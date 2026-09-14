"""Training loop.

Adam at the hyper-parameters recovered from the released checkpoint (see FINDINGS.md),
per-step cosine annealing, and all data -- posed bodies, query points, ground truth --
generated on device inside one jitted step.

Two things are easy to get wrong and are handled explicitly:

**The cosine schedule must not clamp.** `optax.cosine_decay_schedule` flattens at
`T_max`; the original run's `CosineAnnealingLR` is periodic and keeps going, and training
deliberately overruns `T_max = 450,000`. The final steps sit on the *upturn*, where the
learning rate is rising again. A clamped schedule would be a different experiment, so the
closed form is written out below.

**The ragged filter needs a static budget.** See `model.compact_indices`; overflow is
surfaced as a metric rather than silently truncating the loss.

  docker compose run --rm jax python -m training.train --smoke
  docker compose run --rm jax python -m training.train --max-epochs 15 --wandb
"""
from __future__ import annotations

import argparse
import csv
import os
import time

import jax
import jax.numpy as jnp
import optax

from VolumetricSMPL import assets as A
from VolumetricSMPL import checkpoint as C
from VolumetricSMPL import lbs as L
from VolumetricSMPL import model as MD

from . import cache as CA
from . import occupancy as OJ
from . import sampling as S

DEFAULTS = dict(
    lr=5e-4, b1=0.9, b2=0.999, eps=1e-8, weight_decay=0.0,
    t_max=450_000, eta_min=1e-5,
    n_points=512, sigma=0.01, n_unif_samples=512,
    # Static in-box budget per (body, part), because jit needs static shapes. 2048 was
    # the setting for the reported run, and it is too small for the full training split:
    # rows reached at least 3,268 in-box points, overflowing on ~9% of steps and dropping
    # ~0.6% of the occupancy signal. Overflow is logged as `ragged_overflow`. Raising the
    # budget is exact but costs speed linearly; see FINDINGS.md.
    ragged_pad=2048,
)


def cosine_schedule(base_lr: float, t_max: int, eta_min: float):
    """torch's `CosineAnnealingLR`, closed form, deliberately unclamped.

    lr(t) = eta_min + (base_lr - eta_min) * (1 + cos(pi t / T_max)) / 2

    Periodic with period 2*T_max, so past T_max it rises again -- which is exactly what
    the original run does over its final steps.
    """
    def sched(step):
        return eta_min + (base_lr - eta_min) * (1 + jnp.cos(jnp.pi * step / t_max)) / 2
    return sched


def make_optimizer(cfg):
    sched = cosine_schedule(cfg['lr'], cfg['t_max'], cfg['eta_min'])
    tx = optax.adam(learning_rate=sched, b1=cfg['b1'], b2=cfg['b2'], eps=cfg['eps'])
    if cfg['weight_decay']:
        tx = optax.chain(tx, optax.add_decayed_weights(cfg['weight_decay']))
    return tx, sched


def build_steps(a, tx, cfg, cache):
    """One jitted step: bodies, sampling, ground truth, loss, update.

    Ground-truth occupancy is pure JAX (see occupancy.py), so there is no host or
    framework boundary inside the step and the whole thing is one fusion region.

    `a` holds device arrays, so it cannot be a `static_argnames` argument -- hashing a
    frozen dataclass of jnp arrays raises. Closing over it keeps everything static that
    needs to be, with no pytree registration.
    """

    @jax.jit
    def step(params, opt_state, idx, key):
        # The batch is gathered on device from the cache closed over below. Doing it on
        # the host cost 12.6 ms/step -- eight separate device gathers plus a sync, about
        # 8% of the step, for work XLA folds into the first operation that reads it.
        cache_batch = {k: v[idx] for k, v in cache.items()}
        k_pts, k_enc, k_udf = jax.random.split(key, 3)

        verts, joints, full_pose = L.forward(cache_batch, a)
        bone_trans = jnp.linalg.inv(
            MD.compute_abs_transformations(full_pose, joints, a))
        bbox_min, bbox_max = MD.bbox_bounds(verts, bone_trans, a)
        pts = S.sample_query_points(k_pts, verts, bone_trans, bbox_min, bbox_max,
                                    a, cfg['n_points'], sigma=cfg['sigma'])
        B = verts.shape[0]
        points = pts.reshape(B, -1, 3)

        # Ground truth is data, not a function of the parameters.
        gt_occ = jax.lax.stop_gradient(OJ.occupancy(verts, points, a.faces))

        def loss_fn(p):
            impl = MD.encode_body(p, verts, joints, full_pose, a, k_enc)
            out = MD.ragged_losses(p, points, gt_occ, verts, impl, a, k_udf,
                                   pad=cfg['ragged_pad'],
                                   n_unif=cfg['n_unif_samples'])
            return out['total_loss'], out

        (_, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, opt_state = tx.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, aux

    return step


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-path', default='data/body_models')
    ap.add_argument('--train-cache', default='train',
                    help='a split name under data/cache, or a cache .npz')
    ap.add_argument('--out-dir', default='runs/jax_smplx_neutral')
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--max-epochs', type=int, default=15)
    ap.add_argument('--lr', type=float, default=DEFAULTS['lr'])
    ap.add_argument('--t-max', type=int, default=DEFAULTS['t_max'])
    ap.add_argument('--n-points', type=int, default=DEFAULTS['n_points'])
    ap.add_argument('--ragged-pad', type=int, default=DEFAULTS['ragged_pad'])
    ap.add_argument('--ckpt-every', type=int, default=5000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--wandb', action='store_true')
    ap.add_argument('--smoke', action='store_true', help='200 steps, to prove it runs')
    ap.add_argument('--limit-train', type=int, default=None)
    ap.add_argument('--init-from', default=None,
                    help="start from weights ('released', .ckpt or .npz) instead of a "
                         'fresh init')
    args = ap.parse_args()

    cfg = {**DEFAULTS, 'lr': args.lr, 't_max': args.t_max, 'n_points': args.n_points,
           'ragged_pad': args.ragged_pad}
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, 'ckpts'), exist_ok=True)

    from VolumetricSMPL import current_matmul_precision
    a = A.build(args.model_path)
    train = CA.load_params(args.train_cache)
    n_train = len(next(iter(train.values())))
    if args.smoke and args.limit_train is None:
        args.limit_train = 2048
    if args.limit_train:
        train = {k: v[:args.limit_train] for k, v in train.items()}
        n_train = args.limit_train

    print(f'matmul precision: {current_matmul_precision()}')
    print(f'train {n_train:,} bodies   batch {args.batch_size}   '
          f'ragged pad {cfg["ragged_pad"]}')

    params = init_params(args)
    tx, sched = make_optimizer(cfg)
    opt_state = tx.init(params)
    step_fn = build_steps(a, tx, cfg, train)

    steps_per_epoch = n_train // args.batch_size
    total = 200 if args.smoke else steps_per_epoch * args.max_epochs
    print(f'{steps_per_epoch:,} steps/epoch  x {args.max_epochs} = {total:,} steps')

    run = None
    if args.wandb:
        import wandb
        run = wandb.init(project='volumetric-smpl', job_type='train',
                         config={**cfg, 'framework': 'jax', 'batch_size': args.batch_size})

    csv_path = os.path.join(args.out_dir, 'metrics.csv')
    csv_f = open(csv_path, 'a', newline='')
    csv_w = csv.writer(csv_f)
    if csv_f.tell() == 0:
        csv_w.writerow(['step', 'epoch', 'lr', 'mse_occ', 'mse_udf', 'total_loss',
                        'ragged_overflow', 'it_per_s'])

    key = jax.random.PRNGKey(args.seed)
    step = 0
    t0 = time.time()
    overflow_seen = 0

    for epoch in range(args.max_epochs):
        key, k_perm = jax.random.split(key)
        # Whole epoch's batch indices, built once on device: reading them per step
        # would reintroduce the host round trip the gather was moved to avoid.
        perm = jax.random.permutation(k_perm, n_train)
        idx_all = perm[:steps_per_epoch * args.batch_size].reshape(
            steps_per_epoch, args.batch_size)
        for i in range(steps_per_epoch):
            idx = idx_all[i]
            key, k_step = jax.random.split(key)
            params, opt_state, aux = step_fn(params, opt_state, idx, k_step)
            step += 1

            # Every metric read blocks on the device, so this is deliberately
            # infrequent: at 50 it cost more than the ragged filter saved.
            if step % 200 == 0 or step == 1:
                over = int(aux['ragged_overflow'])
                overflow_seen = max(overflow_seen, over)
                its = step / (time.time() - t0)
                row = [step, epoch, float(sched(step)), float(aux['mse_occ']),
                       float(aux['mse_udf']), float(aux['total_loss']), over,
                       round(its, 3)]
                csv_w.writerow(row)
                csv_f.flush()
                if run:
                    run.log({'train/mse_occ': row[3], 'train/mse_udf': row[4],
                             'train/total_loss': row[5], 'lr': row[2],
                             'train/ragged_overflow': over, 'it_per_s': its}, step=step)
                if step % 500 == 0 or step == 1 or step == total:
                    print(f'  step {step:>7,}  loss {row[5]:.6f}  '
                          f'lr {row[2]:.2e}  {its:.2f} it/s'
                          + (f'  OVERFLOW {over}' if over else ''))

            if step % args.ckpt_every == 0 or step == total:
                save(params, opt_state, step, args.out_dir)
            if step >= total:
                break
        if step >= total:
            break

    save(params, opt_state, step, args.out_dir)
    csv_f.close()
    dt = time.time() - t0
    print(f'\ndone: {step:,} steps in {dt / 3600:.2f} h ({step / dt:.2f} it/s)')
    if overflow_seen:
        print(f'WARNING: ragged budget overflowed by up to {overflow_seen} points; '
              f'raise --ragged-pad and re-run -- the loss dropped those points.')
    if run:
        run.finish()


def init_params(args):
    """Fresh init, or existing weights.

    A from-scratch init has to reproduce the original initialisation, which is not
    generic: `ResnetBlockFC` zero-initialises `fc_1` so each block starts as its
    shortcut, and `ImplicitNet`'s geometric init puts the final layer at a sphere of
    radius 1. Those are load-bearing for convergence, not stylistic.
    """
    if args.init_from:
        return C.load_weights(args.init_from)
    from .init import init_weights
    return init_weights(jax.random.PRNGKey(args.seed))


def save(params, opt_state, step, out_dir):
    C.save_weights(params, os.path.join(out_dir, 'ckpts', f'step{step:07d}.npz'))
    C.save_weights(params, os.path.join(out_dir, 'ckpts', 'last.npz'))


if __name__ == '__main__':
    main()
