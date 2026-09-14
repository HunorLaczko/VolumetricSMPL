"""Evaluation harness.

Computes the paper's Table 1 metrics under a fixed protocol. With the default
`--weights released` it scores the released checkpoint, which validates the metrics, the
data path and the occupancy ground truth independently of any training code.

Two quirks of the protocol are kept on purpose rather than tidied:

- **Batches are averaged unweighted.** 316 val bodies at batch 8 leaves a final batch of
  4, which contributes as much as a full one. Fixing that would change the number and
  break comparability with every result already recorded.
- **IoU is per body, then meaned** -- not pooled across the batch.

Sampling is stochastic, so compare seed-averaged runs.

  docker compose run --rm jax python -m training.evaluate
  docker compose run --rm jax python -m training.evaluate --weights runs/jax_smplx_neutral/ckpts/last.npz
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import time

import jax
import jax.numpy as jnp

from VolumetricSMPL import assets as A
from VolumetricSMPL import checkpoint as C
from VolumetricSMPL import geometry as G
from VolumetricSMPL import lbs as L
from VolumetricSMPL import model as MD

from . import cache as CA
from . import occupancy as OJ
from . import sampling as S

# Paper Table 1, SMPL-X neutral, PosePrior. Context only, not the target: the protocol
# behind it is unpublished and inconsistent with these numbers. See FINDINGS.md,
# "Evaluation protocol".
TARGETS = {'iou_mean': 94.67, 'iou_surf': 94.25, 'iou_unif': 95.10,
           'mse_sdf': 3.7e-5, 'mse_abs_sdf': 3.5e-5}

# The actual gate: the *released* smplx_neutral checkpoint scored under this protocol,
# mean of 5 seeds over all 316 val bodies. A trained model must land no more than
# TOLERANCE_IOU *below* this.
#
# One-sided on purpose: scoring below the reference is evidence the reimplementation
# failed to learn the method, which is what the gate exists to catch, whereas scoring
# above it is not a failure mode.
REFERENCE = {'iou_mean': 91.52, 'iou_surf': 88.49, 'iou_unif': 94.55,
             'mse_sdf': 5.75e-5, 'mse_abs_sdf': 5.43e-5}

# ~6x the largest seed-to-seed sd (0.048 on iou_surf), so it separates a real
# regression from sampling noise.
TOLERANCE_IOU = 0.3

# Query-axis chunk for the decoder. The forward on (120, 7680, 143) allocates several GB
# of activations, so it is chunked.
CHUNK = 2048


def compute_iou(pred, gt, level: float = 0.5):
    """Per-body IoU, then the mean. Matches the original `compute_iou`."""
    a = pred >= level
    b = gt >= level
    union = jnp.sum(a | b, axis=-1)
    inter = jnp.sum(a & b, axis=-1)
    return jnp.mean(inter / jnp.maximum(union, 1))


def chunked(fn, points, chunk: int = CHUNK):
    T = points.shape[1]
    return jnp.concatenate([fn(points[:, i:i + chunk]) for i in range(0, T, chunk)],
                           axis=1)


def evaluate(params, a, cache, key, n_points: int = 512, batch_size: int = 8,
             sigma: float = 0.01, limit: int | None = None,
             with_sdf: bool = True):
    _occ = jax.jit(lambda v, p: OJ.occupancy(v, p, a.faces))
    n = len(next(iter(cache.values())))
    if limit is not None:
        n = min(limit, n)

    acc = {k: [] for k in ('iou_unif', 'iou_surf', 'iou_unif_coap', 'iou_surf_coap',
                           'mse_sdf', 'mse_abs_sdf', 'frac_outside_all_bbox')}
    n_bodies = 0
    t0 = time.time()

    for lo in range(0, n, batch_size):
        hi = min(lo + batch_size, n)
        params_b = {k: v[lo:hi] for k, v in cache.items()}
        B = hi - lo
        n_bodies += B

        verts, joints, full_pose = L.forward(params_b, a)

        key, k_enc, k_pts = jax.random.split(key, 3)
        impl = MD.encode_body(params, verts, joints, full_pose, a, k_enc)
        bbox_min, bbox_max = impl['bbox_min'], impl['bbox_max']

        pts = S.sample_query_points(k_pts, verts, impl['bone_trans'],
                                    bbox_min, bbox_max, a, n_points, sigma=sigma)
        K = pts.shape[1]
        flat = pts.reshape(B, K * n_points, 3)

        gt_occ = _occ(verts, flat)
        pred_occ = chunked(lambda c: MD.query_occupancy(params, c, impl, a), flat)

        if with_sdf:
            pred_sdf = chunked(lambda c: MD.query_sdf(params, c, impl, a), flat)
            # Ground-truth SDF: unsigned distance to the whole body, signed by occupancy.
            # This is the expensive term by a wide margin -- 7,680 points against 20,908
            # triangles per body, and it cannot use a BVH nearest-triangle query because
            # the reference distance is the edge-only degenerate branch, not the true
            # closest-point distance (see VolumetricSMPL/geometry.py). Skippable when
            # only IoU is wanted.
            gt_udf = jax.vmap(lambda v, p: G.udf(p, v, a.faces))(verts, flat)
            gt_sdf = jnp.where(gt_occ > 0.5, -gt_udf, gt_udf)

        n_u = n_points // 2
        g = gt_occ.reshape(B, K, n_points)
        p = pred_occ.reshape(B, K, n_points)
        acc['iou_unif'].append(compute_iou(p[..., :n_u].reshape(B, -1),
                                           g[..., :n_u].reshape(B, -1)))
        acc['iou_surf'].append(compute_iou(p[..., n_u:].reshape(B, -1),
                                           g[..., n_u:].reshape(B, -1)))

        # COAP's own split, on the flattened part-major array. Both halves are
        # uniform/surface mixtures, so neither is really "unif" or "surf" -- kept only as
        # evidence about which split the paper's numbers came from.
        T = flat.shape[1]
        acc['iou_unif_coap'].append(compute_iou(pred_occ[:, :T // 2], gt_occ[:, :T // 2]))
        acc['iou_surf_coap'].append(compute_iou(pred_occ[:, T // 2:], gt_occ[:, T // 2:]))

        if with_sdf:
            acc['mse_sdf'].append(jnp.mean((pred_sdf - gt_sdf) ** 2))
            acc['mse_abs_sdf'].append(
                jnp.mean((jnp.abs(pred_sdf) - jnp.abs(gt_sdf)) ** 2))

        _, inside = MD.to_local(flat, impl['bone_trans'],
                                impl['bbox_center'], impl['bbox_size'])
        acc['frac_outside_all_bbox'].append(
            jnp.mean((~jnp.any(inside, axis=1)).astype(jnp.float32)))

    res = {k: float(jnp.mean(jnp.stack(v))) for k, v in acc.items() if v}
    for k in ('iou_unif', 'iou_surf', 'iou_unif_coap', 'iou_surf_coap'):
        res[k] *= 100.0
    res['iou_mean'] = (res['iou_unif'] + res['iou_surf']) * 0.5
    res['iou_mean_coap'] = (res['iou_unif_coap'] + res['iou_surf_coap']) * 0.5
    res['n_bodies'] = n_bodies
    res['seconds'] = round(time.time() - t0, 1)
    return res


def report(runs: list[dict]) -> bool:
    """Print the seed-averaged scorecard against the reference. True if the gate passes."""
    print(f"\n  {runs[0]['n_bodies']} bodies, {len(runs)} seed(s)\n")
    print(f"  {'metric':<14}{'mean':>10}{'sd':>8}{'reference':>12}{'delta':>9}"
          f"{'gate':>7}{'paper':>9}")
    print('  ' + '-' * 69)
    ok = True
    for k in ('iou_mean', 'iou_surf', 'iou_unif'):
        v = [r[k] for r in runs]
        m, sd = st.mean(v), (st.stdev(v) if len(v) > 1 else 0.0)
        d = m - REFERENCE[k]
        passed = d >= -TOLERANCE_IOU
        ok &= passed
        print(f"  {k:<14}{m:>10.2f}{sd:>8.3f}{REFERENCE[k]:>12.2f}{d:>+9.2f}"
              f"{'PASS' if passed else 'FAIL':>7}{TARGETS[k]:>9.2f}")
    for k in ('mse_sdf', 'mse_abs_sdf'):
        if k not in runs[0]:
            continue
        v = [r[k] for r in runs]
        m, sd = st.mean(v), (st.stdev(v) if len(v) > 1 else 0.0)
        print(f"  {k:<14}{m:>10.2e}{sd:>8.0e}{REFERENCE[k]:>12.2e}"
              f"{m - REFERENCE[k]:>+9.0e}{'':>7}{TARGETS[k]:>9.1e}")
    print(f"\n  Gate: no more than {TOLERANCE_IOU} IoU *below* the released checkpoint "
          f"(one-sided). No upper bound; the paper column is context, not a target.")
    print(f"  COAP's part-major split, for comparison: mean "
          f"{st.mean(r['iou_mean_coap'] for r in runs):.2f}   outside every box "
          f"{st.mean(r['frac_outside_all_bbox'] for r in runs) * 100:.2f}%")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-path', default='data/body_models')
    ap.add_argument('--weights', default='released',
                    help="'released', a .ckpt, or an .npz from training.train")
    ap.add_argument('--cache', default='val',
                    help='a split name under data/cache, or a cache .npz')
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    ap.add_argument('--n-points', type=int, default=512)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--sigma', type=float, default=0.01, help="surface jitter; COAP's is 0.01")
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--skip-sdf', action='store_true',
                    help='IoU only; drops the full-body distance term, which dominates')
    ap.add_argument('--json-out', default=None)
    args = ap.parse_args()

    from VolumetricSMPL import current_matmul_precision
    a = A.build(args.model_path)
    params = C.load_weights(args.weights)
    cache = CA.load_params(args.cache)
    print(f'matmul precision: {current_matmul_precision()}')
    print(f"bodies: {len(next(iter(cache.values())))}   weights: {args.weights}")

    runs = []
    for s in args.seeds:
        r = evaluate(params, a, cache, jax.random.PRNGKey(s),
                     n_points=args.n_points, batch_size=args.batch_size,
                     sigma=args.sigma, limit=args.limit,
                     with_sdf=not args.skip_sdf)
        runs.append(r)
        print(f"  seed {s}: iou_mean {r['iou_mean']:.3f}  "
              f"unif {r['iou_unif']:.3f}  surf {r['iou_surf']:.3f}  ({r['seconds']}s)")

    report(runs)
    if args.json_out:
        with open(args.json_out, 'w') as f:
            json.dump(dict(config=vars(args), runs=runs), f, indent=2)


if __name__ == '__main__':
    main()
