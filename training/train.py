"""Training entry point.

  docker compose run --rm train python -m training.train --smoke
  docker compose run --rm train python -m training.train --max-epochs 15
"""
from __future__ import annotations

import argparse
import os

import pytorch_lightning as pl
import torch

from .cache import CACHE_DIR, load
from .module import DEFAULTS, VolumetricSMPLModule


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache-dir', default=CACHE_DIR)
    ap.add_argument('--train-split', default='train')
    ap.add_argument('--model-path', default='data/body_models')
    ap.add_argument('--out-dir', default='runs/smplx_neutral')
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--max-epochs', type=int, default=15)
    ap.add_argument('--lr', type=float, default=DEFAULTS['lr'])
    ap.add_argument('--t-max', type=int, default=DEFAULTS['t_max'])
    ap.add_argument('--n-points', type=int, default=DEFAULTS['n_points'])
    ap.add_argument('--ckpt-every-n-steps', type=int, default=5000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--wandb', action='store_true')
    ap.add_argument('--wandb-id', default=None,
                    help='continue an existing W&B run (use with --resume)')
    ap.add_argument('--resume', default=None)
    ap.add_argument('--smoke', action='store_true',
                    help='200 steps on a small subset, to prove the loop runs')
    ap.add_argument('--limit-train', type=int, default=None)
    args = ap.parse_args()

    pl.seed_everything(args.seed, workers=True)
    torch.set_float32_matmul_precision('high')

    train_cache, train_man = load(os.path.join(args.cache_dir, f'{args.train_split}.pt'))
    val_cache, _ = load(os.path.join(args.cache_dir, 'val.pt'))
    train_params = train_cache.params
    if args.smoke and args.limit_train is None:
        args.limit_train = 2048
    if args.limit_train:
        train_params = {k: v[:args.limit_train] for k, v in train_params.items()}

    print(f"train {len(next(iter(train_params.values()))):,} bodies "
          f"(cache {train_man['n_bodies']:,}, hash {train_man['hash'][:16]})")
    print(f'val   {len(val_cache):,} bodies')

    module = VolumetricSMPLModule(
        train_params=train_params, val_params=val_cache.params,
        model_path=args.model_path, batch_size=args.batch_size,
        lr=args.lr, t_max=args.t_max, n_points=args.n_points)

    # Always keep the CSV logger, even with W&B. Passing a single logger *replaces*
    # the default, which leaves metrics reachable only over the network — awkward for a
    # multi-day run and useless offline. The CSV is the local evidence trail.
    from pytorch_lightning.loggers import CSVLogger
    logger = [CSVLogger(save_dir=args.out_dir, name='csv')]
    if args.wandb:
        from pytorch_lightning.loggers import WandbLogger
        # On --resume, pass --wandb-id <original id> to continue the same W&B run.
        # Without it a resume opens a *new* run and the history is split across two,
        # which is exactly when a continuous curve is most useful.
        logger.append(WandbLogger(
            project='volumetric-smpl', save_dir=args.out_dir,
            id=args.wandb_id, resume='must' if args.wandb_id else None))

    callbacks = [
        pl.callbacks.ModelCheckpoint(
            dirpath=os.path.join(args.out_dir, 'ckpts'),
            filename='step{step:07d}-iou{val/iou_mean:.3f}', auto_insert_metric_name=False,
            every_n_train_steps=args.ckpt_every_n_steps, save_top_k=-1, save_last=True),
        pl.callbacks.LearningRateMonitor(logging_interval='step'),
    ]

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        max_steps=200 if args.smoke else -1,
        accelerator='gpu', devices=1, precision='32-true',
        default_root_dir=args.out_dir, logger=logger, callbacks=callbacks,
        log_every_n_steps=10,
        # A full val pass every epoch is 316 bodies; cheap enough to leave on.
        num_sanity_val_steps=2 if args.smoke else 0,
        limit_val_batches=4 if args.smoke else 1.0,
    )
    trainer.fit(module, ckpt_path=args.resume)

    if torch.cuda.is_available():
        print(f'\npeak VRAM: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB '
              f'(reserved {torch.cuda.max_memory_reserved() / 2**30:.2f} GiB)')


if __name__ == '__main__':
    main()
