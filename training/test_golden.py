"""Golden-batch regression test for the data pipeline.

Pins one generated batch and the losses it produces, so any accidental change to the body
model, sampling, occupancy, the pose cache or the loss path fails loudly instead of
quietly shifting the training distribution.

Uses the *released* checkpoint rather than a fresh init, so the recorded losses depend
only on the data path. The loss is computed twice, through the full forward and through
the ragged in-box filter the training step uses; the two must agree, which checks that
the filter is exact rather than an approximation.

JAX's random streams are reproducible, but float results can move with the GPU, driver
or XLA version. After such a change, re-pin with `--write` once the difference is
understood.

  docker compose run --rm jax python -m training.test_golden          # check
  docker compose run --rm jax python -m training.test_golden --write  # re-pin
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os

import jax
import jax.numpy as jnp
import numpy as np

from VolumetricSMPL import assets as A
from VolumetricSMPL import checkpoint as C
from VolumetricSMPL import lbs as L
from VolumetricSMPL import model as MD

from . import cache as CA
from . import occupancy as OJ
from . import sampling as S
from .train import DEFAULTS

GOLDEN_PATH = 'training/golden_batch.json'
# Loss magnitudes are ~1e-2; sampling is seeded, so agreement should be to float32
# round-off. Loose enough to survive a driver update, tight enough to catch a real change
# in the data distribution.
RTOL = 1e-4


def compute(model_path: str, cache_name: str, seed: int = 0, batch: int = 8,
            n_points: int = 512) -> dict:
    cache, man = CA.load(CA.path_for(cache_name))
    a = A.build(model_path)
    weights = C.load_weights('released')

    params = {k: jnp.asarray(v[:batch]) for k, v in cache.params.items()}
    verts, joints, full_pose = L.forward(params, a)
    # The same split of the step key as `train.build_steps`.
    k_pts, k_enc, k_udf = jax.random.split(jax.random.PRNGKey(seed), 3)

    impl = MD.encode_body(weights, verts, joints, full_pose, a, k_enc)
    pts = S.sample_query_points(k_pts, verts, impl['bone_trans'], impl['bbox_min'],
                                impl['bbox_max'], a, n_points, sigma=0.01)
    flat = pts.reshape(pts.shape[0], -1, 3)
    gt_occ = OJ.occupancy(verts, flat, a.faces)

    full = MD.losses(weights, flat, gt_occ, verts, impl, a, k_udf)
    ragged = MD.ragged_losses(weights, flat, gt_occ, verts, impl, a, k_udf,
                              pad=DEFAULTS['ragged_pad'])

    points = np.asarray(flat)
    terms = ('mse_occ', 'mse_udf', 'total_loss')
    return {
        'cache_hash': man['hash'],
        'shape': list(points.shape),
        'points_sha256': hashlib.sha256(points.tobytes()).hexdigest(),
        'points_mean': [round(float(x), 8) for x in points.mean(axis=(0, 1))],
        'points_std': round(float(points.std()), 8),
        'gt_occ_frac_inside': round(float(jnp.mean(gt_occ)), 8),
        'loss': {k: round(float(full[k]), 8) for k in terms},
        'loss_ragged': {k: round(float(ragged[k]), 8) for k in terms},
        'ragged_overflow': int(ragged['ragged_overflow']),
        'config': dict(seed=seed, batch=batch, n_points=n_points, sigma=0.01,
                       ragged_pad=DEFAULTS['ragged_pad']),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-path', default='data/body_models')
    ap.add_argument('--cache', default='val')
    ap.add_argument('--write', action='store_true', help='re-pin the golden values')
    args = ap.parse_args()

    got = compute(args.model_path, args.cache)

    print(f"cache      {got['cache_hash'][:16]}")
    print(f"points     {got['shape']}  sha {got['points_sha256'][:16]}  "
          f"inside {got['gt_occ_frac_inside']:.6f}")
    print('loss       ' + '  '.join(f'{k}={v:.6f}' for k, v in got['loss'].items()))
    print('ragged     ' + '  '.join(f'{k}={v:.6f}' for k, v in got['loss_ragged'].items())
          + f"  overflow={got['ragged_overflow']}")

    fails = []
    # Not pinned: must hold on every machine.
    if got['ragged_overflow']:
        fails.append(f"  ragged budget overflowed by {got['ragged_overflow']}")
    for k, v in got['loss'].items():
        if abs(got['loss_ragged'][k] - v) > 1e-5 * max(abs(v), 1e-9):
            fails.append(f"  ragged {k} {got['loss_ragged'][k]} != full {v}")

    if args.write or not os.path.exists(GOLDEN_PATH):
        if fails:
            raise SystemExit('refusing to pin a batch that fails:\n' + '\n'.join(fails))
        with open(GOLDEN_PATH, 'w') as f:
            json.dump(got, f, indent=2)
            f.write('\n')
        print(f'\nwrote {GOLDEN_PATH}')
        return

    with open(GOLDEN_PATH) as f:
        want = json.load(f)
    for key in ('cache_hash', 'shape', 'points_sha256'):
        if got[key] != want[key]:
            fails.append(f'  {key}: {want[key]} -> {got[key]}')
    for key in ('points_std', 'gt_occ_frac_inside'):
        if abs(got[key] - want[key]) > RTOL * max(abs(want[key]), 1e-9):
            fails.append(f'  {key}: {want[key]} -> {got[key]}')
    for k, v in want['loss'].items():
        if abs(got['loss'][k] - v) > RTOL * max(abs(v), 1e-9):
            fails.append(f'  loss/{k}: {v} -> {got["loss"][k]}')

    if fails:
        print('\nGOLDEN BATCH MISMATCH:')
        print('\n'.join(fails))
        raise SystemExit(1)
    print('\nmatches golden batch; ragged and full losses agree.')


if __name__ == '__main__':
    main()
