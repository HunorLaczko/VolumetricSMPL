"""Golden-batch regression test for the data pipeline.

Pins the generated batch and the losses it produces, so any accidental change to
sampling, occupancy, the pose cache or the loss path fails loudly instead of quietly
shifting the training distribution.

Uses the *released* checkpoint rather than a fresh init, so the recorded losses depend
only on the data pipeline and not on initialisation RNG.

  docker compose run --rm train python -m training.test_golden          # check
  docker compose run --rm train python -m training.test_golden --write  # re-pin
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os

import torch

from . import ragged
from .cache import load
from .evaluate import build_body
from .occupancy import occupancy_kaolin
from .sampling import sample_query_points

GOLDEN_PATH = 'training/golden_batch.json'
# Loss magnitudes are ~1e-1; sampling is seeded, so agreement should be to float32
# round-off. Loose enough to survive a cuDNN/driver update, tight enough to catch a
# real change in the data distribution.
RTOL = 1e-4


def compute(model_path: str, cache_path: str, seed: int = 0, batch: int = 8,
            n_points: int = 512) -> dict:
    torch.manual_seed(seed)
    cache, man = load(cache_path)
    body = build_body(model_path, 'neutral', 'cuda', flat_hand_mean=False)
    vol = body.volume
    faces = torch.from_numpy(body.faces.astype('int64')).cuda()
    gen = torch.Generator(device='cuda').manual_seed(seed)

    params = cache.batch(0, batch, 'cuda')
    params['expression'] = torch.zeros(batch, body.num_expression_coeffs, device='cuda')
    with torch.no_grad():
        out = body(**params, return_verts=True, return_full_pose=True)
        pts = sample_query_points(vol, out, n_points, sigma=0.01, generator=gen)
        B, K = pts.shape[0], pts.shape[1]
        flat = pts.reshape(B, K * n_points, 3)
        gt_occ = occupancy_kaolin(out.vertices, faces, flat)

    torch.manual_seed(seed)
    vol.detach_cache()
    loss = vol.query_training(flat, gt_occ, out)
    vol.detach_cache()

    h = hashlib.sha256(flat.detach().cpu().numpy().tobytes()).hexdigest()
    return {
        'cache_hash': man['hash'],
        'shape': list(flat.shape),
        'points_sha256': h,
        'points_mean': [round(float(x), 8) for x in flat.mean(dim=(0, 1)).tolist()],
        'points_std': round(float(flat.std()), 8),
        'gt_occ_frac_inside': round(float(gt_occ.mean()), 8),
        'loss': {k: round(float(v), 8) for k, v in loss.items()},
        'config': dict(seed=seed, batch=batch, n_points=n_points, sigma=0.01),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-path', default='data/body_models')
    ap.add_argument('--cache', default='data/cache/val.pt')
    ap.add_argument('--write', action='store_true', help='re-pin the golden values')
    args = ap.parse_args()

    got = compute(args.model_path, args.cache)

    if args.write or not os.path.exists(GOLDEN_PATH):
        with open(GOLDEN_PATH, 'w') as f:
            json.dump(got, f, indent=2)
        print(f'wrote {GOLDEN_PATH}')
        print(json.dumps(got, indent=2))
        return

    want = json.load(open(GOLDEN_PATH))
    fails = []
    for key in ('cache_hash', 'shape', 'points_sha256'):
        if got[key] != want[key]:
            fails.append(f'  {key}: {want[key]} -> {got[key]}')
    for key in ('points_std', 'gt_occ_frac_inside'):
        if abs(got[key] - want[key]) > RTOL * max(abs(want[key]), 1e-9):
            fails.append(f'  {key}: {want[key]} -> {got[key]}')
    for k, v in want['loss'].items():
        if abs(got['loss'][k] - v) > RTOL * max(abs(v), 1e-9):
            fails.append(f'  loss/{k}: {v} -> {got["loss"][k]}')

    print(f"cache      {got['cache_hash'][:16]}")
    print(f"points     {got['shape']}  sha {got['points_sha256'][:16]}  "
          f"inside {got['gt_occ_frac_inside']:.6f}")
    print(f"loss       " + '  '.join(f'{k}={v:.6f}' for k, v in got['loss'].items()))
    if fails:
        print('\nGOLDEN BATCH MISMATCH:')
        print('\n'.join(fails))
        raise SystemExit(1)
    print('\nmatches golden batch.')


if __name__ == '__main__':
    main()
