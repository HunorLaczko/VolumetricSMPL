"""Differential sweep over the evaluation protocol and body construction.

The paper states neither the query-point protocol nor the SMPL-X construction flags;
COAP's data loader and configs are the only written sources, and they were authored for
SMPL. This varies one knob at a time against the *released* checkpoint — whose published
numbers act as ground truth — to identify which choices those numbers came from.

  docker compose run --rm train python -m training.sweep --limit 64
"""
from __future__ import annotations

import argparse

from .amass import scan
from .evaluate import TARGETS, build_body, evaluate

# Body construction: (label, num_betas, flat_hand_mean, zero_face)
BODY_VARIANTS = [
    ('betas16 fhm=F', 16, False, False),
    ('betas16 fhm=T', 16, True, False),
    ('betas10 fhm=F', 10, False, False),
    ('betas10 fhm=T', 10, True, False),
    ('betas16 fhm=F zeroface', 16, False, True),
]

# Sampling protocol: (label, evaluate kwargs)
PROTOCOL_VARIANTS = [
    ('sigma=0.01 (COAP)', dict(sigma=0.01)),
    ('sigma=0.02', dict(sigma=0.02)),
    ('sigma=0.05', dict(sigma=0.05)),
    ('sigma=0.10 (paper)', dict(sigma=0.10)),
    ('surface=extended', dict(sigma=0.01, surface='extended')),
    ('surface=mixed', dict(sigma=0.01, surface='mixed')),
]

HEAD = (f"  {'variant':<24}{'iou_mean':>10}{'iou_surf':>10}{'iou_unif':>10}"
        f"{'mse_sdf':>11}{'mse_unif':>11}{'mse_surf':>11}{'d_mean':>9}")


def row(label, r):
    print(f"  {label:<24}{r['iou_mean']:>10.2f}{r['iou_surf']:>10.2f}"
          f"{r['iou_unif']:>10.2f}{r['mse_sdf']:>11.2e}{r['mse_sdf_unif']:>11.2e}"
          f"{r['mse_sdf_surf']:>11.2e}{r['iou_mean'] - TARGETS['iou_mean']:>+9.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', default='data/extracted')
    ap.add_argument('--model-path', default='data/body_models')
    ap.add_argument('--limit', type=int, default=64)
    ap.add_argument('--n-points', type=int, default=512)
    args = ap.parse_args()

    print('=== body construction (sigma=0.01, COAP protocol) ===\n')
    print(HEAD)
    print('  ' + '-' * 96)
    for label, nb, fhm, zero_face in BODY_VARIANTS:
        cache = scan(args.data_root, ['PosePrior'], 500, num_betas=nb)
        if zero_face:
            for k in ('jaw_pose', 'leye_pose', 'reye_pose'):
                cache.params[k] = cache.params[k].zero_()
        body = build_body(args.model_path, 'neutral', 'cuda', fhm, num_betas=nb)
        r = evaluate(body, cache, 'cuda', n_points=args.n_points, limit=args.limit)
        row(label, r)
        if label == BODY_VARIANTS[0][0]:
            print(f"    (points outside every part bbox: "
                  f"{r['frac_outside_all_bbox'] * 100:.2f}%)")

    print('\n=== sampling protocol (betas16, fhm=F) ===\n')
    print(HEAD)
    print('  ' + '-' * 96)
    cache = scan(args.data_root, ['PosePrior'], 500, num_betas=16)
    body = build_body(args.model_path, 'neutral', 'cuda', False, num_betas=16)
    for label, kw in PROTOCOL_VARIANTS:
        r = evaluate(body, cache, 'cuda', n_points=args.n_points, limit=args.limit, **kw)
        row(label, r)

    print(f"\n  {'paper':<24}{TARGETS['iou_mean']:>10.2f}{TARGETS['iou_surf']:>10.2f}"
          f"{TARGETS['iou_unif']:>10.2f}{TARGETS['mse_sdf']:>11.2e}")


if __name__ == '__main__':
    main()
