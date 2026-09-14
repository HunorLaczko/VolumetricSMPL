"""Evaluation harness.

Computes the paper's Table 1 metrics for a checkpoint under a fixed protocol. With no
`--ckpt` it scores the *released* checkpoint, which validates the metrics, the data
path, the occupancy backend and the IoU split independently of any training code.

  docker compose run --rm train python -m training.evaluate
  docker compose run --rm train python -m training.evaluate --ckpt runs/smplx_neutral/ckpts/last.ckpt
"""
from __future__ import annotations

import argparse
import json
import time

import smplx
import torch
import torch.nn.functional as F

from VolumetricSMPL import attach_volume
from VolumetricSMPL.volumetric_smpl import VolumetricSMPL as _VS

from .amass import NUM_BETAS, scan
from .occupancy import occupancy_kaolin
from .sampling import sample_query_points

# Paper Table 1, SMPL-X neutral, PosePrior. Context only, not the target: the protocol
# behind it is unpublished and inconsistent with these numbers. See FINDINGS.md,
# "Evaluation protocol".
TARGETS = {'iou_mean': 94.67, 'iou_surf': 94.25, 'iou_unif': 95.10,
           'mse_sdf': 3.7e-5, 'mse_abs_sdf': 3.5e-5}

# The actual gate: the *released* smplx_neutral checkpoint scored under the frozen
# protocol below, mean of 5 seeds over all 316 val bodies. A trained model must land no
# more than TOLERANCE_IOU *below* this.
#
# One-sided on purpose: scoring below the reference is evidence the reimplementation
# failed to learn the method, which is what the gate exists to catch, whereas scoring
# above it is not a failure mode.
REFERENCE = {'iou_mean': 91.52, 'iou_surf': 88.49, 'iou_unif': 94.55,
             'mse_sdf': 5.75e-5, 'mse_abs_sdf': 5.43e-5}

# ~6x the largest seed-to-seed sd (0.048 on iou_surf), so it separates a real
# regression from sampling noise. Sampling is stochastic; compare seed-averaged runs.
TOLERANCE_IOU = 0.3


def build_body(model_path: str, gender: str, device: str, flat_hand_mean: bool,
               num_betas: int = NUM_BETAS, ckpt: str | None = None):
    """Build the body. With `ckpt`, load *our* trained weights instead of the released
    ones — the same harness must score both, or the comparison means nothing."""
    body = smplx.create(model_path=model_path, model_type='smplx', gender=gender,
                        num_betas=num_betas, use_pca=False,
                        flat_hand_mean=flat_hand_mean, batch_size=1)
    body = attach_volume(body, pretrained=ckpt is None, device=device)
    if ckpt is not None:
        state = torch.load(ckpt, map_location=device, weights_only=False)
        body.volume.load_state_dict(state.get('state_dict', state))
        body = body.to(device)
    return body.eval()


def chunked(fn, points: torch.Tensor, max_queries: int) -> torch.Tensor:
    """Apply `fn` over the query axis in slices, as COAP does to avoid OOM."""
    B = points.shape[0]
    step = max(1, max_queries // B)
    return torch.cat([fn(c) for c in torch.split(points, step, dim=1)], dim=1)


@torch.no_grad()
def evaluate(body, cache, device, n_points=512, batch_size=8, max_queries=100_000,
             drop_last=False, seed=0, limit=None, sigma=0.01, surface='tight'):
    volume = body.volume
    faces = torch.from_numpy(body.faces.astype('int64')).to(device)
    gen = torch.Generator(device=device).manual_seed(seed)

    n = len(cache) if limit is None else min(limit, len(cache))
    acc = {k: [] for k in ('iou_unif', 'iou_surf', 'iou_unif_coap', 'iou_surf_coap',
                           'mse_sdf', 'mse_abs_sdf', 'mse_sdf_unif', 'mse_sdf_surf',
                           'mse_sdf_in', 'mse_abs_sdf_in', 'frac_outside_all_bbox')}
    n_bodies = 0
    t0 = time.time()

    for lo in range(0, n, batch_size):
        hi = min(lo + batch_size, n)
        if drop_last and hi - lo < batch_size:
            break
        params = cache.batch(lo, hi, device)
        # smplx sizes its unset defaults from the batch_size given at construction, so
        # expression must be passed explicitly for the batch to line up — including the
        # short final batch. Zeros matches COAP, which never sets it.
        params['expression'] = torch.zeros(hi - lo, body.num_expression_coeffs, device=device)
        out = body(**params, return_verts=True, return_full_pose=True)
        B = out.vertices.shape[0]
        n_bodies += B

        pts = sample_query_points(volume, out, n_points, sigma=sigma,
                                  surface=surface, generator=gen)         # (B,K,n,3)
        K = pts.shape[1]
        flat = pts.reshape(B, K * n_points, 3)

        gt_occ = occupancy_kaolin(out.vertices, faces, flat)              # (B,T)
        pred_occ = chunked(lambda c: volume.query_occupancy(c, out), flat, max_queries)
        # Uses the package's own distance, deliberately — see training/geometry.py for
        # why lowering PyTorch3D's min_triangle_area makes this worse, not better.
        gt_sdf = volume.get_gt_sdf(out, flat, gt_occ)
        pred_sdf = chunked(lambda c: volume.query(c, out), flat, max_queries)

        # --- IoU, split correctly ------------------------------------------------
        # The uniform/surface boundary lives on the *last* axis of (B,K,n_points).
        n_u = n_points // 2
        g = gt_occ.reshape(B, K, n_points)
        p = pred_occ.reshape(B, K, n_points)
        acc['iou_unif'].append(_VS.compute_iou(p[..., :n_u].reshape(B, -1),
                                               g[..., :n_u].reshape(B, -1)))
        acc['iou_surf'].append(_VS.compute_iou(p[..., n_u:].reshape(B, -1),
                                               g[..., n_u:].reshape(B, -1)))

        # --- IoU, reproducing COAP's split verbatim ------------------------------
        # COAP splits the *flattened* array at T//2. Because the layout is part-major,
        # that separates the first ~K/2 parts from the rest; both halves are
        # uniform/surface mixtures, so neither is really "unif" or "surf". Computed
        # here only as evidence for which split the paper's numbers came from.
        T = flat.shape[1]
        acc['iou_unif_coap'].append(_VS.compute_iou(pred_occ[:, :T // 2], gt_occ[:, :T // 2]))
        acc['iou_surf_coap'].append(_VS.compute_iou(pred_occ[:, T // 2:], gt_occ[:, T // 2:]))

        acc['mse_sdf'].append(F.mse_loss(pred_sdf, gt_sdf))
        acc['mse_abs_sdf'].append(F.mse_loss(pred_sdf.abs(), gt_sdf.abs()))

        # Diagnostics: which half drives the SDF error, and how many points fall
        # outside every part box (where `query` returns an analytic box SDF rather
        # than a learned one, so large errors there are expected, not a model fault).
        gs = gt_sdf.reshape(B, K, n_points)
        ps = pred_sdf.reshape(B, K, n_points)
        acc['mse_sdf_unif'].append(F.mse_loss(ps[..., :n_u], gs[..., :n_u]))
        acc['mse_sdf_surf'].append(F.mse_loss(ps[..., n_u:], gs[..., n_u:]))

        _, inside_bbox = volume.to_local(flat, volume.impl_code['bone_trans'],
                                         volume.impl_code['bbox_center'],
                                         volume.impl_code['bbox_size'])
        keep = inside_bbox.bool().any(1)                                   # (B,T)
        acc['frac_outside_all_bbox'].append((~keep).float().mean())

        # Same restriction the codebase applies during training: `query_training`
        # drops points outside every box (`pred_occ[~all_out]`), because no learned
        # head is evaluated there at all.
        acc['mse_sdf_in'].append(F.mse_loss(pred_sdf[keep], gt_sdf[keep]))
        acc['mse_abs_sdf_in'].append(F.mse_loss(pred_sdf[keep].abs(), gt_sdf[keep].abs()))

        # The pose-keyed cache must be dropped each step, or a later body with an
        # identically-shaped pose could reuse this one's encoding.
        volume.detach_cache()

    res = {k: float(torch.stack(v).mean()) for k, v in acc.items()}
    for k in ('iou_unif', 'iou_surf', 'iou_unif_coap', 'iou_surf_coap'):
        res[k] *= 100.0
    res['iou_mean'] = (res['iou_unif'] + res['iou_surf']) * 0.5
    res['iou_mean_coap'] = (res['iou_unif_coap'] + res['iou_surf_coap']) * 0.5
    res['n_bodies'] = n_bodies
    res['seconds'] = round(time.time() - t0, 1)
    return res


def report(res: dict) -> bool:
    """Print the scorecard against the reference. Returns True if the gate passes."""
    print(f"\n  bodies evaluated: {res['n_bodies']}   ({res['seconds']}s)\n")
    print(f"  {'metric':<14}{'ours':>12}{'reference':>12}{'delta':>10}"
          f"{'gate':>7}{'paper':>10}")
    print('  ' + '-' * 66)
    ok = True
    for key in ('iou_mean', 'iou_surf', 'iou_unif'):
        d = res[key] - REFERENCE[key]
        passed = d >= -TOLERANCE_IOU
        ok &= passed
        print(f'  {key:<14}{res[key]:>12.2f}{REFERENCE[key]:>12.2f}{d:>+10.2f}'
              f"{'PASS' if passed else 'FAIL':>7}{TARGETS[key]:>10.2f}")
    for key in ('mse_sdf', 'mse_abs_sdf'):
        d = res[key] - REFERENCE[key]
        print(f'  {key:<14}{res[key]:>12.3e}{REFERENCE[key]:>12.3e}{d:>+10.1e}'
              f"{'':>7}{TARGETS[key]:>10.1e}")
    print(f"\n  Gate: no more than {TOLERANCE_IOU} IoU *below* the released checkpoint "
          f"(one-sided). No upper bound; the paper column is context, not a target.")
    print(f"  COAP's part-major split, for comparison:"
          f"  mean {res['iou_mean_coap']:.2f}"
          f"  'surf' {res['iou_surf_coap']:.2f}  'unif' {res['iou_unif_coap']:.2f}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', default='data/extracted')
    ap.add_argument('--model-path', default='data/body_models')
    ap.add_argument('--subsets', nargs='+', default=['PosePrior'])
    ap.add_argument('--stride', type=int, default=500)
    ap.add_argument('--gender', default='neutral')
    ap.add_argument('--n-points', type=int, default=512)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--sigma', type=float, default=0.01, help="surface jitter; COAP's is 0.01")
    ap.add_argument('--surface', default='tight', choices=['tight', 'extended', 'mixed', 'full'])
    ap.add_argument('--num-betas', type=int, default=NUM_BETAS)
    ap.add_argument('--expect', type=int, default=316, help='0 to skip the count gate')
    ap.add_argument('--flat-hand-mean', action='store_true')
    ap.add_argument('--drop-last', action='store_true', help="match COAP's val loader")
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--ckpt', default=None,
                    help='score our trained weights instead of the released checkpoint')
    ap.add_argument('--json-out', default=None)
    args = ap.parse_args()

    cache = scan(args.data_root, args.subsets, args.stride, num_betas=args.num_betas)
    print(f'val bodies: {len(cache)}')
    if args.expect and len(cache) != args.expect:
        raise SystemExit(f'GATE FAILED: expected {args.expect} bodies, built {len(cache)}')

    body = build_body(args.model_path, args.gender, 'cuda', args.flat_hand_mean,
                      num_betas=args.num_betas, ckpt=args.ckpt)
    res = evaluate(body, cache, 'cuda', n_points=args.n_points, batch_size=args.batch_size,
                   drop_last=args.drop_last, seed=args.seed, limit=args.limit,
                   sigma=args.sigma, surface=args.surface)
    res['config'] = vars(args)
    report(res)
    if args.json_out:
        with open(args.json_out, 'w') as f:
            json.dump(res, f, indent=2)


if __name__ == '__main__':
    main()
