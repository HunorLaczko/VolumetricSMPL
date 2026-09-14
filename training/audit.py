"""Audit the evaluation: can the reported numbers be trusted?

Run this before believing any comparison against the reference. A trained model beating
the released checkpoint is exactly when a harness bug is most likely to go unnoticed.

Each check is designed to fail loudly if the thing it tests is broken:

  1. Data leak      — no PosePrior sequence may appear in the training split.
  2. Loading        — both checkpoints must load strictly, with identical architecture.
  3. Distinctness   — the two models must actually differ in their weights, and produce
                      identical bodies.
  4. Metric floor   — GT-vs-GT must score 100; empty and inverted predictors must score 0.
  5. Independent    — IoU recomputed from scratch, not via the package's compute_iou.
  6. Symmetry       — both models scored on the *same* points in the *same* process.
  7. Occupancy      — GT re-verified against trimesh on the real eval points.

  docker compose run --rm train python -m training.audit --ckpt <path>
"""
from __future__ import annotations

import argparse

import torch

from VolumetricSMPL.volumetric_smpl import VolumetricSMPL as _VS

from .amass import scan
from .cache import SPLITS, load
from .evaluate import REFERENCE, build_body, chunked
from .occupancy import occupancy_kaolin, occupancy_trimesh
from .sampling import sample_query_points

PASS, FAIL = '\033[32mPASS\033[0m', '\033[31mFAIL\033[0m'
results = []


def check(name, ok, detail=''):
    results.append((bool(ok), name, detail))
    print(f'[{PASS if ok else FAIL}] {name:<44} {detail}')


def iou_independent(pred: torch.Tensor, gt: torch.Tensor, level=0.5) -> float:
    """IoU from first principles — no reuse of the package's compute_iou."""
    p = (pred >= level)
    g = (gt >= level)
    inter = (p & g).sum(dim=-1).double()
    union = (p | g).sum(dim=-1).double()
    return float((inter / union.clamp(min=1)).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--data-root', default='data/extracted')
    ap.add_argument('--model-path', default='data/body_models')
    ap.add_argument('--n-points', type=int, default=512)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--trimesh-bodies', type=int, default=4)
    args = ap.parse_args()

    dev = 'cuda'

    # --- 1. data leak -------------------------------------------------------
    val_cache, val_man = load('data/cache/val.pt')
    val_seqs = set(val_cache.seq_names)
    train_cache, train_man = load('data/cache/train.pt')
    train_seqs = set(train_cache.seq_names)
    overlap = val_seqs & train_seqs
    check('No train/val sequence overlap', not overlap,
          f'{len(train_seqs):,} train vs {len(val_seqs)} val sequences, {len(overlap)} shared')
    check('Val split is PosePrior only', SPLITS['val']['subsets'] == ['PosePrior'],
          f"train subsets {train_man['config']['subsets']}")
    check('Val set is 316 bodies', len(val_cache) == 316, f'{len(val_cache)}')

    # --- 2. loading ---------------------------------------------------------
    released = build_body(args.model_path, 'neutral', dev, False)
    ours = build_body(args.model_path, 'neutral', dev, False, ckpt=args.ckpt)

    raw = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    sd = raw.get('state_dict', raw)
    missing, unexpected = ours.volume.load_state_dict(sd, strict=False)
    check('Our checkpoint loads with no missing keys', not missing and not unexpected,
          f'{len(sd)} tensors, {len(missing)} missing, {len(unexpected)} unexpected')

    ka = {k: tuple(v.shape) for k, v in released.volume.state_dict().items()}
    kb = {k: tuple(v.shape) for k, v in ours.volume.state_dict().items()}
    check('Both models share one architecture', ka == kb,
          f'{len(ka)} tensors' + ('' if ka == kb else '  MISMATCH'))

    # --- 3. distinctness ----------------------------------------------------
    # Trainable parameters only. The state dict also holds the face-index buffers —
    # ~880k integers of magnitude ~1e4, identical in both models — which swamp the
    # ~1e-1 weights and drive any whole-state-dict comparison to ~0 regardless.
    pa = torch.cat([p.flatten() for p in released.volume.parameters()])
    pb = torch.cat([p.flatten() for p in ours.volume.parameters()])
    reldiff = float((pa - pb).norm() / pa.norm())
    check('The two models are actually different', reldiff > 1e-3,
          f'relative parameter L2 difference {reldiff:.4f} over {pa.numel():,} params')

    # --- 4-7. metrics on identical points ----------------------------------
    faces = torch.from_numpy(released.faces.astype('int64')).to(dev)
    gen = torch.Generator(device=dev).manual_seed(0)
    n = args.n_points
    n_u = n // 2

    acc = {k: [] for k in ('ref_pkg', 'our_pkg', 'ref_ind', 'our_ind',
                           'perfect', 'constant', 'inverted')}
    trimesh_stats = []

    for lo in range(0, len(val_cache), args.batch_size):
        hi = min(lo + args.batch_size, len(val_cache))
        params = val_cache.batch(lo, hi, dev)
        params['expression'] = torch.zeros(hi - lo, released.num_expression_coeffs,
                                           device=dev)
        with torch.no_grad():
            out = released(**params, return_verts=True, return_full_pose=True)
            B = out.vertices.shape[0]
            pts = sample_query_points(released.volume, out, n, sigma=0.01, generator=gen)
            K = pts.shape[1]
            flat = pts.reshape(B, K * n, 3)
            gt = occupancy_kaolin(out.vertices, faces, flat)

            # Same points, same smpl_output, both models.
            pr_ref = chunked(lambda c: released.volume.query_occupancy(c, out), flat, 100_000)
            released.volume.detach_cache()
            out2 = ours(**params, return_verts=True, return_full_pose=True)
            pr_our = chunked(lambda c: ours.volume.query_occupancy(c, out2), flat, 100_000)
            ours.volume.detach_cache()

            if lo == 0:
                vdiff = float((out.vertices - out2.vertices).abs().max())
                check('Both models produce identical bodies', vdiff < 1e-6,
                      f'max vertex delta {vdiff:.2e} m')

            if lo // args.batch_size < args.trimesh_bodies // args.batch_size + 1 \
                    and lo < args.trimesh_bodies:
                sub = min(args.trimesh_bodies - lo, B)
                gt_t = occupancy_trimesh(out.vertices[:sub], faces, flat[:sub])
                d = (gt[:sub].bool() != gt_t.bool()).float().mean()
                trimesh_stats.append(float(d))

        g = gt.reshape(B, K, n)
        for tag, pred in (('ref', pr_ref), ('our', pr_our)):
            p = pred.reshape(B, K, n)
            u = _VS.compute_iou(p[..., :n_u].reshape(B, -1), g[..., :n_u].reshape(B, -1))
            s = _VS.compute_iou(p[..., n_u:].reshape(B, -1), g[..., n_u:].reshape(B, -1))
            acc[f'{tag}_pkg'].append(float((u + s) * 50))
            ui = iou_independent(p[..., :n_u].reshape(B, -1), g[..., :n_u].reshape(B, -1))
            si = iou_independent(p[..., n_u:].reshape(B, -1), g[..., n_u:].reshape(B, -1))
            acc[f'{tag}_ind'].append((ui + si) * 50)

        # Controls, on the same GT.
        acc['perfect'].append(iou_independent(g.reshape(B, -1), g.reshape(B, -1)) * 100)
        acc['constant'].append(
            iou_independent(torch.zeros_like(g).reshape(B, -1), g.reshape(B, -1)) * 100)
        acc['inverted'].append(
            iou_independent((1 - g).reshape(B, -1), g.reshape(B, -1)) * 100)

    m = {k: sum(v) / len(v) for k, v in acc.items()}

    check('Perfect predictor scores 100', abs(m['perfect'] - 100) < 1e-6,
          f"{m['perfect']:.4f}")
    check('All-empty predictor scores 0', m['constant'] < 1e-6, f"{m['constant']:.4f}")
    check('Inverted predictor scores 0', m['inverted'] < 1e-6, f"{m['inverted']:.4f}")

    check('Independent IoU agrees with package (released)',
          abs(m['ref_pkg'] - m['ref_ind']) < 0.01,
          f"{m['ref_pkg']:.4f} vs {m['ref_ind']:.4f}")
    check('Independent IoU agrees with package (ours)',
          abs(m['our_pkg'] - m['our_ind']) < 0.01,
          f"{m['our_pkg']:.4f} vs {m['our_ind']:.4f}")

    check('Released reproduces the stored reference',
          abs(m['ref_pkg'] - REFERENCE['iou_mean']) < 0.3,
          f"{m['ref_pkg']:.2f} vs stored {REFERENCE['iou_mean']}")

    if trimesh_stats:
        rate = sum(trimesh_stats) / len(trimesh_stats)
        check('GT occupancy matches trimesh on eval points', rate < 0.002,
              f'{rate * 100:.4f}% disagreement')

    print(f"\n  Scored on identical points, same process:")
    print(f"    released  {m['ref_pkg']:.3f}")
    print(f"    ours      {m['our_pkg']:.3f}")
    print(f"    delta     {m['our_pkg'] - m['ref_pkg']:+.3f}")

    bad = [n for ok, n, _ in results if not ok]
    print(f"\n{len(results) - len(bad)}/{len(results)} checks passed")
    if bad:
        print('FAILED: ' + '; '.join(bad))
        raise SystemExit(1)


if __name__ == '__main__':
    main()
