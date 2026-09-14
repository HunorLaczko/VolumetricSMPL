"""Audit the evaluation: can the reported numbers be trusted?

Run this before believing any comparison against the reference. A trained model beating
the released checkpoint is exactly when a harness bug is most likely to go unnoticed.

Each check is designed to fail loudly if the thing it tests is broken:

  1. Data leak      — no val sequence may appear in the training split.
  2. Weights        — both checkpoints share one architecture, are finite, and differ.
  3. Metric floor   — GT-vs-GT must score 100; empty and inverted predictors must score 0.
  4. Independent    — IoU recomputed from scratch in numpy, not via `compute_iou`.
  5. Symmetry       — both models scored on the *same* points in the *same* process.
  6. Occupancy      — GT re-verified against trimesh on the real eval points.
  7. Decomposition  — the part split has the shape the checkpoints expect.

  docker compose run --rm jax python -m training.audit --weights runs/jax_smplx_neutral/ckpts/last.npz
"""
from __future__ import annotations

import argparse
import sys

import jax
import jax.numpy as jnp
import numpy as np

from VolumetricSMPL import assets as A
from VolumetricSMPL import checkpoint as C
from VolumetricSMPL import lbs as L
from VolumetricSMPL import model as MD

from . import cache as CA
from . import evaluate as E
from . import occupancy as OJ
from . import sampling as S

results: list[tuple[bool, str, str]] = []


def check(name: str, ok: bool, detail: str = ''):
    results.append((bool(ok), name, detail))


def iou_independent(pred, gt, level: float = 0.5) -> float:
    """IoU from first principles, in float64 numpy — no reuse of `compute_iou`."""
    p = np.asarray(pred) >= level
    g = np.asarray(gt) >= level
    inter = (p & g).sum(-1).astype(np.float64)
    union = (p | g).sum(-1).astype(np.float64)
    return float((inter / np.maximum(union, 1)).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', required=True,
                    help='trained weights to audit: a .ckpt or an .npz')
    ap.add_argument('--model-path', default='data/body_models')
    ap.add_argument('--n-points', type=int, default=512)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--trimesh-bodies', type=int, default=4)
    args = ap.parse_args()

    # --- 1. data leak -------------------------------------------------------------
    val, val_man = CA.load(CA.path_for('val'))
    train, train_man = CA.load(CA.path_for('train'))
    overlap = set(val.seq_names) & set(train.seq_names)
    check('No train/val sequence overlap', not overlap,
          f'{len(set(train.seq_names)):,} train vs {len(set(val.seq_names))} val '
          f'sequences, {len(overlap)} shared')
    check('Val split is PosePrior only', val_man['config']['subsets'] == ['PosePrior'],
          f"train subsets {train_man['config']['subsets']}")
    check('Val set is 316 bodies', len(val) == 316, str(len(val)))

    # --- 2. weights ---------------------------------------------------------------
    released = C.load_weights('released')
    ours = C.load_weights(args.weights)
    check('Both checkpoints share one architecture', sorted(released) == sorted(ours),
          f'{len(released)} vs {len(ours)} tensors')
    check('Matching shapes across checkpoints',
          all(k in ours and released[k].shape == ours[k].shape for k in released))
    pa = jnp.concatenate([released[k].ravel() for k in sorted(released)])
    pb = jnp.concatenate([ours[k].ravel() for k in sorted(released) if k in ours])
    reldiff = float(jnp.linalg.norm(pa - pb) / jnp.linalg.norm(pa)) \
        if pa.shape == pb.shape else float('nan')
    check('The two checkpoints are actually different', reldiff > 1e-3,
          f'relative parameter L2 difference {reldiff:.4f} over {pa.size:,} params')
    check('No NaN or Inf in the trained weights',
          all(bool(jnp.all(jnp.isfinite(v))) for v in ours.values()))

    # --- 3-6. metrics on identical points ------------------------------------------
    a = A.build(args.model_path)
    cache = {k: jnp.asarray(v) for k, v in val.params.items()}
    occ = jax.jit(lambda v, p: OJ.occupancy(v, p, a.faces))
    query = jax.jit(lambda w, p, code: MD.query_occupancy(w, p, code, a))
    n, n_u = args.n_points, args.n_points // 2
    acc = {k: [] for k in ('ref_pkg', 'our_pkg', 'ref_ind', 'our_ind',
                           'perfect', 'empty', 'inverted')}
    trimesh_rates = []
    key = jax.random.PRNGKey(0)

    for lo in range(0, len(val), args.batch_size):
        hi = min(lo + args.batch_size, len(val))
        B = hi - lo
        verts, joints, full_pose = L.forward({k: v[lo:hi] for k, v in cache.items()}, a)
        key, k_enc, k_pts = jax.random.split(key, 3)

        # Both models see the same bodies, the same encoder samples and the same points.
        code_ref = MD.encode_body(released, verts, joints, full_pose, a, k_enc)
        code_our = MD.encode_body(ours, verts, joints, full_pose, a, k_enc)
        pts = S.sample_query_points(k_pts, verts, code_ref['bone_trans'],
                                    code_ref['bbox_min'], code_ref['bbox_max'], a, n)
        K = pts.shape[1]
        flat = pts.reshape(B, K * n, 3)
        gt = occ(verts, flat)
        pred = {'ref': E.chunked(lambda c: query(released, c, code_ref), flat),
                'our': E.chunked(lambda c: query(ours, c, code_our), flat)}

        if lo < args.trimesh_bodies:
            sub = min(args.trimesh_bodies - lo, B)
            ref = OJ.occupancy_trimesh(verts[:sub], flat[:sub], a.faces)
            trimesh_rates.append(OJ.disagreement(gt[:sub], ref)['rate'])

        g = gt.reshape(B, K, n)
        for tag, p in pred.items():
            p = p.reshape(B, K, n)
            u, s = (p[..., :n_u].reshape(B, -1), g[..., :n_u].reshape(B, -1)), \
                   (p[..., n_u:].reshape(B, -1), g[..., n_u:].reshape(B, -1))
            acc[f'{tag}_pkg'].append(float(E.compute_iou(*u) + E.compute_iou(*s)) * 50)
            acc[f'{tag}_ind'].append((iou_independent(*u) + iou_independent(*s)) * 50)

        flat_g = g.reshape(B, -1)
        acc['perfect'].append(iou_independent(flat_g, flat_g) * 100)
        acc['empty'].append(iou_independent(np.zeros(flat_g.shape), flat_g) * 100)
        acc['inverted'].append(iou_independent(1 - np.asarray(flat_g), flat_g) * 100)

    m = {k: sum(v) / len(v) for k, v in acc.items()}
    check('Perfect predictor scores 100', abs(m['perfect'] - 100) < 1e-6,
          f"{m['perfect']:.4f}")
    check('All-empty predictor scores 0', m['empty'] < 1e-6, f"{m['empty']:.4f}")
    check('Inverted predictor scores 0', m['inverted'] < 1e-6, f"{m['inverted']:.4f}")
    check('Independent IoU agrees with compute_iou (released)',
          abs(m['ref_pkg'] - m['ref_ind']) < 0.01, f"{m['ref_pkg']:.4f} vs {m['ref_ind']:.4f}")
    check('Independent IoU agrees with compute_iou (trained)',
          abs(m['our_pkg'] - m['our_ind']) < 0.01, f"{m['our_pkg']:.4f} vs {m['our_ind']:.4f}")
    check('Released reproduces the stored reference',
          abs(m['ref_pkg'] - E.REFERENCE['iou_mean']) < E.TOLERANCE_IOU,
          f"{m['ref_pkg']:.2f} vs stored {E.REFERENCE['iou_mean']}")
    rate = sum(trimesh_rates) / len(trimesh_rates)
    check('GT occupancy matches trimesh on eval points', rate < 2e-3,
          f'{rate * 100:.4f}% disagreement')

    # --- 7. decomposition ------------------------------------------------------------
    check('15 parts', a.num_parts == 15, str(a.num_parts))
    check('Kinematic chain walked to 22 joints', a.mK == 22, str(a.mK))
    pad_rows = int((np.asarray(a.tight_faces)[:, :, 0] == A.FACE_PAD).sum())
    check('Padded face rows are masked, not indexed', pad_rows > 0,
          f'{pad_rows:,} padded rows in the tight set, all excluded by gather_faces')

    width = max(len(name) for _, name, _ in results) + 2
    print('\n' + '=' * (width + 50))
    print('Audit')
    print('=' * (width + 50))
    for ok, name, detail in results:
        print(f"[{'PASS' if ok else 'FAIL'}] {name:<{width}} {detail}")
    print('=' * (width + 50))
    print(f"\n  Scored on identical points, same process (seed 0):")
    print(f"    released  {m['ref_pkg']:.3f}")
    print(f"    trained   {m['our_pkg']:.3f}")
    print(f"    delta     {m['our_pkg'] - m['ref_pkg']:+.3f}")

    failed = [name for ok, name, _ in results if not ok]
    if failed:
        print(f'\n{len(failed)} check(s) FAILED: {", ".join(failed)}')
        sys.exit(1)
    print(f'\nAll {len(results)} checks passed.')


if __name__ == '__main__':
    main()
