"""Report: headline metrics, surface metrics and 3D panels for one set of weights.

Marching cubes runs on the host via scikit-image (see `VolumetricSMPL/mesh.py`), so the
surface numbers are a fact about this extractor as well as about the model.

  docker compose run --rm jax python -m training.report --weights released
  docker compose run --rm jax python -m training.report --weights <last.npz> --no-wandb
"""
from __future__ import annotations

import argparse
import statistics as st

import jax
import jax.numpy as jnp
import numpy as np

from VolumetricSMPL import assets as A
from VolumetricSMPL import checkpoint as C
from VolumetricSMPL import geometry as G
from VolumetricSMPL import lbs as L
from VolumetricSMPL import model as MD

from . import cache as CA
from . import evaluate as E
from . import meshmetrics as MM
from . import viz


def body_meshes(params, a, cache, idx: int, key, voxel_mm: float, field: str):
    """Extract one body's predicted mesh and return it with the ground truth."""
    batch = {k: v[idx:idx + 1] for k, v in cache.items()}
    verts, joints, full_pose = L.forward(batch, a)
    impl = MD.encode_body(params, verts, joints, full_pose, a, key)

    v0 = np.asarray(verts[0])
    query = jax.jit(lambda w, pts, code: MD.query_occupancy(w, pts, code, a))

    pv, pf, info = MM.extract_mesh(lambda pts: query(params, pts, impl),
                                   v0.min(0), v0.max(0), voxel_mm=voxel_mm, field=field)
    return dict(gt_verts=v0, gt_faces=np.asarray(a.faces),
                pred_verts=pv, pred_faces=pf, info=info)


def surface_row(m, key, n_sample: int = 100_000):
    """Chamfer / p95 / Hausdorff plus topology for one extracted mesh."""
    if m['pred_verts'] is None:
        return dict(empty=True, **m['info'])

    gt_v = jnp.asarray(m['gt_verts'])
    gt_f = jnp.asarray(m['gt_faces'])
    pr_v = jnp.asarray(m['pred_verts'], dtype=jnp.float32)
    pr_f = jnp.asarray(m['pred_faces'], dtype=jnp.int32)

    k1, k2 = jax.random.split(key)
    gt_pts = G.sample_faces(k1, gt_v, gt_f, n_sample)
    pr_pts = G.sample_faces(k2, pr_v, pr_f, n_sample)

    out = MM.chamfer(pr_pts, gt_v, gt_f, gt_pts, pr_v, pr_f)
    out.update(MM.topology(m['pred_verts'], m['pred_faces']))
    out['n_verts'] = int(m['pred_verts'].shape[0])
    out['empty'] = False
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-path', default='data/body_models')
    ap.add_argument('--weights', default='released',
                    help="'released', a .ckpt, or an .npz from training.train")
    ap.add_argument('--cache', default='val')
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    ap.add_argument('--n-mesh-bodies', type=int, default=6)
    # 4 mm is set by the renderer rather than the extractor: marching cubes interpolates
    # along cell edges, so the surface is far finer than the grid, and smaller voxels
    # mostly add triangles the panels must load in a browser.
    ap.add_argument('--voxel-mm', type=float, default=4.0)
    ap.add_argument('--field', default='logit', choices=['logit', 'occupancy'])
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--project', default='volumetric-smpl')
    ap.add_argument('--name', default='report')
    ap.add_argument('--no-wandb', action='store_true')
    ap.add_argument('--skip-point-metrics', action='store_true')
    args = ap.parse_args()

    from VolumetricSMPL import current_matmul_precision
    a = A.build(args.model_path)
    params = C.load_weights(args.weights)
    cache = CA.load_params(args.cache)
    print(f'matmul precision: {current_matmul_precision()}   weights: {args.weights}')

    run = None
    if not args.no_wandb:
        import wandb
        run = wandb.init(project=args.project, name=args.name, job_type='report',
                         config=dict(weights=args.weights, voxel_mm=args.voxel_mm,
                                     field=args.field, seeds=args.seeds))

    # --- headline ---------------------------------------------------------------
    if not args.skip_point_metrics:
        runs = [E.evaluate(params, a, cache, jax.random.PRNGKey(s),
                           batch_size=args.batch_size)
                for s in args.seeds]
        passed = E.report(runs)
        if run:
            for k in ('iou_mean', 'iou_surf', 'iou_unif', 'mse_sdf', 'mse_abs_sdf'):
                run.summary[f'metrics/{k}'] = st.mean(r[k] for r in runs)
            run.summary['gate/passed'] = passed
    else:
        print('\n[skipped] point metrics')

    # --- surface + panels -------------------------------------------------------
    print(f'\n=== surface metrics (box-fitted grid, {args.voxel_mm} mm voxel, '
          f'field={args.field}) ===')
    rows = []
    for i in range(args.n_mesh_bodies):
        m = body_meshes(params, a, cache, i, jax.random.PRNGKey(1000 + i),
                        args.voxel_mm, args.field)
        r = surface_row(m, jax.random.PRNGKey(2000 + i))
        rows.append(r)
        if r.get('empty'):
            print(f'  body {i}  EMPTY  (field range {r["vmin"]:.3g}..{r["vmax"]:.3g}, '
                  f'level {r["level"]})')
            continue
        print(f"  body {i}  {r['n_verts']:>7,}v  chamfer {r['chamfer_mm']:6.3f} mm  "
              f"p95 {r['p95_mm']:6.3f} mm  hausdorff {r['hausdorff_mm']:7.3f} mm  "
              f"comp {r['n_components']:>3} (stray {r['stray_area_mm2']:7.1f} mm2)")

        if run:
            fig = viz.overlay({'gt': (m['gt_verts'], m['gt_faces']),
                               'ours': (m['pred_verts'], m['pred_faces'])},
                              f'body {i}: ground truth vs prediction')
            run.log({f'body{i}/overlay': viz.to_wandb(fig, f'body{i}/overlay')})
            fig = viz.single(m['pred_verts'], m['pred_faces'], f'body {i}: prediction',
                             viz.COLORS['ours'])
            run.log({f'body{i}/prediction': viz.to_wandb(fig, f'body{i}/prediction')})

    ok = [r for r in rows if not r.get('empty')]
    if ok:
        print(f"\n  mean over {len(ok)} bodies: chamfer "
              f"{st.mean(r['chamfer_mm'] for r in ok):.3f} mm  "
              f"p95 {st.mean(r['p95_mm'] for r in ok):.3f} mm  "
              f"hausdorff {st.mean(r['hausdorff_mm'] for r in ok):.3f} mm")
    if run:
        print(f'\nW&B report: {run.url}')
        run.finish()


if __name__ == '__main__':
    main()
