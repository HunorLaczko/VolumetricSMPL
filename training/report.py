"""Compare a trained checkpoint against the released one, optionally logged to W&B.

Every number is paired with the released checkpoint's number produced by the same code
on the same data, and the visual panels show where the two models differ rather than
only by how much.

Produces:
  · headline metrics, 5 seeds each, mean ± sd, and a per-seed table
  · generalisation table across PosePrior / held-out BMLrub / held-out DFaust
  · surface metrics: Chamfer, p95, Hausdorff, watertightness, components
  · occupancy slice images, colour-coded by agreement
  · extracted meshes as interactive 3D panels

  docker compose run --rm train python -m training.report --ckpt <path>
  docker compose run --rm train python -m training.report --ckpt <path> --no-wandb

Run this only when the GPU is free — never alongside a training run.
"""
from __future__ import annotations

import argparse
import os
import statistics as st

import numpy as np
import torch
import trimesh

from .cache import SPLITS, build, load
from .evaluate import REFERENCE, TARGETS, TOLERANCE_IOU, build_body, evaluate
from . import viz
from .meshmetrics import _p2m, compare, extract_mesh, occupancy_slice, topology
from .occupancy import occupancy_kaolin

METRICS = ('iou_mean', 'iou_surf', 'iou_unif', 'mse_sdf', 'mse_abs_sdf')


def multi_seed(body, cache, seeds, **kw):
    runs = [evaluate(body, cache, 'cuda', seed=s, **kw) for s in seeds]
    out = {}
    for k in METRICS:
        v = [r[k] for r in runs]
        out[k] = (st.mean(v), st.stdev(v) if len(v) > 1 else 0.0)
    return out, runs


def get_cache(name, data_root):
    path = f'data/cache/{name}.pt'
    if os.path.exists(path):
        return load(path)[0]
    cache, man = build(name, data_root)
    return cache


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--data-root', default='data/extracted')
    ap.add_argument('--model-path', default='data/body_models')
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    ap.add_argument('--n-mesh-bodies', type=int, default=6)
    # 3D panels carry a lot of geometry; six bodies x five panels would make the report
    # slow to open for no extra insight.
    ap.add_argument('--n-vis-bodies', type=int, default=3)
    ap.add_argument('--wandb-id', default=None,
                    help='append to an existing report run instead of creating one')
    ap.add_argument('--visuals-only', action='store_true',
                    help='regenerate just the 3D panels and slices; skips the headline, '
                         'generalisation and surface table, which are unchanged')
    ap.add_argument('--skip-point-metrics', action='store_true',
                    help='skip the headline and generalisation sections but still '
                         'recompute the surface table — for when the extraction changed '
                         'and the point metrics, which never touch a mesh, did not')
    # One resolution for metrics and visuals. 4 mm is set by the renderer rather than
    # the extractor: marching cubes interpolates along cell edges, so the surface is far
    # finer than the grid and smaller voxels mostly add triangles (4 mm ~290k faces,
    # 2 mm ~1.2M) that the panels must load in a browser. PyTorch3D's kernel also caps at
    # 1024 cells per axis, which floors the voxel near 1.8 mm.
    ap.add_argument('--voxel-mm', type=float, default=4.0)
    # Which scalar field marching cubes runs on; see meshmetrics.extract_mesh.
    ap.add_argument('--field', default='logit',
                    choices=['logit', 'occupancy', 'sdf'])
    # Taubin smoothing for the display meshes only — never for the measured ones. Off by
    # default, so the panels show exactly the mesh that was measured.
    ap.add_argument('--smooth-iters', type=int, default=0)
    ap.add_argument('--project', default='volumetric-smpl')
    ap.add_argument('--name', default='final-report')
    ap.add_argument('--no-wandb', action='store_true')
    args = ap.parse_args()

    # Provenance, so the reader can check *which* model and *which* data produced
    # these numbers rather than taking the filename on trust.
    prov = {}
    try:
        sd = torch.load(args.ckpt, map_location='cpu', weights_only=False)
        prov['ckpt_step'] = sd.get('global_step')
        prov['ckpt_epoch'] = sd.get('epoch')
    except Exception as e:
        prov['ckpt_read_error'] = str(e)
    for split in ('train', 'val'):
        p = f'data/cache/{split}.pt'
        if os.path.exists(p):
            m = load(p)[1]
            prov[f'{split}_hash'] = m['hash'][:16]
            prov[f'{split}_bodies'] = m['n_bodies']

    import wandb
    run = None if args.no_wandb else wandb.init(
        project=args.project, name=args.name, job_type='report',
        id=args.wandb_id, resume='must' if args.wandb_id else None,
        config=dict(ckpt=args.ckpt, seeds=args.seeds, reference=REFERENCE,
                    tolerance=TOLERANCE_IOU, **prov))
    print('provenance: ' + '  '.join(f'{k}={v}' for k, v in prov.items()))

    released = build_body(args.model_path, 'neutral', 'cuda', False)
    ours = build_body(args.model_path, 'neutral', 'cuda', False, ckpt=args.ckpt)
    models = {'released': released, 'ours': ours}
    val = get_cache('val', args.data_root)

    if not (args.visuals_only or args.skip_point_metrics):
        metric_sections(run, wandb, models, released, ours, val, args)
    else:
        print('\n[skipped] headline and generalisation; they are point metrics, '
              'unchanged by anything mesh-related, and cost ~20 min of GPU')

    surface_and_visuals(run, wandb, models, released, ours, val, args)

    if run:
        print(f'\nW&B report: {run.url}')
        run.finish()


def metric_sections(run, wandb, models, released, ours, val, args):
    # ---- headline, 5 seeds each -------------------------------------------
    print('\n=== headline (PosePrior, 316 bodies, %d seeds) ===' % len(args.seeds))
    head, per_seed = {}, {}
    for tag, m in models.items():
        head[tag], per_seed[tag] = multi_seed(m, val, args.seeds)
    print(f"  {'metric':<14}{'released':>22}{'ours':>22}{'delta':>12}")
    for k in METRICS:
        (r_mu, r_sd), (o_mu, o_sd) = head['released'][k], head['ours'][k]
        if k.startswith('iou'):
            fmt, delta = '{:.3f}', f'{o_mu - r_mu:+.3f}'
        else:
            fmt, delta = '{:.3e}', f'{o_mu - r_mu:+.2e}'
        rel = f'{fmt.format(r_mu)} ± {fmt.format(r_sd)}'
        our = f'{fmt.format(o_mu)} ± {fmt.format(o_sd)}'
        print(f'  {k:<14}{rel:>22}{our:>22}{delta:>12}')

    if run:
        for tag in models:
            for k in METRICS:
                run.summary[f'{tag}/{k}'] = head[tag][k][0]
                run.summary[f'{tag}/{k}_sd'] = head[tag][k][1]
        for k in METRICS:
            run.summary[f'delta/{k}'] = head['ours'][k][0] - head['released'][k][0]
        run.summary['gate/within_tolerance'] = bool(
            abs(head['ours']['iou_mean'][0] - REFERENCE['iou_mean']) <= TOLERANCE_IOU)
        run.summary['gate/at_least_reference'] = bool(
            head['ours']['iou_mean'][0] >= REFERENCE['iou_mean'] - TOLERANCE_IOU)

        t = wandb.Table(columns=['model', 'seed', *METRICS])
        for tag, runs in per_seed.items():
            for s, r in zip(args.seeds, runs):
                t.add_data(tag, s, *[r[k] for k in METRICS])
        run.log({'per_seed': t})

        cmp = wandb.Table(columns=['metric', 'released', 'ours', 'delta', 'paper'])
        for k in METRICS:
            cmp.add_data(k, head['released'][k][0], head['ours'][k][0],
                         head['ours'][k][0] - head['released'][k][0], TARGETS[k])
        run.log({'headline': cmp})

    # ---- generalisation ----------------------------------------------------
    print('\n=== generalisation (single seed) ===')
    # The rows answer different questions and must not be averaged or read as one
    # trend — see the SPLITS comments in cache.py.
    gen = wandb.Table(columns=['split', 'bodies', 'what_it_tests',
                               'released', 'ours', 'delta']) if run else None
    scores = {}
    for split, seen in (
            ('val', 'tuning set — unseen, but every gate was read off it'),
            ('holdout_bmlrub', 'GENERALISATION: unseen subjects and motion style'),
            ('seen_dfaust', 'memorisation pair A: frames that WERE trained on'),
            ('holdout_dfaust', 'memorisation pair B: adjacent frames, never trained on')):
        try:
            c = get_cache(split, args.data_root)
        except Exception as e:
            print(f'  {split:<18} skipped: {e}')
            continue
        r = evaluate(released, c, 'cuda', seed=0)['iou_mean']
        o = evaluate(ours, c, 'cuda', seed=0)['iou_mean']
        scores[split] = (r, o)
        print(f'  {split:<18} {len(c):>6} bodies   released {r:6.2f}   ours {o:6.2f}'
              f'   delta {o - r:+.2f}')
        if gen is not None:
            gen.add_data(split, len(c), seen, r, o, o - r)
    if gen is not None:
        run.log({'generalisation': gen})

    # The memorisation question, answered rather than left to the reader. Same
    # sequences, adjacent frames, matched on subject and shape; for *our* model the only
    # difference is whether the frame was in training, so a gap is memorisation in IoU
    # points and a gap near zero rules it out. That reading stands on its own.
    #
    # The released model's gap is a difficulty control, NOT a no-memorisation baseline:
    # its training set is unknown, and if it used DFaust at any stride dividing 500 it
    # has the same seen/unseen structure and could show its own memorisation. What it
    # does establish is whether frame-0 poses are intrinsically easier than frame-3
    # ones — if its gap is ~0, the poses are equally hard and ours is a clean read.
    if 'seen_dfaust' in scores and 'holdout_dfaust' in scores:
        gap = scores['seen_dfaust'][1] - scores['holdout_dfaust'][1]
        ctrl = scores['seen_dfaust'][0] - scores['holdout_dfaust'][0]
        print(f'\n  seen - unseen, ours         {gap:+.3f} IoU  <- memorisation if > 0')
        print(f'  seen - unseen, released     {ctrl:+.3f} IoU  <- difficulty control')
        if run:
            run.summary['memorisation/gap_ours'] = gap
            run.summary['memorisation/gap_released_control'] = ctrl


def surface_and_visuals(run, wandb, models, released, ours, val, args):
    # ---- surface metrics + visuals ----------------------------------------
    print(f'\n=== surface metrics (box-fitted grid, {args.voxel_mm} mm voxel) ===')
    faces_gt = torch.from_numpy(released.faces.astype('int64')).cuda()
    surf = wandb.Table(columns=['body', 'model', 'chamfer_mm', 'gt_to_pred_mm',
                                'pred_to_gt_mm', 'hausdorff_mm', 'p95_mm',
                                'watertight', 'n_components', 'stray_area_mm2',
                                'volume_l']) \
        if run and not args.visuals_only else None
    images, objects = {}, {}

    n_bodies = args.n_vis_bodies if args.visuals_only else args.n_mesh_bodies
    for i in range(n_bodies):
        params = val.batch(i, i + 1, 'cuda')
        params['expression'] = torch.zeros(1, released.num_expression_coeffs, device='cuda')
        with torch.no_grad():
            out = released(**params, return_verts=True, return_full_pose=True)
            gt_v = out.vertices[0]
            gt_fn = lambda p: occupancy_kaolin(out.vertices, faces_gt, p)

            # The ground truth is the SMPL-X body itself — the surface both models are
            # approximating. Plotted on the same footing so the comparison is complete.
            vis = {}
            if run and i < args.n_vis_bodies:
                gv, gf = gt_v.cpu().numpy(), released.faces.astype('int64')
                vis['gt'] = (gv, gf)
                objects[f'mesh3d/body{i}/gt'] = viz.to_wandb(
                    viz.single(gv, gf, f'body {i} — ground truth (SMPL-X)',
                               viz.COLORS['gt']), f'body{i}/gt')

            panels = []
            for tag, m in models.items():
                # training.meshmetrics.extract_mesh, NOT the package's — the inherited
                # one runs marching cubes on logit(SDF). See the docstring there.
                mesh = extract_mesh(m.volume, out, voxel_mm=args.voxel_mm,
                                    field=args.field)[0]
                m.volume.detach_cache()

                # Appended before anything that can fail, so a bad mesh cannot leave the
                # slice strip with a missing panel.
                panels.append(occupancy_slice(m.volume, out, gt_fn, axis=2))
                if mesh is None:
                    print(f'  body {i} {tag:<9} no surface found — skipped')
                    continue

                vv, ff = np.asarray(mesh.vertices), np.asarray(mesh.faces)
                if not args.visuals_only:
                    pv = torch.as_tensor(vv, dtype=torch.float32, device='cuda')
                    pf = torch.as_tensor(ff, dtype=torch.int64, device='cuda')
                    d = compare(pv, pf, gt_v, faces_gt)
                    topo = topology(mesh)
                    print(f'  body {i} {tag:<9} {len(vv):>7,}v  '
                          f'chamfer {d["chamfer_mm"]:6.3f} mm  p95 {d["p95_mm"]:6.3f} mm  '
                          f'hausdorff {d["hausdorff_mm"]:7.3f} mm  '
                          f'comp {topo["n_components"]:>3} '
                          f'(stray {topo["stray_area_mm2"]:7.1f} mm2)  '
                          f'vol {topo["volume_l"]:6.1f} L')
                    if surf is not None:
                        surf.add_data(i, tag, d['chamfer_mm'], d['gt_to_pred_mm'],
                                      d['pred_to_gt_mm'], d['hausdorff_mm'], d['p95_mm'],
                                      topo['watertight'], topo['n_components'],
                                      topo['stray_area_mm2'], topo['volume_l'])

                if run and i < args.n_vis_bodies:
                    # Display copy only. Metrics above used the raw mesh, deliberately —
                    # see --smooth-iters.
                    disp = trimesh.smoothing.filter_taubin(
                        mesh.copy(), iterations=args.smooth_iters) \
                        if args.smooth_iters else mesh
                    dv, df = np.asarray(disp.vertices), np.asarray(disp.faces)
                    suffix = (f' (display: Taubin x{args.smooth_iters})'
                              if args.smooth_iters else '')
                    vis[tag] = (dv, df)
                    objects[f'mesh3d/body{i}/{tag}'] = viz.to_wandb(
                        viz.single(dv, df, f'body {i} — {tag}{suffix}',
                                   viz.COLORS[tag]), f'body{i}/{tag}')
                    # Where the residual error actually is — a uniform clay colour cannot
                    # show it. Distances are recomputed on the displayed vertices so the
                    # colours match the geometry being shown.
                    d_v = _p2m(torch.as_tensor(dv, dtype=torch.float32, device='cuda'),
                               gt_v, faces_gt).cpu().numpy() * 1000
                    objects[f'mesh3d/body{i}/{tag}_error'] = viz.to_wandb(
                        viz.error_map(dv, df, d_v,
                                      f'body {i} — {tag}, distance to GT (mm){suffix}'),
                        f'body{i}/{tag}_error')

            if run and len(vis) > 1:
                objects[f'mesh3d/body{i}/overlay'] = viz.to_wandb(viz.overlay(
                    vis, f'body {i} — all three, toggle from the legend'),
                    f'body{i}/overlay')

            if run:
                strip = np.concatenate(panels, axis=1)
                images[f'slices/body{i}'] = wandb.Image(
                    strip, caption=f'body {i} — left: released, right: ours. '
                                   'green=correct, red=missed, blue=hallucinated')

    if run:
        if surf is not None:
            run.log({'surface': surf})
        run.log(images)
        run.log(objects)
        legend = np.zeros((60, 480, 3), dtype=np.uint8)
        legend[:, :120] = (60, 200, 90); legend[:, 120:240] = (220, 60, 60)
        legend[:, 240:360] = (70, 120, 240); legend[:, 360:] = 32
        run.log({'slices/legend': wandb.Image(
            legend, caption='green correct · red missed · blue hallucinated · grey outside')})


if __name__ == '__main__':
    main()
