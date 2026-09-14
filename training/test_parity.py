"""Differential and analytic checks on the geometry the metrics depend on.

1. Ground-truth occupancy is ray-stabbing *parity*, matching COAP's
   `leap.tools.libmesh.check_mesh_contains`. This checks that kaolin's `check_sign`
   implements that definition on real posed bodies by comparing it against trimesh's
   independent CPU implementation. A systematic disagreement near the surface would
   depress IoU_surf while leaving IoU_unif untouched.
2. The ground-truth UDF has the right scale (analytic half-normal).
3. The surface-metric distance is exact on analytic triangles, including the small
   triangle case PyTorch3D gets wrong.
4. Mesh extraction puts the surface where the body is.

  docker compose run --rm train python -m training.test_parity --bodies 4
"""
from __future__ import annotations

import argparse

import torch

from .amass import scan
from .evaluate import build_body
from .occupancy import disagreement, occupancy_kaolin, occupancy_trimesh
from .sampling import sample_query_points


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', default='data/extracted')
    ap.add_argument('--model-path', default='data/body_models')
    ap.add_argument('--bodies', type=int, default=8)
    ap.add_argument('--n-points', type=int, default=512)
    ap.add_argument('--sigma', type=float, default=0.01)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    # Needs no data or GPU, so it runs first and fails fast.
    check_surface_distance()

    cache = scan(args.data_root, ['PosePrior'], 500)
    body = build_body(args.model_path, 'neutral', 'cuda', False)
    faces = torch.from_numpy(body.faces.astype('int64')).cuda()
    gen = torch.Generator(device='cuda').manual_seed(args.seed)

    with torch.no_grad():
        params = cache.batch(0, args.bodies, 'cuda')
        params['expression'] = torch.zeros(args.bodies, body.num_expression_coeffs,
                                           device='cuda')
        out = body(**params, return_verts=True, return_full_pose=True)
        pts = sample_query_points(body.volume, out, args.n_points,
                                  sigma=args.sigma, generator=gen)
        B, K, n, _ = pts.shape
        flat = pts.reshape(B, K * n, 3)

        occ_k = occupancy_kaolin(out.vertices, faces, flat)
        occ_t = occupancy_trimesh(out.vertices, faces, flat)

    n_u = n // 2
    print(f'\n  bodies {B} | {K} parts | {n} pts/part | sigma {args.sigma}\n')
    print(f"  {'subset':<12}{'points':>10}{'disagree':>10}{'rate':>10}"
          f"{'kaolin in':>11}{'trimesh in':>12}")
    print('  ' + '-' * 65)
    for label, sl in (('all', slice(None)), ('uniform', slice(0, n_u)),
                      ('surface', slice(n_u, None))):
        a = occ_k.reshape(B, K, n)[..., sl].reshape(B, -1)
        b = occ_t.reshape(B, K, n)[..., sl].reshape(B, -1)
        d = disagreement(a, b)
        print(f"  {label:<12}{d['n_points']:>10,}{d['n_disagree']:>10,}"
              f"{d['rate'] * 100:>9.3f}%{d['a_inside_frac'] * 100:>10.2f}%"
              f"{d['b_inside_frac'] * 100:>11.2f}%")

    overall = disagreement(occ_k, occ_t)
    print(f"\n  Overall disagreement: {overall['rate'] * 100:.4f}% "
          f"({overall['n_disagree']:,} / {overall['n_points']:,})")
    if overall['a_inside_frac'] > overall['b_inside_frac']:
        print('  kaolin classifies MORE points as inside than trimesh.')
    elif overall['a_inside_frac'] < overall['b_inside_frac']:
        print('  kaolin classifies FEWER points as inside than trimesh.')

    check_udf_scale(body, out, pts[..., n_u:, :], args.sigma)
    check_mesh_extraction(body, out)


def check_udf_scale(body, smpl_output, surface_points, sigma):
    """Assert the ground-truth UDF has the right *scale*, not merely that it runs.

    Jittering a surface point by an isotropic Gaussian puts its distance back to the
    (locally planar) surface at a half-normal: mean = sigma*sqrt(2/pi). That analytic
    identity is what catches a mis-specified `min_triangle_area` — see geometry.py,
    where lowering it collapsed this mean by 10x while still returning 0 for points
    exactly on the surface.
    """
    import numpy as np
    from pytorch3d.structures import Meshes, Pointclouds
    from VolumetricSMPL.volumetric_smpl import point_mesh_distance

    from .geometry import HALF_NORMAL_MEAN

    B = surface_points.shape[0]
    pts = surface_points.reshape(B, -1, 3).contiguous()
    faces = torch.from_numpy(
        body.volume.partitioner.faces[None].astype(np.int64)).to(pts.device)
    with torch.no_grad():
        udf = point_mesh_distance(
            Meshes(verts=smpl_output.vertices, faces=faces.expand(B, -1, -1)),
            Pointclouds(pts))

    got, want = float(udf.mean()), sigma * HALF_NORMAL_MEAN
    rel = abs(got - want) / want
    status = 'OK' if rel < 0.05 else 'FAIL'
    print(f'\n  GT UDF scale: mean {got * 1000:.3f} mm vs analytic '
          f'{want * 1000:.3f} mm ({rel * 100:.1f}% off) [{status}]')
    if rel >= 0.05:
        raise SystemExit('GT UDF scale is wrong — check min_triangle_area (geometry.py)')


def check_surface_distance():
    """Analytic cases for the surface-metric distance (`meshmetrics._p2m`).

    The small-triangle case is the regime where PyTorch3D's `point_face_distance`
    wrongly returns the *plane* distance for a point whose projection falls outside the
    triangle. Here that would read 2.5 mm; the true distance, to the hypotenuse, is
    sqrt(2 * 2.5^2 + 2.5^2) = 4.33 mm.
    """
    import math

    from .meshmetrics import _p2m

    faces = torch.tensor([[0, 1, 2]])
    big = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    small = big * 0.0025                                   # legs of 5 mm, in z = 0
    cases = [
        ('above the interior', big, [0.5, 0.5, 1.0], 1.0),
        ('on the surface', big, [0.5, 0.5, 0.0], 0.0),
        ('beyond a vertex', big, [3.0, 0.0, 0.0], 1.0),
        # Projects to (5, 5) mm, outside; nearest surface point is (2.5, 2.5, 0) mm.
        ('small triangle, outside', small, [0.005, 0.005, 0.0025], math.sqrt(1.875e-5)),
    ]

    print('\n  surface distance, analytic cases:')
    for label, verts, point, want in cases:
        got = float(_p2m(torch.tensor([point]), verts, faces)[0])
        ok = math.isclose(got, want, rel_tol=1e-3, abs_tol=1e-6)
        print(f'    {label:<26} got {got:.6f} m  want {want:.6f} m  '
              f'[{"OK" if ok else "FAIL"}]')
        if not ok:
            raise SystemExit(f'surface distance wrong for "{label}" (meshmetrics._p2m)')


def check_mesh_extraction(body, smpl_output, voxel_mm: float = 4.0,
                          tol_m: float = 0.005):
    """Two cheap invariants on the surface path.

    1. **Self-distance.** Points sampled *on* a mesh are 0 m from it. A distance with a
       degenerate-triangle floor (PyTorch3D's default guard) reads ~1.9 mm here.
    2. **Extent.** The extracted mesh must occupy the same space as the body. Running
       marching cubes on the wrong field returns the whole query cube, and an axis-order
       error in the marching-cubes output squashes the body along the wrong axis; both
       exceed the 5 mm tolerance by far.
    """
    import numpy as np
    from pytorch3d.ops import sample_points_from_meshes
    from pytorch3d.structures import Meshes

    from .meshmetrics import _p2m, extract_mesh

    gv = smpl_output.vertices[0]
    gf = torch.from_numpy(body.faces.astype('int64')).to(gv.device)

    with torch.no_grad():
        on_surface = sample_points_from_meshes(
            Meshes(verts=[gv], faces=[gf]), 20_000)[0].contiguous()
        self_d = float(_p2m(on_surface, gv, gf).mean())
    ok_self = self_d < 1e-5
    print(f'\n  self-distance: {self_d:.3e} m [{"OK" if ok_self else "FAIL"}]')
    if not ok_self:
        raise SystemExit('points on the mesh are not at distance 0 — check '
                         'meshmetrics._p2m')

    # The package's own slicer, rather than rebuilding the output type by hand.
    one = body.volume.batchify_smpl_output(smpl_output)[0]
    with torch.no_grad():
        mesh = extract_mesh(body.volume, one, voxel_mm=voxel_mm)[0]
        body.volume.detach_cache()
    if mesh is None:
        raise SystemExit('extract_mesh found no surface')

    pv = torch.as_tensor(np.asarray(mesh.vertices), dtype=torch.float32, device=gv.device)
    d_ext = (pv.max(0).values - pv.min(0).values) - (gv.max(0).values - gv.min(0).values)
    worst = float(d_ext.abs().max())
    ok_ext = worst < tol_m
    print(f'  mesh extent vs body: worst axis {worst * 1000:.1f} mm '
          f'[{"OK" if ok_ext else "FAIL"}]')
    if not ok_ext:
        raise SystemExit('extracted mesh does not match the body extent — the occupancy '
                         'field is probably wrong (meshmetrics.extract_mesh)')


if __name__ == '__main__':
    main()
