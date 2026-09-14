"""Checks on the geometry the metrics depend on, against analytic answers or an
independent implementation.

1. The surface-metric distance is exact on analytic triangles, including the small
   triangle case PyTorch3D gets wrong.
2. Ground-truth occupancy (ray-stabbing parity) agrees with trimesh's independent CPU
   implementation on real posed bodies. A systematic disagreement near the surface would
   depress IoU_surf while leaving IoU_unif untouched.
3. The ground-truth UDF has the right scale, and every SMPL-X triangle is under
   `min_triangle_area`, which is what makes that UDF the degenerate-branch distance.
4. Mesh extraction puts the surface where the body is.

  docker compose run --rm jax python -m training.test_parity --bodies 4
"""
from __future__ import annotations

import argparse
import math

import jax
import jax.numpy as jnp
import numpy as np

from VolumetricSMPL import assets as A
from VolumetricSMPL import checkpoint as C
from VolumetricSMPL import geometry as G
from VolumetricSMPL import lbs as L
from VolumetricSMPL import model as MD

from . import cache as CA
from . import meshmetrics as MM
from . import occupancy as OJ
from . import sampling as S

# Mean distance back to a locally planar surface after isotropic Gaussian jitter, in
# units of sigma: the half-normal mean.
HALF_NORMAL_MEAN = math.sqrt(2.0 / math.pi)

# Mean UDF of points sampled exactly on the body. Not 0: the degenerate branch measures
# to the nearest *edge*. A different value means the distance definition changed.
ON_SURFACE_UDF_MM = 1.91


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-path', default='data/body_models')
    ap.add_argument('--cache', default='val')
    ap.add_argument('--bodies', type=int, default=8)
    ap.add_argument('--n-points', type=int, default=512)
    ap.add_argument('--sigma', type=float, default=0.01)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    # Needs no data, so it runs first and fails fast.
    check_surface_distance()

    a = A.build(args.model_path)
    batch = {k: v[:args.bodies] for k, v in CA.load_params(args.cache).items()}
    verts, joints, full_pose = L.forward(batch, a)
    bone_trans = jnp.linalg.inv(MD.compute_abs_transformations(full_pose, joints, a))
    bbox_min, bbox_max = MD.bbox_bounds(verts, bone_trans, a)
    pts = S.sample_query_points(jax.random.PRNGKey(args.seed), verts, bone_trans,
                                bbox_min, bbox_max, a, args.n_points, sigma=args.sigma)
    B, K, n, _ = pts.shape
    flat = pts.reshape(B, K * n, 3)

    occ_j = np.asarray(OJ.occupancy(verts, flat, a.faces))
    occ_t = OJ.occupancy_trimesh(verts, flat, a.faces)

    n_u = n // 2
    print(f'\n  bodies {B} | {K} parts | {n} pts/part | sigma {args.sigma}\n')
    print(f"  {'subset':<12}{'points':>10}{'disagree':>10}{'rate':>10}"
          f"{'parity in':>11}{'trimesh in':>12}")
    print('  ' + '-' * 65)
    for label, sl in (('all', slice(None)), ('uniform', slice(0, n_u)),
                      ('surface', slice(n_u, None))):
        d = OJ.disagreement(occ_j.reshape(B, K, n)[..., sl], occ_t.reshape(B, K, n)[..., sl])
        print(f"  {label:<12}{d['n_points']:>10,}{d['n_disagree']:>10,}"
              f"{d['rate'] * 100:>9.3f}%{d['a_inside_frac'] * 100:>10.2f}%"
              f"{d['b_inside_frac'] * 100:>11.2f}%")
    overall = OJ.disagreement(occ_j, occ_t)
    ok = overall['rate'] < 2e-3
    print(f"\n  Overall disagreement: {overall['rate'] * 100:.4f}% "
          f"({overall['n_disagree']:,} / {overall['n_points']:,}) [{'OK' if ok else 'FAIL'}]")
    if not ok:
        raise SystemExit('parity occupancy disagrees with trimesh beyond edge-grazing '
                         'ambiguity (training/occupancy.py)')

    check_udf_scale(a, verts, pts[..., n_u:, :], args.sigma)
    check_mesh_extraction(a, verts, joints, full_pose)


def check_surface_distance():
    """Analytic cases for the surface-metric distance (`meshmetrics.surface_distance`).

    The small-triangle case is the regime where PyTorch3D's `point_face_distance`
    wrongly returns the *plane* distance for a point whose projection falls outside the
    triangle. Here that would read 2.5 mm; the true distance, to the hypotenuse, is
    sqrt(2 * 2.5^2 + 2.5^2) = 4.33 mm.
    """
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    big = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=np.float32)
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
        got = float(MM.surface_distance(np.array([point], np.float32), verts, faces)[0])
        ok = math.isclose(got, want, rel_tol=1e-3, abs_tol=1e-6)
        print(f'    {label:<26} got {got:.6f} m  want {want:.6f} m  '
              f'[{"OK" if ok else "FAIL"}]')
        if not ok:
            raise SystemExit(f'surface distance wrong for "{label}" '
                             '(meshmetrics.surface_distance)')


def check_udf_scale(a, verts, surface_points, sigma):
    """Assert the ground-truth UDF has the right *scale*, not merely that it runs.

    Jittering a surface point by an isotropic Gaussian puts its distance back to the
    (locally planar) surface at a half-normal: mean = sigma*sqrt(2/pi). That identity is
    what catches a mis-specified `min_triangle_area`: lowering it collapsed this mean by
    10x while still returning 0 for points exactly on the surface.
    """
    tri, valid = G.gather_faces(verts[0], a.faces)
    max_area = float(jnp.max(G.triangle_areas(tri)))
    ok = max_area < G.MIN_TRIANGLE_AREA
    print(f'\n  max face area {max_area:.2e} m2 vs min_triangle_area '
          f'{G.MIN_TRIANGLE_AREA:.0e} [{"OK" if ok else "FAIL"}]')
    if not ok:
        raise SystemExit('a triangle reaches min_triangle_area, so the ground-truth UDF '
                         'is no longer the degenerate-branch distance (geometry.py)')

    udf = jax.vmap(lambda v, p: G.udf(p, v, a.faces))(
        verts, surface_points.reshape(verts.shape[0], -1, 3))
    got, want = float(jnp.mean(udf)), sigma * HALF_NORMAL_MEAN
    rel = abs(got - want) / want
    ok = rel < 0.05
    print(f'  GT UDF scale: mean {got * 1000:.3f} mm vs analytic {want * 1000:.3f} mm '
          f'({rel * 100:.1f}% off) [{"OK" if ok else "FAIL"}]')
    if not ok:
        raise SystemExit('GT UDF scale is wrong — check min_triangle_area (geometry.py)')

    on_surface = G.sample_faces(jax.random.PRNGKey(1), verts[0], a.faces, 20_000)
    d_on = float(jnp.mean(G.udf(on_surface, verts[0], a.faces))) * 1000
    ok = abs(d_on - ON_SURFACE_UDF_MM) < 0.25
    print(f'  GT UDF of on-surface points: {d_on:.2f} mm vs {ON_SURFACE_UDF_MM} mm '
          f'[{"OK" if ok else "FAIL"}]')
    if not ok:
        raise SystemExit('on-surface UDF changed — the distance definition is different '
                         '(geometry.py)')


def check_mesh_extraction(a, verts, joints, full_pose, voxel_mm: float = 4.0,
                          tol_m: float = 0.005):
    """Two cheap invariants on the surface path.

    1. **Self-distance.** Points sampled *on* a mesh are 0 m from it. A distance with a
       degenerate-triangle floor (PyTorch3D's default guard) reads ~1.9 mm here.
    2. **Extent.** The extracted mesh must occupy the same space as the body. Running
       marching cubes on the wrong field returns the whole query cube, and an axis-order
       error in the marching-cubes output squashes the body along the wrong axis; both
       exceed the 5 mm tolerance by far.
    """
    gv = verts[0]
    on_surface = G.sample_faces(jax.random.PRNGKey(2), gv, a.faces, 20_000)
    self_d = float(MM.surface_distance(on_surface, gv, a.faces).mean())
    ok = self_d < 1e-5
    print(f'\n  self-distance: {self_d:.3e} m [{"OK" if ok else "FAIL"}]')
    if not ok:
        raise SystemExit('points on the mesh are not at distance 0 — check '
                         'meshmetrics.surface_distance')

    weights = C.load_weights('released')
    code = MD.encode_body(weights, verts[:1], joints[:1], full_pose[:1], a,
                          jax.random.PRNGKey(3))
    query = jax.jit(lambda w, p, c: MD.query_occupancy(w, p, c, a))
    v0 = np.asarray(gv)
    pv, _, info = MM.extract_mesh(lambda p: query(weights, p, code),
                                  v0.min(0), v0.max(0), voxel_mm=voxel_mm)
    if pv is None:
        raise SystemExit(f'extract_mesh found no surface ({info})')

    worst = float(np.abs((pv.max(0) - pv.min(0)) - (v0.max(0) - v0.min(0))).max())
    ok = worst < tol_m
    print(f'  mesh extent vs body: worst axis {worst * 1000:.1f} mm '
          f'[{"OK" if ok else "FAIL"}]')
    if not ok:
        raise SystemExit('extracted mesh does not match the body extent — the occupancy '
                         'field is probably wrong (VolumetricSMPL/mesh.py)')


if __name__ == '__main__':
    main()
