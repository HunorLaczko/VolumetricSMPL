"""Surface-level comparison of an extracted mesh against the ground-truth body.

IoU and SDF MSE are point-sampled, so they are blind to failures that occupy little
probability mass -- a detached fragment beside a hand, a hole in an armpit, a surface
that is ragged only at part seams. These metrics look at the surface itself.

----------------------------------------------------------------------------------------
**This module needs a different distance function from `VolumetricSMPL/geometry.py`, and
the difference is not cosmetic.**

`geometry.udf` implements only PyTorch3D's *degenerate* branch, which is correct there:
the training and evaluation targets are computed at `min_triangle_area = 5e-3`, every
SMPL-X triangle falls under it, and lowering the threshold makes the SDF error 10x worse.
That guard is part of the eval protocol.

Surface comparison asks a different question and deliberately uses `1e-12`. Under the
5e-3 guard, points sampled exactly *on* the body come back 1.9 mm away from it, so a
surface metric built on it would have a 1.9 mm floor. Below the threshold the
plane-projection branch runs and the floor collapses to ~1e-9 m.

So this module implements the **full two-branch** distance. The two constants answer two
questions and must not be unified. It matches an exact float64 reference; PyTorch3D's own
two-branch kernel does not at SMPL-X scale (it underestimates ~2x; see FINDINGS.md).
"""
from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np

from VolumetricSMPL import geometry as G
from VolumetricSMPL.mesh import box_grid, extract_mesh  # noqa: F401  (re-exported)

# Well below any real SMPL-X face area, so the plane branch is the one that runs.
# Deliberately NOT geometry.MIN_TRIANGLE_AREA -- see the module docstring.
SURFACE_MIN_TRIANGLE_AREA = 1e-12

EPS = 1e-30


def _inside_triangle(p, v0, v1, v2, n):
    """Containment test for a point already projected onto the triangle's plane.

    Edge-sign rather than barycentric-by-Gram-determinant: no division to guard, and the
    three cross terms are all the same magnitude as the triangle's own normal, so there is
    nothing to cancel. The sign convention does not matter -- both orientations are
    accepted. (A barycentric version gave identical numbers here; this form is kept
    because it is cheaper and harder to get wrong.)
    """
    c0 = jnp.sum(jnp.cross(v1 - v0, p - v0) * n, -1)
    c1 = jnp.sum(jnp.cross(v2 - v1, p - v1) * n, -1)
    c2 = jnp.sum(jnp.cross(v0 - v2, p - v2) * n, -1)
    return ((c0 >= 0) & (c1 >= 0) & (c2 >= 0)) | ((c0 <= 0) & (c1 <= 0) & (c2 <= 0))


def point_tri_sq_full(p, tri, min_area: float = SURFACE_MIN_TRIANGLE_AREA):
    """Squared point-triangle distance, both branches.

    Plane projection when the triangle is non-degenerate *and* the projection lands
    inside it; otherwise the closest of the three edges.
    """
    v0, v1, v2 = tri[..., 0, :], tri[..., 1, :], tri[..., 2, :]
    n = jnp.cross(v2 - v0, v1 - v0)
    ln = jnp.linalg.norm(n, axis=-1)
    area = 0.5 * ln

    nn = n / jnp.where(ln <= EPS, 1.0, ln)[..., None]
    d_plane = jnp.sum((p - v0) * nn, -1)
    proj = p - d_plane[..., None] * nn

    d_edge = jnp.minimum(
        jnp.minimum(G._point_seg_sq(p, v0, v1), G._point_seg_sq(p, v1, v2)),
        G._point_seg_sq(p, v2, v0))

    use_plane = (area >= min_area) & _inside_triangle(proj, v0, v1, v2, n)
    return jnp.where(use_plane, d_plane ** 2, d_edge)


@functools.partial(jax.jit, static_argnames=('chunk',))
def _min_sq_full(points, tri, valid, chunk: int = 1024):
    F = tri.shape[0]
    pad = (-F) % chunk
    if pad:
        tri = jnp.concatenate([tri, jnp.zeros((pad, 3, 3), tri.dtype)], axis=0)
        valid = jnp.concatenate([valid, jnp.zeros((pad,), bool)], axis=0)
    n = tri.shape[0] // chunk
    tri, valid = tri.reshape(n, chunk, 3, 3), valid.reshape(n, chunk)

    def step(best, xs):
        t, v = xs
        d = point_tri_sq_full(points[:, None, :], t[None])
        return jnp.minimum(best, jnp.where(v[None], d, jnp.inf).min(axis=1)), None

    best, _ = jax.lax.scan(step, jnp.full(points.shape[0], jnp.inf), (tri, valid))
    return best


def surface_distance(points, verts, faces, chunk: int = 1024,
                     point_chunk: int = 8192):
    """Unsigned distance from points to a mesh, with the surface-metric guard.

    Blocked over **both** axes. Chunking triangles alone is not enough here: an
    extracted mesh at a 4 mm voxel carries ~300k triangles, and the per-pair
    intermediates (`p - v0`, `cross(s, e1)`) are 3-vectors, so 100k points against one
    1024-triangle block is already ~1.2 GB and the full product is ~106 GiB. Blocking
    points too bounds the working set at `point_chunk x chunk x 3`.
    """
    tri, valid = G.gather_faces(verts, faces)
    points = jnp.asarray(points)
    out = []
    for i in range(0, points.shape[0], point_chunk):
        # Each block is pulled to host immediately. Returning device arrays and
        # concatenating at the end let XLA fuse the whole Python loop back into one
        # program, re-forming the full points x triangles product this blocking exists
        # to avoid -- 106 GiB for a 100k-point cloud against a 285k-triangle mesh.
        # Copying ~8k floats per block is free by comparison.
        out.append(np.asarray(_min_sq_full(points[i:i + point_chunk], tri, valid,
                                           chunk=chunk)))
    return np.sqrt(np.concatenate(out))


# --------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------

def chamfer(pred_pts, gt_verts, gt_faces, gt_pts, pred_verts, pred_faces):
    """Symmetric point-to-surface distance, in millimetres.

    Both directions matter and measure different failures: pred->gt catches material
    where the body is not, gt->pred catches missing material.
    """
    d1 = surface_distance(pred_pts, gt_verts, gt_faces) * 1e3
    d2 = surface_distance(gt_pts, pred_verts, pred_faces) * 1e3
    both = np.concatenate([d1, d2])
    return dict(chamfer_mm=float(both.mean()),
                p95_mm=float(np.percentile(both, 95)),
                hausdorff_mm=float(both.max()))


def topology(verts, faces):
    """Component count and the surface area outside the largest one.

    Count alone is a poor summary: a single stray triangle and a detached forearm both
    read as "2 components". Stray *area* separates them.
    """
    import trimesh

    m = trimesh.Trimesh(np.asarray(verts), np.asarray(faces), process=False)
    comps = m.split(only_watertight=False)
    if not len(comps):
        return dict(n_components=0, stray_area_mm2=0.0, watertight=False)
    areas = np.array([c.area for c in comps])
    return dict(n_components=int(len(comps)),
                stray_area_mm2=float((areas.sum() - areas.max()) * 1e6),
                watertight=bool(m.is_watertight))
