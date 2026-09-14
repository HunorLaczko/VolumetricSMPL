"""Point-to-mesh distance and surface sampling.

The two geometry operations the original package takes from PyTorch3D:
`point_face_distance` (through its `point_mesh_distance` wrapper) and
`sample_points_from_meshes`.

**Why the distance function is only one branch.** PyTorch3D's point-triangle distance
takes a plane projection for healthy triangles and falls back to "closest of the three
edges" for degenerate ones, switching on `area < min_triangle_area` with a default of
5e-3. Every SMPL-X triangle is far below that -- mean face area 7.8e-5 m2, ~64x under --
so *every* triangle takes the fallback, always.

That is not an accident to be corrected. Lowering the threshold makes the SDF error 10x
worse (see FINDINGS.md, "PyTorch3D `min_triangle_area`"), and the released checkpoints
were trained against the fallback. The ground-truth UDF is therefore *defined* by it, and
reproducing the training targets means reproducing the fallback -- not a mathematically
nicer distance. `training/test_parity.py` asserts the premise (max face area vs the
threshold) and the resulting distance scale.

Surface metrics need the true distance instead; see `training/meshmetrics.py`.

Distances are squared internally, matching PyTorch3D; `udf` takes the square root.
"""
from __future__ import annotations

import functools

import jax
import jax.numpy as jnp

# PyTorch3D's `min_triangle_area` default. Defines the ground-truth UDF; not a tunable.
MIN_TRIANGLE_AREA = 5e-3

# PyTorch3D's degenerate-segment guard.
K_EPSILON = 1e-30

# Absent triangles in the padded per-part face tensors.
FACE_PAD = -1


def _point_seg_sq(p, a, b):
    """Squared distance from p to segment [a, b], broadcasting over leading axes."""
    ab = b - a
    l2 = jnp.sum(ab * ab, axis=-1)
    safe = jnp.where(l2 <= K_EPSILON, 1.0, l2)
    t = jnp.sum((p - a) * ab, axis=-1) / safe
    proj = a + jnp.clip(t, 0.0, 1.0)[..., None] * ab
    d = jnp.sum((p - proj) ** 2, axis=-1)
    # Degenerate segment: PyTorch3D returns the distance to the endpoint.
    return jnp.where(l2 <= K_EPSILON, jnp.sum((p - b) ** 2, axis=-1), d)


def point_tri_sq(p, tri):
    """Squared point-triangle distance, degenerate branch only.

    Args:
        p:   (..., 3)
        tri: (..., 3, 3) as (v0, v1, v2)
    """
    v0, v1, v2 = tri[..., 0, :], tri[..., 1, :], tri[..., 2, :]
    return jnp.minimum(
        jnp.minimum(_point_seg_sq(p, v0, v1), _point_seg_sq(p, v1, v2)),
        _point_seg_sq(p, v2, v0))


def triangle_areas(tri):
    """(..., 3, 3) -> (...,) triangle area."""
    v0, v1, v2 = tri[..., 0, :], tri[..., 1, :], tri[..., 2, :]
    return 0.5 * jnp.linalg.norm(jnp.cross(v2 - v0, v1 - v0), axis=-1)


def gather_faces(verts, faces):
    """(V, 3), (F, 3) -> (F, 3, 3) triangles, plus a validity mask for -1 padding.

    Index -1 would silently wrap to the last vertex and produce a plausible-looking wrong
    triangle, so padded rows are clamped to 0 and masked instead of indexed.
    """
    valid = faces[:, 0] != FACE_PAD
    safe = jnp.where(faces < 0, 0, faces)
    return verts[safe], valid


@functools.partial(jax.jit, static_argnames=('chunk',))
def min_sq_distance(points, tri, valid, chunk: int = 1024):
    """Min squared distance from each point to any valid triangle.

    Args:
        points: (P, 3)
        tri:    (F, 3, 3)
        valid:  (F,) bool
    Returns:
        (P,)

    Scanned over triangle chunks with a running minimum. The full pairwise tensor is
    never built: the ground-truth UDF path alone is 8x15x512 points against ~12k
    triangles -- ~735M pairs, ~2.9 GB in float32 -- so materialising it is not an option.
    """
    F = tri.shape[0]
    pad = (-F) % chunk
    if pad:
        tri = jnp.concatenate([tri, jnp.zeros((pad, 3, 3), tri.dtype)], axis=0)
        valid = jnp.concatenate([valid, jnp.zeros((pad,), bool)], axis=0)
    n = tri.shape[0] // chunk
    tri = tri.reshape(n, chunk, 3, 3)
    valid = valid.reshape(n, chunk)

    def step(best, xs):
        t, v = xs
        d = point_tri_sq(points[:, None, :], t[None])         # (P, chunk)
        d = jnp.where(v[None], d, jnp.inf)
        return jnp.minimum(best, d.min(axis=1)), None

    best, _ = jax.lax.scan(step, jnp.full(points.shape[0], jnp.inf), (tri, valid))
    return best


def udf(points, verts, faces, chunk: int = 1024):
    """Unsigned distance from points to a mesh, matching `point_mesh_distance`.

    Args:
        points: (P, 3)
        verts:  (V, 3)
        faces:  (F, 3), -1 padded
    """
    tri, valid = gather_faces(verts, faces)
    return jnp.sqrt(min_sq_distance(points, tri, valid, chunk=chunk))


def udf_per_part(points, verts, part_faces, chunk: int = 1024):
    """Ground-truth UDF for the per-part path.

    Args:
        points:     (B, K, T, 3) in posed space
        verts:      (B, V, 3)
        part_faces: (K, F, 3), -1 padded -- the same face set for every body
    Returns:
        (B, K, T)

    vmapped over bodies and parts rather than looped, so the K part meshes cost one
    kernel. The vertices are shared and only the face index set varies.
    """
    def one_body(v, pts):
        def per_part(f, p):
            tri, valid = gather_faces(v, f)
            return jnp.sqrt(min_sq_distance(p, tri, valid, chunk=chunk))
        return jax.vmap(per_part)(part_faces, pts)

    return jax.vmap(one_body)(verts, points)


# --------------------------------------------------------------------------------------
# Surface sampling
# --------------------------------------------------------------------------------------

def rand_barycentric(key, shape):
    """PyTorch3D's `_rand_barycentric_coords`: uniform over the triangle."""
    k1, k2 = jax.random.split(key)
    u = jnp.sqrt(jax.random.uniform(k1, shape))
    v = jax.random.uniform(k2, shape)
    return 1.0 - u, u * (1.0 - v), u * v


def sample_faces(key, verts, faces, n_samples: int):
    """Area-weighted surface sampling from one -1-padded face set.

    Args:
        verts: (V, 3)
        faces: (F, 3)
    Returns:
        (n_samples, 3)

    Matches `sample_points_from_meshes`: faces drawn with replacement proportional to
    area, then a uniform barycentric point within the chosen triangle. Padded faces get
    zero weight, which is how PyTorch3D's `Meshes` treats -1 rows.
    """
    tri, valid = gather_faces(verts, faces)
    w = jnp.where(valid, triangle_areas(tri), 0.0)

    k_face, k_bary = jax.random.split(key)
    # Inverse-CDF, NOT `jax.random.categorical`. That helper materialises a
    # (n_samples, n_categories) Gumbel array -- for 100k samples over a 285k-triangle
    # extracted mesh that is 28.5G elements, 106 GiB, and it fails. Worse, it is lazy,
    # so the failure surfaces wherever the samples are first *used* and looks like a bug
    # in the consumer. searchsorted over the cumulative area is O(n log F) with no
    # intermediate, and draws from exactly the same distribution.
    cw = jnp.cumsum(w)
    u = jax.random.uniform(k_face, (n_samples,)) * cw[-1]
    idx = jnp.clip(jnp.searchsorted(cw, u), 0, w.shape[0] - 1)
    t = tri[idx]
    w0, w1, w2 = rand_barycentric(k_bary, (n_samples,))
    return (w0[:, None] * t[:, 0] + w1[:, None] * t[:, 1] + w2[:, None] * t[:, 2])


def sample_parts(key, verts, part_faces, n_samples: int):
    """Area-weighted sampling for every (body, part).

    Args:
        verts:      (B, V, 3)
        part_faces: (K, F, 3)
    Returns:
        (B, K, n_samples, 3)
    """
    B, K = verts.shape[0], part_faces.shape[0]
    keys = jax.random.split(key, B * K).reshape(B, K, -1)

    def one_body(v, ks):
        return jax.vmap(lambda f, k: sample_faces(k, v, f, n_samples))(part_faces, ks)

    return jax.vmap(one_body)(verts, keys)
