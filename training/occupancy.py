"""Ground-truth occupancy by ray-stabbing parity, and an independent reference.

Inside/outside is decided by **parity**, the definition COAP used via
`leap.tools.libmesh.check_mesh_contains` (and kaolin's `check_sign` implements). It is not
the same as a generalized-winding-number test on a self-intersecting mesh: where an arm
passes through the torso, parity reports the doubly-covered region as outside and a
winding number reports it as inside. Posed SMPL-X bodies do self-intersect, so the choice
changes the training targets.

No acceleration structure. The vertices move every step, so a BVH would have to be
refitted every step, and a refit costs more than the whole brute-force pass. Brute force
is cheap here because **the ray direction is a constant**: most of Moller-Trumbore is
per-triangle and hoists out of the point loop, leaving ~25 flops per (point, triangle)
pair in one fused reduction. It runs inside the jitted training step, ~10 ms per batch
of 8.
"""
from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np

from VolumetricSMPL import geometry as G

# Deliberately not axis-aligned: an axis-aligned ray hits far more edges and vertices
# exactly, and each such hit is a coin flip on whether the crossing counts once or twice.
RAY_DIR = jnp.array([0.5773502, 0.5773503, 0.5773504], dtype=jnp.float32)

EPS = 1e-9


def _precompute(verts, faces, d):
    """Per-triangle constants that do not depend on the query point.

    With a fixed ray direction, `h = d x e2`, `a = e1.h` and `f = 1/a` are properties of
    the triangle alone. Hoisting them out is what makes brute force affordable: the
    per-pair work drops to a few dot products.
    """
    tri, valid = G.gather_faces(verts, faces)
    v0 = tri[:, 0]
    e1 = tri[:, 1] - v0
    e2 = tri[:, 2] - v0
    h = jnp.cross(d, e2)
    a = jnp.sum(e1 * h, axis=-1)
    parallel = jnp.abs(a) < EPS
    f = jnp.where(parallel, 0.0, 1.0 / jnp.where(parallel, 1.0, a))
    return v0, e1, e2, h, f, valid & ~parallel


@functools.partial(jax.jit, static_argnames=('chunk',))
def _count_crossings(points, v0, e1, e2, h, f, valid, d, chunk: int = 2048):
    """Number of forward ray-triangle intersections per point."""
    F = v0.shape[0]
    pad = (-F) % chunk
    if pad:
        z3 = jnp.zeros((pad, 3), v0.dtype)
        v0, e1, e2, h = (jnp.concatenate([x, z3]) for x in (v0, e1, e2, h))
        f = jnp.concatenate([f, jnp.zeros((pad,), f.dtype)])
        valid = jnp.concatenate([valid, jnp.zeros((pad,), bool)])
    n = v0.shape[0] // chunk
    v0, e1, e2, h = (x.reshape(n, chunk, 3) for x in (v0, e1, e2, h))
    f, valid = f.reshape(n, chunk), valid.reshape(n, chunk)

    def step(count, xs):
        _v0, _e1, _e2, _h, _f, _ok = xs
        s = points[:, None, :] - _v0[None]                       # (P, C, 3)
        u = _f[None] * jnp.sum(s * _h[None], axis=-1)
        q = jnp.cross(s, _e1[None])
        v = _f[None] * jnp.sum(q * d, axis=-1)
        t = _f[None] * jnp.sum(q * _e2[None], axis=-1)
        hit = _ok[None] & (u >= 0.0) & (v >= 0.0) & (u + v <= 1.0) & (t > EPS)
        return count + jnp.sum(hit, axis=1), None

    count, _ = jax.lax.scan(
        step, jnp.zeros(points.shape[0], jnp.int32), (v0, e1, e2, h, f, valid))
    return count


def occupancy_one(verts, points, faces, d=RAY_DIR, chunk: int = 2048):
    """(V,3), (P,3) -> (P,) float32 in {0,1}."""
    v0, e1, e2, h, f, valid = _precompute(verts, faces, d)
    n = _count_crossings(points, v0, e1, e2, h, f, valid, d, chunk=chunk)
    return (n % 2).astype(jnp.float32)


def occupancy(verts, points, faces, chunk: int = 2048):
    """(B,V,3), (B,P,3) -> (B,P). Fully jittable; safe to call inside a training step."""
    return jax.vmap(lambda v, p: occupancy_one(v, p, faces, chunk=chunk))(verts, points)


def occupancy_trimesh(verts, points, faces) -> np.ndarray:
    """CPU reference via trimesh's ray-stabbing `contains`.

    Deliberately a different implementation, so agreement between the two is evidence
    rather than tautology. Slow -- for tests, not the main path.
    """
    import trimesh

    v, p, f = np.asarray(verts), np.asarray(points), np.asarray(faces)
    out = np.zeros(p.shape[:2], dtype=np.float32)
    for b in range(v.shape[0]):
        out[b] = trimesh.Trimesh(v[b], f, process=False).contains(p[b])
    return out


def disagreement(a, b) -> dict:
    """Compare two occupancy fields, reporting where they differ."""
    a, b = np.asarray(a).astype(bool), np.asarray(b).astype(bool)
    return {
        'n_points': int(a.size),
        'n_disagree': int((a != b).sum()),
        'rate': float((a != b).mean()),
        'a_inside_frac': float(a.mean()),
        'b_inside_frac': float(b.mean()),
    }
