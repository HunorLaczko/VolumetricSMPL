"""Generalized winding numbers for inside/outside tests against triangle soups.

The original package's utility (from BUDDI, https://github.com/muelea/buddi): the winding
number of a point is the sum of the solid angles its triangles subtend, over 4*pi. It is 1
inside a closed mesh and 0 outside, and degrades gracefully on open or self-intersecting
ones.

    Robust Inside-Outside Segmentation using Generalized Winding Numbers.
    Jacobson, Kavan, Sorkine-Hornung. SIGGRAPH 2013.

The original materialises a (B, Q, F, 3, 3) tensor. Here both the query and the triangle
axes are blocked, so memory is bounded by `query_chunk x tri_chunk`.
"""
from __future__ import annotations

import functools
import math

import jax
import jax.numpy as jnp


def solid_angles(points, triangles):
    """Solid angle each triangle subtends at each point (Van Oosterom & Strackee, 1983).

    Args:
        points:    (B, Q, 3)
        triangles: (B, F, 3, 3)
    Returns:
        (B, Q, F)
    """
    centered = triangles[:, None] - points[:, :, None, None]        # (B, Q, F, 3, 3)
    norms = jnp.linalg.norm(centered, axis=-1)                       # (B, Q, F, 3)
    a, b, c = centered[..., 0, :], centered[..., 1, :], centered[..., 2, :]
    numerator = jnp.sum(a * jnp.cross(b, c), axis=-1)
    denominator = (norms.prod(axis=-1)
                   + jnp.sum(a * b, axis=-1) * norms[..., 2]
                   + jnp.sum(a * c, axis=-1) * norms[..., 1]
                   + jnp.sum(b * c, axis=-1) * norms[..., 0])
    return 2.0 * jnp.arctan2(numerator, denominator)


@functools.partial(jax.jit, static_argnames=('query_chunk', 'tri_chunk'))
def winding_numbers(points, triangles, query_chunk: int = 256, tri_chunk: int = 2048):
    """Generalized winding number of each point.

    Args:
        points:    (B, Q, 3)
        triangles: (B, F, 3, 3)
    Returns:
        (B, Q)
    """
    B, Q = points.shape[:2]
    F = triangles.shape[1]

    # Padding triangles are collapsed to a single point, which subtends no solid angle.
    pad_t = (-F) % tri_chunk
    tris = jnp.concatenate([triangles, jnp.zeros((B, pad_t, 3, 3), triangles.dtype)], 1)
    tris = jnp.moveaxis(tris.reshape(B, -1, tri_chunk, 3, 3), 1, 0)

    pad_q = (-Q) % query_chunk
    pts = jnp.concatenate([points, jnp.zeros((B, pad_q, 3), points.dtype)], 1)
    pts = jnp.moveaxis(pts.reshape(B, -1, query_chunk, 3), 1, 0)

    def per_query_block(p):
        def step(total, t):
            return total + solid_angles(p, t).sum(-1), None
        total, _ = jax.lax.scan(step, jnp.zeros(p.shape[:2], p.dtype), tris)
        return total

    total = jax.lax.map(per_query_block, pts)                        # (n, B, query_chunk)
    total = jnp.moveaxis(total, 0, 1).reshape(B, -1)[:, :Q]
    return total / (4.0 * math.pi)
