"""Query-point sampling: COAP's protocol, per part.

Half the points per part are uniform inside that part's padded local box; half are drawn
on the part's tight surface and jittered by Gaussian noise. The uniform points occupy
`[..., :n_uniform, :]` and the surface points the remainder -- keeping that axis intact is
what lets evaluation split `iou_unif` from `iou_surf` correctly.

One easy-to-miss detail: the sampling box is `|bbox_max - bbox_min| * padding - 1e-3`,
which is **not** the same as the `bbox_size` carried in the encoded body. That extra 1e-3
shrink keeps samples strictly inside the box; using the encoded size instead would put a
thin shell of samples exactly on the boundary, where `inside_bbox` is decided by a strict
inequality.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from VolumetricSMPL import geometry as G
from VolumetricSMPL import model as MD

POINTS_SIGMA = 0.01
UNIFORM_RATIO = 0.5


def sample_query_points(key, verts, bone_trans, bbox_min, bbox_max, a,
                        n_points: int, sigma: float = POINTS_SIGMA,
                        uniform_ratio: float = UNIFORM_RATIO):
    """(B, K, n_points, 3) in posed space."""
    B, K = bone_trans.shape[:2]
    n_uniform = int(n_points * uniform_ratio)
    n_surface = n_points - n_uniform

    k_u, k_s, k_n = jax.random.split(key, 3)

    size = jnp.abs(bbox_max - bbox_min) * a.bbox_padding - 1e-3
    center = (bbox_min + bbox_max) * 0.5
    bb_min = center - size * 0.5

    unit = jax.random.uniform(k_u, (B, K, n_uniform, 3), dtype=verts.dtype)
    uniform = bb_min + unit * size
    # bone_trans maps posed -> local, so invert it to place the samples in posed space.
    uniform = MD._apply(jnp.linalg.inv(bone_trans), uniform)

    surface = G.sample_parts(k_s, verts, a.tight_faces, n_surface)
    surface = surface + jax.random.normal(k_n, surface.shape, dtype=verts.dtype) * sigma

    return jnp.concatenate([uniform, surface], axis=-2)
