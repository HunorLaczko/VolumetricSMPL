"""The VolumetricSMPL field: body encoding, part-local queries, and the fused outputs.

A transcription of the `VolumetricSMPL` module from the original package, training losses
included. The structure is kept recognisable against that file on purpose -- this is the
part a reader will most want to diff against the original.

The one systematic departure is boolean indexing. The original writes `pred[~all_out]`,
which produces a data-dependent shape; JAX cannot, so the same quantity is computed as a
masked mean. That is exactly equivalent for a mean, not an approximation.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from . import geometry as G
from . import lbs as L
from . import modules as M


def compute_abs_transformations(full_pose, posed_joints, a):
    """Absolute bone transforms for the K parts. (B, K, 4, 4).

    Note this composes **rotations only** down the chain and then pairs the result with
    the posed joint position, rather than composing full 4x4 transforms. That is what the
    package does, and it is not the same matrix as LBS's `A`.

    The chain is walked to mK (22) and then the 15 surviving parts are selected with
    `joint_mapper`, which drops the 7 merged ones.
    """
    B = full_pose.shape[0]
    rot_mats = L.batch_rodrigues(full_pose.reshape(-1, 3)).reshape(B, -1, 3, 3)
    mK = a.mK

    chain = [rot_mats[:, 0]]
    for i in range(1, mK):
        chain.append(chain[a.parents[i]] @ rot_mats[:, i])
    R = jnp.stack(chain, axis=1)                                  # (B, mK, 3, 3)

    t = posed_joints[:, :mK][..., None]                           # (B, mK, 3, 1)
    top = jnp.concatenate([R, t], axis=-1)                        # (B, mK, 3, 4)
    bottom = jnp.broadcast_to(jnp.array([0.0, 0.0, 0.0, 1.0], top.dtype),
                              top.shape[:2] + (1, 4))
    abs_trans = jnp.concatenate([top, bottom], axis=-2)           # (B, mK, 4, 4)
    return abs_trans[:, np.asarray(a.joint_mapper)]                # (B, K, 4, 4)


def bbox_bounds(verts, bone_trans, a):
    """Axis-aligned extent of each part's tight vertices, in part-local space."""
    B, K = bone_trans.shape[:2]
    sel = a.tight_vert_selector                                    # (K, Vt)
    part_v = verts[:, sel.reshape(-1)].reshape(B, K, sel.shape[1], 3)
    local = _apply(bone_trans, part_v)
    return local.min(axis=-2, keepdims=True), local.max(axis=-2, keepdims=True)


def _apply(mat, pts):
    """(B, K, 4, 4) applied to (B, K, T, 3)."""
    homo = jnp.concatenate([pts, jnp.ones(pts.shape[:-1] + (1,), pts.dtype)], axis=-1)
    return jnp.einsum('bkij,bktj->bkti', mat, homo)[..., :3]


def encode_body(params, verts, joints, full_pose, a, key, n_samples: int = 1000):
    """Everything about a body that does not depend on the query point.

    The encoder's input cloud is half tight faces, half extended -- `sample_mesh_points`
    splits `n_samples` evenly between the two face sets.
    """
    abs_trans = compute_abs_transformations(full_pose, joints, a)
    bone_trans = jnp.linalg.inv(abs_trans)
    bbox_min, bbox_max = bbox_bounds(verts, bone_trans, a)

    n_tight = n_samples // 2
    k1, k2 = jax.random.split(key)
    tight = G.sample_parts(k1, verts, a.tight_faces, n_tight)
    ext = G.sample_parts(k2, verts, a.extended_faces, n_samples - n_tight)
    cloud = jnp.concatenate([tight, ext], axis=-2)                 # (B, K, n, 3)

    local_cloud = _apply(bone_trans, cloud)
    B, K = local_cloud.shape[:2]
    latent = M.encode(params, local_cloud.reshape(B * K, -1, 3)).reshape(B, K, -1)

    return dict(
        bone_trans=bone_trans,
        latent_code=latent,
        bbox_min=bbox_min, bbox_max=bbox_max,
        bbox_size=jnp.abs(bbox_max - bbox_min) * a.bbox_padding,
        bbox_center=(bbox_min + bbox_max) * 0.5,
    )


def to_local(points, bone_trans, bbox_center, bbox_size):
    """(B, T, 3) posed -> (B, K, T, 3) part-local, plus the in-box mask."""
    B, K = bone_trans.shape[:2]
    T = points.shape[1]
    homo = jnp.concatenate([points, jnp.ones((B, T, 1), points.dtype)], axis=-1)
    local = jnp.einsum('bkij,btj->bkti', bone_trans, homo)[..., :3]
    inside = jnp.all(jnp.abs(local - bbox_center) < (bbox_size * 0.5), axis=-1)
    return local, inside


def analytic_part_sdf(local, bbox_center, bbox_offset):
    """Exact SDF to the part's box, used wherever the network is not evaluated.

    The outside term is a norm that is exactly zero for every point inside the box, where
    `jnp.linalg.norm` has a NaN gradient. It is computed so the gradient there is 0, as in
    the original.
    """
    q = jnp.abs(local - bbox_center) - bbox_offset
    sq = jnp.sum(jnp.clip(q, 0.0, None) ** 2, axis=-1)
    outside = jnp.where(sq > 0, jnp.sqrt(jnp.where(sq > 0, sq, 1.0)), 0.0)
    return outside + jnp.clip(jnp.max(q, axis=-1), None, 0.0)


def part_indices(B: int, K: int):
    """Row -> part index for the flattened (body, part) batch the decoder sees."""
    return jnp.tile(jnp.arange(K), B)


def fwd_pass(params, local, inside, latent, cond_decoder: bool = True):
    """Run the decoder on part-local queries.

    Returns (occupancy, part_sdf, part_udf) with the package's fusion:
      - per part, `sigmoid(-logit)` masked to the box, then **max** over parts;
      - the udf head is passed through `abs`, then signed by the occupancy side.
    """
    B, K, T = inside.shape
    z = local
    if cond_decoder:
        z = jnp.concatenate(
            [z, jnp.broadcast_to(latent[:, :, None, :],
                                 latent.shape[:2] + (T, latent.shape[-1]))], axis=-1)

    pred = M.decode(params, z.reshape(B * K, T, -1),
                    cond=latent.reshape(B * K, -1),
                    ind=part_indices(B, K)).reshape(B, K, T, 2)
    part_occ, part_udf = pred[..., 0], pred[..., 1]

    part_occupancy = jax.nn.sigmoid(-part_occ) * inside
    occupancy = jnp.max(part_occupancy, axis=1)

    part_udf = jnp.abs(part_udf)
    part_sdf = jnp.where(part_occupancy > 0.5, -part_udf, part_udf)
    return occupancy, part_sdf, part_udf


def query_occupancy(params, points, impl, a):
    local, inside = to_local(points, impl['bone_trans'],
                             impl['bbox_center'], impl['bbox_size'])
    return fwd_pass(params, local, inside, impl['latent_code'])[0]


def fused_part_sdf(params, points, impl, a):
    """(B, K, T) per-part signed distance: the network inside each part box, the exact
    box SDF outside it."""
    local, inside = to_local(points, impl['bone_trans'],
                             impl['bbox_center'], impl['bbox_size'])
    box_sdf = analytic_part_sdf(local, impl['bbox_center'],
                                impl['bbox_size'] * (0.5 / a.bbox_padding))
    part_sdf = fwd_pass(params, local, inside, impl['latent_code'])[1]
    return jnp.where(inside, part_sdf, box_sdf)


def query_sdf(params, points, impl, a):
    """Fused signed distance: the min over parts of `fused_part_sdf`."""
    return jnp.min(fused_part_sdf(params, points, impl, a), axis=1)


def sample_part_udf(key, verts, impl, a, n_unif: int = 512):
    """Uniform samples in each part box, with their true distance to the extended set.

    Returns (local queries, target UDF). The 1e-4 shrink is the package's: it keeps
    samples strictly inside the box after padding.
    """
    center, size = impl['bbox_center'], impl['bbox_size']
    b_size = size - 1e-4
    b_min = center - b_size * 0.5
    B, K = size.shape[:2]

    u = jax.random.uniform(key, (B, K, n_unif, 3), dtype=verts.dtype)
    local = b_min + u * b_size

    posed = _apply(jnp.linalg.inv(impl['bone_trans']), local)
    target = G.udf_per_part(posed, verts, a.extended_faces)
    return local, target


def compact_indices(inside, pad: int):
    """Per (body, part), gather in-box point indices to the front.

    Sizing this axis from `inside.sum(-1).max()` would be data-dependent and illegal
    under `jit`. Here `pad` is a **static budget** and the
    surplus is reported so a silent truncation cannot happen unnoticed: if any
    (body, part) row has more in-box points than `pad`, the extra ones would be dropped
    from the loss entirely, which is a correctness bug rather than a slowdown.

    Returns (order, valid, overflow) where `overflow` is how far the worst row exceeds
    the budget -- zero when the budget holds.
    """
    order = jnp.argsort(~inside, axis=-1, stable=True)[..., :pad]
    valid = jnp.take_along_axis(inside, order, axis=-1)
    overflow = jnp.maximum(jnp.max(jnp.sum(inside, axis=-1)) - pad, 0)
    return order, valid, overflow


def ragged_losses(params, points, gt_occ, verts, impl, a, key, pad: int,
                  n_unif: int = 512, loss_weights: dict | None = None):
    """`losses`, evaluating only the (body, part, point) triples inside a box.

    Equivalent, not an approximation: the box mask multiplies the per-part occupancy, so
    a masked triple contributes neither value nor gradient, and `sigmoid` is strictly
    positive so the max over parts never selects a masked entry when an unmasked one
    exists. About 8.8% of triples are in-box, a ~5x reduction on this path.
    """
    local, inside = to_local(points, impl['bone_trans'],
                             impl['bbox_center'], impl['bbox_size'])
    B, K, T = inside.shape
    order, valid, overflow = compact_indices(inside, pad)

    gathered = jnp.take_along_axis(local, order[..., None], axis=2)
    part_occ_small = fwd_pass_part_occ(params, gathered, valid, impl['latent_code'])

    part_occ = jnp.zeros((B, K, T), part_occ_small.dtype).at[
        jnp.arange(B)[:, None, None], jnp.arange(K)[None, :, None], order
    ].set(part_occ_small)
    pred_occ = jnp.max(part_occ, axis=1)

    keep = jnp.any(inside, axis=1)
    n_keep = jnp.maximum(jnp.sum(keep), 1)
    mse_occ = jnp.sum(jnp.where(keep, (pred_occ - gt_occ) ** 2, 0.0)) / n_keep

    unif_local, udf_target = sample_part_udf(key, verts, impl, a, n_unif)
    pred_udf = fwd_pass(params, unif_local,
                        jnp.ones(unif_local.shape[:-1], bool),
                        impl['latent_code'])[2]
    mse_udf = jnp.mean((udf_target - pred_udf) ** 2)

    w = loss_weights or {}
    out = {'mse_occ': mse_occ * w.get('mse_occ', 1.0),
           'mse_udf': mse_udf * w.get('mse_udf', 1.0)}
    out['total_loss'] = out['mse_occ'] + out['mse_udf']
    out['ragged_overflow'] = overflow
    return out


def fwd_pass_part_occ(params, local, inside, latent, cond_decoder: bool = True):
    """Per-part occupancy only -- the ragged path never needs the udf head here."""
    B, K, T = inside.shape
    z = local
    if cond_decoder:
        z = jnp.concatenate(
            [z, jnp.broadcast_to(latent[:, :, None, :],
                                 latent.shape[:2] + (T, latent.shape[-1]))], axis=-1)
    pred = M.decode(params, z.reshape(B * K, T, -1),
                    cond=latent.reshape(B * K, -1),
                    ind=part_indices(B, K)).reshape(B, K, T, 2)
    return jax.nn.sigmoid(-pred[..., 0]) * inside


def losses(params, points, gt_occ, verts, impl, a, key, n_unif: int = 512,
           loss_weights: dict | None = None):
    """The two training terms.

    `mse_occ` is a **masked** mean over points that fall inside at least one part box.
    The original writes `pred[~all_out]`, which is a data-dependent shape and illegal under
    jit; a masked mean is the same number.
    """
    local, inside = to_local(points, impl['bone_trans'],
                             impl['bbox_center'], impl['bbox_size'])
    pred_occ = fwd_pass(params, local, inside, impl['latent_code'])[0]

    keep = jnp.any(inside, axis=1)
    n_keep = jnp.maximum(jnp.sum(keep), 1)
    mse_occ = jnp.sum(jnp.where(keep, (pred_occ - gt_occ) ** 2, 0.0)) / n_keep

    unif_local, udf_target = sample_part_udf(key, verts, impl, a, n_unif)
    pred_udf = fwd_pass(params, unif_local,
                        jnp.ones(unif_local.shape[:-1], bool),
                        impl['latent_code'])[2]
    mse_udf = jnp.mean((udf_target - pred_udf) ** 2)

    w = loss_weights or {}
    out = {'mse_occ': mse_occ * w.get('mse_occ', 1.0),
           'mse_udf': mse_udf * w.get('mse_udf', 1.0)}
    out['total_loss'] = out['mse_occ'] + out['mse_udf']
    return out
