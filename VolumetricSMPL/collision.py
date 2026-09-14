"""Collision terms: a body against a point cloud, and a body against itself.

Mirrors the original package's `collision_loss*` and `self_collision_loss`.

The self-intersection term needs care in JAX. The original selects the colliding box pairs,
samples candidates only in those, keeps the candidates inside both boxes, and keeps the
points where two parts claim occupancy -- every one of those intermediates has a
data-dependent size. Here each step has a fixed shape and a validity mask instead, and
masked sums replace the selections: the loss is the same function of the same samples.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from . import model as MD

# The original's margin for "a box corner lies inside another box".
EPS = 1e-4

_CORNER_SELECT = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
                           [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]], dtype=bool)


# --------------------------------------------------------------------------------------
# Body vs points
# --------------------------------------------------------------------------------------

def penetration(sdf):
    """How far each point is inside the body; 0 outside."""
    return jnp.maximum(-sdf, 0.0)


def collision_sum(sdf):
    return penetration(sdf).sum(-1)


def collision_mean(sdf):
    """Mean penetration over the points that do penetrate."""
    loss = penetration(sdf)
    return loss.sum(-1) / ((loss > 0).sum(-1) + 1e-6)


def collision_gmof(sdf, rho: float = 5e-2):
    """Geman-McClure-robustified penetration, averaged over penetrating points."""
    loss = penetration(sdf) ** 2
    loss = loss / (loss + rho ** 2)
    return loss.sum(-1) / ((loss > 0).sum(-1) + 1e-6)


# --------------------------------------------------------------------------------------
# Body vs itself
# --------------------------------------------------------------------------------------

def _homo(p):
    return jnp.concatenate([p, jnp.ones(p.shape[:-1] + (1,), p.dtype)], axis=-1)


def box_corners(bb_min, bb_max):
    """(K, 3), (K, 3) -> (K, 8, 3), in the original's corner order."""
    return jnp.where(_CORNER_SELECT[None], bb_max[:, None], bb_min[:, None])


def bbox_penetrations(bb_min, bb_max, abs_trans, disable_mat):
    """(K, K) lower-triangular mask of part pairs whose boxes overlap.

    A pair overlaps when a corner of one part's box, posed, lies strictly inside the other
    part's box. Pairs switched off in `disable_mat` never count.

    Args:
        bb_min, bb_max: (K, 3) part-local box bounds
        abs_trans:      (K, 4, 4) part-local -> posed
        disable_mat:    (K, K) bool
    """
    posed = jnp.einsum('kij,kcj->kci', abs_trans, _homo(box_corners(bb_min, bb_max)))
    back = jnp.linalg.inv(abs_trans)
    proj = jnp.einsum('kij,lcj->klci', back, posed)[..., :3]    # corner c of box l, frame k
    lo, hi = bb_min[:, None, None] + EPS, bb_max[:, None, None] - EPS
    inside = jnp.all((proj > lo) & (proj < hi), axis=-1).any(axis=-1)
    inside = inside & disable_mat
    return jnp.tril(inside | inside.T)


def colliding_pairs(pair_mask, max_pairs: int):
    """Up to `max_pairs` (low, high) part pairs from the mask, plus which are real.

    Row-major order, as the original's `torch.where`. Pairs beyond `max_pairs` are
    dropped; with K(K-1)/2 pairs none can be.
    """
    k, l = jnp.nonzero(pair_mask, size=max_pairs, fill_value=0)
    pairs = jnp.stack([jnp.minimum(k, l), jnp.maximum(k, l)], axis=-1)
    valid = jnp.arange(max_pairs) < pair_mask.sum()
    return pairs, valid


def sample_candidates(key, bb_min, bb_max, abs_trans, pairs, pair_valid,
                      n_points_uniform: int):
    """Uniform samples in both boxes of every pair, kept where they fall in the other box.

    Returns (P*2*n, 3) posed points and a (P*2*n,) validity mask. The samples of each pair's
    first box come first, then those of its second box, as in the original.
    """
    P = pairs.shape[0]
    idx = pairs.reshape(-1)
    lo, hi = bb_min[idx], bb_max[idx]                                   # (2P, 3)
    unit = jax.random.uniform(key, (2 * P, n_points_uniform, 3), dtype=bb_min.dtype)
    local = unit * (hi - lo)[:, None] + lo[:, None]
    posed = jnp.einsum('pij,pnj->pni', abs_trans[idx], _homo(local))[..., :3]
    posed = posed.reshape(P, 2, n_points_uniform, 3)

    back = jnp.linalg.inv(abs_trans)[idx].reshape(P, 2, 4, 4)
    lo, hi = lo.reshape(P, 2, 3), hi.reshape(P, 2, 3)

    def inside(pts, bwd, bmin, bmax):
        can = jnp.einsum('pij,pnj->pni', bwd, _homo(pts))[..., :3]
        return jnp.all((can > bmin[:, None]) & (can < bmax[:, None]), axis=-1)

    in_second = inside(posed[:, 0], back[:, 1], lo[:, 1], hi[:, 1])
    in_first = inside(posed[:, 1], back[:, 0], lo[:, 0], hi[:, 0])

    points = jnp.concatenate([posed[:, 0], posed[:, 1]], axis=0).reshape(-1, 3)
    valid = jnp.concatenate([in_second & pair_valid[:, None],
                             in_first & pair_valid[:, None]], axis=0).reshape(-1)
    return points, valid


def conflict_loss(part_occ, valid, disable_mat, level_set: float = 0.5):
    """Penalty on points that two non-adjacent parts both claim.

    Args:
        part_occ:    (K, N) per-part occupancy
        valid:       (N,) which points are real candidates
        disable_mat: (K, K) bool
    """
    po = part_occ.T                                                     # (N, K)
    claimed = jax.lax.stop_gradient(po > level_set).astype(po.dtype)
    K = po.shape[1]
    penalised = (~jnp.eye(K, dtype=bool) & disable_mat).astype(po.dtype)
    n_pairs = jnp.einsum('nk,nl,kl->n', claimed, claimed, penalised)
    conflicting = (n_pairs >= 2.0) & valid
    per_point = jnp.maximum(po, 0.0).sum(-1) - level_set
    return jnp.sum(jnp.where(conflicting, per_point, 0.0))


def self_collision_loss(params, verts, impl, a, key, n_points_uniform: int = 300,
                        max_pairs: int | None = None, chunk: int = 8192,
                        level_set: float = 0.5):
    """(B,) self-intersection loss.

    Candidates are the box-overlap samples plus every body vertex. The vertices only count
    when at least one candidate exists, which matches the original returning 0 early.
    Bodies are processed one at a time and the decoder runs in rematerialised chunks, so
    memory stays bounded however many pairs overlap.
    """
    K = a.num_parts
    max_pairs = K * (K - 1) // 2 if max_pairs is None else max_pairs
    disable = a.selfpen_disable_mat
    sg = jax.lax.stop_gradient

    def conflict_chunks(points, valid, code):
        pad = (-points.shape[0]) % chunk
        points = jnp.concatenate([points, jnp.zeros((pad, 3), points.dtype)])
        valid = jnp.concatenate([valid, jnp.zeros((pad,), bool)])
        n = points.shape[0] // chunk

        @jax.checkpoint
        def block(xs):
            p, ok = xs
            local, inside = MD.to_local(p[None], code['bone_trans'],
                                        code['bbox_center'], code['bbox_size'])
            occ = MD.fwd_pass_part_occ(params, local, inside, code['latent_code'])[0]
            return conflict_loss(occ, ok, disable, level_set)

        return jax.lax.map(block, (points.reshape(n, chunk, 3),
                                   valid.reshape(n, chunk))).sum()

    def one_body(xs):
        v, code, k = xs
        bmin, bmax = sg(code['bbox_min'][:, 0]), sg(code['bbox_max'][:, 0])
        abs_trans = sg(jnp.linalg.inv(code['bone_trans']))
        pair_mask = bbox_penetrations(bmin, bmax, abs_trans, disable)
        pairs, pair_valid = colliding_pairs(pair_mask, max_pairs)
        cand, cand_valid = sample_candidates(k, bmin, bmax, abs_trans, pairs, pair_valid,
                                             n_points_uniform)
        points = jnp.concatenate([cand, sg(v)])
        valid = jnp.concatenate([cand_valid,
                                 jnp.broadcast_to(cand_valid.any(), (v.shape[0],))])
        code1 = {name: x[None] for name, x in code.items()}
        return conflict_chunks(points, valid, code1)

    keys = jax.random.split(key, verts.shape[0])
    return jax.lax.map(one_body, (verts, impl, keys))
