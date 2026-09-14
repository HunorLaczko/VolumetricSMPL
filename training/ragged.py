"""Ragged in-bbox filtering for the training forward pass.

`query_training` runs the decoder on every (body, part, point) triple — all 15 parts for
all 7,680 points — but only ~9% of those triples are inside the part's bounding box. The
rest contribute nothing:

    part_occupancy = torch.sigmoid(-part_occ).squeeze(-1)
    part_occupancy = part_occupancy * inside_bbox          # zeroes value AND gradient
    occupancy      = part_occupancy.max(dim=1).values

Multiplying by the mask zeroes both the value and its gradient, and `sigmoid(...) > 0`
strictly, so `max` over parts never selects a masked entry when any unmasked one exists.
Points inside no box at all are dropped by `query_training` anyway. Skipping the masked
triples is therefore *equivalent*, not an approximation.

The decoder's weights are synthesised per (body, part) by the NBW layer, so points
cannot simply be flattened into one batch — each needs its own part's weights. Instead
each (body, part) row is compacted to its in-box points and padded to the batch maximum
(typically ~1,500 of 7,680).

The package's `query_fast` filter does not help here: it masks the *query* dimension
across all parts at once, and in training nearly every point is inside some part's box.

Measured: losses bit-identical to `query_training`, gradients within the unfiltered
path's own run-to-run noise, 1.66x faster per step. `verify_equivalence` re-checks it.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def compact_indices(inside_bbox: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per (body, part), gather in-box point indices to the front.

    Returns `order` (B, K, P) with P = max in-box count, and `valid` (B, K, P) marking
    which of those slots are real. A stable descending argsort puts every true index
    first, in ascending order, so the result is deterministic run to run.
    """
    ib = inside_bbox.bool()
    pad = int(ib.sum(-1).max().clamp(min=1))
    order = torch.argsort(ib.float(), dim=-1, descending=True, stable=True)[..., :pad]
    return order, ib.gather(-1, order)


def query_training(volume, points: torch.Tensor, gt_occ: torch.Tensor, smpl_output,
                   n_unif_samples: int = 512) -> dict:
    """Drop-in replacement for `volume.query_training`, evaluating only in-box triples."""
    volume._attach_impl_code(smpl_output)
    local_queries, inside_bbox = volume.to_local(
        points, volume.impl_code['bone_trans'],
        volume.impl_code['bbox_center'], volume.impl_code['bbox_size'])

    B, K, T = inside_bbox.shape
    order, valid = compact_indices(inside_bbox)

    gathered = local_queries.gather(2, order.unsqueeze(-1).expand(-1, -1, -1, 3))
    part_occ_small = volume._fwd_pass(gathered, valid.int(), only_part_occupancy=True)

    part_occ = torch.zeros(B, K, T, device=part_occ_small.device,
                           dtype=part_occ_small.dtype)
    part_occ = part_occ.scatter(2, order, part_occ_small)
    pred_occ = part_occ.max(dim=1).values                       # (B, T)

    loss = {}
    keep = inside_bbox.bool().any(1)
    loss['mse_occ'] = F.mse_loss(pred_occ[keep], gt_occ[keep])

    # The UDF path is already compact — n_unif_samples points per part, all in-box by
    # construction — so it is left exactly as the package computes it.
    unif_local_pts, udf_target = volume._sample_part_udf(
        smpl_output, n_unif_samples=n_unif_samples)
    pred_udf = volume._fwd_pass(
        unif_local_pts, torch.ones_like(unif_local_pts[..., 0]).bool())[-1]
    loss['mse_udf'] = F.mse_loss(udf_target, pred_udf)

    loss = {k: v * volume.loss_weights.get(k, 1.0) for k, v in loss.items()}
    loss['total_loss'] = sum(loss.values())
    return loss


@torch.no_grad()
def verify_equivalence(volume, points, gt_occ, smpl_output, n_unif_samples=512):
    """Compare filtered vs unfiltered losses on the same batch.

    Exact bit-equality is not expected and not required: the decoder sees a differently
    shaped tensor, so cuBLAS picks different tiling and float32 reductions reassociate.
    Agreement to float32 epsilon is the real criterion.
    """
    torch.manual_seed(0)
    volume.detach_cache()
    ref = volume.query_training(points, gt_occ, smpl_output,
                                n_unif_samples=n_unif_samples)
    torch.manual_seed(0)
    volume.detach_cache()
    got = query_training(volume, points, gt_occ, smpl_output,
                         n_unif_samples=n_unif_samples)
    volume.detach_cache()
    # mse_udf resamples internally, so only mse_occ is deterministic across the two.
    return {k: (float(ref[k]), float(got[k]), abs(float(ref[k]) - float(got[k])))
            for k in ref}
