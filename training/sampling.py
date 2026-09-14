"""Query-point sampling, replicating COAP's protocol on GPU.

Half the points per part are uniform inside that part's padded local bounding box;
half are sampled on the part's tight surface and perturbed by Gaussian noise.
Reproduced from COAP `training_code/data.py::SMPLDataset.sample_points`, which is
the only written-down source for the protocol — the paper omits it.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

POINTS_SIGMA = 0.01        # metres; COAP's surface jitter
UNIFORM_RATIO = 0.5


def sample_query_points(volume, smpl_output, n_points: int,
                        sigma: float = POINTS_SIGMA,
                        uniform_ratio: float = UNIFORM_RATIO,
                        surface: str = 'tight',
                        generator: torch.Generator | None = None) -> torch.Tensor:
    """Sample `n_points` query points per body part.

    Returns (B, K, n_points, 3) in *posed* space. The uniform points occupy
    ``[..., :n_uniform, :]`` and the surface points the remainder — keeping that axis
    intact is what lets the evaluation split uniform from surface correctly.
    """
    verts = smpl_output.vertices
    B, device = verts.shape[0], verts.device

    bone_trans = volume.compute_bone_trans(smpl_output.full_pose, smpl_output.joints)
    bbox_min, bbox_max = volume.get_bbox_bounds(verts, bone_trans)  # (B, K, 1, 3)
    K = bbox_max.shape[1]

    n_uniform = int(n_points * uniform_ratio)
    n_surface = n_points - n_uniform

    # --- uniform inside each part's padded local box ---------------------------
    # The -1e-3 is COAP's: it shrinks the box a hair so samples stay strictly inside.
    bbox_size = (bbox_max - bbox_min).abs() * volume.bbox_padding - 1e-3
    bbox_center = (bbox_min + bbox_max) * 0.5
    bb_min = bbox_center - bbox_size * 0.5

    unit = torch.rand((B, K, n_uniform, 3), device=device, generator=generator)
    uniform_points = bb_min + unit * bbox_size

    # Local (canonical) -> posed. bone_trans maps posed->local, so invert it.
    abs_transforms = torch.inverse(bone_trans)  # (B, K, 4, 4)
    uniform_points = (
        abs_transforms.reshape(B, K, 1, 4, 4).expand(-1, -1, n_uniform, -1, -1)
        @ F.pad(uniform_points, [0, 1], "constant", 1.0).unsqueeze(-1)
    )[..., :3, 0]

    # --- near-surface ----------------------------------------------------------
    # Reuses the package's own sampler: it builds one Meshes per (body, part) from the
    # -1-padded tight face tensor, which is exactly COAP's construction.
    if surface == 'tight':
        surface_points = volume.partitioner._sample_mesh_points(
            verts, volume.get_tight_face_tensor(), n_surface)  # (B, K, n_surface, 3)
    elif surface == 'extended':
        surface_points = volume.partitioner._sample_mesh_points(
            verts, volume.partitioner.extended_face_tensor, n_surface)
    elif surface == 'mixed':
        # Half tight, half extended — the split `sample_mesh_points` uses for the
        # PointNet encoder's input cloud.
        surface_points = volume.partitioner.sample_mesh_points(verts, n_surface)
    elif surface == 'full':
        # Area-weighted over the whole body, ignoring the part decomposition.
        # Per-part sampling gives each of the 15 parts an equal share, which
        # massively oversamples small, geometrically hard parts (hands, feet)
        # relative to their surface area.
        from pytorch3d.ops import sample_points_from_meshes
        from pytorch3d.structures import Meshes

        body_faces = torch.as_tensor(volume.partitioner.faces.astype('int64'),
                                     device=device)
        meshes = Meshes(verts=verts, faces=body_faces[None].expand(B, -1, -1))
        surface_points = sample_points_from_meshes(meshes, K * n_surface)
        surface_points = surface_points.reshape(B, K, n_surface, 3)
    else:
        raise ValueError(f'unknown surface mode: {surface}')
    noise = torch.randn(surface_points.shape, device=device, generator=generator) * sigma
    surface_points = surface_points + noise

    return torch.cat((uniform_points, surface_points), dim=-2).float()
