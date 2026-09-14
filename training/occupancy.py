"""Ground-truth occupancy, and an independent reference to check it against.

Both backends decide inside/outside by **ray-stabbing parity**, which is the
definition COAP used via `leap.tools.libmesh.check_mesh_contains`. It is not the same
as a generalized-winding-number test on a self-intersecting mesh: where an arm passes
through the torso, parity reports the doubly-covered region as outside and a winding
number reports it as inside. Posed SMPL-X bodies do self-intersect, so the choice
changes the training targets.
"""
from __future__ import annotations

import numpy as np
import torch


def occupancy_kaolin(verts: torch.Tensor, faces: torch.Tensor,
                     points: torch.Tensor) -> torch.Tensor:
    """GPU parity occupancy. verts (B,V,3), faces (F,3) int64, points (B,N,3) -> (B,N) float."""
    import kaolin

    inside = kaolin.ops.mesh.check_sign(verts, faces, points)
    return inside.float()


def occupancy_trimesh(verts: torch.Tensor, faces: torch.Tensor,
                      points: torch.Tensor) -> torch.Tensor:
    """CPU reference via trimesh's ray-stabbing `contains`.

    Deliberately a different implementation from kaolin's, so agreement between them is
    evidence rather than tautology. Slow — for differential testing, not the main path.
    """
    import trimesh

    v = verts.detach().cpu().numpy()
    f = faces.detach().cpu().numpy()
    p = points.detach().cpu().numpy()

    out = np.zeros(p.shape[:2], dtype=np.float32)
    for b in range(v.shape[0]):
        mesh = trimesh.Trimesh(v[b], f, process=False)
        out[b] = mesh.contains(p[b]).astype(np.float32)
    return torch.from_numpy(out).to(points.device)


def disagreement(a: torch.Tensor, b: torch.Tensor) -> dict:
    """Compare two occupancy fields, reporting where they differ."""
    diff = (a.bool() != b.bool())
    n = diff.numel()
    return {
        'n_points': n,
        'n_disagree': int(diff.sum()),
        'rate': float(diff.float().mean()),
        'a_inside_frac': float(a.float().mean()),
        'b_inside_frac': float(b.float().mean()),
    }
