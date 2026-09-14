"""Surface-level comparison of an extracted mesh against the ground-truth body.

IoU and SDF MSE are both *point-sampled*: they average over query points drawn from a
fixed distribution. That makes them blind to failures which occupy little probability
mass — a detached fragment floating beside the hand, a hole in the armpit, a surface
that is right almost everywhere but ragged at part seams. A model can move IoU up while
producing a worse mesh.

These metrics look at the extracted surface itself, so they fail on exactly those cases.

**The distance function here is not PyTorch3D's.** Two things rule it out for surface
comparison on SMPL-X-sized geometry:

- At its default `min_triangle_area = 5e-3` every SMPL-X face counts as degenerate, so
  distance falls back to the nearest edge and points sampled *on* the body measure
  ~1.9 mm away. That guard is correct for the training targets (see `geometry.py`) and
  wrong for a millimetre-scale metric.
- Below the guard, its containment test stops discriminating on small triangles in
  float32 and returns the *plane* distance for points whose projection falls outside
  the triangle. At a 5 mm edge it returns 0.0025 m where the true distance is 0.0109 m.
  Since the metric takes a min over triangles, one spuriously small value wins, and
  Chamfer came out roughly 2x too low.

`_p2m` is therefore a brute-force point–triangle distance with an edge-sign containment
test, checked against analytic cases in `test_parity.py`.
"""
from __future__ import annotations

import numpy as np
import torch
import trimesh
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes

# Below any real SMPL-X face area, so the plane branch runs for every real triangle.
# Deliberately NOT geometry.MIN_TRIANGLE_AREA — see the module docstring.
SURFACE_MIN_TRIANGLE_AREA = 1e-12

_EPS = 1e-30


def _point_seg_sq(p: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Squared distance from p to segment [a, b], broadcasting over leading axes."""
    ab = b - a
    t = ((p - a) * ab).sum(-1) / (ab * ab).sum(-1).clamp(min=_EPS)
    proj = a + t.clamp(0.0, 1.0).unsqueeze(-1) * ab
    return ((p - proj) ** 2).sum(-1)


def _point_tri_sq(p: torch.Tensor, v0: torch.Tensor, v1: torch.Tensor, v2: torch.Tensor,
                  min_area: float = SURFACE_MIN_TRIANGLE_AREA) -> torch.Tensor:
    """Squared point–triangle distance. p (P, 1, 3), v* (1, T, 3) -> (P, T).

    Plane distance when the triangle is non-degenerate and the projection lands inside
    it; otherwise the nearest of the three edges.
    """
    n = torch.linalg.cross(v2 - v0, v1 - v0)
    ln = n.norm(dim=-1)
    nn = n / torch.where(ln <= _EPS, torch.ones_like(ln), ln).unsqueeze(-1)
    d_plane = ((p - v0) * nn).sum(-1)
    proj = p - d_plane.unsqueeze(-1) * nn

    # Edge-sign containment: all three cross terms share a sign iff proj is inside.
    c0 = (torch.linalg.cross(v1 - v0, proj - v0) * n).sum(-1)
    c1 = (torch.linalg.cross(v2 - v1, proj - v1) * n).sum(-1)
    c2 = (torch.linalg.cross(v0 - v2, proj - v2) * n).sum(-1)
    inside = ((c0 >= 0) & (c1 >= 0) & (c2 >= 0)) | ((c0 <= 0) & (c1 <= 0) & (c2 <= 0))

    d_edge = torch.minimum(torch.minimum(_point_seg_sq(p, v0, v1), _point_seg_sq(p, v1, v2)),
                           _point_seg_sq(p, v2, v0))
    use_plane = (0.5 * ln >= min_area) & inside
    return torch.where(use_plane, d_plane ** 2, d_edge)


@torch.no_grad()
def _p2m(points: torch.Tensor, verts: torch.Tensor, faces: torch.Tensor,
         point_chunk: int = 8192, tri_chunk: int = 1024) -> torch.Tensor:
    """Unsigned distance from each point (P, 3) to the mesh (verts, faces) -> (P,).

    Blocked over both points and triangles, which bounds the working set at
    `point_chunk x tri_chunk x 3`. An extracted mesh at a 4 mm voxel carries ~300k
    triangles, so the unblocked product would be far beyond GPU memory.
    """
    tri = verts[faces.long()]                                        # (T, 3, 3)
    out = torch.empty(points.shape[0], device=points.device, dtype=points.dtype)
    for i in range(0, points.shape[0], point_chunk):
        p = points[i:i + point_chunk].unsqueeze(1)                    # (P, 1, 3)
        best = torch.full((p.shape[0],), float('inf'), device=points.device,
                          dtype=points.dtype)
        for j in range(0, tri.shape[0], tri_chunk):
            t = tri[j:j + tri_chunk].unsqueeze(0)                     # (1, T, 3, 3)
            d = _point_tri_sq(p, t[..., 0, :], t[..., 1, :], t[..., 2, :])
            best = torch.minimum(best, d.min(dim=1).values)
        out[i:i + point_chunk] = best
    return out.sqrt()


def _mc_gpu(vol: torch.Tensor, level: float):
    """PyTorch3D's CUDA marching cubes.

    Axis convention is the trap. It returns verts whose first column indexes the array's
    **last** axis and whose last column indexes the **first** — measured against an
    asymmetric (8, 4, 16) test volume, not inferred. A cubic grid would hide this
    completely; an anisotropic one turns it into a body squashed along the wrong axis.

    Hard limit: the CUDA kernel refuses volumes above 1024 per axis ("Maximum volume size
    allowed 1K x 1K x 1K"). With the longest body axis at ~1.84 m that puts a floor of
    ~1.8 mm on the voxel unless the volume is tiled.
    """
    # `pytorch3d.ops.marching_cubes` is the *module*; the callable lives inside it.
    from pytorch3d.ops.marching_cubes import marching_cubes

    v, f = marching_cubes(vol.unsqueeze(0), isolevel=level, return_local_coords=False)
    return v[0], f[0]


@torch.no_grad()
def extract_mesh(volume, smpl_output, grid_res: int | None = None,
                 voxel_mm: float | None = None, max_queries: int = 100_000,
                 pad: float = 1.1, level: float | None = None,
                 field: str = 'logit') -> list:
    """Marching cubes over the model's field, on a grid fitted to each body's box.

    Use this rather than the package's inherited `BasicBodyModel.extract_mesh`, which is
    broken for `VolumetricSMPL`: the subclass overrides `query` to return an SDF, while
    the inherited method still inverse-sigmoids `self.query` — so it runs marching cubes
    on `logit(SDF)` and returns a mesh spanning the whole query cube. `query_occupancy`
    is the correct entry point.

    The grid is fitted per axis to the padded body box rather than to a cube on the
    longest dimension, ~4.8x fewer queries for the same voxel. Resolution is `voxel_mm`;
    `grid_res`, if given instead, is the cell count along the longest axis.

    `field` matters more than the voxel size. Measured on one val body at 4 mm:

        field        normal error   mean dihedral
        occupancy       10.72 deg        9.28 deg
        logit            4.63 deg        3.19 deg
        sdf             12.55 deg       12.34 deg

      'occupancy'  `query_occupancy`, level 0.5. The sigmoid goes 0.95 -> 0.05 in ~2 mm,
                   narrower than the voxel, so marching cubes finds it saturated at both
                   ends of a cell edge and vertices snap toward cell midpoints.
      'logit'      the same field inverse-sigmoided, level 0 — the decoder's own smooth
                   pre-activation output. The default.
      'sdf'        `query`, level 0. Metric, but rougher than the occupancy it derives from.

    Returns one trimesh per body, or None where the field never crosses the level set.
    """
    if field not in ('occupancy', 'logit', 'sdf'):
        raise ValueError(f"field must be 'occupancy', 'logit' or 'sdf', got {field!r}")
    if level is None:
        level = 0.5 if field == 'occupancy' else 0.0
    if voxel_mm is None and grid_res is None:
        voxel_mm = 2.0
    verts = smpl_output.vertices
    device, B = verts.device, verts.shape[0]
    b_min, b_max = verts.min(dim=1).values, verts.max(dim=1).values
    centre = (b_min + b_max) * 0.5                      # (B,3)
    box = (b_max - b_min) * pad                         # (B,3) per-axis, padded

    per_body = volume.batchify_smpl_output(smpl_output)
    meshes = []
    for b in range(B):
        bx = box[b]
        step_m = (float(bx.max()) / grid_res) if voxel_mm is None else voxel_mm / 1000.0
        # 1024 is PyTorch3D's kernel limit, not a tuning choice.
        n = torch.clamp((bx / step_m).round().long(), min=8, max=1024)
        nx, ny, nz = (int(v) for v in n)
        origin = centre[b] - bx * 0.5
        step = bx / (n - 1).to(bx.dtype)

        # Coordinates are generated per chunk from flat indices. Materialising the whole
        # grid would be gigabytes: at 2 mm this is 162M points, i.e. 1.9 GB just for xyz.
        # `max_queries` stays at 100k because the decoder expands each point across all
        # K=15 parts — at 1M it tried to allocate 11.6 GiB in a single concat and OOMed.
        total = nx * ny * nz
        vals = torch.empty(total, device=device, dtype=torch.float32)
        for s in range(0, total, max_queries):
            idx = torch.arange(s, min(s + max_queries, total), device=device)
            ix = idx // (ny * nz)
            iy = (idx // nz) % ny
            iz = idx % nz
            pts = torch.stack((origin[0] + ix * step[0],
                               origin[1] + iy * step[1],
                               origin[2] + iz * step[2]), dim=-1)
            fn = volume.query if field == 'sdf' else volume.query_occupancy
            vals[s:s + idx.numel()] = fn(
                pts.unsqueeze(0), per_body[b]).squeeze(0).float()
        volume.detach_cache()

        vol = vals.view(nx, ny, nz)
        if field != 'sdf':
            # An occupancy field is a sigmoid times a 0/1 mask and cannot leave [0,1].
            # Asserting that catches the wrong query method being called (an SDF here
            # ranged -0.087 to 1.134).
            lo, hi = float(vol.min()), float(vol.max())
            if not (-1e-6 <= lo and hi <= 1 + 1e-6):
                raise RuntimeError(
                    f'occupancy field out of range [{lo:.4f}, {hi:.4f}] — '
                    'query_occupancy is not returning occupancy for this model')
        if field == 'logit':
            # 1e-7 matches float32: sigmoid already saturates to exactly 1.0 for a logit
            # above ~17, so the clamp discards nothing that survived the forward pass.
            vol = torch.logit(vol, eps=1e-7)
        lo, hi = float(vol.min()), float(vol.max())
        if not (lo < level < hi):
            meshes.append(None)
            continue

        mv, mf = _mc_gpu(vol, level)
        # PyTorch3D returns x indexing the last array axis and z the first.
        mv = mv.flip(-1) * step + origin
        mesh = trimesh.Trimesh(mv.cpu().numpy(), mf.cpu().numpy())
        # A closed surface with outward normals encloses positive volume, so a negative
        # one means the faces are inside-out.
        if mesh.is_watertight and mesh.volume < 0:
            mesh.invert()
        meshes.append(mesh)
    return meshes


@torch.no_grad()
def compare(pred_verts: torch.Tensor, pred_faces: torch.Tensor,
            gt_verts: torch.Tensor, gt_faces: torch.Tensor,
            n_samples: int = 50_000) -> dict:
    """Bidirectional surface distances between a predicted mesh and the GT body.

    Chamfer is the mean of both directions; Hausdorff the worst point in either. The
    two directions answer different questions — `pred->gt` catches spurious geometry,
    `gt->pred` catches missing geometry — so both are reported separately.
    """
    pred_pts = sample_points_from_meshes(
        Meshes(verts=[pred_verts], faces=[pred_faces]), n_samples)[0].contiguous()
    gt_pts = sample_points_from_meshes(
        Meshes(verts=[gt_verts], faces=[gt_faces]), n_samples)[0].contiguous()

    d_p2g = _p2m(pred_pts, gt_verts, gt_faces)   # spurious surface
    d_g2p = _p2m(gt_pts, pred_verts, pred_faces)  # missing surface

    return {
        'chamfer_mm': float((d_p2g.mean() + d_g2p.mean()) * 0.5 * 1000),
        'pred_to_gt_mm': float(d_p2g.mean() * 1000),
        'gt_to_pred_mm': float(d_g2p.mean() * 1000),
        'hausdorff_mm': float(max(d_p2g.max(), d_g2p.max()) * 1000),
        'p95_mm': float(torch.quantile(torch.cat([d_p2g, d_g2p]), 0.95) * 1000),
    }


def topology(mesh) -> dict:
    """Cheap structural checks on a trimesh. A second connected component is the
    signature of a floating artefact, which point metrics will not report.

    The count alone is not readable — nineteen components where the largest holds
    99.99 % of the area means nineteen specks, not nineteen blobs — so the area outside
    the largest component is reported alongside it. Without that number a finer
    extraction looks like a regression purely because it resolves fragments that a
    coarser one merged into the body.
    """
    try:
        parts = mesh.split(only_watertight=False)
        n_components = len(parts)
        areas = sorted(float(p.area) for p in parts)
        stray_mm2 = sum(areas[:-1]) * 1e6 if len(areas) > 1 else 0.0
    except Exception:
        n_components, stray_mm2 = -1, float('nan')
    return {
        'n_vertices': int(len(mesh.vertices)),
        'n_faces': int(len(mesh.faces)),
        'watertight': bool(mesh.is_watertight),
        'n_components': n_components,
        'stray_area_mm2': stray_mm2,
        'volume_l': float(mesh.volume * 1000) if mesh.is_watertight else float('nan'),
    }


def occupancy_slice(volume, smpl_output, gt_occ_fn, axis: int = 2, res: int = 192,
                    pad: float = 1.15) -> np.ndarray:
    """An RGB image of one planar slice, colour-coded by agreement.

    green  both inside (correct)
    red    GT inside, prediction outside (missed)
    blue   prediction inside, GT outside (hallucinated)
    grey   both outside

    Deliberately not a plot: a per-pixel agreement map shows *where* a model is wrong,
    which a scalar IoU cannot.
    """
    verts = smpl_output.vertices[0]
    device = verts.device
    lo, hi = verts.min(0).values, verts.max(0).values
    c, s = (lo + hi) * 0.5, (hi - lo).max() * pad

    ax = [a for a in range(3) if a != axis]
    g = torch.linspace(-0.5, 0.5, res, device=device) * s
    u, v = torch.meshgrid(g, g, indexing='ij')
    pts = torch.zeros(res * res, 3, device=device)
    pts[:, ax[0]] = u.reshape(-1) + c[ax[0]]
    pts[:, ax[1]] = v.reshape(-1) + c[ax[1]]
    pts[:, axis] = c[axis]
    pts = pts.unsqueeze(0)

    gt = gt_occ_fn(pts).reshape(res, res) > 0.5
    pred = volume.query_occupancy(pts, smpl_output).reshape(res, res) > 0.5
    volume.detach_cache()

    img = np.full((res, res, 3), 32, dtype=np.uint8)
    g_, p_ = gt.cpu().numpy(), pred.cpu().numpy()
    img[g_ & p_] = (60, 200, 90)
    img[g_ & ~p_] = (220, 60, 60)
    img[~g_ & p_] = (70, 120, 240)
    return np.flipud(img)
