"""Mesh extraction: marching cubes over the learned field.

The original package's `extract_mesh` does not work for `VolumetricSMPL`: it inverts a
sigmoid over the output of `query`, which is a signed distance, not an occupancy (see
FINDINGS.md). This runs marching cubes on the occupancy field instead, on a grid fitted to
the body's box.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np

# The original's part colour table; the first K rows colour the K parts.
PART_COLORS = np.array([
    [8.94117647e-01, 1.01960784e-01, 1.09803922e-01],
    [2.15686275e-01, 4.94117647e-01, 7.21568627e-01],
    [3.01960784e-01, 6.86274510e-01, 2.90196078e-01],
    [5.96078431e-01, 3.05882353e-01, 6.39215686e-01],
    [1.00000000e+00, 4.98039216e-01, 0.0],
    [1.00000000e+00, 1.00000000e+00, 2.00000000e-01],
    [6.50980392e-01, 3.37254902e-01, 1.56862745e-01],
    [9.68627451e-01, 5.05882353e-01, 7.49019608e-01],
    [6.00000000e-01, 6.00000000e-01, 6.00000000e-01],
    [1.00000000e+00, 9.17647059e-01, 8.47058824e-01],
    [4.94117647e-01, 0.0, 1.84313725e-01],
    [7.92156863e-01, 7.29411765e-01, 3.72549020e-01],
    [5.25490196e-01, 4.78431373e-01, 0.0],
    [9.29411765e-01, 6.50980392e-01, 5.76470588e-01],
    [5.05882353e-01, 4.03921569e-01, 4.15686275e-01],
    [7.80392157e-01, 3.56862745e-01, 4.43137255e-01],
    [6.86274510e-01, 5.37254902e-01, 3.52941176e-01],
    [6.27450980e-01, 8.23529412e-02, 0.0],
    [1.00000000e+00, 4.35294118e-01, 3.80392157e-01],
    [7.37254902e-01, 0.0, 2.82352941e-01],
    [1.00000000e+00, 9.09803922e-01, 5.52941176e-01],
    [1.00000000e+00, 7.84313725e-02, 4.15686275e-01],
    [7.72549020e-01, 7.52941176e-01, 6.66666667e-01],
    [5.68627451e-01, 2.62745098e-01, 2.98039216e-01],
    [0.0, 0.0, 0.0],
])


def box_grid(bmin, bmax, voxel_m: float, pad: float = 1.1):
    """Axis-fitted sample grid.

    Fitted to the body's box per axis rather than to a cube on its longest dimension.
    A typical body is 1.63 x 0.43 x 1.84 m; a cube on the longest axis wastes ~79% of
    every query on space the body cannot occupy, and gives the thin depth axis the same
    resolution as the tall one. Per-axis fitting is ~4.8x cheaper for the same voxel.
    """
    center = (bmin + bmax) * 0.5
    size = (bmax - bmin) * pad
    n = np.maximum(np.ceil(size / voxel_m).astype(int), 2)
    axes = [np.linspace(center[i] - size[i] / 2, center[i] + size[i] / 2, n[i])
            for i in range(3)]
    grid = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1)
    return grid.astype(np.float32), n, axes


def extract_mesh(query_fn, bmin, bmax, voxel_mm: float = 4.0, field: str = 'logit',
                 level: float | None = None, max_queries: int = 16_384):
    """Marching cubes over an occupancy query, on a box-fitted grid.

    `query_fn` maps (1, T, 3) points to (1, T) occupancy. Returns (verts, faces, info), or
    (None, None, info) when the level set does not cross the grid.

    `field` matters far more than the voxel size. The occupancy field is a *steep*
    sigmoid -- it goes 0.95 -> 0.05 in about 2 mm, narrower than a 4 mm voxel -- so
    marching cubes, which places each vertex by linear interpolation along a cell edge,
    finds the field already saturated at both ends of the edge and snaps vertices toward
    cell midpoints. `logit` inverts the sigmoid to recover the decoder's own smooth
    pre-activation output, which cuts the normal error from 10.7 deg to 4.6 deg.

    `max_queries` counts *grid points*, not decoder rows. The decoder evaluates every
    point against all K=15 parts, so the working set is 15x the chunk.
    """
    from skimage import measure

    if level is None:
        level = 0.5 if field == 'occupancy' else 0.0

    grid, n, axes = box_grid(np.asarray(bmin), np.asarray(bmax), voxel_mm / 1000.0)
    flat = grid.reshape(1, -1, 3)

    vals = []
    for i in range(0, flat.shape[1], max_queries):
        vals.append(np.asarray(query_fn(jnp.asarray(flat[:, i:i + max_queries]))))
    vol = np.concatenate(vals, axis=1).reshape(tuple(n))

    if field == 'logit':
        vol = np.log(np.clip(vol, 1e-7, 1 - 1e-7) / (1 - np.clip(vol, 1e-7, 1 - 1e-7)))
    elif field != 'occupancy':
        raise ValueError(f'unknown field {field!r}')

    lo, hi = float(vol.min()), float(vol.max())
    if not (lo < level < hi):
        return None, None, dict(empty=True, vmin=lo, vmax=hi, level=level)

    verts, faces, _normals, _ = measure.marching_cubes(vol, level=level)
    # Index space -> world. Each axis has its own spacing.
    for i in range(3):
        step = axes[i][1] - axes[i][0]
        verts[:, i] = verts[:, i] * step + axes[i][0]
    return verts, faces, dict(empty=False, vmin=lo, vmax=hi, level=level,
                              grid=tuple(int(x) for x in n))


def part_colors(labels: np.ndarray) -> np.ndarray:
    """Part index -> RGB uint8."""
    return np.round(np.clip(PART_COLORS[labels], 0.0, 1.0) * 255).astype(np.uint8)
