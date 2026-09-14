"""Why the ground-truth UDF must keep PyTorch3D's default `min_triangle_area`.

This module exists to stop a plausible-looking "fix" from being applied: lowering the
guard makes the SDF error 10x worse.

**The tempting observation.** The package's `point_mesh_distance` calls
`point_face_distance` with five positional arguments, so it inherits PyTorch3D's
`min_triangle_area` default of 5e-3. Every SMPL-X triangle is far below that (mean face
area 7.8e-5 m², max 7.8e-4 m²), so the "degenerate triangle" fallback applies to all
20,908 faces. Points sampled exactly on the body surface then come back at a mean
distance of 1.91 mm instead of 0. That looks like a bug, and the parameter did not exist
in the PyTorch3D contemporary with the authors' checkpoints.

**Why lowering it is wrong for the training targets.** In this code path, below the
guard the kernel's projection branch loses float32 precision on triangles this small.
Measured on real query points, with `min_triangle_area=1e-8`:

| Point set | True | default 5e-3 | 1e-8 |
|---|---|---|---|
| on-surface | 0 mm | 1.91 mm | 0.00 mm |
| surface + N(0, 0.01) | 7.98 mm | **8.01 mm** | 0.81 mm |
| uniform in part bbox | up to ~150 mm | max 151.7 mm | max 17.7 mm |

The true mean for the jittered set is analytic: an isotropic Gaussian displacement from
a locally planar surface puts the distance at a half-normal, mean sigma*sqrt(2/pi) =
7.98 mm. The default reproduces it to 0.3%. The lowered value collapses distances toward
zero and silently caps far-field ones — it is wrong everywhere except exactly on the
surface, which is the one case that made it look right.

**Conclusion.** Keep the default. Its only inaccuracy is over-estimating for points
within ~2 mm of the surface, where the nearest-edge distance exceeds the perpendicular
one; beyond that the perpendicular dominates and the fallback is accurate. It is also
what the released checkpoint was trained against. `training/test_parity.py` asserts the
half-normal identity so a change that breaks this fails loudly.

Surface metrics are a different question and use their own exact distance — see
`training/meshmetrics.py`. The two must not be unified.
"""
from __future__ import annotations

import math

# PyTorch3D's default. Do not lower it — see the module docstring.
MIN_TRIANGLE_AREA = 5e-3

# E|z| for z ~ N(0, sigma): the mean distance from a jittered surface point back to a
# locally planar surface. The check that catches a broken distance function.
HALF_NORMAL_MEAN = math.sqrt(2.0 / math.pi)
