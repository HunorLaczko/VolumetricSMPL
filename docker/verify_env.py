"""Environment check: prove the container can run every geometry op the pipeline uses.

Deliberately exercises the *real* dependency surface rather than a generic smoke
test — the exact PyTorch3D entry points the model package imports, and the kaolin ops
that ground-truth generation is built on — each against a unit cube with analytically
known answers.

Run:  docker compose run --rm train python docker/verify_env.py
Exits non-zero if any gate fails.
"""

import math
import sys
import traceback

# Unit cube centred on the origin, side 1, outward-facing winding.
# Known answers: the origin is inside and exactly 0.5 from the surface;
# (2, 0, 0) is outside and exactly 1.5 from it.
CUBE_VERTS = [
    (-0.5, -0.5, -0.5), (0.5, -0.5, -0.5), (0.5, 0.5, -0.5), (-0.5, 0.5, -0.5),
    (-0.5, -0.5, 0.5), (0.5, -0.5, 0.5), (0.5, 0.5, 0.5), (-0.5, 0.5, 0.5),
]
CUBE_FACES = [
    (0, 2, 1), (0, 3, 2),  # -z
    (4, 5, 6), (4, 6, 7),  # +z
    (0, 1, 5), (0, 5, 4),  # -y
    (3, 7, 6), (3, 6, 2),  # +y
    (0, 4, 7), (0, 7, 3),  # -x
    (1, 2, 6), (1, 6, 5),  # +x
]
PROBE_POINTS = [(0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (0.0, 0.9, 0.0)]
PROBE_INSIDE = [True, False, False]
PROBE_DIST = [0.5, 1.5, 0.4]

results = []
notes = []


def gate(name):
    """Register a gate; the decorated function returns a detail string or raises."""
    def wrap(fn):
        try:
            detail = fn()
            results.append((True, name, detail))
        except Exception as exc:  # noqa: BLE001 - we want every failure reported, not the first
            results.append((False, name, f"{type(exc).__name__}: {exc}"))
            notes.append(traceback.format_exc())
        return fn
    return wrap


import torch  # noqa: E402


def cube(device):
    v = torch.tensor(CUBE_VERTS, dtype=torch.float32, device=device)
    f = torch.tensor(CUBE_FACES, dtype=torch.int64, device=device)
    p = torch.tensor(PROBE_POINTS, dtype=torch.float32, device=device)
    return v, f, p


@gate("CUDA device visible")
def _cuda():
    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is False")
    return f"{torch.cuda.get_device_name(0)} | torch {torch.__version__} | cuda {torch.version.cuda}"


@gate("GPU architecture supported by this torch build")
def _capability():
    cap = torch.cuda.get_device_capability()
    arch = f"sm_{cap[0]}{cap[1]}"
    if arch not in torch.cuda.get_arch_list():
        raise RuntimeError(f"{arch} not in torch build {torch.cuda.get_arch_list()}")
    # A real kernel launch: the arch list alone does not prove the kernels execute.
    x = torch.randn(1024, 1024, device="cuda")
    torch.mm(x, x).sum().item()
    return f"{arch}, matmul executed"


@gate("PyTorch3D sample_points_from_meshes on GPU")
def _p3d_sample():
    from pytorch3d.structures import Meshes
    from pytorch3d.ops import sample_points_from_meshes
    import pytorch3d

    v, f, _ = cube("cuda")
    pts = sample_points_from_meshes(Meshes(verts=[v], faces=[f]), 4096)
    if pts.device.type != "cuda":
        raise RuntimeError(f"result landed on {pts.device}, not cuda")
    # Every sample must lie on the cube's surface: max |coord| == 0.5.
    off = (pts.abs().amax(-1) - 0.5).abs().max().item()
    if off > 1e-4:
        raise RuntimeError(f"samples are not on the surface (max deviation {off:.2e})")
    return f"pytorch3d {pytorch3d.__version__} | 4096 pts, max surface deviation {off:.2e}"


@gate("PyTorch3D point_face_distance on GPU (the package's own wrapper)")
def _p3d_distance():
    from pytorch3d.structures import Meshes, Pointclouds
    from VolumetricSMPL.volumetric_smpl import point_mesh_distance

    v, f, p = cube("cuda")
    d = point_mesh_distance(Meshes(verts=[v], faces=[f]), Pointclouds([p]))
    if d.device.type != "cuda":
        raise RuntimeError(f"result landed on {d.device}, not cuda")
    got = [round(x, 4) for x in d.flatten().tolist()]
    for g, want in zip(got, PROBE_DIST):
        if not math.isclose(g, want, abs_tol=1e-3):
            raise RuntimeError(f"unsigned distances {got} != expected {PROBE_DIST}")
    return f"distances {got} match analytic {PROBE_DIST}"


@gate("kaolin check_sign on GPU (parity occupancy)")
def _kaolin_sign():
    import kaolin

    v, f, p = cube("cuda")
    inside = kaolin.ops.mesh.check_sign(v.unsqueeze(0), f, p.unsqueeze(0))
    if inside.device.type != "cuda":
        raise RuntimeError(f"result landed on {inside.device}, not cuda")
    got = inside.flatten().tolist()
    if got != PROBE_INSIDE:
        raise RuntimeError(f"inside/outside {got} != expected {PROBE_INSIDE}")
    return f"kaolin {kaolin.__version__} | occupancy {got} matches analytic"


@gate("kaolin sample_points + point_to_mesh_distance on GPU")
def _kaolin_geom():
    import kaolin
    from kaolin.ops.mesh import index_vertices_by_faces
    from kaolin.metrics.trianglemesh import point_to_mesh_distance

    v, f, p = cube("cuda")
    pts, _ = kaolin.ops.mesh.sample_points(v.unsqueeze(0), f, 4096)
    off = (pts.abs().amax(-1) - 0.5).abs().max().item()
    if off > 1e-4:
        raise RuntimeError(f"samples are not on the surface (max deviation {off:.2e})")

    face_verts = index_vertices_by_faces(v.unsqueeze(0), f)
    raw = point_to_mesh_distance(p.unsqueeze(0), face_verts)[0].flatten()
    if raw.device.type != "cuda":
        raise RuntimeError(f"result landed on {raw.device}, not cuda")

    # kaolin returns squared distance; determine it from the analytic answers rather
    # than assume it, since a missing sqrt produces a plausible-looking loss curve.
    want = torch.tensor(PROBE_DIST, device=raw.device)
    if torch.allclose(raw, want, atol=1e-3):
        convention = "raw (NOT squared)"
    elif torch.allclose(raw.sqrt(), want, atol=1e-3):
        convention = "SQUARED - callers must sqrt()"
    else:
        raise RuntimeError(f"got {raw.tolist()}, matches neither {PROBE_DIST} nor its square")
    notes.append(f"kaolin point_to_mesh_distance returns {convention}")
    return f"surface deviation {off:.2e} | distance convention: {convention}"


@gate("VolumetricSMPL package imports")
def _package():
    import VolumetricSMPL

    return f"VolumetricSMPL from {VolumetricSMPL.__file__}"


width = max(len(n) for _, n, _ in results) + 2
print("\n" + "=" * (width + 60))
print("Environment check")
print("=" * (width + 60))
for ok, name, detail in results:
    print(f"[{'PASS' if ok else 'FAIL'}] {name:<{width}} {detail}")
print("=" * (width + 60))

failed = [n for ok, n, _ in results if not ok]
if failed:
    print(f"\n{len(failed)} gate(s) FAILED: {', '.join(failed)}\n")
    for n in notes:
        if "Traceback" in n:
            print(n)
    sys.exit(1)

print("\nAll gates passed.")
for n in notes:
    if "Traceback" not in n:
        print(f"  note: {n}")
sys.exit(0)
