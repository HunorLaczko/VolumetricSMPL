"""Environment check: prove the container can run what the code needs.

Every geometry gate is held to known analytic answers on a unit cube, so a pass means the
stack computes the right thing, not merely that it runs.

Run:  docker compose run --rm jax python docker/verify_env.py
Exits non-zero if any gate fails.
"""

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
        except Exception as exc:  # noqa: BLE001 - report every failure, not just the first
            results.append((False, name, f"{type(exc).__name__}: {exc}"))
            notes.append(traceback.format_exc())
        return fn
    return wrap


@gate("JAX CUDA device visible")
def _jax_device():
    import jax

    devs = jax.devices()
    if not any(d.platform == "gpu" for d in devs):
        raise RuntimeError(f"no GPU device; jax.devices() = {devs}")
    d = devs[0]
    return f"{d.device_kind} | jax {jax.__version__} | backend {d.platform}"


@gate("JAX runs a real kernel on the GPU")
def _jax_matmul():
    import jax
    import jax.numpy as jnp

    dev = jax.devices()[0]
    # Compute capability alone would not prove the build carries kernels for this
    # architecture, so this compiles and executes rather than just querying.
    x = jnp.ones((1024, 1024), dtype=jnp.float32)
    got = float(jax.jit(lambda a: (a @ a).sum())(x))
    want = 1024.0 ** 3
    if abs(got - want) / want > 1e-5:
        raise RuntimeError(f"matmul wrong: {got} != {want}")
    cc = getattr(dev, "compute_capability", "unknown")
    return f"compute capability {cc}, 1024^2 matmul executed and correct"


@gate("Compiled decoder matches eager execution")
def _compiled_decoder():
    import jax
    import jax.numpy as jnp
    import numpy as np

    from VolumetricSMPL import modules as M

    # Random weights with the decoder's real shapes. XLA's Triton GEMM fusion gets this
    # wrong on GPU (by ~36% here) when the weights are runtime inputs; the image disables
    # it via XLA_FLAGS. This runs after the device gates, so it tests the environment's
    # setting rather than the one the package applies at import.
    rng = np.random.default_rng(0)
    K, T, R, P, C = 15, 2048, 80, 10, 128
    w = {}
    for layer, (o, i) in enumerate([(64, 143), (64, 64), (64, 64), (64, 207),
                                    (64, 64), (64, 64), (2, 64)]):
        p = f"decoder.lin{layer}"
        w[f"{p}.weight"] = rng.uniform(-1, 1, (o, i)) / np.sqrt(i)
        w[f"{p}.bias"] = rng.uniform(-1, 1, (o,)) / np.sqrt(i)
        w[f"{p}.rf_weight"] = rng.standard_normal((1, R, o * i)) * 1e-3
        w[f"{p}.cond2weight.0.weight"] = rng.uniform(-0.09, 0.09, (R, C))
        w[f"{p}.cond2weight.0.bias"] = rng.uniform(-0.09, 0.09, (R,))
        w[f"{p}.part_weight"] = rng.standard_normal((P, o * i)) * 1e-3
        w[f"{p}.part_weight2rank"] = rng.standard_normal((K, P))
    w = {k: jnp.asarray(v, jnp.float32) for k, v in w.items()}
    z = jnp.asarray(np.concatenate([rng.uniform(-0.3, 0.3, (K, T, 3)),
                                    rng.standard_normal((K, T, C)) * 0.5], -1), jnp.float32)
    cond = jnp.asarray(rng.standard_normal((K, C)), jnp.float32)
    ind = jnp.arange(K)

    eager = M.decode(w, z, cond, ind)
    compiled = jax.jit(M.decode)(w, z, cond, ind)
    rel = float(jnp.abs(eager - compiled).max() / jnp.abs(eager).max())
    if rel > 1e-4:
        raise RuntimeError(f"jit output differs from eager by {rel:.2e} relative; "
                           "set XLA_FLAGS=--xla_gpu_enable_triton_gemm=false")
    return f"max relative difference {rel:.1e}"


@gate("Parity occupancy reproduces analytic answers (training.occupancy)")
def _parity():
    import jax.numpy as jnp

    from training import occupancy as O

    # The inside probe is off-centre on purpose. Occupancy casts along a fixed
    # near-(1,1,1) direction, which from the origin leaves the cube exactly through a
    # corner vertex -- the grazing case where parity is genuinely ambiguous. From
    # (0.1, -0.2, 0.05) the ray exits through the interior of the +x face.
    probes = [(0.1, -0.2, 0.05)] + PROBE_POINTS[1:]
    v = jnp.asarray(CUBE_VERTS, dtype=jnp.float32)[None]
    p = jnp.asarray(probes, dtype=jnp.float32)[None]
    f = jnp.asarray(CUBE_FACES, dtype=jnp.int32)
    got = [bool(x) for x in O.occupancy(v, p, f)[0]]
    if got != PROBE_INSIDE:
        raise RuntimeError(f"occupancy {got} at {probes} != analytic {PROBE_INSIDE}")
    return f"occupancy {got} matches analytic"


@gate("Surface distance reproduces analytic answers (training.meshmetrics)")
def _distance():
    import numpy as np

    from training import meshmetrics as MM

    got = [round(float(x), 4) for x in MM.surface_distance(
        np.asarray(PROBE_POINTS, dtype=np.float32),
        np.asarray(CUBE_VERTS, dtype=np.float32),
        np.asarray(CUBE_FACES, dtype=np.int32))]
    for g, want in zip(got, PROBE_DIST):
        if abs(g - want) > 1e-3:
            raise RuntimeError(f"distances {got} != analytic {PROBE_DIST}")
    return f"distances {got} match analytic {PROBE_DIST}"


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
sys.exit(0)
