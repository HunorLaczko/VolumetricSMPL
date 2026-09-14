"""Checks on the package's public API, on real bodies.

Each check is an invariant any correct implementation satisfies, so no stored reference
outputs are needed:

1. Weights: loading is lossless through an `.npz` round trip.
2. Decomposition: no face is in two tight part sets, and each tight set lies inside its
   extended set.
3. Queries: the SDF is negative exactly where the occupancy exceeds 0.5, and a compiled
   query returns what an eager one does.
4. Collision loss: points inside the body penetrate, points far from it do not, and the
   loss has finite gradients with respect to the body.
5. Self-intersection: the rest pose does not intersect itself, an arm driven into the
   hip does, and the loss has finite gradients.
6. Winding numbers: 1 inside a closed cube and 0 outside; 1 inside a posed body and 0 far
   from it.
7. Mesh extraction: part-coloured meshes with the body's extent.

  docker compose run --rm jax python -m training.test_api
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile

import jax
import jax.numpy as jnp
import numpy as np

from VolumetricSMPL import VolumetricSMPL, checkpoint, winding_numbers
from VolumetricSMPL import model as MD
from VolumetricSMPL.assets import FACE_PAD

from . import cache as CA

results: list[tuple[bool, str, str]] = []

# SMPL-X body joints that lie inside the torso and head: pelvis, hips, spine, neck, head.
INNER_JOINTS = [0, 1, 2, 3, 6, 9, 12, 15]
LEFT_SHOULDER = 16

CUBE_VERTS = np.array([
    (-0.5, -0.5, -0.5), (0.5, -0.5, -0.5), (0.5, 0.5, -0.5), (-0.5, 0.5, -0.5),
    (-0.5, -0.5, 0.5), (0.5, -0.5, 0.5), (0.5, 0.5, 0.5), (-0.5, 0.5, 0.5),
], dtype=np.float32)
CUBE_FACES = np.array([
    (0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7), (0, 1, 5), (0, 5, 4),
    (3, 7, 6), (3, 6, 2), (0, 4, 7), (0, 7, 3), (1, 2, 6), (1, 6, 5),
])


def check(name: str, ok: bool, detail: str = ''):
    results.append((bool(ok), name, detail))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-path', default='data/body_models')
    ap.add_argument('--cache', default='val')
    ap.add_argument('--bodies', type=int, default=4)
    args = ap.parse_args()

    model = VolumetricSMPL.create(args.model_path)
    cache = {k: v[:args.bodies] for k, v in CA.load_params(args.cache).items()}
    key = jax.random.PRNGKey(0)

    check_weights(model)
    check_decomposition(model)

    body = model.forward(**cache)
    code = model.encode(body, key)
    check_queries(model, body, code)
    check_collision(model, cache, body, code)
    check_self_collision(model)
    check_winding_numbers(model, body)
    check_extraction(model, body, code)

    width = max(len(name) for _, name, _ in results) + 2
    print('\n' + '=' * (width + 50))
    print('Package API')
    print('=' * (width + 50))
    for ok, name, detail in results:
        print(f"[{'PASS' if ok else 'FAIL'}] {name:<{width}} {detail}")
    print('=' * (width + 50))
    failed = [name for ok, name, _ in results if not ok]
    if failed:
        print(f'\n{len(failed)} check(s) FAILED: {", ".join(failed)}')
        sys.exit(1)
    print(f'\nAll {len(results)} checks passed.')


def check_weights(model):
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'weights.npz')
        checkpoint.save_weights(model.params, path)
        back = checkpoint.load_weights(path)
    same = sorted(back) == sorted(model.params) and all(
        back[k].dtype == v.dtype and bool(jnp.array_equal(back[k], v))
        for k, v in model.params.items())
    n = sum(int(v.size) for v in model.params.values())
    check('Weights survive an .npz round trip', same,
          f'{len(model.params)} tensors, {n:,} parameters')


def check_decomposition(model):
    a = model.assets
    tight, ext = np.asarray(a.tight_faces), np.asarray(a.extended_faces)
    counts = np.zeros(a.faces.shape[0], dtype=int)
    face_id = {tuple(f): i for i, f in enumerate(np.asarray(a.faces))}
    subset = True
    for k in range(a.num_parts):
        t = {tuple(f) for f in tight[k] if f[0] != FACE_PAD}
        e = {tuple(f) for f in ext[k] if f[0] != FACE_PAD}
        subset &= t <= e
        for f in t:
            counts[face_id[f]] += 1
    check('No face is in two tight part sets', counts.max() == 1,
          f'{(counts == 1).sum():,} of {counts.size:,} faces assigned '
          f'({(counts == 0).sum():,} eye faces in none)')
    check('Every tight set lies inside its extended set', subset)


def check_queries(model, body, code):
    points = sample_near_surface(body, jax.random.PRNGKey(1))
    sdf = np.asarray(model.query(points, code))
    occ = np.asarray(model.query_occupancy(points, code))
    # sdf == 0 only where the udf head outputs exactly 0; its sign is then undefined.
    mismatch = ((sdf < 0) != (occ > 0.5)) & (sdf != 0)
    check('SDF < 0 exactly where occupancy > 0.5', mismatch.sum() == 0,
          f'{int(mismatch.sum())} of {sdf.size:,} points disagree, '
          f'{(sdf < 0).mean() * 100:.1f}% inside')

    # Compiled with the weights as arguments: the case XLA's Triton GEMM fusion gets
    # wrong (see VolumetricSMPL/__init__.py).
    a = model.assets
    compiled = jax.jit(lambda w, p, c: MD.query_occupancy(w, p, c, a))
    got = np.asarray(compiled(model.params, points, code))
    diff = float(np.abs(got - occ).max())
    check('Compiled query matches eager', diff < 1e-4, f'max |d| {diff:.1e}')


def check_collision(model, cache, body, code):
    inner = body.joints[:, INNER_JOINTS]
    loss_in, mask_in = model.collision_loss(inner, code, return_mask=True)
    far = body.vertices.mean(axis=1, keepdims=True) + jnp.array([[[3.0, 0.0, 0.0]]])
    far = jnp.broadcast_to(far, inner.shape)
    loss_far = model.collision_loss(far, code)
    check('Joints inside the torso penetrate', bool(jnp.all(mask_in)),
          f'{int(mask_in.sum())} of {mask_in.size}, mean depth '
          f'{float(jnp.mean(-model.query(inner, code))) * 1000:.0f} mm')
    check('Points 3 m away do not', bool(jnp.all(loss_far == 0)),
          f'loss {float(jnp.max(loss_far)):.3g}')
    variants = (model.collision_loss_mean(inner, code),
                model.collision_loss_gmof(inner, code))
    check('Mean and GMoF variants are positive where the sum is',
          all(bool(jnp.all((v > 0) == (loss_in > 0))) for v in variants))

    def objective(betas, body_pose):
        b = model.forward(**{**cache, 'betas': betas, 'body_pose': body_pose})
        c = model.encode(b, jax.random.PRNGKey(0))
        return jnp.sum(model.collision_loss(inner, c))

    g_betas, g_pose = jax.grad(objective, argnums=(0, 1))(cache['betas'],
                                                          cache['body_pose'])
    finite = bool(jnp.all(jnp.isfinite(g_betas)) & jnp.all(jnp.isfinite(g_pose)))
    check('Collision loss has finite, nonzero gradients',
          finite and float(jnp.abs(g_pose).sum()) > 0,
          f'|d/dpose| {float(jnp.abs(g_pose).sum()):.3g}, '
          f'|d/dbetas| {float(jnp.abs(g_betas).sum()):.3g}')


def check_self_collision(model):
    def pose(angle):
        body_pose = np.zeros((1, 63), np.float32)
        # Rest pose is a T-pose with the left arm along +x; rotating the shoulder about
        # z by -135 deg swings the arm down and inward, through the hip.
        body_pose[0, (LEFT_SHOULDER - 1) * 3 + 2] = angle
        return jnp.asarray(body_pose)

    def loss(body_pose):
        b = model.forward(body_pose=body_pose)
        c = model.encode(b, jax.random.PRNGKey(0))
        return model.self_collision_loss(b, c, jax.random.PRNGKey(1))[0]

    rest, crossed = float(loss(pose(0.0))), float(loss(pose(-2.35)))
    check('Rest pose does not intersect itself', rest == 0.0, f'loss {rest:.4g}')
    check('Arm driven into the hip does', crossed > 0.0, f'loss {crossed:.4g}')
    grad = jax.grad(loss)(pose(-2.35))
    check('Self-intersection loss has finite, nonzero gradients',
          bool(jnp.all(jnp.isfinite(grad))) and float(jnp.abs(grad).sum()) > 0,
          f'|d/dpose| {float(jnp.abs(grad).sum()):.3g}')


def check_winding_numbers(model, body):
    tris = jnp.asarray(CUBE_VERTS[CUBE_FACES])[None]
    inside = jnp.array([[[0.0, 0.0, 0.0], [0.1, -0.2, 0.05], [0.45, 0.45, 0.45]]])
    outside = jnp.array([[[2.0, 0.0, 0.0], [0.0, 0.9, 0.0], [0.6, 0.6, 0.6]]])
    w_in = np.asarray(winding_numbers(inside, tris))
    w_out = np.asarray(winding_numbers(outside, tris))
    check('Winding number is 1 inside a cube and 0 outside',
          np.allclose(w_in, 1, atol=1e-4) and np.allclose(w_out, 0, atol=1e-4),
          f'inside {np.round(w_in[0], 5).tolist()}, outside {np.round(w_out[0], 5).tolist()}')

    # On a posed body, away from self-intersections: 1 at the joints inside the torso,
    # 0 far away.
    tris = body.vertices[:, np.asarray(model.assets.faces)]
    inner = body.joints[:, INNER_JOINTS]
    far = inner + jnp.array([3.0, 0.0, 0.0])
    w_inner = np.asarray(winding_numbers(inner, tris))
    w_far = np.asarray(winding_numbers(far, tris))
    check('Winding number is 1 inside a posed body and 0 far from it',
          np.allclose(w_inner, 1, atol=0.05) and np.allclose(w_far, 0, atol=0.05),
          f'inner joints {w_inner.min():.3f}..{w_inner.max():.3f}, '
          f'far {np.abs(w_far).max():.1e}')


def check_extraction(model, body, code, tol_m: float = 0.005):
    one = type(body)(*(x[:1] for x in body))
    mesh = model.extract_mesh(one, {k: v[:1] for k, v in code.items()})[0]
    if mesh is None:
        check('extract_mesh finds a surface', False)
        return
    v0 = np.asarray(one.vertices[0])
    worst = float(np.abs(mesh.extents - (v0.max(0) - v0.min(0))).max())
    colours = len(np.unique(mesh.visual.vertex_colors[:, :3], axis=0))
    check('Extracted mesh has the body extent', worst < tol_m,
          f'worst axis {worst * 1000:.1f} mm, {len(mesh.vertices):,} vertices')
    check('Extracted mesh is coloured by part', colours == model.num_parts,
          f'{colours} distinct colours for {model.num_parts} parts')


def sample_near_surface(body, key, n: int = 20_000, sigma: float = 0.02):
    """Points around the surface: jittered vertices, so both signs are well represented."""
    B, V = body.vertices.shape[:2]
    idx = jax.random.randint(key, (n,), 0, V)
    noise = jax.random.normal(jax.random.split(key)[1], (B, n, 3)) * sigma
    return body.vertices[:, idx] + noise


if __name__ == '__main__':
    main()
