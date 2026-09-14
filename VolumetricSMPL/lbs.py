"""SMPL-X forward pass: linear blend skinning.

A transcription of `smplx.lbs` and `smplx.SMPLX.forward` (smplx 0.1.28). The model buffers
come from `assets.load_smplx`, which applies the same processing smplx does at
construction.

Two details are load-bearing and easy to get subtly wrong:

- `batch_rodrigues` computes `angle = ||rot_vec + 1e-8||`. The epsilon is added to the
  **vector**, not to the norm. For a zero rotation the two differ, and SMPL-X bodies are
  full of exactly-zero joint rotations, so getting this wrong is not a corner case here.
- `full_pose += pose_mean` after assembly. With `flat_hand_mean=False` -- the setting the
  released checkpoints were trained with -- the mean is nonzero on the 30 hand joints, so
  omitting it moves every finger.

Only `J_transformed[:mK]` (mK = 22) is consumed by the volume, so the forward pass stops at
LBS: no vertex-joint selector and none of smplx's 51 appended landmarks.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

# SMPL-X pose layout: 1 global + 21 body + 1 jaw + 2 eyes + 30 hand = 55 joints.
POSE_BLOCKS = (
    ('global_orient', 1),
    ('body_pose', 21),
    ('jaw_pose', 1),
    ('leye_pose', 1),
    ('reye_pose', 1),
    ('left_hand_pose', 15),
    ('right_hand_pose', 15),
)


def assemble_full_pose(params: dict, pose_mean: jnp.ndarray) -> jnp.ndarray:
    """(B, 165) axis-angle, in SMPL-X's joint order, with the model's mean pose added."""
    parts = []
    for name, n in POSE_BLOCKS:
        parts.append(params[name].reshape(-1, n, 3))
    return jnp.concatenate(parts, axis=1).reshape(-1, 165) + pose_mean


def batch_rodrigues(rot_vecs: jnp.ndarray) -> jnp.ndarray:
    """(N, 3) axis-angle -> (N, 3, 3) rotation matrices."""
    # Epsilon on the vector, matching smplx exactly. See the module docstring.
    angle = jnp.linalg.norm(rot_vecs + 1e-8, axis=1, keepdims=True)
    rot_dir = rot_vecs / angle

    cos = jnp.expand_dims(jnp.cos(angle), axis=1)
    sin = jnp.expand_dims(jnp.sin(angle), axis=1)

    rx, ry, rz = jnp.split(rot_dir, 3, axis=1)
    zeros = jnp.zeros_like(rx)
    K = jnp.concatenate([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros],
                        axis=1).reshape(-1, 3, 3)

    ident = jnp.eye(3, dtype=rot_vecs.dtype)[None]
    return ident + sin * K + (1 - cos) * (K @ K)


def _transform_mat(R: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
    """(N,3,3), (N,3,1) -> (N,4,4) homogeneous transform."""
    top = jnp.concatenate([R, t], axis=2)                       # (N, 3, 4)
    bottom = jnp.broadcast_to(jnp.array([0.0, 0.0, 0.0, 1.0], dtype=R.dtype),
                              (R.shape[0], 1, 4))
    return jnp.concatenate([top, bottom], axis=1)


def batch_rigid_transform(rot_mats: jnp.ndarray, joints: jnp.ndarray,
                          parents: np.ndarray):
    """Walk the kinematic chain.

    Returns posed joints (B, J, 3) and the rest-pose-removed transforms (B, J, 4, 4)
    that skinning blends.

    `parents` is a host array so the chain unrolls at trace time. J is 55, which is a
    perfectly reasonable unroll and avoids a scan with a data-dependent gather.
    """
    joints = joints[..., None]                                  # (B, J, 3, 1)
    rel = joints.at[:, 1:].add(-joints[:, parents[1:]])

    B, J = rel.shape[0], rel.shape[1]
    mats = _transform_mat(rot_mats.reshape(-1, 3, 3),
                          rel.reshape(-1, 3, 1)).reshape(B, J, 4, 4)

    chain = [mats[:, 0]]
    for i in range(1, len(parents)):
        chain.append(chain[parents[i]] @ mats[:, i])
    transforms = jnp.stack(chain, axis=1)                       # (B, J, 4, 4)

    posed_joints = transforms[:, :, :3, 3]

    # Subtract each joint's rest position, expressed in its posed frame, so the result
    # maps rest-pose vertices rather than already-posed ones.
    joints_homogen = jnp.concatenate(
        [joints, jnp.zeros_like(joints[:, :, :1])], axis=2)     # (B, J, 4, 1)
    shifted = transforms @ joints_homogen                       # (B, J, 4, 1)
    pad = jnp.zeros(shifted.shape[:-1] + (3,), dtype=shifted.dtype)
    rel_transforms = transforms - jnp.concatenate([pad, shifted], axis=3)

    return posed_joints, rel_transforms


def lbs(shape_components: jnp.ndarray, full_pose: jnp.ndarray, a):
    """Linear blend skinning.

    Args:
        shape_components: (B, n_betas + n_expr)
        full_pose: (B, 165) axis-angle, mean already added
        a: `assets.Assets`
    Returns:
        vertices (B, V, 3), posed joints (B, J, 3)
    """
    B = full_pose.shape[0]
    shapedirs = jnp.concatenate([a.shapedirs, a.expr_dirs], axis=-1)

    v_shaped = a.v_template + jnp.einsum('bl,mkl->bmk', shape_components, shapedirs)
    J = jnp.einsum('bik,ji->bjk', v_shaped, a.J_regressor)

    rot_mats = batch_rodrigues(full_pose.reshape(-1, 3)).reshape(B, -1, 3, 3)

    # Pose-corrective blend shapes, driven by every joint except the root.
    ident = jnp.eye(3, dtype=full_pose.dtype)
    pose_feature = (rot_mats[:, 1:] - ident).reshape(B, -1)
    pose_offsets = (pose_feature @ a.posedirs).reshape(B, -1, 3)
    v_posed = pose_offsets + v_shaped

    posed_joints, A = batch_rigid_transform(rot_mats, J, a.parents)

    n_joints = a.n_joints
    T = (a.lbs_weights @ A.reshape(B, n_joints, 16)).reshape(B, -1, 4, 4)

    v_homo = jnp.concatenate(
        [v_posed, jnp.ones(v_posed.shape[:-1] + (1,), dtype=v_posed.dtype)], axis=-1)
    verts = jnp.einsum('bvij,bvj->bvi', T, v_homo)[..., :3]

    return verts, posed_joints


def forward(params: dict, a):
    """SMPL-X forward from a dict of smplx-named parameters.

    Returns (vertices, joints, full_pose). `joints` is LBS's `J_transformed`, i.e. the
    55 skeleton joints; smplx appends tips and landmarks after these, which the volume
    never reads. `expression` and `transl` are optional and default to zero.
    """
    full_pose = assemble_full_pose(params, a.pose_mean)
    n_expr = int(a.expr_dirs.shape[-1])
    expression = params.get('expression')
    if expression is None:
        expression = jnp.zeros((full_pose.shape[0], n_expr), dtype=full_pose.dtype)
    shape_components = jnp.concatenate([params['betas'], expression], axis=-1)
    verts, joints = lbs(shape_components, full_pose, a)
    transl = params.get('transl')
    if transl is not None:
        verts = verts + transl[:, None, :]
        joints = joints + transl[:, None, :]
    return verts, joints, full_pose
