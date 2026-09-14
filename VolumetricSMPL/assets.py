"""SMPL-X model buffers and the part decomposition, built from the model file.

Nothing here is learned. `load_smplx` reads `SMPLX_<GENDER>.npz` and applies the processing
`smplx.SMPLX` does at construction, so the forward pass starts from the same buffers:
shape directions are sliced to `num_betas`, expression directions are split off, the pose
correctives are flattened, and the hand means are folded into `pose_mean`.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import jax.numpy as jnp
import numpy as np

from . import partition as P

# SMPL-X shape space layout: 300 shape components, then 100 expression components.
SHAPE_SPACE_DIM = 300

# VolumetricSMPL's fixed hyper-parameters (the original package's `_init`).
BBOX_PADDING = 1.125
N_SAMPLES = 1000
LEVEL_SET = 0.5

FACE_PAD = P.FACE_PAD


@dataclass(frozen=True)
class Assets:
    """SMPL-X buffers and the part decomposition, as device arrays."""

    # --- SMPL-X ---------------------------------------------------------------
    v_template: jnp.ndarray        # (V, 3)
    shapedirs: jnp.ndarray         # (V, 3, n_betas)
    expr_dirs: jnp.ndarray         # (V, 3, n_expr)
    posedirs: jnp.ndarray          # (9*(J-1), V*3)
    J_regressor: jnp.ndarray       # (J, V)
    lbs_weights: jnp.ndarray       # (V, J)
    parents: np.ndarray            # (J,) kept on host: it indexes an unrolled chain
    faces: jnp.ndarray             # (F, 3)
    pose_mean: jnp.ndarray         # (J*3,)

    # --- decomposition --------------------------------------------------------
    tight_faces: jnp.ndarray       # (K, Ft, 3), -1 padded
    extended_faces: jnp.ndarray    # (K, Fe, 3), -1 padded
    tight_vert_selector: jnp.ndarray   # (K, Vt)
    joint_mapper: np.ndarray       # (mK,) bool, host
    selfpen_disable_mat: jnp.ndarray   # (K, K) bool

    meta: dict = field(default_factory=dict)

    @property
    def num_parts(self) -> int:
        return int(self.tight_faces.shape[0])

    @property
    def mK(self) -> int:
        """How far down the kinematic chain the part transforms are taken.

        15 parts + 7 merged = 22 for SMPL-X, so only `J_transformed[:mK]` is needed.
        """
        return int(self.joint_mapper.shape[0])

    @property
    def bbox_padding(self) -> float:
        return BBOX_PADDING

    @property
    def n_verts(self) -> int:
        return int(self.v_template.shape[0])

    @property
    def n_joints(self) -> int:
        return int(self.J_regressor.shape[0])


def find_model_file(model_path: str, gender: str = 'neutral') -> str:
    """Accept the `.npz` itself, a `smplx/` directory, or its parent (as `smplx.create`)."""
    if os.path.isfile(model_path):
        return model_path
    name = f'SMPLX_{gender.upper()}.npz'
    for candidate in (os.path.join(model_path, 'smplx', name),
                      os.path.join(model_path, name)):
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(f'no {name} under {model_path}')


def load_smplx(path: str, num_betas: int = 16, num_expression_coeffs: int = 10,
               flat_hand_mean: bool = False) -> dict:
    """SMPL-X buffers as numpy arrays, processed the way `smplx.SMPLX` processes them."""
    with np.load(path, allow_pickle=True) as d:
        data = {k: d[k] for k in d.files}

    shapedirs = data['shapedirs']
    if shapedirs.shape[-1] < SHAPE_SPACE_DIM + num_expression_coeffs:
        raise ValueError(f'{path} has {shapedirs.shape[-1]} shape components; '
                         'the SMPL-X 1.1 model with 400 is required')
    n_pose_basis = data['posedirs'].shape[-1]
    parents = data['kintree_table'][0].astype(np.int64)
    parents[0] = -1

    left_hand_mean = data['hands_meanl']
    right_hand_mean = data['hands_meanr']
    if flat_hand_mean:
        left_hand_mean = np.zeros_like(left_hand_mean)
        right_hand_mean = np.zeros_like(right_hand_mean)
    # global orient, 21 body joints, jaw and both eyes carry no mean.
    pose_mean = np.concatenate([np.zeros(3 + 21 * 3 + 3 + 3 + 3),
                                left_hand_mean, right_hand_mean])

    f32 = np.float32
    return dict(
        v_template=data['v_template'].astype(f32),
        shapedirs=shapedirs[:, :, :num_betas].astype(f32),
        expr_dirs=shapedirs[:, :, SHAPE_SPACE_DIM:SHAPE_SPACE_DIM + num_expression_coeffs]
        .astype(f32),
        posedirs=np.reshape(data['posedirs'], [-1, n_pose_basis]).T.astype(f32),
        J_regressor=np.asarray(data['J_regressor']).astype(f32),
        lbs_weights=data['weights'].astype(f32),
        parents=parents,
        faces=data['f'].astype(np.int32),
        pose_mean=pose_mean.astype(f32),
    )


def build(model_path: str, gender: str = 'neutral', num_betas: int = 16,
          num_expression_coeffs: int = 10, flat_hand_mean: bool = False) -> Assets:
    path = find_model_file(model_path, gender)
    body = load_smplx(path, num_betas, num_expression_coeffs, flat_hand_mean)
    parts = P.partition(body['lbs_weights'], body['faces'], body['parents'])

    return Assets(
        v_template=jnp.asarray(body['v_template']),
        shapedirs=jnp.asarray(body['shapedirs']),
        expr_dirs=jnp.asarray(body['expr_dirs']),
        posedirs=jnp.asarray(body['posedirs']),
        J_regressor=jnp.asarray(body['J_regressor']),
        lbs_weights=jnp.asarray(body['lbs_weights']),
        # parents drives a Python-level unrolled loop over the kinematic chain, so it
        # must stay a host array -- a traced value cannot index a list.
        parents=body['parents'].astype(np.int32),
        faces=jnp.asarray(body['faces']),
        pose_mean=jnp.asarray(body['pose_mean']),
        tight_faces=jnp.asarray(parts['tight_faces']),
        extended_faces=jnp.asarray(parts['extended_faces']),
        tight_vert_selector=jnp.asarray(parts['tight_vert_selector']),
        joint_mapper=parts['joint_mapper'],
        selfpen_disable_mat=jnp.asarray(parts['selfpen_disable_mat']),
        meta=dict(model_file=path, gender=gender, num_betas=num_betas,
                  num_expression_coeffs=num_expression_coeffs,
                  flat_hand_mean=flat_hand_mean),
    )
