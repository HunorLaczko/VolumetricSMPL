"""The public model: an SMPL-X body with a learned volume attached.

Where the original attaches a volume to a PyTorch `smplx` model and caches the body
encoding internally, this is pure-functional: `forward` poses the body, `encode` computes
everything about it that does not depend on the query point, and every query takes that
encoding explicitly. All methods are differentiable and can be wrapped in `jax.jit` or
`jax.grad`.

    model = VolumetricSMPL.create('data/body_models')
    body = model.forward(betas=betas, body_pose=body_pose, transl=transl)
    code = model.encode(body, jax.random.PRNGKey(0))
    sdf = model.query(points, code)                     # (B, T), negative inside
    loss = model.collision_loss(points, code)           # (B,)
    self_pen = model.self_collision_loss(body, code, jax.random.PRNGKey(1))
    meshes = model.extract_mesh(body, code)
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from . import assets as A
from . import checkpoint as C
from . import collision as CL
from . import lbs as L
from . import mesh as MS
from . import model as MD

# smplx parameter name -> size per body.
POSE_PARAMS = {
    'global_orient': 3, 'body_pose': 63, 'jaw_pose': 3, 'leye_pose': 3, 'reye_pose': 3,
    'left_hand_pose': 45, 'right_hand_pose': 45,
}


class BodyOutput(NamedTuple):
    vertices: jnp.ndarray    # (B, V, 3)
    joints: jnp.ndarray      # (B, 55, 3)
    full_pose: jnp.ndarray   # (B, 165)


class VolumetricSMPL:
    """An SMPL-X body model with a volumetric signed distance field."""

    def __init__(self, assets: A.Assets, params: dict):
        self.assets = assets
        self.params = params

    @classmethod
    def create(cls, model_path: str, gender: str = 'neutral', weights: str = 'released',
               model_type: str = 'smplx', num_betas: int = 16,
               num_expression_coeffs: int = 10, flat_hand_mean: bool = False):
        """Build the body from the SMPL-X model file and load the volume's weights.

        `model_path` is the `.npz`, a directory holding `SMPLX_<GENDER>.npz`, or its parent.
        `weights` is `'released'` (downloaded on first use), a PyTorch `.ckpt`, or an
        `.npz` written by `training.train`. The released checkpoints were trained with
        16 betas and `flat_hand_mean=False`.
        """
        if model_type != 'smplx':
            raise NotImplementedError(f'only SMPL-X is supported, not {model_type!r}')
        assets = A.build(model_path, gender, num_betas, num_expression_coeffs,
                         flat_hand_mean)
        return cls(assets, C.load_weights(weights, gender))

    @property
    def faces(self) -> np.ndarray:
        return np.asarray(self.assets.faces)

    @property
    def num_parts(self) -> int:
        return self.assets.num_parts

    # --- body --------------------------------------------------------------------

    def forward(self, betas=None, expression=None, transl=None, **pose) -> BodyOutput:
        """Pose the body. Every parameter is optional and defaults to zero.

        Accepts smplx's names: `betas` (B, num_betas), `expression` (B, 10), `transl`
        (B, 3), and axis-angle `global_orient`, `body_pose`, `jaw_pose`, `leye_pose`,
        `reye_pose`, `left_hand_pose`, `right_hand_pose` (full, not PCA).
        """
        unknown = set(pose) - set(POSE_PARAMS)
        if unknown:
            raise TypeError(f'unknown pose parameters: {sorted(unknown)}')
        given = [x for x in (betas, expression, transl, *pose.values()) if x is not None]
        B = given[0].shape[0] if given else 1

        def zeros(n):
            return jnp.zeros((B, n), jnp.float32)

        params = {name: jnp.asarray(pose[name]).reshape(B, n) if pose.get(name) is not None
                  else zeros(n) for name, n in POSE_PARAMS.items()}
        params['betas'] = zeros(self.assets.shapedirs.shape[-1]) if betas is None else betas
        params['expression'] = expression
        params['transl'] = transl
        return BodyOutput(*L.forward(params, self.assets))

    def encode(self, body: BodyOutput, key=None) -> dict:
        """Part transforms, part boxes and the per-part latent codes of a posed body.

        The encoder sees points sampled from the body's surface, so the code depends
        (slightly) on `key`.
        """
        key = jax.random.PRNGKey(0) if key is None else key
        return MD.encode_body(self.params, body.vertices, body.joints, body.full_pose,
                              self.assets, key)

    # --- queries -----------------------------------------------------------------

    def query(self, points, code: dict):
        """(B, T, 3) -> (B, T) signed distance, negative inside."""
        return MD.query_sdf(self.params, points, code, self.assets)

    def query_occupancy(self, points, code: dict):
        """(B, T, 3) -> (B, T) occupancy probability."""
        return MD.query_occupancy(self.params, points, code, self.assets)

    def part_labels(self, points, code: dict):
        """(B, T, 3) -> (B, T) index of the part closest to each point."""
        return jnp.argmin(MD.fused_part_sdf(self.params, points, code, self.assets), axis=1)

    # --- collisions --------------------------------------------------------------

    def collision_loss(self, points, code: dict, return_mask: bool = False):
        """(B,) summed penetration depth of the points into the body."""
        sdf = self.query(points, code)
        loss = CL.collision_sum(sdf)
        return (loss, CL.penetration(sdf) > 0) if return_mask else loss

    def collision_loss_mean(self, points, code: dict, return_mask: bool = False):
        """(B,) mean penetration depth over the penetrating points."""
        sdf = self.query(points, code)
        loss = CL.collision_mean(sdf)
        return (loss, CL.penetration(sdf) > 0) if return_mask else loss

    def collision_loss_gmof(self, points, code: dict, rho: float = 5e-2,
                            return_mask: bool = False):
        """(B,) Geman-McClure-robustified penetration, mean over penetrating points."""
        sdf = self.query(points, code)
        loss = CL.collision_gmof(sdf, rho)
        return (loss, CL.penetration(sdf) > 0) if return_mask else loss

    def self_collision_loss(self, body: BodyOutput, code: dict, key=None,
                            n_points_uniform: int = 300, max_pairs: int | None = None):
        """(B,) penalty on space claimed by two non-adjacent body parts.

        Samples `n_points_uniform` points in each overlapping pair of part boxes. See
        `collision.self_collision_loss`.
        """
        key = jax.random.PRNGKey(0) if key is None else key
        return CL.self_collision_loss(self.params, body.vertices, code, self.assets, key,
                                      n_points_uniform=n_points_uniform,
                                      max_pairs=max_pairs,
                                      level_set=A.LEVEL_SET)

    # --- meshes ------------------------------------------------------------------

    def extract_mesh(self, body: BodyOutput, code: dict, voxel_mm: float = 4.0,
                     field: str = 'logit'):
        """One part-coloured `trimesh.Trimesh` per body, or None where no surface is found.

        `field='logit'` (the default) runs marching cubes on the decoder's
        pre-sigmoid output, which places the surface far more accurately than the
        saturated occupancy; see `mesh.extract_mesh`.
        """
        import trimesh

        a = self.assets
        chunk = 16_384
        query = jax.jit(lambda w, pts, c: MD.query_occupancy(w, pts, c, a))
        labels = jax.jit(lambda w, pts, c: jnp.argmin(MD.fused_part_sdf(w, pts, c, a), 1))
        meshes = []
        for b in range(body.vertices.shape[0]):
            code_b = {k: v[b:b + 1] for k, v in code.items()}
            v = np.asarray(body.vertices[b])
            verts, faces, _ = MS.extract_mesh(lambda p: query(self.params, p, code_b),
                                              v.min(0), v.max(0), voxel_mm, field)
            if verts is None:
                meshes.append(None)
                continue
            pts = verts.astype(np.float32)[None]
            part = np.concatenate([
                np.asarray(labels(self.params, jnp.asarray(pts[:, i:i + chunk]), code_b))[0]
                for i in range(0, pts.shape[1], chunk)])
            meshes.append(trimesh.Trimesh(verts, faces, vertex_colors=MS.part_colors(part),
                                          process=False))
        return meshes
