"""AMASS SMPL-X sequence scanning and pose caching.

Reads AMASS `*_stageii.npz` files and produces flat tensors of SMPL-X parameters,
one row per selected body.
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass

import numpy as np
import torch

# Number of shape coefficients. COAP's smplx config sets num_betas: 16, and the
# AMASS SMPL-X npz files carry exactly 16 — the SMPL config's 10 does not apply here.
NUM_BETAS = 16

# The SMPL-X parameters a body needs, and the AMASS field each is sliced from.
# Deliberately driven by AMASS's *named* fields rather than by slicing the 165-dim
# `poses` vector. COAP slices `poses[66:111]` as the left hand, which is correct for
# SMPL/SMPL+H but wrong for SMPL-X — there the layout is
# root(3) body(63) jaw(3) eye(6) hand(90), so that range straddles jaw, both eyes and
# only part of the left hand. COAP's own config admits it: "if you use smplh or smplx,
# make sure to adjust data loaders accordingly".
PARAM_SPEC = {
    'global_orient': ('root_orient', slice(0, 3)),
    'body_pose': ('pose_body', slice(0, 63)),
    'left_hand_pose': ('pose_hand', slice(0, 45)),
    'right_hand_pose': ('pose_hand', slice(45, 90)),
    'jaw_pose': ('pose_jaw', slice(0, 3)),
    'leye_pose': ('pose_eye', slice(0, 3)),
    'reye_pose': ('pose_eye', slice(3, 6)),
}


@dataclass
class PoseCache:
    """SMPL-X parameters for a set of bodies, plus their provenance."""
    params: dict[str, torch.Tensor]  # each (N, D)
    seq_names: list[str]
    frame_ids: list[int]

    def __len__(self) -> int:
        return len(self.frame_ids)

    def batch(self, lo: int, hi: int, device) -> dict[str, torch.Tensor]:
        return {k: v[lo:hi].to(device) for k, v in self.params.items()}


def is_usable(path: str) -> bool:
    """COAP excludes `*shape.npz` and `*neutral_stagei.npz`.

    We exclude every `*_stagei.npz`: 102 of them across our three subsets carry no
    `poses` field at all, and COAP's narrower filter would have tried to ingest them.
    """
    return not (path.endswith('shape.npz') or path.endswith('stagei.npz'))


def scan(data_root: str, subsets: list[str], stride, num_betas: int = NUM_BETAS,
         offset: int = 0) -> PoseCache:
    """Collect every `stride`-th frame of every usable sequence in `subsets`.

    `stride` is an int applied to all subsets, or a {subset: stride} mapping — the
    original recipe used different rates per subset (DFaust is two orders of magnitude
    smaller than BMLmovi, so a common stride would swamp it).

    `offset` shifts the first selected frame. Needed to build genuinely held-out
    evaluation sets from a subset that is *in* training: with offset 0 and a stride that
    divides the training stride, every selected frame would already have been seen.
    """
    acc: dict[str, list[np.ndarray]] = {k: [] for k in PARAM_SPEC}
    acc['betas'] = []
    seq_names: list[str] = []
    frame_ids: list[int] = []

    for subset in subsets:
        step = stride[subset] if isinstance(stride, dict) else stride
        root = os.path.join(data_root, subset)
        if not os.path.isdir(root):
            raise FileNotFoundError(f'AMASS subset not found: {root}')
        for subject in sorted(glob.glob(os.path.join(root, '*'))):
            if not os.path.isdir(subject):
                continue
            for path in sorted(glob.glob(os.path.join(subject, '*.npz'))):
                if not is_usable(path):
                    continue
                with np.load(path, allow_pickle=True) as d:
                    if 'poses' not in d.files:
                        continue
                    keep = np.arange(offset, d['poses'].shape[0], step)
                    for name, (field, sl) in PARAM_SPEC.items():
                        acc[name].append(d[field][keep, sl].astype(np.float32))
                    betas = d['betas'][:num_betas].reshape(1, -1).astype(np.float32)
                    acc['betas'].append(np.repeat(betas, len(keep), axis=0))

                name = os.path.join(os.path.basename(subject),
                                    os.path.splitext(os.path.basename(path))[0])
                seq_names.extend([name] * len(keep))
                frame_ids.extend(keep.tolist())

    params = {k: torch.from_numpy(np.concatenate(v, axis=0)) for k, v in acc.items()}
    return PoseCache(params=params, seq_names=seq_names, frame_ids=frame_ids)
