"""The body decomposition: which faces and vertices belong to each part.

A numpy port of `Partitioner` from the original package, SMPL-X only. Every vertex takes the
joint with its largest skinning weight; the jaw folds into its parent, each hand into its
wrist, and the eyes are excluded. Seven small parts are then merged into their parents,
leaving 15. A face takes the majority label of its three vertices. Each part gets a *tight*
face set (its own faces) and an *extended* one (its own plus its parent's and children's),
both padded with -1 to a common length.

The decomposition is static per body model, so it is computed once at load time.
"""
from __future__ import annotations

import numpy as np

# Merged into their parent, in this order. The order matters: each merge also rewrites the
# kinematic table that later merges read.
MERGE_BODY_PARTS = sorted([
    15,      # face into neck
    10,      # left toes into left foot
    11,      # right toes into right foot
    3,       # mid stomach into lower stomach
    13, 14,  # left and right shoulder blade
    9,       # upper body with upper stomach
], reverse=True)

# Part pairs never penalised against each other by the self-intersection loss: they touch
# by construction.
SELFPEN_DISABLE_PARTS = [
    (1, 2),                      # L_Hip - R_Hip
    (1, 3), (2, 3),              # hips - Spine1
    (3, 9),                      # Spine1 - Spine3
    (9, 15), (9, 16), (9, 17),   # Spine3 - Head, shoulders
    (12, 6), (12, 16), (12, 17), # Neck - Spine2, shoulders
    (13, 6), (13, 12),           # L_Collar - Spine2, Neck
    (14, 6), (14, 12),           # R_Collar - Spine2, Neck
    (0, 6), (0, 9),              # pelvis - Spine2, Spine3
    (0, 16), (0, 17),            # pelvis - shoulders
    (6, 16), (6, 17),            # Spine2 - shoulders
]

N_JOINT_PARTS = 22          # SMPL-X body joints that can own a part
EYE_LABEL = 100             # vertices that belong to no part
FACE_PAD = -1


def _face_labels(max_part: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Majority vertex label per face; a three-way tie goes to the smallest label.

    The same rule as the original's `np.bincount(labels).argmax()` over each face's
    vertices, vectorised.
    """
    l0, l1, l2 = (max_part[faces[:, i]] for i in range(3))
    return np.where((l0 == l1) | (l0 == l2), l0,
                    np.where(l1 == l2, l1, np.minimum(np.minimum(l0, l1), l2)))


def _pad(rows: list[np.ndarray], fill) -> np.ndarray:
    width = max(r.shape[0] for r in rows)
    out = []
    for r in rows:
        fill_value = r[0] if fill is None else fill
        padded = np.full((width,) + r.shape[1:], fill_value, dtype=np.int32)
        padded[:r.shape[0]] = r
        out.append(padded)
    return np.stack(out)


def partition(lbs_weights: np.ndarray, faces: np.ndarray, parents: np.ndarray) -> dict:
    """Decompose an SMPL-X body into parts.

    Returns:
        tight_faces       (K, Ft, 3) int32, -1 padded
        extended_faces    (K, Fe, 3) int32, -1 padded
        tight_vert_selector (K, Vt) int32, padded with each row's first vertex
        joint_mapper      (22,) bool, False for merged joints
        selfpen_disable_mat (K, K) bool, False for pairs the self-intersection loss skips
    """
    faces = np.asarray(faces, dtype=np.int64)
    kintree = np.asarray(parents, dtype=np.int64).copy()
    max_part = np.asarray(lbs_weights).argmax(axis=1)

    max_part[max_part == 22] = kintree[22]                           # jaw
    max_part[(max_part >= 25) & (max_part < 40)] = kintree[25 + 3]   # left hand
    max_part[(max_part >= 40) & (max_part < 55)] = kintree[40 + 3]   # right hand
    max_part[max_part >= 22] = EYE_LABEL

    kintree = kintree[:N_JOINT_PARTS]
    valid_labels = list(range(N_JOINT_PARTS))
    for ind in MERGE_BODY_PARTS:
        max_part[max_part == ind] = kintree[ind]
        valid_labels[ind] = kintree[ind]
        kintree[kintree == ind] = kintree[ind]   # children take the grandparent's id

    face_labels = _face_labels(max_part, faces)

    tight, extended, selectors = [], [], []
    for part in range(N_JOINT_PARTS):
        if part in MERGE_BODY_PARTS:
            continue
        label = valid_labels[part]
        own = face_labels == label
        ext = own.copy()
        parent = kintree[part]
        if parent != -1:
            ext |= face_labels == valid_labels[parent]
        for child in (i for i in range(N_JOINT_PARTS) if kintree[i] == part):
            ext |= face_labels == valid_labels[child]
        tight.append(faces[own])
        extended.append(faces[ext])
        selectors.append(np.nonzero(max_part == label)[0])

    joint_mapper = np.ones(N_JOINT_PARTS, dtype=bool)
    joint_mapper[MERGE_BODY_PARTS] = False

    return dict(
        tight_faces=_pad(tight, FACE_PAD),
        extended_faces=_pad(extended, FACE_PAD),
        tight_vert_selector=_pad(selectors, None),
        joint_mapper=joint_mapper,
        selfpen_disable_mat=selfpen_disable_mat(parents, joint_mapper),
    )


def selfpen_disable_mat(parents: np.ndarray, joint_mapper: np.ndarray,
                        n_parts: int = 24) -> np.ndarray:
    """(K, K) True where two parts may be penalised for intersecting.

    Parents, children and the hand-listed `SELFPEN_DISABLE_PARTS` are switched off, then
    the matrix is restricted to the parts that survive merging.
    """
    kintree = np.asarray(parents)[:n_parts]
    mat = np.ones((n_parts, n_parts), dtype=bool)
    for part in range(n_parts):
        children = [i for i in range(n_parts) if kintree[i] == part]
        mat[part, children] = False
        for child in children:
            mat[child, kintree[child]] = False
    for i, j in SELFPEN_DISABLE_PARTS:
        mat[i, j] = mat[j, i] = False
    keep = np.nonzero(joint_mapper)[0]
    return mat[np.ix_(keep, keep)]
