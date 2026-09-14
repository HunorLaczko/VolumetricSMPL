"""Pose cache: AMASS npz tree → one file, plus a manifest that pins its contents.

Scanning the AMASS tree costs minutes and opens tens of thousands of npz files. The
cache turns that into a single load, and the manifest hash makes "did the dataset
change?" a cheap, exact question rather than a guess.

  docker compose run --rm jax python -m training.cache --split val
  docker compose run --rm jax python -m training.cache --split train
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os

import jax.numpy as jnp
import numpy as np

from .amass import NUM_BETAS, PoseCache, scan

# Frozen splits. `stride` may be an int or a per-subset mapping.
# The checkpoint pins bodies/batch_size, not the subsets, so several combinations fit.
# `train` is the chosen one; the alternates stay runnable. See FINDINGS.md, "Data".
SPLITS = {
    'val': dict(subsets=['PosePrior'], stride=500, expect=316),
    # Chosen: the paper's stated datasets, 256,045 bodies (+1.79% vs the checkpoint).
    'train': dict(subsets=['BMLmovi', 'DFaust'],
                  stride={'BMLmovi': 5, 'DFaust': 10}, expect=256045),
    # COAP's smplx config verbatim. Only consistent with the checkpoint at batch 24.
    'train_bmlrub': dict(subsets=['DFaust', 'BMLrub'], stride=5, expect=762180),
    # Closest single-subset fit, +0.13%.
    'train_bmlmovi_only': dict(subsets=['BMLmovi'], stride=5, expect=251866),
    # DFaust alone — small enough to iterate on the training loop.
    'smoke': dict(subsets=['DFaust'], stride=5, expect=8296),

    # --- hold-outs ----------------------------------------------------------
    # Generalisation: BMLrub is in no training split above, so subjects and motion
    # style are unseen. It may be in the *released* checkpoint's training set, so a win
    # here is strong evidence and a loss is ambiguous.
    'holdout_bmlrub': dict(subsets=['BMLrub'], stride=2000, expect=None),
    # Memorisation control — read ONLY as a pair. DFaust is trained at stride 10, so
    # frames == 0 (mod 10) were seen and frames == 3 (mod 10) were not. Same sequences,
    # adjacent frames: the only difference is training membership, so the seen - held-out
    # gap measures memorisation directly.
    'holdout_dfaust': dict(subsets=['DFaust'], stride=500, offset=3, expect=None),
    'seen_dfaust': dict(subsets=['DFaust'], stride=500, offset=0, expect=None),
}

CACHE_DIR = 'data/cache'
_PROVENANCE = ('seq_names', 'frame_ids')


def manifest(cache: PoseCache, cfg: dict) -> dict:
    """Content hash over provenance, not array bytes.

    Keyed on the (sequence, frame) list and the config, so it is stable across runs and
    machines but changes the moment the selection does. Float arrays are deliberately
    excluded: they are a deterministic function of this selection plus the AMASS files,
    and hashing 185 MB on every build to learn nothing is wasteful.
    """
    h = hashlib.sha256()
    h.update(json.dumps(cfg, sort_keys=True).encode())
    for name, frame in zip(cache.seq_names, cache.frame_ids):
        h.update(f'{name}:{frame}\n'.encode())
    return {
        'hash': h.hexdigest(),
        'n_bodies': len(cache),
        'n_sequences': len(set(cache.seq_names)),
        'config': cfg,
        'params': {k: list(v.shape) for k, v in sorted(cache.params.items())},
    }


def build(split: str, data_root: str, num_betas: int = NUM_BETAS) -> tuple[PoseCache, dict]:
    spec = SPLITS[split]
    offset = spec.get('offset', 0)
    cfg = dict(split=split, subsets=spec['subsets'], stride=spec['stride'],
               num_betas=num_betas, offset=offset)
    cache = scan(data_root, spec['subsets'], spec['stride'], num_betas=num_betas,
                 offset=offset)
    if spec['expect'] is not None and len(cache) != spec['expect']:
        raise SystemExit(
            f'GATE FAILED: {split} expected {spec["expect"]} bodies, built {len(cache)}')
    return cache, manifest(cache, cfg)


def path_for(name: str, cache_dir: str = CACHE_DIR) -> str:
    """A split name (`val`) or a path to a cache file."""
    return name if name.endswith('.npz') else os.path.join(cache_dir, f'{name}.npz')


def save(cache: PoseCache, man: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, **cache.params, seq_names=np.asarray(cache.seq_names),
             frame_ids=np.asarray(cache.frame_ids, dtype=np.int64))
    with open(path.replace('.npz', '.manifest.json'), 'w') as f:
        json.dump(man, f, indent=2)


def load(path: str) -> tuple[PoseCache, dict]:
    with np.load(path) as d:
        params = {k: d[k] for k in d.files if k not in _PROVENANCE}
        cache = PoseCache(params=params, seq_names=d['seq_names'].tolist(),
                          frame_ids=d['frame_ids'].tolist())
    with open(path.replace('.npz', '.manifest.json')) as f:
        return cache, json.load(f)


def load_params(name: str) -> dict:
    """Just the parameters of a cache, as device arrays."""
    with np.load(path_for(name)) as d:
        return {k: jnp.asarray(d[k]) for k in d.files if k not in _PROVENANCE}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', required=True, choices=sorted(SPLITS))
    ap.add_argument('--data-root', default='data/extracted')
    ap.add_argument('--out-dir', default=CACHE_DIR)
    ap.add_argument('--num-betas', type=int, default=NUM_BETAS)
    args = ap.parse_args()

    path = os.path.join(args.out_dir, f'{args.split}.npz')
    cache, man = build(args.split, args.data_root, args.num_betas)
    save(cache, man, path)
    size_mb = os.path.getsize(path) / 1e6
    print(f"{args.split}: {man['n_bodies']:,} bodies from {man['n_sequences']:,} sequences")
    print(f"  subsets {man['config']['subsets']} stride {man['config']['stride']}")
    print(f"  hash    {man['hash'][:32]}")
    print(f"  wrote   {path}  ({size_mb:.1f} MB)")


if __name__ == '__main__':
    main()
