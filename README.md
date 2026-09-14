# VolumetricSMPL in JAX

> **Unofficial.** This is a fork of [markomih/VolumetricSMPL](https://github.com/markomih/VolumetricSMPL)
> (ICCV 2025). This branch ports the VolumetricSMPL package to JAX, with no PyTorch dependency, and
> adds a JAX implementation of the paper's training and evaluation pipeline, which was never
> released. The PyTorch implementation of the pipeline is on the
> [`main`](https://github.com/HunorLaczko/VolumetricSMPL/tree/main) branch. It is not affiliated with
> the authors.

[![Paper](https://img.shields.io/badge/Paper-ICCV%202025%20Highlight-brightgreen)](https://arxiv.org/abs/2506.23236) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Results

SMPL-X neutral, AMASS PosePrior validation set (316 bodies), mean ± sd over 5 sampling seeds, all
scored by this branch's evaluation harness under the same protocol.

| Model | IoU mean | IoU surface | IoU uniform |
|---|---:|---:|---:|
| Released checkpoint | 91.51 ± 0.03 | 88.49 ± 0.05 | 94.53 ± 0.02 |
| **Trained from scratch on this branch** | **93.47 ± 0.03** | **91.11 ± 0.05** | **95.83 ± 0.04** |
| Trained with the PyTorch implementation (`main`) | 93.63 ± 0.02 | 91.34 ± 0.03 | 95.92 ± 0.02 |
| Paper, Table 1 | 94.67 | 94.25 | 95.10 |

**The target is the released checkpoint, not the paper's table.** The paper does not state its
evaluation protocol, and no single protocol reproduces all of its numbers (see
[FINDINGS.md](FINDINGS.md#4-evaluation-protocol)). A trained model passes if its IoU mean is no more
than 0.3 below the released checkpoint's. The model trained here passes by +1.96. Training takes
16.3 h on one 16 GB GPU, 1.23× faster than the PyTorch implementation.

The recovered training recipe, the design decisions, the checks against the PyTorch implementation
and the bugs found along the way are in **[FINDINGS.md](FINDINGS.md)**.

## The package

`VolumetricSMPL` attaches a learned signed distance field to an SMPL-X body. This port reads the
SMPL-X model file and the released checkpoints directly, so it needs neither PyTorch nor `smplx`.
Only SMPL-X is supported.

```bash
pip install "jax[cuda12]"
pip install git+https://github.com/HunorLaczko/VolumetricSMPL.git@jax
```

You need the SMPL-X model file (see [Data](#data)). The released weights are downloaded on first use.

```python
import jax
from VolumetricSMPL import VolumetricSMPL, winding_numbers

model = VolumetricSMPL.create('data/body_models', gender='neutral')

# smplx's parameter names; anything left out is zero
body = model.forward(betas=betas, global_orient=global_orient, body_pose=body_pose, transl=transl)
code = model.encode(body, jax.random.PRNGKey(0))

sdf = model.query(points, code)                      # (B, T), negative inside
occupancy = model.query_occupancy(points, code)      # (B, T)
penetration = model.collision_loss(scan_points, code)                 # (B,)
self_intersection = model.self_collision_loss(body, code, jax.random.PRNGKey(1))  # (B,)
meshes = model.extract_mesh(body, code)              # one part-coloured trimesh.Trimesh per body
```

Every method is a pure function of its inputs, so losses can be differentiated with respect to the
body parameters:

```python
def objective(body_pose):
    body = model.forward(betas=betas, body_pose=body_pose)
    return model.collision_loss(scan_points, model.encode(body, key)).sum()

grad = jax.jit(jax.grad(objective))(body_pose)
```

| Method | Returns |
|---|---|
| `VolumetricSMPL.create(model_path, gender, weights)` | The model. `weights` is `'released'`, a PyTorch `.ckpt`, or an `.npz` from `training.train` |
| `forward(**params)` | `BodyOutput(vertices, joints, full_pose)` |
| `encode(body, key)` | Part transforms, part boxes and latent codes; the input to every query |
| `query`, `query_occupancy` | Signed distance and occupancy at query points |
| `part_labels` | Index of the nearest body part per point |
| `collision_loss`, `collision_loss_mean`, `collision_loss_gmof` | Penetration of external points into the body |
| `self_collision_loss` | Penalty on space claimed by two non-adjacent parts |
| `extract_mesh(body, code, voxel_mm, field)` | Marching-cubes meshes of the field |
| `winding_numbers(points, triangles)` | Generalized winding numbers against a triangle soup |

Importing the package disables XLA's Triton GEMM fusion, which computes wrong values for this model
on GPU, and sets float32 matmul precision to `highest`. Import it before running any other JAX code.
See [FINDINGS.md](FINDINGS.md#8-pitfalls-and-upstream-issues).

## Repository layout

| Path | Contents |
|---|---|
| `VolumetricSMPL/` | The package |
| `training/` | Data pipeline, training loop, evaluation harness, audit, report and tests |
| `docker/`, `compose.yaml` | Containerised environment (JAX 0.10.2, CUDA 12) |
| `FINDINGS.md` | What was measured, decided and found, with the numbers |

| Module | Responsibility |
|---|---|
| `VolumetricSMPL/volumetric_smpl.py` | The public model |
| `VolumetricSMPL/assets.py`, `partition.py` | SMPL-X buffers from the model file; the 15-part body decomposition |
| `VolumetricSMPL/checkpoint.py` | Reads PyTorch checkpoints without PyTorch; downloads the released ones |
| `VolumetricSMPL/lbs.py`, `modules.py`, `model.py` | SMPL-X skinning, the PointNet encoder and NBW decoder, the fused field and its losses |
| `VolumetricSMPL/geometry.py`, `collision.py`, `mesh.py`, `winding_numbers.py` | Distances and sampling, collision terms, mesh extraction, winding numbers |
| `training/amass.py`, `cache.py` | AMASS npz tree → pose cache with a content-addressed manifest; split definitions |
| `training/sampling.py`, `occupancy.py` | Query-point protocol; ray-stabbing parity occupancy and a trimesh reference |
| `training/train.py`, `init.py` | Jitted training step with on-device data generation; the original initialisation |
| `training/evaluate.py` | IoU uniform/surface/mean, MSE SDF, MSE \|SDF\|; the reference gate |
| `training/audit.py` | Checks that the evaluation itself can be trusted |
| `training/report.py`, `meshmetrics.py`, `viz.py` | Surface metrics with an exact distance; 3D panels |
| `training/test_parity.py`, `test_api.py`, `test_golden.py` | Geometry against analytic and independent answers; package invariants; pinned-batch regression test |

## Setup

### Requirements

- An NVIDIA GPU with a recent driver, Docker, and the NVIDIA Container Toolkit (on Windows, Docker
  Desktop with the WSL 2 backend).
- Optional: a [Weights & Biases](https://wandb.ai) account for `--wandb`. Copy `.env.example` to `.env`
  and set `WANDB_API_KEY`.

### Data

Both datasets are licence-gated, so they are not included. Register and download them yourself:

- **SMPL-X body models** from [smpl-x.is.tue.mpg.de](https://smpl-x.is.tue.mpg.de), version 1.1. Only
  the `.npz` files are needed.
- **AMASS**, *SMPL-X G* flavour, from [amass.is.tue.mpg.de](https://amass.is.tue.mpg.de). Training and
  evaluation need BMLmovi, DFaust and PosePrior. BMLrub is only needed for the `holdout_bmlrub` split.

```
data/
├── body_models/smplx/SMPLX_NEUTRAL.npz
└── extracted/
    ├── BMLmovi/<subject>/*_stageii.npz
    ├── DFaust/...
    ├── PosePrior/...
    └── BMLrub/...            # optional
```

### Environment

```bash
docker compose build jax
docker compose run --rm jax python docker/verify_env.py
```

The image contains only dependencies. The repository, including `data/`, is bind-mounted at
`/workspace`, so code changes never need a rebuild. `verify_env.py` checks the GPU, the compiler
setting above and the geometry ops against analytic answers. Exact resolved versions are in
[`docker/requirements.lock.txt`](docker/requirements.lock.txt).

## Usage

All commands run inside the container: prefix each with `docker compose run --rm jax`.

**Build the pose caches.** This is a one-off. Splits are declared in `training/cache.py`, and each
build checks the expected body count:

```bash
python -m training.cache --split val      # 316 bodies
python -m training.cache --split train    # 256,045 bodies
```

**Evaluate the released checkpoint.** This establishes the reference and needs no training:

```bash
python -m training.evaluate
```

**Train:**

```bash
python -m training.train --smoke                                  # 200 steps, to check the loop runs
python -m training.train --wandb --out-dir runs/jax_smplx_neutral # the full 15 epochs
```

Checkpoints are written every 5,000 steps as `.npz`, and `metrics.csv` in the output directory logs
every 200 steps.

**Evaluate, audit and report on a trained model:**

```bash
python -m training.evaluate --weights runs/jax_smplx_neutral/ckpts/last.npz
python -m training.audit    --weights runs/jax_smplx_neutral/ckpts/last.npz
python -m training.report   --weights runs/jax_smplx_neutral/ckpts/last.npz [--no-wandb]
```

Sampling is stochastic, so compare seed-averaged results. `evaluate` runs 5 seeds by default.

**Tests:**

```bash
python -m training.test_parity --bodies 4   # distances, occupancy vs trimesh, distance scale, extraction
python -m training.test_api                 # package invariants: queries, collisions, winding numbers, meshes
python -m training.test_golden              # pinned batch and losses; fails on any pipeline drift
```

### Notes

- **The ragged training budget.** The jitted step evaluates at most `--ragged-pad` (2,048) in-box
  points per body part. On the full training split that overflowed on ~9% of steps, dropping ~0.6%
  of the occupancy signal. A larger budget is exact but slower. Overflow is logged as
  `ragged_overflow`.
- **Do not run other GPU work while training.** On Windows (WDDM), two heavy GPU processes can trigger
  a driver reset that kills both.
- In Git Bash, prefix commands with `MSYS_NO_PATHCONV=1`, or container paths get rewritten.

## Citation

The model and the released checkpoints are the work of the original authors:

```bibtex
@inproceedings{ICCV25:VolumetricSMPL,
   title={{VolumetricSMPL}: A Neural Volumetric Body Model for Efficient Interactions, Contacts, and Collisions},
   author={Mihajlovic, Marko and Zhang, Siwei and Li, Gen and Zhao, Kaifeng and M{\"u}ller, Lea and Tang, Siyu},
   booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
   year={2025}
}
```

- **Paper**: [arXiv](https://arxiv.org/abs/2506.23236)
- **Project page**: [markomih.github.io/VolumetricSMPL](https://markomih.github.io/VolumetricSMPL)
- **Original package**: [markomih/VolumetricSMPL](https://github.com/markomih/VolumetricSMPL)

**Authors:** [Marko Mihajlovic](https://markomih.github.io/), [Siwei Zhang](https://sanweiliti.github.io/),
[Gen Li](https://vlg.inf.ethz.ch/team/Gen-Li.html), [Kaifeng Zhao](https://zkf1997.github.io/),
[Lea Müller](https://muelea.github.io/) and [Siyu Tang](https://vlg.inf.ethz.ch/team/Prof-Dr-Siyu-Tang.html).

## License

MIT, as the original. See [LICENSE](LICENSE).
