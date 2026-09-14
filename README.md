# VolumetricSMPL — training reimplementation

> **Unofficial.** This is a fork of [markomih/VolumetricSMPL](https://github.com/markomih/VolumetricSMPL),
> which ships the VolumetricSMPL inference package (ICCV 2025). The paper's training and evaluation
> code was never released. This fork adds an independent reimplementation of both, and trains the
> SMPL-X neutral model from scratch. It is not affiliated with the authors. The original README
> follows [below](#volumetricsmpl-package).

[![Paper](https://img.shields.io/badge/Paper-ICCV%202025%20Highlight-brightgreen)](https://arxiv.org/abs/2506.23236) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Results

SMPL-X neutral, AMASS PosePrior validation set (316 bodies), mean ± sd over 5 sampling seeds. The
released checkpoint and the model trained here are scored by the same harness, on the same bodies,
under the same protocol.

| Metric | Released checkpoint | Trained here | Δ | Paper, Table 1 |
|---|---:|---:|---:|---:|
| IoU mean | 91.52 | **93.68 ± 0.04** | +2.16 | 94.67 |
| IoU surface | 88.49 | **91.39 ± 0.06** | +2.90 | 94.25 |
| IoU uniform | 94.55 | **95.97 ± 0.02** | +1.42 | 95.10 |
| MSE SDF | 5.75e-5 | **3.85e-5** | −1.9e-5 | 3.7e-5 |
| MSE \|SDF\| | 5.43e-5 | **3.69e-5** | −1.7e-5 | 3.5e-5 |

**The target is the released checkpoint, not the paper's table.** The paper does not state its
evaluation protocol, and no single protocol reproduces all five published numbers (see
[FINDINGS.md](FINDINGS.md#4-evaluation-protocol)). The paper column is context only.

The trained model passes the gate (IoU mean no more than 0.3 below the released checkpoint) with a
margin of +2.16. That margin survives controls for memorisation, validation-set overfitting, model
selection, metric bugs and training-set size. The full evidence trail, the recovered training recipe,
the design decisions and the upstream bugs found along the way are in **[FINDINGS.md](FINDINGS.md)**.

The [`jax`](../../tree/jax) branch ports the package and this pipeline to JAX, with no PyTorch dependency.

## What this fork adds

| Path | Contents |
|---|---|
| `training/` | Data pipeline, training loop, evaluation harness, audit, report and tests |
| `docker/`, `compose.yaml` | Containerised environment (torch 2.8 + CUDA 12.9, PyTorch3D, kaolin) |
| `FINDINGS.md` | What was measured, decided and found, with the numbers |

The `VolumetricSMPL/` package itself is unmodified. The training code imports it and wraps it
rather than editing it, so any difference from the original can be attributed to `training/`.

| Module | Responsibility |
|---|---|
| `amass.py` | AMASS npz tree → SMPL-X parameter tensors, via AMASS's named fields |
| `cache.py` | Pose cache with a content-addressed manifest; split definitions |
| `sampling.py` | Query-point protocol (per-part uniform + tight surface + jitter) |
| `occupancy.py` | Ray-stabbing parity occupancy on GPU (kaolin) and a CPU reference (trimesh) |
| `geometry.py` | Why the ground-truth distance keeps PyTorch3D's default guard |
| `ragged.py` | Exact in-box compaction of the training forward pass (1.66x faster) |
| `module.py`, `train.py` | Lightning module with on-GPU data generation; training CLI |
| `evaluate.py` | IoU uniform/surface/mean, MSE SDF, MSE \|SDF\|; the reference gate |
| `audit.py` | Checks that the evaluation itself can be trusted |
| `report.py`, `meshmetrics.py`, `viz.py` | Model comparison: generalisation, surface metrics, 3D panels |
| `sweep.py` | One-knob-at-a-time sweep of the evaluation protocol |
| `test_parity.py`, `test_golden.py` | Geometry differential tests; pinned-batch regression test |

## Setup

### Requirements

- An NVIDIA GPU with a recent driver, Docker, and the NVIDIA Container Toolkit (on Windows, Docker
  Desktop with the WSL 2 backend). Training peaks at ~3.5 GiB of VRAM.
- The image compiles PyTorch3D for Blackwell GPUs (`TORCH_CUDA_ARCH_LIST=12.0+PTX`) by default. For
  other architectures, pass your own list:
  `docker compose build --build-arg TORCH_CUDA_ARCH_LIST="8.6;8.9" train`.
- Optional: a [Weights & Biases](https://wandb.ai) account for `--wandb`. Copy `.env.example` to `.env`
  and set `WANDB_API_KEY`.

### Data

Both datasets are licence-gated, so they are not included. Register and download them yourself:

- **SMPL-X body models** from [smpl-x.is.tue.mpg.de](https://smpl-x.is.tue.mpg.de). Only the `.npz`
  files are needed. The `.pkl` files require `chumpy`, which is broken on modern NumPy.
- **AMASS**, *SMPL-X G* flavour, from [amass.is.tue.mpg.de](https://amass.is.tue.mpg.de). You need
  the subsets BMLmovi, DFaust and PosePrior, plus BMLrub for the generalisation hold-out.

```
data/
├── body_models/smplx/SMPLX_NEUTRAL.npz
└── extracted/
    ├── BMLmovi/<subject>/*_stageii.npz
    ├── DFaust/...
    ├── PosePrior/...
    └── BMLrub/...            # optional, report hold-out only
```

The released checkpoint is downloaded automatically on first use.

### Environment

```bash
docker compose build train
docker compose run --rm train python docker/verify_env.py
```

The image contains only dependencies. The repository, including `data/`, is bind-mounted at
`/workspace`, so code changes never need a rebuild. `verify_env.py` checks every geometry op the
pipeline uses against analytic answers. Exact resolved versions are in
[`docker/requirements.lock.txt`](docker/requirements.lock.txt).

## Usage

All commands run inside the container: prefix each with `docker compose run --rm train`.

**Evaluate the released checkpoint.** This establishes the reference and needs no training:

```bash
python -m training.evaluate
```

**Build the pose caches.** This is a one-off. Splits are declared in `training/cache.py`, and each build
checks the expected body count:

```bash
python -m training.cache --split val      # 316 bodies
python -m training.cache --split train    # 256,045 bodies
python -m training.cache --split smoke    # DFaust only, for a quick loop check
```

**Train:**

```bash
# 200 steps, to check that the loop learns
python -m training.train --smoke --train-split smoke --out-dir runs/smoke

# the full 15-epoch run (~22-24 h on an RTX 5060 Ti)
python -m training.train --wandb --out-dir runs/smplx_neutral

# resume; --wandb-id continues the same W&B run
python -m training.train --wandb --out-dir runs/smplx_neutral \
    --resume runs/smplx_neutral/ckpts/last.ckpt --wandb-id <run-id>
```

Checkpoints are written every 5,000 steps. Every resume starts a new `csv/version_N/` directory, so
read all of them together to get the full history.

**Evaluate, audit and compare a trained model:**

```bash
python -m training.evaluate --ckpt runs/smplx_neutral/ckpts/last.ckpt
python -m training.audit    --ckpt runs/smplx_neutral/ckpts/last.ckpt
python -m training.report   --ckpt runs/smplx_neutral/ckpts/last.ckpt [--no-wandb]
```

Sampling is stochastic. Compare seed-averaged results (`evaluate --seed 0` through `--seed 4`, or
`report`, which runs all five), never single seeds. The report builds its hold-out caches on first
use.

**Tests:**

```bash
python -m training.test_parity --bodies 4   # occupancy vs trimesh, distance scale, surface distance, extraction
python -m training.test_golden              # pinned batch and losses; fails on any pipeline drift
```

### Notes

- **Do not run other GPU work while training.** On Windows (WDDM), two heavy CUDA processes can trigger a
  driver reset that kills both. If it happens, recover with `--resume`.
- In Git Bash, prefix commands with `MSYS_NO_PATHCONV=1`, or container paths get rewritten.
- The CUDA base image prints a licence banner to stdout. Account for it when redirecting output to a
  file.

---

# VolumetricSMPL package

*The original README of [markomih/VolumetricSMPL](https://github.com/markomih/VolumetricSMPL):*

[![PyPI version](https://badge.fury.io/py/VolumetricSMPL.svg)](https://pypi.org/project/VolumetricSMPL/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Paper](https://img.shields.io/badge/Paper-ICCV%202025%20Highlight-brightgreen)](https://arxiv.org/abs/2506.23236) [![Video](https://img.shields.io/badge/Video-YouTube-red)](https://youtu.be/XmY_W_F58cA)

<div align="center">
  <img src="https://markomih.github.io/VolumetricSMPL/assets/teaser.jpeg" alt="VolumetricSMPL Teaser" width="600"/>
</div>

## 🌟 TL;DR

**VolumetricSMPL** is a lightweight, plug-and-play extension for SMPL(-X) models that adds volumetric functionality via Signed Distance Fields (SDFs). With minimal integration—just a single line of code—users gain access to fast and differentiable SDF queries, collision detection, and self-intersection resolution.

## ✨ Key Features

- 🔌 **Single-line integration** with existing SMPL models
- ⚡ **Fast and differentiable** SDF queries
- 🛡️ **Built-in collision detection** and self-intersection resolution
- 🔄 **Compatible** with SMPL, SMPLH, and SMPL-X
- 🎯 **Efficient interaction modeling** for perception and reconstruction tasks

## 📚 Paper & Resources

- **📄 Paper**: [arXiv](https://arxiv.org/abs/2506.23236)
- **🎥 Video**: [YouTube](https://youtu.be/XmY_W_F58cA)
- **🌐 Project Page**: [markomih.github.io/VolumetricSMPL](https://markomih.github.io/VolumetricSMPL)
- **📦 Applications**: [VolumetricSMPL_applications](https://github.com/markomih/VolumetricSMPL_applications)

## 🚀 Quick Start

### Installation

Ensure that PyTorch and PyTorch3D are installed with GPU support. Then install VolumetricSMPL:

```bash
pip install VolumetricSMPL
```

### Basic Usage

Extend an existing [SMPL-X](https://github.com/vchoutas/smplx) model with volumetric functionalities:

```python
import smplx
from VolumetricSMPL import attach_volume

# Create a SMPL body and extend it with volumetric functionalities
# Supports SMPL, SMPLH, and SMPL-X
model = smplx.create(**smpl_parameters)
attach_volume(model)

# Forward pass
smpl_output = model(**smpl_data)  

# Ensure valid SMPL variables (pose parameters, joints, and vertices)
assert model.joint_mapper is None, "VolumetricSMPL requires valid SMPL joints as input."

# Access volumetric functionalities
model.volume.query(scan_point_cloud)                 # Query SDF for given points
model.volume.selfpen_loss(smpl_output)               # Compute self-intersection loss
model.volume.collision_loss(smpl_output, scan_point_cloud)  # Compute collisions with external geometries
```

## 📖 Detailed Usage

VolumetricSMPL extends the interface of the [SMPL-X package](https://github.com/vchoutas/smplx) by attaching a volumetric representation to the body model. This allows for:

- **Querying signed distance fields** for arbitrary points
- **Accessing collision loss terms** for optimization
- **Self-intersection detection** and resolution
- **Efficient interaction modeling** with 3D geometries

For further examples and use cases, check out our [Applications repository](https://github.com/markomih/VolumetricSMPL_applications). 


## 📦 Pretrained Models

Pretrained models are automatically fetched and loaded when you first use VolumetricSMPL. They can also be found in the `dev` branch inside the `./models` directory.

## 🔧 Requirements

- Python 3.7+
- PyTorch 
- PyTorch3D
- SMPL-X

## 📄 Citation

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{ICCV25:VolumetricSMPL,
   title={{VolumetricSMPL}: A Neural Volumetric Body Model for Efficient Interactions, Contacts, and Collisions},
   author={Mihajlovic, Marko and Zhang, Siwei and Li, Gen and Zhao, Kaifeng and M{\"u}ller, Lea and Tang, Siyu},
   booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
   year={2025}
}
```

## 👥 Authors

- [Marko Mihajlovic](https://markomih.github.io/) (ETH Zurich)
- [Siwei Zhang](https://sanweiliti.github.io/) (ETH Zurich)
- [Gen Li](https://vlg.inf.ethz.ch/team/Gen-Li.html) (ETH Zurich)
- [Kaifeng Zhao](https://zkf1997.github.io/) (ETH Zurich)
- [Lea Müller](https://muelea.github.io/) (UC Berkeley)
- [Siyu Tang](https://vlg.inf.ethz.ch/team/Prof-Dr-Siyu-Tang.html) (ETH Zurich)

## Contact

For questions, please contact [Marko Mihajlovic](mailto:markomih@ethz.ch) or raise an issue on [GitHub](https://github.com/markomih/VolumetricSMPL).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
