# Findings

Findings from reimplementing the VolumetricSMPL training pipeline: the recovered training recipe,
the evaluation protocol, design decisions, results, and issues found in upstream code.

**Hardware.** One RTX 5060 Ti (16 GB, Blackwell sm_120), Ryzen 5 5600X, 64 GB RAM, Windows 11 with
Docker Desktop (WSL 2).

**Contents**

1. [Summary](#1-summary)
2. [Recovered training recipe](#2-recovered-training-recipe)
3. [Data](#3-data)
4. [Evaluation protocol](#4-evaluation-protocol)
5. [Design decisions](#5-design-decisions)
6. [Results](#6-results)
7. [Pitfalls and upstream issues](#7-pitfalls-and-upstream-issues)
8. [Methodology notes](#8-methodology-notes)
9. [Open questions](#9-open-questions)
10. [Appendix: terminology](#appendix-terminology)

---

## 1. Summary

The upstream repository ships the inference package only; no training or evaluation code exists
in either of its branches. The training pipeline was rebuilt from the paper, the COAP codebase it
derives from, and the optimizer state left inside the released checkpoints. It trains a SMPL-X
neutral model from scratch in ~24 h on one 16 GB consumer GPU, at 3.5 GiB peak VRAM.

| Metric | Released checkpoint | Trained here (5 seeds) | Δ | Paper, Table 1 |
|---|---:|---:|---:|---:|
| IoU mean | 91.52 | **93.68 ± 0.041** | +2.16 | 94.67 |
| IoU surface | 88.49 | **91.39 ± 0.063** | +2.90 | 94.25 |
| IoU uniform | 94.55 | **95.97 ± 0.024** | +1.42 | 95.10 |
| MSE SDF | 5.75e-5 | **3.849e-5** | −1.9e-5 | 3.7e-5 |
| MSE \|SDF\| | 5.43e-5 | **3.691e-5** | −1.7e-5 | 3.5e-5 |

**What is established:**

| Claim | Evidence |
|---|---|
| The pipeline learns the method | 15 epochs, converged; audit passes |
| It beats the released checkpoint | +2.134 ± 0.014 paired across 5 seeds on identical points |
| Not memorisation | Matched seen/unseen DFaust gap indistinguishable from zero |
| Not a validation-set artefact | +0.80 on 3,938 unseen BMLrub bodies |
| Not model selection | Final checkpoint reported; validation never used to pick a model |
| Not extra training data | A size-matched training set still leads by +1.96 |
| The metrics are correct | Independent IoU agrees to 4 dp; controls score 100 / 0 / 0 |
| The surface is better too | Chamfer 1.33 vs 1.73 mm, lower on 6/6 bodies |

**What is not established:**

- **The paper's Table 1 is not reproduced, and cannot be.** Its protocol is unpublished, and no
  single protocol reproduces all five of its numbers ([§4](#4-evaluation-protocol)).
- **The lead is not uniform.** It is +2.16 on PosePrior and +0.80 on BMLrub.
- **Which subsets the released model was trained on is unknown** ([§3](#which-training-subsets)).

---

## 2. Recovered training recipe

The released Lightning checkpoints (on upstream's `dev` branch) still carry `epoch`, `global_step`,
`optimizer_states` and `lr_schedulers`. Where they contradict the paper, the checkpoint wins,
because the checkpoint is what produced the published weights.

| Item | Paper | Used | Source |
|---|---|---|---|
| Learning rate | 1e-4 | **5e-4** | `optimizer_states[0].param_groups[0].initial_lr` |
| Optimizer | Adam | `Adam(betas=(0.9, 0.999), eps=1e-8, weight_decay=0)` | checkpoint |
| Schedule | — | `CosineAnnealingLR(T_max=450_000, eta_min=1e-5)`, stepped **per step** | checkpoint; `last_epoch == global_step` |
| Iterations | "450k" | **471,644** over 15 epochs | `global_step`, `epoch=14`; 450k is the cosine `T_max` |
| Validation stride | every 300th frame | **every 500th** | 500 gives exactly the paper's stated 316 bodies; 300 gives 517 |
| Batch size | — | 8 | COAP config |
| Query points | 512 per part | 256 uniform + 256 surface per part | paper, COAP |
| Surface noise | N(0, 0.1) | **σ = 0.01 m** | COAP; 10 cm is implausible for near-surface sampling |
| Loss | — | `mse_occ + mse_udf`, equal weight | ships in the package's `query_training()` |
| Architecture | — | K=15 parts, PointNet 128-d, decoder 7×64 with skip at 3, PE multires 2, NBW rank 80, part rank 10 | checkpoint tensor shapes |

Training deliberately overruns the schedule: `T_max` is 450k, but the original ran 471,644 steps,
so the cosine turns back up at the end. This is reproduced as-is. The released male and female
checkpoints are undertrained (they stop at epochs 5–6). The neutral one is fully scheduled.

The checkpoint also pins `steps / epoch = 471,644 / 15 = 31,443`. At batch 8, that means the
original training set held 251,544 bodies.

---

## 3. Data

AMASS, *SMPL-X G* flavour. Frame counts are exact, read from the npz headers:

| Subset | Sequences | Frames | Role | Stride | Bodies |
|---|---:|---:|---|---:|---:|
| BMLmovi | 1,864 | 1,255,584 | train | 5 | 251,866 |
| DFaust | 129 | 41,220 | train | 10 | 4,179 |
| **train** | | | | | **256,045** |
| PosePrior | 35 | 149,939 | validation | 500 | **316** |
| BMLrub | 3,061 | 3,763,367 | generalisation hold-out | 2000 | 3,938 |

- **Poses are read from AMASS's named fields** (`root_orient`, `pose_body`, `pose_hand`, `pose_jaw`,
  `pose_eye`). COAP slices `poses[66:111]` as the left hand. That is correct for SMPL, but wrong
  for SMPL-X, whose layout is root(3) body(63) jaw(3) eye(6) hand(90), so the COAP range straddles
  the jaw, both eyes and part of the left hand.
- **All 102 `*_stagei.npz` files are excluded.** They are shape-estimation artefacts with no `poses`
  field. COAP's filter only excludes `neutral_stagei.npz` and would have tried to ingest them.
- **Body construction** follows COAP's SMPL-X config: `num_betas=16` (AMASS SMPL-X carries exactly
  16), `use_pca=False`, `flat_hand_mean=False`. As in the original, the *neutral* model is trained
  on gender-specific fits.
- **The pose cache is content-addressed.** Its manifest hashes the (sequence, frame) selection and
  config rather than tensor bytes, so "did the data change?" has an exact answer.

### Which training subsets

The checkpoint constrains only `bodies / batch_size`, so several combinations fit:

| Combination | Bodies | vs implied 251,544 | Note |
|---|---:|---:|---|
| **BMLmovi@5 + DFaust@10** | **256,045** | **+1.79%** | chosen: the paper's stated datasets |
| BMLmovi@5 | 251,866 | +0.13% | closest, but drops DFaust, which every source names |
| BMLrub@15 | 252,317 | +0.31% | stride 15 appears in no source |
| DFaust@5 + BMLrub@5, **batch 24** | 762,180 | +1.0% | COAP's SMPL-X config, only if the batch was 24 |

COAP's SMPL-X config (`DFaust + BMLrub` at stride 5) gives 762,180 bodies, 3.03× too many at batch
8. It becomes consistent at batch 24, and a batch jump from 8 to 24 would pair naturally with the
learning-rate change from 1e-4 to 5e-4. That option was rejected on compute (~56 h), not on
evidence. It remains a genuine ambiguity.

**Training-set size does not explain the overshoot.** `train_bmlmovi_only` (+0.13%) was trained
to completion under an otherwise identical config, and all three models were scored in one
process on identical points (seeds 0–4):

| Model | IoU mean | IoU surface | IoU uniform | vs released, paired |
|---|---:|---:|---:|---:|
| released | 91.548 ± 0.027 | 88.540 ± 0.050 | 94.556 ± 0.019 | — |
| BMLmovi + DFaust (256,045) | 93.682 ± 0.021 | 91.398 ± 0.026 | 95.966 ± 0.021 | **+2.134 ± 0.014** |
| BMLmovi only (251,866) | 93.510 ± 0.019 | 91.153 ± 0.027 | 95.867 ± 0.019 | **+1.962 ± 0.020** |

A size-matched set still leads by +1.96. The 4,179 DFaust bodies help measurably (−0.172 ± 0.024
without them, ~7σ), but they account for only 8% of the lead. This experiment varies size while
holding composition 98.4% fixed, so it says nothing about *composition*. `train_bmlrub` would test
that, and it has not been run.

---

## 4. Evaluation protocol

### Why the paper's Table 1 cannot be the target

The paper never states its query-point protocol, and no evaluation code exists upstream. COAP's data
loader is the only written source, and it was written for SMPL. Measured on the released checkpoint
over all 316 validation bodies:

| Protocol | IoU uniform | IoU surface | MSE SDF |
|---|---:|---:|---:|
| Per-part boxes + per-part tight surface, σ=0.01 (COAP) | 94.51 | 88.49 | 5.44e-5 |
| Per-part, σ=0.03 | 94.51 | 93.91 | — |
| Global body box + full-mesh surface, σ=0.01 (ONet-style) | 94.94 | 86.47 | 7.9e-4 |
| Global, σ=0.03 | 94.92 | 93.51 | 8.3e-4 |
| **Paper** | **95.10** | **94.25** | **3.7e-5** |

The three metrics demand incompatible choices:

- **IoU uniform** is matched only by *global* sampling (94.94–95.12 across box paddings 1.0–1.25, so
  this is structural rather than a tuned fit). Per-part sampling stays at 94.51 regardless of σ.
- **MSE SDF** is matched only by *per-part* sampling. Under global sampling it is 21× off, because
  most global points fall outside every part box, where `query` returns an analytic box SDF rather
  than a learned one.
- **IoU surface** needs σ ≈ 0.03 under either scheme, against COAP's documented 0.01.

No single point set reproduces all five published numbers.

### The σ sweep

Surface jitter σ is the one unknown that moves IoU surface materially. Both models were swept. This
was a diagnostic, not a change to the protocol.

| σ | Model | IoU surface | vs paper 94.25 | IoU uniform | vs paper 95.10 |
|---|---|---:|---:|---:|---:|
| 0.01 | released | 88.54 | −5.71 | 94.53 | −0.57 |
| 0.01 | trained | 91.44 | −2.81 | 95.94 | +0.84 |
| 0.02 | released | 92.55 | −1.70 | 94.51 | −0.59 |
| 0.02 | trained | 94.49 | +0.24 | 95.92 | +0.82 |
| 0.03 | released | 93.92 | −0.33 | 94.51 | −0.59 |
| 0.03 | trained | 95.59 | +1.34 | 95.95 | +0.85 |
| 0.04 | released | 94.51 | +0.26 | 94.50 | −0.60 |
| 0.04 | trained | 95.93 | +1.68 | 95.89 | +0.79 |

- **IoU uniform does not depend on σ** (it varies by 0.03 across the sweep). That is expected,
  since σ perturbs only the surface samples, and it is the control that makes the table readable.
- **IoU surface spans 5.97 points on fixed weights** from this one unpublished parameter. The
  released model reaches the paper's 94.25 at σ ≈ 0.036; the trained model reaches it at σ ≈ 0.019.
  The gap between either model and the paper on this metric says nothing about model quality.
- **The like-for-like ordering holds at every σ** (+2.90, +1.94, +1.67, +1.42). The margin narrows
  as the task gets easier, but it never reverses.
- **σ cannot rescue Table 1.** IoU uniform does not move with σ, and neither model sits at 95.10
  under per-part sampling.

### The frozen protocol

COAP's protocol is used verbatim, because it is the only documented one:

| Parameter | Value |
|---|---|
| Validation set | AMASS PosePrior, stride 500 → 316 bodies |
| Points per part | 512 (256 uniform + 256 surface) |
| Uniform points | inside each part's local box, padding 1.125, minus 1e-3 |
| Surface points | per-part tight faces, area-weighted, plus N(0, 0.01) m |
| Ground-truth occupancy | ray-stabbing parity ([§5](#ground-truth-occupancy-ray-stabbing-parity)) |
| Ground-truth distance | the package's `point_mesh_distance` (PyTorch3D, default guard; [§7](#pytorch3d-min_triangle_area-keep-the-default-for-training-targets)) |
| IoU split | on the last axis of `(B, K, 512)` |
| Body | SMPL-X neutral, `num_betas=16`, `use_pca=False`, `flat_hand_mean=False` |

### The reference and the gate

The reference is the released `smplx_neutral` checkpoint under this protocol, averaged over 5 seeds:

| Metric | Mean | Seed sd |
|---|---:|---:|
| IoU mean | **91.52** | 0.022 |
| IoU surface | **88.49** | 0.048 |
| IoU uniform | **94.55** | 0.024 |
| MSE SDF | **5.75e-5** | 1.1e-7 |
| MSE \|SDF\| | **5.43e-5** | — |

**Gate: PASS if seed-averaged IoU mean ≥ 91.22 (reference − 0.3). There is no upper bound.**

- The 0.3 tolerance is ~6× the largest seed-to-seed sd, so it separates a real regression from
  sampling noise.
- The gate is one-sided because the two directions answer different questions. Scoring materially
  *below* the reference is evidence that the reimplementation failed to learn the method. Scoring
  above it is not a failure mode: the released weights are one training run over unknown data.
- The gate was originally two-sided (±0.3). It was made one-sided after the overshoot had been
  tested for artefacts ([§6](#6-results)).
- Sampling is stochastic: uniform points, surface points and the encoder's input cloud are redrawn
  on every call. Always compare seed-averaged runs.
- The trade-off is that this work reproduces the *model*, not the publication's *numbers*. If the
  authors publish their protocol, both checkpoints can be re-scored under it.

### Hypotheses eliminated before accepting this

| Hypothesis for the gap to Table 1 | Verdict |
|---|---|
| Released checkpoint undertrained | ✗ `epoch=14`, fully scheduled |
| kaolin parity ≠ ONet parity | ✗ 0.02–0.04% disagreement with trimesh |
| Ground-truth distance wrong | ✗ within 0.3% of the analytic half-normal |
| `num_betas` 16 vs 10 | ✗ < 0.4 IoU either way |
| `flat_hand_mean` | ✗ < 0.3 IoU |
| Jaw/eye poses set vs zeroed | ✗ < 0.2 IoU |
| Surface faces: extended / mixed / full mesh | ✗ all *worse*; tight faces are best |
| Points outside every part box skew MSE | ✗ 1.39% of points at σ=0.01, no effect |
| Occupancy from `sign(sdf)` vs the occupancy head | ✗ identical to 2 dp |
| Encoder input cloud size (1k → 3k → 10k) | ✗ monotonically worse; 1k is the trained value |
| IoU aggregated per part vs per body | ✗ 0.2 IoU |
| Upstream evaluation code to copy | ✗ none exists in either branch |

### COAP's IoU split

COAP computes "uniform" and "surface" IoU by splitting the flattened points at `T // 2`. The layout
is part-major, so that split separates the first ~7 parts from the last ~8, and both halves mix
uniform and surface points. Under COAP's split the two numbers come out nearly equal (~91 each),
which is not the shape of the published numbers. The split used here reshapes to `(B, K, 512)` and
splits the last axis. `evaluate.py` still prints COAP's version for comparison.

---

## 5. Design decisions

### Ground-truth occupancy: ray-stabbing parity

Occupancy labels use ray-stabbing parity, matching `leap.tools.libmesh.check_mesh_contains` from
the original pipeline. A generalized winding number, which a fresh implementation would reach for,
is not used. The two definitions disagree on self-intersecting meshes. Where an arm passes through
the torso, parity reports the doubly-covered region as *outside* and a winding number reports it
as *inside*. AMASS is full of such poses, and resolving them is the point of the model's
self-penetration loss.

kaolin's `check_sign` is parity-based. Against trimesh's independent CPU ray-stabbing it disagrees
on 0.02–0.04% of points, concentrated near the surface, where rays graze edges and the answer is
genuinely ambiguous.

### Data generation on the GPU

COAP generates query points and labels per sample in CPU `DataLoader` workers, and runs the SMPL-X
forward pass twice per sample (once on CPU for sampling, once on GPU for the loss). Here, sampling
and ground truth run inside the training step on the GPU, and the dataset yields only indices into
the pose cache. This removes the redundant forward pass as well as the CPU bottleneck. Per-step cost
at batch 8, before the ragged filter:

| Phase | ms | Share |
|---|---:|---:|
| SMPL-X forward | 10.3 | 3.6% |
| Point sampling | 11.4 | 4.0% |
| kaolin occupancy | 12.7 | 4.4% |
| `query_training` forward | 121.9 | 42.3% |
| Backward | 128.8 | 44.7% |

Data generation costs 12% of the step; the model costs 87%. The RNG stream differs from COAP's, so
generated data is statistically, not bitwise, identical. Bitwise agreement was never attainable
anyway, because the authors' RNG stream is unrecoverable.

### Environment: torch 2.8, PyTorch3D from source

Blackwell (sm_120) needs torch ≥ 2.7 with CUDA ≥ 12.8. The bottom of that window (torch 2.8.0 +
cu129) is pinned rather than the newest release, because PyTorch3D has no wheel for this stack and
compiles more reliably against an older API. It built on the first attempt (0.7.9, 398 s). kaolin
0.18.0 ships a matching wheel.

PyTorch3D cannot be swapped for kaolin at runtime, because the model package imports it
unconditionally at module top level. Losing it would mean either a shim package or editing the
model, and both are worse than building it.

### The model package is left unmodified

`VolumetricSMPL/` is not edited. Roughly 70% of the training logic already ships in it:
`query_training()` (both loss terms), `_sample_part_udf()`, `get_gt_sdf()`, `compute_iou()`,
`Partitioner`, the PointNet encoder and the NBW decoder. Keeping it untouched means any failure can
be attributed to `training/` rather than to drift in the model. Known frictions are handled from the
outside:

- `detach_cache()` must be called every step, or the pose-keyed `impl_code` cache can reuse a stale
  autograd graph.
- The ragged filter and the mesh extractor are wrappers rather than edits.

### Ragged in-box filter

`query_training` evaluates all 15 parts for all 7,680 points, but only **8.76%** of (body, part,
point) triples lie inside the part's box. The rest are multiplied by the box mask, which zeroes
value and gradient alike. Because `sigmoid > 0`, the max over parts never selects a masked entry.
Skipping those triples is therefore exact, not an approximation. Each (body, part) row is compacted
to its in-box points and padded to the batch maximum (1,527 of 7,680).

| | Unfiltered | Filtered |
|---|---:|---:|
| Step time | 288 ms | **169 ms** (1.66×) |
| Peak VRAM | 5.55 GiB | **2.97 GiB** |
| Projected 15 epochs | 33.9 h | **~22 h** |

- **Losses are bit-identical** (`mse_occ`, `mse_udf` and total agree to 0.0).
- **Gradients agree** to cosine 0.99999 (relative L2 1.5e-3 to 9.0e-3). Bit-equality on gradients is
  not achievable: the *unfiltered* path is not bit-equal to itself on an unchanged batch (1.9e-3 to
  7.0e-3), because float32 backward is nondeterministic. The criterion that holds is agreement
  within the reference path's own run-to-run noise.
- The package's `query_fast` filter does not help in training. It masks the query dimension across
  all parts, and nearly every training point is inside *some* part's box.

---

## 6. Results

### Training run

| | |
|---|---|
| Data | BMLmovi@5 + DFaust@10, 256,045 bodies; validation 316 |
| Schedule | batch 8, 32,005 steps/epoch × 15 = 480,075 steps |
| Throughput | ~5.7 it/s, ~1.5 h per epoch, ~24 h total including one crash and resume |
| Peak VRAM | 3.46 GiB allocated |

Validation IoU mean per epoch (in-training, one seed), with the BMLmovi-only run for comparison:

| Epoch | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| BMLmovi + DFaust | 87.48 | 89.20 | 89.80 | 90.85 | 91.35 | 91.86 | 92.25 | 92.63 | 92.83 | 93.23 | 93.32 | 93.48 | 93.66 | 93.68 | 93.71 |
| BMLmovi only | 86.43 | 89.33 | 90.24 | 90.79 | 91.28 | 91.53 | 92.01 | 92.37 | 92.74 | 92.93 | 93.05 | 93.29 | 93.41 | 93.44 | 93.45 |

The curve came within the gate at epoch 4, crossed the reference at epoch 5, and converged: the last
three epochs moved +0.18, +0.01 and +0.04. Scoring the step-190,000 checkpoint through the evaluation harness gave 91.93, against
91.90 from in-training validation, so the two measurements agree to 0.03.

### Final measurement

`last.ckpt` was scored in **five separate processes**, seeds 0–4. Each process re-scans AMASS,
rebuilds the body and reloads the checkpoint, so neither a stale cache nor leaked state can hide in
the mean.

| Metric | Trained (mean ± sd) | Released | Δ | Paper |
|---|---:|---:|---:|---:|
| IoU mean | **93.68 ± 0.041** | 91.52 | +2.16 | 94.67 |
| IoU surface | **91.39 ± 0.063** | 88.49 | +2.90 | 94.25 |
| IoU uniform | **95.97 ± 0.024** | 94.55 | +1.42 | 95.10 |
| MSE SDF | **3.849e-5 ± 1.2e-7** | 5.750e-5 | −1.9e-5 | 3.7e-5 |
| MSE \|SDF\| | **3.691e-5 ± 1.1e-7** | 5.430e-5 | −1.7e-5 | 3.5e-5 |

Per-seed IoU mean: 93.70, 93.72, 93.66, 93.62, 93.70. IoU mean is exactly the average of surface and
uniform, so only two of the IoU numbers are independent. Against the paper they point in opposite
directions: uniform *exceeds* it, surface falls short. The whole shortfall sits in the one metric
whose protocol is unknown ([the σ sweep](#the-σ-sweep)).

`last.ckpt` holds the step-480,000 weights, not step 480,075, because Lightning's `save_last`
follows the 5,000-step interval. At lr 1.5e-5 the difference is immaterial.

### Generalisation and memorisation

Single seed; both models scored by the same harness on the same points.

| Split | Bodies | Released | Trained | Δ | What it tests |
|---|---:|---:|---:|---:|---|
| `val` (PosePrior) | 316 | 91.55 | 93.71 | +2.17 | unseen, but every gate was read off it |
| **`holdout_bmlrub`** | 3,938 | 94.39 | **95.18** | **+0.80** | unseen subjects and motion style |
| `seen_dfaust` | 148 | 93.31 | 96.85 | +3.54 | frames that *were* trained on |
| `holdout_dfaust` | 148 | 93.24 | 96.81 | +3.57 | adjacent frames, never trained on |

- **Memorisation is ruled out.** The DFaust pair is matched on sequence, subject, shape and motion:
  frames ≡ 0 (mod 10) were all trained on, and frames ≡ 3 (mod 10) never were. The trained model's
  seen − unseen gap is +0.043 (+0.029 on a repeat). The released model's control gap moved from
  +0.066 to −0.090 between identical runs, so the noise floor is ±0.1, and the gap is
  indistinguishable from zero.
- **The trained model wins on true generalisation.** BMLrub is in no training split here, but it
  appears in COAP's config, so it may well be in the released model's training data. Leading there
  is the favourable direction for this test.
- **The margin is not uniform.** Both models find BMLrub easier than PosePrior, and the lead shrinks
  from +2.17 to +0.80. The lead is consistent across the splits, but not uniform.

### Evaluation audit

`training/audit.py` was written when the trained model overtook the reference, which is when a
harness bug is most likely to go unnoticed. On the final checkpoint:

| Check | Result |
|---|---|
| Train/validation sequence overlap | 1,993 train vs 35 validation sequences, **0 shared** |
| Validation split is PosePrior only, 316 bodies | ✓ |
| Checkpoint loads strictly | 79 tensors, 0 missing, 0 unexpected |
| Same architecture, different weights | relative parameter L2 difference 1.449 over 3,960,908 parameters |
| Both models see identical bodies | max vertex delta 0.0 m |
| Perfect / all-empty / inverted predictor | 100.0000 / 0.0000 / 0.0000 |
| Independent IoU implementation vs package | agree to 4 dp, both models |
| Released model reproduces the stored reference | 91.51 vs 91.52 |
| Ground-truth occupancy vs trimesh on real evaluation points | 0.0195% disagreement |

Scored on identical points in one process: released 91.510, trained 93.664, Δ **+2.154**.

### Surface quality

IoU and SDF MSE are point-sampled averages, so they cannot see a floating fragment or a hole. Meshes
were extracted from both models with marching cubes at a 4 mm box-fitted voxel, and compared with
the ground-truth SMPL-X surface using 50,000 samples per direction and an exact point–triangle
distance ([§7](#pytorch3d-point_face_distance-is-wrong-on-small-triangles)). Six validation bodies,
`logit` field (see below), distances in mm:

| Body | Chamfer, released | Chamfer, trained | p95, released | p95, trained | Hausdorff, released | Hausdorff, trained |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 0.975 | **0.750** | 2.947 | **2.144** | 17.87 | **9.93** |
| 1 | 2.166 | **1.881** | 6.896 | **6.215** | 56.83 | **41.64** |
| 2 | 1.964 | **1.336** | 6.186 | **4.079** | 66.26 | **44.88** |
| 3 | 2.138 | **1.599** | 7.209 | **5.063** | 68.93 | **67.36** |
| 4 | 1.635 | **1.243** | 5.304 | **3.786** | **52.10** | 54.41 |
| 5 | 1.500 | **1.159** | 4.656 | **3.607** | **38.25** | 41.46 |
| **mean** | 1.730 | **1.328** | 5.533 | **4.149** | 50.04 | **43.28** |

The trained model's surface is closer to the ground truth on every body by Chamfer and p95 (−23% and
−25% on average), and on 4 of 6 by Hausdorff. Both models produce a few detached fragments and are
often not watertight at this resolution. Component counts are not compared, because they are not
stable between runs ([§7](#the-body-encoder-is-nondeterministic-on-gpu)).

The same checkpoint's body 0 and body 1, scored by the independent JAX implementation (different
marching-cubes extractor, same exact distance), read 0.742 and 1.901 mm.

**The extraction field matters more than the voxel size.** The occupancy field is a steep sigmoid.
Sampled along 1,500 ground-truth normals, it goes from 0.95 to 0.05 in 2.08 mm (median), which is
narrower than the voxel. Marching cubes places vertices by interpolating along cell edges; when the
field is already saturated at both ends of an edge, vertices snap toward cell midpoints and the
surface terraces. Inverse-sigmoiding the occupancy recovers the decoder's own smooth pre-activation
output (`logit(sigmoid(−x)) = −x`). One body, 4 mm voxel:

| Field | Normal error | Mean dihedral |
|---|---:|---:|
| occupancy @ 0.5 | 10.72° | 9.28° |
| **logit @ 0** | **4.63°** | **3.19°** |
| SDF head @ 0 | 12.55° | 12.34° |
| occupancy + Taubin ×5 smoothing | 6.79° | 4.59° |

A sphere control isolates the extractor from the network. It is an analytic sphere at the same
voxel, with fields of different steepness:

| Field on a sphere | Radius error | Normal error |
|---|---:|---:|
| metric SDF | 0.003 mm | 0.13° |
| sigmoid at the model's steepness (2,836 /m) | 0.350 mm | 7.81° |
| hard step | 0.668 mm | 12.60° |

So most of the terracing comes from extracting a saturated field, not from the model. `logit` is
the default. Smoothing is available for display only and is never applied to measured meshes.

---

## 7. Pitfalls and upstream issues

### smplx 0.1.28 fails on a batch smaller than its construction size

`SMPLX.forward` mixes the runtime batch size with the construction-time one, two lines apart
(`body_models.py:1236-1239`):

```python
lmk_faces_idx   = self.lmk_faces_idx.unsqueeze(0).expand(batch_size, -1)          # runtime
lmk_bary_coords = self.lmk_bary_coords.unsqueeze(0).repeat(self.batch_size, 1, 1)  # construction-time
```

The validation set's last batch is 316 % 8 = 4 bodies, so validation crashed at the end of every
epoch with `einsum(): subscript b has size 8 ... does not broadcast`. The fix is to construct the
body with `batch_size=1`, so the default buffers broadcast to any runtime batch.

### The package's `extract_mesh` is broken for `VolumetricSMPL`

`VolumetricSMPL` subclasses `BasicBodyModel` and overrides `query` to return an SDF. The inherited
`extract_mesh` still calls `self.query` and applies an inverse-sigmoid, so it runs marching cubes on
`logit(SDF)`. On the released checkpoint the field ranged from −0.087 to 1.134 (an occupancy cannot
leave [0, 1]), the logit produced NaNs, and the "mesh" spanned the whole query cube (extent 1.836 m
against a body height of 1.669 m). `query_occupancy` is the correct entry point, and
`training/meshmetrics.py` reimplements extraction on top of it.

### PyTorch3D `min_triangle_area`: keep the default for training targets

The package's `point_mesh_distance` inherits PyTorch3D's `min_triangle_area = 5e-3`. Every SMPL-X
face is far below that (mean area 7.8e-5 m²), so the degenerate-triangle fallback runs on all 20,908
faces, and points exactly on the surface read 1.91 mm instead of 0. It looks like a bug, but
lowering the guard makes the ground-truth distance ~10× worse:

| Point set | True | Default 5e-3 | 1e-8 |
|---|---|---:|---:|
| on the surface | 0 mm | 1.91 mm | 0.00 mm |
| surface + N(0, 0.01) | 7.98 mm (analytic half-normal) | **8.01 mm** | 0.81 mm |
| uniform in part box | up to ~150 mm | max 151.7 mm | max 17.7 mm |

The default is also what the released model was trained against. `test_parity.py` asserts the
half-normal identity, so a change that breaks this fails loudly.

### PyTorch3D `point_face_distance` is wrong on small triangles

Below the guard, PyTorch3D's containment test stops discriminating on small triangles in float32. For
a point whose projection falls outside the triangle, it returns the *plane* distance instead of the
distance to the nearest edge:

| Triangle edge | PyTorch3D | Exact | Ratio |
|---|---:|---:|---:|
| 1.000 m | 2.179450 | 2.179449 | 1.000 |
| 0.100 m | 0.217945 | 0.217945 | 1.000 |
| 0.010 m | 0.021794 | 0.021794 | 1.000 |
| **0.005 m** | **0.002499** | **0.010897** | **0.229** |

SMPL-X's mean edge is ~12 mm, and many faces (hands, face, feet) are much smaller. A mesh distance
takes a min over triangles, so a single spuriously small value wins the query. On a 200-vertex
population, an exact float64 reference gave a mean of 0.7627 mm and PyTorch3D gave 0.3165 mm. Chamfer
on extracted meshes was underestimated several-fold, because the ground-truth → prediction direction
measures against marching-cubes triangles, which are smaller still: the released checkpoint's
six-body mean read 0.21 mm with PyTorch3D and is 1.73 mm with the exact distance. The ordering of the
two models was the same under both.

This was found when an independent JAX implementation of the surface metrics (branch `jax`) scored
the same mesh ~4× higher, and an exact reference sided with it. Surface metrics now use a
brute-force point–triangle distance (`meshmetrics._p2m`), and `test_parity.py` checks it on analytic
cases, including one where PyTorch3D reads 2.50 mm for a true 4.33 mm. Point-sampled metrics (IoU,
SDF MSE) never touch this code path and are unaffected.

### kaolin returns squared distance

`kaolin.metrics.trianglemesh.point_to_mesh_distance` returns squared distance, as PyTorch3D's
`point_face_distance` does. Both are 0 at the surface, so a missing `sqrt` produces a
plausible-looking loss curve rather than an error. `docker/verify_env.py` determines the convention
from analytic answers.

### The body encoder is nondeterministic on GPU

Two `encode_body` calls on identical input give latent codes differing by up to 2.4e-2. Querying
200,000 fixed points twice with the cache reset changed 17,162 of the values, by up to 0.71 in
occupancy near the level set. The decoder and marching cubes are exact. The consequences:

- A single extraction is internally consistent, because all chunks share one cached latent.
- **Connected-component counts are not a stable measurement.** One body gave 3 and 18 components in
  two processes.
- Surface metrics carry a small run-to-run jitter. The 5-seed IoU spread already contains this
  effect.

The same holds for the released checkpoint, since this is upstream behaviour.

### PyTorch Lightning

- Passing a `WandbLogger` to `Trainer` *replaces* the default CSV logger rather than adding to it.
  Both are attached explicitly, so metrics are always recorded locally.
- Every resume starts a new `csv/version_N/` directory. Reading only the newest one silently drops
  everything before the last restart. Pass `--wandb-id` so the W&B run stays continuous.
- Metric names get an `_epoch` suffix only when both `on_step` and `on_epoch` are set. Read column
  names from the CSV rather than assuming them.
- `last.ckpt` is rewritten only at the checkpoint interval, so it is stale between saves and is not a
  progress indicator.
- The dataloader is not resumable, so a resume replays the interrupted epoch from its start. Some
  samples in that epoch are seen twice.

### Windows: concurrent GPU jobs can kill training

The training run died at step 260,000 with `CUDA error: unknown error` while a second GPU job (the
audit) was running. On Windows/WDDM, heavy contention can trigger a driver timeout reset that
destroys every CUDA context. Resuming from `last.ckpt` restored step, optimizer and scheduler. Do not
run GPU work alongside training.

Two process pitfalls hid a crash for a while: piping training output through `tail` returns *tail's*
exit status, and a missing metric column read as "validation never ran" rather than "the parser is
wrong".

---

## 8. Methodology notes

Lessons from conclusions that were initially wrong.

- **Tuning on a subset misleads.** A σ=0.02 protocol matched Table 1's IoU to within 0.06 on the
  first 64 validation bodies, then fell 1.1 points short on all 316. The leading bodies are easier
  than the set. The same trap applies to parity checks between implementations: 32 bodies showed a
  0.39 IoU "discrepancy" that was pure sampling variance.
- **A null result is only as good as its resolution.** At a 9.56 mm voxel the two models' surfaces
  looked equivalent, and the differences changed sign from body to body. That was a statement about
  the grid: its own error was far larger than the effect. At 4 mm the sign was consistent.
- **A control must vary the thing under suspicion.** A sphere test "cleared" marching cubes of the
  mesh terracing, but it used a metric SDF, not the saturated sigmoid actually being extracted. Run
  on the right field, the same test reproduced the artefact.
- **Consistency with a reference is not correctness.** A surface distance that matched the package's
  own call argument for argument was still wrong for the question being asked, twice over (the
  degenerate guard, then PyTorch3D's small-triangle defect).
- **A diluted comparison can "prove" two things are the same.** Comparing whole state dicts reported
  the trained and released models as nearly identical (0.0002), because ~880k identical integer
  face-index buffers swamped the weights. Restricted to trainable parameters, the difference is 1.45.
- **Do not read a trend into an unfinished curve.** Mid-run, the BMLmovi-only curve was called
  "converging" at epoch 2 and a "settled −0.28 offset" at epochs 5–7, and each was contradicted
  within an epoch or two. The result is the final number, measured over paired seeds.
- **Sanity checks beat static review.** Two surface-path bugs passed a careful code review and were
  caught in minutes by physically impossible numbers (537 mm Chamfer against 93% IoU). Both are now
  assertions.

---

## 9. Open questions

- **The paper's evaluation protocol.** Every number here is conditional on a σ and a sampling scheme
  inferred from COAP. σ alone moves IoU surface by ~6 points on fixed weights. A statement of the
  protocol from the authors would replace the largest assumption in this work.
- **Training-set composition.** Size is ruled out as the explanation for the lead. Composition is
  not, and `train_bmlrub` (COAP's SMPL-X subsets) has not been trained.
- **Training run-to-run variance.** There is one training run per configuration. Seed-to-seed
  *evaluation* variance is measured; training variance is not, so small differences between two
  trained models (for example the −0.17 from dropping DFaust) are not yet calibrated against it.

---

## Appendix: terminology

| Term | Meaning |
|---|---|
| **Body** | One posed SMPL-X mesh instance; one sample in a batch |
| **Part** | One of K rigid segments from the dominant-blend-weight decomposition (K=15 for SMPL-X) |
| **Tight face set** | The triangles assigned to one part; used for surface sampling and box extents |
| **Extended face set** | A part's tight faces plus its parent's and children's; used for ground-truth part distances |
| **Part-local space** | The frame obtained by applying a part's inverse bone transform; makes the model pose-independent |
| **Part box** | Axis-aligned box of a part's vertices in part-local space, padded by 1.125. Points outside every box get an analytic answer |
| **Query point** | A 3D location where occupancy or distance is evaluated; 512 per part per body |
| **Uniform / surface sample** | A query point drawn uniformly inside a part box / on the part surface plus N(0, σ) jitter |
| **Occupancy** | Inside/outside value in [0, 1]: max over parts of each part's sigmoid, masked to its box. Level set 0.5 |
| **Ground-truth occupancy** | Binary label by ray-stabbing parity against the posed mesh |
| **UDF / SDF** | Unsigned distance (the second decoder head) / signed distance (UDF signed by occupancy) |
| **NBW** | Neural Blend Weights: per-part, per-pose decoder weights from a base matrix plus a low-rank blend of R=80 bases |
| **Latent code** | The 128-d PointNet encoding of a part's local point cloud |
| **Reference** | The released checkpoint's score under the frozen protocol; the gate's target |
| **Ragged filter** | Restricting the training forward pass to in-box (body, part, point) triples; exact, not approximate |
| **Stride** | Per-dataset frame subsampling interval when building a split |
| **Pose cache / manifest** | The single preprocessed parameter file for a split / the hash of its selection and config |
| **Golden batch** | A pinned batch and its losses under the released checkpoint; a regression fixture |
