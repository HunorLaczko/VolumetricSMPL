# Findings

Findings from reimplementing the VolumetricSMPL training pipeline and porting the package to JAX: the
recovered training recipe, the evaluation protocol, design decisions, results, and issues found in
upstream code and in the toolchain.

The pipeline was built twice, first in PyTorch (the `main` branch) and then in JAX (this branch).
Results that were measured with the PyTorch implementation are labelled as such.

**Hardware.** One RTX 5060 Ti (16 GB, Blackwell sm_120), Ryzen 5 5600X, 64 GB RAM, Windows 11 with
Docker Desktop (WSL 2).

**Contents**

1. [Summary](#1-summary)
2. [Recovered training recipe](#2-recovered-training-recipe)
3. [Data](#3-data)
4. [Evaluation protocol](#4-evaluation-protocol)
5. [Design decisions](#5-design-decisions)
6. [Results](#6-results)
7. [Validation against the PyTorch implementation](#7-validation-against-the-pytorch-implementation)
8. [Pitfalls and upstream issues](#8-pitfalls-and-upstream-issues)
9. [Methodology notes](#9-methodology-notes)
10. [Open questions](#10-open-questions)
11. [Appendix: terminology](#appendix-terminology)

---

## 1. Summary

The upstream repository ships the inference package only; no training or evaluation code exists in
either of its branches. The training pipeline was rebuilt from the paper, the COAP codebase it
derives from, and the optimizer state left inside the released checkpoints. This branch runs the
package and the pipeline in JAX, and trains a SMPL-X neutral model from scratch in 16.3 h on one
16 GB consumer GPU.

Scored by this branch's harness, 316 validation bodies, 5 seeds:

| Model | IoU mean | IoU surface | IoU uniform | MSE SDF |
|---|---:|---:|---:|---:|
| Released checkpoint | 91.51 ± 0.03 | 88.49 ± 0.05 | 94.53 ± 0.02 | — |
| **Trained on this branch** | **93.47 ± 0.03** | **91.11 ± 0.05** | **95.83 ± 0.04** | **3.80e-5** |
| Trained with the PyTorch implementation | 93.63 ± 0.02 | 91.34 ± 0.03 | 95.92 ± 0.02 | 3.86e-5 |
| Paper, Table 1 | 94.67 | 94.25 | 95.10 | 3.7e-5 |

**What is established:**

| Claim | Evidence |
|---|---|
| The pipeline learns the method | Both implementations converge and pass the gate: +1.96 (JAX), +2.16 (PyTorch) |
| It beats the released checkpoint | +2.134 ± 0.014 paired across 5 seeds on identical points (PyTorch-trained) |
| Not memorisation | Matched seen/unseen DFaust gap indistinguishable from zero (PyTorch-trained) |
| Not a validation-set artefact | +0.80 on 3,938 unseen BMLrub bodies (PyTorch-trained) |
| Not model selection | Final checkpoints reported; validation never used to pick a model |
| Not extra training data | A size-matched training set still leads by +1.96 (PyTorch) |
| The metrics are correct | Independent IoU agrees; controls score 100 / 0 / 0, in both audits |
| The surface is better too | Chamfer 1.33 vs 1.73 mm, lower on 6/6 bodies (PyTorch-trained) |
| The two implementations compute the same model | Components agree to float32 round-off; one checkpoint scores within 0.05 IoU under both harnesses ([§7](#7-validation-against-the-pytorch-implementation)) |

**What is not established:**

- **The paper's Table 1 is not reproduced, and cannot be.** Its protocol is unpublished, and no
  single protocol reproduces all five of its numbers ([§4](#4-evaluation-protocol)).
- **The lead is not uniform.** It is +2.16 on PosePrior and +0.80 on BMLrub.
- **Which subsets the released model was trained on is unknown** ([§3](#which-training-subsets)).
- **Why the JAX-trained model trails the PyTorch-trained one by 0.17 IoU.** A known contributor is
  the ragged-budget overflow; with one training run per framework, the rest cannot be separated from
  training run-to-run variance ([§6](#the-ragged-budget-overflowed)).

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
so the cosine turns back up at the end. This is reproduced as-is. `optax`'s cosine schedule clamps at
`T_max`, so `training/train.py` writes out the periodic closed form instead. The released male and
female checkpoints are undertrained (they stop at epochs 5–6). The neutral one is fully scheduled.

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
  config rather than array bytes, so "did the data change?" has an exact answer.

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

**Training-set size does not explain the overshoot.** Measured with the PyTorch implementation:
`train_bmlmovi_only` (+0.13%) was trained to completion under an otherwise identical config, and all
three models were scored in one process on identical points (seeds 0–4):

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
over all 316 validation bodies, with the PyTorch implementation:

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

Surface jitter σ is the one unknown that moves IoU surface materially. Both models were swept
(PyTorch implementation). This was a diagnostic, not a change to the protocol.

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
| Ground-truth distance | PyTorch3D's point–face distance with its default guard, reproduced ([§8](#pytorch3d-min_triangle_area-keep-the-default-for-training-targets)) |
| IoU split | on the last axis of `(B, K, 512)` |
| Body | SMPL-X neutral, `num_betas=16`, `use_pca=False`, `flat_hand_mean=False` |

### The reference and the gate

The reference is the released `smplx_neutral` checkpoint under this protocol, averaged over 5 seeds
by the PyTorch harness. This branch's harness measures 91.51 ± 0.03 on the same checkpoint.

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

Tested with the PyTorch implementation:

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

### A JAX package without PyTorch

The original package attaches a volume to a PyTorch `smplx` model. The port reads everything itself:

- **SMPL-X buffers** come from the model `.npz`, with the processing `smplx.SMPLX` applies at
  construction (shape directions sliced to `num_betas`, expression directions split off, hand means
  folded into `pose_mean`). All nine buffers are bit-identical to the built `smplx` module's.
- **The part decomposition** is a numpy port of the package's `Partitioner`. The tight and extended
  face sets (15 × 7,596 and 15 × 11,952 rows), the vertex selector, the joint mapper and the
  self-intersection pair mask are bit-identical to the original's.
- **Checkpoints** are PyTorch zip archives. `checkpoint.py` reads them with a restricted unpickler
  that resolves only `OrderedDict`, the typed storages and the tensor rebuild functions, and refuses
  any other global. Both the released and a PyTorch-trained checkpoint load bit-identically to
  `torch.load` (73 float tensors, 3,960,908 parameters).
- **The API is pure-functional.** The original caches the body encoding inside the module and
  re-encodes when the pose changes; here `encode` returns the encoding and every query takes it, so
  everything composes with `jax.jit` and `jax.grad`.

Parameters stay a flat dict keyed by the checkpoint's names (`decoder.lin3.rf_weight`, ...). There
is no mapping table, which is the likeliest home for a silent weight-port bug. Only SMPL-X is
supported; it is the only model trained and validated here.

### Ground-truth occupancy: ray-stabbing parity

Occupancy labels use ray-stabbing parity, matching `leap.tools.libmesh.check_mesh_contains` from
the original pipeline. A generalized winding number, which a fresh implementation would reach for,
is not used. The two definitions disagree on self-intersecting meshes. Where an arm passes through
the torso, parity reports the doubly-covered region as *outside* and a winding number reports it
as *inside*. AMASS is full of such poses, and resolving them is the point of the model's
self-penetration loss.

Parity is computed brute force, in JAX, inside the training step. A BVH was tried first (Warp), but
the vertices move every step, so the BVH must be refitted every step:

| Backend | ms per batch of 8 | Disagreement with kaolin |
|---|---:|---:|
| Warp (BVH refit 120.7 + ray casts 103.4) | 224.3 | 0.034% |
| kaolin (brute force, PyTorch implementation) | 12.5 | — |
| **pure JAX (brute force)** | **10.1** | **0.026%** |

Brute force is cheap because the ray direction is constant, so most of Möller–Trumbore is
per-triangle and hoists out of the point loop (~25 flops per point–triangle pair remain). Against
trimesh's independent CPU ray-stabbing it disagrees on 0.02–0.04% of points, concentrated near the
surface, where rays graze edges and the answer is genuinely ambiguous.

### The whole step is one jitted function

COAP generates query points and labels per sample in CPU `DataLoader` workers, and runs the SMPL-X
forward pass twice per sample. Here the step gathers a batch from the pose cache on device, poses
the bodies, samples points, computes parity occupancy, runs the model and updates the weights, all
inside one `jax.jit`. Gathering the batch on the host had cost 12.6 ms per step, ~8% of it.

The port was proposed on the thesis that the PyTorch step was Python-bound, so fusing it would win.
Profiling the PyTorch step refuted that: an uninstrumented loop took 150.18 ms against 153.01 ms of
summed per-region GPU time, so there was no hidden Python overhead. The step is GPU-bound but not
compute-bound: the decoder does 57.13 GFLOPs per step in 90 ms, 635 GFLOP/s against a measured GEMM
ceiling of 25,428 GFLOP/s on the same card, **2.5% utilisation**. The cost is kernel granularity
(120 independent 64×64 matrix products per layer) and the NBW layer materialising per-(body, part)
weights. The JAX step still ends up 1.23× faster ([§6](#training-run)), from kernel fusion.

### Ragged in-box filter

The loss evaluates all 15 parts for all 7,680 points, but only **8.76%** of (body, part, point)
triples lie inside the part's box. The rest are multiplied by the box mask, which zeroes value and
gradient alike. Because `sigmoid > 0`, the max over parts never selects a masked entry. Skipping
those triples is therefore exact, not an approximation. In the PyTorch implementation this made the
step 1.66× faster with bit-identical losses.

`jit` needs static shapes, so each (body, part) row is compacted to a **fixed budget** of in-box
points (`--ragged-pad`, 2,048 of 7,680) rather than the batch maximum, and the step reports how far
the worst row exceeded it. `test_golden.py` checks that the filtered and unfiltered losses agree on
the pinned batch. The budget was too small for the full training split ([§6](#the-ragged-budget-overflowed)).

### Two distance functions, deliberately

| | `VolumetricSMPL/geometry.py` | `training/meshmetrics.py` |
|---|---|---|
| `min_triangle_area` | 5e-3 (PyTorch3D default) | 1e-12 |
| Branch that runs | nearest edge, always | plane projection when the projection is inside |
| Used for | training and evaluation targets | surface metrics |

The first reproduces what the model was trained against ([§8](#pytorch3d-min_triangle_area-keep-the-default-for-training-targets)).
It is valid as a single branch because the *largest* SMPL-X face (7.8e-4 m²) is 6× under the guard,
which `test_parity.py` asserts. The second is exact: for a point above a single large triangle's
interior, the two give 1.0000 (exact) and 1.0308; for a point on the triangle, 0.0000 and 0.2500.

### The self-intersection loss with fixed shapes

The original selects the part pairs whose boxes overlap, samples 300 candidate points in each box of
each pair, keeps the candidates inside the other box, appends the body vertices, and penalises
points that two non-adjacent parts both claim. Every intermediate has a data-dependent size. The
port gives every one of the 105 part pairs a sample budget and a validity mask, and replaces each
selection with a masked sum, so the loss is the same function of the same samples. The decoder runs
in rematerialised chunks, one body at a time, so memory stays bounded however many pairs overlap.

Against the original on 8 validation bodies and one pose with an arm driven into the hip
([§7](#7-validation-against-the-pytorch-implementation)): the overlapping pairs are identical, the
loss on the original's own candidate points is identical (283.3870 vs 283.3871), and the seeded loss
agrees in distribution (286 ± 17 vs 270 ± 26 over 10 seeds).

---

## 6. Results

### Training run

Same split, hyper-parameters and 15 epochs as the PyTorch run, fresh initialisation. `training/init.py`
reproduces the package's initialisation, which is not generic: zero-initialised residual branches, a
geometric init that starts the field as a sphere, and near-zero NBW terms.

| | PyTorch | JAX |
|---|---:|---:|
| Sustained rate | 6.66 it/s | **8.17 it/s** |
| Wall clock | 19.7 h | **16.3 h** (480,075 steps) |

That is a **1.23×** speedup. The isolated fused step measured 147.6 ms, but the real loop sustained
122 ms: timing each step forces a sync that the real loop does not pay. The schedule ended at lr
1.54e-5, on the upturn past `T_max`.

Validation IoU mean per epoch of the PyTorch run (in-training, one seed), with the BMLmovi-only run
for comparison:

| Epoch | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| BMLmovi + DFaust | 87.48 | 89.20 | 89.80 | 90.85 | 91.35 | 91.86 | 92.25 | 92.63 | 92.83 | 93.23 | 93.32 | 93.48 | 93.66 | 93.68 | 93.71 |
| BMLmovi only | 86.43 | 89.33 | 90.24 | 90.79 | 91.28 | 91.53 | 92.01 | 92.37 | 92.74 | 92.93 | 93.05 | 93.29 | 93.41 | 93.44 | 93.45 |

The curve came within the gate at epoch 4, crossed the reference at epoch 5, and converged: the last
three epochs moved +0.18, +0.01 and +0.04.

### Final measurement

| | IoU mean | IoU surface | IoU uniform | MSE SDF |
|---|---:|---:|---:|---:|
| PyTorch-trained, PyTorch harness | 93.682 ± 0.021 | 91.398 ± 0.026 | 95.966 ± 0.021 | 3.849e-5 |
| PyTorch-trained, JAX harness | 93.634 ± 0.022 | 91.344 ± 0.028 | 95.923 ± 0.024 | 3.864e-5 |
| **JAX-trained, JAX harness** | **93.468 ± 0.031** | **91.111 ± 0.047** | **95.825 ± 0.040** | **3.800e-5** |

The JAX-trained model passes the gate (+1.96 over the released checkpoint). The middle row is the
same checkpoint under both harnesses, which separates scoring from training. Of the 0.214 total gap,
**0.048 is the harness**: a consistent offset of −0.048 / −0.054 / −0.043 across the three metrics,
from a different framework, occupancy backend and distance kernel. **0.166 is the training.** That
is ~5σ of evaluation noise but 8% of the lead over the released checkpoint.

The PyTorch-trained model, scored in five separate processes by the PyTorch harness, gave per-seed
IoU mean 93.70, 93.72, 93.66, 93.62, 93.70, with MSE |SDF| 3.691e-5. IoU mean is exactly the
average of surface and uniform, so only two of the IoU numbers are independent. Against the paper
they point in opposite directions: uniform *exceeds* it, surface falls short. The whole shortfall
sits in the one metric whose protocol is unknown ([the σ sweep](#the-σ-sweep)).

### The ragged budget overflowed

The budget was set to 2,048 from a maximum of 1,527 in-box points measured on **one** batch. Over the
full training split, rows reached at least 3,268:

- Overflow occurred on **9.4% of sampled steps**, steady across all 15 epochs.
- Overflowed points drop out of that part's occupancy (`part_occ = 0`), so if no other part covers
  them the loss sees a hard 0 prediction.
- Detrended against neighbouring clean steps, affected steps show a residual `mse_occ` of
  +0.00106 ± 0.00021 (n=88), against +0.00000 ± 0.00006 on clean steps (n=796). That is ~6%
  relative on affected steps and ~0.6% of the occupancy signal overall.

The run was not restarted for it. Raising the budget is exact but linear in cost: 2,048 of 7,680 is a
3.75× reduction, and 4,096 would halve that and give back most of the speedup.

### Generalisation and memorisation

PyTorch-trained model and harness. Single seed; both models scored on the same points.

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
  from +2.17 to +0.80.

### Evaluation audit

`training/audit.py` exists because a trained model overtaking the reference is exactly when a harness
bug is most likely to go unnoticed. On the JAX-trained checkpoint:

| Check | Result |
|---|---|
| Train/validation sequence overlap | 1,993 train vs 35 validation sequences, **0 shared** |
| Validation split is PosePrior only, 316 bodies | ✓ |
| Same architecture, different weights | 73 tensors each; relative parameter L2 difference 1.444 over 3,960,908 parameters |
| Trained weights finite | ✓ |
| Perfect / all-empty / inverted predictor | 100.0000 / 0.0000 / 0.0000 |
| Independent IoU implementation vs `compute_iou` | agree to 4 dp, both models |
| Released model reproduces the stored reference | 91.54 vs 91.52 |
| Ground-truth occupancy vs trimesh on real evaluation points | 0.036% disagreement |
| Part decomposition | 15 parts, chain walked to 22 joints, padded face rows masked |

Scored on identical points in one process: released 91.537, trained 93.431, Δ **+1.893**. The
PyTorch audit on the PyTorch-trained checkpoint gave released 91.510, trained 93.664, Δ +2.154.

### Surface quality

IoU and SDF MSE are point-sampled averages, so they cannot see a floating fragment or a hole. Meshes
were extracted with marching cubes at a 4 mm box-fitted voxel, and compared with the ground-truth
SMPL-X surface using an exact point–triangle distance
([§8](#pytorch3d-point_face_distance-is-wrong-on-small-triangles)). Six validation bodies, `logit`
field (see below), distances in mm.

Released and PyTorch-trained checkpoints, PyTorch implementation (50,000 samples per direction):

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
often not watertight at this resolution.

JAX-trained checkpoint, this branch (scikit-image marching cubes, 100,000 samples per direction):

| Body | Vertices | Chamfer | p95 | Hausdorff | Components | Stray area |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 143,362 | 0.784 | 2.324 | 36.31 | 1 | 0 mm² |
| 1 | 145,414 | 2.115 | 6.939 | 87.70 | 6 | 7,944 mm² |
| 2 | 146,923 | 1.638 | 5.852 | 64.40 | 10 | 305 mm² |
| 3 | 145,628 | 1.657 | 5.568 | 68.02 | 3 | 119 mm² |
| 4 | 146,800 | 1.236 | 3.794 | 53.48 | 10 | 2,250 mm² |
| 5 | 146,939 | 1.243 | 3.921 | 40.27 | 7 | 85 mm² |
| **mean** | | **1.445** | **4.733** | | | |

Under this branch's harness the PyTorch-trained checkpoint reads 0.742 and 1.901 mm on bodies 0 and
1, against 0.750 and 1.881 under the PyTorch implementation's, and against 0.784 and 2.115 for the
JAX-trained model. That matches the direction of the IoU gap, though two bodies are not enough to say
more.

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
the default for both `training.report` and the package's `extract_mesh`.

---

## 7. Validation against the PyTorch implementation

These comparisons need both frameworks, so they are not among this branch's tests, which check
analytic answers and invariants instead.

**Component parity.** A batch was pinned as arrays (31 of them) rather than as an RNG stream, built
so each component could be tested alone: real arrays where the computation is deterministic, seeded
synthetic inputs where torch's RNG would leak in, and RNG-dependent quantities stored as inputs. All
at `matmul precision = highest`:

| Component | Agreement |
|---|---|
| `full_pose` | exact |
| vertices / joints (LBS) | 4.2e-7 / 2.4e-7 |
| ground-truth point–face distance | 1.2e-7 |
| PointNet encoder | 3.6e-7 |
| bone transforms, box bounds, local queries, box SDF | ~3e-7 |
| `inside_bbox` | 0 of 921,600 differ |
| decoder occupancy / UDF / SDF | 1.2e-5 / 1.2e-6 |
| parity occupancy vs kaolin | 0.026% of points |
| surface sampler + distance, on-surface / jittered | 1.75 / 7.96 mm (PyTorch: 1.91 / 8.01; analytic 7.98) |

Details that mattered in the transcription:

- `batch_rodrigues` adds its 1e-8 to the rotation *vector* before the norm, and SMPL-X poses are full
  of exactly-zero rotations.
- `pose_mean` must be added after assembly.
- `smplx`'s 127 joints are the 55 LBS joints with tips and landmarks *appended*, so the port can stop
  at LBS.

**Weight-port gate.** On the released checkpoint (316 bodies, 5 seeds), JAX gave
**91.511 ± 0.025 / 88.488 ± 0.046 / 94.534 ± 0.023** against the PyTorch harness's 91.52 / 88.49 /
94.55, all within seed spread. On the first 32 bodies only, IoU surface differed by +0.39 and looked
like a port bug. It was small-sample variance.

**The package without PyTorch.** Compared with the PyTorch-side artefacts and the original package:

| Item | Agreement |
|---|---|
| SMPL-X buffers, part decomposition, joint mapper | bit-identical |
| Released and PyTorch-trained checkpoints, read without torch | 73 of 73 tensors bit-identical |
| `val` and `train` pose caches, numpy scanner | bit-identical; same (sequence, frame) selection hash |
| `query` (SDF) / `query_occupancy`, given the original's body encoding | 1.7e-6 / 4.7e-5 relative |
| `collision_loss`, `_mean`, `_gmof` | ≤ 3.4e-7 relative |
| part labels (mesh colours) | 0 of 36,864 differ |
| `winding_numbers` | 1.8e-6 relative |
| self-intersection: overlapping part pairs | identical on 9 bodies |
| self-intersection loss on the original's candidates | 283.3870 vs 283.3871 |
| self-intersection loss, own sampling, 10 seeds | 286 ± 17 vs 270 ± 26 |

---

## 8. Pitfalls and upstream issues

### XLA's Triton GEMM fusion computes wrong values on GPU

With jax 0.10.2 on this GPU, compiled code that fuses the decoder's matrix products with the
surrounding elementwise ops returns wrong values whenever the weights are runtime inputs rather than
compile-time constants. That covers `jax.jit` with the parameters passed as arguments, and anything
inside `lax.map` or `lax.scan`. The same code run eagerly, on CPU, or compiled with the weights
closed over as constants is correct.

| Decoder depth | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---:|---:|---:|---:|---:|---:|---:|
| jit (weights as arguments) vs eager, max abs difference | 0 | 0.124 | 0.453 | 1.40 | 2.19 | 3.65 | 46.1 |
| output scale | 2.78 | 5.64 | 6.94 | 22.4 | 32.6 | 33.7 | 378 |

With `XLA_FLAGS=--xla_gpu_enable_triton_gemm=false` the difference drops to float32 round-off at
every depth. Random weights of the decoder's shapes reproduce it (36% relative error), so it is not
specific to the checkpoint. Changing cuDNN (9.24 vs 9.26), disabling cuBLASLt, the command buffer or
autotuning did not help. With the flag set, float32 matmul precision also takes effect again under
`jit`; the fused kernels had been ignoring it.

It surfaced as a mesh whose surface sat 1.5 cm inside the body, and as a self-intersection loss 2.5×
too small. The training step was not affected: its jitted losses matched eager execution on a real
batch (0.025304 both ways), consistent with the JAX-trained model scoring well under the eager
evaluation harness. The package disables the fusion at import, the Docker image sets the flag,
`docker/verify_env.py` fails without it, and `test_api.py` compares a compiled query against an eager
one.

### JAX defaults to TF32; torch does not

JAX uses TF32 for float32 matmuls on Ampere and later by default; torch at default settings does not.
The SMPL-X port matched `full_pose` exactly and then missed vertices by **7.5e-4 relative, 0.75 mm**.
Setting `jax_default_matmul_precision = 'highest'` brings it to 4.2e-7. The error is too small to
flip many occupancy labels, so an IoU-only check would likely have passed while every surface metric
was corrupted. The package sets `highest` at import.

### NaN gradients from a norm at zero

The analytic box SDF takes the norm of `max(q, 0)`, which is the zero vector for every point inside
the box. PyTorch defines the gradient of a norm at zero as 0; JAX's is NaN, and `jnp.where` does not
stop a NaN in the unselected branch from reaching the gradient. Collision-loss gradients came back
NaN. The norm is now computed so its gradient at zero is 0. Training never used this term.

### The package's `extract_mesh` is broken for `VolumetricSMPL`

`VolumetricSMPL` subclasses `BasicBodyModel` and overrides `query` to return an SDF. The inherited
`extract_mesh` still calls `self.query` and applies an inverse-sigmoid, so it runs marching cubes on
`logit(SDF)`. On the released checkpoint the field ranged from −0.087 to 1.134 (an occupancy cannot
leave [0, 1]), the logit produced NaNs, and the "mesh" spanned the whole query cube (extent 1.836 m
against a body height of 1.669 m). `query_occupancy` is the correct entry point, and
`VolumetricSMPL/mesh.py` extracts from it.

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

The default is also what the released model was trained against, so `VolumetricSMPL/geometry.py`
reproduces it. `test_parity.py` asserts the half-normal identity and the on-surface value, so a
change that breaks either fails loudly.

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
on extracted meshes was underestimated several-fold: the released checkpoint's six-body mean read
0.21 mm with PyTorch3D and is 1.73 mm with the exact distance. The ordering of the two models was the
same under both.

This was found when the JAX surface metrics scored the same mesh ~4× higher than the PyTorch
implementation, and an exact reference sided with JAX. Both implementations now use a brute-force
point–triangle distance, and `test_parity.py` checks it on analytic cases, including one where
PyTorch3D reads 2.50 mm for a true 4.33 mm. Point-sampled metrics (IoU, SDF MSE) never touch this
code path.

### `jax.random.categorical` materialises a Gumbel array

Area-weighted surface sampling with `jax.random.categorical` builds a (samples × categories) array:
106 GiB for 100,000 samples over a 285,000-triangle extracted mesh. The allocation is lazy, so it
fails wherever the samples are first used. `geometry.sample_faces` uses an inverse-CDF
`searchsorted` over the cumulative area instead, which draws from the same distribution.

### Windows: concurrent GPU jobs can kill training

The PyTorch training run died at step 260,000 with `CUDA error: unknown error` while a second GPU job
was running. On Windows/WDDM, heavy contention can trigger a driver timeout reset that destroys every
CUDA context. Do not run GPU work alongside training.

---

## 9. Methodology notes

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
- **A check on one code path says nothing about another.** The jitted training step matched eager
  execution, while jitted queries with the same weights were off by tens of percent. Only comparing
  the compiled query itself against eager execution found it.
- **A diluted comparison can "prove" two things are the same.** Comparing whole state dicts reported
  the trained and released models as nearly identical (0.0002), because ~880k identical integer
  face-index buffers swamped the weights. Restricted to trainable parameters, the difference is 1.45.
- **Do not read a trend into an unfinished curve.** Mid-run, the BMLmovi-only curve was called
  "converging" at epoch 2 and a "settled −0.28 offset" at epochs 5–7, and each was contradicted
  within an epoch or two. The result is the final number, measured over paired seeds.
- **Sanity checks beat static review.** Surface-path bugs passed code review and were caught by
  physically impossible numbers (537 mm Chamfer against 93% IoU; a mesh 3 cm smaller than its body).
  Both are now assertions.
- **Measure a static budget over the whole dataset.** The ragged budget was sized from one batch and
  overflowed on 9.4% of training steps.

---

## 10. Open questions

- **The paper's evaluation protocol.** Every number here is conditional on a σ and a sampling scheme
  inferred from COAP. σ alone moves IoU surface by ~6 points on fixed weights. A statement of the
  protocol from the authors would replace the largest assumption in this work.
- **Training-set composition.** Size is ruled out as the explanation for the lead. Composition is
  not, and `train_bmlrub` (COAP's SMPL-X subsets) has not been trained.
- **Training run-to-run variance.** There is one training run per configuration. Seed-to-seed
  *evaluation* variance is measured; training variance is not, so small differences between two
  trained models (the −0.17 from dropping DFaust, or the 0.17 between the JAX- and PyTorch-trained
  models) are not yet calibrated against it.

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
| **Body encoding** | Everything about a posed body that does not depend on the query point: part transforms, boxes and latent codes |
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
