# Training environment: torch 2.8 + CUDA 12.9, PyTorch3D built from source, kaolin.
#
# The -devel base is mandatory: PyTorch3D ships no wheel for this torch/CUDA stack and
# must be compiled, which needs nvcc and the CUDA headers. torch 2.8 is the bottom of
# the window that supports Blackwell (sm_120) GPUs; PyTorch3D compiles far more reliably
# against it than against newer releases.
FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel

# GPU architectures to compile PyTorch3D's CUDA kernels for. The default targets
# Blackwell (e.g. RTX 50xx); "+PTX" embeds portable IR alongside the binaries. Override
# for other GPUs, e.g. `docker compose build --build-arg TORCH_CUDA_ARCH_LIST="8.6;8.9" train`.
ARG TORCH_CUDA_ARCH_LIST="12.0+PTX"
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    FORCE_CUDA=1 \
    MAX_JOBS=8 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        ninja-build \
        build-essential \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# --- Layer 1: pure-Python dependencies -------------------------------------
# Ahead of the compiled extensions so that a PyTorch3D build failure leaves this layer
# cached and the retry loop stays short.
#
# smplx 0.1.28 does NOT depend on chumpy; only the .pkl body-model loader needs it, and
# this pipeline uses the .npz models exclusively.
RUN pip install \
        "pytorch-lightning>=2.4,<3" \
        "smplx==0.1.28" \
        trimesh \
        scikit-image \
        wandb \
        pyyaml \
        tqdm \
        pytest \
        fvcore \
        iopath

# --- Layer 2: kaolin (prebuilt wheel) --------------------------------------
# NVIDIA ships a wheel matched to torch-2.8.0_cu129, so this is a download, not a build.
# Provides the parity-based `check_sign` used for ground-truth occupancy and the GPU
# data generation.
RUN pip install "kaolin==0.18.0" \
        -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.8.0_cu129.html

# --- Layer 3: PyTorch3D from source ----------------------------------------
# Last and slowest (~7 min). The model package imports pytorch3d.structures,
# pytorch3d.ops and pytorch3d.loss.point_mesh_distance at module top level, so it is
# required for `import VolumetricSMPL` to succeed at all.
ARG PYTORCH3D_REF=v0.7.9
RUN pip install --no-build-isolation \
        "git+https://github.com/facebookresearch/pytorch3d.git@${PYTORCH3D_REF}"

# --- Layer 4: test and report dependencies ---------------------------------
# After PyTorch3D so adding one never triggers a recompile. rtree backs trimesh's
# ray tracer, the independent CPU reference in the occupancy differential test. plotly
# renders the report's interactive 3D meshes; wandb-workspaces is the W&B Reports API.
RUN pip install rtree plotly wandb-workspaces

# Record exactly what resolved, so the environment is auditable after the fact.
RUN pip freeze > /opt/requirements.lock.txt

WORKDIR /workspace
