# Environment for VolumetricSMPL in JAX: the package, training, evaluation and tests.
#
# No PyTorch: the SMPL-X model is read from its .npz and the released checkpoints are read
# directly from their zip archives (VolumetricSMPL/checkpoint.py).
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive \
    # JAX grabs 75% of VRAM on first allocation otherwise.
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    # Triton GEMM fusion computes wrong values for the decoder on GPU with jax 0.10.2.
    # See VolumetricSMPL/__init__.py; docker/verify_env.py checks it.
    XLA_FLAGS=--xla_gpu_enable_triton_gemm=false

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# --- Layer 1: JAX with the CUDA 12 plugin ----------------------------------
# The CUDA runtime arrives as pip `nvidia-*` wheels, so no -devel base image is needed.
RUN pip install "jax[cuda12]==0.10.2"

# --- Layer 2: optimiser ----------------------------------------------------
RUN pip install optax==0.2.8

# --- Layer 3: geometry, meshes and reporting -------------------------------
# scikit-image provides the host-side marching cubes; trimesh (with rtree) the
# independent occupancy reference and mesh topology.
RUN pip install \
        numpy \
        trimesh \
        rtree \
        scikit-image \
        wandb \
        plotly

# Record exactly what resolved, so this environment is auditable after the fact.
RUN pip freeze > /opt/requirements.lock.txt

WORKDIR /workspace
