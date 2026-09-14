"""VolumetricSMPL in JAX: a neural volumetric body model for SMPL-X.

An unofficial port of the VolumetricSMPL package (Mihajlovic et al., ICCV 2025) that runs
without PyTorch. It reads the SMPL-X model file and the released checkpoints directly.

    import jax
    from VolumetricSMPL import VolumetricSMPL

    model = VolumetricSMPL.create('data/body_models')
    body = model.forward(betas=betas, body_pose=body_pose)
    code = model.encode(body, jax.random.PRNGKey(0))
    sdf = model.query(points, code)

----------------------------------------------------------------------------------------
Two GPU settings are changed at import, on purpose. Both guard against silently wrong
numbers, and neither is the default.

**XLA's Triton GEMM fusion is disabled.** With jax 0.10.2 on GPU, compiled code that fuses
the decoder's matmuls with the surrounding elementwise ops returns wrong values -- off by
tens of percent -- whenever the weights are runtime inputs rather than constants: inside
`jax.jit` with the parameters as arguments, and inside `lax.map`/`lax.scan`. The same code
run eagerly, or on CPU, is correct. The flag must reach XLA before its GPU backend starts,
so import this package before running any JAX computation, or set
`XLA_FLAGS=--xla_gpu_enable_triton_gemm=false` yourself. An explicit setting in
`XLA_FLAGS` is left alone.

**Matmul precision is `highest`.** JAX defaults to TF32 for float32 matmuls on Ampere and
later GPUs. Left alone, that moves SMPL-X vertices by 7.5e-4 relative -- 0.75 mm on a real
body, larger than the differences the surface metrics are meant to resolve -- while
looking like ordinary float32 round-off. Forcing `highest` brings it down to 4.2e-7. Use
`set_matmul_precision('high')` to trade it back for speed explicitly.
"""
from __future__ import annotations

import os

if 'xla_gpu_enable_triton_gemm' not in os.environ.get('XLA_FLAGS', ''):
    os.environ['XLA_FLAGS'] = (os.environ.get('XLA_FLAGS', '')
                               + ' --xla_gpu_enable_triton_gemm=false').strip()

import jax  # noqa: E402

DEFAULT_MATMUL_PRECISION = 'highest'


def set_matmul_precision(mode: str = DEFAULT_MATMUL_PRECISION) -> None:
    """Set the float32 matmul precision. 'highest' is true fp32; 'high' is TF32."""
    jax.config.update('jax_default_matmul_precision', mode)


def current_matmul_precision() -> str:
    return str(jax.config.jax_default_matmul_precision)


set_matmul_precision()

from .volumetric_smpl import BodyOutput, VolumetricSMPL  # noqa: E402
from .winding_numbers import winding_numbers  # noqa: E402

__all__ = [
    'BodyOutput',
    'VolumetricSMPL',
    'current_matmul_precision',
    'set_matmul_precision',
    'winding_numbers',
]
