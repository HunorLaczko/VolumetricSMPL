"""The network: PointNet encoder, positional encoding, and the NBW decoder.

Params are a **flat dict keyed by the checkpoint's parameter names**, verbatim -- so
`params['decoder.lin3.rf_weight']` is exactly `decoder.lin3.rf_weight` from the `.ckpt`.
There is no name mapping step, which is deliberate: a mapping table is the most likely
place for a silent weight-port bug, and the cheapest way to not have that bug is to not
have the table. A flat dict is a perfectly good JAX pytree, so optax differentiates and
updates it directly.

Shapes, from the released checkpoint:
  encoder  fc_pos 3->256, ResnetBlockFC 256->128 at blocks 0,1,3,4, fc_c 128->128
  decoder  lin0 143->64, lin1/2 64->64, lin3 207->64 (skip), lin4/5 64->64, lin6 64->2
           NBW rank R=80, part rank 10, cond dim 128, K=15 parts
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

# ImplicitNet(beta=100). Softplus this sharp is nearly a ReLU but keeps the field C-inf,
# which is what makes the SDF head differentiable everywhere.
SOFTPLUS_BETA = 100.0


def _linear(p, prefix, x):
    w = p[f'{prefix}.weight']
    b = p.get(f'{prefix}.bias')
    y = x @ w.T
    return y if b is None else y + b


def softplus(x, beta: float = SOFTPLUS_BETA):
    # jax.nn.softplus is numerically safe for large beta*x (it uses logaddexp), so the
    # torch `Softplus(beta, threshold=20)` linear-passthrough branch is not needed.
    return jax.nn.softplus(beta * x) / beta


# --------------------------------------------------------------------------------------
# PointNet encoder
# --------------------------------------------------------------------------------------

def _resnet_block(p, prefix, x):
    """ResnetBlockFC: pre-activation, with fc_1 zero-initialised so the block starts as
    identity-through-shortcut."""
    net = _linear(p, f'{prefix}.fc_0', jax.nn.relu(x))
    dx = _linear(p, f'{prefix}.fc_1', jax.nn.relu(net))
    xs = _linear(p, f'{prefix}.shortcut', x) if f'{prefix}.shortcut.weight' in p else x
    return xs + dx


def encode(p, points):
    """ResnetPointnet. (N, T, 3) -> (N, 128).

    Max-pool over the point axis after each block, broadcast back and concatenate: the
    permutation-invariant part of PointNet. `use_block2=False` in this configuration, so
    blocks 0, 1, 3, 4 run and block 2 does not exist -- the numbering gap is in the
    checkpoint, not a mistake here.
    """
    net = _linear(p, 'encoder.fc_pos', points)
    for block in ('block_0', 'block_1', 'block_3'):
        net = _resnet_block(p, f'encoder.{block}', net)
        pooled = jnp.max(net, axis=1, keepdims=True)
        net = jnp.concatenate([net, jnp.broadcast_to(pooled, net.shape)], axis=2)
    net = _resnet_block(p, 'encoder.block_4', net)
    net = jnp.max(net, axis=1)
    return _linear(p, 'encoder.fc_c', jax.nn.relu(net))


# --------------------------------------------------------------------------------------
# Positional encoding
# --------------------------------------------------------------------------------------

def embed(x, multires: int = 2, alpha_ratio: float = 1.0):
    """The package's `Embedder`. (..., 3) -> (..., 3 + 4*multires*... ) = (..., 15) at
    multires=2.

    Order matters and is [identity, sin(pi x), cos(pi x), sin(2pi x), cos(2pi x)].

    `ImplicitNet` hard-wires `alpha_ratio=1.0`, at which every annealing factor evaluates
    to exactly 1, so the frequency-annealing schedule is inert in this model. It is
    implemented anyway rather than dropped, because dropping it would silently change
    behaviour if a future config ever set it.
    """
    freqs = 2.0 ** jnp.linspace(0.0, multires - 1, multires) * jnp.pi
    out = [x]
    for i in range(multires):
        dec = 0.5 * (1.0 - jnp.cos(
            jnp.pi * jnp.clip(alpha_ratio * multires - i, 0.0, 1.0)))
        out.append(jnp.sin(x * freqs[i]) * dec)
        out.append(jnp.cos(x * freqs[i]) * dec)
    return jnp.concatenate(out, axis=-1)


# --------------------------------------------------------------------------------------
# NBW (Neural Blend Weights) decoder
# --------------------------------------------------------------------------------------

def _nbw_linear(p, prefix, x, cond, ind):
    """One conditioned layer.

    The weight matrix is synthesised per row of the batch -- here one row per
    (body, part) -- as a shared base plus two low-rank offsets:

        W_n = W0 + reshape( c_n @ RF  +  P2R[k_n] @ PW )

    where `c_n = cond2weight(latent_n)` is the pose/shape-dependent blend (rank 80) and
    the second term is a per-part offset (rank 10) that does not depend on the latent at
    all. Materialising W_n is what makes this expensive: 120 independent 64x64 matrices
    per layer, which is why the step runs at ~2.5% of the card's GEMM ceiling.

    Args:
        x:    (N, T, f_in)
        cond: (N, cond_dim)
        ind:  (N,) part index per row
    """
    w0 = p[f'{prefix}.weight']                       # (f_out, f_in)
    bias = p[f'{prefix}.bias']                       # (f_out,)
    f_out, f_in = w0.shape

    # rf_weight is (L, R, f_out*f_in) with L = len(cond_dim) = 1 here; the sum over L is
    # kept so a multi-conditioning config would still be correct.
    rf = p[f'{prefix}.rf_weight']
    n_cond = rf.shape[0]
    delta = 0.0
    for l in range(n_cond):
        c = _linear(p, f'{prefix}.cond2weight.{l}', cond)   # (N, R)
        delta = delta + c @ rf[l]                            # (N, f_out*f_in)

    if f'{prefix}.part_weight' in p:
        pw = p[f'{prefix}.part_weight2rank'] @ p[f'{prefix}.part_weight']  # (K, f*f)
        delta = delta + pw[ind]

    w = w0[None] + delta.reshape(-1, f_out, f_in)            # (N, f_out, f_in)
    return jnp.einsum('nof,ntf->nto', w, x) + bias


def decode(p, x, cond, ind, n_layers: int = 7, skip_in=(3,), multires: int = 2):
    """ImplicitNet with NBW layers. (N, T, 3) -> (N, T, 2) as (occupancy logit, udf).

    The skip concatenation divides by sqrt(2), which keeps the activation scale steady
    across the join -- omitting it would rescale everything downstream of layer 3.
    """
    if multires > 0:
        x = jnp.concatenate([embed(x[..., :3], multires), x[..., 3:]], axis=-1)
    inp = x
    for layer in range(n_layers):
        if layer in skip_in:
            x = jnp.concatenate([x, inp], axis=-1) / jnp.sqrt(2.0)
        x = _nbw_linear(p, f'decoder.lin{layer}', x, cond, ind)
        if layer < n_layers - 1:
            x = softplus(x)
    return x
