"""Fresh initialisation, reproducing the package's.

Initialisation here is load-bearing, not stylistic, and three parts of it are not what a
generic init would do:

- **`ResnetBlockFC` zero-initialises `fc_1.weight`**, so every residual block starts as
  exactly its shortcut. The encoder begins as a shallow linear map and grows depth as
  training moves those weights off zero.
- **`ImplicitNet`'s geometric init** puts the field at an approximate unit sphere before
  any training: the final layer's weights are drawn tightly around
  `sqrt(pi)/sqrt(fan_in)` and its bias set to `-radius_init`. Starting an SDF/occupancy
  network from a sphere rather than from noise is what keeps the zero level set from
  being shredded early on.
- **The NBW extras start near zero** (`0.001 * randn`), so the decoder begins as the
  shared base weight matrix with essentially no per-pose or per-part modulation, and the
  conditioning is learned rather than imposed.

Shapes are taken from the released checkpoint rather than re-derived from a config. That
guarantees the fresh parameter tree is structurally identical to the checkpoint's -- if
they ever disagree, loading would fail loudly instead of broadcasting into something
plausible.
"""
from __future__ import annotations

import math

import jax
import jax.numpy as jnp

from VolumetricSMPL import checkpoint as C

# ImplicitNet(radius_init=1): the sphere the field starts as.
RADIUS_INIT = 1.0
# CondResFieldsLinear's scale for the low-rank conditioning parameters.
NBW_SCALE = 0.001


def _linear_uniform(key, shape, fan_in):
    """torch `nn.Linear`'s default: kaiming_uniform(a=sqrt(5)) reduces to
    U(-1/sqrt(fan_in), +1/sqrt(fan_in)), and the bias uses the same bound."""
    bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
    return jax.random.uniform(key, shape, minval=-bound, maxval=bound)


def init_weights(key, template: str = 'released') -> dict:
    template = C.load_weights(template)
    shapes = {k: tuple(v.shape) for k, v in template.items()}

    # Decoder layer count and per-layer fan-in, read off the template.
    lin_ids = sorted({int(k.split('.')[1][3:]) for k in shapes
                      if k.startswith('decoder.lin')})
    last_lin = max(lin_ids)

    out = {}
    keys = jax.random.split(key, len(shapes) + 1)
    for i, name in enumerate(sorted(shapes)):
        shape = shapes[name]
        k = keys[i]

        if name.startswith('encoder.'):
            if name.endswith('.weight'):
                # Every residual block starts as its shortcut.
                out[name] = (jnp.zeros(shape) if name.endswith('fc_1.weight')
                             else _linear_uniform(k, shape, shape[1]))
            else:
                w = shapes[name[:-len('bias')] + 'weight']
                out[name] = _linear_uniform(k, shape, w[1])
            continue

        # --- decoder ----------------------------------------------------------
        layer = int(name.split('.')[1][3:])
        leaf = name.split('.', 2)[2]

        if leaf == 'weight':
            if layer == last_lin:
                # Geometric init: start the field as a sphere of radius `radius_init`.
                mean = math.sqrt(math.pi) / math.sqrt(shape[1])
                out[name] = mean + jax.random.normal(k, shape) * 1e-5
            else:
                std = math.sqrt(2.0) / math.sqrt(shape[0])
                out[name] = jax.random.normal(k, shape) * std
        elif leaf == 'bias':
            out[name] = (jnp.full(shape, -RADIUS_INIT) if layer == last_lin
                         else jnp.zeros(shape))
        elif leaf in ('rf_weight', 'part_weight', 'part_weight2rank'):
            out[name] = jax.random.normal(k, shape) * NBW_SCALE
        elif leaf.startswith('cond2weight'):
            if leaf.endswith('.weight'):
                out[name] = _linear_uniform(k, shape, shape[1])
            else:
                w = shapes[name[:-len('bias')] + 'weight']
                out[name] = _linear_uniform(k, shape, w[1])
        else:
            raise KeyError(f'no init rule for {name}')

    missing = set(shapes) - set(out)
    if missing:
        raise KeyError(f'uninitialised parameters: {sorted(missing)}')
    return out
