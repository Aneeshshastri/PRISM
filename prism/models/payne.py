"""The Payne-style MLP baselines (port of ``build_payne2019``)."""

from __future__ import annotations

import jax.numpy as jnp
from flax import nnx

from prism.config import N_LABELS, N_PIXELS
from prism.data.masks import N_REAL_PIXELS
from prism.models.layers import MLP, expand_gaps


class Payne(nnx.Module):
    """``drop_gaps``: predict the 7514 real pixels only and zero-fill the chip gaps."""

    def __init__(self, depth: int = 2, width: int = 300, *, drop_gaps: bool = False, dtype=jnp.float32,
                 rngs: nnx.Rngs):
        self.drop_gaps = drop_gaps
        n_out = N_REAL_PIXELS if drop_gaps else N_PIXELS
        self.mlp = MLP(N_LABELS, [width] * depth, n_out, hidden_act=nnx.relu, dtype=dtype, rngs=rngs)

    def __call__(self, labels, bad_mask=None):
        del bad_mask
        out = self.mlp(labels)
        return expand_gaps(out) if self.drop_gaps else out
