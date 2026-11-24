"""The Payne-style MLP baselines (port of ``build_payne2019``)."""

from __future__ import annotations

from flax import nnx

from prism.config import N_LABELS, N_PIXELS
from prism.models.layers import MLP


class Payne(nnx.Module):
    def __init__(self, depth: int = 2, width: int = 300, *, rngs: nnx.Rngs):
        self.mlp = MLP(N_LABELS, [width] * depth, N_PIXELS, hidden_act=nnx.relu, rngs=rngs)

    def __call__(self, labels, bad_mask=None):
        del bad_mask
        return self.mlp(labels)
