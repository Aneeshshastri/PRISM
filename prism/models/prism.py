"""PRISM structured emulator and its full-input ablation.

Ported from model-trainer.ipynb: ``build_prism_emulator_masked`` (cell 7) and
``build_final_emulator`` (cell 11, audit-fixed). The old GradientGate is not
ported: it applied stop_gradient to input labels and had no effect on training
(imputed labels are handled by the per-star pixel mask in the loss).
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from flax import nnx

from prism.config import LABELS_27, N_LABELS, N_PIXELS
from prism.models.layers import MLP, ChebyshevContinuum, SparseProjector, beer_lambert

# Input routing (column indices into the 27-vector).
IDX_GLOBAL = [0, 1, 2, 3, 4, 5, 6, 7, 8, 23, 24, 25, 26]  # core + CNO + engineered
IDX_CNO = [0, 1, 3, 4, 5, 6, 7, 8, 23, 25]  # molecules: Teff, logg, broadening, C/N/O, INV_TEFF, C_O_diff
STATE_DIM = 16


def expert_width(n_active: int) -> int:
    return max(64, min(256, n_active))


class _Expert(nnx.Module):
    def __init__(self, in_features: int, indices: np.ndarray, weights: np.ndarray, *, rngs: nnx.Rngs):
        w = expert_width(len(indices))
        self.mlp = MLP(in_features, [w, w // 2], len(indices), out_act=nnx.softplus, rngs=rngs)
        self.project = SparseProjector(indices, weights)

    def __call__(self, x):
        return self.project(self.mlp(x))


class Prism(nnx.Module):
    """Selective routing: atomic experts see only a thermodynamic state vector and their own abundance."""

    def __init__(self, element_idx: dict, *, rngs: nnx.Rngs):
        self.continuum = ChebyshevContinuum(len(IDX_GLOBAL), rngs=rngs)
        self.molecules = MLP(len(IDX_CNO), [256, 1024], N_PIXELS, out_act=nnx.softplus, rngs=rngs)
        self.state = MLP(len(IDX_GLOBAL) + len(IDX_CNO), [32, 16], STATE_DIM, rngs=rngs)
        self.element_cols = tuple(LABELS_27.index(lab) for lab in element_idx)
        self.experts = nnx.List([_Expert(STATE_DIM + 1, ind, w, rngs=rngs) for ind, w in element_idx.values()])

    def __call__(self, labels, bad_mask=None):
        del bad_mask  # accepted for API symmetry with the TF two-input model
        k_cont = self.continuum(labels[:, IDX_GLOBAL])
        tau = self.molecules(labels[:, IDX_CNO])
        state = self.state(labels[:, IDX_GLOBAL + IDX_CNO])
        for col, expert in zip(self.element_cols, self.experts):
            tau = tau + expert(jnp.concatenate([state, labels[:, col:col + 1]], axis=-1))
        return beer_lambert(k_cont, tau)


class PrismAblation(nnx.Module):
    """Full-input ablation: every branch sees all 27 labels; only the sparse projection remains."""

    def __init__(self, element_idx: dict, *, rngs: nnx.Rngs):
        self.continuum = ChebyshevContinuum(N_LABELS, rngs=rngs)
        self.molecules = MLP(N_LABELS, [256, 1024], N_PIXELS, out_act=nnx.softplus, rngs=rngs)
        self.experts = nnx.List([_Expert(N_LABELS, ind, w, rngs=rngs) for ind, w in element_idx.values()])

    def __call__(self, labels, bad_mask=None):
        del bad_mask
        tau = self.molecules(labels)
        for expert in self.experts:
            tau = tau + expert(labels)
        return beer_lambert(self.continuum(labels), tau)
