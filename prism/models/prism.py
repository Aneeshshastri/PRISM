"""PRISM structured emulator and its full-input ablation.

Ported from model-trainer.ipynb: ``build_prism_emulator_masked`` (cell 7) and
``build_final_emulator`` (cell 11, audit-fixed). The old GradientGate is not
ported: it applied stop_gradient to input labels and had no effect on training
(imputed labels are handled by the per-star pixel mask in the loss).

Architecture options (``Config.arch()``; checkpoints without them rebuild the legacy model):

* ``drop_gaps``      every full-spectrum head predicts the 7514 real pixels only; the chip
                     gaps are zero-filled (they never enter the loss).
* ``chip_continuum`` one Chebyshev continuum per detector chip, initialised at k = 1, and
                     optical-depth heads initialised near tau = 0 (start near a normalised
                     spectrum instead of k ~ 1.8 against a tau ~ 0.5 floor).
* ``cno_split``      the all-pixel "molecules" branch, which saw C/N/O and therefore let them
                     act everywhere, is split into a background branch over all pixels that sees
                     only global labels (no individual abundance) and a CNO expert that sees
                     [C/H], [N/H], [O/H] jointly and writes only into the union of the C/N/O
                     windows. The continuum and the state vector drop C/N/O as well, so every
                     abundance (14 metals + C, N, O) acts only inside its own windows.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from flax import nnx

from prism.config import LABELS_27, N_LABELS, N_PIXELS
from prism.data.masks import CNO_KEY, N_REAL_PIXELS
from prism.models.layers import (MLP, ChebyshevContinuum, ChipContinuum, SparseProjector, beer_lambert,
                                 expand_gaps, init_output)

# Input routing (column indices into the 27-vector).
IDX_GLOBAL = [0, 1, 2, 3, 4, 5, 6, 7, 8, 23, 24, 25, 26]  # core + CNO + engineered
IDX_CNO = [0, 1, 3, 4, 5, 6, 7, 8, 23, 25]  # molecules: Teff, logg, broadening, C/N/O, INV_TEFF, C_O_diff
# cno_split routing: global labels without any individual abundance, and the CNO expert's own inputs
IDX_BACKGROUND = [0, 1, 2, 3, 4, 5, 23, 24, 26]  # Teff, logg, [M/H], vmicro, vmacro, vsini, INV_TEFF, vbroad, LOGPE
IDX_CNO_EXPERT = [6, 7, 8, 25]  # [C/H], [N/H], [O/H], C_O_diff
STATE_DIM = 16
TAU_INIT_BIAS = -3.0  # softplus(-3) ~ 0.05: optical depths start near zero (chip_continuum init)
# CNO expert hidden widths: with the [256, 512] background this matches the parameter count of the
# unsplit model (8.6M vs 8.4M on the real masks), so the cno_split comparison is not a capacity one.
CNO_EXPERT_WIDTHS = [256, 768]


def expert_width(n_active: int) -> int:
    return max(64, min(256, n_active))


class _Expert(nnx.Module):
    def __init__(self, in_features: int, indices: np.ndarray, weights: np.ndarray, *, widths=None,
                 tau_init: bool = False, dtype=jnp.float32, rngs: nnx.Rngs):
        w = expert_width(len(indices))
        self.mlp = MLP(in_features, widths or [w, w // 2], len(indices), out_act=nnx.softplus, dtype=dtype,
                       rngs=rngs)
        if tau_init:
            init_output(self.mlp, bias=TAU_INIT_BIAS)
        self.project = SparseProjector(indices, weights)

    def __call__(self, x):
        return self.project(self.mlp(x))


def _split_windows(element_idx: dict, cno_split: bool):
    idx = dict(element_idx)
    cno = idx.pop(CNO_KEY, None)
    if cno_split and cno is None:
        raise ValueError(f"cno_split needs the C/N/O window union under {CNO_KEY!r}: "
                         "element_indices(..., cno_masks=load_cno_masks(...))")
    return idx, cno


class _Structured(nnx.Module):
    """Shared assembly: continuum k, all-pixel optical depth, optional CNO expert, element experts."""

    def _build_common(self, cont_in: int, full_in: int, cno_in: int | None, cno, *, drop_gaps, chip_continuum,
                      cno_split, dtype, rngs):
        self.drop_gaps = drop_gaps
        n_out = N_REAL_PIXELS if drop_gaps else N_PIXELS
        self.continuum = (ChipContinuum(cont_in, dtype=dtype, rngs=rngs) if chip_continuum
                          else ChebyshevContinuum(cont_in, dtype=dtype, rngs=rngs))
        if cno_split:
            self.background = MLP(full_in, [256, 512], n_out, out_act=nnx.softplus, dtype=dtype, rngs=rngs)
            self.cno_expert = _Expert(cno_in, cno[0], cno[1], widths=CNO_EXPERT_WIDTHS, tau_init=chip_continuum,
                                      dtype=dtype, rngs=rngs)
            full = self.background
        else:
            self.molecules = MLP(full_in, [256, 1024], n_out, out_act=nnx.softplus, dtype=dtype, rngs=rngs)
            full = self.molecules
        if chip_continuum:
            init_output(full, bias=TAU_INIT_BIAS)

    def _full_tau(self, branch, x):
        tau = branch(x)
        return expand_gaps(tau) if self.drop_gaps else tau


class Prism(_Structured):
    """Selective routing: atomic experts see only a thermodynamic state vector and their own abundance."""

    def __init__(self, element_idx: dict, *, drop_gaps: bool = False, chip_continuum: bool = False,
                 cno_split: bool = False, dtype=jnp.float32, rngs: nnx.Rngs):
        idx, cno = _split_windows(element_idx, cno_split)
        self.cno_split = cno_split
        self.global_cols = tuple(IDX_BACKGROUND if cno_split else IDX_GLOBAL)
        self.full_cols = tuple(IDX_BACKGROUND if cno_split else IDX_CNO)
        self.state_cols = tuple(IDX_BACKGROUND if cno_split else IDX_GLOBAL + IDX_CNO)
        self._build_common(len(self.global_cols), len(self.full_cols), STATE_DIM + len(IDX_CNO_EXPERT), cno,
                           drop_gaps=drop_gaps, chip_continuum=chip_continuum, cno_split=cno_split, dtype=dtype,
                           rngs=rngs)
        self.state = MLP(len(self.state_cols), [32, 16], STATE_DIM, dtype=dtype, rngs=rngs)
        self.element_cols = tuple(LABELS_27.index(lab) for lab in idx)
        self.experts = nnx.List([_Expert(STATE_DIM + 1, ind, w, tau_init=chip_continuum, dtype=dtype, rngs=rngs)
                                 for ind, w in idx.values()])

    def __call__(self, labels, bad_mask=None):
        del bad_mask  # accepted for API symmetry with the TF two-input model
        cols = lambda c: labels[:, np.asarray(c)]  # noqa: E731
        k_cont = self.continuum(cols(self.global_cols))
        state = self.state(cols(self.state_cols))
        if self.cno_split:
            tau = self._full_tau(self.background, cols(self.full_cols))
            tau = tau + self.cno_expert(jnp.concatenate([state, cols(IDX_CNO_EXPERT)], axis=-1))
        else:
            tau = self._full_tau(self.molecules, cols(self.full_cols))
        for col, expert in zip(self.element_cols, self.experts):
            tau = tau + expert(jnp.concatenate([state, labels[:, col:col + 1]], axis=-1))
        return beer_lambert(k_cont, tau)


class PrismAblation(_Structured):
    """Full-input ablation: every branch sees all 27 labels; only the sparse projections remain."""

    def __init__(self, element_idx: dict, *, drop_gaps: bool = False, chip_continuum: bool = False,
                 cno_split: bool = False, dtype=jnp.float32, rngs: nnx.Rngs):
        idx, cno = _split_windows(element_idx, cno_split)
        self.cno_split = cno_split
        self._build_common(N_LABELS, N_LABELS, N_LABELS, cno, drop_gaps=drop_gaps, chip_continuum=chip_continuum,
                           cno_split=cno_split, dtype=dtype, rngs=rngs)
        self.experts = nnx.List([_Expert(N_LABELS, ind, w, tau_init=chip_continuum, dtype=dtype, rngs=rngs)
                                 for ind, w in idx.values()])

    def __call__(self, labels, bad_mask=None):
        del bad_mask
        if self.cno_split:
            tau = self._full_tau(self.background, labels) + self.cno_expert(labels)
        else:
            tau = self._full_tau(self.molecules, labels)
        for expert in self.experts:
            tau = tau + expert(labels)
        return beer_lambert(self.continuum(labels), tau)
