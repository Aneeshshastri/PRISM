"""CNN warm start / Gaussian prior in physical units."""

from __future__ import annotations

import jax
import numpy as np

from prism.models.cnn_predictor import clean_flux
from prism.models.functional import make_flux_fn

VAR_FLOOR = 1e-5  # same floor as CNN training


def predict_physical(cnn, flux: np.ndarray, mean23: np.ndarray, std23: np.ndarray,
                     batch: int = 512) -> tuple[np.ndarray, np.ndarray]:
    """Return (mu, sigma) in physical units; sigma floored at 1% of the label std."""
    fn, state = make_flux_fn(cnn)
    mus, sds = [], []
    for s in range(0, len(flux), batch):
        mu, raw = fn(state, clean_flux(np.asarray(flux[s:s + batch], np.float32)))
        mus.append(np.asarray(mu))
        sds.append(np.sqrt(np.asarray(jax.nn.softplus(raw)) + VAR_FLOOR))
    mu = np.concatenate(mus) * std23 + mean23
    sd = np.maximum(np.concatenate(sds) * std23, 0.01 * std23)
    return mu.astype(np.float32), sd.astype(np.float32)
