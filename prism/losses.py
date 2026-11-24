"""Losses and metrics (jit-safe; non-finite inputs never produce NaN values or gradients)."""

from __future__ import annotations

import jax
import jax.numpy as jnp

SPL_FLOOR = 1e-5  # variance floor used by the spectral pixel loss weight


def _sanitize(flux, ivar):
    """Return (flux, ivar, ok) with non-finite / non-positive-ivar pixels neutralised."""
    flux = jnp.asarray(flux, jnp.float32)
    ivar = jnp.asarray(ivar, jnp.float32)
    ok = jnp.isfinite(flux) & jnp.isfinite(ivar) & (ivar > 0)
    return jnp.where(ok, flux, 1.0), jnp.where(ok, ivar, 1.0), ok


def masked_spl_loss(pred, flux, ivar, pixel_mask, sample_weight, row_mask, badpix_cutoff: float = 1e-3):
    """Masked spectral pixel loss (Keras ``sum_over_batch_size`` semantics, padding excluded).

    Per pixel: ``w = 1e-5 / (1/ivar + 1e-5)``, ``term = (flux - pred)^2 * w`` on
    pixels with finite flux/ivar, ``ivar > 0``, ``flux > cutoff`` and
    ``pixel_mask == 1``.
    """
    pred = jnp.asarray(pred, jnp.float32)
    flux, ivar, ok = _sanitize(flux, ivar)
    valid = ok & (flux > badpix_cutoff)
    w = SPL_FLOOR / (1.0 / ivar + SPL_FLOOR)
    resid = jnp.where(valid, flux - pred, 0.0)
    term = resid ** 2 * w * jnp.asarray(pixel_mask, jnp.float32)
    row = jnp.asarray(sample_weight, jnp.float32) * jnp.asarray(row_mask, jnp.float32)
    n_rows = jnp.maximum(jnp.sum(row_mask), 1.0)
    return jnp.sum(term * row[:, None]) / (n_rows * flux.shape[-1])


def reduced_chi2(pred, flux, ivar, valid_extra=None, n_params: int = 23, var_floor: float = 1e-5,
                 badpix_cutoff: float = 1e-3):
    """Per-star reduced chi^2 with variance ``1/ivar + var_floor`` (Model-tester definition)."""
    pred = jnp.asarray(pred, jnp.float32)
    flux, ivar, ok = _sanitize(flux, ivar)
    valid = (ok & (flux > badpix_cutoff)).astype(jnp.float32)
    if valid_extra is not None:
        valid = valid * jnp.asarray(valid_extra, jnp.float32)
    var = 1.0 / ivar + var_floor
    chi = jnp.where(valid > 0, (pred - flux) ** 2 / var, 0.0)
    dof = jnp.maximum(jnp.sum(valid, axis=-1) - n_params, 1.0)
    return jnp.sum(chi, axis=-1) / dof


def heteroscedastic_nll(y, mu, raw_var, row_mask, var_floor: float = 1e-5):
    """Gaussian NLL with ``var = softplus(raw) + floor``, averaged over labels and real rows."""
    var = jax.nn.softplus(jnp.asarray(raw_var, jnp.float32)) + var_floor
    nll = 0.5 * (jnp.log(var) + (jnp.asarray(y, jnp.float32) - mu) ** 2 / var)
    per_row = jnp.mean(nll, axis=-1)
    row_mask = jnp.asarray(row_mask, jnp.float32)
    return jnp.sum(per_row * row_mask) / jnp.maximum(jnp.sum(row_mask), 1.0)
