"""RegPayne Jacobian penalty (explicit loss term).

Port of StochasticPerturbation + PhysicsLossLayer (model-trainer_backup.ipynb
cells 6-7, audit-fixed): for each star, perturb one label that has spectral
lines by ``epsilon`` and penalise the flux response on that label's FORBIDDEN
(non-line) pixels.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

# (epochs, penalty weight); None = cfg.epochs. Values from the Optuna search.
REGPAYNE_PHASES = ((5, 0.011906296737322275), (None, 1851.3543948324022))


def constrained_columns(forbidden: np.ndarray) -> np.ndarray:
    return np.where(np.asarray(forbidden).sum(axis=0) > 0)[0].astype(np.int32)


def sample_columns(key, forbidden: np.ndarray, n: int):
    """One constrained label column per row; row i uses ``fold_in(key, i)``."""
    cols = jnp.asarray(constrained_columns(forbidden))
    pick = jax.vmap(lambda i: jax.random.randint(jax.random.fold_in(key, i), (), 0, cols.shape[0]))(jnp.arange(n))
    return cols[pick]


def jacobian_penalty(model, labels, forbidden: np.ndarray, key, *, epsilon: float = 0.05, row_mask):
    labels = jnp.asarray(labels, jnp.float32)
    n, n_labels = labels.shape
    cols = sample_columns(key, forbidden, n)
    perturbed = labels + epsilon * jax.nn.one_hot(cols, n_labels, dtype=labels.dtype)
    diff = jnp.abs(model(labels) - model(perturbed))
    fmask = jnp.asarray(forbidden, jnp.float32).T[cols]  # (n, n_pixels)
    row_mask = jnp.asarray(row_mask, jnp.float32)
    total = jnp.sum(diff * fmask * row_mask[:, None])
    return total / (jnp.maximum(jnp.sum(row_mask), 1.0) * diff.shape[-1])
