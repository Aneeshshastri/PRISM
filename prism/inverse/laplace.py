"""Laplace-approximation uncertainties around a MAP solution.

Port of backward-model-v3-laplace.ipynb cell 2 (audit-fixed): the Hessian of
the per-star negative log posterior is taken in sigmoid-unconstrained space,
stabilised with a per-parameter ridge and an eigenvalue floor, inverted, and
mapped to physical sigma with the delta method. Compiled functions are
module-level (emulator ``fn`` static) so star batches reuse them.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from prism.config import ABUND_INDICES
from prism.inverse.common import HI_CORE, LO_CORE, LOWER, UPPER, Objectives, to_unconstrained
from prism.mesh import data_sharding, n_devices, pad_to_multiple, replicate

RIDGE, REL_RIDGE, MIN_EIG, MAX_COND = 1e-4, 1e-3, 1e-6, 1e6


def stabilised_inverse(H):
    """Symmetrise, add ``max(1e-4, 1e-3 |H_ii|)`` per parameter, floor eigenvalues, invert."""
    H = 0.5 * (H + H.T)
    H = H + jnp.diag(jnp.maximum(RIDGE, REL_RIDGE * jnp.abs(jnp.diag(H))))
    w, V = jnp.linalg.eigh(H)
    w = jnp.maximum(w, jnp.maximum(MIN_EIG, jnp.max(w) / MAX_COND))
    cov = (V / w) @ V.T
    return 0.5 * (cov + cov.T)


def _dtheta_du(u, lo, hi):
    s = jax.nn.sigmoid(u)
    return (hi - lo) * s * (1 - s)


def _sigma(H, u, lo, hi):
    cov = jax.vmap(stabilised_inverse)(H)
    return _dtheta_du(u, lo, hi) * jnp.sqrt(jnp.diagonal(cov, axis1=-2, axis2=-1))


@partial(jax.jit, static_argnames=("fn",))
def _joint(state, mean, std, theta, obs, ivar, mu, sig, *, fn):
    obj = Objectives(fn, state, mean, std)
    u = to_unconstrained(theta, LOWER, UPPER)
    H = jax.vmap(jax.hessian(obj.joint))(u, obs, ivar, mu, sig)
    return _sigma(H, u, LOWER, UPPER)


@partial(jax.jit, static_argnames=("fn",))
def _core(state, mean, std, theta, obs, ivar, mu, sig, *, fn):
    obj = Objectives(fn, state, mean, std)
    u = to_unconstrained(theta[:, :9], LO_CORE, HI_CORE)
    H = jax.vmap(jax.hessian(obj.core))(u, theta[:, 9:], obs, ivar, mu[:, :9], sig[:, :9])
    return _sigma(H, u, LO_CORE, HI_CORE)


@partial(jax.jit, static_argnames=("fn",))
def _element(state, mean, std, theta, j, pix, weight, obs, ivar, mu, sig, *, fn):
    obj = Objectives(fn, state, mean, std)
    lo, hi = jnp.asarray(LOWER)[j], jnp.asarray(UPPER)[j]
    u = to_unconstrained(theta[:, j][:, None], lo, hi)
    H = jax.vmap(jax.hessian(obj.element), in_axes=(0, None, 0, 0, 0, None, 0, 0, None))(
        u, j, theta, obs, ivar, pix, mu[:, j][:, None], sig[:, j][:, None], weight)
    return _sigma(H, u, lo, hi)[:, 0]


def laplace_sigmas(flux_fn, state, theta_map, obs_flux, obs_ivar, cnn_mu, cnn_sigma, element_masks: dict,
                   skip: set, prior_weights: dict, stats, mesh, *, mode: str = "two_stage") -> np.ndarray:
    """Physical-unit 1-sigma per star and label, (N, 23).

    two_stage: 9x9 core block (abundances fixed at the MAP) + 1x1 per element on
    its own pixels with its MAP prior weight; skipped elements get the CNN sigma.
    joint: full 23x23 Hessian.
    """
    if mode not in ("two_stage", "joint"):
        raise ValueError(f"unknown Laplace mode {mode!r}")
    n = len(theta_map)
    nd = n_devices(mesh)
    shard = data_sharding(mesh)
    put = lambda a: jax.device_put(pad_to_multiple(np.asarray(a, np.float32), nd)[0], shard)  # noqa: E731
    theta, obs, ivar, mu, sig = map(put, (theta_map, obs_flux, obs_ivar, cnn_mu, cnn_sigma))
    rep = replicate((state, jnp.asarray(stats[0], jnp.float32), jnp.asarray(stats[1], jnp.float32)), mesh)

    if mode == "joint":
        return np.asarray(_joint(*rep, theta, obs, ivar, mu, sig, fn=flux_fn))[:n]

    out = np.array(cnn_sigma, np.float32, copy=True)
    out[:, :9] = np.asarray(_core(*rep, theta, obs, ivar, mu, sig, fn=flux_fn))[:n]
    for j in ABUND_INDICES:
        if j in skip:
            continue
        out[:, j] = np.asarray(_element(*rep, theta, jnp.int32(j), jnp.asarray(element_masks[j], jnp.float32),
                                        jnp.float32(prior_weights[j]), obs, ivar, mu, sig, fn=flux_fn))[:n]
    return out
