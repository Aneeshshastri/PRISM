"""Batched MAP inversion (port of backward-model.ipynb map_optimize_*).

Two-stage (FERRE-like): 9-D core on the full spectrum with abundances fixed at
the CNN estimate, then each abundance in turn as a 1-D fit on its own line
pixels (results written back before the next element). Joint mode fits all 23
labels at once. Adam with cosine-decayed learning rate in sigmoid-unconstrained
space; stars are independent, sharded over the mesh.

The compiled functions live at module level with the emulator ``fn`` as a
static argument, so every star batch reuses the same executables.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from prism.config import ABUND_INDICES
from prism.inverse.common import LO_CORE, LOWER, UPPER, HI_CORE, Objectives, to_physical, to_unconstrained
from prism.mesh import data_sharding, n_devices, pad_to_multiple, replicate

B1, B2, EPS = 0.9, 0.999, 1e-7
_STATIC = ("fn", "steps", "lr_init", "lr_end")


def _adam(batched_nlp, u0, steps: int, lr_init: float, lr_end: float):
    """Minimise sum(batched_nlp(u)) with Adam (identical per star: updates are elementwise)."""
    grad = jax.grad(lambda u: jnp.sum(batched_nlp(u)))

    def body(i, carry):
        u, m, v = carry
        lr = lr_end + 0.5 * (lr_init - lr_end) * (1.0 + jnp.cos(jnp.pi * i / steps))
        g = grad(u)
        m = B1 * m + (1 - B1) * g
        v = B2 * v + (1 - B2) * g ** 2
        t = i + 1.0
        u = u - lr * (m / (1 - B1 ** t)) / (jnp.sqrt(v / (1 - B2 ** t)) + EPS)
        return u, m, v

    u, _, _ = jax.lax.fori_loop(0, steps, body, (u0, jnp.zeros_like(u0), jnp.zeros_like(u0)))
    return u, batched_nlp(u)


@partial(jax.jit, static_argnames=_STATIC)
def _joint(state, mean, std, obs, ivar, mu, sig, *, fn, steps, lr_init, lr_end):
    obj = Objectives(fn, state, mean, std)
    nlp = jax.vmap(obj.joint)
    u, loss = _adam(lambda u: nlp(u, obs, ivar, mu, sig), to_unconstrained(mu, LOWER, UPPER), steps, lr_init, lr_end)
    return to_physical(u, LOWER, UPPER), loss


@partial(jax.jit, static_argnames=_STATIC)
def _core(state, mean, std, obs, ivar, mu, sig, *, fn, steps, lr_init, lr_end):
    obj = Objectives(fn, state, mean, std)
    nlp = jax.vmap(obj.core)
    u0 = to_unconstrained(mu[:, :9], LO_CORE, HI_CORE)
    u, loss = _adam(lambda u: nlp(u, mu[:, 9:], obs, ivar, mu[:, :9], sig[:, :9]), u0, steps, lr_init, lr_end)
    return to_physical(u, LO_CORE, HI_CORE), loss


@partial(jax.jit, static_argnames=_STATIC)
def _element(state, mean, std, full, j, pix, weight, obs, ivar, mu, sig, *, fn, steps, lr_init, lr_end):
    obj = Objectives(fn, state, mean, std)
    lo, hi = jnp.asarray(LOWER)[j], jnp.asarray(UPPER)[j]
    mu1, sig1 = mu[:, j][:, None], sig[:, j][:, None]
    nlp = jax.vmap(obj.element, in_axes=(0, None, 0, 0, 0, None, 0, 0, None))
    u, loss = _adam(lambda u: nlp(u, j, full, obs, ivar, pix, mu1, sig1, weight), to_unconstrained(mu1, lo, hi),
                    steps, lr_init, lr_end)
    return to_physical(u, lo, hi)[:, 0], loss


def run_map(flux_fn, state, obs_flux, obs_ivar, cnn_mu, cnn_sigma, element_masks: dict, skip: set,
            prior_weights: dict, stats, mesh, *, mode: str = "two_stage", steps_core: int = 1200,
            steps_elem: int = 400, lr_init: float = 0.03, lr_end: float = 1e-4) -> dict:
    if mode not in ("two_stage", "joint"):
        raise ValueError(f"unknown MAP mode {mode!r}")
    n = len(obs_flux)
    nd = n_devices(mesh)
    shard = data_sharding(mesh)
    put = lambda a: jax.device_put(pad_to_multiple(np.asarray(a, np.float32), nd)[0], shard)  # noqa: E731
    obs, ivar, mu, sig = put(obs_flux), put(obs_ivar), put(cnn_mu), put(cnn_sigma)
    rep = replicate((state, jnp.asarray(stats[0], jnp.float32), jnp.asarray(stats[1], jnp.float32)), mesh)
    opt = dict(fn=flux_fn, lr_init=float(lr_init), lr_end=float(lr_end))

    if mode == "joint":
        theta, loss = _joint(*rep, obs, ivar, mu, sig, steps=int(steps_core), **opt)
        return {"theta": np.asarray(theta)[:n], "core_loss": np.asarray(loss)[:n],
                "elem_loss": np.zeros((n, len(ABUND_INDICES)), np.float32)}

    core_theta, core_loss = _core(*rep, obs, ivar, mu, sig, steps=int(steps_core), **opt)
    full = jnp.concatenate([core_theta, mu[:, 9:]], axis=1)
    elem_loss = np.zeros((n, len(ABUND_INDICES)), np.float32)
    for a, j in enumerate(ABUND_INDICES):
        if j in skip:
            continue
        val, loss = _element(*rep, full, jnp.int32(j), jnp.asarray(element_masks[j], jnp.float32),
                             jnp.float32(prior_weights[j]), obs, ivar, mu, sig, steps=int(steps_elem), **opt)
        full = full.at[:, j].set(val)
        elem_loss[:, a] = np.asarray(loss)[:n]
    return {"theta": np.asarray(full)[:n], "core_loss": np.asarray(core_loss)[:n], "elem_loss": elem_loss}
