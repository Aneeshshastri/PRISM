"""Posterior sampling with BlackJAX NUTS / HMC (replaces the TFP Tier-2/3 pipelines).

Each star is an independent posterior in sigmoid-unconstrained space whose log
density INCLUDES the log-det-Jacobian of the bound transform. Chains are a vmap
axis; stars are split over the mesh with ``shard_map`` so every device runs its
own (while-loop heavy) samplers. Two-stage mode samples the 9-D core then each
abundance in turn (1-D, its own pixels); joint mode samples all 23 labels.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

import blackjax
import jax
import jax.numpy as jnp
import numpy as np
from blackjax.diagnostics import effective_sample_size, potential_scale_reduction
from jax.sharding import PartitionSpec as P

from prism.config import ABUND_INDICES
from prism.inverse.common import (HI_CORE, LO_CORE, LOWER, UPPER, Objectives, log_det_jacobian, to_physical,
                                  to_unconstrained)
from prism.mesh import AXIS, n_devices, pad_to_multiple, replicate

RHAT_LIMIT = 1.05


@dataclass(frozen=True)
class MCMCSettings:
    kernel: str  # "nuts" | "hmc"
    n_chains: int
    n_warmup: int
    n_samples: int
    target_accept: float
    max_tree_depth: int = 10
    n_leapfrog: int | None = None


NUTS_CORE = MCMCSettings("nuts", 4, 200, 200, 0.9)
NUTS_ELEM = MCMCSettings("nuts", 2, 100, 100, 0.9)
HMC_CORE = MCMCSettings("hmc", 8, 400, 500, 0.65, n_leapfrog=16)
HMC_ELEM = MCMCSettings("hmc", 4, 200, 300, 0.65, n_leapfrog=10)
NUTS_JOINT = MCMCSettings("nuts", 6, 400, 400, 0.9)


def unconstrained_logdensity(logp_phys, lo, hi):
    """Log density in u-space: ``logp(to_physical(u)) + log|d theta / d u|``."""
    return lambda u: logp_phys(to_physical(u, lo, hi)) + log_det_jacobian(u, lo, hi)


def _algorithm(s: MCMCSettings):
    if s.kernel == "nuts":
        return blackjax.nuts, {"max_num_doublings": s.max_tree_depth}
    if s.kernel == "hmc":
        if not s.n_leapfrog:
            raise ValueError("HMC needs n_leapfrog")
        return blackjax.hmc, {"num_integration_steps": s.n_leapfrog}
    raise ValueError(f"unknown kernel {s.kernel!r}")


def sample_star(logdensity_u, init_u, key, s: MCMCSettings):
    """Window-adapted sampling of one posterior; ``init_u`` (C, D) -> samples (S, C, D), mean acceptance."""
    alg, extra = _algorithm(s)

    def one_chain(k, x0):
        k_warm, k_draw = jax.random.split(k)
        warm = blackjax.window_adaptation(alg, logdensity_u, target_acceptance_rate=s.target_accept, **extra)
        (state, params), _ = warm.run(k_warm, x0, num_steps=s.n_warmup)
        imm = params["inverse_mass_matrix"]

        def step(st, kk):
            k_jit, k_step = jax.random.split(kk)
            eps = params["step_size"]
            if s.kernel == "hmc":
                # Fixed-length HMC can resonate with the target (ESS ~ a few); jittering the
                # step size by +-20% each iteration breaks the periodicity and stays valid.
                eps = eps * jax.random.uniform(k_jit, minval=0.8, maxval=1.2)
            st, info = alg(logdensity_u, step_size=eps, inverse_mass_matrix=imm, **extra).step(k_step, st)
            return st, (st.position, info.acceptance_rate)

        _, (pos, acc) = jax.lax.scan(step, state, jax.random.split(k_draw, s.n_samples))
        return pos, jnp.mean(acc)

    pos, acc = jax.vmap(one_chain)(jax.random.split(key, init_u.shape[0]), init_u)
    return jnp.swapaxes(pos, 0, 1), jnp.mean(acc)


@jax.jit
def _diagnostics(samples):
    """(S, C, D) -> R-hat (D,), ESS (D,); jitted once per shape (blackjax builds closures per call)."""
    x = samples[..., None] if samples.ndim == 2 else samples
    return (potential_scale_reduction(x, chain_axis=1, sample_axis=0).reshape(x.shape[-1]),
            effective_sample_size(x, chain_axis=1, sample_axis=0).reshape(x.shape[-1]))


_batched_diagnostics = jax.jit(jax.vmap(_diagnostics))


def _quantiles(x: np.ndarray, axis) -> dict:
    p16, p50, p84 = np.percentile(x, [16, 50, 84], axis=axis)
    return {"median": p50, "lo": p50 - p16, "hi": p84 - p50}


def posterior_summary(samples_phys) -> dict:
    """Median, 16/84 half-widths, per-label R-hat and ESS from (S, C, D) samples."""
    x = np.asarray(samples_phys, np.float64)
    out = _quantiles(x.reshape(-1, x.shape[-1]), axis=0)
    r_hat, ess = _diagnostics(jnp.asarray(x, jnp.float32))
    return {**out, "r_hat": np.asarray(r_hat), "ess": np.asarray(ess)}


def _summaries(samples) -> dict:
    """(B, S, C, D) -> per-star summaries, each (B, D)."""
    x = np.asarray(samples, np.float64)
    out = _quantiles(x.reshape(x.shape[0], -1, x.shape[-1]), axis=1)
    r_hat, ess = _batched_diagnostics(jnp.asarray(x, jnp.float32))
    return {**out, "r_hat": np.asarray(r_hat), "ess": np.asarray(ess)}


def _chain_inits(k, base, scale, lo, hi, n_chains):
    noise = jax.random.normal(k, (n_chains,) + base.shape) * scale
    return to_unconstrained(base[None] + noise, lo, hi)


def _star_key(key, i, stage):
    return jax.random.fold_in(jax.random.fold_in(key, stage), i)


def _shard(body, mesh, n_rep: int, n_star: int):
    specs = (P(),) * n_rep + (P(AXIS),) * n_star
    return jax.jit(jax.shard_map(body, mesh=mesh, in_specs=specs, out_specs=P(AXIS), check_vma=False))


# Samplers are cached per (emulator fn, mesh, settings) so every star batch reuses
# the compiled executable; data, weights, stats and the PRNG key are arguments.
@functools.lru_cache(maxsize=None)
def _joint_sampler(fn, mesh, s: MCMCSettings):
    lower, upper = jnp.asarray(LOWER), jnp.asarray(UPPER)

    def body(st, mean, std, key, ids, obs, ivar, mu, sig):
        obj = Objectives(fn, st, mean, std)

        def one(i, o, iv, m, s_):
            def logd(u):
                return -obj.joint(u, o, iv, m, s_) + log_det_jacobian(u, lower, upper)

            k = _star_key(key, i, 0)
            init = _chain_inits(k, m, 0.05 * std[:23], lower, upper, s.n_chains)
            samples, acc = sample_star(logd, init, jax.random.fold_in(k, 1), s)
            return to_physical(samples, lower, upper), acc

        return jax.vmap(one)(ids, obs, ivar, mu, sig)

    return _shard(body, mesh, 4, 5)


@functools.lru_cache(maxsize=None)
def _core_sampler(fn, mesh, s: MCMCSettings):
    def body(st, mean, std, key, ids, obs, ivar, mu, sig):
        obj = Objectives(fn, st, mean, std)

        def one(i, o, iv, m, s_):
            def logd(u):
                return -obj.core(u, m[9:], o, iv, m[:9], s_[:9]) + log_det_jacobian(u, LO_CORE, HI_CORE)

            k = _star_key(key, i, 0)
            init = _chain_inits(k, m[:9], 0.05 * std[:9], LO_CORE, HI_CORE, s.n_chains)
            samples, acc = sample_star(logd, init, jax.random.fold_in(k, 1), s)
            return to_physical(samples, LO_CORE, HI_CORE), acc

        return jax.vmap(one)(ids, obs, ivar, mu, sig)

    return _shard(body, mesh, 4, 5)


@functools.lru_cache(maxsize=None)
def _elem_sampler(fn, mesh, s: MCMCSettings):
    lower, upper = jnp.asarray(LOWER), jnp.asarray(UPPER)

    def body(st, mean, std, key, j, pix, ids, full, obs, ivar, mu, sig):
        obj = Objectives(fn, st, mean, std)
        lo, hi = lower[j][None], upper[j][None]

        def one(i, f, o, iv, m, s_):
            def logd(u):
                return (-obj.element(u, j, f, o, iv, pix, m[j][None], s_[j][None], 1.0)
                        + log_det_jacobian(u, lo, hi))

            k = _star_key(key, i, 1 + j)
            init = _chain_inits(k, m[j][None], 0.02, lo, hi, s.n_chains)
            samples, acc = sample_star(logd, init, jax.random.fold_in(k, 1), s)
            return to_physical(samples, lo, hi), acc

        return jax.vmap(one)(ids, full, obs, ivar, mu, sig)

    return _shard(body, mesh, 6, 6)


def run_mcmc(flux_fn, state, obs_flux, obs_ivar, cnn_mu, cnn_sigma, element_masks: dict, skip: set, stats,
             mesh, key, *, mode: str = "two_stage", core: MCMCSettings = NUTS_CORE,
             elem: MCMCSettings = NUTS_ELEM, joint: MCMCSettings = NUTS_JOINT) -> dict:
    """Sample every star; returns median / lower_sigma / upper_sigma / r_hat / ess (N, 23),
    acceptance (N,) and the list of non-converged star rows (max R-hat > 1.05)."""
    if mode not in ("two_stage", "joint"):
        raise ValueError(f"unknown MCMC mode {mode!r}")
    n = len(obs_flux)
    nd = n_devices(mesh)
    pad = lambda a: pad_to_multiple(np.asarray(a, np.float32), nd)[0]  # noqa: E731
    obs, ivar, mu, sig = (jnp.asarray(pad(a)) for a in (obs_flux, obs_ivar, cnn_mu, cnn_sigma))
    star_ids = jnp.arange(obs.shape[0])
    rep = replicate((state, jnp.asarray(stats[0], jnp.float32), jnp.asarray(stats[1], jnp.float32), key), mesh)

    if mode == "joint":
        samples, acc = _joint_sampler(flux_fn, mesh, joint)(*rep, star_ids, obs, ivar, mu, sig)
        summ = _summaries(samples)
        out = {"median": summ["median"], "lower_sigma": summ["lo"], "upper_sigma": summ["hi"],
               "r_hat": summ["r_hat"], "ess": summ["ess"], "acceptance": np.asarray(acc)}
        return _finish(out, n)

    core_samples, core_acc = _core_sampler(flux_fn, mesh, core)(*rep, star_ids, obs, ivar, mu, sig)
    cs = _summaries(core_samples)
    median = np.array(pad(cnn_mu), np.float32, copy=True)
    lo_sig = np.array(pad(cnn_sigma), np.float32, copy=True)
    hi_sig = lo_sig.copy()
    r_hat = np.full(median.shape, np.nan, np.float32)
    ess = np.full(median.shape, np.nan, np.float32)
    median[:, :9], lo_sig[:, :9], hi_sig[:, :9] = cs["median"], cs["lo"], cs["hi"]
    r_hat[:, :9], ess[:, :9] = cs["r_hat"], cs["ess"]
    accs = [np.asarray(core_acc)]
    elem_run = _elem_sampler(flux_fn, mesh, elem)
    for j in ABUND_INDICES:
        if j in skip:
            continue
        samples, acc = elem_run(*rep, jnp.int32(j), jnp.asarray(element_masks[j], jnp.float32),
                                star_ids, jnp.asarray(median), obs, ivar, mu, sig)
        es = _summaries(samples)
        median[:, j], lo_sig[:, j], hi_sig[:, j] = es["median"][:, 0], es["lo"][:, 0], es["hi"][:, 0]
        r_hat[:, j], ess[:, j] = es["r_hat"][:, 0], es["ess"][:, 0]
        accs.append(np.asarray(acc))
    acceptance = 0.5 * (accs[0] + np.mean(accs[1:], axis=0)) if len(accs) > 1 else accs[0]
    out = {"median": median, "lower_sigma": lo_sig, "upper_sigma": hi_sig, "r_hat": r_hat, "ess": ess,
           "acceptance": acceptance}
    return _finish(out, n)


def _finish(out: dict, n: int) -> dict:
    out = {k: np.asarray(v)[:n] for k, v in out.items()}
    worst = np.nanmax(np.where(np.isnan(out["r_hat"]), -np.inf, out["r_hat"]), axis=1)
    out["nonconverged"] = [int(i) for i in np.where(worst > RHAT_LIMIT)[0]]
    for i in out["nonconverged"]:
        print(f"  warning: star row {i} max R-hat {worst[i]:.3f} > {RHAT_LIMIT} (not converged)")
    return out
