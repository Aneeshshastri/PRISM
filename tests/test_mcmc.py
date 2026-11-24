import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats as sps
from toy import make_problem

from prism.mesh import make_mesh


@pytest.mark.parametrize("kernel", ["nuts", "hmc"])
def test_gaussian_target(kernel):
    from prism.inverse.common import to_physical
    from prism.inverse.mcmc import MCMCSettings, sample_star, unconstrained_logdensity

    lo, hi = jnp.array([-50.0, -50.0]), jnp.array([50.0, 50.0])
    mu, sd = jnp.array([1.0, -2.0]), jnp.array([1.0, 0.5])
    logp = unconstrained_logdensity(lambda x: -0.5 * jnp.sum(((x - mu) / sd) ** 2), lo, hi)
    s = MCMCSettings(kernel, 4, 300, 500, 0.8, n_leapfrog=10)
    init = jnp.zeros((4, 2))
    samples_u, acc = jax.jit(lambda k: sample_star(logp, init, k, s))(jax.random.key(0))
    x = np.asarray(to_physical(samples_u, lo, hi)).reshape(-1, 2)
    assert samples_u.shape == (500, 4, 2) and 0 < float(acc) <= 1
    assert np.allclose(x.mean(0), [1.0, -2.0], atol=0.1)
    assert np.allclose(x.std(0), [1.0, 0.5], rtol=0.15)


def test_jacobian_makes_flat_target_uniform():
    from prism.inverse.common import to_physical
    from prism.inverse.mcmc import MCMCSettings, sample_star, unconstrained_logdensity

    lo, hi = jnp.array([0.0]), jnp.array([1.0])
    logp = unconstrained_logdensity(lambda x: jnp.zeros(()), lo, hi)  # flat in physical space
    s = MCMCSettings("nuts", 4, 300, 1000, 0.8)
    samples_u, _ = jax.jit(lambda k: sample_star(logp, jnp.zeros((4, 1)), k, s))(jax.random.key(1))
    x = np.asarray(to_physical(samples_u, lo, hi)).ravel()[::5]  # thin to reduce autocorrelation
    assert sps.kstest(x, "uniform").pvalue > 0.01


def test_posterior_summary_per_star_rhat():
    from prism.inverse.mcmc import posterior_summary

    rng = np.random.default_rng(0)
    good = rng.normal(0, 1, (400, 4, 2))
    bad = good.copy()
    bad[:, 2:, 0] += 5.0  # two chains stuck in another mode, label 0 only
    s_good, s_bad = posterior_summary(good), posterior_summary(bad)
    assert (s_good["r_hat"] < 1.05).all()
    assert s_bad["r_hat"][0] > 1.05 and s_bad["r_hat"][1] < 1.05
    assert np.allclose(s_good["median"], np.median(good.reshape(-1, 2), 0))
    assert np.allclose(s_good["lo"], s_good["median"] - np.percentile(good.reshape(-1, 2), 16, 0))
    assert s_good["ess"].shape == (2,)


@pytest.fixture(scope="module")
def toy_obs():
    from prism.inverse.common import features_27, make_flux_fn

    model, masks, theta, mean, std = make_problem(3, seed=4)
    fn, state = make_flux_fn(model)
    flux = np.asarray(fn(state, (np.asarray(features_27(theta)) - mean) / std))
    ivar = np.full_like(flux, 400.0)
    return fn, state, masks, theta, mean, std, flux, ivar


def _run(mode, mesh, toy_obs, settings, skip=frozenset()):
    from prism.inverse.mcmc import run_mcmc

    fn, state, masks, theta, mean, std, flux, ivar = toy_obs
    return run_mcmc(fn, state, flux, ivar, theta + 0.01, np.full_like(theta, 0.5), masks, set(skip),
                    (mean, std), mesh, jax.random.key(0), mode=mode,
                    core=settings[0], elem=settings[1], joint=settings[2])


def _quick_settings():
    from prism.inverse.mcmc import HMC_CORE, HMC_ELEM, NUTS_JOINT

    q = lambda s: dataclasses.replace(s, n_chains=2, n_warmup=20, n_samples=20)  # noqa: E731
    return q(HMC_CORE), q(HMC_ELEM), q(NUTS_JOINT)


def test_run_mcmc_outputs_and_determinism(toy_obs):
    a = _run("two_stage", make_mesh(), toy_obs, _quick_settings(), skip={14})
    b = _run("two_stage", make_mesh(), toy_obs, _quick_settings(), skip={14})
    for k in ("median", "lower_sigma", "upper_sigma", "r_hat", "ess"):
        assert a[k].shape == (3, 23), k
    assert a["acceptance"].shape == (3,) and isinstance(a["nonconverged"], list)
    assert np.isnan(a["r_hat"][:, 14]).all() and np.allclose(a["median"][:, 14], toy_obs[3][:, 14] + 0.01)
    assert np.array_equal(a["median"], b["median"])  # same mesh + key -> identical


def test_run_mcmc_joint(toy_obs):
    out = _run("joint", make_mesh(), toy_obs, _quick_settings())
    assert out["median"].shape == (3, 23) and np.isfinite(out["median"]).all()


@pytest.mark.slow
def test_run_mcmc_converges_and_is_mesh_consistent(toy_obs):
    """Converged NUTS recovers the truth and 1- vs 2-device runs agree statistically.

    (Bit-exact equality across meshes is not expected: per-device batch shapes change
    float rounding, which MCMC accept/reject amplifies.)
    """
    from prism.inverse.mcmc import NUTS_CORE, NUTS_ELEM, NUTS_JOINT

    q = lambda s: dataclasses.replace(s, n_chains=2, n_warmup=100, n_samples=100)  # noqa: E731
    settings = (q(NUTS_CORE), q(NUTS_ELEM), q(NUTS_JOINT))
    a = _run("two_stage", make_mesh(), toy_obs, settings, skip={14})
    b = _run("two_stage", make_mesh(jax.devices()[:1]), toy_obs, settings, skip={14})
    theta = toy_obs[3]
    width = np.maximum(np.maximum(a["upper_sigma"], a["lower_sigma"]), 1e-9)
    assert np.nanmax(a["r_hat"]) < 1.05 and a["nonconverged"] == []
    assert np.max(np.abs(a["median"] - theta) / width) < 3.0
    assert np.max(np.abs(a["median"] - b["median"]) / width) < 1.0


def test_mcmc_does_not_recompile_per_batch(toy_obs):
    from test_map import _count_compiles

    settings = _quick_settings()
    _run("two_stage", make_mesh(), toy_obs, settings)
    assert _count_compiles(lambda: _run("two_stage", make_mesh(), toy_obs, settings)) == 0
