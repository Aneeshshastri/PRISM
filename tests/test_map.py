import jax
import numpy as np
import pytest
from toy import make_problem

from prism.inverse.common import features_27, make_flux_fn
from prism.mesh import make_mesh


def _observe(model, theta, mean, std):
    fn, state = make_flux_fn(model)
    flux = np.asarray(fn(state, (np.asarray(features_27(theta)) - mean) / std))
    ivar = np.full_like(flux, 1e4)
    return fn, state, flux, ivar


def _run(mode, n=4, mesh=None, seed=0, **kw):
    from prism.data.masks import element_prior_weights
    from prism.inverse.map import run_map

    model, masks, theta, mean, std = make_problem(n, seed)
    fn, state, flux, ivar = _observe(model, theta, mean, std)
    rng = np.random.default_rng(seed + 1)
    cnn_mu = theta + rng.normal(0, 0.05, theta.shape).astype(np.float32)
    cnn_mu[:, 0] = theta[:, 0] + rng.normal(0, 150, n)
    cnn_sigma = np.full_like(theta, 1e3)  # effectively flat prior
    out = run_map(fn, state, flux, ivar, cnn_mu, cnn_sigma, masks, set(), element_prior_weights(masks),
                  (mean, std), mesh or make_mesh(), mode=mode, **kw)
    return out, theta


@pytest.mark.parametrize("mode", ["two_stage", "joint"])
def test_map_recovers_truth(mode):
    out, theta = _run(mode)
    got = out["theta"]
    assert got.shape == theta.shape and out["core_loss"].shape == (4,) and out["elem_loss"].shape == (4, 14)
    assert np.abs(got[:, 0] - theta[:, 0]).max() < 5.0
    assert np.abs(got[:, 1:] - theta[:, 1:]).max() < 1e-2


def test_map_odd_star_count():
    a, _ = _run("two_stage", n=5, steps_core=200, steps_elem=50)
    b, _ = _run("two_stage", n=5, mesh=make_mesh(jax.devices()[:1]), steps_core=200, steps_elem=50)
    assert a["theta"].shape == (5, 23)
    assert np.allclose(a["theta"], b["theta"], rtol=1e-5, atol=1e-5)


def test_map_skipped_element_keeps_cnn_value():
    from prism.data.masks import element_prior_weights
    from prism.inverse.map import run_map

    model, masks, theta, mean, std = make_problem(2, 0)
    fn, state, flux, ivar = _observe(model, theta, mean, std)
    cnn_mu = theta + 0.05
    out = run_map(fn, state, flux, ivar, cnn_mu, np.full_like(theta, 1e3), masks, {14},
                  element_prior_weights(masks), (mean, std), make_mesh(), mode="two_stage",
                  steps_core=50, steps_elem=20)
    assert np.array_equal(out["theta"][:, 14], cnn_mu[:, 14])


def _count_compiles(fn):
    import jax.monitoring as mon

    events = []
    cb = lambda name, *a, **k: events.append(name) if "backend_compile" in name else None  # noqa: E731
    mon.register_event_duration_secs_listener(cb)
    try:
        fn()
    finally:
        mon.unregister_event_duration_listener(cb)
    return len(events)


def test_map_and_laplace_do_not_recompile_per_batch():
    """Review finding: jitted closures were rebuilt per call -> recompiles on every star batch."""
    from prism.data.masks import element_prior_weights
    from prism.inverse.laplace import laplace_sigmas
    from prism.inverse.map import run_map

    model, masks, theta, mean, std = make_problem(2, 0)
    fn, state, flux, ivar = _observe(model, theta, mean, std)
    pw = element_prior_weights(masks)
    mesh = make_mesh()

    def once(shift):
        mu = theta + shift
        out = run_map(fn, state, flux, ivar, mu, np.full_like(theta, 1e3), masks, set(), pw, (mean, std), mesh,
                      steps_core=20, steps_elem=5)
        laplace_sigmas(fn, state, out["theta"], flux, ivar, mu, np.full_like(theta, 1.0), masks, set(), pw,
                       (mean, std), mesh)

    once(0.01)  # warm-up compiles
    assert _count_compiles(lambda: once(0.02)) == 0
