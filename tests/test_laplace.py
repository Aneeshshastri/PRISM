import jax
import jax.numpy as jnp
import numpy as np
import pytest
from toy import make_problem

from prism.inverse.common import IVAR_FLOOR_VAR, features_27, make_flux_fn
from prism.mesh import make_mesh


def test_stabilised_inverse_diagonal():
    from prism.inverse.laplace import stabilised_inverse

    cov = np.asarray(stabilised_inverse(jnp.diag(jnp.array([1e6, 50.0, 2.0]))))
    sig = np.sqrt(np.diag(cov))
    assert np.allclose(sig, 1 / np.sqrt([1e6, 50.0, 2.0]), rtol=1e-3)


@pytest.fixture(scope="module")
def problem():
    model, masks, theta, mean, std = make_problem(3, seed=2)
    fn, state = make_flux_fn(model)
    flux = np.asarray(fn(state, (np.asarray(features_27(theta)) - mean) / std))  # noiseless
    ivar = np.full_like(flux, 400.0)
    sig = np.full_like(theta, 0.3)
    sig[:, 0] = 300.0
    return fn, state, masks, theta, mean, std, flux, ivar, sig


def _jac(fn, state, mean, std, theta_star):
    f = lambda t: fn(state, ((features_27(t) - mean) / std)[None])[0]  # noqa: E731
    return np.asarray(jax.jacfwd(f)(jnp.asarray(theta_star)), np.float64)  # (8575, 23)


def _analytic(J, w_pix, prior_prec):
    H = J.T @ (J * w_pix[:, None]) + np.diag(prior_prec)
    return np.sqrt(np.diag(np.linalg.inv(H)))


@pytest.mark.parametrize("mode", ["two_stage", "joint"])
def test_laplace_matches_linear_gaussian(problem, mode):
    from prism.data.masks import element_prior_weights
    from prism.inverse.laplace import laplace_sigmas

    fn, state, masks, theta, mean, std, flux, ivar, sig = problem
    pw = element_prior_weights(masks)
    # prior centred on the truth + noiseless data => the truth is the exact MAP
    got = laplace_sigmas(fn, state, theta, flux, ivar, theta, sig, masks, set(), pw, (mean, std),
                         make_mesh(), mode=mode)
    assert got.shape == theta.shape
    for s in range(len(theta)):
        # the likelihood only uses pixels with 1e-3 < flux < 1.3
        valid = (flux[s] > 1e-3) & (flux[s] < 1.3)
        w = valid / (1.0 / ivar[s] + IVAR_FLOOR_VAR)
        J = _jac(fn, state, mean, std, theta[s])
        if mode == "joint":
            expect = _analytic(J, w, 1 / sig[s] ** 2)
            assert np.allclose(got[s], expect, rtol=0.02)
        else:
            expect_core = _analytic(J[:, :9], w, 1 / sig[s, :9] ** 2)
            assert np.allclose(got[s, :9], expect_core, rtol=0.02)
            for j in range(9, 23):
                e = 1 / np.sqrt((J[:, j] ** 2 * w * masks[j]).sum() + pw[j] / sig[s, j] ** 2)
                assert np.isclose(got[s, j], e, rtol=0.02), j


def test_laplace_skipped_element_uses_cnn_sigma(problem):
    from prism.data.masks import element_prior_weights
    from prism.inverse.laplace import laplace_sigmas

    fn, state, masks, theta, mean, std, flux, ivar, sig = problem
    got = laplace_sigmas(fn, state, theta, flux, ivar, theta, sig, masks, {14},
                         element_prior_weights(masks), (mean, std), make_mesh(), mode="two_stage")
    assert np.array_equal(got[:, 14], sig[:, 14])
