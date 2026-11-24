import jax
import jax.numpy as jnp
import numpy as np

from prism.inverse import common as C


def test_bounds_shape_and_vsini():
    assert C.LOWER.shape == (23,) and C.UPPER.shape == (23,) and (C.UPPER > C.LOWER).all()
    assert C.LOWER[5] == 0.0  # VSINI >= 0


def test_features_27_matches_numpy_pipeline():
    from prism.data.labels import engineer_features

    x = np.random.default_rng(0).uniform(0.5, 2, (3, 23))
    x[:, 0] = [4000, 5000, 6000]
    assert np.allclose(np.asarray(C.features_27(jnp.asarray(x))), engineer_features(x), rtol=1e-5)


def test_roundtrip_inside_bounds():
    x = (C.LOWER + C.UPPER) / 2 + 0.1 * (C.UPPER - C.LOWER) * np.linspace(-1, 1, 23)
    u = C.to_unconstrained(x, C.LOWER, C.UPPER)
    assert np.allclose(np.asarray(C.to_physical(u, C.LOWER, C.UPPER)), x, rtol=1e-4, atol=1e-4)


def test_to_unconstrained_clips_out_of_bounds():
    x = C.UPPER + 10.0
    x[0] = C.LOWER[0] - 500
    u = np.asarray(C.to_unconstrained(x, C.LOWER, C.UPPER))
    p = np.asarray(C.to_physical(u, C.LOWER, C.UPPER))
    assert np.isfinite(u).all() and (p >= C.LOWER).all() and (p <= C.UPPER).all()


def test_log_det_jacobian():
    u = jnp.asarray(np.random.default_rng(1).normal(size=23), jnp.float32)
    jac = jax.jacfwd(lambda v: C.to_physical(v, C.LOWER, C.UPPER))(u)
    expect = np.sum(np.log(np.abs(np.diag(np.asarray(jac)))))
    assert np.isclose(float(C.log_det_jacobian(u, C.LOWER, C.UPPER)), expect, rtol=1e-5)


def test_loglik_ignores_bad_pixels():
    rng = np.random.default_rng(2)
    obs = rng.uniform(0.6, 1.0, (2, 8575)).astype(np.float32)
    ivar = rng.uniform(100, 1000, (2, 8575)).astype(np.float32)
    model = obs + 0.01
    obs[0, :5] = np.nan
    ivar[0, 5:10] = 0.0
    obs[1, :5] = 1.5
    a = C.log_likelihood(model, obs, ivar)
    model2 = model.copy()
    model2[0, :10] = 7.0  # changing the model on bad pixels must not matter
    model2[1, :5] = 7.0
    b = C.log_likelihood(model2, obs, ivar)
    assert np.isfinite(np.asarray(a)).all() and np.allclose(np.asarray(a), np.asarray(b))
    g = jax.grad(lambda m: C.log_likelihood(m, obs, ivar).sum())(jnp.asarray(model))
    assert np.isfinite(np.asarray(g)).all()
    pm = np.zeros(8575, np.float32)
    pm[100:200] = 1
    assert np.allclose(np.asarray(C.log_likelihood(model2, obs, ivar, pm)),
                       np.asarray(C.log_likelihood(model, obs, ivar, pm)))


def test_gaussian_prior():
    assert np.isclose(float(C.gaussian_prior(jnp.array([1.0, 3.0]), jnp.array([0.0, 1.0]),
                                             jnp.array([1.0, 2.0]), weight=2.0)), -0.5 * 2 * (1 + 1))


def test_select_test_stars():
    rng = np.random.default_rng(3)
    n = 3000
    teff, logg, mh = rng.uniform(3500, 7000, n), rng.uniform(0, 5, n), rng.uniform(-2, 0.5, n)
    snr = rng.uniform(50, 300, n)
    teff[0] = -9999.0
    all_idx = C.select_test_stars(snr, teff, logg, mh, None)
    assert len(all_idx) == n - 1 and 0 not in all_idx
    a = C.select_test_stars(snr, teff, logg, mh, 5, seed=1)
    assert np.array_equal(a, C.select_test_stars(snr, teff, logg, mh, 5, seed=1)) and len(a) == 5
    edges = np.percentile(teff[1:], np.linspace(0, 100, 11))
    assert len(set(np.digitize(teff[a], edges))) >= 4
    big = C.select_test_stars(snr, teff, logg, mh, 2000, seed=1)
    assert len(big) == 2000 and len(set(big.tolist())) == 2000


def test_results_save_and_checkpoint(tmp_path):
    r = C.InverseResults.empty()
    part = C.InverseResults(global_indices=np.array([10, 11]), true_labels=np.zeros((2, 23)),
                            inferred_labels=np.ones((2, 23)), aspcap_errors=np.ones((2, 23)),
                            true_label_imputed=np.zeros((2, 23), bool), wall_seconds=np.ones(2),
                            lower_sigma=np.ones((2, 23)), upper_sigma=np.ones((2, 23)))
    r = r.append(part)
    ck = C.ResultsCheckpoint(tmp_path / "ck.npz", expected_indices=np.array([10, 11, 12]))
    ck.save(r)
    assert ck.load().global_indices.tolist() == [10, 11]
    bad = C.ResultsCheckpoint(tmp_path / "ck.npz", expected_indices=np.array([11, 12, 13]))
    assert bad.load() is None
    r.save(tmp_path / "results.npz")
    d = np.load(tmp_path / "results.npz")
    assert {"global_indices", "inferred_labels", "lower_sigma", "label_names"} <= set(d.files)
    assert "inferred_errors" not in d.files  # unset optional fields are not written


def test_cnn_prior_physical_units():
    from prism.inverse.cnn_prior import predict_physical
    from prism.models.registry import build_model

    cnn = build_model("cnn", None, seed=0)
    mean, std = np.arange(23, dtype=np.float32), np.full(23, 2.0, np.float32)
    flux = np.ones((4, 8575), np.float32)
    flux[0, :10] = np.nan
    mu, sd = predict_physical(cnn, flux, mean, std)
    assert mu.shape == (4, 23) and sd.shape == (4, 23)
    assert np.isfinite(mu).all() and (sd >= 0.01 * std - 1e-7).all()


def test_make_flux_fn_reexported():
    from prism.models.functional import make_flux_fn

    assert C.make_flux_fn is make_flux_fn
