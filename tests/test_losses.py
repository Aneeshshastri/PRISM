import numpy as np

CUT = 1e-3


def _np_spl(pred, flux, ivar, pm, w, rm):
    with np.errstate(all="ignore"):
        wt = np.where(ivar > 0, 1e-5 / (1 / ivar + 1e-5), 0.0)
        valid = (flux > CUT).astype(float)
        safe = np.where(valid > 0, flux, pred)
        term = (safe - pred) ** 2 * wt * valid * pm
    term = np.where(np.isfinite(term), term, 0.0)
    return (term * w[:, None] * rm[:, None]).sum() / (max(rm.sum(), 1) * flux.shape[1])


def _data(rng, b=4, p=8575):
    flux = rng.uniform(0.5, 1.1, (b, p))
    flux[0, :20] = 0.0  # below cutoff
    ivar = rng.uniform(10, 1e4, (b, p))
    ivar[1, :30] = 0.0
    pred = rng.uniform(0.5, 1.1, (b, p))
    pm = (rng.random((b, p)) > 0.1).astype(float)
    w = rng.uniform(0.5, 2, b)
    rm = np.ones(b)
    return pred, flux, ivar, pm, w, rm


def test_masked_spl_matches_numpy():
    from prism.losses import masked_spl_loss

    args = _data(np.random.default_rng(0))
    got = float(masked_spl_loss(*[a.astype(np.float32) for a in args]))
    assert np.isclose(got, _np_spl(*args), rtol=1e-4)


def test_losses_ignore_nonfinite_pixels():
    from prism.losses import masked_spl_loss

    pred, flux, ivar, pm, w, rm = _data(np.random.default_rng(1))
    clean = float(masked_spl_loss(pred, flux, ivar, pm * 1.0, w, rm))
    flux2, ivar2, pm2 = flux.copy(), ivar.copy(), pm.copy()
    flux2[2, 100:110] = np.nan
    flux2[2, 110:120] = np.inf
    ivar2[3, 200:210] = np.nan
    dirty = float(masked_spl_loss(pred, flux2, ivar2, pm2, w, rm))
    pm3 = pm.copy()
    pm3[2, 100:120] = 0
    pm3[3, 200:210] = 0
    dropped = float(masked_spl_loss(pred, flux, ivar, pm3, w, rm))
    assert np.isfinite(dirty) and np.isclose(dirty, dropped, rtol=1e-5) and dirty <= clean


def test_padded_rows_ignored():
    from prism.losses import masked_spl_loss

    pred, flux, ivar, pm, w, rm = _data(np.random.default_rng(2))
    base = float(masked_spl_loss(pred, flux, ivar, pm, w, rm))
    garbage = np.random.default_rng(3).uniform(0, 5, (2, 8575))
    cat = lambda a, g: np.concatenate([a, g])  # noqa: E731
    padded = float(masked_spl_loss(cat(pred, garbage), cat(flux, garbage), cat(ivar, garbage),
                                   cat(pm, np.ones((2, 8575))), cat(w, [9.0, 9.0]), cat(rm, [0.0, 0.0])))
    assert np.isclose(base, padded, rtol=1e-6)


def test_reduced_chi2_matches_numpy():
    from prism.losses import reduced_chi2

    pred, flux, ivar, *_ = _data(np.random.default_rng(4))
    valid = ((ivar > 0) & (flux > CUT)).astype(float)
    with np.errstate(divide="ignore"):
        var = np.where(ivar > 0, 1 / ivar, 0.0) + 1e-5
    expect = ((pred - flux) ** 2 / var * valid).sum(1) / np.maximum(valid.sum(1) - 23, 1)
    assert np.allclose(np.asarray(reduced_chi2(pred, flux, ivar)), expect, rtol=1e-4)
    extra = np.ones_like(flux)
    extra[:, :100] = 0
    v2 = valid * extra
    expect2 = ((pred - flux) ** 2 / var * v2).sum(1) / np.maximum(v2.sum(1) - 23, 1)
    assert np.allclose(np.asarray(reduced_chi2(pred, flux, ivar, valid_extra=extra)), expect2, rtol=1e-4)


def test_nll_minimum():
    from prism.losses import heteroscedastic_nll

    sigma2 = 0.04
    raw = np.log(np.expm1(sigma2 - 1e-5))  # softplus^-1
    y = np.random.default_rng(5).normal(size=(6, 23))
    got = float(heteroscedastic_nll(y, y, np.full_like(y, raw), np.ones(6)))
    assert np.isclose(got, 0.5 * np.log(sigma2), rtol=1e-5)


def test_gradient_finite_with_nonfinite_data():
    import jax

    from prism.losses import masked_spl_loss

    pred, flux, ivar, pm, w, rm = _data(np.random.default_rng(6))
    flux[0, :50] = np.nan
    ivar[1, :50] = np.inf
    g = jax.grad(lambda p: masked_spl_loss(p, flux, ivar, pm, w, rm))(pred.astype(np.float32))
    assert np.isfinite(np.asarray(g)).all()
