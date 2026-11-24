import h5py
import numpy as np

from prism.config import SELECTED_LABELS


def _raw(synthetic_h5, start=0, end=60):
    from prism.data.labels import read_raw_flagged

    with h5py.File(synthetic_h5, "r") as h5:
        return read_raw_flagged(h5, SELECTED_LABELS, start, end)


def test_flags(synthetic_h5, planted):
    raw, bad = _raw(synthetic_h5)
    assert raw.dtype == np.float64 and bad.dtype == bool and bad.shape == (60, 23)
    assert bad[planted["i_mh"], 2]   # M_H -9999 sentinel (no M_H_FLAG column)
    assert bad[planted["i_o"], 8]    # O_FE NaN
    assert bad[planted["i_mg"], 10]  # MG_FE_FLAG != 0
    assert bad.sum() == 3


def test_x_h_and_features():
    from prism.data.labels import engineer_features, to_x_h

    x = np.zeros((1, 23))
    x[0, 0] = 5040.0
    x[0, 4], x[0, 5] = 3.0, 4.0
    x[0, 6], x[0, 8] = 0.2, 0.1
    x[0, 1], x[0, 2] = 2.0, -1.0
    x[0, 9], x[0, 10] = -0.5, 0.3
    xh = to_x_h(x)
    assert x[0, 10] == 0.3  # input not modified
    f = engineer_features(xh)
    assert f.shape == (1, 27)
    assert np.isclose(f[0, 10], -0.2) and np.isclose(f[0, 6], -0.3) and f[0, 9] == -0.5
    # C_O_diff uses [X/H]: (.2-.5)-(.1-.5) = 0.1
    assert np.allclose(f[0, 23:], [1.0, 5.0, 0.1, 0.5])


def test_imputer_train_only(synthetic_h5):
    from prism.data.labels import fit_imputer, impute

    raw, bad = _raw(synthetic_h5)
    imp = fit_imputer(raw[:30], bad[:30], 42)
    out = impute(imp, raw[30:], bad[30:])
    assert out.shape == (30, 23) and np.isfinite(out).all()
    full = impute(imp, raw, bad)
    assert np.isfinite(full).all() and full[3, 2] > -10  # sentinel replaced
    assert np.allclose(full[~bad], raw[~bad])


def test_norm_stats():
    from prism.data.labels import norm_stats

    a = np.array([[1.0, 5.0], [3.0, 5.0]])
    mean, std = norm_stats(a)
    assert np.allclose(mean, [2.0, 5.0]) and np.allclose(std, [1.0, 1.0])  # zero std -> 1


def test_weights_val_uses_train_density():
    from prism.data.labels import stratified_weights

    rng = np.random.default_rng(0)
    tr = rng.normal(size=(2000, 27))
    w_tr = stratified_weights(tr)
    assert np.allclose(stratified_weights(tr[:5], ref_labels_norm27=tr), w_tr[:5])
    assert abs(w_tr.mean() - 1) < 1e-5 and w_tr.dtype == np.float32


def test_errors_sentinel(synthetic_h5, planted):
    from prism.data.labels import read_errors

    with h5py.File(synthetic_h5, "r") as h5:
        e = read_errors(h5, SELECTED_LABELS, 0, 60)
        good = h5["metadata"]["TEFF_ERR"].astype(np.float64)
    assert e.shape == (60, 23) and (e >= 0).all() and np.isfinite(e).all()
    good = np.delete(good, planted["i_teff_err"])
    assert np.isclose(e[planted["i_teff_err"], 0], np.percentile(good, 95))
    assert (e[:, 3:6] == 0).all()  # no VMICRO/VMACRO/VSINI errors
