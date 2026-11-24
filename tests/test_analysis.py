import numpy as np
import pytest

from prism.config import SELECTED_LABELS

STEMS = ["map_one_to_one", "map_residual_histograms", "map_bias_scatter_bars", "map_residual_vs_teff",
         "map_residual_heatmap", "map_wall_time", "map_rmse_vs_aspcap"]


def _write(path, kind, n=60, seed=0):
    rng = np.random.default_rng(seed)
    true = rng.normal(size=(n, 23))
    true[:, 0] = rng.uniform(4000, 6000, n)
    sigma = np.full((n, 23), 0.05)
    sigma[:, 0] = 50.0
    inferred = true + rng.normal(size=true.shape) * sigma
    d = dict(global_indices=np.arange(n), true_labels=true, inferred_labels=inferred,
             aspcap_errors=np.abs(rng.normal(0.02, 0.005, (n, 23))), wall_seconds=rng.uniform(0.1, 1, n),
             true_label_imputed=np.zeros((n, 23), bool), label_names=np.array(SELECTED_LABELS))
    if kind == "laplace":
        d["inferred_errors"] = sigma
    if kind == "mcmc":
        d["lower_sigma"] = sigma
        d["upper_sigma"] = sigma
        d["r_hat"] = np.ones((n, 23))
    np.savez(path, **d)
    return path


@pytest.mark.parametrize("kind", ["map", "laplace", "mcmc"])
def test_report_and_figures(tmp_path, kind):
    from prism.analysis import load_results, report, visualise

    data = load_results(_write(tmp_path / f"{kind}.npz", kind))
    rep = report(data, print_report=False)
    assert set(rep) == set(SELECTED_LABELS)
    figs = visualise(data, save_dir=tmp_path / "figs", close=True)
    assert len(figs) == 7
    for stem in STEMS:
        assert (tmp_path / "figs" / f"{stem}.png").exists()
    if kind == "map":
        assert np.isnan(rep["TEFF"]["ratio"]) and np.isnan(rep["MG_FE"]["sigma_rmse"])
    else:
        assert np.isfinite(rep["TEFF"]["ratio"])


def test_calibration_unit_z(tmp_path):
    from prism.analysis import calibration, load_results

    data = load_results(_write(tmp_path / "l.npz", "laplace", n=4000, seed=1))
    cal = calibration(data)
    for name in ("TEFF", "MG_FE", "CO_FE"):
        assert abs(cal[name]["z_std"] - 1) < 0.15 and abs(cal[name]["frac_within_1sigma"] - 0.683) < 0.05


def test_calibration_requires_errors(tmp_path):
    from prism.analysis import calibration, load_results

    with pytest.raises(ValueError):
        calibration(load_results(_write(tmp_path / "m.npz", "map")))


def test_flag_outliers(tmp_path):
    from prism.analysis import flag_outliers, load_results, report_clean_vs_outlier

    path = _write(tmp_path / "o.npz", "map", n=200)
    d = dict(np.load(path))
    d["inferred_labels"][7, 3] += 100.0
    np.savez(path, **d)
    data = load_results(path)
    is_out, info = flag_outliers(data, sigma_thresh=10.0)
    assert is_out[7] and info["VMICRO"]["n_flagged"] >= 1
    report_clean_vs_outlier(data, is_out)  # runs


def test_cli(tmp_path):
    import subprocess
    import sys

    path = _write(tmp_path / "c.npz", "mcmc")
    out = subprocess.run([sys.executable, "-m", "prism.analysis", "--results", str(path),
                          "--save-dir", str(tmp_path / "cli")], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert (tmp_path / "cli" / "map_one_to_one.png").exists()
