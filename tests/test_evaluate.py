import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx


class _Echo(nnx.Module):
    """'Perfect' model: returns a fixed spectrum table looked up by an id stored in label col 0."""

    def __init__(self, table):
        self.table = nnx.Variable(jnp.asarray(table, jnp.float32))

    def __call__(self, labels, bad_mask=None):
        return self.table[...][labels[:, 0].astype(jnp.int32)]


@pytest.fixture(scope="module")
def element_idx(synthetic_masks):
    from prism.data.masks import element_indices, load_apogee_mask

    return element_indices(load_apogee_mask(synthetic_masks), 0.01)


def _test_set(n=10, seed=0):
    rng = np.random.default_rng(seed)
    flux = rng.uniform(0.6, 1.0, (n, 8575)).astype(np.float32)
    ivar = rng.uniform(100, 1000, (n, 8575)).astype(np.float32)
    labels = rng.normal(size=(n, 27)).astype(np.float32)
    labels[:, 0] = np.arange(n)
    bad = np.zeros((n, 23), bool)
    return {"labels": labels, "flux": flux, "ivar": ivar, "bad_mask": bad}


def test_predict_matches_model_and_pads():
    from prism.evaluate import predict
    from prism.mesh import make_mesh

    t = _test_set(7)
    out = predict(_Echo(t["flux"]), t["labels"], make_mesh(), batch=4)
    assert out.shape == (7, 8575) and np.array_equal(out, t["flux"])


def test_perfect_model_chi2_zero(synthetic_masks):
    from prism.data.masks import load_apogee_mask
    from prism.evaluate import chi2_table
    from prism.mesh import make_mesh

    t = _test_set()
    res = chi2_table({"perfect": _Echo(t["flux"])}, t, load_apogee_mask(synthetic_masks), make_mesh())
    assert np.allclose(res["perfect"]["chi2"], 0) and np.allclose(res["perfect"]["chi2_masked"], 0)


def test_masked_chi2_drops_core_cno_feh_stars(synthetic_masks):
    from prism.data.masks import load_apogee_mask
    from prism.evaluate import chi2_table
    from prism.mesh import make_mesh

    t = _test_set()
    t["bad_mask"][1, 0] = True   # TEFF -> dropped
    t["bad_mask"][2, 7] = True   # N_FE -> dropped
    t["bad_mask"][3, 9] = True   # FE_H -> dropped
    t["bad_mask"][4, 10] = True  # MG_FE -> kept, MG pixels masked
    am = load_apogee_mask(synthetic_masks)
    model = _Echo(t["flux"] + 0.01)
    res = chi2_table({"m": model}, t, am, make_mesh())
    assert res["valid_stars"].tolist() == [True, False, False, False] + [True] * 6
    assert len(res["m"]["chi2_masked"]) == 7
    # star 4 with MG pixels removed has a different chi2 than its unmasked value
    assert not np.isclose(res["m"]["chi2_masked"][1], res["m"]["chi2"][4])


def test_imputed_vs_natural():
    from prism.evaluate import imputed_vs_natural

    chi2 = np.array([1.0, 2.0, 3.0, 4.0])
    bad = np.zeros((4, 23), bool)
    bad[[0, 1], 10] = True
    rows = {r["label"]: r for r in imputed_vs_natural(chi2, bad)}
    assert rows["MG_FE"]["n_imputed"] == 2 and np.isclose(rows["MG_FE"]["ratio"], 1.5 / 3.5)
    assert "TEFF" not in rows


def test_leakage_prism_zero_payne_positive(element_idx, synthetic_masks):
    from prism.data.masks import load_apogee_mask
    from prism.evaluate import leakage
    from prism.models.registry import build_model

    am = load_apogee_mask(synthetic_masks)
    rng = np.random.default_rng(0)
    x = rng.normal(size=(6, 27)).astype(np.float32)
    bad = np.zeros((6, 23), bool)
    std = np.ones(27, np.float32)
    lp = leakage(build_model("prism", element_idx, seed=0), x, bad, am, std, batch=4)
    lq = leakage(build_model("payne", None, seed=0), x, bad, am, std, batch=4)
    assert set(lp) == set(element_idx)
    assert all(v["forbidden"] == 0.0 for v in lp.values())
    assert all(v["forbidden"] > 0 for v in lq.values())


def test_stellar_groups():
    from prism.evaluate import STELLAR_GROUPS

    teff = np.array([3800, 4500, 5000, 5500, 4800, 7000.0])
    logg = np.array([1.0, 2.5, 3.8, 4.5, 2.0, 4.0])
    feh = np.array([0.0, 0.0, 0.0, 0.0, -1.5, 0.0])
    got = {k: f(teff, logg, feh).tolist() for k, f in STELLAR_GROUPS.items()}
    assert got["RGB Tip"] == [True, False, False, False, False, False]
    assert got["RGB Giants"] == [False, True, False, False, True, False]
    assert got["Subgiants"] == [False, False, True, False, False, False]
    assert got["Dwarfs"] == [False, False, False, True, False, False]
    assert got["Metal-poor Giants"] == [False, False, False, False, True, False]
    assert got["Hot Stars"] == [False, False, False, False, False, True]


def test_plots_written(tmp_path):
    from prism.evaluate import binned_median_plots, leakage_plot

    rng = np.random.default_rng(0)
    x = {"Teff": rng.uniform(3500, 7000, 3000)}
    chi = {"a": rng.uniform(1, 2, 3000), "b": rng.uniform(1, 3, 3000)}
    paths = binned_median_plots(x, chi, tmp_path)
    leak = {"a": {"FE_H": {"ratio": 0.5}}, "b": {"FE_H": {"ratio": 1.2}}}
    p2 = leakage_plot(leak, tmp_path)
    assert paths[0].name == "median_chi2_vs_Teff_masked.png" and paths[0].exists() and p2.exists()
