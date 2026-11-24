import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx


@pytest.fixture(scope="module")
def element_idx(synthetic_masks):
    from prism.data.masks import element_indices, load_apogee_mask

    return element_indices(load_apogee_mask(synthetic_masks), 0.01)


@pytest.mark.parametrize("name", ["prism", "ablation", "payne", "expanded_payne", "reg_payne"])
def test_forward_shapes(name, element_idx):
    from prism.models.registry import build_model

    m = build_model(name, element_idx, seed=0)
    out = m(jnp.zeros((3, 27)))
    assert out.shape == (3, 8575) and np.isfinite(np.asarray(out)).all()
    assert m.prism_name == name


def _jvp_col(model, col, x):
    t = jnp.zeros_like(x).at[:, col].set(1.0)
    return np.asarray(jax.jvp(model, (x,), (t,))[1])


def test_prism_is_local(element_idx):
    from prism.models.prism import IDX_CNO, IDX_GLOBAL
    from prism.models.registry import build_model

    assert set(range(10, 23)).isdisjoint(IDX_GLOBAL + IDX_CNO)
    m = build_model("prism", element_idx, seed=0)
    x = jnp.asarray(np.random.default_rng(0).normal(size=(4, 27)), jnp.float32)
    d = _jvp_col(m, 10, x)  # MG_FE
    off = np.ones(8575, bool)
    off[element_idx["MG_FE"][0]] = False
    assert (d[:, off] == 0).all() and np.abs(d[:, ~off]).max() > 0


def test_ablation_leaks(element_idx):
    from prism.models.registry import build_model

    m = build_model("ablation", element_idx, seed=0)
    x = jnp.asarray(np.random.default_rng(0).normal(size=(4, 27)), jnp.float32)
    d = _jvp_col(m, 10, x)
    off = np.ones(8575, bool)
    off[element_idx["MG_FE"][0]] = False
    assert np.abs(d[:, off]).max() > 0


def test_same_seed_same_model(element_idx):
    from prism.models.registry import build_model

    a, b = build_model("prism", element_idx, seed=3), build_model("prism", element_idx, seed=3)
    x = jnp.ones((2, 27))
    assert np.array_equal(np.asarray(a(x)), np.asarray(b(x)))
    n = lambda m: sum(p.size for p in jax.tree_util.tree_leaves(nnx.state(m, nnx.Param)))  # noqa: E731
    assert n(a) == n(b)


def test_prism_bad_mask_ignored(element_idx):
    from prism.models.registry import build_model

    m = build_model("prism", element_idx, seed=0)
    x = jnp.ones((2, 27))
    assert np.array_equal(np.asarray(m(x)), np.asarray(m(x, jnp.ones((2, 23)))))


def test_unknown_model_name(element_idx):
    from prism.models.registry import build_model

    with pytest.raises(ValueError):
        build_model("nope", element_idx, seed=0)
