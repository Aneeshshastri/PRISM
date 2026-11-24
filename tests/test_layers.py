import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx


def test_chebyshev_basis():
    from prism.models.layers import chebyshev_basis

    b = chebyshev_basis(4, 5)
    x = np.linspace(-1, 1, 5)
    assert b.shape == (4, 5)
    assert np.allclose(b, np.polynomial.chebyshev.chebvander(x, 3).T, atol=1e-6)


def test_continuum_shape_and_basis_not_param():
    from prism.models.layers import ChebyshevContinuum

    m = ChebyshevContinuum(13, degree=20, rngs=nnx.Rngs(0))
    out = m(jnp.ones((3, 13)))
    assert out.shape == (3, 8575) and np.isfinite(np.asarray(out)).all()
    params = jax.tree_util.tree_leaves(nnx.state(m, nnx.Param))
    assert all(p.shape != (20, 8575) for p in params)  # basis is a constant, not trained


def test_sparse_projector_exact_zeros_and_gradient():
    from prism.models.layers import SparseProjector

    idx = np.array([5, 100, 8000], np.int32)
    w = np.array([0.5, 1.0, 0.2], np.float32)
    proj = SparseProjector(idx, w)
    local = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    out = np.asarray(proj(local))
    off = np.ones(8575, bool)
    off[idx] = False
    assert out.shape == (2, 8575) and (out[:, off] == 0).all()
    assert np.allclose(out[:, idx], np.asarray(local) * w)
    r = jnp.asarray(np.random.default_rng(0).normal(size=(2, 8575)), jnp.float32)
    g = jax.grad(lambda x: jnp.sum(proj(x) * r))(local)
    assert np.allclose(np.asarray(g), w * np.asarray(r)[:, idx])
    assert jax.tree_util.tree_leaves(nnx.state(proj, nnx.Param)) == []


def test_mlp_and_beer_lambert():
    from prism.models.layers import MLP, beer_lambert

    mlp = MLP(4, [8, 8], 3, out_act=nnx.softplus, rngs=nnx.Rngs(0))
    y = mlp(jnp.ones((2, 4)))
    assert y.shape == (2, 3) and (np.asarray(y) > 0).all()
    assert np.allclose(beer_lambert(jnp.array([2.0]), jnp.array([0.0])), 2.0)
    assert np.allclose(beer_lambert(jnp.array([1.0]), jnp.array([np.log(2)])), 0.5)
