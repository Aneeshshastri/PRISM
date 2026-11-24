import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx


class _TeffOnly(nnx.Module):
    def __init__(self):
        self.lin = nnx.Linear(1, 8575, rngs=nnx.Rngs(0))

    def __call__(self, x, bad_mask=None):
        return self.lin(x[:, :1])


def _forbidden(synthetic_masks):
    from prism.data.masks import line_mask_to_forbidden, load_apogee_mask

    return line_mask_to_forbidden(load_apogee_mask(synthetic_masks), 0.01)


def test_penalty_zero_for_line_free_dependence(synthetic_masks):
    from prism.models.regularised import jacobian_penalty

    x = jnp.asarray(np.random.default_rng(0).normal(size=(8, 27)), jnp.float32)
    p = jacobian_penalty(_TeffOnly(), x, _forbidden(synthetic_masks), jax.random.key(0), row_mask=jnp.ones(8))
    assert float(p) == 0.0


def test_penalty_positive_for_payne_and_differentiable(synthetic_masks):
    from prism.models.registry import build_model
    from prism.models.regularised import jacobian_penalty

    m = build_model("payne", None, seed=0)
    x = jnp.asarray(np.random.default_rng(0).normal(size=(8, 27)), jnp.float32)
    forb = _forbidden(synthetic_masks)
    assert float(jacobian_penalty(m, x, forb, jax.random.key(0), row_mask=jnp.ones(8))) > 0

    def loss(model):
        return jacobian_penalty(model, x, forb, jax.random.key(0), row_mask=jnp.ones(8))

    grads = nnx.grad(loss)(m)
    total = sum(float(jnp.abs(g).sum()) for g in jax.tree_util.tree_leaves(grads))
    assert total > 0


def test_only_line_columns_are_perturbed(synthetic_masks):
    from prism.models.regularised import sample_columns

    cols = np.asarray(sample_columns(jax.random.key(1), _forbidden(synthetic_masks), 1000))
    assert set(cols.tolist()) == set(range(9, 23))


def test_padded_rows_do_not_count(synthetic_masks):
    from prism.models.registry import build_model
    from prism.models.regularised import jacobian_penalty

    m = build_model("payne", None, seed=0)
    forb = _forbidden(synthetic_masks)
    x = jnp.asarray(np.random.default_rng(0).normal(size=(8, 27)), jnp.float32)
    xp = jnp.concatenate([x, 50.0 * jnp.ones((2, 27))])
    a = jacobian_penalty(m, x, forb, jax.random.key(2), row_mask=jnp.ones(8))
    b = jacobian_penalty(m, xp, forb, jax.random.key(2), row_mask=jnp.r_[jnp.ones(8), jnp.zeros(2)])
    # columns are drawn per row with fold_in(key, row), so padding cannot change real rows
    assert float(a) > 0 and np.isclose(float(a), float(b), rtol=1e-6)


def test_phases_constant():
    from prism.models.regularised import REGPAYNE_PHASES

    assert REGPAYNE_PHASES == ((5, 0.011906296737322275), (None, 1851.3543948324022))
