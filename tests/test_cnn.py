import jax.numpy as jnp
import numpy as np


def test_cnn_shapes():
    from prism.models.registry import build_model

    m = build_model("cnn", None, seed=0)
    mu, raw = m(jnp.ones((3, 8575)))
    assert mu.shape == (3, 23) and raw.shape == (3, 23)


def test_dropout_only_in_train_mode():
    from prism.models.registry import build_model

    m = build_model("cnn", None, seed=0)
    x = jnp.asarray(np.random.default_rng(0).uniform(0.8, 1.0, (4, 8575)), jnp.float32)
    m.train()
    a, b = m(x)[0], m(x)[0]
    assert not np.allclose(np.asarray(a), np.asarray(b))
    m.eval()
    c, d = m(x)[0], m(x)[0]
    assert np.array_equal(np.asarray(c), np.asarray(d))


def test_clean_flux():
    from prism.models.cnn_predictor import clean_flux

    f = jnp.array([[np.nan, -1.0, 0.0005, 0.9, np.inf]])
    assert np.allclose(np.asarray(clean_flux(f)), [[1.0, 1.0, 1.0, 0.9, 1.0]])


def test_cnn_trains_one_epoch(tiny_config):
    from prism.data.loader import BatchLoader
    from prism.data.prepare import prepare_all
    from prism.mesh import make_mesh
    from prism.models.cnn_predictor import make_cnn_loss
    from prism.models.registry import build_model
    from prism.train import Trainer, make_optimizer

    d = prepare_all(tiny_config)
    fields = ("labels", "flux")
    tr = BatchLoader(d, "train", 8, shuffle=True, seed=0, fields=fields, n_devices=2)
    va = BatchLoader(d, "val", 8, shuffle=False, seed=0, fields=fields, n_devices=2)
    t = Trainer(build_model("cnn", None, seed=0), tiny_config, make_mesh(), loss_fn=make_cnn_loss(),
                optimizer=make_optimizer(tiny_config), plateau={"patience": 5, "factor": 0.5, "min_lr": 1e-6})
    r = t.fit(tr, va, epochs=1)
    assert np.isfinite(r.history["val_loss"][0]) and "val_mse" in r.history


def test_cnn_checkpoint_roundtrip(tmp_path):
    import jax
    from flax import nnx

    from prism.checkpoint import load_model, save_model
    from prism.models.registry import build_model

    m = build_model("cnn", None, seed=0)
    x = jnp.asarray(np.random.default_rng(0).uniform(0.8, 1.0, (4, 8575)), jnp.float32)
    m.train()
    m(x)  # updates BatchNorm running statistics
    bs = nnx.state(m, nnx.BatchStat)
    assert any(float(jnp.abs(v).sum()) > 0 for v in jax.tree_util.tree_leaves(bs))
    save_model(tmp_path / "cnn", m, {"name": "cnn", "seed": 0})
    m2 = load_model(tmp_path / "cnn")
    m.eval()
    m2.eval()
    assert np.array_equal(np.asarray(m(x)[0]), np.asarray(m2(x)[0]))
