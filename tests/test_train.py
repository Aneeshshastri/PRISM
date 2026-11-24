import jax
import numpy as np
import pytest
from flax import nnx


def test_schedule_matches_keras_formula():
    from prism.schedules import cosine_decay_restarts

    s = cosine_decay_restarts(1e-3, 10, 2.0, 0.85, 1e-6)
    assert np.isclose(s(0), 1e-3)
    assert np.isclose(s(10), 0.85e-3, rtol=1e-4)       # second period starts at step 10
    assert np.isclose(s(30), 0.85 ** 2 * 1e-3, rtol=1e-4)  # third at 10 + 20
    assert float(s(9)) < 1e-4 and float(s(29)) < 1e-4


@pytest.fixture
def prepared(tiny_config):
    from prism.data.prepare import prepare_all

    return prepare_all(tiny_config)


def _loaders(prepared, n_devices=2, batch=8):
    from prism.data.loader import BatchLoader

    tr = BatchLoader(prepared, "train", batch, shuffle=True, seed=0, n_devices=n_devices)
    va = BatchLoader(prepared, "val", batch, shuffle=False, seed=0, n_devices=n_devices)
    return tr, va


def _trainer(cfg, prepared, mesh, name="payne", seed=0, **kw):
    from prism.models.registry import build_model
    from prism.train import Trainer, make_forward_loss, make_optimizer

    model = build_model(name, None, seed=seed)
    return Trainer(model, cfg, mesh, loss_fn=make_forward_loss(np.load(prepared / "suppress.npy")),
                   optimizer=make_optimizer(cfg), **kw)


def test_sharded_step_equals_single_device(tiny_config, prepared):
    from prism.mesh import make_mesh

    tr, _ = _loaders(prepared)
    batch = next(iter(tr))
    t2 = _trainer(tiny_config, prepared, make_mesh())
    t1 = _trainer(tiny_config, prepared, make_mesh(jax.devices()[:1]))
    t2.train_batch(batch, step=0)
    t1.train_batch(batch, step=0)
    p2 = jax.tree_util.tree_leaves(jax.device_get(nnx.state(t2.model, nnx.Param)))
    p1 = jax.tree_util.tree_leaves(jax.device_get(nnx.state(t1.model, nnx.Param)))
    assert all(np.allclose(a, b, atol=1e-6) for a, b in zip(p1, p2))


def test_trainer_reduces_loss(tiny_config, prepared):
    from prism.mesh import make_mesh

    tr, va = _loaders(prepared)
    r = _trainer(tiny_config.replace(patience=10), prepared, make_mesh()).fit(tr, va, epochs=3)
    assert len(r.history["loss"]) == 3 and r.history["loss"][-1] < r.history["loss"][0]
    assert {"loss", "val_loss", "lr", "val_spl"} <= set(r.history)


def test_trainer_ignores_padded_rows(tiny_config, prepared):
    from prism.mesh import make_mesh

    tr, _ = _loaders(prepared)
    batch = next(iter(tr))
    six = {k: v[:6] for k, v in batch.items()}
    padded = {k: v.copy() for k, v in batch.items()}
    padded["row_mask"][6:] = 0.0
    padded["flux"][6:] = 123.0
    padded["weights"][6:] = 50.0
    t = _trainer(tiny_config, prepared, make_mesh())
    t_single = _trainer(tiny_config, prepared, make_mesh(jax.devices()[:1]))
    loss_pad, _ = t.eval_batch(padded)
    loss_six, _ = t_single.eval_batch(six)
    assert np.isclose(loss_pad, loss_six, rtol=1e-5)


def test_checkpoint_roundtrip(tmp_path, synthetic_masks):
    import jax.numpy as jnp

    from prism.checkpoint import load_model, save_model
    from prism.data.masks import element_indices, load_apogee_mask
    from prism.models.registry import MODEL_NAMES, build_model

    idx = element_indices(load_apogee_mask(synthetic_masks), 0.01)
    x = jnp.asarray(np.random.default_rng(0).normal(size=(2, 27)), jnp.float32)
    for name in MODEL_NAMES:
        m = build_model(name, idx, seed=1)
        # perturb params so we are not just comparing two fresh inits
        state = nnx.state(m, nnx.Param)
        nnx.update(m, jax.tree.map(lambda p: p + 0.01, state))
        save_model(tmp_path / name, m, {"name": name, "seed": 1}, element_idx=idx)
        m2 = load_model(tmp_path / name)
        assert np.array_equal(np.asarray(m(x)), np.asarray(m2(x))), name


def test_early_stopping_restores_best(tiny_config, prepared):
    from prism.mesh import make_mesh
    from prism.train import Trainer

    seq = iter([5.0, 3.0, 4.0, 4.5, 4.6])

    class Scripted(Trainer):
        def _validate(self, loader):
            return {"val_loss": next(seq)}

    snapshots = {}

    def cb(epoch, logs):
        snapshots[epoch] = jax.device_get(nnx.state(t.model, nnx.Param))
        return False

    from prism.models.registry import build_model
    from prism.train import make_forward_loss, make_optimizer

    t = Scripted(build_model("payne", None, seed=0), tiny_config.replace(patience=2), make_mesh(),
                 loss_fn=make_forward_loss(np.load(prepared / "suppress.npy")),
                 optimizer=make_optimizer(tiny_config))
    tr, va = _loaders(prepared)
    r = t.fit(tr, va, epochs=5, on_epoch_end=cb)
    assert r.best_epoch == 1 and r.best_val == 3.0 and len(r.history["loss"]) == 4  # stops after 2 bad epochs
    now = jax.tree_util.tree_leaves(jax.device_get(nnx.state(t.model, nnx.Param)))
    assert all(np.array_equal(a, b) for a, b in zip(now, jax.tree_util.tree_leaves(snapshots[1])))


def test_callback_can_stop(tiny_config, prepared):
    from prism.mesh import make_mesh

    tr, va = _loaders(prepared)
    r = _trainer(tiny_config.replace(patience=10), prepared, make_mesh()).fit(
        tr, va, epochs=5, on_epoch_end=lambda e, logs: e >= 1)
    assert len(r.history["loss"]) == 2


def test_plateau_reduces_lr(tiny_config, prepared):
    from prism.mesh import make_mesh

    seq = iter([5.0, 6.0, 6.0, 6.0, 6.0])
    from prism.train import Trainer

    class Scripted(Trainer):
        def _validate(self, loader):
            return {"val_loss": next(seq)}

    from prism.models.registry import build_model
    from prism.train import make_forward_loss, make_optimizer

    t = Scripted(build_model("payne", None, seed=0), tiny_config.replace(patience=10), make_mesh(),
                 loss_fn=make_forward_loss(np.load(prepared / "suppress.npy")),
                 optimizer=make_optimizer(tiny_config), plateau={"patience": 2, "factor": 0.5, "min_lr": 1e-6})
    tr, va = _loaders(prepared)
    r = t.fit(tr, va, epochs=5)
    assert np.isclose(r.history["lr"][0], 1e-3) and np.isclose(r.history["lr"][-1], 5e-4)


def test_monitor_selects_best_epoch_by_named_metric(tiny_config, prepared):
    """Review finding: RegPayne must pick/stop on the spectral loss, not the penalised val_loss."""
    from prism.mesh import make_mesh
    from prism.models.registry import build_model
    from prism.train import Trainer, make_forward_loss, make_optimizer

    seq = iter([(1.0, 5.0), (9.0, 3.0), (0.5, 4.0)])  # (val_loss, val_spl)

    class Scripted(Trainer):
        def _validate(self, loader):
            vl, vs = next(seq)
            return {"val_loss": vl, "val_spl": vs}

    t = Scripted(build_model("payne", None, seed=0), tiny_config.replace(patience=10), make_mesh(),
                 loss_fn=make_forward_loss(np.load(prepared / "suppress.npy")),
                 optimizer=make_optimizer(tiny_config), monitor="val_spl")
    tr, va = _loaders(prepared)
    r = t.fit(tr, va, epochs=3)
    assert r.best_epoch == 1 and r.best_val == 3.0
