"""Generic data-parallel trainer for NNX models.

Parameters and optimizer state are replicated over the mesh; every batch is
sharded along ``'data'``. ``nnx.jit`` inserts the gradient all-reduce.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from prism.checkpoint import save_model
from prism.data.masks import pixel_mask_from_bad
from prism.losses import masked_spl_loss
from prism.mesh import replicate, shard_batch
from prism.models.regularised import jacobian_penalty

LossFn = Callable[[nnx.Module, dict, jax.Array], tuple[jax.Array, dict]]


def make_optimizer(cfg, schedule: optax.Schedule | None = None) -> optax.GradientTransformation:
    """Adam with global-norm clipping; the learning rate is an injected hyper-parameter."""
    lr = schedule if schedule is not None else cfg.learning_rate
    return optax.chain(optax.clip_by_global_norm(cfg.clip_norm),
                       optax.inject_hyperparams(optax.adam)(learning_rate=lr))


def make_forward_loss(suppress: np.ndarray, penalty: float | None = None,
                      forbidden: np.ndarray | None = None) -> LossFn:
    """Masked spectral pixel loss (+ optional RegPayne Jacobian penalty)."""
    suppress = jnp.asarray(suppress, jnp.float32)
    if penalty and forbidden is None:
        raise ValueError("a penalty needs the forbidden-pixel mask")

    def loss_fn(model, batch, key):
        pred = model(batch["labels"])
        pm = pixel_mask_from_bad(batch["bad_mask"], suppress)
        spl = masked_spl_loss(pred, batch["flux"], batch["ivar"], pm, batch["weights"], batch["row_mask"])
        loss = spl
        if penalty:
            loss = loss + penalty * jacobian_penalty(model, batch["labels"], forbidden, key,
                                                     row_mask=batch["row_mask"])
        return loss, {"spl": spl}

    return loss_fn


@dataclass
class TrainResult:
    history: dict[str, list[float]] = field(default_factory=dict)
    best_epoch: int = -1
    best_val: float = float("inf")


def _lr_variable(optimizer: nnx.Optimizer):
    return optimizer.opt_state[1].hyperparams["learning_rate"]


class Trainer:
    def __init__(self, model: nnx.Module, cfg, mesh, *, loss_fn: LossFn,
                 optimizer: optax.GradientTransformation | nnx.Optimizer,
                 plateau: dict | None = None, ckpt_dir: Path | None = None,
                 ckpt_meta: dict | None = None, element_idx: dict | None = None, monitor: str = "val_loss"):
        self.model = model
        self.cfg = cfg
        self.mesh = mesh
        self.loss_fn = loss_fn
        self.optimizer = optimizer if isinstance(optimizer, nnx.Optimizer) else \
            nnx.Optimizer(model, optimizer, wrt=nnx.Param)
        self.plateau = plateau
        # Metric that drives best-epoch selection, early stopping and the LR plateau.
        # RegPayne uses "val_spl": its val_loss includes the Jacobian penalty.
        self.monitor = monitor
        self.ckpt_dir = Path(ckpt_dir) if ckpt_dir else None
        self.ckpt_meta = ckpt_meta or {"name": getattr(model, "prism_name", "model"), "seed": cfg.seed}
        self.element_idx = element_idx
        self.key = jax.random.key(cfg.seed)
        self.global_step = 0

        state = replicate(nnx.state((self.model, self.optimizer)), mesh)
        nnx.update((self.model, self.optimizer), state)

        loss = self.loss_fn

        @nnx.jit
        def train_step(model, opt, batch, key):
            (value, metrics), grads = nnx.value_and_grad(lambda m: loss(m, batch, key), has_aux=True)(model)
            opt.update(model, grads)
            return value, metrics

        @nnx.jit
        def eval_step(model, batch, key):
            return loss(model, batch, key)

        self._train_step = train_step
        self._eval_step = eval_step

    # --- learning rate -------------------------------------------------
    @property
    def lr(self) -> float:
        return float(jax.device_get(_lr_variable(self.optimizer)[...]))

    def _set_lr(self, value: float) -> None:
        var = _lr_variable(self.optimizer)
        var[...] = jax.device_put(jnp.asarray(value, var[...].dtype), var[...].sharding)

    # --- single batches (also used by tests) ----------------------------
    def train_batch(self, batch: dict, step: int) -> float:
        self.model.train()
        loss, _ = self._train_step(self.model, self.optimizer, shard_batch(batch, self.mesh),
                                   jax.random.fold_in(self.key, step))
        return float(loss)

    def eval_batch(self, batch: dict) -> tuple[float, dict]:
        self.model.eval()
        loss, metrics = self._eval_step(self.model, shard_batch(batch, self.mesh), self.key)
        return float(loss), {k: float(v) for k, v in metrics.items()}

    # --- epochs ---------------------------------------------------------
    def _validate(self, loader) -> dict:
        totals: dict[str, float] = {}
        n_total = 0.0
        for batch in loader:
            n = float(batch["row_mask"].sum())
            loss, metrics = self.eval_batch(batch)
            totals["val_loss"] = totals.get("val_loss", 0.0) + loss * n
            for k, v in metrics.items():
                totals[f"val_{k}"] = totals.get(f"val_{k}", 0.0) + v * n
            n_total += n
        return {k: v / max(n_total, 1.0) for k, v in totals.items()}

    def _train_epoch(self, loader, epoch: int) -> float:
        loader.set_epoch(epoch)
        total, n_total = 0.0, 0.0
        for batch in loader:
            n = float(batch["row_mask"].sum())
            total += self.train_batch(batch, self.global_step) * n
            n_total += n
            self.global_step += 1
        return total / max(n_total, 1.0)

    def fit(self, train_loader, val_loader, *, epochs: int,
            on_epoch_end: Callable[[int, dict], bool] | None = None) -> TrainResult:
        res = TrainResult()
        best_state = None
        wait = plateau_wait = 0
        plateau_best = float("inf")
        for epoch in range(epochs):
            t0 = time.time()
            lr_used = self.lr
            train_loss = self._train_epoch(train_loader, epoch)
            logs = {"loss": train_loss, "lr": lr_used, **self._validate(val_loader)}
            for k, v in logs.items():
                res.history.setdefault(k, []).append(float(v))
            val = logs[self.monitor]
            extra = f"  {self.monitor} {val:.6g}" if self.monitor != "val_loss" else ""
            print(f"  epoch {epoch + 1:>3}/{epochs}  loss {train_loss:.6g}  val_loss {logs['val_loss']:.6g}{extra}"
                  f"  lr {lr_used:.3g}  ({time.time() - t0:.1f}s)")

            if val < res.best_val - self.cfg.min_delta:
                res.best_val, res.best_epoch, wait = val, epoch, 0
                best_state = jax.device_get(nnx.state(self.model))
                if self.ckpt_dir is not None:
                    save_model(self.ckpt_dir, self.model, {**self.ckpt_meta, "epoch": epoch, self.monitor: val},
                               element_idx=self.element_idx)
            else:
                wait += 1

            if self.plateau:
                if val < plateau_best - self.plateau.get("min_delta", 1e-4):
                    plateau_best, plateau_wait = val, 0
                else:
                    plateau_wait += 1
                    if plateau_wait >= self.plateau["patience"]:
                        new_lr = max(self.lr * self.plateau["factor"], self.plateau["min_lr"])
                        self._set_lr(new_lr)
                        plateau_wait = 0
                        print(f"  reducing learning rate to {new_lr:.3g}")

            stop = on_epoch_end(epoch, logs) if on_epoch_end else False
            if stop or wait >= self.cfg.patience:
                break

        if best_state is not None:
            nnx.update(self.model, replicate(best_state, self.mesh))
        return res
