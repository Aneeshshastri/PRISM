"""Generic data-parallel trainer for NNX models.

Parameters and optimizer state are replicated over the mesh; every batch is
sharded along ``'data'``. ``nnx.jit`` inserts the gradient all-reduce.
"""

from __future__ import annotations

import faulthandler
import sys
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from prism.checkpoint import save_model
from prism.data.masks import pixel_mask_from_bad
from prism.losses import masked_spl_loss_w
from prism.mesh import replicate, shard_batch
from prism.precision import check_precision, matmul_precision
from prism.models.regularised import jacobian_penalty
from prism.schedules import cosine_decay_restarts, warmup_cosine

LossFn = Callable[[nnx.Module, dict, jax.Array], tuple[jax.Array, dict]]


def make_optimizer(cfg, schedule: optax.Schedule | None = None) -> optax.GradientTransformation:
    """Adam with global-norm clipping; the learning rate is an injected hyper-parameter."""
    lr = schedule if schedule is not None else cfg.learning_rate
    return optax.chain(optax.clip_by_global_norm(cfg.clip_norm),
                       optax.inject_hyperparams(optax.adam)(learning_rate=lr))


def make_schedule(cfg, steps_per_epoch: int, epochs: int, *, resume_step: int = 0) -> optax.Schedule:
    """Learning-rate schedule for ``epochs`` epochs of ``steps_per_epoch`` steps (``cfg.lr_schedule``).

    ``resume_step > 0``: training resumes there with fresh Adam moments, so the LR ramps linearly
    back up to the schedule's value over one epoch instead of jumping straight to it."""
    if cfg.lr_schedule == "cosine":
        base = warmup_cosine(cfg.learning_rate, steps_per_epoch * epochs, cfg.warmup_epochs * steps_per_epoch,
                             cfg.final_lr_fraction)
    elif cfg.lr_schedule == "cosine_restarts":
        base = cosine_decay_restarts(cfg.learning_rate, cfg.first_decay_steps, cfg.t_mul, cfg.m_mul, cfg.alpha)
    else:
        raise ValueError(f"unknown lr_schedule {cfg.lr_schedule!r}; use 'cosine' or 'cosine_restarts'")
    if resume_step <= 0:
        return base
    ramp = max(steps_per_epoch, 1)
    return lambda count: base(count) * jnp.clip((count - resume_step + 1) / ramp, 0.0, 1.0)


def make_forward_loss(suppress: np.ndarray, penalty: float | None = None,
                      forbidden: np.ndarray | None = None, scale: float = 1.0) -> LossFn:
    """``scale`` x (masked spectral pixel loss + optional RegPayne Jacobian penalty).

    The ``spl`` metric is reported unscaled."""
    suppress = jnp.asarray(suppress, jnp.float32)
    if penalty and forbidden is None:
        raise ValueError("a penalty needs the forbidden-pixel mask")

    def loss_fn(model, batch, key):
        pred = model(batch["labels"])
        pm = pixel_mask_from_bad(batch["bad_mask"], suppress)
        spl = masked_spl_loss_w(pred, batch["flux"], batch["spl_weight"], pm, batch["weights"], batch["row_mask"])
        loss = spl
        if penalty:
            loss = loss + penalty * jacobian_penalty(model, batch["labels"], forbidden, key,
                                                     row_mask=batch["row_mask"])
        return scale * loss, {"spl": spl}

    return loss_fn


@dataclass
class TrainResult:
    history: dict[str, list[float]] = field(default_factory=dict)
    best_epoch: int = -1
    best_val: float = float("inf")


def _lr_variable(optimizer: nnx.Optimizer):
    return optimizer.opt_state[1].hyperparams["learning_rate"]


@contextmanager
def _epoch_watchdog(timeout_s: float):
    """Yields ``arm()``; call it at each epoch start. If an epoch then runs longer than
    ``timeout_s`` every thread's stack goes to stderr and the process exits (code 1), so a hung
    run fails loudly instead of sitting idle. ``timeout_s <= 0`` disables it."""
    def arm():
        if timeout_s > 0:  # the real stderr (fd 2 -> run.out), even when sys.stderr is captured
            faulthandler.dump_traceback_later(timeout_s, exit=True, file=sys.__stderr__)
    try:
        yield arm
    finally:
        if timeout_s > 0:
            faulthandler.cancel_dump_traceback_later()


class Trainer:
    def __init__(self, model: nnx.Module, cfg, mesh, *, loss_fn: LossFn,
                 optimizer: optax.GradientTransformation | nnx.Optimizer,
                 plateau: dict | None = None, ckpt_dir: Path | None = None,
                 ckpt_meta: dict | None = None, element_idx: dict | None = None, monitor: str = "val_loss",
                 precision: str = "float32"):
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
        # Matmul precision the train/eval steps are traced with (see prism.precision). For
        # "bfloat16" the model must also be built with dtype=bfloat16 (stages do this).
        self.precision = check_precision(precision)
        self.ckpt_meta = {**(ckpt_meta or {"name": getattr(model, "prism_name", "model"), "seed": cfg.seed}),
                          "precision": precision}
        if getattr(model, "prism_arch", None) is not None:  # load_model rebuilds this architecture
            self.ckpt_meta["arch"] = dict(model.prism_arch)
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

    def _improved(self, val: float, best: float) -> bool:
        if not np.isfinite(best):
            return bool(np.isfinite(val))
        delta = max(self.cfg.min_delta, getattr(self.cfg, "rel_min_delta", 0.0) * abs(best))
        return val < best - delta

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
        with matmul_precision(self.precision):
            loss, _ = self._train_step(self.model, self.optimizer, shard_batch(batch, self.mesh),
                                       jax.random.fold_in(self.key, step))
        return float(loss)

    def eval_batch(self, batch: dict) -> tuple[float, dict]:
        self.model.eval()
        with matmul_precision(self.precision):
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

    def _resume_at(self, start_epoch: int, steps_per_epoch: int) -> None:
        """Continue the step counter (RNG folds, LR schedule) from ``start_epoch``; Adam's own
        bias-correction count stays at 0 because its moments restart from zero."""
        self.global_step = start_epoch * steps_per_epoch
        inject = self.optimizer.opt_state[1]
        # a callable LR is wrapped in a stateful schedule with its own step counter
        counters = [inject.count] + [st.count for st in inject.hyperparams_states.values() if hasattr(st, "count")]
        for count in counters:
            count[...] = jax.device_put(jnp.asarray(self.global_step, count[...].dtype), count[...].sharding)

    def fit(self, train_loader, val_loader, *, epochs: int,
            on_epoch_end: Callable[[int, dict], bool] | None = None,
            start_epoch: int = 0, best_val: float = float("inf")) -> TrainResult:
        """Train epochs ``start_epoch .. epochs-1``. Resuming: load the checkpoint's weights into the
        model first and pass its next epoch and monitored value so early stopping and
        best-checkpoint selection carry on from there."""
        res = TrainResult(best_val=best_val, best_epoch=start_epoch - 1 if start_epoch else -1)
        best_state = None
        if start_epoch:
            self._resume_at(start_epoch, len(train_loader))
            best_state = jax.device_get(nnx.state(self.model))
            print(f"  resuming at epoch {start_epoch + 1} (best {self.monitor} {best_val:.6g}), "
                  f"lr re-warms over 1 epoch")
        wait = plateau_wait = 0
        plateau_best = float("inf")
        with _epoch_watchdog(float(getattr(self.cfg, "epoch_timeout_s", 0) or 0)) as arm_watchdog:
            for epoch in range(start_epoch, epochs):
                arm_watchdog()
                t0 = time.time()
                lr_used = self.lr
                train_loss = self._train_epoch(train_loader, epoch)
                logs = {"loss": train_loss, "lr": lr_used, **self._validate(val_loader)}
                for k, v in logs.items():
                    res.history.setdefault(k, []).append(float(v))
                val = logs[self.monitor]
                extra = f"  {self.monitor} {val:.6g}" if self.monitor != "val_loss" else ""
                print(f"  epoch {epoch + 1:>3}/{epochs}  loss {train_loss:.6g}  val_loss {logs['val_loss']:.6g}"
                      f"{extra}  lr {lr_used:.3g}  ({time.time() - t0:.1f}s)")

                if self._improved(val, res.best_val):
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
