"""Precision benchmark: is bf16 / TF32 training worth it on this GPU?

* ``matmul_precision_report`` — relative error of a float32 matmul under JAX's
  default, TF32 and highest precision. If ``default`` is ~1e-3-1e-4 the GPU silently
  uses TF32 (Ampere and newer); on a T4 or CPU all three agree at ~1e-7.
* ``benchmark_training`` — for each (model, precision): median training-step time,
  throughput, peak memory, validation spectral loss at the training precision AND with
  the same weights evaluated in full fp32 (what evaluation/inversion will see), and the
  relative RMS flux difference between the two.

Run on Kaggle and locally (notebook 09 or ``python -m prism.bench``) and compare the
two ``precision.json`` files.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from prism.config import Config
from prism.data.loader import DEFAULT_FIELDS, BatchLoader
from prism.data.masks import element_indices, load_apogee_mask, load_cno_masks
from prism.data.prepare import prepare_all
from prism.mesh import make_mesh, n_devices
from prism.models.registry import NEEDS_ELEMENT_IDX, build_model
from prism.precision import TRAIN_PRECISIONS, compute_dtype
from prism.train import Trainer, make_forward_loss, make_optimizer

_LEVELS = {"default": None, "tensorfloat32": "tensorfloat32", "highest": "highest"}


def matmul_precision_report(n: int = 1024, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((n, n)).astype(np.float32)
    b = rng.standard_normal((n, n)).astype(np.float32)
    ref = a.astype(np.float64) @ b.astype(np.float64)
    errs = {}
    for name, level in _LEVELS.items():
        f = jax.jit(lambda x, y, p=level: jnp.matmul(x, y, precision=p))
        c = np.asarray(f(a, b), np.float64)
        errs[name] = float(np.max(np.abs(c - ref)) / np.max(np.abs(ref)))
    dev = jax.devices()[0]
    return {"device": dev.device_kind, "platform": dev.platform,
            "compute_capability": getattr(dev, "compute_capability", None),
            "relative_error": errs,
            "default_uses_reduced_precision": errs["default"] > 10 * max(errs["highest"], 1e-9)}


def _peak_memory_mb() -> float | None:
    try:
        stats = jax.devices()[0].memory_stats()
        return round(stats["peak_bytes_in_use"] / 2 ** 20, 1) if stats else None
    except Exception:  # noqa: BLE001 - not available on every backend
        return None


def _fp32_copy(model, name: str, idx, seed: int):
    m32 = build_model(name, idx, seed=seed, arch=model.prism_arch)  # fp32 compute
    nnx.update(m32, jax.device_get(nnx.state(model)))
    return m32


def benchmark_training(cfg: Config, names=("prism", "payne"), precisions=TRAIN_PRECISIONS, epochs: int = 3,
                       max_train: int | None = None, mesh=None) -> list[dict]:
    mesh = mesh if mesh is not None else make_mesh()
    nd = n_devices(mesh)
    prep = prepare_all(cfg)
    suppress = np.load(prep / "suppress.npy")
    bs = -(-cfg.batch_size // nd) * nd
    cno = load_cno_masks(cfg.mask_dir, cfg.mask_threshold) if cfg.cno_split else None
    idx = element_indices(load_apogee_mask(cfg.mask_dir), cfg.mask_threshold, cno_masks=cno)
    kw = dict(fields=DEFAULT_FIELDS, n_devices=nd, prefetch=cfg.prefetch, seed=cfg.seed,
              in_memory=cfg.load_in_memory)
    train = BatchLoader(prep, "train", bs, shuffle=True, limit=max_train, **kw)
    val = BatchLoader(prep, "val", bs, shuffle=False, limit=max_train, **kw)
    loss_fn = make_forward_loss(suppress, scale=cfg.loss_scale)
    rows = []
    for name in names:
        e_idx = idx if name in NEEDS_ELEMENT_IDX else None
        for prec in precisions:
            model = build_model(name, e_idx, seed=cfg.seed, dtype=compute_dtype(prec), arch=cfg.arch())
            t = Trainer(model, cfg, mesh, loss_fn=loss_fn, optimizer=make_optimizer(cfg), precision=prec)
            times, step = [], 0
            for epoch in range(epochs):
                train.set_epoch(epoch)
                for batch in train:
                    t0 = time.perf_counter()
                    t.train_batch(batch, step)  # returns a host float -> synchronised
                    if epoch > 0 or (epochs == 1 and step > 0):  # skip the compile epoch/step
                        times.append(time.perf_counter() - t0)
                    step += 1
            val_prec = t._validate(val)["val_spl"]
            m32 = _fp32_copy(t.model, name, e_idx, cfg.seed)
            t32 = Trainer(m32, cfg, mesh, loss_fn=loss_fn, optimizer=make_optimizer(cfg), precision="float32")
            val_fp32 = t32._validate(val)["val_spl"]
            x = jnp.asarray(next(iter(val))["labels"])
            t.model.eval()
            o_prec = np.asarray(t.model(x), np.float64)
            o_32 = np.asarray(m32(x), np.float64)
            rel = float(np.sqrt(np.mean((o_prec - o_32) ** 2)) / max(np.sqrt(np.mean(o_32 ** 2)), 1e-12))
            step_ms = 1e3 * float(np.median(times)) if times else float("nan")
            rows.append({"model": name, "precision": prec, "step_ms": round(step_ms, 3),
                         "stars_per_s": round(bs / (step_ms / 1e3), 1) if times else float("nan"),
                         "peak_memory_mb": _peak_memory_mb(), "val_spl": val_prec, "val_spl_fp32_eval": val_fp32,
                         "flux_rel_rms_vs_fp32": rel, "epochs": epochs, "global_batch": bs, "n_devices": nd})
    out = {"matmul": matmul_precision_report(), "training": rows}
    path = Path(cfg.artifact_dir) / "benchmark" / "precision.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1, default=float))
    _print_table(out)
    return rows


def _print_table(out: dict) -> None:
    m = out["matmul"]
    print(f"\nDevice: {m['device']} ({m['platform']}, cc {m['compute_capability']})")
    print("float32 matmul relative error: " + ", ".join(f"{k} {v:.1e}" for k, v in m["relative_error"].items())
          + ("   <- default silently uses reduced precision" if m["default_uses_reduced_precision"] else ""))
    print(f"\n{'model':<15} {'precision':<14} {'step ms':>9} {'stars/s':>9} {'val SPL':>11} "
          f"{'val SPL(fp32)':>14} {'flux rel diff':>14}")
    for r in out["training"]:
        print(f"{r['model']:<15} {r['precision']:<14} {r['step_ms']:>9.2f} {r['stars_per_s']:>9.0f} "
              f"{r['val_spl']:>11.4g} {r['val_spl_fp32_eval']:>14.4g} {r['flux_rel_rms_vs_fp32']:>14.2e}")


def main() -> None:
    ap = argparse.ArgumentParser(description="PRISM precision benchmark")
    ap.add_argument("--report-only", action="store_true", help="only the matmul precision check")
    ap.add_argument("--models", default="prism,payne")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--max-train", type=int, default=None, help="use only the first N train/val stars")
    args = ap.parse_args()
    if args.report_only:
        print(json.dumps(matmul_precision_report(), indent=1))
        return
    benchmark_training(Config.for_kaggle(), names=tuple(args.models.split(",")), epochs=args.epochs,
                       max_train=args.max_train)


if __name__ == "__main__":
    main()
