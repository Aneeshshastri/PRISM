"""Runtime environment check — run first in every Kaggle notebook."""

from __future__ import annotations

import importlib
import os
import subprocess

import jax
import jax.numpy as jnp
import numpy as np

from prism.mesh import data_sharding, make_mesh, replicated

MIN_CUDA13_DRIVER = 580


def _version(module: str) -> str | None:
    try:
        return getattr(importlib.import_module(module), "__version__", "?")
    except Exception:  # noqa: BLE001 - optional packages
        return None


def _driver_version() -> str | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10, check=True,
        )
        return out.stdout.strip().splitlines()[0]
    except Exception:  # noqa: BLE001 - no GPU / no nvidia-smi
        return None


def _fallback_hint() -> str:
    return (
        "If the GPUs are missing because the host driver is too old for CUDA-13 JAX "
        f"(needs driver >= {MIN_CUDA13_DRIVER}), install the CUDA-12 build of the SAME JAX version:\n"
        f'  !pip install -c /tmp/kaggle-pins.txt "jax[cuda12]=={jax.__version__}"\n'
        "then restart the session. Also check Settings -> Accelerator is 'GPU T4 x2'."
    )


def default_expected_gpus() -> int:
    """GPUs the notebooks expect: 2 on Kaggle T4 x2; set PRISM_EXPECTED_GPUS=1 for one local GPU."""
    return int(os.environ.get("PRISM_EXPECTED_GPUS", "2"))


_DEFAULT = object()


def check_env(expected_gpus=_DEFAULT, require_gpu: bool = True) -> dict:
    """Print versions/devices, run a sharded smoke test, and fail loudly on a bad setup.

    ``expected_gpus`` defaults to ``default_expected_gpus()``; pass None to accept any count.
    """
    if expected_gpus is _DEFAULT:
        expected_gpus = default_expected_gpus()
    info: dict = {
        "versions": {m: _version(m) for m in
                     ("jax", "jaxlib", "flax", "optax", "orbax.checkpoint", "blackjax", "numpy")},
        "backend": jax.default_backend(),
    }
    devices = jax.devices()
    gpus = [d for d in devices if d.platform == "gpu"]
    info["n_devices"] = len(devices)
    info["n_gpus"] = len(gpus)
    info["compute_capability"] = [getattr(d, "compute_capability", None) for d in gpus]
    info["driver"] = _driver_version()

    print("PRISM environment")
    for name, ver in info["versions"].items():
        print(f"  {name:<17} {ver}")
    print(f"  backend           {info['backend']}  ({len(devices)} device(s): "
          f"{', '.join(d.device_kind for d in devices)})")
    if gpus:
        print(f"  compute capability {info['compute_capability']}  driver {info['driver']}")
        try:
            if info["driver"] and int(info["driver"].split(".")[0]) < MIN_CUDA13_DRIVER:
                print(f"  WARNING: driver {info['driver']} < {MIN_CUDA13_DRIVER}; "
                      "CUDA-13 JAX wheels may not work.")
        except ValueError:
            pass

    if require_gpu and not gpus:
        raise RuntimeError("No GPU visible to JAX.\n" + _fallback_hint())
    if expected_gpus is not None and require_gpu and len(gpus) != expected_gpus:
        raise RuntimeError(
            f"Expected {expected_gpus} GPUs but JAX sees {len(gpus)}.\n"
            "Running on one local GPU? Set PRISM_EXPECTED_GPUS=1 (or call check_env(expected_gpus=1)).\n"
            + _fallback_hint())

    # Sharded smoke test: rows split across the mesh, weights replicated.
    mesh = make_mesh()
    nd = len(devices)
    x = np.arange(nd * 4 * 8, dtype=np.float32).reshape(nd * 4, 8) / 100.0
    w = np.linspace(-1, 1, 8 * 3, dtype=np.float32).reshape(8, 3)
    xs = jax.device_put(x, data_sharding(mesh))
    ws = jax.device_put(w, replicated(mesh))
    y = jax.jit(lambda a, b: jnp.tanh(a @ b))(xs, ws)
    if not np.allclose(np.asarray(y), np.tanh(x @ w), atol=1e-5):
        raise RuntimeError("Sharded smoke test produced wrong results.")
    info["smoke_test"] = "ok"
    print(f"  sharded smoke test ok on {nd} device(s)")
    return info
