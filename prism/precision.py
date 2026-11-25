"""Numerical precision policy.

* Training precision is opt-in (``Config.train_precision``):
    - ``"float32"``        true fp32 matmuls everywhere (default; what a T4 always does)
    - ``"tensorfloat32"``  fp32 storage, TF32 tensor-core matmuls (Ampere and newer GPUs)
    - ``"bfloat16"``       bf16 compute in the layers, fp32 master weights / loss / optimizer
* Inference (evaluation and every inverse method) always runs at full fp32
  (``inference_precision``): chi^2, Hessians and MCMC accept/reject need it.

Why the explicit pinning: on Ampere and newer GPUs JAX's *default* matmul precision
may use TF32 for float32 inputs, so "fp32" results would silently differ between an
RTX-class GPU and a T4 (which has no TF32).
"""

from __future__ import annotations

import contextlib

import jax
import jax.numpy as jnp

TRAIN_PRECISIONS = ("float32", "tensorfloat32", "bfloat16")
_MATMUL = {"float32": "highest", "tensorfloat32": "tensorfloat32", "bfloat16": "highest"}


def check_precision(name: str) -> str:
    if name not in TRAIN_PRECISIONS:
        raise ValueError(f"unknown precision {name!r}; choose from {TRAIN_PRECISIONS}")
    return name


def compute_dtype(name: str):
    """Layer compute dtype for a training precision (parameters always stay fp32)."""
    return jnp.bfloat16 if check_precision(name) == "bfloat16" else jnp.float32


@contextlib.contextmanager
def matmul_precision(name: str):
    """Trace/compile everything inside with the matmul precision of ``name``."""
    with jax.default_matmul_precision(_MATMUL[check_precision(name)]):
        yield


def inference_precision():
    """Full fp32 matmuls, regardless of GPU generation."""
    return matmul_precision("float32")
