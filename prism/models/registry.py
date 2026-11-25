"""Name -> model constructor, shared by training, evaluation, inversion and checkpoints."""

from __future__ import annotations

import jax.numpy as jnp
from flax import nnx

from prism.models.cnn_predictor import HeteroscedasticCNNPredictor
from prism.models.payne import Payne
from prism.models.prism import Prism, PrismAblation

MODEL_NAMES = ("prism", "ablation", "payne", "expanded_payne", "reg_payne")  # forward emulators
CNN_NAME = "cnn"  # spectrum -> label warm-start predictor
NEEDS_ELEMENT_IDX = ("prism", "ablation")
# Architecture options (see prism.models.prism). Missing keys = legacy behaviour, which is what
# checkpoints written before the options existed were trained with.
LEGACY_ARCH = {"drop_gaps": False, "chip_continuum": False, "cno_split": False}


class Arch(dict):
    """Hashable architecture options: stored on the module as static data, so NNX graph
    definitions (jit cache keys) stay hashable. Compares equal to a plain dict."""

    def __hash__(self):
        return hash(tuple(sorted(self.items())))

    def __setitem__(self, key, value):
        raise TypeError("architecture options are immutable")


def resolve_arch(arch: dict | None) -> Arch:
    unknown = set(arch or {}) - set(LEGACY_ARCH)
    if unknown:
        raise ValueError(f"unknown architecture options {sorted(unknown)}; known: {sorted(LEGACY_ARCH)}")
    return Arch({**LEGACY_ARCH, **(arch or {})})


def build_model(name: str, element_idx: dict | None, *, seed: int, dtype=jnp.float32,
                arch: dict | None = None) -> nnx.Module:
    """``dtype`` is the layer compute dtype (bf16 for mixed-precision training); parameters are fp32.
    ``arch`` (``Config.arch()``) defaults to the legacy architecture.

    Checkpoints don't record dtype, so ``load_model`` always rebuilds an fp32-compute model."""
    rngs = nnx.Rngs(seed)
    arch = resolve_arch(arch)
    if name in NEEDS_ELEMENT_IDX and not element_idx:
        raise ValueError(f"model {name!r} needs element indices")
    if name == "prism":
        model = Prism(element_idx, **arch, dtype=dtype, rngs=rngs)
    elif name == "ablation":
        model = PrismAblation(element_idx, **arch, dtype=dtype, rngs=rngs)
    elif name in ("payne", "reg_payne"):
        model = Payne(2, 300, drop_gaps=arch["drop_gaps"], dtype=dtype, rngs=rngs)
    elif name == "expanded_payne":
        model = Payne(4, 985, drop_gaps=arch["drop_gaps"], dtype=dtype, rngs=rngs)
    elif name == CNN_NAME:
        model = HeteroscedasticCNNPredictor(dtype=dtype, rngs=rngs)
    else:
        raise ValueError(f"unknown model {name!r}; choose from {MODEL_NAMES}")
    model.prism_name = name
    model.prism_arch = arch
    return model
