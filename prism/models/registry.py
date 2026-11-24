"""Name -> model constructor, shared by training, evaluation, inversion and checkpoints."""

from __future__ import annotations

from flax import nnx

from prism.models.cnn_predictor import HeteroscedasticCNNPredictor
from prism.models.payne import Payne
from prism.models.prism import Prism, PrismAblation

MODEL_NAMES = ("prism", "ablation", "payne", "expanded_payne", "reg_payne")  # forward emulators
CNN_NAME = "cnn"  # spectrum -> label warm-start predictor
NEEDS_ELEMENT_IDX = ("prism", "ablation")


def build_model(name: str, element_idx: dict | None, *, seed: int) -> nnx.Module:
    rngs = nnx.Rngs(seed)
    if name in NEEDS_ELEMENT_IDX and not element_idx:
        raise ValueError(f"model {name!r} needs element indices")
    if name == "prism":
        model = Prism(element_idx, rngs=rngs)
    elif name == "ablation":
        model = PrismAblation(element_idx, rngs=rngs)
    elif name in ("payne", "reg_payne"):
        model = Payne(2, 300, rngs=rngs)
    elif name == "expanded_payne":
        model = Payne(4, 985, rngs=rngs)
    elif name == CNN_NAME:
        model = HeteroscedasticCNNPredictor(rngs=rngs)
    else:
        raise ValueError(f"unknown model {name!r}; choose from {MODEL_NAMES}")
    model.prism_name = name
    return model
