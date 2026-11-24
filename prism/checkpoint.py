"""Model checkpoints: Orbax-saved NNX state + JSON metadata (+ element windows)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import jax
import numpy as np
import orbax.checkpoint as ocp
from flax import nnx


def _save_element_idx(path: Path, element_idx: dict) -> None:
    arrays = {}
    for lab, (ind, w) in element_idx.items():
        arrays[f"{lab}__idx"] = ind
        arrays[f"{lab}__w"] = w
    np.savez(path, order=np.array(list(element_idx)), **arrays)


def _load_element_idx(path: Path) -> dict:
    with np.load(path) as d:
        return {str(lab): (d[f"{lab}__idx"], d[f"{lab}__w"]) for lab in d["order"]}


def save_model(path: Path, model: nnx.Module, meta: dict, element_idx: dict | None = None) -> None:
    """Write ``path/{state, meta.json, element_idx.npz}``; overwrites an existing checkpoint."""
    path = Path(path).absolute()
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    state = jax.device_get(nnx.state(model))
    with ocp.StandardCheckpointer() as ckpt:
        ckpt.save(path / "state", state)
        ckpt.wait_until_finished()
    (path / "meta.json").write_text(json.dumps(meta, indent=1))
    if element_idx is not None:
        _save_element_idx(path / "element_idx.npz", element_idx)


def load_meta(path: Path) -> dict:
    return json.loads((Path(path) / "meta.json").read_text())


def load_model(path: Path) -> nnx.Module:
    from prism.models.registry import build_model

    path = Path(path).absolute()
    meta = load_meta(path)
    idx_file = path / "element_idx.npz"
    element_idx = _load_element_idx(idx_file) if idx_file.exists() else None
    model = build_model(meta["name"], element_idx, seed=int(meta.get("seed", 0)))
    target = jax.tree.map(ocp.utils.to_shape_dtype_struct, nnx.state(model))
    with ocp.StandardCheckpointer() as ckpt:
        restored = ckpt.restore(path / "state", target)
    nnx.update(model, restored)
    return model
