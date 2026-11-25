"""APOGEE line-window masks.

Convention: ``apogee_mask.npy`` is (8575, 27); values are line weights in
[0, 1], > threshold means the pixel lies in a line window of that label.
Ported from model-trainer.ipynb cells 1 and 10, the RegPayne notebook and the
inverse pipelines (audit-fixed).
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np

from prism.config import ABUND_INDICES, ATOMIC_LABELS, CNO_LABELS, LABELS_27, N_PIXELS, SELECTED_LABELS, TRACE_LABELS

_CNO_FILES = {"C_FE": "C.filt", "N_FE": "N.filt", "O_FE": "O.filt"}
N_PIXELS_FILT = 7514
N_REAL_PIXELS = N_PIXELS_FILT
# Detector chips (blue, green, red) as [start, stop) on the 8575-pixel grid; the rest are gaps
# (flux 0 in the data, never in the loss).
CHIPS: tuple[tuple[int, int], ...] = ((246, 3274), (3585, 6080), (6344, 8335))
# element_idx key for the union of the C/N/O windows (the CNO expert of Config.cno_split models)
CNO_KEY = "CNO"


def gap_mask() -> np.ndarray:
    """(8575,) True on chip-gap / edge pixels."""
    gap = np.ones(N_PIXELS, bool)
    for a, b in CHIPS:
        gap[a:b] = False
    return gap


def insert_gaps(x7514: np.ndarray) -> np.ndarray:
    """Map the 7514-pixel chip-gap-free grid onto the 8575-pixel detector grid."""
    x = np.asarray(x7514, dtype=np.float32)
    if x.shape[0] != N_PIXELS_FILT:
        raise ValueError(f"expected {N_PIXELS_FILT} values, got {x.shape[0]}")
    full = np.zeros(N_PIXELS, np.float32)
    off = 0
    for a, b in CHIPS:  # blue 0:3028, green 3028:5523, red 5523:7514
        full[a:b] = x[off:off + b - a]
        off += b - a
    return full


def load_apogee_mask(mask_dir: Path) -> np.ndarray:
    mask = np.load(Path(mask_dir) / "apogee_mask.npy").astype(np.float32)
    if mask.shape != (N_PIXELS, len(LABELS_27)):
        raise ValueError(f"apogee_mask.npy has shape {mask.shape}, expected ({N_PIXELS}, 27)")
    return mask


def load_cno_masks(mask_dir: Path, threshold: float) -> dict[str, np.ndarray]:
    return {lab: insert_gaps(np.loadtxt(Path(mask_dir) / fname)) > threshold
            for lab, fname in _CNO_FILES.items()}


def element_indices(apogee_mask: np.ndarray, threshold: float,
                    cno_masks: dict[str, np.ndarray] | None = None) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Active pixel indices and weights per atomic label (ATOMIC_LABELS order).

    With ``cno_masks`` (``load_cno_masks``) the dict also holds ``CNO_KEY``: the union of the
    C/N/O windows with unit weights — a pure routing mask for the CNO expert.
    """
    out = {}
    for lab in ATOMIC_LABELS:
        col = LABELS_27.index(lab)
        active = np.where(apogee_mask[:, col] > threshold)[0]
        if len(active):
            out[lab] = (active.astype(np.int32), apogee_mask[active, col].astype(np.float32))
    if cno_masks is not None:
        union = np.zeros(N_PIXELS, bool)
        for lab in CNO_LABELS:
            union |= np.asarray(cno_masks[lab], bool)
        active = np.where(union)[0].astype(np.int32)
        out[CNO_KEY] = (active, np.ones(len(active), np.float32))
    return out


def suppression_matrix(apogee_mask: np.ndarray, cno_masks: dict[str, np.ndarray], threshold: float) -> np.ndarray:
    """(23, 8575): row j marks the pixels whose loss is dropped when label j was imputed.

    Core labels and FE_H never suppress pixels (as in the TF trainer).
    """
    sup = np.zeros((len(SELECTED_LABELS), N_PIXELS), np.float32)
    for lab in CNO_LABELS:
        sup[SELECTED_LABELS.index(lab)] = cno_masks[lab]
    for lab in TRACE_LABELS:
        j = SELECTED_LABELS.index(lab)
        sup[j] = apogee_mask[:, j] > threshold
    return sup


def pixel_mask_from_bad(bad_mask, suppress):
    """Per-star keep-mask (B, 8575): 0 where any imputed label suppresses the pixel."""
    bad = jnp.asarray(bad_mask, jnp.float32)
    return 1.0 - jnp.clip(bad @ jnp.asarray(suppress, jnp.float32), 0.0, 1.0)


def line_mask_to_forbidden(apogee_mask: np.ndarray, threshold: float) -> np.ndarray:
    """1 where the pixel is NOT a line pixel of a label that has lines (RegPayne penalty)."""
    lines = apogee_mask > threshold
    has_lines = lines.any(axis=0, keepdims=True)
    return ((~lines) & has_lines).astype(np.float32)


def inference_element_masks(apogee_mask: np.ndarray, threshold: float) -> tuple[dict[int, np.ndarray], set[int]]:
    """Per-abundance pixel masks for 1-D element fits; columns without lines are skipped.

    No full-spectrum fallback: few-pixel elements (S_FE, CO_FE) are fit on
    their own pixels only.
    """
    masks: dict[int, np.ndarray] = {}
    skip: set[int] = set()
    for j in ABUND_INDICES:
        m = (apogee_mask[:, j] > threshold).astype(np.float32)
        n = int(m.sum())
        if n == 0:
            skip.add(j)
            print(f"  warning: {SELECTED_LABELS[j]} has no line pixels; keeping the CNN estimate")
        elif n < 10:
            print(f"  warning: {SELECTED_LABELS[j]} has only {n} line pixels")
        masks[j] = m
    return masks, skip


def element_prior_weights(element_masks: dict[int, np.ndarray], n_reference: float = 500.0) -> dict[int, float]:
    """MAP/Laplace prior up-weighting for few-pixel elements: clip(500/max(n,10), 0.1, 20)."""
    return {j: float(np.clip(n_reference / max(float(m.sum()), 10.0), 0.1, 20.0))
            for j, m in element_masks.items()}
