"""Label pipeline: ASPCAP flags, train-only imputation, [X/Fe] -> [X/H],
engineered features, normalisation and stratification weights.

Ported from model-trainer.ipynb cell 2 and backward_model_hmc.py (audit-fixed).
"""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge

from prism.config import ABUNDANCE_INDICES, FE_H_INDEX, SELECTED_LABELS, SENTINEL_LABELS, STRUCTURAL_ZERO_LABELS

_ZERO_ERR = ("VMICRO", "VMACRO", "VSINI")


def _flag_column(meta, name: str, start: int, end: int) -> np.ndarray:
    flg = meta[name][start:end]
    if flg.dtype.names:
        flg = flg[flg.dtype.names[0]]
    if flg.dtype.kind == "V":
        flg = flg.view("<i4")
    return flg.astype(np.int64) != 0


def read_raw_flagged(h5, labels: list[str], start: int, end: int) -> tuple[np.ndarray, np.ndarray]:
    """Raw label values (float64) and a bad mask for rows ``[start, end)``.

    Bad = non-finite, ``<LABEL>_FLAG != 0``, or a ``< -5000`` sentinel for the
    core labels that have no flag column (including M_H). Labels in
    STRUCTURAL_ZERO_LABELS (VSINI) are not measured for giants by design: their
    NaNs become 0 and are not counted as bad.
    """
    meta = h5["metadata"]
    keys = meta.dtype.names
    raw = np.stack([meta[lab][start:end] for lab in labels], axis=1).astype(np.float64)
    for i, lab in enumerate(labels):
        if lab in STRUCTURAL_ZERO_LABELS:
            raw[np.isnan(raw[:, i]), i] = 0.0
    bad = ~np.isfinite(raw)
    for i, lab in enumerate(labels):
        flag = f"{lab}_FLAG"
        if flag in keys:
            bad[:, i] |= _flag_column(meta, flag, start, end)
        elif lab in SENTINEL_LABELS:
            bad[:, i] |= raw[:, i] < -5000
    return raw, bad


def fit_imputer(raw_train: np.ndarray, bad_train: np.ndarray, seed: int) -> IterativeImputer:
    vals = raw_train.astype(np.float64).copy()
    vals[bad_train] = np.nan
    imputer = IterativeImputer(estimator=BayesianRidge(), max_iter=10,
                               initial_strategy="median", random_state=seed)
    imputer.fit(vals)
    return imputer


def impute(imputer: IterativeImputer, raw: np.ndarray, bad: np.ndarray) -> np.ndarray:
    vals = raw.astype(np.float64).copy()
    vals[bad] = np.nan
    return imputer.transform(vals)


def to_x_h(labels23: np.ndarray) -> np.ndarray:
    """[X/Fe] -> [X/H] for every *_FE column (FE_H itself unchanged)."""
    out = np.array(labels23, dtype=np.float64, copy=True)
    out[:, ABUNDANCE_INDICES] += out[:, FE_H_INDEX:FE_H_INDEX + 1]
    return out


def engineer_features(labels23: np.ndarray) -> np.ndarray:
    """Append INV_TEFF, vbroad, C_O_diff, LOGPE -> (N, 27)."""
    idx = SELECTED_LABELS.index
    x = np.asarray(labels23, dtype=np.float64)
    eng = np.stack([
        5040.0 / x[:, idx("TEFF")],
        np.sqrt(x[:, idx("VMACRO")] ** 2 + x[:, idx("VSINI")] ** 2),
        x[:, idx("C_FE")] - x[:, idx("O_FE")],
        0.5 * x[:, idx("LOGG")] + 0.5 * x[:, idx("M_H")],
    ], axis=1)
    return np.concatenate([x, eng], axis=1)


def norm_stats(train27: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(train27, axis=0)
    std = np.std(train27, axis=0)
    std[std == 0] = 1.0
    return mean, std


def stratified_weights(labels_norm27: np.ndarray, ref_labels_norm27: np.ndarray | None = None,
                       bins: int = 15) -> np.ndarray:
    """Inverse 3-D (TEFF, LOGG, M_H) density weights.

    Histogram, 99th-percentile clip and mean normalisation all come from the
    reference set (default: the labels themselves), so validation weights use
    the training density.
    """
    ref = labels_norm27 if ref_labels_norm27 is None else ref_labels_norm27
    cols = [0, 1, 2]
    hist, edges = np.histogramdd(ref[:, cols], bins=bins)
    weights_map = 1.0 / (hist + 1.0)

    def lookup(arr):
        idx = [np.clip(np.digitize(arr[:, c], edges[k]) - 1, 0, bins - 1) for k, c in enumerate(cols)]
        return weights_map[idx[0], idx[1], idx[2]]

    ref_w = lookup(ref)
    clip = np.percentile(ref_w, 99)
    norm = np.mean(np.clip(ref_w, None, clip))
    return (np.clip(lookup(labels_norm27), None, clip) / norm).astype(np.float32)


def read_errors(h5, labels: list[str], start: int, end: int) -> np.ndarray:
    """ASPCAP ``<LABEL>_ERR`` for rows ``[start, end)``.

    Bad errors (flagged label, non-finite, negative sentinel) are replaced by
    the column's 95th percentile over the slice; broadening errors are zero.
    """
    meta = h5["metadata"]
    keys = meta.dtype.names
    n = end - start
    raw = np.zeros((n, len(labels)), np.float64)
    bad = np.zeros((n, len(labels)), bool)
    for i, lab in enumerate(labels):
        if lab in _ZERO_ERR or f"{lab}_ERR" not in keys:
            continue
        raw[:, i] = meta[f"{lab}_ERR"][start:end]
        bad[:, i] = ~np.isfinite(raw[:, i]) | (raw[:, i] < 0)
        if f"{lab}_FLAG" in keys:
            bad[:, i] |= _flag_column(meta, f"{lab}_FLAG", start, end)
    vals = raw.copy()
    vals[bad] = np.nan
    with warnings.catch_warnings():  # all-NaN columns warn; handled below
        warnings.simplefilter("ignore", RuntimeWarning)
        p95 = np.nanpercentile(vals, 95, axis=0)
    p95 = np.where(np.isfinite(p95), p95, 0.0)
    return np.where(np.isnan(vals), p95, vals)
