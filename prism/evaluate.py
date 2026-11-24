"""Forward-model evaluation (port of Model-tester.ipynb).

* reduced chi^2 on the held-out test split, plain and imputation-masked
* imputed-vs-natural chi^2 per label
* median masked chi^2 per stellar population (paper Table tab:chi2_stellar)
* information leakage: |dF/d[X/H]| outside vs inside each element's line windows
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from prism.config import ATOMIC_LABELS, CNO_LABELS, CORE_LABELS, FEH_LABELS, LABELS_27, SELECTED_LABELS, TRACE_LABELS
from prism.losses import reduced_chi2
from prism.mesh import data_sharding, n_devices, pad_to_multiple, replicate
from prism.models.functional import make_flux_fn

STELLAR_GROUPS: dict[str, Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]] = {
    "RGB Tip": lambda t, g, f: (t < 4000) & (g < 1.5),
    "RGB Giants": lambda t, g, f: (t > 4000) & (t < 5000) & (g > 1.5) & (g < 3.5),
    "Subgiants": lambda t, g, f: (t > 4500) & (t < 6500) & (g > 3.5) & (g < 4.2),
    "Dwarfs": lambda t, g, f: g > 4.2,
    "Metal-poor Giants": lambda t, g, f: (f < -1) & (g < 3.5),
    "Hot Stars": lambda t, g, f: t > 6500,
}


def predict(model, labels: np.ndarray, mesh, batch: int = 512) -> np.ndarray:
    """Model output for all rows, in fixed-size sharded batches."""
    nd = n_devices(mesh)
    batch = -(-batch // nd) * nd
    fn, state = make_flux_fn(model)
    state = replicate(state, mesh)
    sharding = data_sharding(mesh)
    outs = []
    for s in range(0, len(labels), batch):
        chunk = np.asarray(labels[s:s + batch], np.float32)
        n = len(chunk)
        if n < batch:
            chunk, _ = pad_to_multiple(chunk, batch)
        out = fn(state, jax.device_put(chunk, sharding))
        outs.append(np.asarray(out)[:n])
    return np.concatenate(outs)


def _chi2_chunks(pred, flux, ivar, extra=None, chunk: int = 512) -> np.ndarray:
    res = []
    for s in range(0, len(pred), chunk):
        e = None if extra is None else extra[s:s + chunk]
        res.append(np.asarray(reduced_chi2(pred[s:s + chunk], flux[s:s + chunk], ivar[s:s + chunk], valid_extra=e)))
    return np.concatenate(res)


def chi2_table(models: dict, test: dict, apogee_mask: np.ndarray, mesh, threshold: float = 0.01) -> dict:
    """Plain and masked reduced chi^2 per model.

    Masked protocol: drop stars with any imputed core/CNO/FE_H label; for kept
    stars, remove the line pixels of every imputed trace element.
    """
    bad = np.asarray(test["bad_mask"], bool)
    drop_cols = [SELECTED_LABELS.index(lab) for lab in CORE_LABELS + CNO_LABELS + FEH_LABELS]
    valid = ~bad[:, drop_cols].any(axis=1)
    extra = np.ones((int(valid.sum()), apogee_mask.shape[0]), np.float32)
    bad_v = bad[valid]
    for lab in TRACE_LABELS:
        j = SELECTED_LABELS.index(lab)
        rows = bad_v[:, j]
        if rows.any():
            extra[np.ix_(rows, apogee_mask[:, j] > threshold)] = 0.0

    out: dict = {"valid_stars": valid}
    for name, model in models.items():
        pred = predict(model, test["labels"], mesh)
        out[name] = {
            "chi2": _chi2_chunks(pred, test["flux"], test["ivar"]),
            "chi2_masked": _chi2_chunks(pred[valid], test["flux"][valid], test["ivar"][valid], extra),
        }
    return out


def imputed_vs_natural(chi2: np.ndarray, bad_mask: np.ndarray) -> list[dict]:
    rows = []
    for i, lab in enumerate(SELECTED_LABELS):
        imp = np.asarray(bad_mask[: len(chi2), i], bool)
        if imp.any() and (~imp).any():
            ci, cn = float(np.mean(chi2[imp])), float(np.mean(chi2[~imp]))
            rows.append({"label": lab, "n_imputed": int(imp.sum()), "chi2_imp": ci, "chi2_nat": cn, "ratio": ci / cn})
    return rows


def population_table(chi2_masked: dict[str, np.ndarray], phys_labels: np.ndarray, valid: np.ndarray) -> dict:
    """Median masked chi^2 per stellar group; ``phys_labels`` (N, 23) for all test stars."""
    p = np.asarray(phys_labels)[valid]
    teff, logg, feh = p[:, 0], p[:, 1], p[:, SELECTED_LABELS.index("FE_H")]
    table = {}
    for group, sel in STELLAR_GROUPS.items():
        mask = sel(teff, logg, feh)
        table[group] = {"N": int(mask.sum())}
        for name, chi2 in chi2_masked.items():
            table[group][name] = float(np.median(chi2[mask])) if mask.any() else float("nan")
    return table


def leakage(model, labels_norm: np.ndarray, bad_mask: np.ndarray, apogee_mask: np.ndarray,
            std: np.ndarray, batch: int = 512) -> dict[str, dict]:
    """Per element: mean |dF/dX| on forbidden pixels / mean on allowed pixels (physical units)."""
    fn, state = make_flux_fn(model, jit=False)

    @jax.jit
    def abs_jvp(s, x, col):
        t = jnp.zeros_like(x).at[:, col].set(1.0)
        return jnp.abs(jax.jvp(lambda xx: fn(s, xx), (x,), (t,))[1])

    results = {}
    for lab in ATOMIC_LABELS:
        col = LABELS_27.index(lab)
        allowed = apogee_mask[:, col] > 0
        if not allowed.any():
            continue
        rows = np.asarray(labels_norm, np.float32)[~np.asarray(bad_mask, bool)[:, col]]
        sums_f, sums_a, count = 0.0, 0.0, 0
        for s in range(0, len(rows), batch):
            chunk = rows[s:s + batch]
            n = len(chunk)
            chunk, _ = pad_to_multiple(chunk, batch) if n < batch else (chunk, None)
            g = np.asarray(abs_jvp(state, jnp.asarray(chunk), col))[:n] / std[col]
            sums_f += float(g[:, ~allowed].mean())
            sums_a += float(g[:, allowed].mean())
            count += 1
        f, a = sums_f / max(count, 1), sums_a / max(count, 1)
        results[lab] = {"forbidden": f, "allowed": a, "ratio": f / a if a > 0 else float("inf")}
    return results


def _quantile_bins(x: np.ndarray, min_per_bin: int = 200) -> np.ndarray:
    n_bins = max(5, len(x) // min_per_bin)
    return np.quantile(x, np.linspace(0, 1, n_bins + 1))


def binned_median_plots(x_dict: dict[str, np.ndarray], chi2_masked: dict[str, np.ndarray], out_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import binned_statistic

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for xname, x in x_dict.items():
        bins = _quantile_bins(x)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for name, y in chi2_masked.items():
            vals, edges, _ = binned_statistic(x, y, statistic="median", bins=bins)
            count, _, _ = binned_statistic(x, y, statistic="count", bins=bins)
            centres = 0.5 * (edges[:-1] + edges[1:])
            keep = count > 10
            ax.plot(centres[keep], vals[keep], "o-", label=name)
        ax.axhline(1.0, ls="--", color="black", lw=1)
        ax.set_xlabel(xname)
        ax.set_ylabel(r"Median reduced $\chi^2$")
        ax.set_title(fr"Median $\chi^2_{{red}}$ (masked) vs {xname}")
        ax.legend()
        fig.tight_layout()
        path = out_dir / f"median_chi2_vs_{xname}_masked.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)
    return paths


def leakage_plot(all_leakage: dict[str, dict], out_dir: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(ATOMIC_LABELS))
    width = 0.8 / max(len(all_leakage), 1)
    fig, ax = plt.subplots(figsize=(14, 5))
    for i, (name, leak) in enumerate(all_leakage.items()):
        ax.bar(x + i * width, [leak.get(el, {}).get("ratio", 0.0) for el in ATOMIC_LABELS], width, label=name)
    ax.set_xticks(x + width * (len(all_leakage) - 1) / 2)
    ax.set_xticklabels(ATOMIC_LABELS, rotation=45, ha="right")
    ax.set_ylabel("Leakage ratio (forbidden / allowed)")
    ax.set_title("Information leakage")
    ax.axhline(1.0, ls="--", color="grey", lw=0.8)
    ax.legend()
    fig.tight_layout()
    path = out_dir / "leakage_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
