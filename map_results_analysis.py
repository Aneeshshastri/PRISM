"""
map_results_analysis.py — Load, report and visualise MAP inference results
==========================================================================
Designed to work with results.npz produced by backward_model_map.py.

Usage (standalone):
    python map_results_analysis.py --results path/to/results.npz

Usage (as a library):
    from map_results_analysis import load_map_results, report_map_results, visualise_map_results
    data    = load_map_results("map_results/results.npz")
    report  = report_map_results(data)
    figs    = visualise_map_results(data, save_dir="map_results/figures")
"""

import argparse
import os
import textwrap
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")              # safe default for scripts / Kaggle
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import LogNorm

# ─────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────

PRETTY_NAMES = {
    "TEFF": r"$T_{\rm eff}$ [K]",
    "LOGG": r"$\log g$",
    "M_H": "[M/H]",
    "VMICRO": r"$v_{\rm micro}$ [km/s]",
    "VMACRO": r"$v_{\rm macro}$ [km/s]",
    "VSINI": r"$v \sin i$ [km/s]",
    "C_FE": "[C/Fe]",
    "N_FE": "[N/Fe]",
    "O_FE": "[O/Fe]",
    "FE_H": "[Fe/H]",
    "MG_FE": "[Mg/Fe]",
    "SI_FE": "[Si/Fe]",
    "CA_FE": "[Ca/Fe]",
    "TI_FE": "[Ti/Fe]",
    "S_FE": "[S/Fe]",
    "AL_FE": "[Al/Fe]",
    "MN_FE": "[Mn/Fe]",
    "NI_FE": "[Ni/Fe]",
    "CR_FE": "[Cr/Fe]",
    "K_FE": "[K/Fe]",
    "NA_FE": "[Na/Fe]",
    "V_FE": "[V/Fe]",
    "CO_FE": "[Co/Fe]",
}

CORE_LABELS  = ["TEFF", "LOGG", "M_H", "VMICRO", "VMACRO", "VSINI", "C_FE", "N_FE", "O_FE"]
ABUND_LABELS = ["FE_H", "MG_FE", "SI_FE", "CA_FE", "TI_FE", "S_FE",
                "AL_FE", "MN_FE", "NI_FE", "CR_FE", "K_FE", "NA_FE", "V_FE", "CO_FE"]

TEFF_SCALE = 100.0   # report Teff residuals in units of 100 K


# ─────────────────────────────────────────────────────────────────
# 1. LOAD
# ─────────────────────────────────────────────────────────────────

def load_map_results(path: str) -> dict:
    """
    Load a MAP results.npz file and return a standardised dict.

    Returns
    -------
    dict with keys:
        label_names      : list[str]  — 23 label names
        n_stars          : int
        n_labels         : int
        true_labels      : (N, 23) array, physical units
        inferred_labels  : (N, 23) array, physical units
        residuals        : (N, 23) array  (inferred − true)
        aspcap_errors    : (N, 23) array
        wall_seconds     : (N,) array
        global_indices   : (N,) array
    """
    path = str(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Results file not found: {path}")

    raw = np.load(path, allow_pickle=True)

    label_names = [s.item() if isinstance(s, np.generic) else str(s)
                   for s in raw["label_names"]]
    true      = raw["true_labels"].astype(np.float64)
    inferred  = raw["inferred_labels"].astype(np.float64)
    aspcap_err = raw["aspcap_errors"].astype(np.float64)
    wall      = raw["wall_seconds"].astype(np.float64)
    gidx      = raw["global_indices"]

    return {
        "label_names":     label_names,
        "n_stars":         len(gidx),
        "n_labels":        len(label_names),
        "true_labels":     true,
        "inferred_labels": inferred,
        "residuals":       inferred - true,
        "aspcap_errors":   aspcap_err,
        "wall_seconds":    wall,
        "global_indices":  gidx,
    }


# ─────────────────────────────────────────────────────────────────
# 2. REPORT
# ─────────────────────────────────────────────────────────────────

def _iqr(x):
    """Interquartile range."""
    return np.percentile(x, 75) - np.percentile(x, 25)


def report_map_results(data: dict, print_report: bool = True) -> dict:
    """
    Compute per-label summary statistics and optionally print a table.

    Returns
    -------
    dict  label_name → {bias, mad, rmse, iqr, median_aspcap_err, n_stars}
    """
    label_names = data["label_names"]
    residuals   = data["residuals"]
    aspcap_err  = data["aspcap_errors"]
    n_stars     = data["n_stars"]
    wall        = data["wall_seconds"]

    report = {}
    for j, name in enumerate(label_names):
        r = residuals[:, j]
        report[name] = {
            "bias":              np.median(r),
            "mad":               np.median(np.abs(r - np.median(r))),
            "rmse":              np.sqrt(np.mean(r**2)),
            "iqr":               _iqr(r),
            "median_aspcap_err": np.median(aspcap_err[:, j]),
            "n_stars":           n_stars,
        }

    if print_report:
        hdr  = f"{'Label':<10} {'Bias':>10} {'MAD':>10} {'RMSE':>10} {'IQR':>10} {'ASPCAP σ':>10}"
        rule = "─" * len(hdr)
        lines = [
            "",
            "╔══════════════════════════════════════════════════════════════════╗",
            "║            MAP Results — Summary Statistics                     ║",
            f"║  {n_stars} stars,  median wall-time = {np.median(wall):.2f} s/star{' '*(21 - len(f'{np.median(wall):.2f}'))}║",
            "╚══════════════════════════════════════════════════════════════════╝",
            "",
            "  Core physics parameters",
            f"  {rule}",
            f"  {hdr}",
            f"  {rule}",
        ]
        for name in CORE_LABELS:
            if name not in report:
                continue
            s = report[name]
            lines.append(
                f"  {PRETTY_NAMES.get(name, name):<10} "
                f"{s['bias']:>+10.4f} {s['mad']:>10.4f} {s['rmse']:>10.4f} "
                f"{s['iqr']:>10.4f} {s['median_aspcap_err']:>10.4f}"
            )
        lines += [
            f"  {rule}",
            "",
            "  Individual abundances",
            f"  {rule}",
            f"  {hdr}",
            f"  {rule}",
        ]
        for name in ABUND_LABELS:
            if name not in report:
                continue
            s = report[name]
            lines.append(
                f"  {PRETTY_NAMES.get(name, name):<10} "
                f"{s['bias']:>+10.4f} {s['mad']:>10.4f} {s['rmse']:>10.4f} "
                f"{s['iqr']:>10.4f} {s['median_aspcap_err']:>10.4f}"
            )
        lines.append(f"  {rule}")
        print("\n".join(lines))

    return report


# ─────────────────────────────────────────────────────────────────
# 3. VISUALISE
# ─────────────────────────────────────────────────────────────────

# ---- colour palette (publication-quality) ----
_C_CORE   = "#2c7bb6"       # blue
_C_ABUND  = "#d7191c"       # red
_C_HIST   = "#636363"       # grey
_C_UNITY  = "#1a1a1a"       # dark
_C_BG     = "#fafafa"


def _nice_label(name):
    return PRETTY_NAMES.get(name, name)


def _save(fig, save_dir, stem):
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(save_dir, f"{stem}.{ext}"),
                        dpi=200, bbox_inches="tight")


# ---- (a) 1-to-1 scatter grid ----

def plot_one_to_one(data, save_dir=None):
    """23-panel grid: inferred vs true for every label."""
    label_names = data["label_names"]
    true        = data["true_labels"]
    inferred    = data["inferred_labels"]
    n_labels    = len(label_names)
    n_cols      = 6
    n_rows      = int(np.ceil(n_labels / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.2, n_rows * 3.2))
    fig.patch.set_facecolor("white")
    axes = axes.ravel()

    for j in range(n_labels):
        ax  = axes[j]
        t   = true[:, j]
        inf = inferred[:, j]
        name = label_names[j]
        c = _C_CORE if name in CORE_LABELS else _C_ABUND

        ax.scatter(t, inf, s=12, alpha=0.65, edgecolors="none", c=c, zorder=2)

        lo = min(t.min(), inf.min())
        hi = max(t.max(), inf.max())
        margin = 0.05 * (hi - lo) if hi > lo else 0.5
        ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin],
                ls="--", lw=1, c=_C_UNITY, alpha=0.5, zorder=1)
        ax.set_xlim(lo - margin, hi + margin)
        ax.set_ylim(lo - margin, hi + margin)
        ax.set_xlabel("ASPCAP", fontsize=8)
        ax.set_ylabel("MAP", fontsize=8)
        ax.set_title(_nice_label(name), fontsize=9, fontweight="bold")
        ax.tick_params(labelsize=7)
        ax.set_aspect("equal", adjustable="box")

    for j in range(n_labels, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("MAP Inference — Inferred vs ASPCAP (True)", fontsize=14,
                 fontweight="bold", y=1.01)
    fig.tight_layout()
    _save(fig, save_dir, "map_one_to_one")
    return fig


# ---- (b) residual histograms ----

def plot_residual_histograms(data, save_dir=None):
    """23-panel grid: residual (inferred − true) histogram per label."""
    label_names = data["label_names"]
    residuals   = data["residuals"]
    n_labels    = len(label_names)
    n_cols      = 6
    n_rows      = int(np.ceil(n_labels / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.0, n_rows * 2.8))
    fig.patch.set_facecolor("white")
    axes = axes.ravel()

    for j in range(n_labels):
        ax   = axes[j]
        r    = residuals[:, j]
        name = label_names[j]
        c = _C_CORE if name in CORE_LABELS else _C_ABUND

        n_bins = max(10, min(50, int(np.sqrt(len(r)))))
        ax.hist(r, bins=n_bins, color=c, alpha=0.75, edgecolor="white", linewidth=0.5)
        ax.axvline(0, ls="--", lw=1.0, c=_C_UNITY, alpha=0.5)
        ax.axvline(np.median(r), ls="-", lw=1.2, c="k", alpha=0.8, label=f"med={np.median(r):.3f}")

        ax.set_xlabel("Residual", fontsize=8)
        ax.set_title(_nice_label(name), fontsize=9, fontweight="bold")
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6, loc="upper right")
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    for j in range(n_labels, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("MAP Inference — Residual Distributions (Inferred − ASPCAP)",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    _save(fig, save_dir, "map_residual_histograms")
    return fig


# ---- (c) bias + scatter bar chart ----

def plot_bias_scatter_bars(data, save_dir=None):
    """Horizontal bar chart: median bias ± MAD for every label."""
    label_names = data["label_names"]
    residuals   = data["residuals"]
    n_labels    = len(label_names)

    biases = np.array([np.median(residuals[:, j]) for j in range(n_labels)])
    mads   = np.array([np.median(np.abs(residuals[:, j] - biases[j])) for j in range(n_labels)])

    order = np.arange(n_labels)[::-1]

    fig, ax = plt.subplots(figsize=(7, 0.45 * n_labels + 1.5))
    fig.patch.set_facecolor("white")
    colours = [_C_CORE if label_names[j] in CORE_LABELS else _C_ABUND for j in order]

    ax.barh(range(n_labels), biases[order], xerr=mads[order],
            color=colours, alpha=0.8, edgecolor="white", linewidth=0.5,
            error_kw=dict(lw=1, capsize=3, capthick=1, ecolor="#333"))
    ax.axvline(0, ls="-", lw=0.8, c=_C_UNITY, alpha=0.5)
    ax.set_yticks(range(n_labels))
    ax.set_yticklabels([_nice_label(label_names[j]) for j in order], fontsize=8)
    ax.set_xlabel("Median bias  ±  MAD", fontsize=10)
    ax.set_title("MAP Inference — Per-Label Bias & Scatter", fontsize=13, fontweight="bold")
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    _save(fig, save_dir, "map_bias_scatter_bars")
    return fig


# ---- (d) Teff-coloured residual map ----

def plot_residual_vs_teff(data, save_dir=None):
    """
    For each abundance, scatter residual vs Teff  
    to reveal temperature-dependent biases.
    """
    label_names = data["label_names"]
    true        = data["true_labels"]
    residuals   = data["residuals"]

    teff_idx = label_names.index("TEFF")
    teff     = true[:, teff_idx]

    abund_indices = [j for j, n in enumerate(label_names) if n in ABUND_LABELS]
    n = len(abund_indices)
    n_cols = 5
    n_rows = int(np.ceil(n / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.2, n_rows * 2.8),
                              sharey=False)
    fig.patch.set_facecolor("white")
    axes = axes.ravel()

    for k, j in enumerate(abund_indices):
        ax   = axes[k]
        r    = residuals[:, j]
        name = label_names[j]
        sc = ax.scatter(teff, r, c=teff, cmap="RdYlBu_r", s=10, alpha=0.7,
                        edgecolors="none")
        ax.axhline(0, ls="--", lw=0.8, c=_C_UNITY, alpha=0.4)
        ax.set_xlabel(r"$T_{\rm eff}$ [K]", fontsize=7)
        ax.set_ylabel("Residual", fontsize=7)
        ax.set_title(_nice_label(name), fontsize=9, fontweight="bold")
        ax.tick_params(labelsize=6)

    for k in range(n, len(axes)):
        axes[k].set_visible(False)

    fig.suptitle("MAP Residuals vs $T_{\\rm eff}$  — Temperature-Dependent Biases",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    _save(fig, save_dir, "map_residual_vs_teff")
    return fig


# ---- (e) residual heatmap (label × star) ----

def plot_residual_heatmap(data, save_dir=None):
    """
    Heatmap of absolute residuals:  rows = labels,  cols = stars.
    Helps spot outlier stars and systematically hard labels.
    """
    label_names = data["label_names"]
    residuals   = data["residuals"]
    n_stars     = data["n_stars"]

    # Normalise residuals by MAD per label for comparability
    mads = np.array([np.median(np.abs(residuals[:, j] - np.median(residuals[:, j])))
                     for j in range(len(label_names))])
    mads = np.where(mads < 1e-8, 1.0, mads)
    normed = np.abs(residuals) / mads[None, :]    # (n_stars, n_labels)

    fig, ax = plt.subplots(figsize=(max(6, 0.15 * n_stars + 2), 7))
    fig.patch.set_facecolor("white")

    im = ax.imshow(normed.T, aspect="auto", cmap="inferno",
                   interpolation="nearest", vmin=0, vmax=5)
    ax.set_yticks(range(len(label_names)))
    ax.set_yticklabels([_nice_label(n) for n in label_names], fontsize=7)
    ax.set_xlabel("Star index", fontsize=10)
    ax.set_title("MAP |Residuals|  /  MAD   (per label)", fontsize=13, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("× MAD", fontsize=9)
    fig.tight_layout()
    _save(fig, save_dir, "map_residual_heatmap")
    return fig


# ---- (f) wall-time histogram ----

def plot_wall_time(data, save_dir=None):
    """Histogram of per-star wall-clock time."""
    wall = data["wall_seconds"]

    fig, ax = plt.subplots(figsize=(6, 3.5))
    fig.patch.set_facecolor("white")
    ax.hist(wall, bins=max(10, min(40, len(wall) // 3)),
            color="#636363", alpha=0.8, edgecolor="white", linewidth=0.5)
    ax.axvline(np.median(wall), ls="-", lw=1.5, c=_C_CORE,
               label=f"median = {np.median(wall):.2f} s")
    ax.set_xlabel("Wall time per star [s]", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title("MAP Inference — Per-Star Timing", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    fig.tight_layout()
    _save(fig, save_dir, "map_wall_time")
    return fig


# ---- (g) RMSE comparison: MAP vs ASPCAP pipeline errors ----

def plot_rmse_vs_aspcap(data, save_dir=None):
    """
    Side-by-side bar chart: RMSE of MAP residuals vs median
    ASPCAP pipeline error, per label.
    """
    label_names = data["label_names"]
    residuals   = data["residuals"]
    aspcap_err  = data["aspcap_errors"]
    n_labels    = len(label_names)

    rmses       = np.array([np.sqrt(np.mean(residuals[:, j]**2)) for j in range(n_labels)])
    med_aspcap  = np.array([np.median(aspcap_err[:, j]) for j in range(n_labels)])

    x = np.arange(n_labels)
    w = 0.35

    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor("white")
    ax.bar(x - w/2, rmses, w, label="MAP RMSE", color=_C_CORE, alpha=0.85, edgecolor="white")
    ax.bar(x + w/2, med_aspcap, w, label="ASPCAP median σ", color=_C_ABUND, alpha=0.65, edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels([_nice_label(n) for n in label_names], fontsize=7, rotation=45, ha="right")
    ax.set_ylabel("Error scale", fontsize=10)
    ax.set_title("MAP RMSE  vs  ASPCAP Pipeline Errors", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    fig.tight_layout()
    _save(fig, save_dir, "map_rmse_vs_aspcap")
    return fig


# ---- master function ----

def visualise_map_results(data: dict, save_dir: str = None) -> dict:
    """
    Generate all diagnostic plots.

    Parameters
    ----------
    data      : dict returned by load_map_results()
    save_dir  : if not None, each figure is saved as PNG + PDF here

    Returns
    -------
    dict  name → matplotlib.figure.Figure
    """
    figs = {}
    figs["one_to_one"]           = plot_one_to_one(data, save_dir)
    figs["residual_histograms"]  = plot_residual_histograms(data, save_dir)
    figs["bias_scatter_bars"]    = plot_bias_scatter_bars(data, save_dir)
    figs["residual_vs_teff"]     = plot_residual_vs_teff(data, save_dir)
    figs["residual_heatmap"]     = plot_residual_heatmap(data, save_dir)
    figs["wall_time"]            = plot_wall_time(data, save_dir)
    figs["rmse_vs_aspcap"]       = plot_rmse_vs_aspcap(data, save_dir)

    if save_dir:
        print(f"\n  📊  7 figures saved to {save_dir}/")
    return figs


# ─────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Load, report and visualise MAP inference results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Example
            -------
              python map_results_analysis.py --results map_results/results.npz
              python map_results_analysis.py --results map_results/results.npz --save-dir map_results/figures
        """),
    )
    parser.add_argument("--results", required=True, help="Path to results.npz from backward_model_map.py")
    parser.add_argument("--save-dir", default=None, help="Directory to save figures (PNG + PDF)")
    args = parser.parse_args()

    data   = load_map_results(args.results)
    report = report_map_results(data)
    figs   = visualise_map_results(data, save_dir=args.save_dir)

    print(f"\nDone. {len(figs)} figures generated.")


if __name__ == "__main__":
    main()
