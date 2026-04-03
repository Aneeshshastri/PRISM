"""
map_fixes.py — All fixes for backward_model_map.py + outlier analysis
======================================================================

This file is a REFERENCE.  Copy the relevant sections into
backward_model_map.py (and map_results_analysis.py) yourself.

Sections
--------
  FIX 1 — Hyperparameter updates  (replace lines 283-288)
  FIX 2 — Cosine-decay learning-rate schedule for core optimizer  (replace map_optimize_core)
  FIX 3 — Cosine-decay + scaled prior for element optimizer  (replace map_optimize_element)
  FIX 4 — Per-element prior-weight lookup  (add after line 310)
  FIX 5 — Record per-star final loss in checkpoint  (update checkpoint dict + pipeline loop)
  FIX 6 — Outlier flagging & diagnostic plots  (add to map_results_analysis.py)
"""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

tfb = tfp.bijectors


# ════════════════════════════════════════════════════════════════════
# FIX 1 — Hyperparameters
# ════════════════════════════════════════════════════════════════════
# Replace lines 283-288 in backward_model_map.py with:

N_STEPS_CORE = 500          # was 300  — more headroom for Teff convergence
N_STEPS_ELEM = 400          # was 150  — critical: 150 is too few for sigmoid bijector
LR_CORE_INIT = 0.03         # was 0.01 — start higher, cosine-decay handles the rest
LR_CORE_END  = 1e-4         # final lr after decay
LR_ELEM_INIT = 0.03         # was 0.01
LR_ELEM_END  = 1e-4
PRIOR_WEIGHT_CORE = 1.0     # unchanged — core has full-spectrum chi2, prior is fine
# Element prior weights are now PER-ELEMENT — see FIX 4


# ════════════════════════════════════════════════════════════════════
# FIX 2 — Core optimizer with cosine-decay LR
# ════════════════════════════════════════════════════════════════════
# Replace the entire map_optimize_core function (lines 336-401).

# These constants must be visible at module level (they're used inside
# the @tf.function so they get captured as Python ints / floats).
_N_STEPS_CORE = N_STEPS_CORE
_LR_CORE_INIT = LR_CORE_INIT
_LR_CORE_END  = LR_CORE_END

# You still need these from the original file:
#   _core_bij, CORE_INDICES, ABUND_INDICES, N_LABELS_RAW,
#   MEAN_TENSOR, STD_TENSOR, model_pure_fp32, get_27_features,
#   N_PIXELS, PRIOR_WEIGHT_CORE

# @tf.function(jit_compile=True)
def map_optimize_core_FIXED(theta_unc, obs_flux, obs_ivar,
                             fixed_abund, cnn_mu, cnn_std):
    """
    Batched Adam MAP for 9D core physics — WITH cosine LR decay.

    Changes vs original
    -------------------
    * N_STEPS_CORE  300 → 500
    * Learning rate  0.01 constant → 0.03 cosine-decayed to 1e-4
    * Returns (theta_phys, final_loss)  so we can flag non-converged stars
    """
    CORE_INDICES_PY  = [0,1,2,3,4,5,6,7,8]
    ABUND_INDICES_PY = [9,10,11,12,13,14,15,16,17,18,19,20,21,22]
    N_LABELS_RAW_PY  = 23
    PRIOR_WEIGHT     = 1.0

    core_cnn_mu  = tf.gather(cnn_mu, CORE_INDICES_PY, axis=1)
    core_cnn_std = tf.gather(cnn_std, CORE_INDICES_PY, axis=1)

    lr_init = tf.constant(_LR_CORE_INIT, tf.float32)
    lr_end  = tf.constant(_LR_CORE_END, tf.float32)
    beta1   = tf.constant(0.9, tf.float32)
    beta2   = tf.constant(0.999, tf.float32)
    eps     = tf.constant(1e-7, tf.float32)
    n_total = tf.constant(float(_N_STEPS_CORE), tf.float32)

    m     = tf.zeros_like(theta_unc)
    v     = tf.zeros_like(theta_unc)
    theta = theta_unc

    for step in tf.range(_N_STEPS_CORE):
        # ---- cosine-decay learning rate ----
        t_f  = tf.cast(step, tf.float32)
        frac = t_f / n_total                              # 0 → 1
        lr   = lr_end + 0.5 * (lr_init - lr_end) * (1.0 + tf.cos(frac * 3.14159265))

        with tf.GradientTape() as tape:
            tape.watch(theta)
            # theta_phys = _core_bij.forward(theta)       # ← use your existing bijector
            #
            # --- assemble full_23 exactly as before ---
            # parts = []
            # for i in range(N_LABELS_RAW_PY):
            #     if i in CORE_INDICES_PY:
            #         ci = CORE_INDICES_PY.index(i)
            #         parts.append(theta_phys[:, ci:ci+1])
            #     else:
            #         ai = ABUND_INDICES_PY.index(i)
            #         parts.append(fixed_abund[:, ai:ai+1])
            # full_23 = tf.concat(parts, axis=1)
            #
            # labels_27   = get_27_features(full_23)
            # labels_norm = (labels_27 - MEAN_TENSOR) / STD_TENSOR
            # model_flux  = tf.cast(model_pure_fp32(labels_norm, training=False), tf.float32)
            # model_flux  = tf.squeeze(model_flux, axis=-1)
            #
            # safe_flux = tf.where(tf.math.is_finite(obs_flux), obs_flux, 0.0)
            # safe_ivar = tf.where(
            #     tf.math.is_finite(obs_ivar) & tf.math.is_finite(obs_flux),
            #     obs_ivar, 0.0)
            # safe_ivar = 1.0 / (1.0 / (safe_ivar + 1e-12) + 1e-5)
            # fv   = tf.cast((safe_flux > 1e-3) & (safe_flux < 1.3), tf.float32)
            # iv   = tf.cast(safe_ivar > 0.0, tf.float32)
            # mask = fv * iv

            # chi2  = tf.reduce_sum(tf.square(safe_flux - model_flux) * safe_ivar * mask, axis=1)
            # prior = tf.reduce_sum(
            #     tf.square((theta_phys - core_cnn_mu) / core_cnn_std), axis=1)
            # loss  = tf.reduce_mean(0.5 * chi2 + PRIOR_WEIGHT * 0.5 * prior)
            #
            # ---- THE ABOVE IS IDENTICAL TO YOUR ORIGINAL ----
            # ---- only `lr` and `n_steps` change          ----
            loss = tf.constant(0.0)   # placeholder — remove this line
            pass

        grad = tape.gradient(loss, theta)
        t_adam = tf.cast(step + 1, tf.float32)
        m = beta1 * m + (1.0 - beta1) * grad
        v = beta2 * v + (1.0 - beta2) * tf.square(grad)
        m_hat = m / (1.0 - tf.pow(beta1, t_adam))
        v_hat = v / (1.0 - tf.pow(beta2, t_adam))
        theta = theta - lr * m_hat / (tf.sqrt(v_hat) + eps)

    # ---- NEW: return final per-star loss alongside the result ----
    # final_loss is (B,) — used later to flag outlier stars
    # return _core_bij.forward(theta), loss     # ← uncomment in real code
    pass


# ════════════════════════════════════════════════════════════════════
# FIX 3 — Element optimizer with cosine-decay LR + scaled prior
# ════════════════════════════════════════════════════════════════════
# Replace the entire map_optimize_element function (lines 404-467).

_N_STEPS_ELEM = N_STEPS_ELEM
_LR_ELEM_INIT = LR_ELEM_INIT
_LR_ELEM_END  = LR_ELEM_END

# @tf.function(jit_compile=True)
def map_optimize_element_FIXED(theta_unc_1d, obs_flux, obs_ivar,
                                fixed_full_23, elem_col, pixel_mask,
                                elem_lo, elem_hi, elem_cnn_mu, elem_cnn_std,
                                prior_weight):          # ← NEW argument
    """
    Batched Adam MAP for a single 1D abundance — FIXED.

    Changes vs original
    -------------------
    * N_STEPS_ELEM  150 → 400
    * Learning rate  0.01 constant → 0.03 cosine-decayed to 1e-4
    * prior_weight is now a per-element tf.constant (see FIX 4)
      so weak-line elements get stronger regularisation
    * Returns (theta_phys, final_per_star_loss_vector)
    """
    N_PIXELS_PY = 8575

    lo = tf.reshape(elem_lo, [1])
    hi = tf.reshape(elem_hi, [1])
    bij = tfb.Chain([tfb.Shift(lo), tfb.Scale(hi - lo), tfb.Sigmoid()])

    lr_init = tf.constant(_LR_ELEM_INIT, tf.float32)
    lr_end  = tf.constant(_LR_ELEM_END, tf.float32)
    beta1   = tf.constant(0.9, tf.float32)
    beta2   = tf.constant(0.999, tf.float32)
    eps     = tf.constant(1e-7, tf.float32)
    n_total = tf.constant(float(_N_STEPS_ELEM), tf.float32)

    m     = tf.zeros_like(theta_unc_1d)
    v     = tf.zeros_like(theta_unc_1d)
    theta = theta_unc_1d

    for step in tf.range(_N_STEPS_ELEM):
        # ---- cosine-decay learning rate ----
        t_f  = tf.cast(step, tf.float32)
        frac = t_f / n_total
        lr   = lr_end + 0.5 * (lr_init - lr_end) * (1.0 + tf.cos(frac * 3.14159265))

        with tf.GradientTape() as tape:
            tape.watch(theta)
            theta_phys = bij.forward(theta)  # (B, 1)

            bc = tf.shape(fixed_full_23)[0]
            indices = tf.stack([tf.range(bc), tf.fill([bc], elem_col)], axis=1)
            full_spliced = tf.tensor_scatter_nd_update(
                fixed_full_23, indices, theta_phys[:, 0]
            )

            # labels_27   = get_27_features(full_spliced)
            # labels_norm = (labels_27 - MEAN_TENSOR) / STD_TENSOR
            # model_flux  = tf.cast(model_pure_fp32(labels_norm, training=False), tf.float32)
            # model_flux  = tf.squeeze(model_flux, axis=-1)

            # safe_flux = tf.where(tf.math.is_finite(obs_flux), obs_flux, 0.0)
            # safe_ivar = tf.where(
            #     tf.math.is_finite(obs_ivar) & tf.math.is_finite(obs_flux),
            #     obs_ivar, 0.0)
            # safe_ivar = 1.0 / (1.0 / (safe_ivar + 1e-12) + 1e-5)
            # fv   = tf.cast((safe_flux > 1e-3) & (safe_flux < 1.3), tf.float32)
            # iv   = tf.cast(safe_ivar > 0.0, tf.float32)
            # mask = fv * iv * tf.reshape(pixel_mask, [1, N_PIXELS_PY])

            # chi2 = tf.reduce_sum(tf.square(safe_flux - model_flux) * safe_ivar * mask, axis=1)

            # ──────────────────────────────────────────────────────
            #  KEY FIX: scale the prior by prior_weight (per-element)
            # ──────────────────────────────────────────────────────
            # prior = tf.reduce_sum(
            #     tf.square((theta_phys - elem_cnn_mu) / elem_cnn_std), axis=1
            # )
            # loss = tf.reduce_mean(0.5 * chi2 + prior_weight * 0.5 * prior)
            #                                    ^^^^^^^^^^^^
            #                                    was PRIOR_WEIGHT (=1.0 for all)
            loss = tf.constant(0.0)   # placeholder — remove
            pass

        grad = tape.gradient(loss, theta)
        t_adam = tf.cast(step + 1, tf.float32)
        m = beta1 * m + (1.0 - beta1) * grad
        v = beta2 * v + (1.0 - beta2) * tf.square(grad)
        m_hat = m / (1.0 - tf.pow(beta1, t_adam))
        v_hat = v / (1.0 - tf.pow(beta2, t_adam))
        theta = theta - lr * m_hat / (tf.sqrt(v_hat) + eps)

    # Return physical value + per-star loss for outlier flagging
    # final_phys = bij.forward(theta)[:, 0]
    # return final_phys, loss      # ← uncomment in real code
    pass


# ════════════════════════════════════════════════════════════════════
# FIX 4 — Per-element prior weight (scaled by sensitive pixel count)
# ════════════════════════════════════════════════════════════════════
# Add this block AFTER line 310 (after ELEMENT_PIXEL_MASKS is built).
#
# Rationale: chi2 sums over N_masked pixels while prior sums over 1.
# To balance them we set prior_weight ∝ N_masked / N_reference.
# N_reference = 500 is a typical "well-constrained" element.
# This means:
#   - A well-constrained element (500 pix) gets prior_weight ≈ 1
#   - A weak element (50 pix) gets prior_weight ≈ 10  → stronger regularisation
#   - A very strong element (2000 pix) gets prior_weight ≈ 0.25

N_REFERENCE_PIXELS = 500.0   # tune this — roughly the pixel count for Mg or Si

ELEMENT_PRIOR_WEIGHTS = {}
for idx in [9,10,11,12,13,14,15,16,17,18,19,20,21,22]:   # ABUND_INDICES
    # n_pix = float(ELEMENT_PIXEL_MASKS[idx].numpy().sum())  # ← uncomment in real code
    n_pix = 100.0  # placeholder
    pw = N_REFERENCE_PIXELS / max(n_pix, 10.0)
    pw = np.clip(pw, 0.1, 20.0)          # safety clamp
    ELEMENT_PRIOR_WEIGHTS[idx] = tf.constant(pw, tf.float32)
    # print(f"  {LABEL_NAMES[idx]:>6s}: {int(n_pix)} pix → prior_weight = {pw:.2f}")


# Then in the pipeline loop (line 588), pass the per-element weight:
#
#   e_result, e_loss = map_optimize_element_FIXED(
#       elem_init_unc, obs_flux_tf, obs_ivar_tf,
#       tf.constant(fixed_23, tf.float32), elem_idx,
#       ELEMENT_PIXEL_MASKS[elem_idx],
#       lo_tf, hi_tf,
#       cnn_mu_tf[:, elem_idx:elem_idx+1],
#       cnn_sig_tf[:, elem_idx:elem_idx+1],
#       ELEMENT_PRIOR_WEIGHTS[elem_idx],          # ← NEW
#   )


# ════════════════════════════════════════════════════════════════════
# FIX 5 — Record per-star final loss for outlier flagging
# ════════════════════════════════════════════════════════════════════
#
# 5a.  Update the checkpoint dict (line 474) to include 'final_loss':
#
#   fresh = {
#       'global_indices': [], 'true_labels': [], 'inferred_labels': [],
#       'aspcap_errors': [], 'wall_seconds': [],
#       'core_loss': [],              # ← NEW
#       'elem_loss': [],              # ← NEW  (14-vector per star)
#   }
#
# 5b.  In run_inference_pipeline, after calling map_optimize_core_FIXED:
#
#   core_result, core_final_loss = map_optimize_core_FIXED(...)
#                                  ^^^^^^^^^^^^^^^^^
#   core_loss_np = core_final_loss.numpy()   # (B,) per-star chi2+prior
#
# 5c.  Collect element losses too:
#
#   elem_losses = np.zeros((BATCH_SIZE_STARS, N_ABUND), np.float32)
#   for ai, elem_idx in enumerate(ABUND_INDICES):
#       e_result, e_loss = map_optimize_element_FIXED(...)
#       elem_losses[:, ai] = e_loss.numpy()
#       ...
#
# 5d.  Append in the per-star loop:
#
#   results['core_loss'].append(core_loss_np[b])
#   results['elem_loss'].append(elem_losses[b])
#
# 5e.  Update save_final_results to include both new keys.


# ════════════════════════════════════════════════════════════════════
# FIX 6 — Outlier flagging & diagnostic plots
# ════════════════════════════════════════════════════════════════════
# Add these functions to map_results_analysis.py (or import this file).

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import os

PRETTY_NAMES = {
    "TEFF": r"$T_{\rm eff}$ [K]", "LOGG": r"$\log g$", "M_H": "[M/H]",
    "VMICRO": r"$v_{\rm micro}$", "VMACRO": r"$v_{\rm macro}$",
    "VSINI": r"$v \sin i$",
    "C_FE": "[C/Fe]", "N_FE": "[N/Fe]", "O_FE": "[O/Fe]", "FE_H": "[Fe/H]",
    "MG_FE": "[Mg/Fe]", "SI_FE": "[Si/Fe]", "CA_FE": "[Ca/Fe]",
    "TI_FE": "[Ti/Fe]", "S_FE": "[S/Fe]", "AL_FE": "[Al/Fe]",
    "MN_FE": "[Mn/Fe]", "NI_FE": "[Ni/Fe]", "CR_FE": "[Cr/Fe]",
    "K_FE": "[K/Fe]", "NA_FE": "[Na/Fe]", "V_FE": "[V/Fe]", "CO_FE": "[Co/Fe]",
}

CORE_LABELS  = ["TEFF","LOGG","M_H","VMICRO","VMACRO","VSINI","C_FE","N_FE","O_FE"]
ABUND_LABELS = ["FE_H","MG_FE","SI_FE","CA_FE","TI_FE","S_FE",
                "AL_FE","MN_FE","NI_FE","CR_FE","K_FE","NA_FE","V_FE","CO_FE"]


def flag_outliers(data, sigma_thresh=3.0):
    """
    Flag outlier stars using a per-label robust z-score.

    A star is an outlier if |residual - median| > sigma_thresh * MAD
    for ANY label.  Returns a boolean mask (n_stars,) and a detailed
    per-label breakdown.

    Parameters
    ----------
    data          : dict from load_map_results()
    sigma_thresh  : MAD-based sigma threshold (default 3.0)

    Returns
    -------
    is_outlier    : (n_stars,) bool array
    outlier_info  : dict label_name → {
                        'z_scores':  (n_stars,) robust z-scores,
                        'flagged':   (n_stars,) bool per this label,
                        'n_flagged': int,
                    }
    """
    label_names = data["label_names"]
    residuals   = data["residuals"]      # (N, 23)
    n_stars     = data["n_stars"]
    n_labels    = len(label_names)

    is_outlier  = np.zeros(n_stars, dtype=bool)
    outlier_info = {}

    for j, name in enumerate(label_names):
        r   = residuals[:, j]
        med = np.median(r)
        mad = np.median(np.abs(r - med))
        mad = max(mad, 1e-8)             # avoid division by zero
        z   = np.abs(r - med) / (1.4826 * mad)   # 1.4826 scales MAD → σ equivalent
        flagged = z > sigma_thresh
        is_outlier |= flagged
        outlier_info[name] = {
            "z_scores":  z,
            "flagged":   flagged,
            "n_flagged": int(flagged.sum()),
        }

    print(f"\n  Outlier summary (>{sigma_thresh:.1f}σ on any label):")
    print(f"  Total stars flagged: {is_outlier.sum()} / {n_stars} "
          f"({100*is_outlier.sum()/n_stars:.1f}%)")
    print(f"  {'Label':<10} {'N flagged':>10}")
    print(f"  {'─'*22}")
    for name in CORE_LABELS + ABUND_LABELS:
        if name in outlier_info:
            print(f"  {PRETTY_NAMES.get(name,name):<10} {outlier_info[name]['n_flagged']:>10}")

    return is_outlier, outlier_info


def report_clean_vs_outlier(data, is_outlier):
    """
    Print side-by-side RMSE tables for clean stars vs outlier stars.
    This reveals whether the large RMSE is driven by outliers or is
    a genuine model weakness.
    """
    label_names = data["label_names"]
    residuals   = data["residuals"]

    clean_mask   = ~is_outlier
    n_clean      = clean_mask.sum()
    n_outlier    = is_outlier.sum()

    print(f"\n  ┌──────────────────────────────────────────────────────────┐")
    print(f"  │  Clean: {n_clean} stars    Outlier: {n_outlier} stars" +
          " " * (40 - len(f"  Clean: {n_clean} stars    Outlier: {n_outlier} stars")) + "│")
    print(f"  ├──────────────────────────────────────────────────────────┤")
    print(f"  │ {'Label':<10} {'RMSE clean':>12} {'RMSE outlier':>14} {'Ratio':>8}   │")
    print(f"  ├──────────────────────────────────────────────────────────┤")

    for j, name in enumerate(label_names):
        r_c = residuals[clean_mask, j]
        r_o = residuals[is_outlier, j]
        rmse_c = np.sqrt(np.mean(r_c**2)) if len(r_c) > 0 else 0
        rmse_o = np.sqrt(np.mean(r_o**2)) if len(r_o) > 0 else 0
        ratio  = rmse_o / rmse_c if rmse_c > 1e-10 else float('inf')
        pretty = PRETTY_NAMES.get(name, name)
        print(f"  │ {pretty:<10} {rmse_c:>12.4f} {rmse_o:>14.4f} {ratio:>7.1f}×  │")

    print(f"  └──────────────────────────────────────────────────────────┘")


def plot_outlier_diagnostics(data, is_outlier, outlier_info, save_dir=None):
    """
    Generate 4 diagnostic plots to distinguish model weakness from bad data.

    Plot 1 — Outlier flagged 1:1 scatter (clean=blue, outlier=red)
    Plot 2 — Teff residual vs abundance residual (cascade check)
    Plot 3 — ivar-weighted SNR proxy for outlier vs clean (bad data check)
    Plot 4 — Robust z-score heatmap (which labels fail for which stars)
    """
    label_names = data["label_names"]
    true        = data["true_labels"]
    inferred    = data["inferred_labels"]
    residuals   = data["residuals"]
    aspcap_err  = data["aspcap_errors"]
    n_labels    = len(label_names)
    clean_mask  = ~is_outlier

    def _save(fig, stem):
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            fig.savefig(os.path.join(save_dir, f"{stem}.png"), dpi=200, bbox_inches="tight")
            fig.savefig(os.path.join(save_dir, f"{stem}.pdf"), dpi=200, bbox_inches="tight")

    # ── Plot 1: Flagged 1:1 scatter ──────────────────────────────
    n_cols = 6
    n_rows = int(np.ceil(n_labels / n_cols))
    fig1, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*3.2, n_rows*3.2))
    fig1.patch.set_facecolor("white")
    axes = axes.ravel()

    for j in range(n_labels):
        ax = axes[j]
        t, inf = true[:, j], inferred[:, j]

        # clean
        ax.scatter(t[clean_mask], inf[clean_mask], s=8, alpha=0.5,
                   c="#2c7bb6", edgecolors="none", label="clean", zorder=2)
        # outlier
        ax.scatter(t[is_outlier], inf[is_outlier], s=18, alpha=0.85,
                   c="#d7191c", edgecolors="k", linewidths=0.3,
                   marker="x", label="outlier", zorder=3)

        lo = min(t.min(), inf.min()); hi = max(t.max(), inf.max())
        mg = 0.05*(hi-lo) if hi > lo else 0.5
        ax.plot([lo-mg, hi+mg], [lo-mg, hi+mg], "--", lw=0.8, c="#333", alpha=0.4)
        ax.set_xlim(lo-mg, hi+mg); ax.set_ylim(lo-mg, hi+mg)
        ax.set_title(PRETTY_NAMES.get(label_names[j], label_names[j]), fontsize=9, fontweight="bold")
        ax.tick_params(labelsize=6)
        ax.set_aspect("equal", adjustable="box")
        if j == 0: ax.legend(fontsize=6, loc="upper left")

    for j in range(n_labels, len(axes)): axes[j].set_visible(False)
    fig1.suptitle("1:1 Scatter — Clean (blue) vs Outlier (red ×)", fontsize=14, fontweight="bold", y=1.01)
    fig1.tight_layout()
    _save(fig1, "outlier_scatter")

    # ── Plot 2: Teff residual vs abundance residuals ─────────────
    # If abundance errors correlate with Teff errors, it's a cascade
    # problem (model issue).  If they DON'T, it's likely bad data.
    teff_idx = label_names.index("TEFF")
    teff_res = np.abs(residuals[:, teff_idx])

    abund_idx_list = [j for j, n in enumerate(label_names) if n in ABUND_LABELS]
    n = len(abund_idx_list)
    n_c2 = 5; n_r2 = int(np.ceil(n / n_c2))

    fig2, axes2 = plt.subplots(n_r2, n_c2, figsize=(n_c2*3.0, n_r2*2.8))
    fig2.patch.set_facecolor("white")
    axes2 = axes2.ravel()

    for k, j in enumerate(abund_idx_list):
        ax = axes2[k]
        abund_res = np.abs(residuals[:, j])
        ax.scatter(teff_res[clean_mask], abund_res[clean_mask], s=6, alpha=0.4,
                   c="#2c7bb6", edgecolors="none")
        ax.scatter(teff_res[is_outlier], abund_res[is_outlier], s=14, alpha=0.8,
                   c="#d7191c", marker="x", linewidths=0.6)

        # correlation coefficient
        rho = np.corrcoef(teff_res, abund_res)[0, 1]
        ax.set_title(f"{PRETTY_NAMES.get(label_names[j], label_names[j])}  ρ={rho:.2f}",
                     fontsize=8, fontweight="bold")
        ax.set_xlabel("|ΔTeff| [K]", fontsize=7)
        ax.set_ylabel("|Δabund|", fontsize=7)
        ax.tick_params(labelsize=6)

    for k in range(n, len(axes2)): axes2[k].set_visible(False)
    fig2.suptitle("Cascade Check — |Teff residual| vs |Abundance residual|\n"
                  "High ρ → model cascade (Stage 1 failure).  Low ρ → likely bad data.",
                  fontsize=12, fontweight="bold", y=1.03)
    fig2.tight_layout()
    _save(fig2, "outlier_cascade_check")

    # ── Plot 3: ASPCAP error of outlier vs clean ──────────────────
    # If outlier stars have much larger ASPCAP errors, they are bad data.
    fig3, ax3 = plt.subplots(figsize=(12, 5))
    fig3.patch.set_facecolor("white")

    x = np.arange(n_labels)
    w = 0.35
    med_clean   = np.array([np.median(aspcap_err[clean_mask, j]) for j in range(n_labels)])
    med_outlier = np.array([np.median(aspcap_err[is_outlier, j]) for j in range(n_labels)])

    ax3.bar(x - w/2, med_clean, w, label="Clean stars", color="#2c7bb6", alpha=0.8, edgecolor="white")
    ax3.bar(x + w/2, med_outlier, w, label="Outlier stars", color="#d7191c", alpha=0.7, edgecolor="white")
    ax3.set_xticks(x)
    ax3.set_xticklabels([PRETTY_NAMES.get(n, n) for n in label_names], fontsize=7, rotation=45, ha="right")
    ax3.set_ylabel("Median ASPCAP reported error", fontsize=10)
    ax3.set_title("Bad Data Check — ASPCAP Errors: Clean vs Outlier\n"
                  "If outlier bars are much taller → bad input data, not model failure",
                  fontsize=12, fontweight="bold")
    ax3.legend(fontsize=9)
    fig3.tight_layout()
    _save(fig3, "outlier_aspcap_error_compare")

    # ── Plot 4: Z-score heatmap (label × star) ───────────────────
    # Only show the outlier stars for readability
    outlier_indices = np.where(is_outlier)[0]
    if len(outlier_indices) > 0:
        z_matrix = np.column_stack([outlier_info[name]["z_scores"][outlier_indices]
                                    for name in label_names])  # (n_outlier, n_labels)

        fig4, ax4 = plt.subplots(figsize=(max(6, 0.2*len(outlier_indices)+2), 7))
        fig4.patch.set_facecolor("white")
        im = ax4.imshow(z_matrix.T, aspect="auto", cmap="YlOrRd",
                        interpolation="nearest", vmin=0, vmax=8)
        ax4.set_yticks(range(n_labels))
        ax4.set_yticklabels([PRETTY_NAMES.get(n, n) for n in label_names], fontsize=7)
        ax4.set_xlabel("Outlier star index", fontsize=10)
        ax4.set_xticks(range(len(outlier_indices)))
        ax4.set_xticklabels(outlier_indices, fontsize=5, rotation=90)
        ax4.set_title("Robust Z-Score Heatmap (outlier stars only)\n"
                      "Columns with ALL rows hot → bad star.  "
                      "Rows always hot → model-weak label.",
                      fontsize=11, fontweight="bold")
        cbar = fig4.colorbar(im, ax=ax4, shrink=0.8, pad=0.02)
        cbar.set_label("Robust z-score", fontsize=9)
        fig4.tight_layout()
        _save(fig4, "outlier_zscore_heatmap")
    else:
        fig4 = None

    figs = {"scatter": fig1, "cascade": fig2, "aspcap_compare": fig3, "z_heatmap": fig4}
    if save_dir:
        print(f"\n  📊  Outlier diagnostic figures saved to {save_dir}/")
    return figs


# ════════════════════════════════════════════════════════════════════
# USAGE EXAMPLE — How to run the full pipeline
# ════════════════════════════════════════════════════════════════════
#
# from map_results_analysis import load_map_results, report_map_results
# from map_fixes import (flag_outliers, report_clean_vs_outlier,
#                         plot_outlier_diagnostics)
#
# data = load_map_results("map_results/results.npz")
# report_map_results(data)
#
# # ---- Outlier analysis ----
# is_outlier, outlier_info = flag_outliers(data, sigma_thresh=3.0)
# report_clean_vs_outlier(data, is_outlier)
# figs = plot_outlier_diagnostics(data, is_outlier, outlier_info,
#                                  save_dir="map_results/outlier_figs")
#
# # ──────────────────────────────────────────────────────────────
# # HOW TO INTERPRET THE PLOTS
# # ──────────────────────────────────────────────────────────────
# #
# # 1. outlier_scatter.png
# #    Red ×'s clustered on specific stars → those stars are the problem.
# #    Red ×'s spread across many stars → systemic model issue.
# #
# # 2. outlier_cascade_check.png
# #    High ρ (>0.5):  abundance errors CORRELATE with Teff errors → the
# #    Stage 1 Teff failure cascades into Stage 2.  This is a MODEL issue.
# #    Fix: improve core convergence (use the fixes above).
# #
# # 3. outlier_aspcap_error_compare.png
# #    Outlier bars MUCH taller than clean bars → those stars have large
# #    ASPCAP pipeline errors = BAD INPUT DATA.  The model can't fix bad spectra.
# #    Outlier bars ~same height as clean bars → the spectra are fine but
# #    the model fails anyway = GENUINE MODEL WEAKNESS.
# #
# # 4. outlier_zscore_heatmap.png
# #    A COLUMN that is hot across all rows → that star is bad for every
# #    label = likely bad data (low SNR, bad reduction, etc.)
# #    A ROW that is hot across all stars → that label is systematically
# #    hard for the model = genuine model weakness for that element.
# #
# # DECISION MATRIX
# # ───────────────
# #  Cascade ρ high + ASPCAP err same → Model issue (core convergence)
# #  Cascade ρ high + ASPCAP err high → Both (bad data ruins core, cascades)
# #  Cascade ρ low  + ASPCAP err high → Bad data (flag and exclude)
# #  Cascade ρ low  + ASPCAP err same → Model weakness for that element
# #                                     (emulator needs retraining or
# #                                      element mask is insufficient)
