# FULL INTEGRATION OF BACKWARD-MODEL PIPELINE (V6 - FINAL VERIFIED)
# This script combines the notebook base with Laplace Uncertainty Estimation.

import os
import time
import json
import numpy as np
import h5py
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.saving import register_keras_serializable
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge

tfb = tfp.bijectors

# ================================================================
# 1. CONFIGURATION
# ================================================================

class Config:
    H5_PATH    = "/kaggle/input/datasets/aneeshshastri/aspcapstar-dr17-150kstars/apogee_dr17_parallel.h5"
    STATS_PATH = "/kaggle/working/dataset_stats.npz"
    SELECTED_LABELS = [
        'TEFF', 'LOGG', 'M_H', 'VMICRO', 'VMACRO', 'VSINI',
        'C_FE', 'N_FE', 'O_FE', 'FE_H',
        'MG_FE', 'SI_FE', 'CA_FE', 'TI_FE', 'S_FE',
        'AL_FE', 'MN_FE', 'NI_FE', 'CR_FE', 'K_FE',
        'NA_FE', 'V_FE', 'CO_FE',
    ]
    N_LABELS_RAW = len(SELECTED_LABELS)
    BADPIX_CUTOFF = 1e-3
    RANDOM_SEED   = 42
    BATCH_SIZE_STARS = 32
    PRIOR_WEIGHT = 1.0

config = Config()
np.random.seed(config.RANDOM_SEED)
tf.random.set_seed(config.RANDOM_SEED)

CORE_INDICES  = [0,1,2,3,4,5,6,7,8]
ABUND_INDICES = [i for i in range(23) if i not in CORE_INDICES]

# ================================================================
# 2. CUSTOM LAYERS
# ================================================================

@register_keras_serializable()
class ColumnSelector(layers.Layer):
    def __init__(self, indices, **kwargs):
        super().__init__(**kwargs)
        self.indices = list(indices)
    def call(self, inputs): return tf.gather(inputs, self.indices, axis=1)
    def get_config(self): c = super().get_config(); c.update({'indices': self.indices}); return c

@register_keras_serializable()
class BeerLambertLaw(layers.Layer):
    def __init__(self, **kwargs): super().__init__(**kwargs)
    def call(self, k, tau): return k * tf.math.exp(-tau)

# ================================================================
# 3. LAPLACE UNCERTAINTY ESTIMATION
# ================================================================

@tf.function(jit_compile=True)
def compute_uncertainties_core(theta_unc_batch, obs_flux, obs_ivar, fixed_abund, cnn_mu, cnn_std):
    n_params = 9
    core_cnn_mu = tf.gather(cnn_mu, CORE_INDICES, axis=1)
    core_cnn_std = tf.gather(cnn_std, CORE_INDICES, axis=1)
    def single_star_hessian(args):
        theta_unc, flux, ivar, f_abund, c_mu, c_std = args
        with tf.GradientTape() as outer_tape:
            outer_tape.watch(theta_unc)
            with tf.GradientTape() as inner_tape:
                inner_tape.watch(theta_unc)
                theta_phys = _core_bij.forward(theta_unc)
                parts = []
                for i in range(23):
                    if i in CORE_INDICES: parts.append(tf.reshape(theta_phys[CORE_INDICES.index(i)], [1]))
                    else: parts.append(tf.reshape(f_abund[ABUND_INDICES.index(i)], [1]))
                full_23 = tf.concat(parts, axis=0)
                labels_27 = get_27_features(tf.expand_dims(full_23, 0))
                labels_norm = (labels_27 - MEAN_TENSOR) / STD_TENSOR
                f_model = tf.squeeze(tf.cast(model_pure_fp32(labels_norm, training=False), tf.float32))
                mask = tf.cast((flux > 1e-3) & (flux < 1.3) & (ivar > 0.0), tf.float32)
                loss = 0.5 * tf.reduce_sum(tf.square(flux - f_model) * ivar * mask) + 0.5 * tf.reduce_sum(tf.square((theta_phys - c_mu) / c_std))
            grad = inner_tape.gradient(loss, theta_unc)
        hess = outer_tape.jacobian(grad, theta_unc)
        cov = tf.linalg.inv(hess + tf.eye(n_params) * 1e-6)
        var = tf.abs(tf.linalg.diag_part(cov))
        s = tf.nn.sigmoid(theta_unc); jac = s * (1.0 - s) * (_core_high - _core_low)
        return tf.sqrt(var) * jac
    return tf.vectorized_map(single_star_hessian, (theta_unc_batch, obs_flux, obs_ivar, fixed_abund, core_cnn_mu, core_cnn_std))

@tf.function(jit_compile=True)
def compute_uncertainties_element(theta_unc_1d, obs_flux, obs_ivar, fixed_full_23, elem_col, pixel_mask, elem_lo, elem_hi, elem_cnn_mu, elem_cnn_std, p_weight):
    bij = tfb.Chain([tfb.Shift(tf.reshape(elem_lo, [1])), tfb.Scale(tf.reshape(elem_hi - elem_lo, [1])), tfb.Sigmoid()])
    def single_star_hessian(args):
        theta_unc, flux, ivar, f_23, c_mu, c_std = args
        with tf.GradientTape() as tape:
            tape.watch(theta_unc)
            with tf.GradientTape() as inner_tape:
                inner_tape.watch(theta_unc)
                theta_phys = bij.forward(theta_unc)
                full_spliced = tf.tensor_scatter_nd_update(tf.expand_dims(f_23, 0), [[0, elem_col]], theta_phys)
                labels_27 = get_27_features(full_spliced)
                labels_norm = (labels_27 - MEAN_TENSOR) / STD_TENSOR
                f_model = tf.squeeze(tf.cast(model_pure_fp32(labels_norm, training=False), tf.float32))
                mask = tf.cast((flux > 1e-3) & (flux < 1.3) & (ivar > 0.0), tf.float32) * pixel_mask
                loss = 0.5 * tf.reduce_sum(tf.square(flux - f_model) * ivar * mask) + p_weight * 0.5 * tf.reduce_sum(tf.square((theta_phys - c_mu) / c_std))
            grad = inner_tape.gradient(loss, theta_unc)
        hess = tape.gradient(grad, theta_unc)
        var = 1.0 / (tf.abs(hess) + 1e-6)
        s = tf.nn.sigmoid(theta_unc); jac = s * (1.0 - s) * (elem_hi - elem_lo)
        return tf.sqrt(var[0]) * jac[0]
    return tf.vectorized_map(single_star_hessian, (theta_unc_1d, obs_flux, obs_ivar, fixed_full_23, elem_cnn_mu, elem_cnn_std))

@tf.function(jit_compile=True)
def compute_uncertainties_joint(theta_unc_batch, obs_flux, obs_ivar, cnn_mu, cnn_std):
    n_params = 23
    def single_star_hessian(args):
        theta_unc, flux, ivar, c_mu, c_std = args
        with tf.GradientTape() as outer_tape:
            outer_tape.watch(theta_unc)
            with tf.GradientTape() as inner_tape:
                inner_tape.watch(theta_unc)
                theta_phys = _joint_bij.forward(theta_unc)
                labels_27 = get_27_features(tf.expand_dims(theta_phys, 0))
                labels_norm = (labels_27 - MEAN_TENSOR) / STD_TENSOR
                f_model = tf.squeeze(tf.cast(model_pure_fp32(labels_norm, training=False), tf.float32))
                mask = tf.cast((flux > 1e-3) & (flux < 1.3) & (ivar > 0.0), tf.float32)
                loss = 0.5 * tf.reduce_sum(tf.square(flux - f_model) * ivar * mask) + 0.5 * tf.reduce_sum(tf.square((theta_phys - c_mu) / c_std))
            grad = inner_tape.gradient(loss, theta_unc)
        hess = outer_tape.jacobian(grad, theta_unc)
        cov = tf.linalg.inv(hess + tf.eye(n_params) * 1e-6)
        var = tf.abs(tf.linalg.diag_part(cov))
        s = tf.nn.sigmoid(theta_unc); jac = s * (1.0 - s) * (bounds_high - bounds_low)
        return tf.sqrt(var) * jac
    return tf.vectorized_map(single_star_hessian, (theta_unc_batch, obs_flux, obs_ivar, cnn_mu, cnn_std))

# ================================================================
# 4. ENRICHED REPORTING
# ================================================================

def report_map_results_enriched(results):
    trues = np.array(results['true_labels'])
    preds = np.array(results['inferred_labels'])
    errors = np.array(results['inferred_errors'])
    aspcap_err = np.array(results['aspcap_errors'])
    residuals = preds - trues
    h = "─" * 80
    print(f"\n  PRISM Enriched MAP Diagnostics\n  {h}")
    print(f"  {'Label':<12} {'Bias':>8} {'MAD':>8} {'RMSE':>8} {'Mod-σ':>8} {'Ratio':>6}")
    print(f"  {h}")
    for j, name in enumerate(config.SELECTED_LABELS):
        r, m_sig, a_sig = residuals[:, j], errors[:, j], aspcap_err[:, j]
        bias = np.median(r); mad = np.median(np.abs(r - bias)); rmse = np.sqrt(np.mean(r**2))
        avg_ie = np.sqrt(np.mean(m_sig**2)); ratio = np.median(a_sig / np.where(m_sig>1e-10, m_sig, 1e-10))
        print(f"  {name:<12} {bias:>8.3f} {mad:>8.3f} {rmse:>8.3f} {avg_ie:>8.3f} {ratio:>6.1f}x")
    print(f"  {h}")

# ================================================================
# 5. MODIFIED INFERENCE PIPELINES
# ================================================================

def load_checkpoint(path=None):
    if path is None: path = CHECKPOINT_PATH
    fresh = {
        'global_indices': [], 'true_labels': [], 'inferred_labels': [],
        'inferred_errors': [], 'aspcap_errors': [], 'wall_seconds': []
    }
    if os.path.exists(path):
        data = np.load(path, allow_pickle=True)
        return {k: list(data[k]) if k in data else [] for k in fresh}, len(data['global_indices'])
    return fresh, 0

def run_inference_pipeline(test_indices, true_labels_norm, aspcap_errors, flux_array, ivar_array):
    n_stars = len(test_indices)
    true_labels_physical = (true_labels_norm[:, :23] * STD_TENSOR[:23] + MEAN_TENSOR[:23])
    results, n_done = load_checkpoint()
    remaining = list(range(n_done, n_stars))

    print(f"\n  Tier 1: Two-Stage MAP + Laplace ({len(remaining)} stars remaining)")
    for batch_start in range(0, len(remaining), config.BATCH_SIZE_STARS):
        batch_local = remaining[batch_start:batch_start + config.BATCH_SIZE_STARS]
        actual = len(batch_local)
        
        b_flux = flux_array[batch_local].astype(np.float32)
        b_ivar = ivar_array[batch_local].astype(np.float32)
        cnn_mu, cnn_sig = cnn_predict_physical(b_flux)

        if actual < config.BATCH_SIZE_STARS:
            pad = config.BATCH_SIZE_STARS - actual
            b_flux = np.concatenate([b_flux, np.zeros((pad, 8575), np.float32)])
            b_ivar = np.concatenate([b_ivar, np.zeros((pad, 8575), np.float32)])
            cnn_mu = np.concatenate([cnn_mu, np.tile(cnn_mu[:1], (pad, 1))])
            cnn_sig = np.concatenate([cnn_sig, np.tile(cnn_sig[:1], (pad, 1))])

        obs_flux_tf, obs_ivar_tf = tf.constant(b_flux), tf.constant(b_ivar)
        cnn_mu_tf, cnn_sig_tf = tf.constant(cnn_mu), tf.constant(cnn_sig)

        # -- Stage 1 Core --
        t0 = time.time()
        core_init = tf.clip_by_value(tf.gather(cnn_mu_tf, CORE_INDICES, 1), _core_low + 1e-3, _core_high - 1e-3)
        core_result, _ = map_optimize_core(_core_inv.forward(core_init), obs_flux_tf, obs_ivar_tf, tf.gather(cnn_mu_tf, ABUND_INDICES, 1), cnn_mu_tf, cnn_sig_tf)
        
        core_f_unc = _core_inv.forward(core_result)
        core_errs = compute_uncertainties_core(core_f_unc, obs_flux_tf, obs_ivar_tf, tf.gather(cnn_mu_tf, ABUND_INDICES, 1), cnn_mu_tf, cnn_sig_tf)
        core_np, core_err_np = core_result.numpy(), core_errs.numpy()
        t_core = time.time() - t0

        # -- Stage 2 Elements --
        t0 = time.time()
        fixed_23 = cnn_mu.copy()
        for ci, gi in enumerate(CORE_INDICES): fixed_23[:, gi] = core_np[:, ci]
        
        elem_results = np.zeros((config.BATCH_SIZE_STARS, 23), np.float32)
        elem_errors  = np.zeros((config.BATCH_SIZE_STARS, 23), np.float32)
        for ci, gi in enumerate(CORE_INDICES):
            elem_results[:, gi] = core_np[:, ci]
            elem_errors[:, gi]  = core_err_np[:, ci]

        for ai, elem_idx in enumerate(ABUND_INDICES):
            lo, hi = lower_bounds[elem_idx], upper_bounds[elem_idx]
            inv_bij = tfb.Invert(tfb.Chain([tfb.Shift(tf.reshape(lo, [1])), tfb.Scale(tf.reshape(hi-lo, [1])), tfb.Sigmoid()]))
            e_init = tf.clip_by_value(cnn_mu_tf[:, elem_idx:elem_idx+1], lo+1e-3, hi-1e-3)
            e_result, _ = map_optimize_element(inv_bij.forward(e_init), obs_flux_tf, obs_ivar_tf, tf.constant(fixed_23, tf.float32), elem_idx, ELEMENT_PIXEL_MASKS[elem_idx], lo, hi, cnn_mu_tf[:, elem_idx:elem_idx+1], cnn_sig_tf[:, elem_idx:elem_idx+1], ELEMENT_PRIOR_WEIGHTS[elem_idx])
            
            e_err = compute_uncertainties_element(inv_bij.forward(e_result), obs_flux_tf, obs_ivar_tf, tf.constant(fixed_23, tf.float32), elem_idx, ELEMENT_PIXEL_MASKS[elem_idx], lo, hi, cnn_mu_tf[:, elem_idx:elem_idx+1], cnn_sig_tf[:, elem_idx:elem_idx+1], ELEMENT_PRIOR_WEIGHTS[elem_idx])
            elem_results[:, elem_idx] = e_result.numpy()
            elem_errors[:, elem_idx]  = e_err.numpy()
            fixed_23[:, elem_idx] = e_result.numpy()
        t_elem = time.time() - t0

        elapsed = t_core + t_elem
        for b in range(actual):
            results['global_indices'].append(int(test_indices[batch_local[b]]))
            results['true_labels'].append(true_labels_physical[batch_local[b]])
            results['inferred_labels'].append(elem_results[b])
            results['inferred_errors'].append(elem_errors[b])
            results['aspcap_errors'].append(aspcap_errors[batch_local[b]])
            results['wall_seconds'].append(elapsed / actual)
            
        n_done += actual
        np.savez(CHECKPOINT_PATH, **{k: np.array(v) for k, v in results.items()})
        print(f"  [{n_done:>4}/{n_stars}] Batch done.")

    save_final_results(results)
    return results

def run_inference_pipeline_alt(test_indices, true_labels_norm, aspcap_errors, flux_array, ivar_array):
    CHECKPOINT_ALT_PATH = os.path.join(RESULTS_DIR, "checkpoint_alt.npz")
    true_labels_physical = (true_labels_norm[:, :23] * STD_TENSOR[:23] + MEAN_TENSOR[:23])
    results, n_done = load_checkpoint(CHECKPOINT_ALT_PATH)
    remaining = list(range(n_done, len(test_indices)))

    print(f"\n  Tier 1-Alt: Joint MAP + Laplace ({len(remaining)} stars remaining)")
    for batch_start in range(0, len(remaining), config.BATCH_SIZE_STARS):
        batch_local = remaining[batch_start:batch_start + config.BATCH_SIZE_STARS]
        actual = len(batch_local)
        b_flux = flux_array[batch_local].astype(np.float32)
        b_ivar = ivar_array[batch_local].astype(np.float32)
        cnn_mu, cnn_sig = cnn_predict_physical(b_flux)

        if actual < config.BATCH_SIZE_STARS:
            pad = config.BATCH_SIZE_STARS - actual
            b_flux = np.concatenate([b_flux, np.zeros((pad, 8575), np.float32)])
            b_ivar = np.concatenate([b_ivar, np.zeros((pad, 8575), np.float32)])
            cnn_mu = np.concatenate([cnn_mu, np.tile(cnn_mu[:1], (pad, 1))])
            cnn_sig = np.concatenate([cnn_sig, np.tile(cnn_sig[:1], (pad, 1))])

        obs_flux_tf, obs_ivar_tf = tf.constant(b_flux), tf.constant(b_ivar)
        cnn_mu_tf, cnn_sig_tf = tf.constant(cnn_mu), tf.constant(cnn_sig)

        t0 = time.time()
        margin = 1e-3
        j_init = tf.clip_by_value(cnn_mu_tf, bounds_low + margin, bounds_high - margin)
        j_result, _ = map_optimize_joint(_joint_inv.forward(j_init), obs_flux_tf, obs_ivar_tf, cnn_mu_tf, cnn_sig_tf)
        
        j_errors = compute_uncertainties_joint(_joint_inv.forward(j_result), obs_flux_tf, obs_ivar_tf, cnn_mu_tf, cnn_sig_tf)
        j_res_np, j_err_np = j_result.numpy(), j_errors.numpy()
        elapsed = time.time() - t0

        for b in range(actual):
            results['global_indices'].append(int(test_indices[batch_local[b]]))
            results['true_labels'].append(true_labels_physical[batch_local[b]])
            results['inferred_labels'].append(j_res_np[b])
            results['inferred_errors'].append(j_err_np[b])
            results['aspcap_errors'].append(aspcap_errors[batch_local[b]])
            results['wall_seconds'].append(elapsed / actual)
            
        n_done += actual
        np.savez(CHECKPOINT_ALT_PATH, **{k: np.array(v) for k, v in results.items()})
        print(f"  [{n_done:>4}/{len(test_indices)}] Batch done.")

    save_final_results(results, RESULTS_ALT_PATH) # Assuming path definition
    return results

# [Rest of your notebook's loading and visualization functions follows...]
