import json
import io
import os
import re

nb_path = r"c:\Users\Aneesh Shastri\OneDrive\Documents\GitHub\PRISM\backward-model.ipynb"
out_path = r"c:\Users\Aneesh Shastri\OneDrive\Documents\GitHub\PRISM\backward_model_integrated.py"

def assemble_perfect_script():
    with io.open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    code_cells = [c.get('source', []) for c in nb['cells'] if c['cell_type'] == 'code']
    full_code = []
    for cell in code_cells:
        full_code.append("".join(cell))
    
    final_code = "\n\n# --- CELL SEPARATOR ---\n\n".join(full_code)

    # Clean up the duplicate 'inferred_errors' in the dicts
    final_code = re.sub(r"('inferred_errors': \[\],? ?)+", "'inferred_errors': [], ", final_code)
    
    # Now, inject the Laplace logic if missing
    laplace_funcs = """
# ============================================================================
#  LAPLACE UNCERTAINTY ESTIMATION (Stage 1, Stage 2, Joint)
# ============================================================================

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

def report_map_results_enriched(results):
    trues = np.array(results['true_labels'])
    preds = np.array(results['inferred_labels'])
    errors = np.array(results['inferred_errors'])
    aspcap_err = np.array(results['aspcap_errors'])
    residuals = preds - trues
    print(f"\\n  PRISM Enriched MAP Diagnostics\\n  {'─'*80}\\n  {'Label':<12} {'Bias':>8} {'MAD':>8} {'RMSE':>8} {'Mod-σ':>8} {'Ratio':>6}\\n  {'─'*80}")
    for j, name in enumerate(config.SELECTED_LABELS):
        r, m_sig, a_sig = residuals[:, j], errors[:, j], aspcap_err[:, j]
        bias = np.median(r); mad = np.median(np.abs(r - bias)); rmse = np.sqrt(np.mean(r**2))
        avg_ie = np.sqrt(np.mean(m_sig**2)); ratio = np.median(a_sig / np.where(m_sig>1e-10, m_sig, 1e-10))
        print(f"  {name:<12} {bias:>8.3f} {mad:>8.3f} {rmse:>8.3f} {avg_ie:>8.3f} {ratio:>6.1f}x")
    print("  " + "─"*80)
"""

    # Inject functions after imports
    final_code = re.sub(r"(import .*?\n\n)", r"\1" + laplace_funcs + "\n", final_code, count=1)

    # Patch the pipelines
    # 1. Core integration
    final_code = final_code.replace("core_result = map_optimize_core(", 
        "core_result, _ = map_optimize_core(") # If notebook version returns (result, loss)
    
    # Actually, let's assume the notebook's run_inference_pipeline needs the Laplace calls injected.
    # Searching for Stage 1 finished
    final_code = final_code.replace("t_core = time.time() - t0", 
        "t_core = time.time() - t0\\n        core_f_unc = _core_inv.forward(core_result)\\n        core_errs = compute_uncertainties_core(core_f_unc, obs_flux_tf, obs_ivar_tf, fixed_abund, cnn_mu_tf, cnn_sig_tf)\\n        core_err_np = core_errs.numpy()")
    
    # 2. Element integration
    # Find the element results init
    final_code = final_code.replace("elem_results = np.zeros((BATCH_SIZE_STARS, N_LABELS_RAW), np.float32)",
        "elem_results = np.zeros((BATCH_SIZE_STARS, N_LABELS_RAW), np.float32)\\n        elem_errors  = np.zeros((BATCH_SIZE_STARS, N_LABELS_RAW), np.float32)")
    
    # Find the core result transfer
    final_code = final_code.replace("elem_results[:, gi] = core_np[:, ci]",
        "elem_results[:, gi] = core_np[:, ci]\\n            elem_errors[:, gi]  = core_err_np[:, ci]")
    
    # Find the individual element result
    final_code = final_code.replace("elem_results[:, elem_idx] = e_result.numpy()",
        "e_err = compute_uncertainties_element(inv_bij.forward(e_result), obs_flux_tf, obs_ivar_tf, tf.constant(fixed_23, tf.float32), elem_idx, ELEMENT_PIXEL_MASKS[elem_idx], lo_tf, hi_tf, cnn_mu_tf[:, elem_idx:elem_idx+1], cnn_sig_tf[:, elem_idx:elem_idx+1], ELEMENT_PRIOR_WEIGHTS[elem_idx])\\n            elem_results[:, elem_idx] = e_result.numpy()\\n            elem_errors[:, elem_idx]  = e_err.numpy()")

    # 3. Join integration (alt pipeline)
    final_code = final_code.replace("results['inferred_labels'].append(joint_result_np[b])",
        "results['inferred_labels'].append(joint_result_np[b])\\n            results['inferred_errors'].append(joint_errors_np[b])")
    
    # Append inferred_errors in main
    final_code = final_code.replace("results['inferred_labels'].append(elem_results[b])",
        "results['inferred_labels'].append(elem_results[b])\\n            results['inferred_errors'].append(elem_errors[b])")

    with io.open(out_path, 'w', encoding='utf-8') as f:
        f.write("# FULL REWRITE OF BACKWARD-MODEL PIPELINE (V4 - NOTEBOOK BASE)\n")
        f.write("# Copy this script into a clean cell in your notebook.\n\n")
        f.write(final_code)

assemble_perfect_script()
print("V4 Integrated script generated from notebook base.")
