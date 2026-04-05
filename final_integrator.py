import json
import io
import re

notebook_path = r"c:\Users\Aneesh Shastri\OneDrive\Documents\GitHub\PRISM\backward-model.ipynb"
output_path   = r"c:\Users\Aneesh Shastri\OneDrive\Documents\GitHub\PRISM\backward_model_integrated.py"

def integrate_everything():
    with io.open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # 1. Gather all code
    full_code = []
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            full_code.append("".join(cell.get('source', [])))
    
    combined_code = "\n\n".join(full_code)

    # 2. Define Laplace functions if not present
    laplace_defs = """
# --- LAPLACE UNCERTAINTY LOGIC ---
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
                safe_flux = tf.where(tf.math.is_finite(flux), flux, 0.0)
                safe_ivar = tf.where(tf.math.is_finite(ivar) & tf.math.is_finite(flux), ivar, 0.0)
                mask = tf.cast((safe_flux > 1e-3) & (safe_flux < 1.3) & (safe_ivar > 0.0), tf.float32)
                chi2 = tf.reduce_sum(tf.square(safe_flux - f_model) * safe_ivar * mask)
                prior = tf.reduce_sum(tf.square((theta_phys - c_mu) / c_std))
                loss = 0.5 * chi2 + PRIOR_WEIGHT * 0.5 * prior
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
                safe_flux = tf.where(tf.math.is_finite(flux), flux, 0.0)
                safe_ivar = tf.where(tf.math.is_finite(ivar) & tf.math.is_finite(flux), ivar, 0.0)
                mask = tf.cast((safe_flux > 1e-3) & (safe_flux < 1.3) & (safe_ivar > 0.0), tf.float32) * pixel_mask
                chi2 = tf.reduce_sum(tf.square(safe_flux - f_model) * safe_ivar * mask)
                prior = tf.reduce_sum(tf.square((theta_phys - c_mu) / c_std))
                loss = 0.5 * chi2 + p_weight * 0.5 * prior
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
                safe_flux = tf.where(tf.math.is_finite(flux), flux, 0.0)
                safe_ivar = tf.where(tf.math.is_finite(ivar) & tf.math.is_finite(flux), ivar, 0.0)
                mask = tf.cast((safe_flux > 1e-3) & (safe_flux < 1.3) & (safe_ivar > 0.0), tf.float32)
                chi2 = tf.reduce_sum(tf.square(safe_flux - f_model) * safe_ivar * mask)
                prior = tf.reduce_sum(tf.square((theta_phys - c_mu) / c_std))
                loss = 0.5 * chi2 + PRIOR_WEIGHT * 0.5 * prior
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
    print("\\n  PRISM Enriched MAP Diagnostics\\n  " + "─"*80)
    print(f"  {'Label':<12} {'Bias':>8} {'MAD':>8} {'RMSE':>8} {'Mod-σ':>8} {'Ratio':>6}")
    print("  " + "─"*80)
    for j, name in enumerate(config.SELECTED_LABELS):
        r, m_sig, a_sig = residuals[:, j], errors[:, j], aspcap_err[:, j]
        bias = np.median(r); mad = np.median(np.abs(r - bias)); rmse = np.sqrt(np.mean(r**2))
        avg_ie = np.sqrt(np.mean(m_sig**2)); ratio = np.median(a_sig / np.where(m_sig>1e-10, m_sig, 1e-10))
        print(f"  {name:<12} {bias:>8.3f} {mad:>8.3f} {rmse:>8.3f} {avg_ie:>8.3f} {ratio:>6.1f}x")
    print("  " + "─"*80)
"""

    # 3. Integrate into Pipelines
    # Stage 1 Laplace
    if "compute_uncertainties_core" not in combined_code:
        combined_code = combined_code.replace("t_core = time.time() - t0", 
            "t_core = time.time() - t0\\n        core_f_unc = _core_inv.forward(core_result)\\n        core_errors = compute_uncertainties_core(core_f_unc, obs_flux_tf, obs_ivar_tf, fixed_abund, cnn_mu_tf, cnn_sig_tf)\\n        core_err_np = core_errors.numpy()")
        combined_code = combined_code.replace("elem_results = np.zeros((BATCH_SIZE_STARS, N_LABELS_RAW), np.float32)",
            "elem_results = np.zeros((BATCH_SIZE_STARS, N_LABELS_RAW), np.float32)\\n        elem_errors  = np.zeros((BATCH_SIZE_STARS, N_LABELS_RAW), np.float32)")
        combined_code = re.sub(r"(elem_results\[:, gi\] = core_np\[:, ci\])", r"\\1\\n            elem_errors[:, gi]  = core_err_np[:, ci]", combined_code)
        
        # Stage 2 Laplace
        old_body = """            elem_results[:, elem_idx] = e_result.numpy()
            fixed_23[:, elem_idx] = e_result.numpy()"""
        new_body = """            e_f_unc = inv_bij.forward(e_result)
            e_err = compute_uncertainties_element(e_f_unc, obs_flux_tf, obs_ivar_tf, tf.constant(fixed_23, tf.float32), elem_idx, ELEMENT_PIXEL_MASKS[elem_idx], lo_tf, hi_tf, cnn_mu_tf[:, elem_idx:elem_idx+1], cnn_sig_tf[:, elem_idx:elem_idx+1], ELEMENT_PRIOR_WEIGHTS[elem_idx])
            elem_results[:, elem_idx] = e_result.numpy()
            elem_errors[:, elem_idx]  = e_err.numpy()
            fixed_23[:, elem_idx] = e_result.numpy()"""
        combined_code = combined_code.replace(old_body, new_body)

        combined_code = combined_code.replace("results['inferred_labels'].append(elem_results[b])",
            "results['inferred_labels'].append(elem_results[b])\\n            results['inferred_errors'].append(elem_errors[b])")

    # Joint Laplace
    if "compute_uncertainties_joint" not in combined_code:
        combined_code = combined_code.replace("joint_result_np = joint_result.numpy()",
            "joint_result_np = joint_result.numpy()\\n        joint_errors_np = compute_uncertainties_joint(_joint_inv.forward(joint_result), obs_flux_tf, obs_ivar_tf, cnn_mu_tf, cnn_sig_tf).numpy()")
        combined_code = combined_code.replace("results['inferred_labels'].append(joint_result_np[b])",
            "results['inferred_labels'].append(joint_result_np[b])\\n            results['inferred_errors'].append(joint_errors_np[b])")

    # 4. Final Patch for Definitions
    if "def compute_uncertainties_core" not in combined_code:
        combined_code = laplace_defs + "\n" + combined_code

    with io.open(output_path, 'w', encoding='utf-8') as f:
        f.write("# FULL REWRITE OF BACKWARD-MODEL PIPELINE\\n")
        f.write("# Copy this script into a clean cell in your notebook.\\n\\n")
        f.write(combined_code)

integrate_everything()
