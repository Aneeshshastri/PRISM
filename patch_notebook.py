import json
import os

notebook_path = r"c:\Users\Aneesh Shastri\OneDrive\Documents\GitHub\PRISM\backward-model.ipynb"

if not os.path.exists(notebook_path):
    print(f"File not found: {notebook_path}")
    exit(1)

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Define the new content for the cells we want to modify

# 1. Hessian Functions and updated Constants
hessian_code = [
    "LABEL_NAMES  = config.SELECTED_LABELS\n",
    "BATCH_SIZE_STARS = 32\n",
    "CORE_INDICES  = [0,1,2,3,4,5,6,7,8]\n",
    "ABUND_INDICES = [9,10,11,12,13,14,15,16,17,18,19,20,21,22]\n",
    "N_LABELS_RAW  = 23\n",
    "N_ABUND       = len(ABUND_INDICES)\n",
    "PRIOR_WEIGHT  = 1.0\n",
    "\n",
    "@tf.function(jit_compile=True)\n",
    "def compute_uncertainties_core(theta_unc_batch, obs_flux, obs_ivar, fixed_abund, cnn_mu, cnn_std):\n",
    "    \"\"\"Laplace approximation for 9D core physics.\"\"\"\n",
    "    n_params = 9\n",
    "    core_cnn_mu = tf.gather(cnn_mu, CORE_INDICES, axis=1)\n",
    "    core_cnn_std = tf.gather(cnn_std, CORE_INDICES, axis=1)\n",
    "\n",
    "    def single_star_hessian(args):\n",
    "        theta_unc, flux, ivar, f_abund, c_mu, c_std = args\n",
    "        with tf.GradientTape() as outer_tape:\n",
    "            outer_tape.watch(theta_unc)\n",
    "            with tf.GradientTape() as inner_tape:\n",
    "                inner_tape.watch(theta_unc)\n",
    "                theta_phys = _core_bij.forward(theta_unc)\n",
    "                parts = []\n",
    "                for i in range(23):\n",
    "                    if i in CORE_INDICES:\n",
    "                        ci = CORE_INDICES.index(i)\n",
    "                        parts.append(tf.reshape(theta_phys[ci], [1]))\n",
    "                    else:\n",
    "                        ai = ABUND_INDICES.index(i)\n",
    "                        parts.append(tf.reshape(f_abund[ai], [1]))\n",
    "                full_23 = tf.concat(parts, axis=0)\n",
    "                labels_27 = get_27_features(tf.expand_dims(full_23, 0))\n",
    "                labels_norm = (labels_27 - MEAN_TENSOR) / STD_TENSOR\n",
    "                model_flux = tf.cast(model_pure_fp32(labels_norm, training=False), tf.float32)\n",
    "                model_flux = tf.squeeze(model_flux)\n",
    "                safe_flux = tf.where(tf.math.is_finite(flux), flux, 0.0)\n",
    "                safe_ivar = tf.where(tf.math.is_finite(ivar) & tf.math.is_finite(flux), ivar, 0.0)\n",
    "                safe_ivar = 1.0 / (1.0 / (safe_ivar + 1e-12) + 1e-5)\n",
    "                mask = tf.cast((safe_flux > 1e-3) & (safe_flux < 1.3) & (safe_ivar > 0.0), tf.float32)\n",
    "                chi2 = tf.reduce_sum(tf.square(safe_flux - model_flux) * safe_ivar * mask)\n",
    "                prior = tf.reduce_sum(tf.square((theta_phys - c_mu) / c_std))\n",
    "                loss = 0.5 * chi2 + PRIOR_WEIGHT * 0.5 * prior\n",
    "            grad = inner_tape.gradient(loss, theta_unc)\n",
    "        hess = outer_tape.jacobian(grad, theta_unc)\n",
    "        hess_reg = hess + tf.eye(n_params) * 1e-6\n",
    "        cov_unc = tf.linalg.inv(hess_reg)\n",
    "        var_unc = tf.abs(tf.linalg.diag_part(cov_unc))\n",
    "        s = tf.nn.sigmoid(theta_unc)\n",
    "        jacobian_diag = s * (1.0 - s) * (_core_high - _core_low)\n",
    "        return tf.sqrt(var_unc) * jacobian_diag\n",
    "\n",
    "    return tf.vectorized_map(single_star_hessian, (theta_unc_batch, obs_flux, obs_ivar, fixed_abund, core_cnn_mu, core_cnn_std))\n",
    "\n",
    "@tf.function(jit_compile=True)\n",
    "def compute_uncertainties_element(theta_unc_1d, obs_flux, obs_ivar, fixed_full_23, \n",
    "                                 elem_col, pixel_mask, elem_lo, elem_hi, \n",
    "                                 elem_cnn_mu, elem_cnn_std, prior_weight):\n",
    "    \"\"\"Laplace approximation for 1D single element.\"\"\"\n",
    "    lo = tf.reshape(elem_lo, [1]); hi = tf.reshape(elem_hi, [1])\n",
    "    bij = tfb.Chain([tfb.Shift(lo), tfb.Scale(hi - lo), tfb.Sigmoid()])\n",
    "    def single_star_hessian(args):\n",
    "        theta_unc, flux, ivar, f_23, c_mu, c_std = args\n",
    "        with tf.GradientTape() as tape:\n",
    "            tape.watch(theta_unc)\n",
    "            with tf.GradientTape() as inner_tape:\n",
    "                inner_tape.watch(theta_unc)\n",
    "                theta_phys = bij.forward(theta_unc)\n",
    "                indices = tf.constant([[0, elem_col]])\n",
    "                full_spliced = tf.tensor_scatter_nd_update(tf.expand_dims(f_23, 0), indices, theta_phys)\n",
    "                labels_27 = get_27_features(full_spliced)\n",
    "                labels_norm = (labels_27 - MEAN_TENSOR) / STD_TENSOR\n",
    "                model_flux = tf.cast(model_pure_fp32(labels_norm, training=False), tf.float32)\n",
    "                model_flux = tf.squeeze(model_flux)\n",
    "                safe_flux = tf.where(tf.math.is_finite(flux), flux, 0.0)\n",
    "                safe_ivar = tf.where(tf.math.is_finite(ivar) & tf.math.is_finite(flux), ivar, 0.0)\n",
    "                safe_ivar = 1.0 / (1.0 / (safe_ivar + 1e-12) + 1e-5)\n",
    "                mask = tf.cast((safe_flux > 1e-3) & (safe_flux < 1.3) & (safe_ivar > 0.0), tf.float32) * pixel_mask\n",
    "                chi2 = tf.reduce_sum(tf.square(safe_flux - model_flux) * safe_ivar * mask)\n",
    "                prior = tf.reduce_sum(tf.square((theta_phys - c_mu) / c_std))\n",
    "                loss = 0.5 * chi2 + prior_weight * 0.5 * prior\n",
    "            grad = inner_tape.gradient(loss, theta_unc)\n",
    "        hess = tape.gradient(grad, theta_unc)\n",
    "        var_unc = 1.0 / tf.abs(hess + 1e-6)\n",
    "        s = tf.nn.sigmoid(theta_unc)\n",
    "        jacobian = s * (1.0 - s) * (hi - lo)\n",
    "        return tf.sqrt(var_unc[0]) * jacobian[0]\n",
    "\n",
    "    return tf.vectorized_map(single_star_hessian, (theta_unc_1d, obs_flux, obs_ivar, fixed_full_23, elem_cnn_mu, elem_cnn_std))\n",
    "\n",
    "@tf.function(jit_compile=True)\n",
    "def compute_uncertainties_joint(theta_unc_batch, obs_flux, obs_ivar, cnn_mu, cnn_std):\n",
    "    \"\"\"Laplace approximation for 23D joint MAP.\"\"\"\n",
    "    n_params = 23\n",
    "    def single_star_hessian(args):\n",
    "        theta_unc, flux, ivar, c_mu, c_std = args\n",
    "        with tf.GradientTape() as outer_tape:\n",
    "            outer_tape.watch(theta_unc)\n",
    "            with tf.GradientTape() as inner_tape:\n",
    "                inner_tape.watch(theta_unc)\n",
    "                theta_phys = _joint_bij.forward(theta_unc)\n",
    "                labels_27 = get_27_features(tf.expand_dims(theta_phys, 0))\n",
    "                labels_norm = (labels_27 - MEAN_TENSOR) / STD_TENSOR\n",
    "                model_flux = tf.cast(model_pure_fp32(labels_norm, training=False), tf.float32)\n",
    "                model_flux = tf.squeeze(model_flux)\n",
    "                safe_flux = tf.where(tf.math.is_finite(flux), flux, 0.0)\n",
    "                safe_ivar = tf.where(tf.math.is_finite(ivar) & tf.math.is_finite(flux), ivar, 0.0)\n",
    "                safe_ivar = 1.0 / (1.0 / (safe_ivar + 1e-12) + 1e-5)\n",
    "                mask = tf.cast((safe_flux > 1e-3) & (safe_flux < 1.3) & (safe_ivar > 0.0), tf.float32)\n",
    "                chi2 = tf.reduce_sum(tf.square(safe_flux - model_flux) * safe_ivar * mask)\n",
    "                prior = tf.reduce_sum(tf.square((theta_phys - c_mu) / c_std))\n",
    "                loss = 0.5 * chi2 + PRIOR_WEIGHT * 0.5 * prior\n",
    "            grad = inner_tape.gradient(loss, theta_unc)\n",
    "        hess = outer_tape.jacobian(grad, theta_unc)\n",
    "        hess_reg = hess + tf.eye(n_params) * 1e-6\n",
    "        cov_unc = tf.linalg.inv(hess_reg)\n",
    "        var_unc = tf.abs(tf.linalg.diag_part(cov_unc))\n",
    "        s = tf.nn.sigmoid(theta_unc)\n",
    "        jacobian_diag = s * (1.0 - s) * (bounds_high - bounds_low)\n",
    "        return tf.sqrt(var_unc) * jacobian_diag\n",
    "\n",
    "    return tf.vectorized_map(single_star_hessian, (theta_unc_batch, obs_flux, obs_ivar, cnn_mu, cnn_std))\n"
]

# 2. Update report_map_results
report_code = [
    "def report_map_results(data: dict, print_report: bool = True) -> dict:\n",
    "    \"\"\"Compute per-label summary statistics and print analysis table with Laplace errors.\"\"\"\n",
    "    label_names = data[\"label_names\"]\n",
    "    residuals   = data[\"residuals\"]\n",
    "    aspcap_err  = data[\"aspcap_errors\"]\n",
    "    inf_err     = data[\"inferred_errors\" if \"inferred_errors\" in data else \"residuals\"]\n",
    "    n_stars     = data[\"n_stars\"]\n",
    "    wall        = data[\"wall_seconds\"]\n",
    "\n",
    "    report = {}\n",
    "    for j, name in enumerate(label_names):\n",
    "        r = residuals[:, j]\n",
    "        ie = inf_err[:, j] if \"inferred_errors\" in data else np.zeros_like(r)\n",
    "        ae = aspcap_err[:, j]\n",
    "        safe_ie = np.where(ie > 1e-10, ie, 1e-10)\n",
    "        ratios = ae / safe_ie\n",
    "        \n",
    "        report[name] = {\n",
    "            \"bias\":              np.median(r),\n",
    "            \"mad\":               np.median(np.abs(r - np.median(r))),\n",
    "            \"rmse\":              np.sqrt(np.mean(r**2)),\n",
    "            \"sigma_rmse\":        np.sqrt(np.mean(ie**2)),\n",
    "            \"sigma_mad\":         np.median(np.abs(ie - np.median(ie))),\n",
    "            \"ratio\":             np.median(ratios),\n",
    "            \"median_aspcap_err\": np.median(ae),\n",
    "            \"median_model_err\":  np.median(ie),\n",
    "            \"n_stars\":           n_stars,\n",
    "        }\n",
    "\n",
    "    if print_report:\n",
    "        hdr  = f\"{'Label':<10} {'Bias':>8} {'MAD':>8} {'RMSE':>8} {'Mod-σ':>8} {'σ-MAD':>8} {'Ratio':>6}\"\n",
    "        rule = \"─\" * len(hdr)\n",
    "        lines = [\"\", \"  Result Summary\", f\"  {rule}\", f\"  {hdr}\", f\"  {rule}\"]\n",
    "        for name in label_names:\n",
    "            s = report[name]\n",
    "            lines.append(f\"  {name:<10} {s['bias']:>8.3f} {s['mad']:>8.3f} {s['rmse']:>8.3f} {s['sigma_rmse']:>8.3f} {s['sigma_mad']:>8.3f} {s['ratio']:>6.1f}x\")\n",
    "        lines.append(f\"  {rule}\")\n",
    "        print(\"\\n\".join(lines))\n",
    "    return report\n"
]

# Process cells
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        
        # Insert Hessian functions before Stage 1/2 cells
        if "def map_optimize_core" in source and "compute_uncertainties_core" not in source:
            cell['source'] = hessian_code + ["\n"] + cell['source']
            
        # Update inference loop logic to use Laplace
        if "def run_inference_pipeline" in source and "run_inference_pipeline_alt" not in source:
            source = source.replace("t_core = time.time() - t0", 
                                    "t_core = time.time() - t0\n        # Laplace\n        core_f_unc = _core_inv.forward(core_result)\n        core_errors = compute_uncertainties_core(core_f_unc, obs_flux_tf, obs_ivar_tf, fixed_abund, cnn_mu_tf, cnn_sig_tf)\n        core_err_np = core_errors.numpy()")
            source = source.replace("elem_results = np.zeros((BATCH_SIZE_STARS, N_LABELS_RAW), np.float32)",
                                    "elem_results = np.zeros((BATCH_SIZE_STARS, N_LABELS_RAW), np.float32)\n        elem_errors  = np.zeros((BATCH_SIZE_STARS, N_LABELS_RAW), np.float32)")
            source = source.replace("elem_results[:, gi] = core_np[:, ci]",
                                    "elem_results[:, gi] = core_np[:, ci]\n            elem_errors[:, gi]  = core_err_np[:, ci]")
            source = source.replace("elem_results[:, elem_idx] = e_result.numpy()",
                                    "e_f_unc = inv_bij.forward(e_result)\n            e_err = compute_uncertainties_element(e_f_unc, obs_flux_tf, obs_ivar_tf, tf.constant(fixed_23, tf.float32), elem_idx, ELEMENT_PIXEL_MASKS[elem_idx], lo_tf, hi_tf, cnn_mu_tf[:, elem_idx:elem_idx+1], cnn_sig_tf[:, elem_idx:elem_idx+1], ELEMENT_PRIOR_WEIGHTS[elem_idx])\n            elem_results[:, elem_idx] = e_result.numpy()\n            elem_errors[:, elem_idx]  = e_err.numpy()")
            source = source.replace("results['inferred_labels'].append(elem_results[b])",
                                    "results['inferred_labels'].append(elem_results[b])\n            results['inferred_errors'].append(elem_errors[b])")
            cell['source'] = [line + ("" if line.endswith("\n") else "\n") for line in source.split("\n")]

        if "def run_inference_pipeline_alt" in source:
             source = source.replace("joint_result, joint_final_loss = map_optimize_joint(",
                                     "joint_result, joint_final_loss = map_optimize_joint(")
             source = source.replace("joint_result_np = joint_result.numpy()",
                                     "joint_result_np = joint_result.numpy()\n        # Laplace\n        joint_f_unc = _joint_inv.forward(joint_result)\n        joint_errors = compute_uncertainties_joint(joint_f_unc, obs_flux_tf, obs_ivar_tf, cnn_mu_tf, cnn_sig_tf)\n        joint_errors_np = joint_errors.numpy()")
             source = source.replace("results['inferred_labels'].append(joint_result_np[b])",
                                     "results['inferred_labels'].append(joint_result_np[b])\n            results['inferred_errors'].append(joint_errors_np[b])")
             cell['source'] = [line + ("" if line.endswith("\n") else "\n") for line in source.split("\n")]

        # Update reporting
        if "def report_map_results" in source:
            cell['source'] = report_code

# Add inferred_errors to load_checkpoint dictionary
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if "def load_checkpoint" in source and "'inferred_errors'" not in source:
             source = source.replace("'true_labels': [],", "'true_labels': [], 'inferred_errors': [],")
             source = source.replace("results = {k: list(data[k]) for k in fresh}", "results = {k: list(data[k]) if k in data else [] for k in fresh}")
             cell['source'] = [line + ("" if line.endswith("\n") else "\n") for line in source.split("\n")]

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Notebook patched successfully.")
