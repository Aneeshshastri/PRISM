import json
import os
import re

notebook_path = r"c:\Users\Aneesh Shastri\OneDrive\Documents\GitHub\PRISM\backward-model.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# The Laplace Logic
laplace_cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["# Laplace Uncertainty Estimation (Enriched)\n"]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import time\n",
            "\n",
            "@tf.function(jit_compile=True)\n",
            "def compute_uncertainties_core(theta_unc_batch, obs_flux, obs_ivar, fixed_abund, cnn_mu, cnn_std):\n",
            "    \"\"\"Laplace approx for Stage 1 (9D).\"\"\"\n",
            "    n_params = 9\n",
            "    core_cnn_mu = tf.gather(cnn_mu, CORE_INDICES, axis=1)\n",
            "    core_cnn_std = tf.gather(cnn_std, CORE_INDICES, axis=1)\n",
            "    def single_star_hessian(args):\n",
            "        theta_unc, flux, ivar, f_abund, c_mu, c_std = args\n",
            "        with tf.GradientTape() as outer_tape:\n",
            "            outer_tape.watch(theta_unc)\n",
            "            with tf.GradientTape() as inner_tape:\n",
            "                inner_tape.watch(theta_unc)\n",
            "                theta_phys = _core_bij.forward(theta_unc)\n",
            "                parts = []\n",
            "                for i in range(23):\n",
            "                    if i in CORE_INDICES: ci = CORE_INDICES.index(i); parts.append(tf.reshape(theta_phys[ci], [1]))\n",
            "                    else: ai = ABUND_INDICES.index(i); parts.append(tf.reshape(f_abund[ai], [1]))\n",
            "                full_23 = tf.concat(parts, axis=0)\n",
            "                labels_27 = get_27_features(tf.expand_dims(full_23, 0))\n",
            "                labels_norm = (labels_27 - MEAN_TENSOR) / STD_TENSOR\n",
            "                model_flux = tf.squeeze(tf.cast(model_pure_fp32(labels_norm, training=False), tf.float32))\n",
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
            "        s = tf.nn.sigmoid(theta_unc); jacobian_diag = s * (1.0 - s) * (_core_high - _core_low)\n",
            "        return tf.sqrt(var_unc) * jacobian_diag\n",
            "    return tf.vectorized_map(single_star_hessian, (theta_unc_batch, obs_flux, obs_ivar, fixed_abund, core_cnn_mu, core_cnn_std))\n",
            "\n",
            "@tf.function(jit_compile=True)\n",
            "def compute_uncertainties_element(theta_unc_1d, obs_flux, obs_ivar, fixed_full_23, \n",
            "                                 elem_col, pixel_mask, elem_lo, elem_hi, \n",
            "                                 elem_cnn_mu, elem_cnn_std, prior_weight):\n",
            "    \"\"\"Laplace approx for Stage 2 (1D).\"\"\"\n",
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
            "                model_flux = tf.squeeze(tf.cast(model_pure_fp32(labels_norm, training=False), tf.float32))\n",
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
            "        s = tf.nn.sigmoid(theta_unc); jacobian = s * (1.0 - s) * (hi - lo)\n",
            "        return tf.sqrt(var_unc[0]) * jacobian[0]\n",
            "    return tf.vectorized_map(single_star_hessian, (theta_unc_1d, obs_flux, obs_ivar, fixed_full_23, elem_cnn_mu, elem_cnn_std))\n",
            "\n",
            "@tf.function(jit_compile=True)\n",
            "def compute_uncertainties_joint(theta_unc_batch, obs_flux, obs_ivar, cnn_mu, cnn_std):\n",
            "    \"\"\"Laplace approx for Joint MAP (23D).\"\"\"\n",
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
            "                model_flux = tf.squeeze(tf.cast(model_pure_fp32(labels_norm, training=False), tf.float32))\n",
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
            "        s = tf.nn.sigmoid(theta_unc); jacobian_diag = s * (1.0 - s) * (bounds_high - bounds_low)\n",
            "        return tf.sqrt(var_unc) * jacobian_diag\n",
            "    return tf.vectorized_map(single_star_hessian, (theta_unc_batch, obs_flux, obs_ivar, cnn_mu, cnn_std))\n",
            "\n",
            "def report_map_results_enriched(data: dict, print_report: bool = True) -> dict:\n",
            "    \"\"\"Summary statistics including Laplace uncertainties.\"\"\"\n",
            "    label_names, residuals, aspcap_err = data[\"label_names\"], data[\"residuals\"], data[\"aspcap_errors\"]\n",
            "    inf_err = data.get(\"inferred_errors\", np.zeros_like(residuals))\n",
            "    n_stars, wall = data[\"n_stars\"], data[\"wall_seconds\"]\n",
            "    report = {}\n",
            "    for j, name in enumerate(label_names):\n",
            "        r, ie, ae = residuals[:, j], inf_err[:, j], aspcap_err[:, j]\n",
            "        report[name] = {\n",
            "            \"bias\": np.median(r), \"mad\": np.median(np.abs(r - np.median(r))), \"rmse\": np.sqrt(np.mean(r**2)),\n",
            "            \"sigma_rmse\": np.sqrt(np.mean(ie**2)), \"sigma_mad\": np.median(np.abs(ie - np.median(ie))),\n",
            "            \"ratio\": np.median(ae / np.where(ie > 1e-10, ie, 1e-10)), \"n_stars\": n_stars,\n",
            "        }\n",
            "    if print_report:\n",
            "        hdr  = f\"{'Label':<10} {'Bias':>8} {'MAD':>8} {'RMSE':>8} {'Mod-\\u03c3':>8} {'\\u03c3-MAD':>8} {'Ratio':>6}\"\n",
            "        rule = \"\u2500\" * len(hdr)\n",
            "        print(f\"\\n  Enriched Summary\\n  {rule}\\n  {hdr}\\n  {rule}\")\n",
            "        for n in label_names:\n",
            "            s = report[n]\n",
            "            print(f\"  {n:<10} {s['bias']:>8.3f} {s['mad']:>8.3f} {s['rmse']:>8.3f} {s['sigma_rmse']:>8.3f} {s['sigma_mad']:>8.3f} {s['ratio']:>6.1f}x\")\n",
            "        print(f\"  {rule}\")\n",
            "    return report\n"
        ]
    }
]

# Find a good spot to insert: after the bijectors/constants setup
insertion_idx = 0
for i, cell in enumerate(nb['cells']):
    src = "".join(cell.get('source', []))
    if "_core_bij" in src:
        insertion_idx = i + 1
        break

if insertion_idx > 0:
    print(f"Inserting Laplace logic after cell {insertion_idx}")
    nb['cells'] = nb['cells'][:insertion_idx] + laplace_cells + nb['cells'][insertion_idx:]
else:
    print("Warning: Could not find bijectors cell. Inserting at first code cell.")
    nb['cells'].insert(2, laplace_cells[0])
    nb['cells'].insert(3, laplace_cells[1])

# Patch Call Sites
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source_text = "".join(cell['source'])
        
        # Checkpoint loading update
        if "def load_checkpoint" in source_text and "'inferred_errors'" not in source_text:
            source_text = source_text.replace("'aspcap_errors': [],", "'inferred_errors': [], 'aspcap_errors': [],")
            source_text = source_text.replace("results = {k: list(data[k]) for k in fresh}", 
                                             "results = {k: list(data[k]) if k in data else [] for k in fresh}")
            cell['source'] = [l + ("" if l.endswith("\n") else "\n") for l in source_text.split("\n")]
        
        # Pipeline update
        if "def run_inference_pipeline" in source_text and "run_inference_pipeline_alt" not in source_text:
            source_text = source_text.replace("results['inferred_labels'].append(elem_results[b])", 
                                             "results['inferred_labels'].append(elem_results[b])\n            results['inferred_errors'].append(elem_errors[b])")
            source_text = source_text.replace("t_core = time.time() - t0", 
                                             "t_core = time.time() - t0\n        core_errors = compute_uncertainties_core(_core_inv.forward(core_result), obs_flux_tf, obs_ivar_tf, fixed_abund, cnn_mu_tf, cnn_sig_tf)\n        core_err_np = core_errors.numpy()")
            source_text = source_text.replace("elem_results = np.zeros((BATCH_SIZE_STARS, N_LABELS_RAW), np.float32)",
                                             "elem_results = np.zeros((BATCH_SIZE_STARS, N_LABELS_RAW), np.float32)\n        elem_errors  = np.zeros((BATCH_SIZE_STARS, N_LABELS_RAW), np.float32)")
            source_text = re.sub(r"(elem_results\[:, gi\] = core_np\[:, ci\])", r"\1\n            elem_errors[:, gi]  = core_err_np[:, ci]", source_text)
            
            # Element loop update
            old_elem_call = "elem_results[:, elem_idx] = e_result.numpy()"
            new_elem_call = "e_err = compute_uncertainties_element(inv_bij.forward(e_result), obs_flux_tf, obs_ivar_tf, tf.constant(fixed_23, tf.float32), elem_idx, ELEMENT_PIXEL_MASKS[elem_idx], lo_tf, hi_tf, cnn_mu_tf[:, elem_idx:elem_idx+1], cnn_sig_tf[:, elem_idx:elem_idx+1], ELEMENT_PRIOR_WEIGHTS[elem_idx])\n            elem_results[:, elem_idx] = e_result.numpy()\n            elem_errors[:, elem_idx]  = e_err.numpy()"
            source_text = source_text.replace(old_elem_call, new_elem_call)
            
            cell['source'] = [l + ("" if l.endswith("\n") else "\n") for l in source_text.split("\n")]

        # Joint pipeline update
        if "def run_inference_pipeline_alt" in source_text:
            source_text = source_text.replace("'aspcap_errors': [],", "'inferred_errors': [], 'aspcap_errors': [],")
            source_text = source_text.replace("results = {k: list(data[k]) for k in fresh}", 
                                             "results = {k: list(data[k]) if k in data else [] for k in fresh}")
            source_text = source_text.replace("joint_result_np = joint_result.numpy()",
                                             "joint_result_np = joint_result.numpy()\n        joint_errors_np = compute_uncertainties_joint(_joint_inv.forward(joint_result), obs_flux_tf, obs_ivar_tf, cnn_mu_tf, cnn_sig_tf).numpy()")
            source_text = source_text.replace("results['inferred_labels'].append(joint_result_np[b])",
                                             "results['inferred_labels'].append(joint_result_np[b])\n            results['inferred_errors'].append(joint_errors_np[b])")
            cell['source'] = [l + ("" if l.endswith("\n") else "\n") for l in source_text.split("\n")]

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
print("Notebook patched successfully (Safe Version).")
