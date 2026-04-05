import json
import os
import re

notebook_path = r"c:\Users\Aneesh Shastri\OneDrive\Documents\GitHub\PRISM\backward-model.ipynb"

def patch_notebook():
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    for cell in nb['cells']:
        if cell['cell_type'] != 'code': continue
        source_text = "".join(cell.get('source', []))

        # 1. Clean up duplicate keys/args in load_alt_checkpoint
        if "def load_alt_checkpoint" in source_text:
             source_text = re.sub(r'\'inferred_errors\': \[\], (\'inferred_errors\': \[\], )+', "'inferred_errors': [], ", source_text)
             # Update dictionary assignment to handle missing values
             source_text = source_text.replace("results = {k: list(data[k]) for k in fresh}", 
                                              "results = {k: list(data[k]) if k in data else [] for k in fresh}")
             cell['source'] = [l + ("" if l.endswith("\n") else "\n") for l in source_text.split("\n") if l.strip()]

        # 2. Update run_inference_pipeline_alt (Joint)
        if "def run_inference_pipeline_alt" in source_text:
            # Join lines correctly to handle multiple spaces/indentation
            source_text = "".join(cell['source'])
            
            # Use regex to find and replace the joint result logic
            pattern = r"(joint_result_np = joint_result\.numpy\(\))\n"
            if "joint_errors_np =" not in source_text:
                source_text = re.sub(pattern, r"\1\n        joint_errors_np = compute_uncertainties_joint(_joint_inv.forward(joint_result), obs_flux_tf, obs_ivar_tf, cnn_mu_tf, cnn_sig_tf).numpy()\n", source_text)
            
            # Clean up duplicates if they already exist
            source_text = re.sub(r"(joint_errors_np = [^\n]+\n)\s+\1", r"\1", source_text)
            
            # Append result result correctly
            append_src = "results['inferred_labels'].append(joint_result_np[b])"
            if f"{append_src}\n            results['inferred_errors'].append(joint_errors_np[b])" not in source_text:
                source_text = source_text.replace(append_src, f"{append_src}\n            results['inferred_errors'].append(joint_errors_np[b])")
            
            # Clean up duplicate append result
            source_text = re.sub(r"(results\['inferred_errors'\]\.append\(joint_errors_np\[b\]\)\n)\s+\1", r"\1", source_text)
            
            cell['source'] = [l + ("" if l.endswith("\n") else "\n") for l in source_text.split("\n") if l.strip()]

        # 3. Update run_inference_pipeline (Stage 1/2)
        if "def run_inference_pipeline(" in source_text and "alt" not in source_text:
            source_text = "".join(cell['source'])
            # Stage 1: Integrate compute_uncertainties_core
            s1_marker = "t_core = time.time() - t0"
            if "core_errors = compute_uncertainties_core" not in source_text:
                source_text = source_text.replace(s1_marker, f"{s1_marker}\n        core_errors = compute_uncertainties_core(_core_inv.forward(core_result), obs_flux_tf, obs_ivar_tf, fixed_abund, cnn_mu_tf, cnn_sig_tf)\n        core_err_np = core_errors.numpy()")
            
            # Update elem_results init
            if "elem_errors  = np.zeros" not in source_text:
                source_text = source_text.replace("elem_results = np.zeros((BATCH_SIZE_STARS, N_LABELS_RAW), np.float32)", 
                                                 "elem_results = np.zeros((BATCH_SIZE_STARS, N_LABELS_RAW), np.float32)\n        elem_errors  = np.zeros((BATCH_SIZE_STARS, N_LABELS_RAW), np.float32)")
            
            # Update core copying
            if "elem_errors[:, gi]  = core_err_np[:, ci]" not in source_text:
                source_text = re.sub(r"elem_results\[:, gi\] = core_np\[:, ci\]", 
                                     r"elem_results[:, gi] = core_np[:, ci]\n            elem_errors[:, gi]  = core_err_np[:, ci]", source_text)
            
            # Stage 2: Integrate compute_uncertainties_element in the loop
            if "e_err = compute_uncertainties_element" not in source_text:
                loop_pat = r"(elem_results\[:, elem_idx\] = e_result\.numpy\(\))\n(\s+fixed_23\[:, elem_idx\] = e_result\.numpy\(\))"
                source_text = re.sub(loop_pat, 
                                     r"e_err = compute_uncertainties_element(inv_bij.forward(e_result), obs_flux_tf, obs_ivar_tf, tf.constant(fixed_23, tf.float32), elem_idx, ELEMENT_PIXEL_MASKS[elem_idx], lo_tf, hi_tf, cnn_mu_tf[:, elem_idx:elem_idx+1], cnn_sig_tf[:, elem_idx:elem_idx+1], ELEMENT_PRIOR_WEIGHTS[elem_idx])\n            \1\n            elem_errors[:, elem_idx]  = e_err.numpy()\n\2", 
                                     source_text)

            # Store result
            if "results['inferred_errors'].append(elem_errors[b])" not in source_text:
                source_text = source_text.replace("results['inferred_labels'].append(elem_results[b])", 
                                                 "results['inferred_labels'].append(elem_results[b])\n            results['inferred_errors'].append(elem_errors[b])")
            
            cell['source'] = [l + ("" if l.endswith("\n") else "\n") for l in source_text.split("\n") if l.strip()]

        # 4. Update reporting calls anywhere in the notebook
        if "report_map_results(" in source_text and "def " not in source_text and "_enriched" not in source_text:
            source_text = source_text.replace("report_map_results(", "report_map_results_enriched(")
            cell['source'] = [l + ("" if l.endswith("\n") else "\n") for l in source_text.split("\n") if l.strip()]

    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("Notebook integration complete.")

if __name__ == "__main__":
    patch_notebook()
