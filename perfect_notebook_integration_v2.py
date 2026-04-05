import json
import os
import re

notebook_path = r"c:\Users\Aneesh Shastri\OneDrive\Documents\GitHub\PRISM\backward-model.ipynb"

def patch_notebook():
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # 1. Locate Laplace functions in Cell 3
    laplace_cell = nb['cells'][3]
    laplace_source = laplace_cell.get('source', [])
    
    # 2. Get Pipeline Definitions in Cell 1
    pipeline_cell = nb['cells'][1]
    pipeline_source_text = "".join(pipeline_cell.get('source', []))

    # 3. Patch run_inference_pipeline in the text
    if "compute_uncertainties_core" not in pipeline_source_text:
        # Stage 1 integration
        pipeline_source_text = pipeline_source_text.replace(
            "t_core = time.time() - t0",
            "t_core = time.time() - t0\n        # Laplace Uncertainties\n        core_final_unc = _core_inv.forward(core_result)\n        core_errors = compute_uncertainties_core(core_final_unc, obs_flux_tf, obs_ivar_tf, fixed_abund, cnn_mu_tf, cnn_sig_tf)\n        core_err_np = core_errors.numpy()"
        )
        
        pipeline_source_text = pipeline_source_text.replace(
            "elem_results = np.zeros((BATCH_SIZE_STARS, N_LABELS_RAW), np.float32)",
            "elem_results = np.zeros((BATCH_SIZE_STARS, N_LABELS_RAW), np.float32)\n        elem_errors  = np.zeros((BATCH_SIZE_STARS, N_LABELS_RAW), np.float32)"
        )
        
        pipeline_source_text = re.sub(
            r"(elem_results\[:, gi\] = core_np\[:, ci\])",
            r"\1\n            elem_errors[:, gi]  = core_err_np[:, ci]",
            pipeline_source_text
        )

        # Stage 2 integration (Element Loop)
        old_loop_body = """            elem_results[:, elem_idx] = e_result.numpy()
            fixed_23[:, elem_idx] = e_result.numpy()"""
        new_loop_body = """            # Laplace Uncertainties
            e_final_unc = inv_bij.forward(e_result)
            e_err = compute_uncertainties_element(e_final_unc, obs_flux_tf, obs_ivar_tf, tf.constant(fixed_23, tf.float32), elem_idx, ELEMENT_PIXEL_MASKS[elem_idx], lo_tf, hi_tf, cnn_mu_tf[:, elem_idx:elem_idx+1], cnn_sig_tf[:, elem_idx:elem_idx+1], ELEMENT_PRIOR_WEIGHTS[elem_idx])
            elem_results[:, elem_idx] = e_result.numpy()
            elem_errors[:, elem_idx]  = e_err.numpy()
            fixed_23[:, elem_idx] = e_result.numpy()"""
        pipeline_source_text = pipeline_source_text.replace(old_loop_body, new_loop_body)

        pipeline_source_text = pipeline_source_text.replace(
            "results['inferred_labels'].append(elem_results[b])",
            "results['inferred_labels'].append(elem_results[b])\n            results['inferred_errors'].append(elem_errors[b])"
        )

    # 4. Patch run_inference_pipeline_alt in the text
    if "compute_uncertainties_joint" not in pipeline_source_text:
        pipeline_source_text = pipeline_source_text.replace(
            "joint_result_np = joint_result.numpy()",
            "joint_result_np = joint_result.numpy()\n        # Laplace Uncertainties\n        batch_unc = _joint_inv.forward(joint_result)\n        joint_errors_np = compute_uncertainties_joint(batch_unc, obs_flux_tf, obs_ivar_tf, cnn_mu_tf, cnn_sig_tf).numpy()"
        )
        pipeline_source_text = pipeline_source_text.replace(
            "results['inferred_labels'].append(joint_result_np[b])",
            "results['inferred_labels'].append(joint_result_np[b])\n            results['inferred_errors'].append(joint_errors_np[b])"
        )
        # Clean up duplicate inferred_errors keys if any
        pipeline_source_text = re.sub(r'\'inferred_errors\': \[\], (\'inferred_errors\': \[\], )+', "'inferred_errors': [], ", pipeline_source_text)

    # 5. Insert Laplace functions at the top of Cell 1
    # Convert Laplace source list to correctly format list of lines
    laplace_lines = [l + ("" if l.endswith("\n") else "\n") for l in laplace_source]
    new_cell_1_lines = laplace_lines + ["\n"] + [l + "\n" for l in pipeline_source_text.split("\n")]
    nb['cells'][1]['source'] = new_cell_1_lines

    # 6. Delete old Cell 3
    del nb['cells'][3]

    # 7. Update Reporting calls in ALL code cells
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            src_list = cell.get('source', [])
            new_src_list = []
            for line in src_list:
                # If not a function definition and contains report_map_results
                if "report_map_results(" in line and "def " not in line:
                    line = line.replace("report_map_results(", "report_map_results_enriched(")
                new_src_list.append(line)
            cell['source'] = new_src_list

    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("Notebook perfectly integrated and cleaned.")

if __name__ == "__main__":
    patch_notebook()
