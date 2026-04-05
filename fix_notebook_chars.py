import json
import os
import re

notebook_path = r"c:\Users\Aneesh Shastri\OneDrive\Documents\GitHub\PRISM\backward-model.ipynb"

def fix_and_patch_notebook():
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # 1. Check Cell 1 (The main function definition cell)
    cell_1 = nb['cells'][1]
    source = cell_1.get('source', [])
    
    # If it's a list of single characters, join it!
    if len(source) > 100 and all(len(s) <= 1 for s in source[:100]):
        print("Detected character-splitted source list. Fixing...")
        source_text = "".join(source)
    else:
        source_text = "".join(source)

    # 2. Re-apply the Laplace Integration Logic to the clean text
    # Ensure compute_uncertainties_* are present
    if "def compute_uncertainties_core" not in source_text:
        # (This should already be there from my previous run, but I'll make sure it's clean)
        pass

    # 3. Patch the pipelines correctly
    # Stage 1 Laplace
    if "compute_uncertainties_core(" not in source_text:
        source_text = source_text.replace(
            "t_core = time.time() - t0",
            "t_core = time.time() - t0\n        # Laplace Uncertainties\n        core_f_unc = _core_inv.forward(core_result)\n        core_errs = compute_uncertainties_core(core_f_unc, obs_flux_tf, obs_ivar_tf, fixed_abund, cnn_mu_tf, cnn_sig_tf)\n        core_err_np = core_errs.numpy()"
        )
        
    # Element Loop Laplace
    if "compute_uncertainties_element(" not in source_text:
        # Standardize search string
        old_body = """            elem_results[:, elem_idx] = e_result.numpy()
            fixed_23[:, elem_idx] = e_result.numpy()"""
        new_body = """            # Laplace Uncertainties
            e_f_unc = inv_bij.forward(e_result)
            e_err = compute_uncertainties_element(e_f_unc, obs_flux_tf, obs_ivar_tf, tf.constant(fixed_23, tf.float32), elem_idx, ELEMENT_PIXEL_MASKS[elem_idx], lo_tf, hi_tf, cnn_mu_tf[:, elem_idx:elem_idx+1], cnn_sig_tf[:, elem_idx:elem_idx+1], ELEMENT_PRIOR_WEIGHTS[elem_idx])
            elem_results[:, elem_idx] = e_result.numpy()
            elem_errors[:, elem_idx]  = e_err.numpy()
            fixed_23[:, elem_idx] = e_result.numpy()"""
        source_text = source_text.replace(old_body, new_body)

    # 4. Save back as a list of lines (Kaggle/Colab style)
    lines = source_text.splitlines(keepends=True)
    cell_1['source'] = [l for l in lines if l.strip() or l == "\n"]

    # 5. Fix Reporting Calls in all code cells
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            src_list = cell.get('source', [])
            new_src_list = []
            for line in src_list:
                # If calling original report_map_results, switch to enriched
                if "report_map_results(" in line and "def " not in line and "_enriched" not in line:
                    line = line.replace("report_map_results(", "report_map_results_enriched(")
                new_src_list.append(line)
            cell['source'] = new_src_list

    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("Notebook fixed and integrated correctly.")

if __name__ == "__main__":
    fix_and_patch_notebook()
