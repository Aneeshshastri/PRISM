import json
import glob

def patch_notebook(filepath, is_cnn=False):
    import os
    with open(filepath, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # Imputer path string to inject
    if is_cnn:
        # In CNNwarmstart.ipynb they are building models, maybe they use local kaggle paths or the same standard paths?
        # we'll just hardcode it to the path from stargen-comparison for consistency
        imputer_path = '"/kaggle/input/models/aneeshshastri/stargen-comparision/tensorflow2/default/14/imputer.pkl"'
    else:
        imputer_path = '"/kaggle/input/models/aneeshshastri/stargen-comparision/tensorflow2/default/14/imputer.pkl"'

    for cell in nb['cells']:
        if cell['cell_type'] != 'code':
            continue
        source = "".join(cell['source'])

        changed = False
        if not is_cnn:
            if "imputer = IterativeImputer(estimator=BayesianRidge(), max_iter=10, initial_strategy='median')" in source and "clean   = imputer.fit_transform(vals)" in source:
                old_code = "imputer = IterativeImputer(estimator=BayesianRidge(), max_iter=10, initial_strategy='median')\n    clean   = imputer.fit_transform(vals)"
                new_code = f"import joblib\n    imputer = joblib.load({imputer_path})\n    clean   = imputer.transform(vals)"
                source = source.replace(old_code, new_code)
                changed = True
        else:
            # CNNwarmstart
            if "imputer = IterativeImputer(estimator=BayesianRidge(), max_iter=10," in source and "train_labels = imputer.fit_transform(train_imp)" in source:
                # Find exact substring
                old_code_sub = "imputer = IterativeImputer(estimator=BayesianRidge(), max_iter=10,\n                               initial_strategy='median')\n    train_labels = imputer.fit_transform(train_imp)"
                new_code = f"import joblib\n    imputer = joblib.load({imputer_path})\n    train_labels = imputer.transform(train_imp)"
                if old_code_sub in source:
                    source = source.replace(old_code_sub, new_code)
                    changed = True

        if changed:
            # Reconstruct cell source lines
            if "\n" in source and not isinstance(cell['source'], str):
                lines = source.split('\n')
                cell['source'] = [line + '\n' for line in lines[:-1]] + [lines[-1]] if lines[-1] else [line + '\n' for line in lines[:-1]]
            else:
                cell['source'] = [source]

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
        f.write("\n")
    print(f"Patched {filepath}")

patch_notebook(r"c:\Users\Aneesh Shastri\OneDrive\Documents\GitHub\PRISM\Model-tester.ipynb")
patch_notebook(r"c:\Users\Aneesh Shastri\OneDrive\Documents\GitHub\PRISM\backward-model.ipynb")
patch_notebook(r"c:\Users\Aneesh Shastri\OneDrive\Documents\GitHub\PRISM\CNNwarmstart.ipynb", is_cnn=True)
