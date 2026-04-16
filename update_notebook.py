import json
import os

NOTEBOOK_PATH = "c:/Users/ANEESHSHASTRI/Documents/GitHub/PRISM/backward-model-nuts.ipynb"

with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb.get('cells', []):
    if cell.get('cell_type') == 'code':
        source = cell.get('source', [])
        is_string = isinstance(source, str)
        text = source if is_string else "".join(source)
        
        # Check if this cell has the hyperparameters and run_element_mcmc
        if "NUM_CHAINS_CORE  = 8;" in text or "lo_val = bounds_low[elem_idx];" in text:
            # 1. Hyperparameters
            text = text.replace(
                "NUM_CHAINS_CORE  = 8;   NUM_RESULTS_CORE = 800;  NUM_BURNIN_CORE = 600;  NUM_ADAPT_CORE = 500",
                "NUM_CHAINS_CORE  = 4;   NUM_RESULTS_CORE = 200;  NUM_BURNIN_CORE = 200;  NUM_ADAPT_CORE = 150"
            )
            text = text.replace(
                "NUM_CHAINS_ELEM  = 4;   NUM_RESULTS_ELEM = 500;  NUM_BURNIN_ELEM = 400;  NUM_ADAPT_ELEM = 300",
                "NUM_CHAINS_ELEM  = 2;   NUM_RESULTS_ELEM = 100;  NUM_BURNIN_ELEM = 100;  NUM_ADAPT_ELEM = 80"
            )
            text = text.replace(
                "MAX_TREE_DEPTH   = 8;   TARGET_ACCEPT    = 0.8",
                "MAX_TREE_DEPTH   = 6;   TARGET_ACCEPT    = 0.85"
            )
            
            # 2. Step sizes
            text = text.replace(
                "step_size = tf.fill([1, BATCH_SIZE_STARS, N_CORE], 0.01)",
                "step_size = tf.fill([1, BATCH_SIZE_STARS, N_CORE], 0.05)"
            )
            text = text.replace(
                "step_size = tf.fill([1, BATCH_SIZE_STARS, 1], 0.01)",
                "step_size = tf.fill([1, BATCH_SIZE_STARS, 1], 0.1)"
            )
            
            # 3. Bug fix in run_element_mcmc
            bug_old = '''    lo_val = bounds_low[elem_idx]; hi_val = bounds_high[elem_idx]
    bij = tfb.Chain([tfb.Shift(tf.constant([lo_val])), tfb.Scale(tf.constant([hi_val - lo_val])), tfb.Sigmoid()])'''
            bug_new = '''    lo_val = float(lower_bounds[elem_idx])
    hi_val = float(upper_bounds[elem_idx])
    bij = tfb.Chain([tfb.Shift(tf.constant([lo_val], tf.float32)), tfb.Scale(tf.constant([hi_val - lo_val], tf.float32)), tfb.Sigmoid()])'''
            text = text.replace(bug_old, bug_new)
            
            # 4. Joint MCMC alternative hyperparams
            text = text.replace(
                "NUM_CHAINS_JOINT = 8; NUM_RESULTS_JOINT = 800; NUM_BURNIN_JOINT = 600; NUM_ADAPT_JOINT = 500",
                "NUM_CHAINS_JOINT = 4; NUM_RESULTS_JOINT = 200; NUM_BURNIN_JOINT = 200; NUM_ADAPT_JOINT = 150"
            )
            text = text.replace(
                "step_size = tf.fill([1, BATCH_SIZE_STARS, N_LABELS_RAW], 0.005)",
                "step_size = tf.fill([1, BATCH_SIZE_STARS, N_LABELS_RAW], 0.05)"
            )
            
            if is_string:
                cell['source'] = text
            else:
                # Need to chunk into lines since original was likely lines
                import io
                cell['source'] = [line + '\n' for line in text.split('\n')]
                # remove trailing newline from last element
                if cell['source']:
                    cell['source'][-1] = cell['source'][-1].rstrip('\n')

with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
print("Updated notebook.")
