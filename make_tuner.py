import json
import os

source_nb = r"c:\Users\Aneesh Shastri\OneDrive\Documents\GitHub\PRISM\model-trainer.ipynb"
target_nb = r"c:\Users\Aneesh Shastri\OneDrive\Documents\GitHub\PRISM\optuna-tuner.ipynb"

with open(source_nb, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Extract first 10 code cells
setup_cells = []
code_idx = 0
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        setup_cells.append(cell)
        code_idx += 1
        if code_idx >= 10:
            break

# We will need to make sure the dataset is loaded.
# Cell 7 defines config overrides, cell 9 loads mask.
# We should probably append a cell that actually instantiates `train_ds`, `val_ds` if they aren't instantiated.
# Let's check cell 10 to see if `train_ds` is instantiated there.
# wait, my previous run_command showed:
# 7: # ========================================== # 1. CONFIGURATION & INDICES # ======================
# 8: @tf.keras.utils.register_keras_serializable() class ColumnSelector...
# 9: # ========================================== # 2. MASK LOADER # ====================================

# Let's just create the Optuna tuning cell.

optuna_code = """# ============================================================================
# OPTUNA FINE-TUNING STUDIES
# ============================================================================
import optuna
from optuna.integration import TFKerasPruningCallback
import tensorflow as tf
from tensorflow.keras import layers, Model

print("Installing Optuna if running in standard Kaggle env... (!pip install optuna if needed)")

# ----------------------------------------------------------------------------
# 1. EXPANDED PAYNE TUNING
# ----------------------------------------------------------------------------
def build_tunable_payne(n_labels, n_pixels, depth, width):
    inputs = layers.Input(shape=(n_labels,), name="labels")
    x = inputs
    for i in range(depth):
        x = layers.Dense(width, activation="relu", name=f"dense_{i+1}")(x)
    outputs = layers.Dense(n_pixels, activation=None, name="output_flux")(x)
    return Model(inputs=inputs, outputs=outputs, name="TunablePayne")

def objective_expanded_payne(trial):
    depth = trial.suggest_int('depth', 1, 5)
    width = trial.suggest_int('width', 200, 1500)
    
    model = build_tunable_payne(config.N_LABELS, config.OUTPUT_LENGTH, depth=depth, width=width)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(config.LEARNING_RATE, clipnorm=config.CLIP_NORM),
        loss=spl_loss
    )
    
    # We use a smaller epochs boundary but still full 35 to allow pruning
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        TFKerasPruningCallback(trial, 'val_loss')
    ]
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=35,
        verbose=0,
        callbacks=callbacks
    )
    
    # Return best val_loss
    return min(history.history['val_loss'])

# ----------------------------------------------------------------------------
# 2. REG PAYNE TUNING
# ----------------------------------------------------------------------------
def objective_reg_payne(trial):
    penalty_1 = trial.suggest_float('penalty_1', 1e-2, 1e2, log=True)
    penalty_2 = trial.suggest_float('penalty_2', 50.0, 2000.0, log=True)
    
    # Base structure is fixed payload for RegPayne, or we can use the best configuration from Expanded Payne.
    # Here we stick to original RegPayne base dimensionality (or Payne2019 base dimensionality).
    # Payne default: depth=2, width=300
    base_model = build_tunable_payne(config.N_LABELS, config.OUTPUT_LENGTH, depth=2, width=300)
    
    # Phase 1: Learn spectral lines loosely
    trainer_phase_1 = build_disentangled_model(base_model, mask_matrix=mask, penalty_weight=penalty_1)
    trainer_phase_1.compile(
        optimizer=tf.keras.optimizers.Adam(config.LEARNING_RATE, clipnorm=config.CLIP_NORM), 
        loss=spl_loss
    )
    
    trainer_phase_1.fit(
        train_ds,
        validation_data=val_ds,
        epochs=5,
        verbose=0
    )
    
    # Phase 2: Discipline via heavy penalty
    # Re-wrap the SAME base_model
    trainer_phase_2 = build_disentangled_model(base_model, mask_matrix=mask, penalty_weight=penalty_2)
    trainer_phase_2.compile(
        optimizer=tf.keras.optimizers.Adam(config.LEARNING_RATE, clipnorm=config.CLIP_NORM), 
        loss=spl_loss
    )
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        TFKerasPruningCallback(trial, 'val_loss')
    ]
    
    history = trainer_phase_2.fit(
        train_ds,
        validation_data=val_ds,
        epochs=30,
        verbose=0,
        callbacks=callbacks
    )
    
    return min(history.history['val_loss'])

# ============================================================================
# EXECUTION
# ============================================================================
if __name__ == "__main__":
    print("--- Starting Expanded Payne Tuning ---")
    study_ep = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner(n_warmup_steps=5))
    study_ep.optimize(objective_expanded_payne, n_trials=30)
    print("Best Expanded Payne Params:", study_ep.best_params)
    
    print("\\n--- Starting Reg Payne Tuning ---")
    study_rp = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner(n_warmup_steps=5))
    # We prune phase 2
    study_rp.optimize(objective_reg_payne, n_trials=30)
    print("Best Reg Payne Params:", study_rp.best_params)
    
    # Save the studies
    import joblib
    joblib.dump(study_ep, "expanded_payne_study.pkl")
    joblib.dump(study_rp, "reg_payne_study.pkl")
"""

optuna_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [line + '\n' for line in optuna_code.split('\n')]
}

visualization_code = """# ============================================================================
# OPTUNA VISUALIZATIONS
# ============================================================================
# Uncomment these to view plots in Kaggle
# import optuna.visualization as vis
# vis.plot_optimization_history(study_ep).show()
# vis.plot_param_importances(study_ep).show()
# vis.plot_contour(study_ep).show()

# vis.plot_optimization_history(study_rp).show()
# vis.plot_param_importances(study_rp).show()
# vis.plot_contour(study_rp).show()
"""

vis_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [line + '\n' for line in visualization_code.split('\n')]
}

new_nb = {
    "cells": setup_cells + [optuna_cell, vis_cell],
    "metadata": nb.get("metadata", {}),
    "nbformat": nb.get("nbformat", 4),
    "nbformat_minor": nb.get("nbformat_minor", 5)
}

# The dataset needs to be initialized. Just in case it's not present in first 10 cells, let's append dataset loading right before optuna cell.
ds_loader_code = """
# Ensure datasets are loaded before fine-tuning
train_ds, val_ds = build_datasets(config.H5_PATH)
mask = np.load("/kaggle/input/datasets/aneeshshastri/element-masks/apogee_mask.npy")
"""
ds_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [line + '\n' for line in ds_loader_code.split('\n')]
}
new_nb['cells'].insert(len(setup_cells), ds_cell)


with open(target_nb, 'w', encoding='utf-8') as f:
    json.dump(new_nb, f, indent=1)
    f.write("\n")
