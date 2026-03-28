"""
backward-model-corrected.py
============================
Complete, self-contained script equivalent to backward-model.ipynb with
all GPU-optimisation and correctness fixes applied.

Run on Kaggle with GPU enabled. Sections are labelled by their original
notebook-cell numbers for easy cross-referencing.

Changes vs original notebook:
  - BATCH_SIZE_STARS  5  → 10
  - NUM_BURNIN       400 → 1000,  NUM_ADAPT 320 → 800
  - Per-star step-size adaptation  (shape 1,B,23  not  23,)
  - tf.repeat broadcast  (explicit star↔chain alignment)
  - .numpy() calls removed from graph-traced code
  - Single XLA compilation via tf.Variable
  - Last-batch padding + reduced checkpoint frequency
"""

# ╔══════════════════════════════════════════════════════════════╗
# ║ CELL 1 — Imports & Config                                   ║
# ╚══════════════════════════════════════════════════════════════╝

import os
import glob
import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras.saving import register_keras_serializable
from tensorflow.keras import layers, models, Input, callbacks, regularizers
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic
import seaborn as sns
from scipy.stats import spearmanr
from tensorflow.keras import mixed_precision
from matplotlib.lines import Line2D
mixed_precision.set_global_policy('mixed_float16')

#-----------------------------------------------------------------
# Configs
#-----------------------------------------------------------------

class Config:
    # --- Paths ---
    H5_PATH = "/kaggle/input/datasets/aneeshshastri/aspcapstar-dr17-150kstars/apogee_dr17_parallel.h5" 
    TFREC_DIR = "/kaggle/working/tfrecords"
    STATS_PATH = "/kaggle/working/dataset_stats.npz"
    start=140_000
    end=149_500
    # --- System ---
    TESTING_MODE = True
    TEST_LIMIT = None
    NUM_SHARDS = 16 
    TRAIN_INTEGRATED=True
    TRAIN_BASE_MODEL=True
    RANDOM_SEED=42
    # --- Model Hyperparameters ---
    BATCH_SIZE = 64     
    LEARNING_RATE = 1e-3  
    EPOCHS = 50
    LATENT_DIM = 268
    OUTPUT_LENGTH = 8575
    # --loss related---  
    IVAR_SCALE = 1000.0   
    CLIP_NORM = 1.0     
    BADPIX_CUTOFF=1e-3  
    #----predictor-labels--------
    #CAUTION: LITERALLY EVERYTHING IS IN THE SAME ORDER AS THESE LABELS. DO NOT TOUCH THE ORDER OF THESE LABELS
    SELECTED_LABELS = [
        # 1. Core
        'TEFF', 'LOGG', 'M_H', 'VMICRO', 'VMACRO', 'VSINI',
        # 2. CNO
        'C_FE', 'N_FE', 'O_FE',
        #3. metals
        'FE_H',
        'MG_FE', 'SI_FE', 'CA_FE', 'TI_FE', 'S_FE',
        'AL_FE', 'MN_FE', 'NI_FE', 'CR_FE','K_FE',
        'NA_FE','V_FE','CO_FE'
    ]
    ABUNDANCE_INDICES =[i for i, label in enumerate(SELECTED_LABELS) if '_FE' in label]
    FE_H_INDEX = SELECTED_LABELS.index('FE_H')
    N_LABELS = len(SELECTED_LABELS) + 4
    ERRORS=[f'{label}_ERR' for label in SELECTED_LABELS ]
    #GRAPHING:
    WAVELENGTH_START = 1514
    WAVELENGTH_END = 1694 


config = Config()
np.random.seed(config.RANDOM_SEED)
tf.random.set_seed(config.RANDOM_SEED)
os.makedirs(config.TFREC_DIR, exist_ok=True)


# ╔══════════════════════════════════════════════════════════════╗
# ║ CELL 2 — Data Loading / Imputation Helpers                  ║
# ╚══════════════════════════════════════════════════════════════╝

from sklearn.decomposition import PCA


def get_clean_imputed_data(h5_path, selected_labels, limit=None):
    
    print("Read data for imputation")
    
    with h5py.File(h5_path, 'r') as f:
        get_col = lambda k: f['metadata'][k]
        keys = f['metadata'].dtype.names
        print(np.mean((f['metadata']['SNR'])[config.start:]))
        raw_values = np.stack([get_col(p) for p in selected_labels], axis=1)
        bad_mask = np.zeros_like(raw_values, dtype=bool)
        
        for i, label in enumerate(selected_labels):
            flag_name = f"{label}_FLAG"
            if flag_name in keys:
                flg = get_col(flag_name)
                #Handle Void/Structured Types safely
                if flg.dtype.names: flg = flg[flg.dtype.names[0]]
                if flg.dtype.kind == 'V': flg = flg.view('<i4')
                is_bad = (flg.astype(int) != 0)
            elif label in ['TEFF', 'LOGG', 'VMICRO', 'VMACRO', 'VSINI']:
                is_bad = (raw_values[:, i] < -5000)
            else:
                is_bad = np.zeros_like(raw_values[:, i], dtype=bool)
            bad_mask[:, i] = is_bad

    if limit:
        print(f"Truncating to first {limit} stars.")
        raw_values = raw_values[:limit]
        bad_mask = bad_mask[:limit]

    print(f"Imputing Labels for {len(raw_values)} stars...")
    vals_to_impute = raw_values.copy()
    vals_to_impute[bad_mask] = np.nan
    
    imputer = IterativeImputer(estimator=BayesianRidge(), max_iter=10, initial_strategy='median')
    clean_labels = imputer.fit_transform(vals_to_impute)
    print("Converting [X/Fe] to [X/H]...")
    
    # Get the Iron Abundance column
    fe_h_col = clean_labels[:, config.FE_H_INDEX : config.FE_H_INDEX+1] 
    
    # Loop through all metal columns and add iron
    for idx in config.ABUNDANCE_INDICES:
        clean_labels[:, idx] = clean_labels[:, idx] + fe_h_col[:, 0]
        
    try:
        #calculate Inverse temperature to help with spectral lines
        teff_idx = selected_labels.index('TEFF')
        teff_vals = clean_labels[:, teff_idx]
        
        inv_teff = 5040.0 / (teff_vals + 1e-6)
        inv_teff = inv_teff.reshape(-1, 1)
        
        # Stack it onto the end of the array
        clean_labels = np.hstack([clean_labels, inv_teff])
        
    except ValueError:
        print("TEFF not found in labels")
         
    vmacro_idx = selected_labels.index('VMACRO')
    vmacro_vals = clean_labels[:, vmacro_idx]
       
    vsini_idx = selected_labels.index('VSINI')
    vsini_vals = clean_labels[:, vsini_idx]
    quadrature=np.sqrt(np.square(vmacro_vals)+np.square(vsini_vals))
    quadrature = quadrature.reshape(-1, 1)
    clean_labels = np.hstack([clean_labels, quadrature])
    
    try:
        C_idx = selected_labels.index('C_FE')
        C_vals = clean_labels[:, C_idx]
       
        O_idx = selected_labels.index('O_FE')
        O_vals = clean_labels[:, O_idx]
        C_O_diff=C_vals-O_vals
        C_O_diff=C_O_diff.reshape(-1,1)
        clean_labels = np.hstack([clean_labels, C_O_diff])
        
    except ValueError:
        print("C_FE or O_FE not found in labels")
    
    try:
        G_idx = selected_labels.index('LOGG')
        G_vals = clean_labels[:, G_idx]
       
        M_idx = selected_labels.index('M_H')
        M_vals = clean_labels[:, M_idx]
        LOGPE=0.5*G_vals+0.5*M_vals
        LOGPE=LOGPE.reshape(-1,1)
        clean_labels = np.hstack([clean_labels, LOGPE])
        
    except ValueError:
        print("M_H or LOGG not found in labels")

    print(f"Transformation Complete. Final Input Shape: {clean_labels.shape}")
    
    return clean_labels

def get_err(h5_path, selected_labels, limit=None):
    
    print("Read data for imputation")
    
    with h5py.File(h5_path, 'r') as f:
        get_col = lambda k: f['metadata'][k]
        keys = f['metadata'].dtype.names
        print(np.mean(f['metadata']['SNR']))
        raw_values = np.stack([get_col(p) if p not in ['VMACRO_ERR','VMICRO_ERR','VSINI_ERR'] else np.zeros_like(get_col('TEFF_ERR'))  for p in selected_labels 
                              ], axis=1)
        
        bad_mask = np.zeros_like(raw_values, dtype=bool)
        for i, label in enumerate(selected_labels):
            flag_name = f"{label}_FLAG"
            if flag_name in keys:
                flg = get_col(flag_name)
                #Handle Void/Structured Types safely
                if flg.dtype.names: flg = flg[flg.dtype.names[0]]
                if flg.dtype.kind == 'V': flg = flg.view('<i4')
                is_bad = (flg.astype(int) != 0)
            elif label in ['TEFF', 'LOGG', 'VMICRO', 'VMACRO', 'VSINI']:
                is_bad = (raw_values[:, i] < -5000)
            else:
                is_bad = np.zeros_like(raw_values[:, i], dtype=bool)
            bad_mask[:, i] = is_bad

        if limit:
            print(f"Truncating to first {limit} stars.")
            raw_values = raw_values[:limit]
            bad_mask = bad_mask[:limit]

    vals_to_impute = raw_values.copy()
    vals_to_impute[bad_mask] = np.nan
    bad_error_values = np.nanpercentile(vals_to_impute, 95, axis=0)

    nan_indices = np.isnan(vals_to_impute)
    clean_labels = np.where(nan_indices, bad_error_values, vals_to_impute)
         
    return clean_labels


# ╔══════════════════════════════════════════════════════════════╗
# ║ CELL 3 — Custom Loss Functions                              ║
# ╚══════════════════════════════════════════════════════════════╝

@register_keras_serializable()
def chi2_estimate(y_true,y_pred):
    y_true=tf.cast(y_true,tf.float32)
    y_pred=tf.cast(y_pred,tf.float32)
    if len(y_pred.shape) == 2:
        y_pred = tf.expand_dims(y_pred, -1)
    real_flux = y_true[:, :, 0:1]
    ivar = y_true[:, :, 1:2]
    valid_mask = tf.cast(real_flux > config.BADPIX_CUTOFF, tf.float32)    
    safe_flux = tf.where(valid_mask == 1.0, real_flux, y_pred)
    weight=1.0/(1/ivar+1e-5)
    weight=tf.where(ivar>0,weight,0.0)
    wmse_term = tf.square(safe_flux - y_pred) * weight * valid_mask
    n_valid_pixels = tf.reduce_sum(valid_mask, 1)
    n_params = len(config.SELECTED_LABELS)
    dof = n_valid_pixels - n_params
    return tf.reduce_sum(wmse_term,1)/dof
    


@register_keras_serializable()
def spl_loss(y_true, y_pred):
    if len(y_pred.shape) == 2:
        y_pred = tf.expand_dims(y_pred, -1)
    real_flux = y_true[:, :, 0:1]
    ivar = y_true[:, :, 1:2]
    valid_mask = tf.cast(real_flux > config.BADPIX_CUTOFF, tf.float32)    
    safe_flux = tf.where(valid_mask == 1.0, real_flux, y_pred)
    weight=1e-5/(1/ivar+1e-5)
    weight=tf.where(ivar>0,weight,0.0)
    wmse_term = tf.square(safe_flux - y_pred) * weight * valid_mask
    
    loss = tf.where(tf.math.is_finite(wmse_term), wmse_term, tf.zeros_like(wmse_term))
    
    return loss


# ╔══════════════════════════════════════════════════════════════╗
# ║ CELL 4 — Custom Keras Layers                                ║
# ╚══════════════════════════════════════════════════════════════╝

@tf.keras.utils.register_keras_serializable()
class ColumnSelector(layers.Layer):
    """
    A robust layer that selects specific columns from the input.
    Replaces: Lambda(lambda x: tf.gather(x, indices, axis=1))
    """
    def __init__(self, indices, **kwargs):
        super().__init__(**kwargs)
        self.indices = list(indices) 

    def call(self, inputs):
        return tf.gather(inputs, self.indices, axis=1)

    def get_config(self):
        config = super().get_config()
        config.update({'indices': self.indices})
        return config

@tf.keras.utils.register_keras_serializable()
class BeerLambertLaw(layers.Layer):
    """
    Applies the Beer-Lambert law: Flux = exp(-Tau).
    Replaces: Lambda(lambda t: tf.math.exp(-t))
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self,k ,tau):
        return k*tf.math.exp(-tau)

@tf.keras.utils.register_keras_serializable()
class GetAbundances(layers.Layer):
    """
     Returns the abundance column for each element 
    """
    def __init__(self,col_id, **kwargs):
        super().__init__(**kwargs)
        self.index=col_id

    def call(self, inputs):
        return inputs[:, self.index:self.index+1]


# ╔══════════════════════════════════════════════════════════════╗
# ║ CELL 5 — Load Data & Stats                                  ║
# ╚══════════════════════════════════════════════════════════════╝

n_pixels = 8575
h5_path="/kaggle/input/datasets/aneeshshastri/aspcapstar-dr17-150kstars/apogee_dr17_parallel.h5"
stats_path="/kaggle/input/models/aneeshshastri/stargen-comparision/tensorflow2/default/9/dataset_stats_140k.npz"
model_path="/kaggle/input/models/aneeshshastri/stargen-comparision/tensorflow2/default/9/SPECTROGRAM_GENERATOR.keras"
jacobian_mask_path="/kaggle/input/models/aneeshshastri/stargen-comparision/tensorflow2/default/9/jacobian_mask.npy"
model_path_P="/kaggle/input/models/aneeshshastri/stargen-comparision/tensorflow2/default/9/ThePayne.keras"
model_path_EP="/kaggle/input/models/aneeshshastri/stargen-comparision/tensorflow2/default/9/CNN_model.keras"
model_path_CNN="/kaggle/input/models/aneeshshastri/stargen-comparision/tensorflow2/default/9/CNN_model.keras"
model_path_RP="/kaggle/input/models/aneeshshastri/stargen-comparision/tensorflow2/default/9/CNN_model.keras"

with h5py.File(h5_path, 'r') as f:
    real_data = f['flux'][config.start:config.end]
    real_data_ivar = f['ivar'][config.start:config.end]

with np.load(stats_path) as data:
    MEAN_TENSOR= data['mean'].astype(np.float32)
    STD_TENSOR = data['std'].astype(np.float32)
    labels=get_clean_imputed_data(h5_path, config.SELECTED_LABELS)
    label_err=get_err(h5_path, config.ERRORS)[config.start:config.end]
    print(labels.shape,MEAN_TENSOR.shape,STD_TENSOR.shape)
    labels=((labels-MEAN_TENSOR)/(STD_TENSOR))[config.start:config.end]


# ╔══════════════════════════════════════════════════════════════╗
# ║ CELL 6 — Load Model                                         ║
# ╚══════════════════════════════════════════════════════════════╝

model=tf.keras.models.load_model(model_path)


# ╔══════════════════════════════════════════════════════════════╗
# ║ CELL 7 — Stratified Test Sample Selection                   ║
# ╚══════════════════════════════════════════════════════════════╝

def select_stratified_test_sample(
    h5_path: str,
    stats_path: str,
    selected_labels: list,
    test_start: int,
    test_end: int,
    target_n: int = 500,
    logg_bins: int = 5,
    teff_bins: int = 10,
    mh_bins: int = 10,
) -> np.ndarray:
    """
    Selects a stratified test sample from the test window [test_start, test_end]
    by partitioning (LOGG, TEFF, M_H) space into a 3D histogram and picking
    the median-SNR star from each occupied bin.
    """

    max_bins = logg_bins * teff_bins * mh_bins
    if target_n > max_bins:
        raise ValueError(
            f"target_n ({target_n}) exceeds the number of bins "
            f"({logg_bins}x{teff_bins}x{mh_bins} = {max_bins}). "
            f"Reduce target_n or increase bin counts."
        )

    teff_idx = selected_labels.index('TEFF')
    logg_idx = selected_labels.index('LOGG')
    mh_idx   = selected_labels.index('M_H')

    print(f"Loading test window [{test_start}:{test_end}] "
          f"({test_end - test_start} stars)...")

    with h5py.File(h5_path, 'r') as f:
        meta  = f['metadata']
        teff  = meta['TEFF'][test_start:test_end].astype(np.float32)
        logg  = meta['LOGG'][test_start:test_end].astype(np.float32)
        m_h   = meta['M_H'][test_start:test_end].astype(np.float32)
        snr   = meta['SNR'][test_start:test_end].astype(np.float32)

    n_test = test_end - test_start

    SENTINEL = -5000.0
    valid = (teff > SENTINEL) & (logg > SENTINEL) & (m_h > SENTINEL)
    print(f"  Valid stars (no sentinel values): {valid.sum()} / {n_test}")

    def percentile_edges(values, mask, n_bins):
        pts = np.linspace(0, 100, n_bins + 1)
        edges = np.percentile(values[mask], pts)
        edges[0]  -= 1e-6
        edges[-1] += 1e-6
        return edges

    logg_edges = percentile_edges(logg, valid, logg_bins)
    teff_edges = percentile_edges(teff, valid, teff_bins)
    mh_edges   = percentile_edges(m_h,  valid, mh_bins)

    print(f"  LOGG edges: {np.round(logg_edges, 2)}")
    print(f"  TEFF edges: {np.round(teff_edges, 0)}")
    print(f"  M_H  edges: {np.round(mh_edges,  2)}")

    logg_bin = np.clip(np.digitize(logg, logg_edges) - 1, 0, logg_bins - 1)
    teff_bin = np.clip(np.digitize(teff, teff_edges) - 1, 0, teff_bins - 1)
    mh_bin   = np.clip(np.digitize(m_h,  mh_edges)  - 1, 0, mh_bins  - 1)

    local_indices_selected = []
    bin_summary = []

    for i in range(logg_bins):
        for j in range(teff_bins):
            for k in range(mh_bins):
                in_bin = valid & (logg_bin == i) & (teff_bin == j) & (mh_bin == k)
                bin_local_idx = np.where(in_bin)[0]

                if len(bin_local_idx) == 0:
                    continue

                bin_snr    = snr[bin_local_idx]
                median_snr = np.median(bin_snr)
                closest    = bin_local_idx[np.argmin(np.abs(bin_snr - median_snr))]

                local_indices_selected.append(closest)
                bin_summary.append({
                    'bin': (i, j, k),
                    'n_stars': len(bin_local_idx),
                    'median_snr': median_snr,
                    'selected_snr': snr[closest],
                })

    local_indices_selected = np.array(local_indices_selected)
    n_from_bins = len(local_indices_selected)
    print(f"\n  Occupied bins: {n_from_bins} / {max_bins}")

    if n_from_bins < target_n:
        shortfall = target_n - n_from_bins
        already_selected = set(local_indices_selected.tolist())
        remaining = np.array([
            idx for idx in np.where(valid)[0]
            if idx not in already_selected
        ])
        if len(remaining) < shortfall:
            print(f"  Warning: only {len(remaining)} additional valid stars "
                  f"available; test set will be smaller than target_n.")
            shortfall = len(remaining)

        rng = np.random.default_rng(42)
        extra = rng.choice(remaining, size=shortfall, replace=False)
        local_indices_selected = np.concatenate([local_indices_selected, extra])
        print(f"  Topped up with {shortfall} random valid stars "
              f"to reach {len(local_indices_selected)} total.")

    elif n_from_bins > target_n:
        occupancies = np.array([b['n_stars'] for b in bin_summary])
        order = np.argsort(occupancies)
        keep  = order[:target_n]
        local_indices_selected = local_indices_selected[keep]
        print(f"  Downsampled from {n_from_bins} to {target_n}, "
              f"prioritising sparse bins.")

    global_indices = np.sort(local_indices_selected + test_start)

    selected_teff = teff[global_indices - test_start]
    selected_logg = logg[global_indices - test_start]
    selected_mh   = m_h[global_indices  - test_start]
    selected_snr  = snr[global_indices  - test_start]

    print(f"\n{'='*50}")
    print(f"  Final test set: {len(global_indices)} stars")
    print(f"  TEFF  — min: {selected_teff.min():.0f}  "
               f"max: {selected_teff.max():.0f}  "
               f"median: {np.median(selected_teff):.0f}")
    print(f"  LOGG  — min: {selected_logg.min():.2f}  "
               f"max: {selected_logg.max():.2f}  "
               f"median: {np.median(selected_logg):.2f}")
    print(f"  [M/H] — min: {selected_mh.min():.2f}   "
               f"max: {selected_mh.max():.2f}   "
               f"median: {np.median(selected_mh):.2f}")
    print(f"  SNR   — min: {selected_snr.min():.1f}   "
               f"max: {selected_snr.max():.1f}   "
               f"median: {np.median(selected_snr):.1f}")
    print(f"{'='*50}\n")

    return global_indices


def load_test_spectra(h5_path, global_indices):
    """
    Loads flux and ivar for the selected test stars.
    """
    with h5py.File(h5_path, 'r') as f:
        flux = f['flux'][global_indices].astype(np.float32)
        ivar = f['ivar'][global_indices].astype(np.float32)
    return flux, ivar


# --- Run stratified selection ---
TEST_START = 140_000
TEST_END   = 149_500

global_indices = select_stratified_test_sample(
    h5_path        = config.H5_PATH,
    stats_path     = config.STATS_PATH,
    selected_labels= config.SELECTED_LABELS,
    test_start     = TEST_START,
    test_end       = TEST_END,
    target_n       = 500,
    logg_bins      = 5,
    teff_bins      = 10,
    mh_bins        = 10,
)

np.save("test_indices.npy", global_indices)
print(f"Saved {len(global_indices)} test indices to test_indices.npy")

flux, ivar = load_test_spectra(config.H5_PATH, global_indices)
print(f"Flux shape: {flux.shape}, IVAR shape: {ivar.shape}")


# ╔══════════════════════════════════════════════════════════════╗
# ║ CELL 8 — Physical Bounds & Bijectors                        ║
# ╚══════════════════════════════════════════════════════════════╝

import tensorflow_probability as tfp
tfb = tfp.bijectors

# Ordered exactly as SELECTED_LABELS
lower_bounds = [
    # Core
    3000.0, -0.5, -2.5,  0.0,  0.0,   0.0,  
    # CNO
    -1.5,  -1.0, -1.0, 
    # Metals
    -2.5,  -1.5, -1.0, -1.0, -1.5, -1.0, 
    -1.5,  -1.5, -1.5, -4.0, -2.5, -4.5, 
    -2.5,  -2.5
]

upper_bounds = [
    # Core
    20000.0, 5.0,  1.0,  3.0, 15.0, 80.0, 
    # CNO
    1.5,   3.5,  1.0,  
    # Metals
    1.0,   1.0,  1.5,  1.0,  1.0,  1.0,  
    1.5,   1.5,  1.0,  1.5,  2.0,  3.5,  
    1.5,   2.0
]

bounds_low = tf.constant(lower_bounds, dtype=tf.float32)
bounds_high = tf.constant(upper_bounds, dtype=tf.float32)

# Bijector to map Unconstrained space → Bounded space (Physical Labels)
unconstrained_to_physical = tfb.Chain([
    tfb.Shift(shift=bounds_low),
    tfb.Scale(scale=(bounds_high - bounds_low)),
    tfb.Sigmoid()
])

# Inverse: Physical → Unconstrained (for HMC initialisation)
physical_to_unconstrained = tfb.Invert(unconstrained_to_physical)


# ╔══════════════════════════════════════════════════════════════╗
# ║ CELL 9 — Rebuild Model in Pure FP32                         ║
# ╚══════════════════════════════════════════════════════════════╝

import json

# 1. Extract the raw architecture blueprint
config_m = model.get_config()
config_str = json.dumps(config_m)

# 2. Aggressively purge all low-precision flags
config_str = config_str.replace('"float16"', '"float32"')
config_str = config_str.replace('"mixed_float16"', '"float32"')

# 3. Rebuild the dictionary
clean_config = json.loads(config_str)

# 4. Construct a brand new graph using the sanitized config
model_pure_fp32 = model.__class__.from_config(clean_config)

# 5. Inject the trained weights
model_pure_fp32.set_weights(model.get_weights())

print("Model aggressively rebuilt in pure FP32.")


# ╔══════════════════════════════════════════════════════════════╗
# ║ CELL 10 — MCMC Constants & Shape Contract   [CHANGED]       ║
# ╚══════════════════════════════════════════════════════════════╝

import time

# ============================================================
# CONSTANTS
# ============================================================
BATCH_SIZE_STARS = 10       # ← was 5;  stars processed in parallel per MCMC call
NUM_CHAINS       = 8        # independent chains per star
NUM_RESULTS      = 500      # posterior samples to keep (after burn-in)
NUM_BURNIN       = 1000     # ← was 400;  burn-in steps
NUM_ADAPT        = 800      # ← was 320;  80 % of burn-in
MAX_TREE_DEPTH   = 6
TARGET_ACCEPT    = 0.8
TOTAL_STARS      = 500

RESULTS_DIR = "/kaggle/working/mcmc_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

CHECKPOINT_PATH = os.path.join(RESULTS_DIR, "checkpoint.npz")
RESULTS_PATH    = os.path.join(RESULTS_DIR, "results.npz")

N_LABELS_RAW = 23   # physical labels fed to MCMC
N_LABELS_ALL = 27   # raw + 4 engineered features
N_PIXELS     = 8575

LABEL_NAMES = config.SELECTED_LABELS   # length 23, ordered


# ============================================================
# SHAPE CONTRACT
# ============================================================
# NUTS current_state : (num_chains, batch_stars, 23)
# obs_flux / obs_ivar: (batch_stars, 8575)          — broadcast over chains
# model input        : (batch_stars * num_chains, 27) — flattened for one forward pass
# log_prob output    : (num_chains, batch_stars)     — TFP expects this shape
#
# CRITICAL: step_size shape is (1, batch_stars, 23) so each star
# gets its own adapted step size, shared only across chains.


# ╔══════════════════════════════════════════════════════════════╗
# ║ CELL 11 — Feature Engineering  (unchanged logic)            ║
# ╚══════════════════════════════════════════════════════════════╝

def get_27_features(labels_23):
    """
    labels_23 : (..., 23)  — any leading batch dimensions
    returns    : (..., 27)
    """
    teff   = labels_23[ :, config.SELECTED_LABELS.index('TEFF')]
    logg   = labels_23[ :, config.SELECTED_LABELS.index('LOGG')]
    vmacro = labels_23[ :, config.SELECTED_LABELS.index('VMACRO')]
    vsini  = labels_23[ :, config.SELECTED_LABELS.index('VSINI')]
    c_fe   = labels_23[ :, config.SELECTED_LABELS.index('C_FE')]
    o_fe   = labels_23[ :, config.SELECTED_LABELS.index('O_FE')]
    m_h    = labels_23[ :, config.SELECTED_LABELS.index('M_H')]

    inv_teff  = 5040.0 / (teff + 1e-6)
    v_broad   = tf.sqrt(tf.square(vmacro) + tf.square(vsini))
    c_minus_o = c_fe - o_fe
    log_pe    = 0.5 * logg + 0.5 * m_h

    eng = tf.stack([inv_teff, v_broad, c_minus_o, log_pe], axis=-1)
    return tf.concat([labels_23, eng], axis=-1)


# ╔══════════════════════════════════════════════════════════════╗
# ║ CELL 12 — Log-Probability  [FIXED: tf.repeat + transpose]   ║
# ╚══════════════════════════════════════════════════════════════╝

@tf.function(reduce_retracing=True)
def target_log_prob_fn(theta, obs_flux, obs_ivar):
    """
    theta     : (num_chains, batch_stars, 23)  — physical space
    obs_flux  : (batch_stars, 8575)
    obs_ivar  : (batch_stars, 8575)
    returns   : (num_chains, batch_stars)

    Flattening order:
        theta  → transpose to (B, C, 23) → reshape to (B*C, 23)
        obs    → tf.repeat each star C times       → (B*C, 8575)
    This guarantees star i's spectrum is always paired with star i's
    proposals across all chains — no cross-contamination.
    """
    num_chains  = tf.shape(theta)[0]
    batch_stars = tf.shape(theta)[1]

    # --- flatten: (C, B, 23) → (B, C, 23) → (B*C, 23) ---
    theta_BCL  = tf.transpose(theta, [1, 0, 2])             # (B, C, 23)
    theta_flat = tf.reshape(theta_BCL, [-1, N_LABELS_RAW])   # (B*C, 23)

    labels_27   = get_27_features(theta_flat)                # (B*C, 27)
    labels_norm = (labels_27 - MEAN_TENSOR) / STD_TENSOR     # (B*C, 27)

    model_flux  = model_pure_fp32(labels_norm, training=False)
    if len(model_flux.shape) == 3:
        model_flux = model_flux[:, :, 0]                      # (B*C, 8575)

    # --- repeat observed data: each star repeated C times ---
    obs_flux_rep = tf.repeat(obs_flux, num_chains, axis=0)    # (B*C, 8575)
    obs_ivar_rep = tf.repeat(obs_ivar, num_chains, axis=0)    # (B*C, 8575)

    # --- sanitise ---
    safe_flux = tf.where(tf.math.is_finite(obs_flux_rep), obs_flux_rep, 0.0)
    safe_ivar = tf.where(
        tf.math.is_finite(obs_ivar_rep) & tf.math.is_finite(obs_flux_rep),
        obs_ivar_rep, 0.0
    )
    safe_ivar = 1.0 / (1.0 / (safe_ivar) + 1e-5)

    # --- pixel mask ---
    flux_valid = tf.cast((safe_flux > 1e-3) & (safe_flux < 1.3), tf.float32)
    ivar_valid = tf.cast(safe_ivar > 0.0, tf.float32)
    mask       = flux_valid * ivar_valid                       # (B*C, 8575)

    # --- χ² ---
    chi2_flat = tf.reduce_sum(
        tf.square(safe_flux - model_flux) * safe_ivar * mask,
        axis=1
    )                                                          # (B*C,)

    # --- reshape: (B*C,) → (B, C) → transpose to (C, B) ---
    chi2_BC = tf.reshape(chi2_flat, [batch_stars, num_chains])  # (B, C)
    return -0.5 * tf.transpose(chi2_BC)                        # (C, B)


# ╔══════════════════════════════════════════════════════════════╗
# ║ CELL 13 — Initialisation Helper  [FIXED: no .numpy()]       ║
# ╚══════════════════════════════════════════════════════════════╝

def make_init_state(batch_stars, num_chains, physical_init=None):
    """
    Returns initial state in PHYSICAL space: (num_chains, batch_stars, 23).

    physical_init : (batch_stars, 23) optional — ASPCAP labels as warm start.
                    If None, uses training mean for all stars.
    batch_stars   : int or tf.Tensor — both work (no .numpy() needed).
    """
    margin     = 1e-3
    safe_lower = bounds_low  + margin          # (23,)
    safe_upper = bounds_high - margin          # (23,)

    if physical_init is not None:
        base = tf.cast(physical_init, tf.float32)
        base = tf.expand_dims(base, 0)
        base = tf.tile(base, [num_chains, 1, 1])
    else:
        base = tf.reshape(MEAN_TENSOR[:N_LABELS_RAW], [1, 1, N_LABELS_RAW])
        base = tf.tile(base, [num_chains, batch_stars, 1])

    # Use tf.shape(base) so this works inside tf.function
    jitter = tf.random.normal(
        tf.shape(base)
    ) * (0.05 * STD_TENSOR[:N_LABELS_RAW])

    return tf.clip_by_value(base + jitter, safe_lower, safe_upper)


# ╔══════════════════════════════════════════════════════════════╗
# ║ CELL 14 — Compiled Sampler  [NEW: single XLA via tf.Variable]║
# ╚══════════════════════════════════════════════════════════════╝

# Persistent variables — assigned new data each batch, but shape stays fixed
# so XLA only compiles once.
_obs_flux_var = tf.Variable(
    tf.zeros([BATCH_SIZE_STARS, N_PIXELS], tf.float32), trainable=False
)
_obs_ivar_var = tf.Variable(
    tf.zeros([BATCH_SIZE_STARS, N_PIXELS], tf.float32), trainable=False
)
_init_state_var = tf.Variable(
    tf.zeros([NUM_CHAINS, BATCH_SIZE_STARS, N_LABELS_RAW], tf.float32),
    trainable=False,
)


@tf.function(jit_compile=True)
def sample_compiled():
    """
    Runs NUTS with dual-averaging step-size adaptation.
    
    Reads observed data and initial state from the tf.Variable feeds
    (_obs_flux_var, _obs_ivar_var, _init_state_var).
    
    Returns
    -------
    samples : (NUM_RESULTS, NUM_CHAINS, BATCH_SIZE_STARS, 23) unconstrained
    is_accepted : (NUM_RESULTS, NUM_CHAINS, BATCH_SIZE_STARS) bool
    """

    def log_prob_closure(theta_unconstrained):
        theta_physical = unconstrained_to_physical(theta_unconstrained)
        return target_log_prob_fn(theta_physical, _obs_flux_var, _obs_ivar_var)

    # Per-star step size: (1, B, 23) — shared across chains, independent per star
    initial_step_size = tf.fill(
        [1, BATCH_SIZE_STARS, N_LABELS_RAW], 0.05
    )

    nuts = tfp.mcmc.NoUTurnSampler(
        target_log_prob_fn=log_prob_closure,
        step_size=initial_step_size,
        max_tree_depth=MAX_TREE_DEPTH,
    )

    adaptive_kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
        inner_kernel=nuts,
        num_adaptation_steps=NUM_ADAPT,
        target_accept_prob=TARGET_ACCEPT,
        step_size_setter_fn=lambda pkr, s: pkr._replace(step_size=s),
        step_size_getter_fn=lambda pkr: pkr.step_size,
        log_accept_prob_getter_fn=lambda pkr: pkr.log_accept_ratio,
    )

    # Map initial state to unconstrained space for NUTS
    init_unconstrained = physical_to_unconstrained(_init_state_var)

    return tfp.mcmc.sample_chain(
        num_results=NUM_RESULTS,
        num_burnin_steps=NUM_BURNIN,
        current_state=init_unconstrained,
        kernel=adaptive_kernel,
        trace_fn=lambda _, pkr: pkr.inner_results.is_accepted,
    )


# ╔══════════════════════════════════════════════════════════════╗
# ║ CELL 15 — Single-Batch MCMC  [REWRITTEN]                    ║
# ╚══════════════════════════════════════════════════════════════╝

def run_mcmc_batch(obs_flux, obs_ivar, physical_init=None):
    """
    obs_flux      : (BATCH_SIZE_STARS, 8575)
    obs_ivar      : (BATCH_SIZE_STARS, 8575)
    physical_init : (BATCH_SIZE_STARS, 23) optional warm start in physical units

    Returns
    -------
    medians      : (BATCH_SIZE_STARS, 23)  — posterior median, physical units
    lower_sigma  : (BATCH_SIZE_STARS, 23)  — median − 16th percentile
    upper_sigma  : (BATCH_SIZE_STARS, 23)  — 84th percentile − median
    accept_rate  : float
    r_hat        : (23,)  — Gelman-Rubin per label (should be < 1.01)
    """
    batch_stars = BATCH_SIZE_STARS  # always fixed (padded if necessary)
    obs_flux = tf.cast(obs_flux, tf.float32)
    obs_ivar = tf.cast(obs_ivar, tf.float32)

    init_state = make_init_state(
        batch_stars, NUM_CHAINS, physical_init
    )  # (C, B, 23) physical

    # Feed data into the persistent variables
    _obs_flux_var.assign(obs_flux)
    _obs_ivar_var.assign(obs_ivar)
    _init_state_var.assign(init_state)

    # Run the compiled sampler (XLA compiled only on first call)
    samples_unc, is_accepted = sample_compiled()
    # samples_unc: (NUM_RESULTS, NUM_CHAINS, BATCH_SIZE_STARS, 23) — unconstrained

    # Map samples back to physical space
    samples_phys = unconstrained_to_physical(samples_unc)
    # shape: (NUM_RESULTS, NUM_CHAINS, BATCH_SIZE_STARS, 23)

    # --- Gelman-Rubin per label ---
    r_hat_per_star = tfp.mcmc.diagnostic.potential_scale_reduction(
        samples_phys, independent_chain_ndims=1
    )
    # r_hat_per_star: (BATCH_SIZE_STARS, 23) → mean over stars
    r_hat = tf.reduce_mean(r_hat_per_star, axis=0).numpy()  # (23,)

    # --- Pool chains for percentile computation ---
    n_total = NUM_RESULTS * NUM_CHAINS
    flat = tf.reshape(samples_phys, [n_total, -1, N_LABELS_RAW])

    pcts = tfp.stats.percentile(flat, [16.0, 50.0, 84.0], axis=0)
    # pcts: (3, B, 23)

    medians     = pcts[1].numpy()              # (B, 23)
    lower_sigma = (pcts[1] - pcts[0]).numpy()  # (B, 23)
    upper_sigma = (pcts[2] - pcts[1]).numpy()  # (B, 23)

    accept_rate = tf.reduce_mean(
        tf.cast(is_accepted, tf.float32)
    ).numpy()

    return medians, lower_sigma, upper_sigma, accept_rate, r_hat


# ╔══════════════════════════════════════════════════════════════╗
# ║ CELL 16 — Checkpoint Helpers  (unchanged)                   ║
# ╚══════════════════════════════════════════════════════════════╝

def load_checkpoint():
    """Returns (results_dict, completed_count) or (fresh_dict, 0)."""
    fresh = {
        'global_indices':    [],
        'true_labels':       [],
        'inferred_labels':   [],
        'lower_sigma':       [],
        'upper_sigma':       [],
        'aspcap_errors':     [],
        'acceptance_rates':  [],
        'r_hat':             [],
        'wall_seconds':      [],
    }
    if os.path.exists(CHECKPOINT_PATH):
        data = np.load(CHECKPOINT_PATH, allow_pickle=True)
        results = {k: list(data[k]) for k in fresh}
        n_done  = len(results['global_indices'])
        print(f"  Resuming from checkpoint: {n_done} stars already done.")
        return results, n_done
    return fresh, 0


def save_checkpoint(results):
    """Saves current state to checkpoint (overwrites previous)."""
    np.savez(CHECKPOINT_PATH, **{
        k: np.array(v) for k, v in results.items()
    })


def save_final_results(results):
    """Saves the completed results with human-readable label names."""
    arrays = {k: np.array(v) for k, v in results.items()}
    arrays['label_names'] = np.array(LABEL_NAMES)
    np.savez(RESULTS_PATH, **arrays)
    print(f"\nFinal results saved to {RESULTS_PATH}")
    print(f"  Keys: {list(arrays.keys())}")


# ╔══════════════════════════════════════════════════════════════╗
# ║ CELL 17 — Outer Loop  [FIXED: padding + checkpoint freq]    ║
# ╚══════════════════════════════════════════════════════════════╝

def run_inference_pipeline(
    test_indices,       # (500,) global HDF5 indices from stratified sampler
    true_labels_norm,   # (N_test, 27) normalised labels from get_clean_imputed_data
    aspcap_errors,      # (N_test, 23) label errors from get_err
    flux_array,         # (N_test, 8575) pre-loaded flux
    ivar_array,         # (N_test, 8575) pre-loaded ivar
):
    """
    Runs batched NUTS inference over all test stars in fixed-size
    batches of BATCH_SIZE_STARS, checkpointing periodically.

    The last batch is padded to BATCH_SIZE_STARS with zeros to
    avoid an XLA recompilation; padded results are discarded.
    """

    n_stars = len(test_indices)
    assert n_stars <= len(flux_array), \
        "test_indices contains indices outside the loaded flux window"

    # Denormalise true labels back to physical units for storage
    true_labels_physical = (
        true_labels_norm[:, :N_LABELS_RAW] * STD_TENSOR[:N_LABELS_RAW]
        + MEAN_TENSOR[:N_LABELS_RAW]
    )  # (N_test, 23)

    # --- Load checkpoint (enables resume after crash) ---
    results, n_done = load_checkpoint()
    remaining_local = list(range(n_done, n_stars))

    print(f"\n{'='*60}")
    print(f"  Inference pipeline: {n_stars} stars, "
          f"batch size {BATCH_SIZE_STARS}, {NUM_CHAINS} chains")
    print(f"  Burn-in: {NUM_BURNIN}, Samples: {NUM_RESULTS}, "
          f"Adaptation: {NUM_ADAPT}")
    print(f"  To do: {len(remaining_local)} stars")
    print(f"{'='*60}\n")

    total_start = time.time()
    batch_count = 0
    CHECKPOINT_EVERY = 5  # checkpoint every N batches

    # Iterate in chunks of BATCH_SIZE_STARS
    for batch_start in range(0, len(remaining_local), BATCH_SIZE_STARS):

        batch_local = remaining_local[batch_start:batch_start + BATCH_SIZE_STARS]
        actual_batch = len(batch_local)  # last batch may be smaller

        # --- Gather data for this batch ---
        batch_flux = flux_array[batch_local]            # (actual, 8575)
        batch_ivar = ivar_array[batch_local]            # (actual, 8575)
        batch_init = true_labels_physical[batch_local]  # (actual, 23)

        # --- Pad last batch to fixed BATCH_SIZE_STARS ---
        if actual_batch < BATCH_SIZE_STARS:
            pad_n = BATCH_SIZE_STARS - actual_batch
            batch_flux = np.concatenate([
                batch_flux,
                np.zeros((pad_n, N_PIXELS), dtype=np.float32)
            ])
            batch_ivar = np.concatenate([
                batch_ivar,
                np.zeros((pad_n, N_PIXELS), dtype=np.float32)
            ])
            # Pad init with copies of first star (value doesn't matter,
            # zero ivar means these contribute zero likelihood)
            batch_init = np.concatenate([
                batch_init,
                np.tile(batch_init[:1], (pad_n, 1))
            ])

        # --- Run MCMC ---
        t0 = time.time()
        medians, lo_sig, hi_sig, acc_rate, r_hat = run_mcmc_batch(
            obs_flux       = batch_flux,
            obs_ivar       = batch_ivar,
            physical_init  = batch_init,
        )
        elapsed = time.time() - t0

        # --- Discard padding, keep only real stars ---
        medians = medians[:actual_batch]
        lo_sig  = lo_sig[:actual_batch]
        hi_sig  = hi_sig[:actual_batch]

        # --- Store results star by star ---
        for b in range(actual_batch):
            local_idx  = batch_local[b]
            global_idx = int(test_indices[local_idx])

            results['global_indices'].append(global_idx)
            results['true_labels'].append(
                true_labels_physical[local_idx]
            )
            results['inferred_labels'].append(medians[b])
            results['lower_sigma'].append(lo_sig[b])
            results['upper_sigma'].append(hi_sig[b])
            results['aspcap_errors'].append(
                aspcap_errors[local_idx, :N_LABELS_RAW]
            )
            results['acceptance_rates'].append(acc_rate)
            results['r_hat'].append(r_hat)
            results['wall_seconds'].append(elapsed / actual_batch)

        n_done += actual_batch
        batch_count += 1

        # --- Checkpoint periodically ---
        if batch_count % CHECKPOINT_EVERY == 0 or n_done >= n_stars:
            save_checkpoint(results)

        # --- Progress report ---
        max_rhat = r_hat.max()
        converged = "✓" if max_rhat < 1.05 else f"⚠ max R̂={max_rhat:.3f}"
        print(
            f"  [{n_done:>4}/{n_stars}]  "
            f"batch {batch_count}  |  "
            f"accept={acc_rate:.2f}  |  "
            f"R̂ {converged}  |  "
            f"{elapsed:.1f}s  "
            f"(~{(n_stars - n_done) * elapsed / actual_batch / 60:.0f} min left)"
        )

    total_elapsed = time.time() - total_start
    print(f"\nAll {n_stars} stars done in {total_elapsed/60:.1f} min.")

    save_final_results(results)
    return results


# ╔══════════════════════════════════════════════════════════════╗
# ║ CELL 18 — Loading Saved Results  (unchanged)                ║
# ╚══════════════════════════════════════════════════════════════╝

def load_results(path=RESULTS_PATH):
    """
    Loads the saved results and returns a clean dict with named arrays.

    Example usage:
        r = load_results()
        teff_inferred = r['inferred_labels'][:, 0]
        teff_true     = r['true_labels'][:, 0]
        teff_err      = r['aspcap_errors'][:, 0]
    """
    data = np.load(path, allow_pickle=True)
    r = {k: data[k] for k in data.files}

    if 'label_names' in r:
        r['label_index'] = {
            name: i for i, name in enumerate(r['label_names'])
        }

    print("Loaded results:")
    print(f"  Stars:  {len(r['global_indices'])}")
    print(f"  Labels: {r.get('label_names', ['(not stored)'])}")
    print(f"  Keys:   {[k for k in r if k != 'label_names']}")
    return r


# ╔══════════════════════════════════════════════════════════════╗
# ║ CELL 19 — Execution                                         ║
# ╚══════════════════════════════════════════════════════════════╝

test_indices = np.load("/kaggle/working/test_indices.npy")
results = run_inference_pipeline(
    test_indices      = test_indices,
    true_labels_norm  = labels,
    aspcap_errors     = label_err,
    flux_array        = real_data,
    ivar_array        = real_data_ivar,
)
