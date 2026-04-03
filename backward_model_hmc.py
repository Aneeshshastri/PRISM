"""
backward_model_hmc.py — Tier 2: Fixed-Step HMC (XLA-compiled)
==============================================================
Replaces NUTS with tfp.mcmc.HamiltonianMonteCarlo using fixed
num_leapfrog_steps, enabling full @tf.function(jit_compile=True).

Two-stage FERRE-style inference:
  Stage 1 — 9D core physics (TEFF, LOGG, M_H, VMICRO, VMACRO,
            VSINI, C_FE, N_FE, O_FE) on full spectrum
  Stage 2 — 14 × 1D individual abundances on element-masked pixels

CNN warm-start provides initialisation + Gaussian prior.

Run on Kaggle with GPU enabled.
"""

import os
import time
import json
import numpy as np
import h5py
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.saving import register_keras_serializable
from tensorflow.keras import layers
from sklearn.experimental import enable_iterative_imputer   # noqa
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge

tfb = tfp.bijectors

# ================================================================
# CONFIG
# ================================================================

class Config:
    H5_PATH    = "/kaggle/input/datasets/aneeshshastri/aspcapstar-dr17-150kstars/apogee_dr17_parallel.h5"
    TFREC_DIR  = "/kaggle/working/tfrecords"
    STATS_PATH = "/kaggle/working/dataset_stats.npz"
    start = 140_000
    end   = 149_500
    TESTING_MODE  = True
    RANDOM_SEED   = 42
    OUTPUT_LENGTH = 8575
    BADPIX_CUTOFF = 1e-3
    SELECTED_LABELS = [
        'TEFF', 'LOGG', 'M_H', 'VMICRO', 'VMACRO', 'VSINI',
        'C_FE', 'N_FE', 'O_FE',
        'FE_H',
        'MG_FE', 'SI_FE', 'CA_FE', 'TI_FE', 'S_FE',
        'AL_FE', 'MN_FE', 'NI_FE', 'CR_FE', 'K_FE',
        'NA_FE', 'V_FE', 'CO_FE',
    ]
    ABUNDANCE_INDICES = [i for i, l in enumerate(SELECTED_LABELS) if '_FE' in l]
    FE_H_INDEX = SELECTED_LABELS.index('FE_H')
    N_LABELS   = len(SELECTED_LABELS) + 4
    ERRORS     = [f'{l}_ERR' for l in SELECTED_LABELS]

config = Config()
np.random.seed(config.RANDOM_SEED)
tf.random.set_seed(config.RANDOM_SEED)

# ================================================================
# CUSTOM LAYERS (needed for model loading)
# ================================================================

@register_keras_serializable()
class ColumnSelector(layers.Layer):
    def __init__(self, indices, **kwargs):
        super().__init__(**kwargs)
        self.indices = list(indices)
    def call(self, inputs):
        return tf.gather(inputs, self.indices, axis=1)
    def get_config(self):
        c = super().get_config(); c['indices'] = self.indices; return c

@register_keras_serializable()
class BeerLambertLaw(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def call(self, k, tau):
        return k * tf.math.exp(-tau)

@register_keras_serializable()
class GetAbundances(layers.Layer):
    def __init__(self, col_id, **kwargs):
        super().__init__(**kwargs)
        self.index = col_id
    def call(self, inputs):
        return inputs[:, self.index:self.index+1]
    def get_config(self):
        c = super().get_config(); c['col_id'] = self.index; return c

@register_keras_serializable()
class SparseProjector(tf.keras.layers.Layer):
    def __init__(self, active_indices, weights, total_pixels, label_name="unknown", **kwargs):
        kwargs['name'] = f"Sparse_Projector_{label_name}"
        super().__init__(**kwargs)
        self.total_pixels = total_pixels
        self.label_name = label_name
        self.active_indices = tf.constant(active_indices, dtype=tf.int32)
        self.weights_tensor = tf.constant(weights, dtype=tf.float32)

    def call(self, local_tau):
        local_tau_32 = tf.cast(local_tau, tf.float32)
        weighted = local_tau_32 * self.weights_tensor[None, :]
        batch_size = tf.shape(local_tau)[0]
        n_active = tf.shape(self.active_indices)[0]
        batch_ids = tf.repeat(tf.range(batch_size), n_active)
        pixel_ids = tf.tile(self.active_indices, tf.expand_dims(batch_size, axis=0))
        scatter_idx = tf.stack([batch_ids, pixel_ids], axis=-1)
        flat_weighted = tf.reshape(weighted, [-1])
        full = tf.scatter_nd(
            scatter_idx, flat_weighted,
            shape=tf.stack([batch_size, tf.constant(self.total_pixels, dtype=tf.int32)])
        )
        return full

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.total_pixels)

    def get_config(self):
        c = super().get_config()
        c.update({
            'active_indices': self.active_indices.numpy().tolist(),
            'weights': self.weights_tensor.numpy().tolist(),
            'total_pixels': self.total_pixels,
            'label_name': self.label_name,
        })
        return c

    @classmethod
    def from_config(cls, config):
        config['active_indices'] = np.array(config['active_indices'], dtype=np.int32)
        config['weights'] = np.array(config['weights'], dtype=np.float32)
        config.pop('name', None)
        return cls(**config)

# CNN predictor classes (needed for model loading)
N_LABELS_RAW = 23
N_PIXELS     = 8575

@register_keras_serializable()
class HeteroscedasticCNNPredictor(tf.keras.Model):
    def __init__(self, n_labels=N_LABELS_RAW, **kwargs):
        super().__init__(**kwargs)
        self.n_labels = n_labels
        self.reshape_in = layers.Reshape((N_PIXELS, 1))
        self.conv1 = layers.Conv1D(32, kernel_size=16, strides=4, activation='relu', padding='same')
        self.bn1   = layers.BatchNormalization()
        self.conv2 = layers.Conv1D(64, kernel_size=8, strides=4, activation='relu', padding='same')
        self.bn2   = layers.BatchNormalization()
        self.conv3 = layers.Conv1D(128, kernel_size=4, strides=2, activation='relu', padding='same')
        self.bn3   = layers.BatchNormalization()
        self.conv4 = layers.Conv1D(256, kernel_size=4, strides=2, activation='relu', padding='same')
        self.bn4   = layers.BatchNormalization()
        self.gap   = layers.GlobalAveragePooling1D()
        self.dense1 = layers.Dense(512, activation='relu')
        self.drop1  = layers.Dropout(0.3)
        self.dense2 = layers.Dense(256, activation='relu')
        self.drop2  = layers.Dropout(0.2)
        self.mean_head    = layers.Dense(n_labels, name='label_mean')
        self.log_var_head = layers.Dense(n_labels, name='label_log_var')

    def call(self, x, training=False):
        h = self.reshape_in(x)
        h = self.bn1(self.conv1(h), training=training)
        h = self.bn2(self.conv2(h), training=training)
        h = self.bn3(self.conv3(h), training=training)
        h = self.bn4(self.conv4(h), training=training)
        h = self.gap(h)
        h = self.drop1(self.dense1(h), training=training)
        h = self.drop2(self.dense2(h), training=training)
        return self.mean_head(h), self.log_var_head(h)

    def get_config(self):
        c = super().get_config(); c['n_labels'] = self.n_labels; return c

@register_keras_serializable()
class TrainableCNNPredictor(HeteroscedasticCNNPredictor):
    pass

# ================================================================
# DATA LOADING
# ================================================================

def get_clean_imputed_data(h5_path, selected_labels, limit=None):
    print("Read data for imputation")
    with h5py.File(h5_path, 'r') as f:
        get_col = lambda k: f['metadata'][k]
        keys = f['metadata'].dtype.names
        raw_values = np.stack([get_col(p) for p in selected_labels], axis=1)
        bad_mask = np.zeros_like(raw_values, dtype=bool)
        for i, label in enumerate(selected_labels):
            flag_name = f"{label}_FLAG"
            if flag_name in keys:
                flg = get_col(flag_name)
                if flg.dtype.names: flg = flg[flg.dtype.names[0]]
                if flg.dtype.kind == 'V': flg = flg.view('<i4')
                bad_mask[:, i] = (flg.astype(int) != 0)
            elif label in ['TEFF', 'LOGG', 'VMICRO', 'VMACRO', 'VSINI']:
                bad_mask[:, i] = (raw_values[:, i] < -5000)

    if limit:
        raw_values = raw_values[:limit]
        bad_mask = bad_mask[:limit]

    vals_to_impute = raw_values.copy()
    vals_to_impute[bad_mask] = np.nan
    imputer = IterativeImputer(estimator=BayesianRidge(), max_iter=10, initial_strategy='median')
    clean_labels = imputer.fit_transform(vals_to_impute)

    fe_h_col = clean_labels[:, config.FE_H_INDEX:config.FE_H_INDEX+1]
    for idx in config.ABUNDANCE_INDICES:
        clean_labels[:, idx] += fe_h_col[:, 0]

    teff_vals = clean_labels[:, selected_labels.index('TEFF')]
    inv_teff = (5040.0 / (teff_vals + 1e-6)).reshape(-1, 1)
    clean_labels = np.hstack([clean_labels, inv_teff])

    vmacro_vals = clean_labels[:, selected_labels.index('VMACRO')]
    vsini_vals = clean_labels[:, selected_labels.index('VSINI')]
    quadrature = np.sqrt(vmacro_vals**2 + vsini_vals**2).reshape(-1, 1)
    clean_labels = np.hstack([clean_labels, quadrature])

    c_vals = clean_labels[:, selected_labels.index('C_FE')]
    o_vals = clean_labels[:, selected_labels.index('O_FE')]
    clean_labels = np.hstack([clean_labels, (c_vals - o_vals).reshape(-1, 1)])

    g_vals = clean_labels[:, selected_labels.index('LOGG')]
    m_vals = clean_labels[:, selected_labels.index('M_H')]
    clean_labels = np.hstack([clean_labels, (0.5*g_vals + 0.5*m_vals).reshape(-1, 1)])

    print(f"Transformation Complete. Final Input Shape: {clean_labels.shape}")
    return clean_labels

def get_err(h5_path, selected_labels, limit=None):
    with h5py.File(h5_path, 'r') as f:
        get_col = lambda k: f['metadata'][k]
        keys = f['metadata'].dtype.names
        raw_values = np.stack([
            get_col(p) if p not in ['VMACRO_ERR','VMICRO_ERR','VSINI_ERR']
            else np.zeros_like(get_col('TEFF_ERR'))
            for p in selected_labels
        ], axis=1)
        bad_mask = np.zeros_like(raw_values, dtype=bool)
        for i, label in enumerate(selected_labels):
            flag_name = f"{label}_FLAG"
            if flag_name in keys:
                flg = get_col(flag_name)
                if flg.dtype.names: flg = flg[flg.dtype.names[0]]
                if flg.dtype.kind == 'V': flg = flg.view('<i4')
                bad_mask[:, i] = (flg.astype(int) != 0)
            elif label in ['TEFF', 'LOGG', 'VMICRO', 'VMACRO', 'VSINI']:
                bad_mask[:, i] = (raw_values[:, i] < -5000)
        if limit:
            raw_values = raw_values[:limit]
            bad_mask = bad_mask[:limit]
    vals_to_impute = raw_values.copy()
    vals_to_impute[bad_mask] = np.nan
    p95 = np.nanpercentile(vals_to_impute, 95, axis=0)
    return np.where(np.isnan(vals_to_impute), p95, vals_to_impute)

# ================================================================
# LOAD EVERYTHING
# ================================================================

print("Loading data...")
h5_path   = config.H5_PATH
stats_path = "/kaggle/input/models/aneeshshastri/stargen-comparision/tensorflow2/default/14/dataset_stats_120k.npz"
model_path = "/kaggle/input/models/aneeshshastri/stargen-comparision/tensorflow2/default/14/APOGEE_SGEN.keras"

cnn_model_path = "/kaggle/input/models/aneeshshastri/backward-warmstart/tensorflow2/default/1/cnn_label_predictor_inner (1).keras"
cnn_stats_path = "/kaggle/input/models/aneeshshastri/backward-warmstart/tensorflow2/default/1/cnn_label_stats.npz"

apogee_mask_path = "/kaggle/input/datasets/aneeshshastri/element-masks/apogee_mask.npy"

with h5py.File(h5_path, 'r') as f:
    real_data      = f['flux'][config.start:config.end]
    real_data_ivar = f['ivar'][config.start:config.end]

with np.load(stats_path) as data:
    MEAN_TENSOR = data['mean'].astype(np.float32)
    STD_TENSOR  = data['std'].astype(np.float32)

labels    = get_clean_imputed_data(h5_path, config.SELECTED_LABELS)
label_err = get_err(h5_path, config.ERRORS)[config.start:config.end]
labels    = ((labels - MEAN_TENSOR) / STD_TENSOR)[config.start:config.end]

# Load forward model and rebuild in FP32
model_raw = tf.keras.models.load_model(model_path)
config_m = model_raw.get_config()
config_str = json.dumps(config_m)
config_str = config_str.replace('"float16"', '"float32"')
config_str = config_str.replace('"mixed_float16"', '"float32"')
clean_config = json.loads(config_str)
model_pure_fp32 = model_raw.__class__.from_config(clean_config)
model_pure_fp32.set_weights(model_raw.get_weights())
print("Forward model rebuilt in FP32.")

# Load CNN predictor
cnn_predictor = tf.keras.models.load_model(cnn_model_path)
with np.load(cnn_stats_path) as cnn_stats:
    CNN_LABEL_MEAN = cnn_stats['mean'].astype(np.float32)
    CNN_LABEL_STD  = cnn_stats['std'].astype(np.float32)
print(f"CNN label predictor loaded.")

def cnn_predict_physical(flux_batch):
    flux_clean = flux_batch.copy()
    bad = ~np.isfinite(flux_clean) | (flux_clean <= 1e-3)
    flux_clean[bad] = 1.0
    mu_norm, raw_logvar = cnn_predictor(
        tf.constant(flux_clean, dtype=tf.float32), training=False
    )
    mu_phys = mu_norm.numpy() * CNN_LABEL_STD + CNN_LABEL_MEAN
    var_norm = tf.nn.softplus(raw_logvar).numpy() + 1e-4
    std_phys = np.sqrt(var_norm) * CNN_LABEL_STD
    std_phys = np.maximum(std_phys, 0.01 * CNN_LABEL_STD)
    return mu_phys.astype(np.float32), std_phys.astype(np.float32)

# ================================================================
# CONSTANTS
# ================================================================

N_LABELS_ALL = 27
LABEL_NAMES  = config.SELECTED_LABELS

BATCH_SIZE_STARS = 10
TOTAL_STARS      = 5    # adjust for your test set

CORE_INDICES  = [0, 1, 2, 3, 4, 5, 6, 7, 8]
N_CORE        = len(CORE_INDICES)
ABUND_INDICES = [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
N_ABUND       = len(ABUND_INDICES)

# --- HMC hyperparameters (replacing NUTS) ---
NUM_CHAINS_CORE     = 8
NUM_RESULTS_CORE    = 500
NUM_BURNIN_CORE     = 400
NUM_ADAPT_CORE      = 320
NUM_LEAPFROG_CORE   = 16     # Fixed leapfrog steps — enables XLA
TARGET_ACCEPT_CORE  = 0.65   # Lower than NUTS's 0.8 for fixed HMC

NUM_CHAINS_ELEM     = 4
NUM_RESULTS_ELEM    = 300
NUM_BURNIN_ELEM     = 200
NUM_ADAPT_ELEM      = 160
NUM_LEAPFROG_ELEM   = 10     # 1D elements need fewer steps

PRIOR_WEIGHT = 1.0

RESULTS_DIR     = "/kaggle/working/hmc_results"
os.makedirs(RESULTS_DIR, exist_ok=True)
CHECKPOINT_PATH = os.path.join(RESULTS_DIR, "checkpoint.npz")
RESULTS_PATH    = os.path.join(RESULTS_DIR, "results.npz")

# Bounds
lower_bounds = [
    3000.0, -0.5, -2.5, 0.0, 0.0, 0.0,
    -1.5, -1.0, -1.0,
    -2.5, -1.5, -1.0, -1.0, -1.5, -1.0,
    -1.5, -1.5, -1.5, -4.0, -2.5, -4.5,
    -2.5, -2.5
]
upper_bounds = [
    20000.0, 5.0, 1.0, 3.0, 15.0, 80.0,
    1.5, 3.5, 1.0,
    1.0, 1.0, 1.5, 1.0, 1.0, 1.0,
    1.5, 1.5, 1.0, 1.5, 2.0, 3.5,
    1.5, 2.0
]
bounds_low  = tf.constant(lower_bounds, dtype=tf.float32)
bounds_high = tf.constant(upper_bounds, dtype=tf.float32)

# Element pixel masks
_raw_mask = np.load(apogee_mask_path)
ELEMENT_PIXEL_MASKS = {}
for abund_idx in ABUND_INDICES:
    mask_col = _raw_mask[:, abund_idx]
    sensitive = (mask_col > 0.01).astype(np.float32)
    if sensitive.sum() < 10:
        sensitive = np.ones(N_PIXELS, dtype=np.float32)
        print(f"  ⚠ {LABEL_NAMES[abund_idx]} has <10 pixels, using full spectrum")
    ELEMENT_PIXEL_MASKS[abund_idx] = tf.constant(sensitive, tf.float32)
    print(f"  {LABEL_NAMES[abund_idx]:>6s}: {int(sensitive.sum()):>5d} sensitive pixels")

# ================================================================
# FEATURE ENGINEERING
# ================================================================

def get_27_features(labels_23):
    teff   = labels_23[:, config.SELECTED_LABELS.index('TEFF')]
    logg   = labels_23[:, config.SELECTED_LABELS.index('LOGG')]
    vmacro = labels_23[:, config.SELECTED_LABELS.index('VMACRO')]
    vsini  = labels_23[:, config.SELECTED_LABELS.index('VSINI')]
    c_fe   = labels_23[:, config.SELECTED_LABELS.index('C_FE')]
    o_fe   = labels_23[:, config.SELECTED_LABELS.index('O_FE')]
    m_h    = labels_23[:, config.SELECTED_LABELS.index('M_H')]
    inv_teff  = 5040.0 / (teff + 1e-6)
    v_broad   = tf.sqrt(tf.square(vmacro) + tf.square(vsini))
    c_minus_o = c_fe - o_fe
    log_pe    = 0.5 * logg + 0.5 * m_h
    eng = tf.stack([inv_teff, v_broad, c_minus_o, log_pe], axis=-1)
    return tf.concat([labels_23, eng], axis=-1)

# ================================================================
# TF.VARIABLE STATE (avoids retracing)
# ================================================================

_cnn_mu_var  = tf.Variable(tf.zeros([BATCH_SIZE_STARS, N_LABELS_RAW], tf.float32), trainable=False)
_cnn_std_var = tf.Variable(tf.ones([BATCH_SIZE_STARS, N_LABELS_RAW], tf.float32), trainable=False)
_fixed_abund_var = tf.Variable(tf.zeros([BATCH_SIZE_STARS, N_ABUND], tf.float32), trainable=False)
_fixed_full_var  = tf.Variable(tf.zeros([BATCH_SIZE_STARS, N_LABELS_RAW], tf.float32), trainable=False)
_elem_col_var    = tf.Variable(0, dtype=tf.int32, trainable=False)
_elem_pixel_mask_var = tf.Variable(tf.ones([N_PIXELS], tf.float32), trainable=False)

_obs_flux_var = tf.Variable(tf.zeros([BATCH_SIZE_STARS, N_PIXELS], tf.float32), trainable=False)
_obs_ivar_var = tf.Variable(tf.zeros([BATCH_SIZE_STARS, N_PIXELS], tf.float32), trainable=False)

# ================================================================
# LOG PROBABILITY FUNCTIONS (graph-safe)
# ================================================================

@tf.function(reduce_retracing=True)
def _assemble_full_23(core_params, fixed_abund):
    parts = []
    for i in range(N_LABELS_RAW):
        if i in CORE_INDICES:
            ci = CORE_INDICES.index(i)
            parts.append(core_params[:, ci:ci+1])
        else:
            ai = ABUND_INDICES.index(i)
            parts.append(fixed_abund[:, ai:ai+1])
    return tf.concat(parts, axis=1)


@tf.function(reduce_retracing=True)
def core_log_prob_fn(theta_core, obs_flux, obs_ivar):
    """theta_core: (C, B, 9) physical → returns (C, B)"""
    num_chains  = tf.shape(theta_core)[0]
    batch_stars = tf.shape(theta_core)[1]

    theta_BCL  = tf.transpose(theta_core, [1, 0, 2])
    theta_flat = tf.reshape(theta_BCL, [-1, N_CORE])
    abund_rep  = tf.repeat(_fixed_abund_var, num_chains, axis=0)
    full_23    = _assemble_full_23(theta_flat, abund_rep)

    labels_27   = get_27_features(full_23)
    labels_norm = (labels_27 - MEAN_TENSOR) / STD_TENSOR

    model_flux = model_pure_fp32(labels_norm, training=False)
    model_flux = tf.cast(model_flux, tf.float32)
    model_flux = tf.squeeze(model_flux, axis=-1)  # safe for (B,8575) or (B,8575,1)

    obs_flux_rep = tf.repeat(obs_flux, num_chains, axis=0)
    obs_ivar_rep = tf.repeat(obs_ivar, num_chains, axis=0)

    safe_flux = tf.where(tf.math.is_finite(obs_flux_rep), obs_flux_rep, 0.0)
    safe_ivar = tf.where(
        tf.math.is_finite(obs_ivar_rep) & tf.math.is_finite(obs_flux_rep),
        obs_ivar_rep, 0.0
    )
    safe_ivar = 1.0 / (1.0 / (safe_ivar + 1e-12) + 1e-5)

    flux_valid = tf.cast((safe_flux > 1e-3) & (safe_flux < 1.3), tf.float32)
    ivar_valid = tf.cast(safe_ivar > 0.0, tf.float32)
    mask = flux_valid * ivar_valid

    chi2_flat = tf.reduce_sum(
        tf.square(safe_flux - model_flux) * safe_ivar * mask, axis=1
    )
    log_lik_flat = -0.5 * chi2_flat

    # CNN prior
    cnn_mu_core  = tf.gather(_cnn_mu_var, CORE_INDICES, axis=1)
    cnn_std_core = tf.gather(_cnn_std_var, CORE_INDICES, axis=1)
    cnn_mu_rep   = tf.repeat(cnn_mu_core, num_chains, axis=0)
    cnn_std_rep  = tf.repeat(cnn_std_core, num_chains, axis=0)
    log_prior = -0.5 * tf.reduce_sum(
        tf.square((theta_flat - cnn_mu_rep) / cnn_std_rep), axis=1
    )

    log_post = log_lik_flat + PRIOR_WEIGHT * log_prior
    log_post_BC = tf.reshape(log_post, [batch_stars, num_chains])
    return tf.transpose(log_post_BC)


@tf.function(reduce_retracing=True)
def element_log_prob_fn(theta_1d, obs_flux, obs_ivar):
    """theta_1d: (C, B, 1) physical → returns (C, B)"""
    num_chains  = tf.shape(theta_1d)[0]
    batch_stars = tf.shape(theta_1d)[1]
    elem_col    = _elem_col_var

    theta_BCL  = tf.transpose(theta_1d, [1, 0, 2])
    theta_flat = tf.reshape(theta_BCL, [-1, 1])

    full_rep = tf.repeat(_fixed_full_var, num_chains, axis=0)
    bc = tf.shape(full_rep)[0]
    indices = tf.stack([tf.range(bc), tf.fill([bc], elem_col)], axis=1)
    full_spliced = tf.tensor_scatter_nd_update(full_rep, indices, theta_flat[:, 0])

    labels_27   = get_27_features(full_spliced)
    labels_norm = (labels_27 - MEAN_TENSOR) / STD_TENSOR

    model_flux = model_pure_fp32(labels_norm, training=False)
    model_flux = tf.cast(model_flux, tf.float32)
    model_flux = tf.squeeze(model_flux, axis=-1)

    obs_flux_rep = tf.repeat(obs_flux, num_chains, axis=0)
    obs_ivar_rep = tf.repeat(obs_ivar, num_chains, axis=0)

    safe_flux = tf.where(tf.math.is_finite(obs_flux_rep), obs_flux_rep, 0.0)
    safe_ivar = tf.where(
        tf.math.is_finite(obs_ivar_rep) & tf.math.is_finite(obs_flux_rep),
        obs_ivar_rep, 0.0
    )
    safe_ivar = 1.0 / (1.0 / (safe_ivar + 1e-12) + 1e-5)

    flux_valid = tf.cast((safe_flux > 1e-3) & (safe_flux < 1.3), tf.float32)
    ivar_valid = tf.cast(safe_ivar > 0.0, tf.float32)
    elem_mask  = tf.reshape(_elem_pixel_mask_var, [1, N_PIXELS])
    mask = flux_valid * ivar_valid * elem_mask

    chi2_flat = tf.reduce_sum(
        tf.square(safe_flux - model_flux) * safe_ivar * mask, axis=1
    )
    log_lik_flat = -0.5 * chi2_flat

    cnn_mu_elem  = _cnn_mu_var[:, elem_col:elem_col+1]
    cnn_std_elem = _cnn_std_var[:, elem_col:elem_col+1]
    cnn_mu_rep   = tf.repeat(cnn_mu_elem, num_chains, axis=0)
    cnn_std_rep  = tf.repeat(cnn_std_elem, num_chains, axis=0)
    log_prior = -0.5 * tf.reduce_sum(
        tf.square((theta_flat - cnn_mu_rep) / cnn_std_rep), axis=1
    )

    log_post = log_lik_flat + PRIOR_WEIGHT * log_prior
    log_post_BC = tf.reshape(log_post, [batch_stars, num_chains])
    return tf.transpose(log_post_BC)

# ================================================================
# BIJECTORS
# ================================================================

_core_low_tf  = tf.gather(bounds_low, CORE_INDICES)
_core_high_tf = tf.gather(bounds_high, CORE_INDICES)
_core_bijector = tfb.Chain([
    tfb.Shift(_core_low_tf),
    tfb.Scale(_core_high_tf - _core_low_tf),
    tfb.Sigmoid(),
])
_core_inv_bijector = tfb.Invert(_core_bijector)

# ================================================================
# INIT HELPERS
# ================================================================

def make_init_state_core(batch_stars, num_chains, physical_init_23):
    margin = 1e-3
    safe_lower = bounds_low + margin
    safe_upper = bounds_high - margin
    core_init = tf.gather(tf.cast(physical_init_23, tf.float32), CORE_INDICES, axis=1)
    base = tf.tile(tf.expand_dims(core_init, 0), [num_chains, 1, 1])
    core_std = tf.gather(STD_TENSOR[:N_LABELS_RAW], CORE_INDICES)
    jitter = tf.random.normal(tf.shape(base)) * (0.05 * core_std)
    core_low  = tf.gather(safe_lower, CORE_INDICES)
    core_high = tf.gather(safe_upper, CORE_INDICES)
    return tf.clip_by_value(base + jitter, core_low, core_high)


def make_init_state_elem(batch_stars, num_chains, phys_value_1d):
    base = tf.reshape(tf.cast(phys_value_1d, tf.float32), [1, -1, 1])
    base = tf.tile(base, [num_chains, 1, 1])
    jitter = tf.random.normal(tf.shape(base)) * 0.02
    return base + jitter

# ================================================================
# HMC SAMPLING — XLA COMPILED (the key difference from NUTS)
# ================================================================

_init_core_var = tf.Variable(
    tf.zeros([NUM_CHAINS_CORE, BATCH_SIZE_STARS, N_CORE], tf.float32),
    trainable=False,
)

# NOTE: jit_compile=True may fail with DualAveragingStepSizeAdaptation on some
# TFP versions. If you get XLA errors, the log-prob is still JIT-compiled via
# core_log_prob_fn's @tf.function, so this is still fast.
@tf.function(jit_compile=True)
def sample_core_compiled():
    """Fixed-step HMC for Stage 1: 9D core physics, full spectrum. XLA compiled."""
    def log_prob_closure(theta_unc):
        return core_log_prob_fn(
            _core_bijector.forward(theta_unc),
            _obs_flux_var, _obs_ivar_var
        )

    step_size = tf.fill([1, BATCH_SIZE_STARS, N_CORE], 0.01)
    hmc = tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=log_prob_closure,
        step_size=step_size,
        num_leapfrog_steps=NUM_LEAPFROG_CORE,
    )
    adaptive = tfp.mcmc.DualAveragingStepSizeAdaptation(
        inner_kernel=hmc,
        num_adaptation_steps=NUM_ADAPT_CORE,
        target_accept_prob=TARGET_ACCEPT_CORE,
    )
    init_unc = _core_inv_bijector.forward(_init_core_var)
    return tfp.mcmc.sample_chain(
        num_results=NUM_RESULTS_CORE,
        num_burnin_steps=NUM_BURNIN_CORE,
        current_state=init_unc,
        kernel=adaptive,
        trace_fn=lambda _, pkr: pkr.inner_results.is_accepted,
    )


_elem_low_var  = tf.Variable(-3.0, dtype=tf.float32, trainable=False)
_elem_high_var = tf.Variable(3.0, dtype=tf.float32, trainable=False)
_init_elem_var = tf.Variable(
    tf.zeros([NUM_CHAINS_ELEM, BATCH_SIZE_STARS, 1], tf.float32),
    trainable=False,
)

@tf.function(jit_compile=True)
def sample_element_compiled():
    # If XLA fails here, change jit_compile=True to reduce_retracing=True above
    """Fixed-step HMC for Stage 2: 1D abundance, element-masked spectrum. XLA compiled."""
    lo = tf.reshape(_elem_low_var, [1])
    hi = tf.reshape(_elem_high_var, [1])
    elem_bij = tfb.Chain([
        tfb.Shift(lo), tfb.Scale(hi - lo), tfb.Sigmoid(),
    ])
    elem_inv = tfb.Invert(elem_bij)

    def log_prob_closure(theta_unc):
        return element_log_prob_fn(
            elem_bij.forward(theta_unc),
            _obs_flux_var, _obs_ivar_var
        )

    step_size = tf.fill([1, BATCH_SIZE_STARS, 1], 0.01)
    hmc = tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=log_prob_closure,
        step_size=step_size,
        num_leapfrog_steps=NUM_LEAPFROG_ELEM,
    )
    adaptive = tfp.mcmc.DualAveragingStepSizeAdaptation(
        inner_kernel=hmc,
        num_adaptation_steps=NUM_ADAPT_ELEM,
        target_accept_prob=TARGET_ACCEPT_CORE,
    )
    init_unc = elem_inv.forward(_init_elem_var)
    return tfp.mcmc.sample_chain(
        num_results=NUM_RESULTS_ELEM,
        num_burnin_steps=NUM_BURNIN_ELEM,
        current_state=init_unc,
        kernel=adaptive,
        trace_fn=lambda _, pkr: pkr.inner_results.is_accepted,
    )

# ================================================================
# MCMC RUNNERS
# ================================================================

def run_core_mcmc(obs_flux, obs_ivar, physical_init_23, cnn_mean, cnn_std):
    obs_flux = tf.cast(obs_flux, tf.float32)
    obs_ivar = tf.cast(obs_ivar, tf.float32)

    _cnn_mu_var.assign(tf.cast(cnn_mean, tf.float32))
    _cnn_std_var.assign(tf.cast(cnn_std, tf.float32))

    abund_init = tf.gather(tf.cast(cnn_mean, tf.float32), ABUND_INDICES, axis=1)
    _fixed_abund_var.assign(abund_init)

    init_core = make_init_state_core(BATCH_SIZE_STARS, NUM_CHAINS_CORE, physical_init_23)
    _obs_flux_var.assign(obs_flux)
    _obs_ivar_var.assign(obs_ivar)
    _init_core_var.assign(init_core)

    samples_unc, is_accepted = sample_core_compiled()
    samples_phys = _core_bijector.forward(samples_unc)

    r_hat = tf.reduce_mean(
        tfp.mcmc.diagnostic.potential_scale_reduction(
            samples_phys, independent_chain_ndims=1
        ), axis=0
    ).numpy()

    n_total = NUM_RESULTS_CORE * NUM_CHAINS_CORE
    flat = tf.reshape(samples_phys, [n_total, -1, N_CORE])
    pcts = tfp.stats.percentile(flat, [16.0, 50.0, 84.0], axis=0)

    medians = pcts[1].numpy()
    lo_sig  = (pcts[1] - pcts[0]).numpy()
    hi_sig  = (pcts[2] - pcts[1]).numpy()
    accept  = tf.reduce_mean(tf.cast(is_accepted, tf.float32)).numpy()

    return medians, lo_sig, hi_sig, accept, r_hat


def run_element_mcmc(obs_flux, obs_ivar, fixed_labels_23, elem_idx, cnn_mean, cnn_std):
    obs_flux = tf.cast(obs_flux, tf.float32)
    obs_ivar = tf.cast(obs_ivar, tf.float32)

    _fixed_full_var.assign(tf.cast(fixed_labels_23, tf.float32))
    _elem_col_var.assign(elem_idx)
    _elem_pixel_mask_var.assign(ELEMENT_PIXEL_MASKS[elem_idx])
    _elem_low_var.assign(bounds_low[elem_idx])
    _elem_high_var.assign(bounds_high[elem_idx])

    init_val = cnn_mean[:, elem_idx]
    init_state = make_init_state_elem(BATCH_SIZE_STARS, NUM_CHAINS_ELEM, init_val)

    _obs_flux_var.assign(obs_flux)
    _obs_ivar_var.assign(obs_ivar)
    _init_elem_var.assign(init_state)

    samples_unc, is_accepted = sample_element_compiled()

    lo_val = bounds_low[elem_idx]
    hi_val = bounds_high[elem_idx]
    bij = tfb.Chain([
        tfb.Shift(tf.constant([lo_val])),
        tfb.Scale(tf.constant([hi_val - lo_val])),
        tfb.Sigmoid(),
    ])
    samples_phys = bij.forward(samples_unc)

    r_hat = tf.reduce_mean(
        tfp.mcmc.diagnostic.potential_scale_reduction(
            samples_phys, independent_chain_ndims=1
        )
    ).numpy()

    n_total = NUM_RESULTS_ELEM * NUM_CHAINS_ELEM
    flat = tf.reshape(samples_phys, [n_total, -1, 1])
    pcts = tfp.stats.percentile(flat, [16.0, 50.0, 84.0], axis=0)

    median = pcts[1, :, 0].numpy()
    lo_sig = (pcts[1, :, 0] - pcts[0, :, 0]).numpy()
    hi_sig = (pcts[2, :, 0] - pcts[1, :, 0]).numpy()
    accept = tf.reduce_mean(tf.cast(is_accepted, tf.float32)).numpy()

    return median, lo_sig, hi_sig, accept, r_hat

# ================================================================
# CHECKPOINT / RESULTS
# ================================================================

def load_checkpoint():
    fresh = {
        'global_indices': [], 'true_labels': [], 'inferred_labels': [],
        'lower_sigma': [], 'upper_sigma': [], 'aspcap_errors': [],
        'acceptance_rates': [], 'r_hat': [], 'wall_seconds': [],
    }
    if os.path.exists(CHECKPOINT_PATH):
        data = np.load(CHECKPOINT_PATH, allow_pickle=True)
        results = {k: list(data[k]) for k in fresh}
        n_done = len(results['global_indices'])
        print(f"  Resuming from checkpoint: {n_done} stars already done.")
        return results, n_done
    return fresh, 0

def save_checkpoint(results):
    np.savez(CHECKPOINT_PATH, **{k: np.array(v) for k, v in results.items()})

def save_final_results(results):
    arrays = {k: np.array(v) for k, v in results.items()}
    arrays['label_names'] = np.array(LABEL_NAMES)
    np.savez(RESULTS_PATH, **arrays)
    print(f"\nFinal results saved to {RESULTS_PATH}")

# ================================================================
# INFERENCE PIPELINE
# ================================================================

def run_inference_pipeline(test_indices, true_labels_norm, aspcap_errors,
                           flux_array, ivar_array):
    n_stars = len(test_indices)
    true_labels_physical = (
        true_labels_norm[:, :N_LABELS_RAW] * STD_TENSOR[:N_LABELS_RAW]
        + MEAN_TENSOR[:N_LABELS_RAW]
    )
    results, n_done = load_checkpoint()
    remaining_local = list(range(n_done, n_stars))

    print(f"\n{'='*65}")
    print(f"  Tier 2: Fixed-step HMC (XLA compiled)")
    print(f"  {n_stars} stars, batch {BATCH_SIZE_STARS}")
    print(f"  Stage 1 (core {N_CORE}D): {NUM_CHAINS_CORE}ch, "
          f"{NUM_LEAPFROG_CORE} leapfrog, {NUM_BURNIN_CORE} burn, {NUM_RESULTS_CORE} samp")
    print(f"  Stage 2 ({N_ABUND} elem × 1D): {NUM_CHAINS_ELEM}ch, "
          f"{NUM_LEAPFROG_ELEM} leapfrog, {NUM_BURNIN_ELEM} burn, {NUM_RESULTS_ELEM} samp")
    print(f"  To do: {len(remaining_local)} stars")
    print(f"{'='*65}\n")

    total_start = time.time()
    batch_count = 0

    for batch_start in range(0, len(remaining_local), BATCH_SIZE_STARS):
        batch_local = remaining_local[batch_start:batch_start + BATCH_SIZE_STARS]
        actual_batch = len(batch_local)

        batch_flux = flux_array[batch_local]
        batch_ivar = ivar_array[batch_local]
        cnn_mu, cnn_sig = cnn_predict_physical(batch_flux)

        # Pad if needed
        if actual_batch < BATCH_SIZE_STARS:
            pad_n = BATCH_SIZE_STARS - actual_batch
            batch_flux = np.concatenate([batch_flux, np.zeros((pad_n, N_PIXELS), np.float32)])
            batch_ivar = np.concatenate([batch_ivar, np.zeros((pad_n, N_PIXELS), np.float32)])
            cnn_mu  = np.concatenate([cnn_mu, np.tile(cnn_mu[:1], (pad_n, 1))])
            cnn_sig = np.concatenate([cnn_sig, np.tile(cnn_sig[:1], (pad_n, 1))])

        # Stage 1: Core
        t_s1 = time.time()
        core_med, core_lo, core_hi, s1_acc, s1_rhat = run_core_mcmc(
            batch_flux, batch_ivar, cnn_mu, cnn_mu, cnn_sig
        )
        t_s1_elapsed = time.time() - t_s1
        s1_tag = "✓" if s1_rhat.max() < 1.05 else f"⚠ {s1_rhat.max():.3f}"
        print(f"    S1 core: acc={s1_acc:.2f}  R̂ {s1_tag}  {t_s1_elapsed:.1f}s")

        fixed_23 = cnn_mu.copy()
        for ci, gi in enumerate(CORE_INDICES):
            fixed_23[:, gi] = core_med[:, ci]

        # Stage 2: Elements
        t_s2 = time.time()
        abund_med = np.zeros((BATCH_SIZE_STARS, N_ABUND), np.float32)
        abund_lo  = np.zeros((BATCH_SIZE_STARS, N_ABUND), np.float32)
        abund_hi  = np.zeros((BATCH_SIZE_STARS, N_ABUND), np.float32)
        s2_rhats, s2_accs = [], []

        for ai, elem_idx in enumerate(ABUND_INDICES):
            e_med, e_lo, e_hi, e_acc, e_rhat = run_element_mcmc(
                batch_flux, batch_ivar, fixed_23, elem_idx, cnn_mu, cnn_sig
            )
            abund_med[:, ai] = e_med
            abund_lo[:, ai] = e_lo
            abund_hi[:, ai] = e_hi
            s2_rhats.append(e_rhat)
            s2_accs.append(e_acc)
            fixed_23[:, elem_idx] = e_med

        t_s2_elapsed = time.time() - t_s2
        s2_tag = "✓" if max(s2_rhats) < 1.05 else f"⚠ {max(s2_rhats):.3f}"
        print(f"    S2 elem: acc={np.mean(s2_accs):.2f}  R̂ {s2_tag}  "
              f"{t_s2_elapsed:.1f}s  ({N_ABUND} elements)")

        # Assemble
        full_med  = np.zeros((BATCH_SIZE_STARS, N_LABELS_RAW), np.float32)
        full_lo   = np.zeros((BATCH_SIZE_STARS, N_LABELS_RAW), np.float32)
        full_hi   = np.zeros((BATCH_SIZE_STARS, N_LABELS_RAW), np.float32)
        full_rhat = np.zeros(N_LABELS_RAW, np.float32)

        for ci, gi in enumerate(CORE_INDICES):
            full_med[:, gi] = core_med[:, ci]
            full_lo[:, gi]  = core_lo[:, ci]
            full_hi[:, gi]  = core_hi[:, ci]
            full_rhat[gi]   = s1_rhat[ci]

        for ai, gi in enumerate(ABUND_INDICES):
            full_med[:, gi] = abund_med[:, ai]
            full_lo[:, gi]  = abund_lo[:, ai]
            full_hi[:, gi]  = abund_hi[:, ai]
            full_rhat[gi]   = s2_rhats[ai]

        elapsed = t_s1_elapsed + t_s2_elapsed
        avg_acc = (s1_acc + np.mean(s2_accs)) / 2

        for b in range(actual_batch):
            local_idx  = batch_local[b]
            global_idx = int(test_indices[local_idx])
            results['global_indices'].append(global_idx)
            results['true_labels'].append(true_labels_physical[local_idx])
            results['inferred_labels'].append(full_med[b])
            results['lower_sigma'].append(full_lo[b])
            results['upper_sigma'].append(full_hi[b])
            results['aspcap_errors'].append(aspcap_errors[local_idx, :N_LABELS_RAW])
            results['acceptance_rates'].append(avg_acc)
            results['r_hat'].append(full_rhat)
            results['wall_seconds'].append(elapsed / actual_batch)

        n_done += actual_batch
        batch_count += 1
        save_checkpoint(results)

        est_left = (n_stars - n_done) * elapsed / actual_batch / 60
        print(f"  [{n_done:>4}/{n_stars}]  batch {batch_count}  |  "
              f"{elapsed:.1f}s  (~{est_left:.0f} min left)\n")

    total_elapsed = time.time() - total_start
    print(f"\nAll {n_stars} stars done in {total_elapsed/60:.1f} min.")
    save_final_results(results)
    return results

# ================================================================
# MAIN — TEST SET SELECTION + RUN
# ================================================================

if __name__ == "__main__":

    def select_stratified_test_sample(h5_path, stats_path, selected_labels,
                                       test_start, test_end, target_n=5,
                                       logg_bins=5, teff_bins=10, mh_bins=10):
        """Identical to notebook; selects stratified test stars."""
        max_bins = logg_bins * teff_bins * mh_bins
        with h5py.File(h5_path, 'r') as f:
            meta = f['metadata']
            teff = meta['TEFF'][test_start:test_end].astype(np.float32)
            logg = meta['LOGG'][test_start:test_end].astype(np.float32)
            m_h  = meta['M_H'][test_start:test_end].astype(np.float32)
            snr  = meta['SNR'][test_start:test_end].astype(np.float32)

        SENTINEL = -5000.0
        valid = (teff > SENTINEL) & (logg > SENTINEL) & (m_h > SENTINEL)

        def percentile_edges(values, mask, n_bins):
            pts = np.linspace(0, 100, n_bins + 1)
            edges = np.percentile(values[mask], pts)
            edges[0] -= 1e-6; edges[-1] += 1e-6
            return edges

        logg_edges = percentile_edges(logg, valid, logg_bins)
        teff_edges = percentile_edges(teff, valid, teff_bins)
        mh_edges   = percentile_edges(m_h, valid, mh_bins)

        logg_bin = np.clip(np.digitize(logg, logg_edges) - 1, 0, logg_bins - 1)
        teff_bin = np.clip(np.digitize(teff, teff_edges) - 1, 0, teff_bins - 1)
        mh_bin   = np.clip(np.digitize(m_h, mh_edges) - 1, 0, mh_bins - 1)

        local_indices_selected = []
        bin_summary = []
        for i in range(logg_bins):
            for j in range(teff_bins):
                for k in range(mh_bins):
                    in_bin = valid & (logg_bin == i) & (teff_bin == j) & (mh_bin == k)
                    idx = np.where(in_bin)[0]
                    if len(idx) == 0: continue
                    median_snr = np.median(snr[idx])
                    closest = idx[np.argmin(np.abs(snr[idx] - median_snr))]
                    local_indices_selected.append(closest)
                    bin_summary.append({'n_stars': len(idx)})

        local_indices_selected = np.array(local_indices_selected)
        n_from_bins = len(local_indices_selected)

        if n_from_bins < target_n:
            shortfall = target_n - n_from_bins
            remaining = np.array([i for i in np.where(valid)[0]
                                  if i not in set(local_indices_selected.tolist())])
            rng = np.random.default_rng(42)
            extra = rng.choice(remaining, size=min(shortfall, len(remaining)), replace=False)
            local_indices_selected = np.concatenate([local_indices_selected, extra])
        elif n_from_bins > target_n:
            occupancies = np.array([b['n_stars'] for b in bin_summary])
            order = np.argsort(occupancies)
            local_indices_selected = local_indices_selected[order[:target_n]]

        global_indices = np.sort(local_indices_selected + test_start)
        print(f"Selected {len(global_indices)} test stars.")
        return global_indices

    TEST_START = 140_000
    TEST_END   = 149_500

    global_indices = select_stratified_test_sample(
        config.H5_PATH, config.STATS_PATH, config.SELECTED_LABELS,
        TEST_START, TEST_END, target_n=TOTAL_STARS, logg_bins=5, teff_bins=10, mh_bins=10,
    )
    np.save("test_indices.npy", global_indices)

    test_indices = np.load("test_indices.npy")
    results = run_inference_pipeline(
        test_indices=test_indices,
        true_labels_norm=labels,
        aspcap_errors=label_err,
        flux_array=real_data,
        ivar_array=real_data_ivar,
    )
