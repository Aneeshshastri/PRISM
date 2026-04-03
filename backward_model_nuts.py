"""
backward_model_nuts.py — Tier 3: JIT Log-Prob + Eager NUTS
===========================================================
Keeps NUTS for adaptive trajectory lengths, but JIT-compiles the
expensive log-probability function for speed.

This is the graph-safe version of backward-model.ipynb with all
.numpy() calls removed, Python if replaced with tf.squeeze, and
FP32 properly enforced.

NUTS itself runs eagerly (via @tf.function without jit_compile)
because its dynamic tree-doubling is incompatible with XLA.

Two-stage FERRE-style:
  Stage 1 — 9D core physics (NUTS, full spectrum)
  Stage 2 — 14 × 1D abundances (NUTS, element-masked pixels)

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
# CUSTOM LAYERS
# ================================================================

@register_keras_serializable()
class ColumnSelector(layers.Layer):
    def __init__(self, indices, **kwargs):
        super().__init__(**kwargs); self.indices = list(indices)
    def call(self, inputs): return tf.gather(inputs, self.indices, axis=1)
    def get_config(self):
        c = super().get_config(); c['indices'] = self.indices; return c

@register_keras_serializable()
class BeerLambertLaw(layers.Layer):
    def __init__(self, **kwargs): super().__init__(**kwargs)
    def call(self, k, tau): return k * tf.math.exp(-tau)

@register_keras_serializable()
class GetAbundances(layers.Layer):
    def __init__(self, col_id, **kwargs):
        super().__init__(**kwargs); self.index = col_id
    def call(self, inputs): return inputs[:, self.index:self.index+1]
    def get_config(self):
        c = super().get_config(); c['col_id'] = self.index; return c

@register_keras_serializable()
class SparseProjector(tf.keras.layers.Layer):
    def __init__(self, active_indices, weights, total_pixels, label_name="unknown", **kwargs):
        kwargs['name'] = f"Sparse_Projector_{label_name}"
        super().__init__(**kwargs)
        self.total_pixels = total_pixels; self.label_name = label_name
        self.active_indices = tf.constant(active_indices, dtype=tf.int32)
        self.weights_tensor = tf.constant(weights, dtype=tf.float32)
    def call(self, local_tau):
        local_tau_32 = tf.cast(local_tau, tf.float32)
        weighted = local_tau_32 * self.weights_tensor[None, :]
        bs = tf.shape(local_tau)[0]; na = tf.shape(self.active_indices)[0]
        batch_ids = tf.repeat(tf.range(bs), na)
        pixel_ids = tf.tile(self.active_indices, tf.expand_dims(bs, axis=0))
        idx = tf.stack([batch_ids, pixel_ids], axis=-1)
        return tf.scatter_nd(idx, tf.reshape(weighted, [-1]),
                             shape=tf.stack([bs, tf.constant(self.total_pixels, dtype=tf.int32)]))
    def compute_output_shape(self, s): return (s[0], self.total_pixels)
    def get_config(self):
        c = super().get_config()
        c.update({'active_indices': self.active_indices.numpy().tolist(),
                  'weights': self.weights_tensor.numpy().tolist(),
                  'total_pixels': self.total_pixels, 'label_name': self.label_name})
        return c
    @classmethod
    def from_config(cls, cfg):
        cfg['active_indices'] = np.array(cfg['active_indices'], dtype=np.int32)
        cfg['weights'] = np.array(cfg['weights'], dtype=np.float32)
        cfg.pop('name', None); return cls(**cfg)

N_LABELS_RAW = 23; N_PIXELS = 8575

@register_keras_serializable()
class HeteroscedasticCNNPredictor(tf.keras.Model):
    def __init__(self, n_labels=N_LABELS_RAW, **kwargs):
        super().__init__(**kwargs); self.n_labels = n_labels
        self.reshape_in = layers.Reshape((N_PIXELS, 1))
        self.conv1 = layers.Conv1D(32,16,strides=4,activation='relu',padding='same')
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv1D(64,8,strides=4,activation='relu',padding='same')
        self.bn2 = layers.BatchNormalization()
        self.conv3 = layers.Conv1D(128,4,strides=2,activation='relu',padding='same')
        self.bn3 = layers.BatchNormalization()
        self.conv4 = layers.Conv1D(256,4,strides=2,activation='relu',padding='same')
        self.bn4 = layers.BatchNormalization()
        self.gap = layers.GlobalAveragePooling1D()
        self.dense1 = layers.Dense(512,activation='relu'); self.drop1 = layers.Dropout(0.3)
        self.dense2 = layers.Dense(256,activation='relu'); self.drop2 = layers.Dropout(0.2)
        self.mean_head = layers.Dense(n_labels,name='label_mean')
        self.log_var_head = layers.Dense(n_labels,name='label_log_var')
    def call(self, x, training=False):
        h = self.reshape_in(x)
        h = self.bn1(self.conv1(h),training=training)
        h = self.bn2(self.conv2(h),training=training)
        h = self.bn3(self.conv3(h),training=training)
        h = self.bn4(self.conv4(h),training=training)
        h = self.gap(h)
        h = self.drop1(self.dense1(h),training=training)
        h = self.drop2(self.dense2(h),training=training)
        return self.mean_head(h), self.log_var_head(h)
    def get_config(self):
        c = super().get_config(); c['n_labels'] = self.n_labels; return c

@register_keras_serializable()
class TrainableCNNPredictor(HeteroscedasticCNNPredictor): pass

# ================================================================
# DATA LOADING
# ================================================================

def get_clean_imputed_data(h5_path, selected_labels, limit=None):
    with h5py.File(h5_path, 'r') as f:
        get_col = lambda k: f['metadata'][k]; keys = f['metadata'].dtype.names
        raw = np.stack([get_col(p) for p in selected_labels], axis=1)
        bad = np.zeros_like(raw, dtype=bool)
        for i, l in enumerate(selected_labels):
            fn = f"{l}_FLAG"
            if fn in keys:
                flg = get_col(fn)
                if flg.dtype.names: flg = flg[flg.dtype.names[0]]
                if flg.dtype.kind == 'V': flg = flg.view('<i4')
                bad[:, i] = (flg.astype(int) != 0)
            elif l in ['TEFF','LOGG','VMICRO','VMACRO','VSINI']:
                bad[:, i] = (raw[:, i] < -5000)
    if limit: raw = raw[:limit]; bad = bad[:limit]
    v = raw.copy(); v[bad] = np.nan
    clean = IterativeImputer(estimator=BayesianRidge(),max_iter=10,initial_strategy='median').fit_transform(v)
    fe = clean[:, config.FE_H_INDEX:config.FE_H_INDEX+1]
    for idx in config.ABUNDANCE_INDICES: clean[:, idx] += fe[:, 0]
    t = clean[:, selected_labels.index('TEFF')]
    clean = np.hstack([clean, (5040./(t+1e-6)).reshape(-1,1)])
    vm = clean[:, selected_labels.index('VMACRO')]; vs = clean[:, selected_labels.index('VSINI')]
    clean = np.hstack([clean, np.sqrt(vm**2+vs**2).reshape(-1,1)])
    c = clean[:, selected_labels.index('C_FE')]; o = clean[:, selected_labels.index('O_FE')]
    clean = np.hstack([clean, (c-o).reshape(-1,1)])
    g = clean[:, selected_labels.index('LOGG')]; m = clean[:, selected_labels.index('M_H')]
    clean = np.hstack([clean, (0.5*g+0.5*m).reshape(-1,1)])
    return clean

def get_err(h5_path, selected_labels, limit=None):
    with h5py.File(h5_path, 'r') as f:
        get_col = lambda k: f['metadata'][k]; keys = f['metadata'].dtype.names
        raw = np.stack([get_col(p) if p not in ['VMACRO_ERR','VMICRO_ERR','VSINI_ERR']
                        else np.zeros_like(get_col('TEFF_ERR')) for p in selected_labels], axis=1)
        bad = np.zeros_like(raw, dtype=bool)
        for i, l in enumerate(selected_labels):
            fn = f"{l}_FLAG"
            if fn in keys:
                flg = get_col(fn)
                if flg.dtype.names: flg = flg[flg.dtype.names[0]]
                if flg.dtype.kind == 'V': flg = flg.view('<i4')
                bad[:, i] = (flg.astype(int) != 0)
            elif l in ['TEFF','LOGG','VMICRO','VMACRO','VSINI']:
                bad[:, i] = (raw[:, i] < -5000)
        if limit: raw = raw[:limit]; bad = bad[:limit]
    v = raw.copy(); v[bad] = np.nan
    p95 = np.nanpercentile(v, 95, axis=0)
    return np.where(np.isnan(v), p95, v)

# ================================================================
# LOAD EVERYTHING
# ================================================================

print("Loading data...")
h5_path    = config.H5_PATH
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

model_raw = tf.keras.models.load_model(model_path)
cfg_str = json.dumps(model_raw.get_config()).replace('"float16"','"float32"').replace('"mixed_float16"','"float32"')
model_pure_fp32 = model_raw.__class__.from_config(json.loads(cfg_str))
model_pure_fp32.set_weights(model_raw.get_weights())
print("Forward model rebuilt in FP32.")

cnn_predictor = tf.keras.models.load_model(cnn_model_path)
with np.load(cnn_stats_path) as cs:
    CNN_LABEL_MEAN = cs['mean'].astype(np.float32)
    CNN_LABEL_STD  = cs['std'].astype(np.float32)

def cnn_predict_physical(flux_batch):
    fc = flux_batch.copy(); bad = ~np.isfinite(fc)|(fc<=1e-3); fc[bad] = 1.0
    mu_n, rlv = cnn_predictor(tf.constant(fc, tf.float32), training=False)
    mu_p = mu_n.numpy() * CNN_LABEL_STD + CNN_LABEL_MEAN
    std_p = np.sqrt(tf.nn.softplus(rlv).numpy() + 1e-4) * CNN_LABEL_STD
    std_p = np.maximum(std_p, 0.01*CNN_LABEL_STD)
    return mu_p.astype(np.float32), std_p.astype(np.float32)

# ================================================================
# CONSTANTS
# ================================================================

LABEL_NAMES  = config.SELECTED_LABELS
BATCH_SIZE_STARS = 10
TOTAL_STARS      = 5

CORE_INDICES  = [0,1,2,3,4,5,6,7,8]; N_CORE = 9
ABUND_INDICES = [9,10,11,12,13,14,15,16,17,18,19,20,21,22]; N_ABUND = 14

# NUTS hyperparameters (same as notebook)
NUM_CHAINS_CORE  = 8;  NUM_RESULTS_CORE = 500; NUM_BURNIN_CORE = 400; NUM_ADAPT_CORE = 320
NUM_CHAINS_ELEM  = 4;  NUM_RESULTS_ELEM = 300; NUM_BURNIN_ELEM = 200; NUM_ADAPT_ELEM = 160
MAX_TREE_DEPTH   = 6;  TARGET_ACCEPT    = 0.8
PRIOR_WEIGHT     = 1.0

RESULTS_DIR     = "/kaggle/working/nuts_results"
os.makedirs(RESULTS_DIR, exist_ok=True)
CHECKPOINT_PATH = os.path.join(RESULTS_DIR, "checkpoint.npz")
RESULTS_PATH    = os.path.join(RESULTS_DIR, "results.npz")

lower_bounds = [3000.,-0.5,-2.5,0.,0.,0.,-1.5,-1.,-1.,-2.5,-1.5,-1.,-1.,-1.5,-1.,-1.5,-1.5,-1.5,-4.,-2.5,-4.5,-2.5,-2.5]
upper_bounds = [20000.,5.,1.,3.,15.,80.,1.5,3.5,1.,1.,1.,1.5,1.,1.,1.,1.5,1.5,1.,1.5,2.,3.5,1.5,2.]
bounds_low  = tf.constant(lower_bounds, tf.float32)
bounds_high = tf.constant(upper_bounds, tf.float32)

_raw_mask = np.load(apogee_mask_path)
ELEMENT_PIXEL_MASKS = {}
for idx in ABUND_INDICES:
    s = (_raw_mask[:, idx] > 0.01).astype(np.float32)
    if s.sum() < 10: s = np.ones(N_PIXELS, np.float32)
    ELEMENT_PIXEL_MASKS[idx] = tf.constant(s, tf.float32)

# ================================================================
# FEATURE ENGINEERING
# ================================================================

def get_27_features(labels_23):
    teff = labels_23[:, config.SELECTED_LABELS.index('TEFF')]
    logg = labels_23[:, config.SELECTED_LABELS.index('LOGG')]
    vmacro = labels_23[:, config.SELECTED_LABELS.index('VMACRO')]
    vsini = labels_23[:, config.SELECTED_LABELS.index('VSINI')]
    c_fe = labels_23[:, config.SELECTED_LABELS.index('C_FE')]
    o_fe = labels_23[:, config.SELECTED_LABELS.index('O_FE')]
    m_h = labels_23[:, config.SELECTED_LABELS.index('M_H')]
    eng = tf.stack([5040./(teff+1e-6), tf.sqrt(tf.square(vmacro)+tf.square(vsini)),
                    c_fe-o_fe, 0.5*logg+0.5*m_h], axis=-1)
    return tf.concat([labels_23, eng], axis=-1)

# ================================================================
# TF.VARIABLE STATE
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
# LOG PROBABILITY — JIT COMPILED (the speed boost)
# The key difference: log_prob is XLA-compiled, but the NUTS loop is not.
# ================================================================

@tf.function(reduce_retracing=True)
def _assemble_full_23(core_params, fixed_abund):
    parts = []
    for i in range(N_LABELS_RAW):
        if i in CORE_INDICES:
            parts.append(core_params[:, CORE_INDICES.index(i):CORE_INDICES.index(i)+1])
        else:
            parts.append(fixed_abund[:, ABUND_INDICES.index(i):ABUND_INDICES.index(i)+1])
    return tf.concat(parts, axis=1)

# JIT-compiled log-prob for the expensive forward model evaluation
@tf.function(jit_compile=True)
def _forward_model_jit(labels_norm):
    """XLA-compiled forward model call."""
    model_flux = model_pure_fp32(labels_norm, training=False)
    return tf.cast(tf.squeeze(model_flux, axis=-1), tf.float32)

@tf.function(reduce_retracing=True)
def core_log_prob_fn(theta_core, obs_flux, obs_ivar):
    """theta_core: (C, B, 9) physical → (C, B). Log-prob uses JIT'd forward model."""
    num_chains  = tf.shape(theta_core)[0]
    batch_stars = tf.shape(theta_core)[1]

    theta_BCL  = tf.transpose(theta_core, [1, 0, 2])
    theta_flat = tf.reshape(theta_BCL, [-1, N_CORE])
    abund_rep  = tf.repeat(_fixed_abund_var, num_chains, axis=0)
    full_23    = _assemble_full_23(theta_flat, abund_rep)

    labels_27   = get_27_features(full_23)
    labels_norm = (labels_27 - MEAN_TENSOR) / STD_TENSOR

    # Use JIT-compiled forward model
    model_flux = _forward_model_jit(labels_norm)

    obs_flux_rep = tf.repeat(obs_flux, num_chains, axis=0)
    obs_ivar_rep = tf.repeat(obs_ivar, num_chains, axis=0)

    safe_flux = tf.where(tf.math.is_finite(obs_flux_rep), obs_flux_rep, 0.0)
    safe_ivar = tf.where(
        tf.math.is_finite(obs_ivar_rep) & tf.math.is_finite(obs_flux_rep),
        obs_ivar_rep, 0.0)
    safe_ivar = 1.0 / (1.0 / (safe_ivar + 1e-12) + 1e-5)
    fv = tf.cast((safe_flux > 1e-3) & (safe_flux < 1.3), tf.float32)
    iv = tf.cast(safe_ivar > 0.0, tf.float32)
    mask = fv * iv

    chi2 = tf.reduce_sum(tf.square(safe_flux - model_flux) * safe_ivar * mask, axis=1)
    log_lik = -0.5 * chi2

    cnn_mu_rep  = tf.repeat(tf.gather(_cnn_mu_var, CORE_INDICES, axis=1), num_chains, axis=0)
    cnn_std_rep = tf.repeat(tf.gather(_cnn_std_var, CORE_INDICES, axis=1), num_chains, axis=0)
    log_prior = -0.5 * tf.reduce_sum(tf.square((theta_flat - cnn_mu_rep) / cnn_std_rep), axis=1)

    log_post = log_lik + PRIOR_WEIGHT * log_prior
    return tf.transpose(tf.reshape(log_post, [batch_stars, num_chains]))


@tf.function(reduce_retracing=True)
def element_log_prob_fn(theta_1d, obs_flux, obs_ivar):
    """theta_1d: (C, B, 1) physical → (C, B). Log-prob uses JIT'd forward model."""
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
    model_flux  = _forward_model_jit(labels_norm)

    obs_flux_rep = tf.repeat(obs_flux, num_chains, axis=0)
    obs_ivar_rep = tf.repeat(obs_ivar, num_chains, axis=0)

    safe_flux = tf.where(tf.math.is_finite(obs_flux_rep), obs_flux_rep, 0.0)
    safe_ivar = tf.where(
        tf.math.is_finite(obs_ivar_rep) & tf.math.is_finite(obs_flux_rep),
        obs_ivar_rep, 0.0)
    safe_ivar = 1.0 / (1.0 / (safe_ivar + 1e-12) + 1e-5)
    fv = tf.cast((safe_flux > 1e-3) & (safe_flux < 1.3), tf.float32)
    iv = tf.cast(safe_ivar > 0.0, tf.float32)
    mask = fv * iv * tf.reshape(_elem_pixel_mask_var, [1, N_PIXELS])

    chi2 = tf.reduce_sum(tf.square(safe_flux - model_flux) * safe_ivar * mask, axis=1)
    log_lik = -0.5 * chi2

    cnn_mu_rep  = tf.repeat(_cnn_mu_var[:, elem_col:elem_col+1], num_chains, axis=0)
    cnn_std_rep = tf.repeat(_cnn_std_var[:, elem_col:elem_col+1], num_chains, axis=0)
    log_prior = -0.5 * tf.reduce_sum(tf.square((theta_flat - cnn_mu_rep) / cnn_std_rep), axis=1)

    log_post = log_lik + PRIOR_WEIGHT * log_prior
    return tf.transpose(tf.reshape(log_post, [batch_stars, num_chains]))

# ================================================================
# BIJECTORS + INIT HELPERS
# ================================================================

_core_low_tf  = tf.gather(bounds_low, CORE_INDICES)
_core_high_tf = tf.gather(bounds_high, CORE_INDICES)
_core_bijector = tfb.Chain([tfb.Shift(_core_low_tf), tfb.Scale(_core_high_tf - _core_low_tf), tfb.Sigmoid()])
_core_inv_bijector = tfb.Invert(_core_bijector)

def make_init_state_core(batch_stars, num_chains, physical_init_23):
    margin = 1e-3
    core_init = tf.gather(tf.cast(physical_init_23, tf.float32), CORE_INDICES, axis=1)
    base = tf.tile(tf.expand_dims(core_init, 0), [num_chains, 1, 1])
    core_std = tf.gather(STD_TENSOR[:N_LABELS_RAW], CORE_INDICES)
    jitter = tf.random.normal(tf.shape(base)) * (0.05 * core_std)
    return tf.clip_by_value(base + jitter,
                            tf.gather(bounds_low + margin, CORE_INDICES),
                            tf.gather(bounds_high - margin, CORE_INDICES))

def make_init_state_elem(batch_stars, num_chains, phys_value_1d):
    base = tf.reshape(tf.cast(phys_value_1d, tf.float32), [1, -1, 1])
    base = tf.tile(base, [num_chains, 1, 1])
    return base + tf.random.normal(tf.shape(base)) * 0.02

# ================================================================
# NUTS SAMPLING — NO jit_compile (dynamic tree-doubling is XLA-incompatible)
# Speed comes from JIT-compiled log-prob instead.
# ================================================================

_init_core_var = tf.Variable(
    tf.zeros([NUM_CHAINS_CORE, BATCH_SIZE_STARS, N_CORE], tf.float32), trainable=False)

@tf.function(reduce_retracing=True)  # NOTE: NO jit_compile — NUTS is eager-compatible only
def sample_core_compiled():
    """NUTS for Stage 1: 9D core physics. Log-prob is JIT-compiled internally."""
    def log_prob_closure(theta_unc):
        return core_log_prob_fn(
            _core_bijector.forward(theta_unc), _obs_flux_var, _obs_ivar_var)

    step_size = tf.fill([1, BATCH_SIZE_STARS, N_CORE], 0.01)
    nuts = tfp.mcmc.NoUTurnSampler(
        target_log_prob_fn=log_prob_closure,
        step_size=step_size,
        max_tree_depth=MAX_TREE_DEPTH,
    )
    adaptive = tfp.mcmc.DualAveragingStepSizeAdaptation(
        inner_kernel=nuts,
        num_adaptation_steps=NUM_ADAPT_CORE,
        target_accept_prob=TARGET_ACCEPT,
        step_size_setter_fn=lambda pkr, s: pkr._replace(step_size=s),
        step_size_getter_fn=lambda pkr: pkr.step_size,
        log_accept_prob_getter_fn=lambda pkr: pkr.log_accept_ratio,
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
    tf.zeros([NUM_CHAINS_ELEM, BATCH_SIZE_STARS, 1], tf.float32), trainable=False)

@tf.function(reduce_retracing=True)  # NO jit_compile
def sample_element_compiled():
    """NUTS for Stage 2: 1D abundance. Log-prob is JIT-compiled internally."""
    lo = tf.reshape(_elem_low_var, [1]); hi = tf.reshape(_elem_high_var, [1])
    elem_bij = tfb.Chain([tfb.Shift(lo), tfb.Scale(hi - lo), tfb.Sigmoid()])
    elem_inv = tfb.Invert(elem_bij)

    def log_prob_closure(theta_unc):
        return element_log_prob_fn(
            elem_bij.forward(theta_unc), _obs_flux_var, _obs_ivar_var)

    step_size = tf.fill([1, BATCH_SIZE_STARS, 1], 0.01)
    nuts = tfp.mcmc.NoUTurnSampler(
        target_log_prob_fn=log_prob_closure,
        step_size=step_size,
        max_tree_depth=MAX_TREE_DEPTH,
    )
    adaptive = tfp.mcmc.DualAveragingStepSizeAdaptation(
        inner_kernel=nuts,
        num_adaptation_steps=NUM_ADAPT_ELEM,
        target_accept_prob=TARGET_ACCEPT,
        step_size_setter_fn=lambda pkr, s: pkr._replace(step_size=s),
        step_size_getter_fn=lambda pkr: pkr.step_size,
        log_accept_prob_getter_fn=lambda pkr: pkr.log_accept_ratio,
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
# MCMC RUNNERS + PIPELINE (identical structure to HMC file)
# ================================================================

def run_core_mcmc(obs_flux, obs_ivar, physical_init_23, cnn_mean, cnn_std):
    _cnn_mu_var.assign(tf.cast(cnn_mean, tf.float32))
    _cnn_std_var.assign(tf.cast(cnn_std, tf.float32))
    _fixed_abund_var.assign(tf.gather(tf.cast(cnn_mean, tf.float32), ABUND_INDICES, axis=1))
    init_core = make_init_state_core(BATCH_SIZE_STARS, NUM_CHAINS_CORE, physical_init_23)
    _obs_flux_var.assign(tf.cast(obs_flux, tf.float32))
    _obs_ivar_var.assign(tf.cast(obs_ivar, tf.float32))
    _init_core_var.assign(init_core)

    samples_unc, is_accepted = sample_core_compiled()
    samples_phys = _core_bijector.forward(samples_unc)
    r_hat = tf.reduce_mean(
        tfp.mcmc.diagnostic.potential_scale_reduction(samples_phys, independent_chain_ndims=1), axis=0
    ).numpy()
    flat = tf.reshape(samples_phys, [NUM_RESULTS_CORE * NUM_CHAINS_CORE, -1, N_CORE])
    pcts = tfp.stats.percentile(flat, [16., 50., 84.], axis=0)
    return pcts[1].numpy(), (pcts[1]-pcts[0]).numpy(), (pcts[2]-pcts[1]).numpy(), \
           tf.reduce_mean(tf.cast(is_accepted, tf.float32)).numpy(), r_hat

def run_element_mcmc(obs_flux, obs_ivar, fixed_labels_23, elem_idx, cnn_mean, cnn_std):
    _fixed_full_var.assign(tf.cast(fixed_labels_23, tf.float32))
    _elem_col_var.assign(elem_idx)
    _elem_pixel_mask_var.assign(ELEMENT_PIXEL_MASKS[elem_idx])
    _elem_low_var.assign(bounds_low[elem_idx])
    _elem_high_var.assign(bounds_high[elem_idx])
    init_state = make_init_state_elem(BATCH_SIZE_STARS, NUM_CHAINS_ELEM, cnn_mean[:, elem_idx])
    _obs_flux_var.assign(tf.cast(obs_flux, tf.float32))
    _obs_ivar_var.assign(tf.cast(obs_ivar, tf.float32))
    _init_elem_var.assign(init_state)

    samples_unc, is_accepted = sample_element_compiled()
    lo_val = bounds_low[elem_idx]; hi_val = bounds_high[elem_idx]
    bij = tfb.Chain([tfb.Shift(tf.constant([lo_val])), tfb.Scale(tf.constant([hi_val - lo_val])), tfb.Sigmoid()])
    samples_phys = bij.forward(samples_unc)
    r_hat = tf.reduce_mean(
        tfp.mcmc.diagnostic.potential_scale_reduction(samples_phys, independent_chain_ndims=1)).numpy()
    flat = tf.reshape(samples_phys, [NUM_RESULTS_ELEM * NUM_CHAINS_ELEM, -1, 1])
    pcts = tfp.stats.percentile(flat, [16., 50., 84.], axis=0)
    return pcts[1,:,0].numpy(), (pcts[1,:,0]-pcts[0,:,0]).numpy(), (pcts[2,:,0]-pcts[1,:,0]).numpy(), \
           tf.reduce_mean(tf.cast(is_accepted, tf.float32)).numpy(), r_hat

def load_checkpoint():
    fresh = {'global_indices':[],'true_labels':[],'inferred_labels':[],
             'lower_sigma':[],'upper_sigma':[],'aspcap_errors':[],
             'acceptance_rates':[],'r_hat':[],'wall_seconds':[]}
    if os.path.exists(CHECKPOINT_PATH):
        data = np.load(CHECKPOINT_PATH, allow_pickle=True)
        r = {k: list(data[k]) for k in fresh}
        print(f"  Resuming: {len(r['global_indices'])} done.")
        return r, len(r['global_indices'])
    return fresh, 0

def save_checkpoint(results):
    np.savez(CHECKPOINT_PATH, **{k: np.array(v) for k, v in results.items()})

def save_final_results(results):
    a = {k: np.array(v) for k, v in results.items()}
    a['label_names'] = np.array(LABEL_NAMES)
    np.savez(RESULTS_PATH, **a)
    print(f"\nFinal results saved to {RESULTS_PATH}")

def run_inference_pipeline(test_indices, true_labels_norm, aspcap_errors,
                           flux_array, ivar_array):
    n_stars = len(test_indices)
    true_phys = true_labels_norm[:, :N_LABELS_RAW] * STD_TENSOR[:N_LABELS_RAW] + MEAN_TENSOR[:N_LABELS_RAW]
    results, n_done = load_checkpoint()
    remaining = list(range(n_done, n_stars))

    print(f"\n{'='*65}")
    print(f"  Tier 3: NUTS with JIT-compiled log-prob")
    print(f"  {n_stars} stars, batch {BATCH_SIZE_STARS}")
    print(f"  Stage 1 (core {N_CORE}D): {NUM_CHAINS_CORE}ch, NUTS depth {MAX_TREE_DEPTH}")
    print(f"  Stage 2 ({N_ABUND} elem × 1D): {NUM_CHAINS_ELEM}ch, NUTS depth {MAX_TREE_DEPTH}")
    print(f"  To do: {len(remaining)} stars")
    print(f"{'='*65}\n")

    total_start = time.time()
    batch_count = 0

    for bstart in range(0, len(remaining), BATCH_SIZE_STARS):
        bloc = remaining[bstart:bstart + BATCH_SIZE_STARS]
        actual = len(bloc)
        bf = flux_array[bloc]; bi = ivar_array[bloc]
        cnn_mu, cnn_sig = cnn_predict_physical(bf)

        if actual < BATCH_SIZE_STARS:
            pad = BATCH_SIZE_STARS - actual
            bf = np.concatenate([bf, np.zeros((pad, N_PIXELS), np.float32)])
            bi = np.concatenate([bi, np.zeros((pad, N_PIXELS), np.float32)])
            cnn_mu  = np.concatenate([cnn_mu, np.tile(cnn_mu[:1], (pad, 1))])
            cnn_sig = np.concatenate([cnn_sig, np.tile(cnn_sig[:1], (pad, 1))])

        t_s1 = time.time()
        core_med, core_lo, core_hi, s1_acc, s1_rhat = run_core_mcmc(bf, bi, cnn_mu, cnn_mu, cnn_sig)
        t_s1e = time.time() - t_s1
        print(f"    S1 core: acc={s1_acc:.2f}  R̂ max={s1_rhat.max():.3f}  {t_s1e:.1f}s")

        fixed_23 = cnn_mu.copy()
        for ci, gi in enumerate(CORE_INDICES): fixed_23[:, gi] = core_med[:, ci]

        t_s2 = time.time()
        amed = np.zeros((BATCH_SIZE_STARS, N_ABUND), np.float32)
        alo = np.zeros_like(amed); ahi = np.zeros_like(amed)
        s2r, s2a = [], []
        for ai, ei in enumerate(ABUND_INDICES):
            em, el, eh, ea, er = run_element_mcmc(bf, bi, fixed_23, ei, cnn_mu, cnn_sig)
            amed[:, ai] = em; alo[:, ai] = el; ahi[:, ai] = eh
            s2r.append(er); s2a.append(ea)
            fixed_23[:, ei] = em
        t_s2e = time.time() - t_s2
        print(f"    S2 elem: acc={np.mean(s2a):.2f}  R̂ max={max(s2r):.3f}  {t_s2e:.1f}s")

        full_med = np.zeros((BATCH_SIZE_STARS, N_LABELS_RAW), np.float32)
        full_lo = np.zeros_like(full_med); full_hi = np.zeros_like(full_med)
        full_rhat = np.zeros(N_LABELS_RAW, np.float32)
        for ci, gi in enumerate(CORE_INDICES):
            full_med[:, gi] = core_med[:, ci]; full_lo[:, gi] = core_lo[:, ci]
            full_hi[:, gi] = core_hi[:, ci]; full_rhat[gi] = s1_rhat[ci]
        for ai, gi in enumerate(ABUND_INDICES):
            full_med[:, gi] = amed[:, ai]; full_lo[:, gi] = alo[:, ai]
            full_hi[:, gi] = ahi[:, ai]; full_rhat[gi] = s2r[ai]

        elapsed = t_s1e + t_s2e
        avg_acc = (s1_acc + np.mean(s2a)) / 2
        for b in range(actual):
            li = bloc[b]; gi = int(test_indices[li])
            results['global_indices'].append(gi)
            results['true_labels'].append(true_phys[li])
            results['inferred_labels'].append(full_med[b])
            results['lower_sigma'].append(full_lo[b])
            results['upper_sigma'].append(full_hi[b])
            results['aspcap_errors'].append(aspcap_errors[li, :N_LABELS_RAW])
            results['acceptance_rates'].append(avg_acc)
            results['r_hat'].append(full_rhat)
            results['wall_seconds'].append(elapsed / actual)

        n_done += actual; batch_count += 1
        save_checkpoint(results)
        el = (n_stars - n_done) * elapsed / actual / 60
        print(f"  [{n_done:>4}/{n_stars}]  {elapsed:.1f}s  (~{el:.0f} min left)\n")

    print(f"\nAll {n_stars} stars done in {(time.time()-total_start)/60:.1f} min.")
    save_final_results(results)
    return results

# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":
    test_indices = np.load("test_indices.npy") if os.path.exists("test_indices.npy") else None
    if test_indices is None:
        print("Run backward_model_hmc.py first to generate test_indices.npy")
    else:
        run_inference_pipeline(test_indices, labels, label_err, real_data, real_data_ivar)
