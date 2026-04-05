"""
backward_model_map.py — Tier 1: Batched MAP Optimization (fastest)
==================================================================
Pure gradient-descent point estimates using Adam, fully XLA-compiled.
No posterior uncertainties — just the maximum a posteriori (MAP) estimate.

Two-stage FERRE-style:
  Stage 1 — 9D core physics via batched Adam on full spectrum
  Stage 2 — 14 × 1D individual abundances via batched Adam on masked pixels

CNN warm-start provides initialisation + Gaussian prior (regulariser).

~50–100× faster than NUTS. Run on Kaggle with GPU enabled.
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
# CONFIG (identical to backward_model_hmc.py)
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
# CUSTOM LAYERS (needed for model loading — identical to HMC file)
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
    def __init__(self, **kwargs): super().__init__(**kwargs)
    def call(self, k, tau): return k * tf.math.exp(-tau)

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
        return tf.scatter_nd(scatter_idx, tf.reshape(weighted, [-1]),
                             shape=tf.stack([batch_size, tf.constant(self.total_pixels, dtype=tf.int32)]))
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.total_pixels)
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
        cfg.pop('name', None)
        return cls(**cfg)

N_LABELS_RAW = 23
N_PIXELS     = 8575

@register_keras_serializable()
class HeteroscedasticCNNPredictor(tf.keras.Model):
    def __init__(self, n_labels=N_LABELS_RAW, **kwargs):
        super().__init__(**kwargs)
        self.n_labels = n_labels
        self.reshape_in = layers.Reshape((N_PIXELS, 1))
        self.conv1 = layers.Conv1D(32, 16, strides=4, activation='relu', padding='same')
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv1D(64, 8, strides=4, activation='relu', padding='same')
        self.bn2 = layers.BatchNormalization()
        self.conv3 = layers.Conv1D(128, 4, strides=2, activation='relu', padding='same')
        self.bn3 = layers.BatchNormalization()
        self.conv4 = layers.Conv1D(256, 4, strides=2, activation='relu', padding='same')
        self.bn4 = layers.BatchNormalization()
        self.gap = layers.GlobalAveragePooling1D()
        self.dense1 = layers.Dense(512, activation='relu')
        self.drop1 = layers.Dropout(0.3)
        self.dense2 = layers.Dense(256, activation='relu')
        self.drop2 = layers.Dropout(0.2)
        self.mean_head = layers.Dense(n_labels, name='label_mean')
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
# DATA LOADING (same as HMC file)
# ================================================================

def get_clean_imputed_data(h5_path, selected_labels, limit=None):
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
        raw_values = raw_values[:limit]; bad_mask = bad_mask[:limit]
    vals = raw_values.copy(); vals[bad_mask] = np.nan
    imputer = IterativeImputer(estimator=BayesianRidge(), max_iter=10, initial_strategy='median')
    clean = imputer.fit_transform(vals)
    fe_h = clean[:, config.FE_H_INDEX:config.FE_H_INDEX+1]
    for idx in config.ABUNDANCE_INDICES: clean[:, idx] += fe_h[:, 0]
    t = clean[:, selected_labels.index('TEFF')]
    clean = np.hstack([clean, (5040.0/(t+1e-6)).reshape(-1,1)])
    vm = clean[:, selected_labels.index('VMACRO')]
    vs = clean[:, selected_labels.index('VSINI')]
    clean = np.hstack([clean, np.sqrt(vm**2+vs**2).reshape(-1,1)])
    c = clean[:, selected_labels.index('C_FE')]
    o = clean[:, selected_labels.index('O_FE')]
    clean = np.hstack([clean, (c-o).reshape(-1,1)])
    g = clean[:, selected_labels.index('LOGG')]
    m = clean[:, selected_labels.index('M_H')]
    clean = np.hstack([clean, (0.5*g+0.5*m).reshape(-1,1)])
    return clean

def get_err(h5_path, selected_labels, limit=None):
    with h5py.File(h5_path, 'r') as f:
        get_col = lambda k: f['metadata'][k]
        keys = f['metadata'].dtype.names
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
# LOAD DATA + MODELS
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
cfg_m = model_raw.get_config()
cfg_str = json.dumps(cfg_m).replace('"float16"', '"float32"').replace('"mixed_float16"', '"float32"')
model_pure_fp32 = model_raw.__class__.from_config(json.loads(cfg_str))
model_pure_fp32.set_weights(model_raw.get_weights())
print("Forward model rebuilt in FP32.")

cnn_predictor = tf.keras.models.load_model(cnn_model_path)
with np.load(cnn_stats_path) as cs:
    CNN_LABEL_MEAN = cs['mean'].astype(np.float32)
    CNN_LABEL_STD  = cs['std'].astype(np.float32)

def cnn_predict_physical(flux_batch):
    fc = flux_batch.copy(); bad = ~np.isfinite(fc) | (fc <= 1e-3); fc[bad] = 1.0
    mu_n, rlv = cnn_predictor(tf.constant(fc, tf.float32), training=False)
    mu_p = mu_n.numpy() * CNN_LABEL_STD + CNN_LABEL_MEAN
    std_p = np.sqrt(tf.nn.softplus(rlv).numpy() + 1e-4) * CNN_LABEL_STD
    std_p = np.maximum(std_p, 0.01 * CNN_LABEL_STD)
    return mu_p.astype(np.float32), std_p.astype(np.float32)

# ================================================================
# CONSTANTS
# ================================================================

LABEL_NAMES  = config.SELECTED_LABELS
BATCH_SIZE_STARS = 32   # MAP is cheap — batch larger
TOTAL_STARS      = 5

CORE_INDICES  = [0,1,2,3,4,5,6,7,8]
N_CORE        = len(CORE_INDICES)
ABUND_INDICES = [9,10,11,12,13,14,15,16,17,18,19,20,21,22]
N_ABUND       = len(ABUND_INDICES)

# MAP hyper-params
N_STEPS_CORE = 300
N_STEPS_ELEM = 150
LEARNING_RATE_CORE = 0.01
LEARNING_RATE_ELEM = 0.01
PRIOR_WEIGHT = 1.0

RESULTS_DIR     = "/kaggle/working/map_results"
os.makedirs(RESULTS_DIR, exist_ok=True)
CHECKPOINT_PATH = os.path.join(RESULTS_DIR, "checkpoint.npz")
RESULTS_PATH    = os.path.join(RESULTS_DIR, "results.npz")

lower_bounds = [3000.,-0.5,-2.5,0.,0.,0.,-1.5,-1.,-1.,-2.5,-1.5,-1.,-1.,-1.5,-1.,-1.5,-1.5,-1.5,-4.,-2.5,-4.5,-2.5,-2.5]
upper_bounds = [20000.,5.,1.,3.,15.,80.,1.5,3.5,1.,1.,1.,1.5,1.,1.,1.,1.5,1.5,1.,1.5,2.,3.5,1.5,2.]
bounds_low  = tf.constant(lower_bounds, tf.float32)
bounds_high = tf.constant(upper_bounds, tf.float32)

_core_low  = tf.gather(bounds_low, CORE_INDICES)
_core_high = tf.gather(bounds_high, CORE_INDICES)
_core_bij  = tfb.Chain([tfb.Shift(_core_low), tfb.Scale(_core_high - _core_low), tfb.Sigmoid()])
_core_inv  = tfb.Invert(_core_bij)

_raw_mask = np.load(apogee_mask_path)
ELEMENT_PIXEL_MASKS = {}
for idx in ABUND_INDICES:
    s = (_raw_mask[:, idx] > 0.01).astype(np.float32)
    if s.sum() < 10: s = np.ones(N_PIXELS, np.float32)
    ELEMENT_PIXEL_MASKS[idx] = tf.constant(s, tf.float32)

@tf.function(jit_compile=True)
def compute_uncertainties_core(theta_unc_batch, obs_flux, obs_ivar, fixed_abund, cnn_mu, cnn_std):
    """
    Laplace approximation for 9D core physics.
    """
    n_params = 9
    core_cnn_mu = tf.gather(cnn_mu, CORE_INDICES, axis=1)
    core_cnn_std = tf.gather(cnn_std, CORE_INDICES, axis=1)

    def single_star_hessian(args):
        theta_unc, flux, ivar, f_abund, c_mu, c_std = args
        with tf.GradientTape() as outer_tape:
            outer_tape.watch(theta_unc)
            with tf.GradientTape() as inner_tape:
                inner_tape.watch(theta_unc)
                theta_phys = _core_bij.forward(theta_unc)
                
                parts = []
                for i in range(23):
                    if i in CORE_INDICES:
                        ci = CORE_INDICES.index(i)
                        parts.append(tf.reshape(theta_phys[ci], [1]))
                    else:
                        ai = ABUND_INDICES.index(i)
                        parts.append(tf.reshape(f_abund[ai], [1]))
                full_23 = tf.concat(parts, axis=0)
                
                labels_27 = get_27_features(tf.expand_dims(full_23, 0))
                labels_norm = (labels_27 - MEAN_TENSOR) / STD_TENSOR
                model_flux = tf.cast(model_pure_fp32(labels_norm, training=False), tf.float32)
                model_flux = tf.squeeze(model_flux)

                safe_flux = tf.where(tf.math.is_finite(flux), flux, 0.0)
                safe_ivar = tf.where(tf.math.is_finite(ivar) & tf.math.is_finite(flux), ivar, 0.0)
                safe_ivar = 1.0 / (1.0 / (safe_ivar + 1e-12) + 1e-5)
                mask = tf.cast((safe_flux > 1e-3) & (safe_flux < 1.3) & (safe_ivar > 0.0), tf.float32)

                chi2 = tf.reduce_sum(tf.square(safe_flux - model_flux) * safe_ivar * mask)
                prior = tf.reduce_sum(tf.square((theta_phys - c_mu) / c_std))
                loss = 0.5 * chi2 + PRIOR_WEIGHT * 0.5 * prior
            grad = inner_tape.gradient(loss, theta_unc)
        hess = outer_tape.jacobian(grad, theta_unc)
        hess_reg = hess + tf.eye(n_params) * 1e-6
        cov_unc = tf.linalg.inv(hess_reg)
        var_unc = tf.abs(tf.linalg.diag_part(cov_unc))
        
        s = tf.nn.sigmoid(theta_unc)
        jacobian_diag = s * (1.0 - s) * (_core_high - _core_low)
        return tf.sqrt(var_unc) * jacobian_diag

    return tf.vectorized_map(single_star_hessian, (theta_unc_batch, obs_flux, obs_ivar, fixed_abund, core_cnn_mu, core_cnn_std))

@tf.function(jit_compile=True)
def compute_uncertainties_element(theta_unc_1d, obs_flux, obs_ivar, fixed_full_23, 
                                 elem_col, pixel_mask, elem_lo, elem_hi, 
                                 elem_cnn_mu, elem_cnn_std):
    """
    Laplace approximation for 1D single element.
    """
    lo = tf.reshape(elem_lo, [1]); hi = tf.reshape(elem_hi, [1])
    bij = tfb.Chain([tfb.Shift(lo), tfb.Scale(hi - lo), tfb.Sigmoid()])

    def single_star_hessian(args):
        theta_unc, flux, ivar, f_23, c_mu, c_std = args
        with tf.GradientTape() as tape:
            tape.watch(theta_unc)
            with tf.GradientTape() as inner_tape:
                inner_tape.watch(theta_unc)
                theta_phys = bij.forward(theta_unc)
                indices = tf.constant([[0, elem_col]])
                full_spliced = tf.tensor_scatter_nd_update(tf.expand_dims(f_23, 0), indices, theta_phys)
                labels_27 = get_27_features(full_spliced)
                labels_norm = (labels_27 - MEAN_TENSOR) / STD_TENSOR
                model_flux = tf.cast(model_pure_fp32(labels_norm, training=False), tf.float32)
                model_flux = tf.squeeze(model_flux)
                safe_flux = tf.where(tf.math.is_finite(flux), flux, 0.0)
                safe_ivar = tf.where(tf.math.is_finite(ivar) & tf.math.is_finite(flux), ivar, 0.0)
                safe_ivar = 1.0 / (1.0 / (safe_ivar + 1e-12) + 1e-5)
                mask = tf.cast((safe_flux > 1e-3) & (safe_flux < 1.3) & (safe_ivar > 0.0), tf.float32) * pixel_mask
                chi2 = tf.reduce_sum(tf.square(safe_flux - model_flux) * safe_ivar * mask)
                prior = tf.reduce_sum(tf.square((theta_phys - c_mu) / c_std))
                loss = 0.5 * chi2 + PRIOR_WEIGHT * 0.5 * prior
            grad = inner_tape.gradient(loss, theta_unc)
        hess = tape.gradient(grad, theta_unc)
        var_unc = 1.0 / tf.abs(hess + 1e-6)
        s = tf.nn.sigmoid(theta_unc)
        jacobian = s * (1.0 - s) * (hi - lo)
        return tf.sqrt(var_unc[0]) * jacobian[0]

    return tf.vectorized_map(single_star_hessian, (theta_unc_1d, obs_flux, obs_ivar, fixed_full_23, elem_cnn_mu, elem_cnn_std))

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
    eng = tf.stack([
        5040.0 / (teff + 1e-6),
        tf.sqrt(tf.square(vmacro) + tf.square(vsini)),
        c_fe - o_fe,
        0.5 * logg + 0.5 * m_h,
    ], axis=-1)
    return tf.concat([labels_23, eng], axis=-1)

# ================================================================
# MAP OPTIMISATION — FULLY XLA COMPILED
# ================================================================

@tf.function(jit_compile=True)
def map_optimize_core(theta_unc, obs_flux, obs_ivar,
                       fixed_abund, cnn_mu, cnn_std):
    """
    Batched Adam MAP for 9D core physics.
    theta_unc: (B, 9) unconstrained
    Returns optimised (B, 9) in physical space.

    Uses a functional Adam update (no tf.Variable / optimizer inside XLA).
    """
    core_cnn_mu  = tf.gather(cnn_mu, CORE_INDICES, axis=1)
    core_cnn_std = tf.gather(cnn_std, CORE_INDICES, axis=1)

    lr = tf.constant(LEARNING_RATE_CORE, tf.float32)
    beta1 = tf.constant(0.9, tf.float32)
    beta2 = tf.constant(0.999, tf.float32)
    eps   = tf.constant(1e-7, tf.float32)

    m = tf.zeros_like(theta_unc)
    v = tf.zeros_like(theta_unc)
    theta = theta_unc

    for step in tf.range(N_STEPS_CORE):
        with tf.GradientTape() as tape:
            tape.watch(theta)
            theta_phys = _core_bij.forward(theta)

            parts = []
            for i in range(N_LABELS_RAW):
                if i in CORE_INDICES:
                    ci = CORE_INDICES.index(i)
                    parts.append(theta_phys[:, ci:ci+1])
                else:
                    ai = ABUND_INDICES.index(i)
                    parts.append(fixed_abund[:, ai:ai+1])
            full_23 = tf.concat(parts, axis=1)

            labels_27 = get_27_features(full_23)
            labels_norm = (labels_27 - MEAN_TENSOR) / STD_TENSOR
            model_flux = tf.cast(model_pure_fp32(labels_norm, training=False), tf.float32)
            model_flux = tf.squeeze(model_flux, axis=-1)

            safe_flux = tf.where(tf.math.is_finite(obs_flux), obs_flux, 0.0)
            safe_ivar = tf.where(
                tf.math.is_finite(obs_ivar) & tf.math.is_finite(obs_flux),
                obs_ivar, 0.0)
            safe_ivar = 1.0 / (1.0 / (safe_ivar + 1e-12) + 1e-5)
            fv = tf.cast((safe_flux > 1e-3) & (safe_flux < 1.3), tf.float32)
            iv = tf.cast(safe_ivar > 0.0, tf.float32)
            mask = fv * iv

            chi2 = tf.reduce_sum(tf.square(safe_flux - model_flux) * safe_ivar * mask, axis=1)
            prior = tf.reduce_sum(
                tf.square((theta_phys - core_cnn_mu) / core_cnn_std), axis=1
            )
            loss = tf.reduce_mean(0.5 * chi2 + PRIOR_WEIGHT * 0.5 * prior)

        grad = tape.gradient(loss, theta)
        t_f = tf.cast(step + 1, tf.float32)
        m = beta1 * m + (1.0 - beta1) * grad
        v = beta2 * v + (1.0 - beta2) * tf.square(grad)
        m_hat = m / (1.0 - tf.pow(beta1, t_f))
        v_hat = v / (1.0 - tf.pow(beta2, t_f))
        theta = theta - lr * m_hat / (tf.sqrt(v_hat) + eps)

    return _core_bij.forward(theta)


@tf.function(jit_compile=True)
def map_optimize_element(theta_unc_1d, obs_flux, obs_ivar,
                          fixed_full_23, elem_col, pixel_mask,
                          elem_lo, elem_hi, elem_cnn_mu, elem_cnn_std):
    """
    Batched Adam MAP for a single 1D abundance.
    theta_unc_1d: (B, 1) unconstrained
    Returns optimised (B,) in physical space.

    Uses a functional Adam update (no tf.Variable / optimizer inside XLA).
    """
    lo = tf.reshape(elem_lo, [1])
    hi = tf.reshape(elem_hi, [1])
    bij = tfb.Chain([tfb.Shift(lo), tfb.Scale(hi - lo), tfb.Sigmoid()])

    lr = tf.constant(LEARNING_RATE_ELEM, tf.float32)
    beta1 = tf.constant(0.9, tf.float32)
    beta2 = tf.constant(0.999, tf.float32)
    eps   = tf.constant(1e-7, tf.float32)

    m = tf.zeros_like(theta_unc_1d)
    v = tf.zeros_like(theta_unc_1d)
    theta = theta_unc_1d

    for step in tf.range(N_STEPS_ELEM):
        with tf.GradientTape() as tape:
            tape.watch(theta)
            theta_phys = bij.forward(theta)  # (B, 1)

            bc = tf.shape(fixed_full_23)[0]
            indices = tf.stack([tf.range(bc), tf.fill([bc], elem_col)], axis=1)
            full_spliced = tf.tensor_scatter_nd_update(
                fixed_full_23, indices, theta_phys[:, 0]
            )

            labels_27 = get_27_features(full_spliced)
            labels_norm = (labels_27 - MEAN_TENSOR) / STD_TENSOR
            model_flux = tf.cast(model_pure_fp32(labels_norm, training=False), tf.float32)
            model_flux = tf.squeeze(model_flux, axis=-1)

            safe_flux = tf.where(tf.math.is_finite(obs_flux), obs_flux, 0.0)
            safe_ivar = tf.where(
                tf.math.is_finite(obs_ivar) & tf.math.is_finite(obs_flux),
                obs_ivar, 0.0)
            safe_ivar = 1.0 / (1.0 / (safe_ivar + 1e-12) + 1e-5)
            fv = tf.cast((safe_flux > 1e-3) & (safe_flux < 1.3), tf.float32)
            iv = tf.cast(safe_ivar > 0.0, tf.float32)
            mask = fv * iv * tf.reshape(pixel_mask, [1, N_PIXELS])

            chi2 = tf.reduce_sum(tf.square(safe_flux - model_flux) * safe_ivar * mask, axis=1)
            prior = tf.reduce_sum(
                tf.square((theta_phys - elem_cnn_mu) / elem_cnn_std), axis=1
            )
            loss = tf.reduce_mean(0.5 * chi2 + PRIOR_WEIGHT * 0.5 * prior)

        grad = tape.gradient(loss, theta)
        t_f = tf.cast(step + 1, tf.float32)
        m = beta1 * m + (1.0 - beta1) * grad
        v = beta2 * v + (1.0 - beta2) * tf.square(grad)
        m_hat = m / (1.0 - tf.pow(beta1, t_f))
        v_hat = v / (1.0 - tf.pow(beta2, t_f))
        theta = theta - lr * m_hat / (tf.sqrt(v_hat) + eps)

    return bij.forward(theta)[:, 0]

@tf.function(jit_compile=True)
def compute_uncertainties_joint(theta_unc_batch, obs_flux, obs_ivar, cnn_mu, cnn_std):
    """
    Laplace approximation for 23D joint MAP.
    """
    batch_size = tf.shape(theta_unc_batch)[0]
    n_params = 23
    
    def single_star_hessian(args):
        theta_unc, flux, ivar, c_mu, c_std = args
        with tf.GradientTape() as outer_tape:
            outer_tape.watch(theta_unc)
            with tf.GradientTape() as inner_tape:
                inner_tape.watch(theta_unc)
                theta_phys = _joint_bij.forward(theta_unc)
                labels_27 = get_27_features(tf.expand_dims(theta_phys, 0))
                labels_norm = (labels_27 - MEAN_TENSOR) / STD_TENSOR
                model_flux = tf.cast(model_pure_fp32(labels_norm, training=False), tf.float32)
                model_flux = tf.squeeze(model_flux)
                safe_flux = tf.where(tf.math.is_finite(flux), flux, 0.0)
                safe_ivar = tf.where(tf.math.is_finite(ivar) & tf.math.is_finite(flux), ivar, 0.0)
                safe_ivar = 1.0 / (1.0 / (safe_ivar + 1e-12) + 1e-5)
                mask = tf.cast((safe_flux > 1e-3) & (safe_flux < 1.3) & (safe_ivar > 0.0), tf.float32)
                chi2 = tf.reduce_sum(tf.square(safe_flux - model_flux) * safe_ivar * mask)
                prior = tf.reduce_sum(tf.square((theta_phys - c_mu) / c_std))
                loss = 0.5 * chi2 + PRIOR_WEIGHT * 0.5 * prior
            grad = inner_tape.gradient(loss, theta_unc)
        hess = outer_tape.jacobian(grad, theta_unc)
        hess_reg = hess + tf.eye(n_params) * 1e-6
        cov_unc = tf.linalg.inv(hess_reg)
        var_unc = tf.abs(tf.linalg.diag_part(cov_unc))
        s = tf.nn.sigmoid(theta_unc)
        jacobian_diag = s * (1.0 - s) * (bounds_high - bounds_low)
        return tf.sqrt(var_unc) * jacobian_diag

    return tf.vectorized_map(single_star_hessian, (theta_unc_batch, obs_flux, obs_ivar, cnn_mu, cnn_std))

# ================================================================
# CHECKPOINT / RESULTS
# ================================================================

def load_checkpoint():
    fresh = {
        'global_indices': [], 'true_labels': [], 'inferred_labels': [],
        'inferred_errors': [], 'aspcap_errors': [], 'wall_seconds': [],
    }
    if os.path.exists(CHECKPOINT_PATH):
        data = np.load(CHECKPOINT_PATH, allow_pickle=True)
        results = {k: list(data[k]) if k in data else [] for k in fresh}
        n_done = len(results['global_indices'])
        print(f"  Resuming: {n_done} stars done.")
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
    remaining = list(range(n_done, n_stars))

    print(f"\n{'='*60}")
    print(f"  Tier 1: MAP Optimization (XLA compiled)")
    print(f"  {n_stars} stars, batch {BATCH_SIZE_STARS}")
    print(f"  Core: {N_STEPS_CORE} Adam steps, lr={LEARNING_RATE_CORE}")
    print(f"  Elements: {N_STEPS_ELEM} Adam steps each, lr={LEARNING_RATE_ELEM}")
    print(f"  To do: {len(remaining)} stars")
    print(f"{'='*60}\n")

    total_start = time.time()
    batch_count = 0

    for batch_start in range(0, len(remaining), BATCH_SIZE_STARS):
        batch_local = remaining[batch_start:batch_start + BATCH_SIZE_STARS]
        actual = len(batch_local)

        b_flux = flux_array[batch_local].astype(np.float32)
        b_ivar = ivar_array[batch_local].astype(np.float32)
        cnn_mu, cnn_sig = cnn_predict_physical(b_flux)

        # Pad
        if actual < BATCH_SIZE_STARS:
            pad = BATCH_SIZE_STARS - actual
            b_flux = np.concatenate([b_flux, np.zeros((pad, N_PIXELS), np.float32)])
            b_ivar = np.concatenate([b_ivar, np.zeros((pad, N_PIXELS), np.float32)])
            cnn_mu  = np.concatenate([cnn_mu, np.tile(cnn_mu[:1], (pad, 1))])
            cnn_sig = np.concatenate([cnn_sig, np.tile(cnn_sig[:1], (pad, 1))])

        obs_flux_tf = tf.constant(b_flux)
        obs_ivar_tf = tf.constant(b_ivar)
        cnn_mu_tf   = tf.constant(cnn_mu)
        cnn_sig_tf  = tf.constant(cnn_sig)

        # Stage 1: Core MAP
        t0 = time.time()
        core_init_phys = tf.gather(cnn_mu_tf, CORE_INDICES, axis=1)
        margin = 1e-3
        core_init_phys = tf.clip_by_value(
            core_init_phys,
            tf.gather(bounds_low + margin, CORE_INDICES),
            tf.gather(bounds_high - margin, CORE_INDICES),
        )
        core_init_unc = _core_inv.forward(core_init_phys)

        fixed_abund = tf.gather(cnn_mu_tf, ABUND_INDICES, axis=1)

        core_result = map_optimize_core(
            core_init_unc, obs_flux_tf, obs_ivar_tf,
            fixed_abund, cnn_mu_tf, cnn_sig_tf
        )
        
        # Core Uncertainties (Laplace)
        core_final_unc = _core_inv.forward(core_result)
        core_errors = compute_uncertainties_core(
            core_final_unc, obs_flux_tf, obs_ivar_tf,
            fixed_abund, cnn_mu_tf, cnn_sig_tf
        )
        
        t_core = time.time() - t0
        print(f"    S1 core MAP + Laplace: {t_core:.2f}s")

        # Build fixed_23 with core results
        fixed_23 = cnn_mu.copy()
        core_np = core_result.numpy()
        core_err_np = core_errors.numpy()
        for ci, gi in enumerate(CORE_INDICES):
            fixed_23[:, gi] = core_np[:, ci]

        # Stage 2: Element MAP
        t0 = time.time()
        elem_results = np.zeros((BATCH_SIZE_STARS, N_LABELS_RAW), np.float32)
        elem_errors  = np.zeros((BATCH_SIZE_STARS, N_LABELS_RAW), np.float32)
        for ci, gi in enumerate(CORE_INDICES):
            elem_results[:, gi] = core_np[:, ci]
            elem_errors[:, gi]  = core_err_np[:, ci]

        for ai, elem_idx in enumerate(ABUND_INDICES):
            elem_init_phys = cnn_mu[:, elem_idx:elem_idx+1]
            lo_val = lower_bounds[elem_idx]
            hi_val = upper_bounds[elem_idx]
            elem_init_phys = np.clip(elem_init_phys, lo_val + margin, hi_val - margin)

            lo_tf = tf.constant(lo_val, tf.float32); hi_tf = tf.constant(hi_val, tf.float32)
            inv_bij = tfb.Invert(tfb.Chain([
                tfb.Shift(tf.reshape(lo_tf, [1])),
                tfb.Scale(tf.reshape(hi_tf - lo_tf, [1])),
                tfb.Sigmoid(),
            ]))
            elem_init_unc = inv_bij.forward(tf.constant(elem_init_phys, tf.float32))

            e_result = map_optimize_element(
                elem_init_unc, obs_flux_tf, obs_ivar_tf,
                tf.constant(fixed_23, tf.float32), elem_idx,
                ELEMENT_PIXEL_MASKS[elem_idx],
                lo_tf, hi_tf,
                cnn_mu_tf[:, elem_idx:elem_idx+1],
                cnn_sig_tf[:, elem_idx:elem_idx+1],
            )
            
            # Element Uncertainties (Laplace)
            e_final_unc = inv_bij.forward(e_result)
            e_err = compute_uncertainties_element(
                e_final_unc, obs_flux_tf, obs_ivar_tf,
                tf.constant(fixed_23, tf.float32), elem_idx,
                ELEMENT_PIXEL_MASKS[elem_idx],
                lo_tf, hi_tf,
                cnn_mu_tf[:, elem_idx:elem_idx+1],
                cnn_sig_tf[:, elem_idx:elem_idx+1],
            )
            
            elem_results[:, elem_idx] = e_result.numpy()
            elem_errors[:, elem_idx]  = e_err.numpy()
            fixed_23[:, elem_idx] = e_result.numpy()

        t_elem = time.time() - t0
        print(f"    S2 elem MAP + Laplace: {t_elem:.2f}s ({N_ABUND} elements)")

        elapsed = t_core + t_elem

        for b in range(actual):
            local_idx = batch_local[b]
            global_idx = int(test_indices[local_idx])
            results['global_indices'].append(global_idx)
            results['true_labels'].append(true_labels_physical[local_idx])
            results['inferred_labels'].append(elem_results[b])
            results['inferred_errors'].append(elem_errors[b])
            results['aspcap_errors'].append(aspcap_errors[local_idx, :N_LABELS_RAW])
            results['wall_seconds'].append(elapsed / actual)

        n_done += actual
        batch_count += 1
        save_checkpoint(results)

        est_left = (n_stars - n_done) * elapsed / actual / 60
        print(f"  [{n_done:>4}/{n_stars}]  {elapsed:.1f}s  (~{est_left:.0f} min left)\n")

    total_elapsed = time.time() - total_start
    print(f"\nAll {n_stars} stars done in {total_elapsed:.1f}s ({total_elapsed/60:.1f} min).")
    save_final_results(results)
    return results

@tf.function(jit_compile=True)
def map_optimize_joint(theta_unc, obs_flux, obs_ivar, cnn_mu, cnn_std):
    """
    Batched Adam MAP for all 23 physics/abundance parameters simultaneously.
    """
    lr = tf.constant(LEARNING_RATE_CORE, tf.float32)
    beta1 = tf.constant(0.9, tf.float32)
    beta2 = tf.constant(0.999, tf.float32)
    eps   = tf.constant(1e-7, tf.float32)
    
    m = tf.zeros_like(theta_unc)
    v = tf.zeros_like(theta_unc)
    theta = theta_unc

    for step in tf.range(N_STEPS_CORE):
        with tf.GradientTape() as tape:
            tape.watch(theta)
            theta_phys = _joint_bij.forward(theta)
            labels_27 = get_27_features(theta_phys)
            labels_norm = (labels_27 - MEAN_TENSOR) / STD_TENSOR
            model_flux = tf.cast(model_pure_fp32(labels_norm, training=False), tf.float32)
            model_flux = tf.squeeze(model_flux, axis=-1)

            safe_flux = tf.where(tf.math.is_finite(obs_flux), obs_flux, 0.0)
            safe_ivar = tf.where(tf.math.is_finite(obs_ivar) & tf.math.is_finite(obs_flux), obs_ivar, 0.0)
            safe_ivar = 1.0 / (1.0 / (safe_ivar + 1e-12) + 1e-5)
            mask = tf.cast((safe_flux > 1e-3) & (safe_flux < 1.3) & (safe_ivar > 0.0), tf.float32)

            chi2 = tf.reduce_sum(tf.square(safe_flux - model_flux) * safe_ivar * mask, axis=1)
            prior = tf.reduce_sum(tf.square((theta_phys - cnn_mu) / cnn_std), axis=1)
            loss = tf.reduce_mean(0.5 * chi2 + PRIOR_WEIGHT * 0.5 * prior)

        grad = tape.gradient(loss, theta)
        t_f = tf.cast(step + 1, tf.float32)
        m = beta1 * m + (1.0 - beta1) * grad
        v = beta2 * v + (1.0 - beta2) * tf.square(grad)
        m_hat = m / (1.0 - tf.pow(beta1, t_f))
        v_hat = v / (1.0 - tf.pow(beta2, t_f))
        theta = theta - lr * m_hat / (tf.sqrt(v_hat) + eps)

    return _joint_bij.forward(theta)

def run_inference_pipeline_alt(test_indices, true_labels_norm, aspcap_errors,
                               flux_array, ivar_array):
    CHECKPOINT_ALT_PATH = os.path.join(RESULTS_DIR, "checkpoint_alt.npz")
    RESULTS_ALT_PATH    = os.path.join(RESULTS_DIR, "results_alt.npz")
    
    def load_alt_checkpoint():
        fresh = {
            'global_indices': [], 'true_labels': [], 'inferred_labels': [],
            'inferred_errors': [], 'aspcap_errors': [], 'wall_seconds': [],
        }
        if os.path.exists(CHECKPOINT_ALT_PATH):
            data = np.load(CHECKPOINT_ALT_PATH, allow_pickle=True)
            results = {k: list(data[k]) if k in data else [] for k in fresh}
            n_done = len(results['global_indices'])
            print(f"  Resuming (Joint MAP): {n_done} stars done.")
            return results, n_done
        return fresh, 0

    n_stars = len(test_indices)
    true_labels_physical = (
        true_labels_norm[:, :N_LABELS_RAW] * STD_TENSOR[:N_LABELS_RAW]
        + MEAN_TENSOR[:N_LABELS_RAW]
    )
    results, n_done = load_alt_checkpoint()
    remaining = list(range(n_done, n_stars))

    print(f"\n{'='*60}")
    print(f"  Tier 1-Alt: JOINT MAP Optimization (XLA compiled)")
    print(f"  To do: {len(remaining)} stars")
    print(f"{'='*60}\n")

    total_start = time.time()
    for batch_start in range(0, len(remaining), BATCH_SIZE_STARS):
        batch_local = remaining[batch_start:batch_start + BATCH_SIZE_STARS]
        actual = len(batch_local)
        b_flux = flux_array[batch_local].astype(np.float32)
        b_ivar = ivar_array[batch_local].astype(np.float32)
        cnn_mu, cnn_sig = cnn_predict_physical(b_flux)

        if actual < BATCH_SIZE_STARS:
            pad = BATCH_SIZE_STARS - actual
            b_flux = np.concatenate([b_flux, np.zeros((pad, N_PIXELS), np.float32)])
            b_ivar = np.concatenate([b_ivar, np.zeros((pad, N_PIXELS), np.float32)])
            cnn_mu  = np.concatenate([cnn_mu, np.tile(cnn_mu[:1], (pad, 1))])
            cnn_sig = np.concatenate([cnn_sig, np.tile(cnn_sig[:1], (pad, 1))])

        obs_flux_tf = tf.constant(b_flux); obs_ivar_tf = tf.constant(b_ivar)
        cnn_mu_tf = tf.constant(cnn_mu); cnn_sig_tf = tf.constant(cnn_sig)

        t0 = time.time()
        margin = 1e-3
        joint_init_phys = tf.clip_by_value(cnn_mu_tf, bounds_low + margin, bounds_high - margin)
        joint_init_unc = _joint_inv.forward(joint_init_phys)

        joint_result = map_optimize_joint(joint_init_unc, obs_flux_tf, obs_ivar_tf, cnn_mu_tf, cnn_sig_tf)
        
        # Laplace
        joint_f_unc = _joint_inv.forward(joint_result)
        joint_errors = compute_uncertainties_joint(joint_f_unc, obs_flux_tf, obs_ivar_tf, cnn_mu_tf, cnn_sig_tf)
        
        joint_result_np = joint_result.numpy(); joint_errors_np = joint_errors.numpy()
        elapsed = time.time() - t0
        print(f"    Joint MAP + Laplace: {elapsed:.2f}s")

        for b in range(actual):
            local_idx = batch_local[b]
            results['global_indices'].append(int(test_indices[local_idx]))
            results['true_labels'].append(true_labels_physical[local_idx])
            results['inferred_labels'].append(joint_result_np[b])
            results['inferred_errors'].append(joint_errors_np[b])
            results['aspcap_errors'].append(aspcap_errors[local_idx, :N_LABELS_RAW])
            results['wall_seconds'].append(elapsed / actual)

        n_done += actual
        np.savez(CHECKPOINT_ALT_PATH, **{k: np.array(v) for k, v in results.items()})

    total_elapsed = time.time() - total_start
    print(f"\nAll {n_stars} stars done in {total_elapsed:.1f}s.")
    
    arrays = {k: np.array(v) for k, v in results.items()}
    arrays['label_names'] = np.array(LABEL_NAMES)
    np.savez(RESULTS_ALT_PATH, **arrays)
    return results

# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":
    test_indices = np.load("test_indices.npy") if os.path.exists("test_indices.npy") else None
    if test_indices is None:
        print("Run backward_model_hmc.py first to generate test_indices.npy")
    else:
        results = run_inference_pipeline(
            test_indices, labels, label_err, real_data, real_data_ivar,
        )
