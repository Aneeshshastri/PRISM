"""
cnn-label-predictor.py
======================
Trains a small 1D CNN to predict 23 stellar labels (+ per-label uncertainty)
directly from observed 8575-pixel APOGEE spectra.

Run on Kaggle with GPU enabled.  The saved model is loaded by
backward-model-corrected.py to warm-start NUTS HMC.

Data split (same as the forward emulator):
    Train : stars  0 – 119,999   (120k)
    Val   : stars  120,000 – 139,999  (20k)
    Test  : stars  140,000 – 149,500  (held out, never touched here)

Output:
    /kaggle/working/cnn_label_predictor.keras
"""

# ╔══════════════════════════════════════════════════════════════╗
# ║ 1 — Imports & Config                                        ║
# ╚══════════════════════════════════════════════════════════════╝

import os
import numpy as np
import h5py
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, Input
from tensorflow.keras.saving import register_keras_serializable
from sklearn.experimental import enable_iterative_imputer          # noqa
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge

# ── reproducibility ──
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# ── paths (Kaggle) ──
H5_PATH    = "/kaggle/input/datasets/aneeshshastri/aspcapstar-dr17-150kstars/apogee_dr17_parallel.h5"
SAVE_PATH  = "/kaggle/working/cnn_label_predictor.keras"

# ── data split ──
TRAIN_LIMIT = 120_000
VAL_LIMIT   = 20_000     # stars 120k–140k

# ── label ordering ──  (must match backward-model-corrected.py exactly)
SELECTED_LABELS = [
    'TEFF', 'LOGG', 'M_H', 'VMICRO', 'VMACRO', 'VSINI',
    'C_FE', 'N_FE', 'O_FE',
    'FE_H',
    'MG_FE', 'SI_FE', 'CA_FE', 'TI_FE', 'S_FE',
    'AL_FE', 'MN_FE', 'NI_FE', 'CR_FE', 'K_FE',
    'NA_FE', 'V_FE', 'CO_FE',
]
N_LABELS   = len(SELECTED_LABELS)          # 23
N_PIXELS   = 8575

ABUNDANCE_INDICES = [i for i, l in enumerate(SELECTED_LABELS) if '_FE' in l]
FE_H_INDEX = SELECTED_LABELS.index('FE_H')

# ── training hyper-params ──
BATCH_SIZE    = 256
LEARNING_RATE = 1e-3
EPOCHS        = 80
PATIENCE      = 15


# ╔══════════════════════════════════════════════════════════════╗
# ║ 2 — Label Imputation (train-only fit)                       ║
# ╚══════════════════════════════════════════════════════════════╝

def _read_raw_flagged(h5_path, selected_labels, start, end):
    """Read raw labels + construct bad-pixel mask for [start:end]."""
    with h5py.File(h5_path, 'r') as f:
        get_col = lambda k: f['metadata'][k]
        keys = f['metadata'].dtype.names
        raw = np.stack([get_col(p)[start:end] for p in selected_labels], axis=1)
        bad = np.zeros_like(raw, dtype=bool)
        for i, label in enumerate(selected_labels):
            flag = f"{label}_FLAG"
            if flag in keys:
                flg = get_col(flag)[start:end]
                if flg.dtype.names:
                    flg = flg[flg.dtype.names[0]]
                if flg.dtype.kind == 'V':
                    flg = flg.view('<i4')
                bad[:, i] = flg.astype(int) != 0
            elif label in ('TEFF', 'LOGG', 'VMICRO', 'VMACRO', 'VSINI'):
                bad[:, i] = raw[:, i] < -5000
    return raw, bad


def get_physical_labels(h5_path, train_limit, val_limit):
    """
    Returns (train_labels, val_labels) in **physical** units, shape (N, 23).
    [X/Fe] abundances are converted to [X/H] by adding [Fe/H].
    Imputer is fit on train only, then applied to val.
    """
    print("Reading training labels …")
    train_raw, train_bad = _read_raw_flagged(h5_path, SELECTED_LABELS, 0, train_limit)
    print("Reading validation labels …")
    val_raw, val_bad = _read_raw_flagged(h5_path, SELECTED_LABELS,
                                          train_limit, train_limit + val_limit)

    # Impute missing values (fit on train)
    train_imp = train_raw.copy().astype(float)
    train_imp[train_bad] = np.nan
    val_imp = val_raw.copy().astype(float)
    val_imp[val_bad] = np.nan

    imputer = IterativeImputer(estimator=BayesianRidge(), max_iter=10,
                               initial_strategy='median')
    train_labels = imputer.fit_transform(train_imp)
    val_labels   = imputer.transform(val_imp)

    # [X/Fe] → [X/H]
    for idx in ABUNDANCE_INDICES:
        train_labels[:, idx] += train_labels[:, FE_H_INDEX]
        val_labels[:, idx]   += val_labels[:, FE_H_INDEX]

    print(f"  Train labels: {train_labels.shape}   Val labels: {val_labels.shape}")
    return train_labels.astype(np.float32), val_labels.astype(np.float32)


# ╔══════════════════════════════════════════════════════════════╗
# ║ 3 — Normalisation                                           ║
# ╚══════════════════════════════════════════════════════════════╝

def compute_label_stats(train_labels):
    """Per-label mean/std from training set (physical units)."""
    mu  = np.mean(train_labels, axis=0)
    sig = np.std(train_labels, axis=0)
    sig[sig == 0] = 1.0
    return mu, sig


# ╔══════════════════════════════════════════════════════════════╗
# ║ 4 — TFRecord Writer + Dataset Builder                       ║
# ╚══════════════════════════════════════════════════════════════╝

NUM_SHARDS   = 16
TFREC_DIR    = "/kaggle/working/cnn_tfrecords"


def _bytes_feature(value):
    """Wrap a serialised tensor as a TFRecord bytes feature."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def write_tfrecords(h5_path, labels_norm, h5_start, split_name):
    """
    Reads flux from the HDF5 file in per-shard chunks and writes
    sharded TFRecords to disk.  Peak memory ≈ 1 shard of flux
    (~7.5k × 8575 × 4 B ≈ 260 MB for 16 shards over 120k stars).
    Bad pixels (non-finite or ≤ 1e-3) are replaced with 1.0.
    """
    total = len(labels_norm)
    shard_size = int(np.ceil(total / NUM_SHARDS))

    with h5py.File(h5_path, 'r') as f:
        ds_flux = f['flux']

        for shard_id in range(NUM_SHARDS):
            local_start = shard_id * shard_size
            local_end   = min((shard_id + 1) * shard_size, total)
            if local_start >= total:
                break

            global_start = h5_start + local_start
            global_end   = h5_start + local_end

            fname = os.path.join(TFREC_DIR,
                                 f"{split_name}_{shard_id:02d}.tfrec")
            print(f"   Writing {fname}  ({local_end - local_start} stars) …")

            # Read one shard of flux at a time
            chunk_flux   = ds_flux[global_start:global_end].astype(np.float32)
            chunk_labels = labels_norm[local_start:local_end]

            # Clean bad pixels
            bad = ~np.isfinite(chunk_flux) | (chunk_flux <= 1e-3)
            chunk_flux[bad] = 1.0

            with tf.io.TFRecordWriter(fname) as writer:
                for i in range(len(chunk_labels)):
                    flux_bytes  = tf.io.serialize_tensor(chunk_flux[i])
                    label_bytes = tf.io.serialize_tensor(chunk_labels[i])
                    feature = {
                        'flux':   _bytes_feature(flux_bytes),
                        'labels': _bytes_feature(label_bytes),
                    }
                    example = tf.train.Example(
                        features=tf.train.Features(feature=feature)
                    )
                    writer.write(example.SerializeToString())

    print(f"  {split_name} TFRecords written ({total} stars, {NUM_SHARDS} shards).")


def _parse_example(example_proto):
    """Parse a single TFRecord example → (flux, labels)."""
    feature_desc = {
        'flux':   tf.io.FixedLenFeature([], tf.string),
        'labels': tf.io.FixedLenFeature([], tf.string),
    }
    parsed = tf.io.parse_single_example(example_proto, feature_desc)
    flux   = tf.io.parse_tensor(parsed['flux'],   out_type=tf.float32)
    labels = tf.io.parse_tensor(parsed['labels'], out_type=tf.float32)
    flux.set_shape([N_PIXELS])
    labels.set_shape([N_LABELS])
    return flux, labels


def build_datasets(h5_path, train_labels_norm, val_labels_norm):
    """
    Writes sharded TFRecords (if not already present), then builds
    lazy tf.data pipelines that stream (flux, labels_norm) from disk.
    """
    import glob as _glob

    os.makedirs(TFREC_DIR, exist_ok=True)

    if not _glob.glob(os.path.join(TFREC_DIR, "*.tfrec")):
        print("Writing TFRecords …")
        write_tfrecords(h5_path, train_labels_norm, h5_start=0,
                        split_name="train")
        write_tfrecords(h5_path, val_labels_norm, h5_start=TRAIN_LIMIT,
                        split_name="val")
    else:
        print("TFRecords already present — skipping write.")

    train_files = sorted(tf.io.gfile.glob(os.path.join(TFREC_DIR, "train_*.tfrec")))
    val_files   = sorted(tf.io.gfile.glob(os.path.join(TFREC_DIR, "val_*.tfrec")))
    print(f"  Train shards: {len(train_files)}   Val shards: {len(val_files)}")

    train_ds = (
        tf.data.TFRecordDataset(train_files, num_parallel_reads=tf.data.AUTOTUNE)
        .map(_parse_example, num_parallel_calls=tf.data.AUTOTUNE)
        .shuffle(10_000, seed=RANDOM_SEED)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_ds = (
        tf.data.TFRecordDataset(val_files, num_parallel_reads=tf.data.AUTOTUNE)
        .map(_parse_example, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )
    return train_ds, val_ds


# ╔══════════════════════════════════════════════════════════════╗
# ║ 5 — Model Architecture                                      ║
# ╚══════════════════════════════════════════════════════════════╝

@register_keras_serializable()
class HeteroscedasticCNNPredictor(tf.keras.Model):
    """
    1D CNN that predicts stellar labels + per-label log-variance
    from an 8575-pixel APOGEE spectrum.

    Forward pass returns (mean, log_var) both shaped (batch, 23).
    """

    def __init__(self, n_labels=N_LABELS, **kwargs):
        super().__init__(**kwargs)
        self.n_labels = n_labels

        # ── convolutional backbone ──
        self.reshape_in = layers.Reshape((N_PIXELS, 1))

        self.conv1 = layers.Conv1D(32,  kernel_size=16, strides=4,
                                   activation='relu', padding='same')
        self.bn1   = layers.BatchNormalization()

        self.conv2 = layers.Conv1D(64,  kernel_size=8, strides=4,
                                   activation='relu', padding='same')
        self.bn2   = layers.BatchNormalization()

        self.conv3 = layers.Conv1D(128, kernel_size=4, strides=2,
                                   activation='relu', padding='same')
        self.bn3   = layers.BatchNormalization()

        self.conv4 = layers.Conv1D(256, kernel_size=4, strides=2,
                                   activation='relu', padding='same')
        self.bn4   = layers.BatchNormalization()

        self.gap   = layers.GlobalAveragePooling1D()

        # ── dense head ──
        self.dense1  = layers.Dense(512, activation='relu')
        self.drop1   = layers.Dropout(0.3)
        self.dense2  = layers.Dense(256, activation='relu')
        self.drop2   = layers.Dropout(0.2)

        # ── output heads ──
        self.mean_head    = layers.Dense(n_labels, name='label_mean')
        self.log_var_head = layers.Dense(n_labels, name='label_log_var')

    def call(self, x, training=False):
        # x: (batch, 8575)
        h = self.reshape_in(x)

        h = self.bn1(self.conv1(h), training=training)
        h = self.bn2(self.conv2(h), training=training)
        h = self.bn3(self.conv3(h), training=training)
        h = self.bn4(self.conv4(h), training=training)
        h = self.gap(h)

        h = self.drop1(self.dense1(h), training=training)
        h = self.drop2(self.dense2(h), training=training)

        mu      = self.mean_head(h)             # (batch, 23)
        log_var = self.log_var_head(h)           # (batch, 23)
        return mu, log_var

    def get_config(self):
        cfg = super().get_config()
        cfg['n_labels'] = self.n_labels
        return cfg


# ╔══════════════════════════════════════════════════════════════╗
# ║ 6 — Custom train / test steps (stabilised heteroscedastic)  ║
# ╚══════════════════════════════════════════════════════════════╝

VAR_FLOOR = 1e-4   # minimum predicted variance (normalised-label units²)


def _heteroscedastic_nll(y_true, mu, raw_log_var):
    """
    Gaussian NLL with learned per-label variance.
    Uses softplus + floor for the variance to avoid precision explosions:
        var = softplus(raw) + VAR_FLOOR
    This keeps gradients smooth even when the mean head is still inaccurate.

    y_true, mu, raw_log_var: all (batch, 23).
    Returns scalar loss.
    """
    # Stable variance: softplus is smooth, bounded below by VAR_FLOOR
    var     = tf.nn.softplus(raw_log_var) + VAR_FLOOR      # (batch, 23)
    log_var = tf.math.log(var)                              # for the log-det term

    # NLL = 0.5 * [log(var) + (y - mu)^2 / var]
    nll = 0.5 * (log_var + tf.square(y_true - mu) / var)

    # Per-sample mean across labels, then mean across batch
    return tf.reduce_mean(nll)


@register_keras_serializable()
class TrainableCNNPredictor(HeteroscedasticCNNPredictor):
    """
    Thin subclass that adds train_step / test_step so Keras' .fit()
    works with our tuple-output (mu, log_var) model — no wrapper needed.
    The model itself is what gets checkpointed and saved.
    """

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            mu, raw_log_var = self(x, training=True)
            loss = _heteroscedastic_nll(y, mu, raw_log_var)
        grads = tape.gradient(loss, self.trainable_variables)
        # Clip gradients to prevent spikes from variance-head feedback
        grads, _ = tf.clip_by_global_norm(grads, 1.0)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        # Track MSE separately so we can see mean quality independent of variance
        mse = tf.reduce_mean(tf.square(y - mu))
        return {"loss": loss, "mse": mse}

    def test_step(self, data):
        x, y = data
        mu, raw_log_var = self(x, training=False)
        loss = _heteroscedastic_nll(y, mu, raw_log_var)
        mse  = tf.reduce_mean(tf.square(y - mu))
        return {"loss": loss, "mse": mse}


# ╔══════════════════════════════════════════════════════════════╗
# ║ 7 — Training Loop                                           ║
# ╚══════════════════════════════════════════════════════════════╝

def train():
    # ── 1. Labels ──
    train_labels_phys, val_labels_phys = get_physical_labels(
        H5_PATH, TRAIN_LIMIT, VAL_LIMIT
    )
    label_mu, label_sig = compute_label_stats(train_labels_phys)

    # Save normalisation stats alongside the model for the backward pipeline
    stats_save_path = "/kaggle/working/cnn_label_stats.npz"
    np.savez(stats_save_path, mean=label_mu, std=label_sig)
    print(f"Label stats saved → {stats_save_path}")

    # Normalise labels (CNN predicts in normalised space; we denorm at inference)
    train_labels_norm = ((train_labels_phys - label_mu) / label_sig).astype(np.float32)
    val_labels_norm   = ((val_labels_phys   - label_mu) / label_sig).astype(np.float32)

    # ── 2. Datasets ──
    train_ds, val_ds = build_datasets(H5_PATH, train_labels_norm, val_labels_norm)

    # ── 3. Model ──
    model = TrainableCNNPredictor(n_labels=N_LABELS)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE))

    # Trigger build so summary and checkpoint shapes are known
    model(tf.zeros((1, N_PIXELS), dtype=tf.float32))
    model.summary()

    # ── 4. Train ──
    cbs = [
        callbacks.ModelCheckpoint(
            SAVE_PATH, save_best_only=True, monitor='val_loss',
            save_weights_only=True,  # weights-only avoids serialisation headaches
        ),
        callbacks.EarlyStopping(
            patience=PATIENCE, restore_best_weights=True,
            monitor='val_loss', min_delta=1e-6,
        ),
        callbacks.ReduceLROnPlateau(
            patience=5, factor=0.5, min_lr=1e-6, verbose=1,
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=cbs,
    )

    # ── 5. Save model ──
    # Load best weights (EarlyStopping already restored them)
    # Save the full model for backward-model-corrected.py
    inner_save = "/kaggle/working/cnn_label_predictor_inner.keras"
    model.save(inner_save)
    print(f"\nModel saved → {inner_save}")

    # ── 6. Quick sanity check ──
    print("\n── Sanity Check (first 5 val stars) ──")
    with h5py.File(H5_PATH, 'r') as f:
        sample_flux = f['flux'][TRAIN_LIMIT:TRAIN_LIMIT + 5].astype(np.float32)
    bad = ~np.isfinite(sample_flux) | (sample_flux <= 1e-3)
    sample_flux[bad] = 1.0

    mu_pred, logvar_pred = model(sample_flux, training=False)
    pred_phys = mu_pred.numpy() * label_sig + label_mu
    true_phys = val_labels_phys[:5]

    for i, name in enumerate(SELECTED_LABELS):
        print(f"  {name:>8s}  pred={pred_phys[0, i]:>10.3f}  "
              f"true={true_phys[0, i]:>10.3f}  "
              f"σ={np.sqrt(np.exp(logvar_pred[0, i].numpy())) * label_sig[i]:>8.3f}")

    print("\n✓ Training complete.")
    return history


# ╔══════════════════════════════════════════════════════════════╗
# ║ 8 — Main                                                    ║
# ╚══════════════════════════════════════════════════════════════╝

if __name__ == "__main__":
    train()

