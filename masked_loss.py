"""
masked_loss.py
==============
Masked loss utilities for PRISM model training.

Handles imputed stellar labels by:
1. Pixel masking: zeroing loss at pixels sensitive to imputed abundance labels
2. Gradient gating: severing gradient flow through imputed CNO/abundance label paths

Usage in model-trainer.ipynb:
    from masked_loss import (
        compute_star_pixel_masks, masked_spl_loss, GradientGate,
        build_prism_emulator_masked, masked_chi2_estimate,
    )
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.saving import register_keras_serializable

# ════════════════════════════════════════════════════════════════════════
# 1. CNO FILTER LOADER  (7514 → 8575 conversion)
# ════════════════════════════════════════════════════════════════════════

N_PIXELS_FULL = 8575
N_PIXELS_FILT = 7514

# Label name → CNO filter filename
CNO_LABEL_TO_FILE = {
    'C_FE': 'C.filt',
    'N_FE': 'N.filt',
    'O_FE': 'O.filt',
}


def _insert_gaps(window_7514):
    """Convert a 7514-pixel array (chip-gaps excluded) to the full 8575-pixel range."""
    full = np.zeros(N_PIXELS_FULL, dtype=np.float32)
    full[246:3274]  = window_7514[0:3028]       # blue chip
    full[3585:6080] = window_7514[3028:5523]    # green chip
    full[6344:8335] = window_7514[5523:7514]    # red chip
    return full


def load_cno_masks(cno_filt_dir, threshold=0.01):
    """
    Load CNO element filter files and convert to boolean pixel masks.

    Parameters
    ----------
    cno_filt_dir : str
        Path to directory containing C.filt, N.filt, O.filt
    threshold : float
        Pixels with filter weight > threshold are considered "active"

    Returns
    -------
    dict : {label_name: np.ndarray(8575,) bool}
        True = pixel is sensitive to this CNO element
    """
    masks = {}
    for label, fname in CNO_LABEL_TO_FILE.items():
        filt_path = os.path.join(cno_filt_dir, fname)
        raw = np.loadtxt(filt_path)
        assert len(raw) == N_PIXELS_FILT, (
            f"Expected {N_PIXELS_FILT} values in {filt_path}, got {len(raw)}"
        )
        full = _insert_gaps(raw.astype(np.float32))
        masks[label] = full > threshold
        print(f"  CNO mask {label}: {masks[label].sum()} active pixels")
    return masks


# ════════════════════════════════════════════════════════════════════════
# 2. PER-STAR PIXEL MASK COMPUTATION
# ════════════════════════════════════════════════════════════════════════

def compute_star_pixel_masks(
    bad_mask,
    apogee_mask_path,
    cno_filt_dir,
    selected_labels,
    threshold=0.01,
):
    """
    Build per-star pixel masks that zero out loss at pixels affected by
    imputed labels.

    Parameters
    ----------
    bad_mask : np.ndarray, shape (N_stars, N_selected_labels), bool
        True where a label was imputed for that star.
    apogee_mask_path : str
        Path to apogee_mask.npy, shape (8575, 27).
    cno_filt_dir : str
        Path to CNO filter directory.
    selected_labels : list of str
        Label names in order (length must match bad_mask.shape[1]).
    threshold : float
        Pixel filter weight threshold for boolean conversion.

    Returns
    -------
    pixel_masks : np.ndarray, shape (N_stars, 8575), float32
        1.0 = pixel contributes to loss, 0.0 = pixel is suppressed.
    """
    n_stars = bad_mask.shape[0]
    pixel_masks = np.ones((n_stars, N_PIXELS_FULL), dtype=np.float32)

    # --- Load abundance masks from apogee_mask.npy ---
    apogee_mask = np.load(apogee_mask_path)  # (8575, 27)
    # Build label → column index mapping for the 27-column mask
    labels_27 = (
        selected_labels
        + ['INV_TEFF', 'vbroad', 'C_O_diff', 'LOGPE']
    )

    # --- Load CNO masks ---
    cno_masks = load_cno_masks(cno_filt_dir, threshold=threshold)

    # --- CNO labels ---
    cno_labels = {'C_FE', 'N_FE', 'O_FE'}

    # --- Identify abundance labels (those with _FE but NOT CNO) ---
    abundance_labels = [
        l for l in selected_labels if '_FE' in l and l not in cno_labels
    ]

    # --- Build pixel masks per star ---
    n_imputed_total = 0
    for star_idx in range(n_stars):
        if not bad_mask[star_idx].any():
            continue  # no imputed labels → full mask (all 1.0)

        for label_idx in range(len(selected_labels)):
            if not bad_mask[star_idx, label_idx]:
                continue

            label = selected_labels[label_idx]

            if label in cno_labels:
                # Use CNO-specific filter
                active_pixels = cno_masks[label]
                pixel_masks[star_idx, active_pixels] = 0.0
                n_imputed_total += 1

            elif label in abundance_labels:
                # Use apogee_mask column
                col_idx = labels_27.index(label)
                active_pixels = apogee_mask[:, col_idx] > threshold
                if active_pixels.sum() > 0:
                    pixel_masks[star_idx, active_pixels] = 0.0
                n_imputed_total += 1

            # Global labels (TEFF, LOGG, etc.) are not masked —
            # they affect all pixels, masking them would zero everything

    active_counts = (pixel_masks < 1.0).any(axis=1).sum()
    print(f"  Pixel masks built: {active_counts}/{n_stars} stars have masked pixels")
    print(f"  Total imputed label-instances masked: {n_imputed_total}")

    return pixel_masks


# ════════════════════════════════════════════════════════════════════════
# 3. MASKED LOSS FUNCTION
# ════════════════════════════════════════════════════════════════════════

@register_keras_serializable()
def masked_spl_loss(y_true, y_pred):
    """
    Spectral pixel loss with per-star pixel masking for imputed labels.

    y_true shape: (batch, 8575, 3)
        channel 0: observed flux
        channel 1: inverse variance
        channel 2: pixel mask (1.0 = keep, 0.0 = suppress)
    y_pred shape: (batch, 8575) or (batch, 8575, 1)
    """
    if len(y_pred.shape) == 2:
        y_pred = tf.expand_dims(y_pred, -1)

    real_flux  = y_true[:, :, 0:1]
    ivar       = y_true[:, :, 1:2]
    pixel_mask = y_true[:, :, 2:3]  # <-- NEW: imputation pixel mask

    BADPIX_CUTOFF = 1e-3
    valid_mask = tf.cast(real_flux > BADPIX_CUTOFF, tf.float32)
    safe_flux  = tf.where(valid_mask == 1.0, real_flux, y_pred)

    weight = 1e-5 / (1.0 / ivar + 1e-5)
    weight = tf.where(ivar > 0, weight, 0.0)

    wmse_term = tf.square(safe_flux - y_pred) * weight * valid_mask
    # Apply imputation pixel mask — zero contribution from imputed-label pixels
    wmse_term = wmse_term * pixel_mask

    loss = tf.where(tf.math.is_finite(wmse_term), wmse_term, tf.zeros_like(wmse_term))
    return loss


@register_keras_serializable()
def masked_chi2_estimate(y_true, y_pred):
    """χ² estimate compatible with 3-channel y_true."""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    if len(y_pred.shape) == 2:
        y_pred = tf.expand_dims(y_pred, -1)

    real_flux  = y_true[:, :, 0:1]
    ivar       = y_true[:, :, 1:2]
    pixel_mask = y_true[:, :, 2:3]

    BADPIX_CUTOFF = 1e-3
    valid_mask = tf.cast(real_flux > BADPIX_CUTOFF, tf.float32)
    safe_flux  = tf.where(valid_mask == 1.0, real_flux, y_pred)

    weight = tf.where(ivar > 0, ivar, 0.0)
    wmse_term = tf.square(safe_flux - y_pred) * weight * valid_mask * pixel_mask

    n_valid_pixels = tf.reduce_sum(valid_mask * pixel_mask, 1)
    # n_params is approximate — doesn't subtract imputed labels
    n_params = 23.0
    dof = n_valid_pixels - n_params
    return tf.reduce_sum(wmse_term, 1) / dof


# ════════════════════════════════════════════════════════════════════════
# 4. GRADIENT GATE LAYER (for PRISM)
# ════════════════════════════════════════════════════════════════════════

@register_keras_serializable()
class GradientGate(layers.Layer):
    """
    Per-sample gradient severing for imputed label inputs.

    Forward pass: unchanged — uses the (imputed) label value normally.
    Backward pass: gradient is zero for imputed labels via tf.stop_gradient.

    Inputs
    ------
    label_value : (batch, 1) float — the label value
    is_imputed  : (batch, 1) float — 1.0 if imputed, 0.0 if real

    Returns
    -------
    gated_value : (batch, 1) float — same values, severed gradients
    """

    def call(self, label_value, is_imputed):
        is_imputed = tf.cast(is_imputed, label_value.dtype)
        live = label_value * (1.0 - is_imputed)
        dead = tf.stop_gradient(label_value) * is_imputed
        return live + dead

    def get_config(self):
        return super().get_config()


# ════════════════════════════════════════════════════════════════════════
# 5. PRISM MODEL BUILDER (with GradientGate)
# ════════════════════════════════════════════════════════════════════════

def build_prism_emulator_masked(
    element_indices,
    config,
    IDX_GLOBAL,
    IDX_CNO,
    LABELS_ORDER,
    ATOMIC_LABELS,
    build_continuum_branch_fn,
    ColumnSelector,
    GetAbundances,
    BeerLambertLaw,
    create_sparse_projector_fn,
):
    """
    Build the PRISM (Sparse Physics Emulator) model with GradientGate
    on all CNO and abundance label inputs.

    This is identical to build_final_emulator but:
    - Accepts a second input: bad_mask (batch, N_selected_labels) float32
    - Gates each CNO label individually before molecular branch
    - Gates each abundance label before its expert branch

    Parameters
    ----------
    element_indices : dict
        From load_element_indices(). Maps label → (active_indices, weights).
    config : Config object
    IDX_GLOBAL, IDX_CNO, LABELS_ORDER, ATOMIC_LABELS : from notebook
    build_continuum_branch_fn : callable
    ColumnSelector, GetAbundances, BeerLambertLaw : layer classes
    create_sparse_projector_fn : callable

    Returns
    -------
    tf.keras.Model with inputs=[full_input, bad_mask_input], outputs=flux
    """
    from tensorflow.keras import Input, Model, models

    n_labels = config.N_LABELS
    n_selected = len(config.SELECTED_LABELS)

    # --- INPUTS ---
    full_input     = Input(shape=(n_labels,),    name="All_Labels")
    bad_mask_input = Input(shape=(n_selected,),  name="Bad_Mask")

    gate = GradientGate()

    # --- CNO indices in selected_labels ---
    cno_label_names = {'C_FE', 'N_FE', 'O_FE'}
    cno_indices_in_selected = {
        l: config.SELECTED_LABELS.index(l)
        for l in cno_label_names
        if l in config.SELECTED_LABELS
    }

    # === BRANCH A: CONTINUUM (Global params — no gating needed) ===
    input_global = ColumnSelector(IDX_GLOBAL, name="Slice_Global")(full_input)
    k_cont = build_continuum_branch_fn(input_global)

    # === BRANCH B: MOLECULES (CNO — gate individual CNO labels) ===
    # IDX_CNO gives indices into the 27-label full_input
    # We rebuild the CNO input by gating each CNO label individually
    cno_parts = []
    for idx in IDX_CNO:
        label_val = GetAbundances(idx, name=f"CNO_Slice_{idx}")(full_input)

        # Check if this index corresponds to a CNO label
        label_name = LABELS_ORDER[idx] if idx < len(LABELS_ORDER) else None
        if label_name in cno_label_names:
            sel_idx = cno_indices_in_selected[label_name]
            is_imputed = GetAbundances(sel_idx, name=f"Mask_{label_name}")(bad_mask_input)
            label_val = gate(label_val, is_imputed)

        cno_parts.append(label_val)

    input_cno = layers.Concatenate(name="CNO_Gated")(cno_parts)

    x_mol = layers.Dense(256, activation='swish')(input_cno)
    x_mol = layers.Dense(1024, activation='swish')(x_mol)
    tau_mol = layers.Dense(8575, activation='softplus', name="Tau_Molecules")(x_mol)

    # === BRANCH C: THERMODYNAMIC STATE (unchanged) ===
    input_EOS = ColumnSelector(IDX_GLOBAL + IDX_CNO, name="Slice_Global_Full")(full_input)
    x_state = layers.Dense(32, activation='swish', name="EOS_Hidden_1")(input_EOS)
    x_state = layers.Dense(16, activation='swish', name="EOS_Hidden_2")(x_state)
    state_vector = layers.Dense(8, activation='linear', name="State_Vector")(x_state)

    # === BRANCH D: SPARSE ATOMIC EXPERTS (gated per element) ===
    tau_atoms = []

    for label, (indices, weights) in element_indices.items():
        n_active = len(indices)
        col_idx = LABELS_ORDER.index(label)

        # Slice the raw abundance
        raw_abundance = GetAbundances(col_idx, name=f"Slice_{label}")(full_input)

        # Gate it if label is in selected_labels
        if label in config.SELECTED_LABELS:
            sel_idx = config.SELECTED_LABELS.index(label)
            is_imputed = GetAbundances(sel_idx, name=f"Mask_{label}")(bad_mask_input)
            raw_abundance = gate(raw_abundance, is_imputed)

        expert_in = layers.Concatenate()([state_vector, raw_abundance])

        x = layers.Dense(64, activation='swish')(expert_in)
        local_tau = layers.Dense(n_active, activation='softplus',
                                 name=f"Tau_{label}_Local")(x)

        projector = create_sparse_projector_fn(indices, weights, label_name=label)
        full_tau = projector(local_tau)
        tau_atoms.append(full_tau)

    # === PHYSICS SUMMATION ===
    all_sources = [tau_mol] + tau_atoms
    total_tau = layers.Add(name="Total_Opacity")(all_sources)

    output_flux = BeerLambertLaw(dtype='float32', name="Final_Flux")(k_cont, total_tau)

    return models.Model(
        inputs=[full_input, bad_mask_input],
        outputs=output_flux,
        name="Sparse_Physics_Emulator_Masked",
    )


# ════════════════════════════════════════════════════════════════════════
# 6. TFRECORD HELPERS
# ════════════════════════════════════════════════════════════════════════

def _float_feature(value):
    """Scalar float feature for TFRecord."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _bytes_feature(value):
    """Bytes feature for TFRecord."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
