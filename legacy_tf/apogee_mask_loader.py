"""
apogee_mask_loader.py

Reads per-element APOGEE window files (`element masks/<El>.mask`; the
`element filters/<El>.filt` files share the same format), converts them from
the 7514-pixel (chip-gap-excluded) grid to the full 8575-pixel detector range,
and assembles an (8575, 27) mask array with one column per model input
(column order = LABELS_ORDER, identical to model-trainer.ipynb).

Default output: apogee_inference_mask.npy (apogee_mask.npy has the same layout).
Convention : values are line weights in [0, 1]; > 0 means the pixel lies in a
             line window of that element ("allowed"/"sensitive"), 0 means it
             does not.  Consumers threshold at > 0.01.
Columns without a window file (TEFF, LOGG, …, C/N/O and the engineered
features) are all zeros.
"""

import os
import numpy as np

# ── Label ordering (must match model-trainer.ipynb exactly) ──────────────
SELECTED_LABELS = [
    # 1. Core
    'TEFF', 'LOGG', 'M_H', 'VMICRO', 'VMACRO', 'VSINI',
    # 2. CNO
    'C_FE', 'N_FE', 'O_FE',
    # 3. Metals
    'FE_H',
    'MG_FE', 'SI_FE', 'CA_FE', 'TI_FE', 'S_FE',
    'AL_FE', 'MN_FE', 'NI_FE', 'CR_FE', 'K_FE',
    'NA_FE', 'V_FE', 'CO_FE',
]

LABELS_ORDER = SELECTED_LABELS + ['INV_TEFF', 'vbroad', 'C_O_diff', 'LOGPE']

# Elements that have sparse projectors (and therefore .filt files)
ATOMIC_LABELS = [
    'FE_H',
    'MG_FE', 'SI_FE', 'CA_FE', 'TI_FE', 'S_FE',
    'AL_FE', 'MN_FE', 'NI_FE', 'CR_FE', 'K_FE',
    'NA_FE', 'V_FE', 'CO_FE',
]

# Mapping from label name → .filt filename (element symbol only)
LABEL_TO_ELEMENT = {
    'FE_H':  'Fe',
    'MG_FE': 'Mg',
    'SI_FE': 'Si',
    'CA_FE': 'Ca',
    'TI_FE': 'Ti',
    'S_FE':  'S',
    'AL_FE': 'Al',
    'MN_FE': 'Mn',
    'NI_FE': 'Ni',
    'CR_FE': 'Cr',
    'K_FE':  'K',
    'NA_FE': 'Na',
    'V_FE':  'V',
    'CO_FE': 'Co',
}

# ── Chip-gap insertion ───────────────────────────────────────────────────
N_PIXELS_FULL = 8575
N_PIXELS_FILT = 7514


def insert_gaps(window_7514):
    """Convert a 7514-pixel array (chip-gaps excluded) to the full 8575-pixel range."""
    full = np.zeros(N_PIXELS_FULL)
    full[246:3274]  = window_7514[0:3028]       # blue chip
    full[3585:6080] = window_7514[3028:5523]    # green chip
    full[6344:8335] = window_7514[5523:7514]    # red chip
    return full


# ── Main builder ─────────────────────────────────────────────────────────
def build_apogee_mask(filt_dir='element masks', output_path='apogee_inference_mask.npy'):
    """
    Build the apogee mask array from per-element .mask files and save it.

    Parameters
    ----------
    filt_dir : str
        Directory containing {Element}.mask files.
    output_path : str
        Where to save the resulting .npy file.

    Returns
    -------
    mask : np.ndarray, shape (8575, 27); > 0 marks line pixels
    """
    n_labels = len(LABELS_ORDER)
    mask = np.zeros((N_PIXELS_FULL, n_labels), dtype=np.float32)

    for label in ATOMIC_LABELS:
        col_idx = LABELS_ORDER.index(label)
        element = LABEL_TO_ELEMENT[label]
        filt_path = os.path.join(filt_dir, f'{element}.mask')

        print(f'Loading {filt_path} for {label} (column {col_idx}) ... ', end='')

        filt_data = np.loadtxt(filt_path)  # reads 7514 values (blank trailing line is ignored)
        assert len(filt_data) == N_PIXELS_FILT, (
            f'Expected {N_PIXELS_FILT} values in {filt_path}, got {len(filt_data)}'
        )

        mask[:, col_idx] = insert_gaps(filt_data)
        print('OK')

    np.save(output_path, mask)
    print(f'\nSaved apogee_mask → {output_path}  (shape {mask.shape})')
    return mask


if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filt_dir = os.path.join(script_dir, 'element masks')
    out_path = os.path.join(script_dir, 'apogee_inference_mask.npy')
    build_apogee_mask(filt_dir=filt_dir, output_path=out_path)
