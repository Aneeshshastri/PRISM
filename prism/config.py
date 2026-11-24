"""Label constants and the run configuration.

CAUTION: everything in PRISM indexes labels by position in ``SELECTED_LABELS``
followed by the four engineered features. Never reorder these lists.

Every ``*_FE`` column is stored as [X/H]: ``FE_H`` is added to [X/Fe] during
preprocessing (see ``prism.data.labels.to_x_h``).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

SELECTED_LABELS: list[str] = [
    # core
    "TEFF", "LOGG", "M_H", "VMICRO", "VMACRO", "VSINI",
    # CNO
    "C_FE", "N_FE", "O_FE",
    # metals
    "FE_H",
    "MG_FE", "SI_FE", "CA_FE", "TI_FE", "S_FE",
    "AL_FE", "MN_FE", "NI_FE", "CR_FE", "K_FE",
    "NA_FE", "V_FE", "CO_FE",
]
ENGINEERED: list[str] = ["INV_TEFF", "vbroad", "C_O_diff", "LOGPE"]
LABELS_27: list[str] = SELECTED_LABELS + ENGINEERED

N_RAW_LABELS = len(SELECTED_LABELS)  # 23
N_LABELS = len(LABELS_27)  # 27
N_PIXELS = 8575

CORE_INDICES: list[int] = list(range(9))
ABUND_INDICES: list[int] = list(range(9, 23))
# Columns that receive +FE_H in the [X/Fe] -> [X/H] conversion (FE_H itself excluded).
ABUNDANCE_INDICES: list[int] = [i for i, lab in enumerate(SELECTED_LABELS) if "_FE" in lab]
FE_H_INDEX = SELECTED_LABELS.index("FE_H")

CORE_LABELS: list[str] = ["TEFF", "LOGG", "M_H", "VMICRO", "VMACRO", "VSINI"]
CNO_LABELS: list[str] = ["C_FE", "N_FE", "O_FE"]
FEH_LABELS: list[str] = ["FE_H"]
TRACE_LABELS: list[str] = SELECTED_LABELS[10:]
ATOMIC_LABELS: list[str] = ["FE_H"] + TRACE_LABELS
# Labels without an ASPCAP *_FLAG column whose bad values are -9999 sentinels.
SENTINEL_LABELS: list[str] = ["TEFF", "LOGG", "M_H", "VMICRO", "VMACRO", "VSINI"]

PREP_VERSION = "1"

KAGGLE_H5 = "/kaggle/input/datasets/aneeshshastri/aspcapstar-dr17-150kstars/apogee_dr17_parallel.h5"
KAGGLE_MASKS = "/kaggle/input/datasets/aneeshshastri/element-masks"

# Fields whose change invalidates the prepared data cache.
_DATA_FIELDS = ("h5_path", "mask_dir", "train_end", "val_end", "test_end", "mask_threshold", "seed")


@dataclass
class Config:
    # --- paths (Kaggle defaults; override for local runs) ---
    h5_path: Path = Path(KAGGLE_H5)
    mask_dir: Path = Path(KAGGLE_MASKS)
    cache_dir: Path = Path("/tmp/prism_data")  # large arrays; not saved as Kaggle output
    artifact_dir: Path = Path("/kaggle/working/prism_artifacts")
    # Checkpoints from earlier Kaggle sessions: a directory, or a glob such as
    # "/kaggle/input/**/prism_artifacts/checkpoints" (recursive; one match per attached notebook output).
    checkpoint_inputs: Path | str | None = None
    # --- splits by HDF5 row ---
    train_end: int = 120_000
    val_end: int = 140_000
    test_end: int = 149_500
    # --- data ---
    seed: int = 42
    badpix_cutoff: float = 1e-3
    mask_threshold: float = 0.01
    # --- training ---
    batch_size: int = 256  # global batch, split across devices
    learning_rate: float = 1e-3
    epochs: int = 150
    patience: int = 20
    min_delta: float = 1e-7
    clip_norm: float = 1.0
    # --- cosine-decay-restarts schedule (Keras semantics) ---
    first_decay_steps: int = 5000
    t_mul: float = 2.0
    m_mul: float = 0.85
    alpha: float = 1e-6
    prefetch: int = 2
    # --- inverse ---
    mcmc_overrides: dict | None = None  # applied to every MCMCSettings preset

    def __post_init__(self) -> None:
        for name in ("h5_path", "mask_dir", "cache_dir", "artifact_dir"):
            setattr(self, name, Path(getattr(self, name)))
        if self.checkpoint_inputs is not None and "*" not in str(self.checkpoint_inputs):
            self.checkpoint_inputs = Path(self.checkpoint_inputs)

    def data_hash(self) -> str:
        """Hash of everything that determines the prepared data cache."""
        payload = {f: str(getattr(self, f)) for f in _DATA_FIELDS}
        payload["labels"] = LABELS_27
        payload["prep_version"] = PREP_VERSION
        blob = json.dumps(payload, sort_keys=True).encode()
        return hashlib.sha1(blob).hexdigest()[:16]

    def replace(self, **changes) -> "Config":
        return dataclasses.replace(self, **changes)

    @classmethod
    def for_kaggle(cls, root: str | Path = "/kaggle/input", **overrides) -> "Config":
        """Config whose data paths are discovered under ``root`` (any Kaggle mount layout)."""
        root = Path(root)

        def find(pattern: str) -> Path:
            hits = sorted(root.rglob(pattern))
            if not hits:
                raise FileNotFoundError(
                    f"{pattern} not found under {root}; attach the APOGEE HDF5 and element-masks "
                    "datasets to the notebook (Add Input)")
            return hits[0]

        paths = {}
        if "h5_path" not in overrides:
            paths["h5_path"] = find("apogee_dr17_parallel.h5")
        if "mask_dir" not in overrides:
            find("apogee_mask.npy")  # raises a clear error if missing
            # prefer the folder that also holds the C/N/O window files
            hits = [p.parent for p in sorted(root.rglob("apogee_mask.npy"))]
            paths["mask_dir"] = next((d for d in hits if (d / "C.filt").exists()), hits[0])
        return cls(**paths, **overrides)
