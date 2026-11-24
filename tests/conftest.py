import os

# Must run before anything imports jax: 2 fake CPU devices emulate Kaggle's T4 x2 mesh.
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=2")
os.environ["JAX_PLATFORMS"] = "cpu"

from pathlib import Path  # noqa: E402

import h5py  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

N_STARS = 60
N_PIX = 8575
# Rows with deliberately bad values in the synthetic catalogue.
PLANTED = {"i_mh": 3, "i_o": 5, "i_mg": 7, "i_teff_err": 11, "i_bad_flux": 2}

_LABELS = [
    "TEFF", "LOGG", "M_H", "VMICRO", "VMACRO", "VSINI", "C_FE", "N_FE", "O_FE", "FE_H",
    "MG_FE", "SI_FE", "CA_FE", "TI_FE", "S_FE", "AL_FE", "MN_FE", "NI_FE", "CR_FE", "K_FE",
    "NA_FE", "V_FE", "CO_FE",
]
_FLAGGED = [lab for lab in _LABELS if "_FE" in lab or lab == "FE_H"]  # 17 labels
_ERR = [lab for lab in _LABELS if lab not in ("VMICRO", "VMACRO", "VSINI")]


def _write_h5(path: Path) -> None:
    rng = np.random.default_rng(0)
    n = N_STARS
    vals = {
        "TEFF": rng.uniform(3500, 7000, n), "LOGG": rng.uniform(0.0, 5.0, n),
        "M_H": rng.uniform(-2.0, 0.4, n), "VMICRO": rng.uniform(0.5, 2.5, n),
        "VMACRO": rng.uniform(1.0, 8.0, n), "VSINI": rng.uniform(0.0, 10.0, n),
    }
    for lab in _LABELS[6:]:
        vals[lab] = rng.normal(0.0, 0.15, n)
    vals["FE_H"] = vals["M_H"] + rng.normal(0, 0.03, n)
    vals["M_H"][PLANTED["i_mh"]] = -9999.0
    vals["O_FE"][PLANTED["i_o"]] = np.nan

    fields = [(lab, "f4") for lab in _LABELS]
    fields += [(f"{lab}_ERR", "f4") for lab in _ERR]
    fields += [(f"{lab}_FLAG", "i4") for lab in _FLAGGED]
    fields += [("SNR", "f4")]
    meta = np.zeros(n, dtype=fields)
    for lab in _LABELS:
        meta[lab] = vals[lab]
    for lab in _ERR:
        meta[f"{lab}_ERR"] = np.abs(rng.normal(0.02, 0.005, n)) * (100 if lab == "TEFF" else 1)
    meta["TEFF_ERR"][PLANTED["i_teff_err"]] = -9999.0
    meta["MG_FE_FLAG"][PLANTED["i_mg"]] = 1
    meta["SNR"] = rng.uniform(50, 400, n)

    flux = (1.0 - 0.1 * rng.random((n, N_PIX))).astype(np.float32)
    ivar = rng.uniform(100, 1e4, (n, N_PIX)).astype(np.float32)
    flux[PLANTED["i_bad_flux"], :5] = np.nan
    flux[PLANTED["i_bad_flux"], 5:10] = -1.0
    ivar[PLANTED["i_bad_flux"], 10:15] = 0.0
    with h5py.File(path, "w") as f:
        f.create_dataset("flux", data=flux)
        f.create_dataset("ivar", data=ivar)
        f.create_dataset("metadata", data=meta)


def _write_masks(directory: Path) -> None:
    rng = np.random.default_rng(1)
    mask = np.zeros((N_PIX, 27), np.float32)
    for col in range(9, 23):
        n_lines = 6 if col == 14 else int(rng.integers(30, 120))  # col 14 = S_FE
        idx = rng.choice(N_PIX, n_lines, replace=False)
        mask[idx, col] = rng.uniform(0.05, 1.0, n_lines)
    np.save(directory / "apogee_mask.npy", mask)
    for el in ("C", "N", "O"):
        filt = np.zeros(7514)
        filt[rng.choice(7514, 200, replace=False)] = rng.uniform(0.05, 1.0, 200)
        np.savetxt(directory / f"{el}.filt", filt, fmt="%.8f")


@pytest.fixture(scope="session")
def planted():
    return dict(PLANTED)


@pytest.fixture(scope="session")
def synthetic_h5(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("h5") / "apogee_synthetic.h5"
    _write_h5(path)
    return path


@pytest.fixture(scope="session")
def synthetic_masks(tmp_path_factory) -> Path:
    directory = tmp_path_factory.mktemp("masks")
    _write_masks(directory)
    return directory


@pytest.fixture
def tiny_config(tmp_path, synthetic_h5, synthetic_masks):
    from prism.config import Config

    return Config(
        h5_path=synthetic_h5, mask_dir=synthetic_masks,
        cache_dir=tmp_path / "cache", artifact_dir=tmp_path / "artifacts",
        train_end=30, val_end=45, test_end=60,
        batch_size=8, epochs=2, patience=1,
    )
