"""HDF5 -> per-split numpy cache.

Layout of ``cfg.cache_dir/<data_hash>/``::

    manifest.json            written last; its presence marks a complete cache
    stats.npz                mean / std of the 27 training inputs
    imputer.pkl              train-only IterativeImputer
    suppress.npy             (23, 8575) pixel suppression matrix for imputed labels
    {train,val,test}/labels.npy     (N, 27) float32, normalised
                     bad_mask.npy   (N, 23) bool, True = imputed
                     flux.npy       (N, 8575) float32, raw
    {train,val}/spl_weight.npy      (N, 8575) float16 loss weight 1e-5/(1/ivar+1e-5)
                                    (training never needs ivar itself; float16 halves the I/O)
    test/ivar.npy                   (N, 8575) float32 (evaluation / inversion need ivar)
                     weights.npy    (N,) float32 stratification weights
                     global_idx.npy (N,) int64 HDF5 row index
    test/label_err.npy (N, 23), test/snr.npy (N,)

The cache is keyed by ``Config.data_hash()``: any change in data-relevant
config yields a new directory, and stale ones are deleted.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import h5py
import joblib
import numpy as np

from prism.config import SELECTED_LABELS, Config
from prism.data import labels as L
from prism.data.masks import load_apogee_mask, load_cno_masks, suppression_matrix

CHUNK = 2048
SPLITS = ("train", "val", "test")
MARKER = ".prism_cache"  # only directories carrying this marker are ever deleted


def _is_prism_cache(path: Path) -> bool:
    return path.is_dir() and (path / MARKER).exists()


def _ranges(cfg: Config) -> dict[str, tuple[int, int]]:
    return {"train": (0, cfg.train_end), "val": (cfg.train_end, cfg.val_end),
            "test": (cfg.val_end, cfg.test_end)}


def _copy_spectra(h5, name: str, start: int, end: int, out: Path) -> None:
    arr = np.lib.format.open_memmap(out, mode="w+", dtype=np.float32, shape=(end - start, h5[name].shape[1]))
    for s in range(start, end, CHUNK):
        e = min(s + CHUNK, end)
        arr[s - start:e - start] = h5[name][s:e]
    arr.flush()
    del arr


def _copy_spl_weight(h5, start: int, end: int, out: Path) -> None:
    from prism.losses import spl_weight

    arr = np.lib.format.open_memmap(out, mode="w+", dtype=np.float16, shape=(end - start, h5["ivar"].shape[1]))
    for s in range(start, end, CHUNK):
        e = min(s + CHUNK, end)
        arr[s - start:e - start] = np.asarray(spl_weight(h5["ivar"][s:e])).astype(np.float16)
    arr.flush()
    del arr


def load_stats(prep_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(Path(prep_dir) / "stats.npz") as d:
        return d["mean"].astype(np.float32), d["std"].astype(np.float32)


def prepare_all(cfg: Config, force: bool = False) -> Path:
    root = Path(cfg.cache_dir)
    out = root / cfg.data_hash()
    if (out / "manifest.json").exists() and not force:
        return out
    if out.exists():
        if not _is_prism_cache(out):
            raise FileExistsError(f"{out} exists and is not a PRISM cache; refusing to overwrite it")
        shutil.rmtree(out)
    out.mkdir(parents=True)
    (out / MARKER).write_text(cfg.data_hash())
    t0 = time.time()
    ranges = _ranges(cfg)
    print(f"Preparing PRISM data cache in {out}")

    with h5py.File(cfg.h5_path, "r") as h5:
        raw, bad = {}, {}
        for split, (s, e) in ranges.items():
            raw[split], bad[split] = L.read_raw_flagged(h5, SELECTED_LABELS, s, e)
        imputer = L.fit_imputer(raw["train"], bad["train"], cfg.seed)
        feats = {split: L.engineer_features(L.to_x_h(L.impute(imputer, raw[split], bad[split])))
                 for split in SPLITS}
        mean, std = L.norm_stats(feats["train"])
        norm = {split: ((feats[split] - mean) / std).astype(np.float32) for split in SPLITS}
        weights = {
            "train": L.stratified_weights(norm["train"]),
            "val": L.stratified_weights(norm["val"], ref_labels_norm27=norm["train"]),
            "test": np.ones(len(norm["test"]), np.float32),
        }
        for split, (s, e) in ranges.items():
            d = out / split
            d.mkdir()
            np.save(d / "labels.npy", norm[split])
            np.save(d / "bad_mask.npy", bad[split])
            np.save(d / "weights.npy", weights[split])
            np.save(d / "global_idx.npy", np.arange(s, e, dtype=np.int64))
            _copy_spectra(h5, "flux", s, e, d / "flux.npy")
            if split == "test":
                _copy_spectra(h5, "ivar", s, e, d / "ivar.npy")
            else:
                _copy_spl_weight(h5, s, e, d / "spl_weight.npy")
            print(f"  {split:<5} {e - s:>7} stars, {int(bad[split].any(1).sum())} with imputed labels")
        s, e = ranges["test"]
        np.save(out / "test" / "label_err.npy", L.read_errors(h5, SELECTED_LABELS, s, e).astype(np.float32))
        np.save(out / "test" / "snr.npy", h5["metadata"]["SNR"][s:e].astype(np.float32))

    am = load_apogee_mask(cfg.mask_dir)
    np.save(out / "suppress.npy", suppression_matrix(am, load_cno_masks(cfg.mask_dir, cfg.mask_threshold),
                                                    cfg.mask_threshold))
    np.savez(out / "stats.npz", mean=mean, std=std)
    joblib.dump(imputer, out / "imputer.pkl")
    manifest = {"hash": cfg.data_hash(), "ranges": ranges, "created": time.time(),
                "h5_path": str(cfg.h5_path), "seconds": round(time.time() - t0, 1)}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))

    art = Path(cfg.artifact_dir)
    art.mkdir(parents=True, exist_ok=True)
    for name in ("stats.npz", "imputer.pkl", "manifest.json"):
        shutil.copy2(out / name, art / name)
    for other in root.iterdir():  # stale caches from other configs; never foreign folders
        if other != out and _is_prism_cache(other):
            shutil.rmtree(other, ignore_errors=True)
    print(f"  done in {time.time() - t0:.1f}s")
    return out
