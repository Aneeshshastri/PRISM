import json

import numpy as np


def test_prepare_outputs(tiny_config):
    from prism.data.prepare import load_stats, prepare_all

    d = prepare_all(tiny_config)
    tr = np.load(d / "train/labels.npy")
    assert tr.shape == (30, 27) and tr.dtype == np.float32
    assert np.allclose(tr.mean(0), 0, atol=1e-5)
    assert np.isfinite(np.load(d / "val/labels.npy")).all()
    assert np.load(d / "val/labels.npy").shape == (15, 27)
    assert np.load(d / "test/label_err.npy").shape == (15, 23)
    assert np.load(d / "test/snr.npy").shape == (15,)
    assert np.load(d / "train/flux.npy", mmap_mode="r").shape == (30, 8575)
    assert np.load(d / "train/bad_mask.npy").dtype == bool
    assert np.load(d / "val/global_idx.npy").tolist() == list(range(30, 45))
    assert np.load(d / "suppress.npy").shape == (23, 8575)
    mean, std = load_stats(d)
    assert mean.shape == (27,) and (std > 0).all()
    assert json.loads((d / "manifest.json").read_text())["hash"] == tiny_config.data_hash()
    for name in ("stats.npz", "imputer.pkl", "manifest.json"):
        assert (tiny_config.artifact_dir / name).exists()


def test_prepare_is_cached(tiny_config):
    from prism.data.prepare import prepare_all

    d1 = prepare_all(tiny_config)
    t = (d1 / "manifest.json").stat().st_mtime_ns
    assert prepare_all(tiny_config) == d1
    assert (d1 / "manifest.json").stat().st_mtime_ns == t


def test_prepare_rebuilds_on_config_change(tiny_config):
    from prism.data.prepare import prepare_all

    d1 = prepare_all(tiny_config)
    d2 = prepare_all(tiny_config.replace(train_end=28))
    assert d1 != d2 and not d1.exists()  # stale cache removed
    assert np.load(d2 / "train/labels.npy").shape == (28, 27)


def test_loader_pads_last_batch(tiny_config):
    from prism.data.loader import BatchLoader
    from prism.data.prepare import prepare_all

    d = prepare_all(tiny_config)
    bl = BatchLoader(d, "train", 8, shuffle=True, seed=0, n_devices=2)
    batches = list(bl)
    assert len(bl) == 4 and len(batches) == 4
    assert batches[-1]["labels"].shape[0] == 8 and batches[-1]["row_mask"].sum() == 6
    assert batches[0]["flux"].shape == (8, 8575) and batches[0]["bad_mask"].dtype == np.float32
    real = np.concatenate([b["labels"][b["row_mask"] > 0] for b in batches])
    stored = np.load(d / "train/labels.npy")
    assert sorted(real[:, 0].tolist()) == sorted(stored[:, 0].tolist())


def test_loader_epoch_reshuffle(tiny_config):
    from prism.data.loader import BatchLoader
    from prism.data.prepare import prepare_all

    d = prepare_all(tiny_config)

    def order(epoch, seed=0):
        bl = BatchLoader(d, "train", 8, shuffle=True, seed=seed, fields=("labels",))
        bl.set_epoch(epoch)
        return np.concatenate([b["labels"][:, 0] for b in bl])

    assert not np.array_equal(order(0), order(1))
    assert np.array_equal(order(1), order(1))


def test_loader_no_shuffle_keeps_order(tiny_config):
    from prism.data.loader import BatchLoader
    from prism.data.prepare import prepare_all

    d = prepare_all(tiny_config)
    bl = BatchLoader(d, "test", 8, shuffle=False, seed=0, fields=("labels", "label_err"))
    got = np.concatenate([b["labels"][b["row_mask"] > 0] for b in bl])
    assert np.array_equal(got, np.load(d / "test/labels.npy"))


def test_prepare_never_deletes_foreign_directories(tiny_config):
    """Review finding: cache cleanup must only remove PRISM cache directories."""
    from prism.data.prepare import prepare_all

    foreign = tiny_config.cache_dir / "prism_artifacts" / "checkpoints"
    foreign.mkdir(parents=True)
    (foreign / "keep.txt").write_text("precious")
    prepare_all(tiny_config)
    old = prepare_all(tiny_config.replace(train_end=28))
    assert (foreign / "keep.txt").read_text() == "precious" and old.exists()
