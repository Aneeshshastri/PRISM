import numpy as np
import pytest


def test_label_constants():
    from prism import config as c

    assert len(c.SELECTED_LABELS) == 23 and len(c.LABELS_27) == 27
    assert c.LABELS_27[23:] == ["INV_TEFF", "vbroad", "C_O_diff", "LOGPE"]
    assert c.FE_H_INDEX == 9 and 9 not in c.ABUNDANCE_INDICES and len(c.ABUNDANCE_INDICES) == 16
    assert c.CORE_INDICES == list(range(9)) and c.ABUND_INDICES == list(range(9, 23))
    assert c.ATOMIC_LABELS[0] == "FE_H" and len(c.ATOMIC_LABELS) == 14
    assert len(c.TRACE_LABELS) == 13 and c.N_PIXELS == 8575 and c.N_LABELS == 27


def test_data_hash_changes():
    from prism.config import Config

    assert Config().data_hash() != Config(train_end=1000).data_hash()
    # training-only parameters must not invalidate prepared data
    assert Config().data_hash() == Config(batch_size=8).data_hash()


def test_pad_and_shard():
    from prism.mesh import make_mesh, pad_to_multiple, shard_batch

    a = np.arange(5 * 3, dtype=np.float32).reshape(5, 3)
    p, m = pad_to_multiple(a, 2)
    assert p.shape == (6, 3) and m.tolist() == [1, 1, 1, 1, 1, 0]
    x = shard_batch({"x": p}, make_mesh())["x"]
    assert len(x.sharding.device_set) == 2 and np.allclose(np.asarray(x), p)
    with pytest.raises(ValueError):
        shard_batch({"x": a}, make_mesh())


def test_check_env_cpu_ok():
    from prism.env import check_env

    info = check_env(expected_gpus=None, require_gpu=False)
    assert info["n_devices"] == 2


def test_check_env_raises_on_missing_devices():
    from prism.env import check_env

    with pytest.raises(RuntimeError, match=r"jax\[cuda12\]"):
        check_env(expected_gpus=2, require_gpu=True)


def test_for_kaggle_discovers_inputs(tmp_path):
    from prism.config import Config

    h5 = tmp_path / "datasets" / "someone" / "aspcap" / "apogee_dr17_parallel.h5"
    masks = tmp_path / "element-masks"
    h5.parent.mkdir(parents=True)
    masks.mkdir()
    h5.write_bytes(b"")
    (masks / "apogee_mask.npy").write_bytes(b"")
    cfg = Config.for_kaggle(root=tmp_path, batch_size=64)
    assert cfg.h5_path == h5 and cfg.mask_dir == masks and cfg.batch_size == 64


def test_for_kaggle_missing_inputs(tmp_path):
    from prism.config import Config

    with pytest.raises(FileNotFoundError, match="apogee_dr17_parallel.h5"):
        Config.for_kaggle(root=tmp_path)
