import json

import numpy as np
import pytest


@pytest.mark.slow
def test_end_to_end(tiny_config):
    from prism import stages

    cfg = tiny_config.replace(mcmc_overrides={"n_chains": 2, "n_warmup": 10, "n_samples": 10})
    prep = stages.prepare_data(cfg)
    assert (prep / "manifest.json").exists()

    for name in ("prism", "reg_payne", "payne"):
        ck = stages.train_forward(cfg, name, epochs=1)
        assert (ck / "meta.json").exists() and json.loads((ck / "meta.json").read_text())["name"] == name
        hist = json.loads((cfg.artifact_dir / f"history_{name}.json").read_text())
        assert hist
    ck = stages.train_cnn(cfg, epochs=1)
    assert (ck / "meta.json").exists()

    summary = stages.evaluate_models(cfg, ["prism", "payne"])
    assert set(summary["chi2"]) == {"prism", "payne"}
    assert all(np.isfinite(v["median_masked"]) for v in summary["chi2"].values())
    assert summary["leakage"]["prism"]["MG_FE"]["forbidden"] == 0.0
    assert (cfg.artifact_dir / "evaluation" / "summary.json").exists()
    assert (cfg.artifact_dir / "evaluation" / "leakage_comparison.png").exists()

    for method, n in (("map", 3), ("laplace", 3), ("nuts", 2)):
        path = stages.run_inverse(cfg, method, target_n=n, batch_stars=2)
        d = np.load(path)
        assert d["inferred_labels"].shape == (n, 23) and np.isfinite(d["inferred_labels"]).all()
        assert d["true_label_imputed"].shape == (n, 23)
        if method == "laplace":
            assert np.isfinite(d["inferred_errors"]).all()
        if method == "nuts":
            assert d["r_hat"].shape == (n, 23) and "lower_sigma" in d.files
        assert (path.parent / "figures" / "map_one_to_one.png").exists()

    # resume: a second call reuses the finished checkpoint and returns the same results
    again = np.load(stages.run_inverse(cfg, "map", target_n=3, batch_stars=2))
    assert np.array_equal(again["inferred_labels"], np.load(path.parent.parent / "map_two_stage" / "results.npz")["inferred_labels"])

    study = stages.run_optuna(cfg, "expanded_payne", n_trials=2, epochs=1)
    assert len(study.trials) == 2 and np.isfinite(study.best_value)
    study = stages.run_optuna(cfg, "reg_payne", n_trials=1, epochs=1)
    assert len(study.trials) == 1


def test_find_checkpoint_prefers_inputs(tmp_path, tiny_config):
    from prism.stages import find_checkpoint

    for root in (tmp_path / "inputs" / "prism", tiny_config.artifact_dir / "checkpoints" / "prism"):
        root.mkdir(parents=True)
        (root / "meta.json").write_text("{}")
    cfg = tiny_config.replace(checkpoint_inputs=tmp_path / "inputs")
    assert find_checkpoint(cfg, "prism") == tmp_path / "inputs" / "prism"
    assert find_checkpoint(tiny_config, "prism") == tiny_config.artifact_dir / "checkpoints" / "prism"
    with pytest.raises(FileNotFoundError):
        find_checkpoint(tiny_config, "payne")


def test_find_checkpoint_glob_over_kaggle_inputs(tmp_path, tiny_config):
    from prism.stages import find_checkpoint

    for nb, name in (("nb02", "prism"), ("nb04", "cnn")):
        d = tmp_path / nb / "prism_artifacts" / "checkpoints" / name
        d.mkdir(parents=True)
        (d / "meta.json").write_text("{}")
    cfg = tiny_config.replace(checkpoint_inputs=str(tmp_path / "*" / "prism_artifacts" / "checkpoints"))
    assert find_checkpoint(cfg, "prism") == tmp_path / "nb02" / "prism_artifacts" / "checkpoints" / "prism"
    assert find_checkpoint(cfg, "cnn") == tmp_path / "nb04" / "prism_artifacts" / "checkpoints" / "cnn"


def test_find_checkpoint_recursive_glob_for_nested_mounts(tmp_path, tiny_config):
    from prism.stages import find_checkpoint

    d = tmp_path / "notebooks" / "owner" / "nb02" / "prism_artifacts" / "checkpoints" / "prism"
    d.mkdir(parents=True)
    (d / "meta.json").write_text("{}")
    cfg = tiny_config.replace(checkpoint_inputs=str(tmp_path / "**" / "prism_artifacts" / "checkpoints"))
    assert find_checkpoint(cfg, "prism") == d
