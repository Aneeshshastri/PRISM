"""One function per pipeline stage — what the Kaggle notebooks call.

Outputs go under ``cfg.artifact_dir``:
``checkpoints/<model>/``, ``history_<model>.json``, ``evaluation/``,
``inverse/<method>_<mode>/{results.npz, checkpoint.npz, figures/}``, ``optuna/``.
"""

from __future__ import annotations

import dataclasses
import functools
import glob
import json
import time
from pathlib import Path

import jax
import numpy as np

from flax import nnx

from prism.checkpoint import load_meta, load_model
from prism.config import Config
from prism.data.loader import DEFAULT_FIELDS, BatchLoader
from prism.data.masks import (element_indices, element_prior_weights, gap_mask, inference_element_masks,
                              line_mask_to_forbidden, load_apogee_mask, load_cno_masks)
from prism.data.prepare import load_stats, prepare_all
from prism.mesh import make_mesh, n_devices
from prism.precision import compute_dtype, inference_precision
from prism.models.registry import LEGACY_ARCH, MODEL_NAMES, NEEDS_ELEMENT_IDX, build_model
from prism.models.regularised import REGPAYNE_PHASES
from prism.train import Trainer, make_forward_loss, make_optimizer, make_schedule

CNN_TRAINING = {"epochs": 80, "patience": 15, "min_delta": 1e-6, "learning_rate": 1e-3,
                "plateau": {"patience": 5, "factor": 0.5, "min_lr": 1e-6}}
INVERSE_METHODS = ("map", "laplace", "nuts", "hmc")


# ----------------------------------------------------------------- helpers
def _mesh(mesh):
    return mesh if mesh is not None else make_mesh()


def _batch_size(cfg: Config, mesh) -> int:
    nd = n_devices(mesh)
    return -(-cfg.batch_size // nd) * nd


def _loaders(prep: Path, cfg: Config, mesh, fields=DEFAULT_FIELDS):
    nd, bs = n_devices(mesh), _batch_size(cfg, mesh)
    kw = dict(fields=fields, n_devices=nd, prefetch=cfg.prefetch, seed=cfg.seed, in_memory=cfg.load_in_memory)
    return BatchLoader(prep, "train", bs, shuffle=True, **kw), BatchLoader(prep, "val", bs, shuffle=False, **kw)


def _element_idx(cfg: Config) -> dict:
    """Model routing windows: the atomic line windows, plus the C/N/O union when ``cfg.cno_split``."""
    cno = load_cno_masks(cfg.mask_dir, cfg.mask_threshold) if cfg.cno_split else None
    return element_indices(load_apogee_mask(cfg.mask_dir), cfg.mask_threshold, cno_masks=cno)


def _ckpt_dir(cfg: Config, name: str) -> Path:
    return Path(cfg.artifact_dir) / "checkpoints" / name


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=float))


def _full_fp32(fn):
    """Run an inference stage with true fp32 matmuls on every GPU (no silent TF32)."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with inference_precision():
            return fn(*args, **kwargs)
    return wrapper


def find_checkpoint(cfg: Config, name: str) -> Path:
    """Checkpoint dir for ``name``: ``cfg.checkpoint_inputs`` first, then this run's artifacts."""
    candidates = []
    if cfg.checkpoint_inputs is not None:
        pattern = str(cfg.checkpoint_inputs)
        bases = sorted(Path(p) for p in glob.glob(pattern, recursive=True)) if "*" in pattern else [Path(pattern)]
        for base in bases:
            candidates += [base / name, base / "checkpoints" / name]
    candidates.append(_ckpt_dir(cfg, name))
    for c in candidates:
        if (c / "meta.json").exists():
            return c
    raise FileNotFoundError(f"no checkpoint for {name!r}; looked in: " + ", ".join(map(str, candidates)))


def _test_split(prep: Path) -> dict:
    d = prep / "test"
    return {k: np.load(d / f"{k}.npy", mmap_mode="r") for k in
            ("labels", "bad_mask", "flux", "ivar", "label_err", "snr", "global_idx")}


# ----------------------------------------------------------------- stages
def prepare_data(cfg: Config) -> Path:
    return prepare_all(cfg)


def train_forward(cfg: Config, name: str, mesh=None, *, epochs: int | None = None) -> Path:
    """Train one forward emulator (``MODEL_NAMES``); returns its checkpoint directory."""
    if name not in MODEL_NAMES:
        raise ValueError(f"unknown model {name!r}; choose from {MODEL_NAMES}")
    mesh = _mesh(mesh)
    prep = prepare_all(cfg)
    idx = _element_idx(cfg) if name in NEEDS_ELEMENT_IDX else None
    model = build_model(name, idx, seed=cfg.seed, dtype=compute_dtype(cfg.train_precision), arch=cfg.arch())
    suppress = np.load(prep / "suppress.npy")
    train, val = _loaders(prep, cfg, mesh)
    ckpt = _ckpt_dir(cfg, name)
    meta = {"name": name, "seed": cfg.seed}
    total = epochs or cfg.epochs
    phases = ([min(e or total, total) for e, _ in REGPAYNE_PHASES] if name == "reg_payne" else [total])
    start_epoch, best_val = _resume_point(cfg, name, ckpt, model)
    # one decay over every phase; a resumed run re-warms from where the checkpoint left off
    schedule = make_schedule(cfg, len(train), sum(phases), resume_step=start_epoch * len(train))
    print(f"  architecture {cfg.arch()}, loss scale {cfg.loss_scale:g}, schedule {cfg.lr_schedule}")
    print(f"Training {name} on {n_devices(mesh)} device(s), global batch {_batch_size(cfg, mesh)}")
    if name == "reg_payne":
        forbidden = line_mask_to_forbidden(load_apogee_mask(cfg.mask_dir), cfg.mask_threshold)
        optimizer = make_optimizer(cfg, schedule)
        history = {}
        for phase, ((_, weight), n_ep) in enumerate(zip(REGPAYNE_PHASES, phases), start=1):
            print(f"  RegPayne phase {phase}: penalty {weight:g}, {n_ep} epochs")
            trainer = Trainer(model, cfg, mesh, loss_fn=make_forward_loss(suppress, weight, forbidden,
                                                                          scale=cfg.loss_scale),
                              optimizer=optimizer, ckpt_dir=ckpt, ckpt_meta=meta, monitor="val_spl",
                              precision=cfg.train_precision)
            history[f"phase{phase}"] = trainer.fit(train, val, epochs=n_ep).history
            optimizer = trainer.optimizer  # Adam state carries over to phase 2
    else:
        trainer = Trainer(model, cfg, mesh, loss_fn=make_forward_loss(suppress, scale=cfg.loss_scale),
                          optimizer=make_optimizer(cfg, schedule),
                          ckpt_dir=ckpt, ckpt_meta=meta, element_idx=idx, precision=cfg.train_precision)
        history = trainer.fit(train, val, epochs=total, start_epoch=start_epoch, best_val=best_val).history
        if start_epoch:
            history = {"resumed_at_epoch": start_epoch, **history}
    _write_json(Path(cfg.artifact_dir) / f"history_{name}.json", history)
    return ckpt


def _resume_point(cfg: Config, name: str, ckpt: Path, model) -> tuple[int, float]:
    """``cfg.resume``: load ``ckpt``'s best weights into ``model`` and return (next epoch, best
    monitored value); (0, inf) when not resuming or there is no checkpoint yet. Only the weights
    are saved, so Adam restarts from zero moments (make_schedule re-warms the LR)."""
    if not cfg.resume:
        return 0, float("inf")
    if name == "reg_payne":
        raise ValueError("resume is not supported for reg_payne (two penalty phases); retrain it")
    if not (ckpt / "meta.json").exists():
        print(f"  resume requested but no checkpoint at {ckpt}; training from scratch")
        return 0, float("inf")
    meta = load_meta(ckpt)
    saved_arch = meta.get("arch", dict(LEGACY_ARCH))
    if meta.get("name") != name or saved_arch != cfg.arch() or meta.get("seed") != cfg.seed:
        raise ValueError(f"checkpoint {ckpt} (name {meta.get('name')}, arch {saved_arch}, seed {meta.get('seed')}) "
                         f"does not match this run (name {name}, arch {cfg.arch()}, seed {cfg.seed})")
    nnx.update(model, nnx.state(load_model(ckpt)))
    return int(meta["epoch"]) + 1, float(meta["val_loss"])


def train_cnn(cfg: Config, mesh=None, *, epochs: int | None = None) -> Path:
    """Train the heteroscedastic CNN warm-start predictor."""
    from prism.models.cnn_predictor import make_cnn_loss

    mesh = _mesh(mesh)
    prep = prepare_all(cfg)
    c = cfg.replace(patience=CNN_TRAINING["patience"], min_delta=CNN_TRAINING["min_delta"], rel_min_delta=0.0,
                    learning_rate=CNN_TRAINING["learning_rate"])
    train, val = _loaders(prep, c, mesh, fields=("labels", "flux"))
    ckpt = _ckpt_dir(cfg, "cnn")
    cnn = build_model("cnn", None, seed=cfg.seed, dtype=compute_dtype(cfg.train_precision))
    trainer = Trainer(cnn, c, mesh, loss_fn=make_cnn_loss(),
                      optimizer=make_optimizer(c), plateau=CNN_TRAINING["plateau"], ckpt_dir=ckpt,
                      ckpt_meta={"name": "cnn", "seed": cfg.seed}, precision=cfg.train_precision)
    history = trainer.fit(train, val, epochs=epochs or CNN_TRAINING["epochs"]).history
    _write_json(Path(cfg.artifact_dir) / "history_cnn.json", history)
    return ckpt


@_full_fp32
def evaluate_models(cfg: Config, names=MODEL_NAMES, mesh=None) -> dict:
    """Chi^2 tables, stellar populations, imputed-vs-natural and leakage for trained emulators."""
    from prism import evaluate as E

    mesh = _mesh(mesh)
    prep = prepare_all(cfg)
    test = {k: np.asarray(v) for k, v in _test_split(prep).items()}
    mean, std = load_stats(prep)
    am = load_apogee_mask(cfg.mask_dir)
    cno = load_cno_masks(cfg.mask_dir, cfg.mask_threshold)
    cno_union = cno["C_FE"] | cno["N_FE"] | cno["O_FE"]
    regions = E.spectral_regions(am, cno, cfg.mask_threshold)
    models = {n: load_model(find_checkpoint(cfg, n)) for n in names}
    table = E.chi2_table(models, test, am, mesh, cfg.mask_threshold, regions=regions)
    valid = table["valid_stars"]
    phys = test["labels"][:, :23] * std[:23] + mean[:23]
    masked = {n: table[n]["chi2_masked"] for n in names}
    out_dir = Path(cfg.artifact_dir) / "evaluation"

    def stats(a):
        return {"mean": float(np.mean(a)), "median": float(np.median(a)), "p95": float(np.percentile(a, 95))}

    summary = {
        "n_test": int(len(valid)), "n_retained": int(valid.sum()),
        "chi2": {n: {**{f"{k}": v for k, v in stats(table[n]["chi2"]).items()},
                     **{f"{k}_masked": v for k, v in stats(masked[n]).items()}} for n in names},
        "populations": E.population_table(masked, phys, valid),
        "imputed_vs_natural": {n: E.imputed_vs_natural(table[n]["chi2"], test["bad_mask"]) for n in names},
        "leakage": {n: E.leakage(models[n], test["labels"][valid], test["bad_mask"][valid], am, std,
                                 cno_window=cno_union) for n in names},
        "wrong_sign": {n: E.wrong_sign_fraction(models[n], test["labels"][valid], test["bad_mask"][valid], am)
                       for n in names},
        "region_chi2": E.region_chi2_summary({n: table[n]["region_chi2"] for n in names}, list(regions),
                                             phys[valid, 0], phys[valid, 1]),
        "region_pixels": {r: int(m.sum()) for r, m in regions.items()},
        "architectures": {n: getattr(models[n], "prism_arch", None) for n in names},
    }
    E.binned_median_plots({"Teff": phys[valid, 0], "log_g": phys[valid, 1], "M_H": phys[valid, 2],
                           "SNR": test["snr"][valid]}, masked, out_dir)
    E.leakage_plot(summary["leakage"], out_dir)
    _write_json(out_dir / "summary.json", summary)
    for n in names:
        c = summary["chi2"][n]
        print(f"  {n:<15} median chi2 {c['median']:.3f}   masked {c['median_masked']:.3f}")
    print("  median chi2 per region (all stars | Teff<4000 giants):")
    for n in names:
        a, cg = summary["region_chi2"]["all"][n], summary["region_chi2"]["Teff<4000 giants"][n]
        print(f"  {n:<15} " + "  ".join(f"{r} {a[r]:.3f} | {cg[r]:.3f}" for r in regions))
    return summary


@_full_fp32
def analyse_entanglement(cfg: Config, names=MODEL_NAMES, residual_models=("prism",), n_perm: int = 20,
                         mesh=None) -> dict:
    """Physical vs non-physical entanglement tests (prism.entanglement) -> evaluation/entanglement.json.

    Test 1 (residual test) runs on ``residual_models`` (PRISM: does forcing zero off-window
    response leave real signal in its residuals?); test 2 (proxy test) on every model in ``names``.
    """
    from prism import entanglement as E

    mesh = _mesh(mesh)
    prep = prepare_all(cfg)
    test = {k: np.asarray(v) for k, v in _test_split(prep).items()}
    mean, std = load_stats(prep)
    am = load_apogee_mask(cfg.mask_dir)
    models = {n: load_model(find_checkpoint(cfg, n)) for n in names}
    out = {"residual_test": {}, "proxy_test": None}
    for n in residual_models:
        if n in models:
            print(f"Residual test (test 1) for {n}")
            out["residual_test"] = E.residual_test(models[n], test, mean, std, am, mesh, n_perm=n_perm,
                                                    seed=cfg.seed)
    train_phys = np.load(prep / "train" / "labels.npy", mmap_mode="r")[:, :23] * std[:23] + mean[:23]
    train_bad = np.load(prep / "train" / "bad_mask.npy")
    print("Proxy test (test 2)")
    out["proxy_test"] = E.proxy_test(models, test, train_phys, train_bad, mean, std, am, seed=cfg.seed)
    E.write_json(Path(cfg.artifact_dir) / "evaluation" / "entanglement.json", out)
    return out


def _mcmc_settings(cfg: Config, method: str):
    from prism.inverse import mcmc as M

    core, elem, joint = ((M.NUTS_CORE, M.NUTS_ELEM, M.NUTS_JOINT) if method == "nuts"
                         else (M.HMC_CORE, M.HMC_ELEM, dataclasses.replace(M.HMC_CORE, n_chains=6)))
    if cfg.mcmc_overrides:
        core, elem, joint = (dataclasses.replace(s, **cfg.mcmc_overrides) for s in (core, elem, joint))
    return core, elem, joint


@_full_fp32
def run_inverse(cfg: Config, method: str, *, mode: str = "two_stage", target_n: int | None = None,
                batch_stars: int | None = None, model_name: str = "prism", mesh=None) -> Path:
    """Invert test-set spectra with MAP / MAP+Laplace / NUTS / HMC; returns the results.npz path."""
    from prism.analysis import load_results, report, visualise
    from prism.inverse.cnn_prior import predict_physical
    from prism.inverse.common import InverseResults, ResultsCheckpoint, make_flux_fn, select_test_stars
    from prism.inverse.laplace import laplace_sigmas
    from prism.inverse.map import run_map
    from prism.inverse.mcmc import run_mcmc

    if method not in INVERSE_METHODS:
        raise ValueError(f"unknown method {method!r}; choose from {INVERSE_METHODS}")
    mesh = _mesh(mesh)
    nd = n_devices(mesh)
    if batch_stars is None:  # MAP/Laplace are cheap per star; MCMC holds chains x samples per star
        batch_stars = 256 if method in ("map", "laplace") else 32
    batch_stars = -(-batch_stars // nd) * nd
    prep = prepare_all(cfg)
    test = _test_split(prep)
    mean, std = load_stats(prep)
    phys = np.asarray(test["labels"][:, :23]) * std[:23] + mean[:23]
    rows_all = select_test_stars(test["snr"], phys[:, 0], phys[:, 1], phys[:, 2], target_n, seed=cfg.seed)
    gidx = np.asarray(test["global_idx"])[rows_all]

    emulator = load_model(find_checkpoint(cfg, model_name))
    cnn = load_model(find_checkpoint(cfg, "cnn"))
    fn, state = make_flux_fn(emulator)
    masks, skip = inference_element_masks(load_apogee_mask(cfg.mask_dir), cfg.mask_threshold)
    weights = element_prior_weights(masks)
    stats = (mean, std)

    out_dir = Path(cfg.artifact_dir) / "inverse" / f"{method}_{mode}"
    ck = ResultsCheckpoint(out_dir / "checkpoint.npz", gidx)
    results = ck.load() or InverseResults.empty()
    print(f"Inverting {len(rows_all)} stars with {method} ({mode}) on {nd} device(s); "
          f"{len(results)} already done")
    key = jax.random.key(cfg.seed)
    prior_map = None  # Laplace = MAP + Hessians: reuse a finished MAP run on the same stars
    if method == "laplace":
        map_path = Path(cfg.artifact_dir) / "inverse" / f"map_{mode}" / "results.npz"
        if map_path.exists():
            prev = InverseResults.load(map_path)
            if np.array_equal(np.asarray(prev.global_indices), gidx):
                prior_map = prev
                print(f"  reusing MAP solutions from {map_path}")
    for start in range(len(results), len(rows_all), batch_stars):
        rows = rows_all[start:start + batch_stars]
        flux = np.asarray(test["flux"][rows], np.float32)
        ivar = np.asarray(test["ivar"][rows], np.float32)
        mu, sd = predict_physical(cnn, flux, mean[:23], std[:23])
        t0 = time.time()
        extra = {}
        if method in ("map", "laplace"):
            if prior_map is not None:
                sl = slice(start, start + len(rows))
                out = {"theta": np.asarray(prior_map.inferred_labels)[sl],
                       "core_loss": np.asarray(prior_map.core_loss)[sl],
                       "elem_loss": np.asarray(prior_map.elem_loss)[sl]}
            else:
                out = run_map(fn, state, flux, ivar, mu, sd, masks, skip, weights, stats, mesh, mode=mode)
            theta = out["theta"]
            extra.update(core_loss=out["core_loss"], elem_loss=out["elem_loss"])
            if method == "laplace":
                extra["inferred_errors"] = laplace_sigmas(fn, state, theta, flux, ivar, mu, sd, masks, skip,
                                                          weights, stats, mesh, mode=mode)
        else:
            core, elem, joint = _mcmc_settings(cfg, method)
            out = run_mcmc(fn, state, flux, ivar, mu, sd, masks, skip, stats, mesh, jax.random.fold_in(key, start),
                           mode=mode, core=core, elem=elem, joint=joint)
            theta = out["median"]
            extra.update(lower_sigma=out["lower_sigma"], upper_sigma=out["upper_sigma"], r_hat=out["r_hat"],
                         ess=out["ess"], acceptance_rates=out["acceptance"])
        dt = time.time() - t0
        part = InverseResults(global_indices=gidx[start:start + len(rows)], true_labels=phys[rows],
                              inferred_labels=theta, aspcap_errors=np.asarray(test["label_err"][rows]),
                              true_label_imputed=np.asarray(test["bad_mask"][rows]),
                              wall_seconds=np.full(len(rows), dt / len(rows)), **extra)
        results = results.append(part)
        ck.save(results)
        print(f"  [{len(results):>6}/{len(rows_all)}] {dt:.1f}s for {len(rows)} stars")

    path = out_dir / "results.npz"
    results.save(path)
    data = load_results(path)
    report(data)
    visualise(data, save_dir=out_dir / "figures", close=True)
    return path


def run_optuna(cfg: Config, study: str, n_trials: int = 30, epochs: int | None = None, mesh=None):
    """Optuna search (optuna-tuner.ipynb): Expanded-Payne size or RegPayne penalties.

    The objective is the minimum validation masked-SPL WITHOUT the penalty term.
    """
    import joblib
    import optuna
    from flax import nnx

    from prism.models.payne import Payne

    if study not in ("expanded_payne", "reg_payne"):
        raise ValueError("study must be 'expanded_payne' or 'reg_payne'")
    mesh = _mesh(mesh)
    prep = prepare_all(cfg)
    suppress = np.load(prep / "suppress.npy")
    train, val = _loaders(prep, cfg, mesh)
    tcfg = cfg.replace(patience=10)
    forbidden = line_mask_to_forbidden(load_apogee_mask(cfg.mask_dir), cfg.mask_threshold)

    def fit(trainer, n_ep, trial=None):
        pruned = {"flag": False}

        def cb(epoch, logs):
            if trial is None:
                return False
            trial.report(logs["val_spl"], epoch)
            pruned["flag"] = trial.should_prune()
            return pruned["flag"]

        res = trainer.fit(train, val, epochs=n_ep, on_epoch_end=cb)
        if pruned["flag"]:
            raise optuna.TrialPruned()
        return min(res.history["val_spl"])

    def objective(trial):
        if study == "expanded_payne":
            depth, width = trial.suggest_int("depth", 1, 5), trial.suggest_int("width", 200, 1500)
            model = Payne(depth, width, drop_gaps=cfg.drop_gaps, dtype=compute_dtype(cfg.train_precision),
                          rngs=nnx.Rngs(cfg.seed))
            model.prism_name = "expanded_payne"
            t = Trainer(model, tcfg, mesh, loss_fn=make_forward_loss(suppress, scale=cfg.loss_scale), optimizer=make_optimizer(tcfg),
                        monitor="val_spl", precision=cfg.train_precision)
            return fit(t, epochs or 35, trial)
        p1 = trial.suggest_float("penalty_1", 1e-2, 1e2, log=True)
        p2 = trial.suggest_float("penalty_2", 50.0, 2000.0, log=True)
        model = Payne(2, 300, drop_gaps=cfg.drop_gaps, dtype=compute_dtype(cfg.train_precision),
                      rngs=nnx.Rngs(cfg.seed))
        model.prism_name = "reg_payne"
        t1 = Trainer(model, tcfg, mesh, loss_fn=make_forward_loss(suppress, p1, forbidden, scale=cfg.loss_scale),
                     optimizer=make_optimizer(tcfg), monitor="val_spl", precision=cfg.train_precision)
        fit(t1, min(5, epochs or 5))
        t2 = Trainer(model, tcfg, mesh, loss_fn=make_forward_loss(suppress, p2, forbidden, scale=cfg.loss_scale),
                     optimizer=make_optimizer(tcfg), monitor="val_spl", precision=cfg.train_precision)
        return fit(t2, epochs or 30, trial)

    s = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
                            sampler=optuna.samplers.TPESampler(seed=cfg.seed))
    s.optimize(objective, n_trials=n_trials)
    out = Path(cfg.artifact_dir) / "optuna" / f"{study}_study.pkl"
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(s, out)
    print(f"Best {study}: {s.best_params} (val SPL {s.best_value:.6g})")
    return s


# ----------------------------------------------------------------- whole pipeline
PIPELINE_STAGES = ("prepare", *MODEL_NAMES, "cnn", "evaluate", "entanglement", "map", "laplace", "nuts", "hmc",
                   "optuna")
DEFAULT_STAGES = tuple(s for s in PIPELINE_STAGES if s != "optuna")
SMOKE_MCMC = {"n_chains": 2, "n_warmup": 20, "n_samples": 20}


def _available_models(cfg: Config) -> list[str]:
    found = []
    for name in MODEL_NAMES:
        try:
            find_checkpoint(cfg, name)
            found.append(name)
        except FileNotFoundError:
            pass
    if not found:
        raise FileNotFoundError("no trained forward-model checkpoints to evaluate")
    return found


def _run_stage(cfg: Config, stage: str, smoke: bool):
    ep = 1 if smoke else None
    if stage == "prepare":
        return str(prepare_data(cfg))
    if stage in MODEL_NAMES:
        return str(train_forward(cfg, stage, epochs=ep))
    if stage == "cnn":
        return str(train_cnn(cfg, epochs=ep))
    if stage == "evaluate":
        summary = evaluate_models(cfg, names=_available_models(cfg))
        return {n: c["median_masked"] for n, c in summary["chi2"].items()}
    if stage == "entanglement":
        res = analyse_entanglement(cfg, names=_available_models(cfg), n_perm=3 if smoke else 20)
        return {e: v.get("forbidden/all", {}).get("perm_z") for e, v in res["residual_test"].items()}
    if stage in ("map", "laplace"):
        return str(run_inverse(cfg, stage, target_n=8 if smoke else None))
    if stage in ("nuts", "hmc"):
        c = cfg.replace(mcmc_overrides=SMOKE_MCMC) if smoke else cfg
        n = 2 if smoke else 10
        return str(run_inverse(c, stage, target_n=n, batch_stars=n))
    if stage == "optuna":
        return {s: run_optuna(cfg, s, n_trials=1 if smoke else 30, epochs=ep).best_value
                for s in ("expanded_payne", "reg_payne")}
    raise ValueError(stage)


def run_pipeline(cfg: Config, stages=DEFAULT_STAGES, *, smoke: bool = False) -> dict:
    """Run stages in order, logging wall time and status to ``artifact_dir/run_log.json``.

    ``smoke=True`` runs every stage minimally (1 epoch, a few stars, short chains) to
    validate a new machine or dataset. A failing stage is logged and re-raised: later
    stages depend on earlier ones.
    """
    import traceback
    from datetime import datetime

    stages = list(stages)
    unknown = [s for s in stages if s not in PIPELINE_STAGES]
    if unknown:
        raise ValueError(f"unknown stage(s) {unknown}; choose from {PIPELINE_STAGES}")
    log = {"smoke": smoke, "started": datetime.now().isoformat(timespec="seconds"),
           "train_precision": cfg.train_precision, "stages": []}
    path = Path(cfg.artifact_dir) / "run_log.json"
    for stage in stages:
        print(f"\n===== stage {stage}{' (smoke)' if smoke else ''} =====", flush=True)
        t0 = time.time()
        entry = {"stage": stage}
        try:
            entry["result"] = _run_stage(cfg, stage, smoke)
            entry["status"] = "ok"
        except Exception as exc:
            entry.update(status="failed", error="".join(traceback.format_exception(exc)))
            raise
        finally:
            entry["seconds"] = round(time.time() - t0, 2)
            log["stages"].append(entry)
            _write_json(path, log)
            print(f"===== {stage}: {entry['status']} in {entry['seconds']:.1f}s =====", flush=True)
    return log
