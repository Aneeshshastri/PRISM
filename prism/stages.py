"""One function per pipeline stage — what the Kaggle notebooks call.

Outputs go under ``cfg.artifact_dir``:
``checkpoints/<model>/``, ``history_<model>.json``, ``evaluation/``,
``inverse/<method>_<mode>/{results.npz, checkpoint.npz, figures/}``, ``optuna/``.
"""

from __future__ import annotations

import dataclasses
import glob
import json
import time
from pathlib import Path

import jax
import numpy as np

from prism.checkpoint import load_model
from prism.config import Config
from prism.data.loader import DEFAULT_FIELDS, BatchLoader
from prism.data.masks import (element_indices, element_prior_weights, inference_element_masks,
                              line_mask_to_forbidden, load_apogee_mask)
from prism.data.prepare import load_stats, prepare_all
from prism.mesh import make_mesh, n_devices
from prism.models.registry import MODEL_NAMES, NEEDS_ELEMENT_IDX, build_model
from prism.models.regularised import REGPAYNE_PHASES
from prism.schedules import cosine_decay_restarts
from prism.train import Trainer, make_forward_loss, make_optimizer

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
    kw = dict(fields=fields, n_devices=nd, prefetch=cfg.prefetch, seed=cfg.seed)
    return BatchLoader(prep, "train", bs, shuffle=True, **kw), BatchLoader(prep, "val", bs, shuffle=False, **kw)


def _element_idx(cfg: Config) -> dict:
    return element_indices(load_apogee_mask(cfg.mask_dir), cfg.mask_threshold)


def _ckpt_dir(cfg: Config, name: str) -> Path:
    return Path(cfg.artifact_dir) / "checkpoints" / name


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=float))


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
    model = build_model(name, idx, seed=cfg.seed)
    schedule = cosine_decay_restarts(cfg.learning_rate, cfg.first_decay_steps, cfg.t_mul, cfg.m_mul, cfg.alpha)
    suppress = np.load(prep / "suppress.npy")
    train, val = _loaders(prep, cfg, mesh)
    ckpt = _ckpt_dir(cfg, name)
    meta = {"name": name, "seed": cfg.seed}
    total = epochs or cfg.epochs
    print(f"Training {name} on {n_devices(mesh)} device(s), global batch {_batch_size(cfg, mesh)}")
    if name == "reg_payne":
        forbidden = line_mask_to_forbidden(load_apogee_mask(cfg.mask_dir), cfg.mask_threshold)
        optimizer = make_optimizer(cfg, schedule)
        history = {}
        for phase, (phase_epochs, weight) in enumerate(REGPAYNE_PHASES, start=1):
            n_ep = min(phase_epochs or total, total)
            print(f"  RegPayne phase {phase}: penalty {weight:g}, {n_ep} epochs")
            trainer = Trainer(model, cfg, mesh, loss_fn=make_forward_loss(suppress, weight, forbidden),
                              optimizer=optimizer, ckpt_dir=ckpt, ckpt_meta=meta, monitor="val_spl")
            history[f"phase{phase}"] = trainer.fit(train, val, epochs=n_ep).history
            optimizer = trainer.optimizer  # Adam state carries over to phase 2
    else:
        trainer = Trainer(model, cfg, mesh, loss_fn=make_forward_loss(suppress), optimizer=make_optimizer(cfg, schedule),
                          ckpt_dir=ckpt, ckpt_meta=meta, element_idx=idx)
        history = trainer.fit(train, val, epochs=total).history
    _write_json(Path(cfg.artifact_dir) / f"history_{name}.json", history)
    return ckpt


def train_cnn(cfg: Config, mesh=None, *, epochs: int | None = None) -> Path:
    """Train the heteroscedastic CNN warm-start predictor."""
    from prism.models.cnn_predictor import make_cnn_loss

    mesh = _mesh(mesh)
    prep = prepare_all(cfg)
    c = cfg.replace(patience=CNN_TRAINING["patience"], min_delta=CNN_TRAINING["min_delta"],
                    learning_rate=CNN_TRAINING["learning_rate"])
    train, val = _loaders(prep, c, mesh, fields=("labels", "flux"))
    ckpt = _ckpt_dir(cfg, "cnn")
    trainer = Trainer(build_model("cnn", None, seed=cfg.seed), c, mesh, loss_fn=make_cnn_loss(),
                      optimizer=make_optimizer(c), plateau=CNN_TRAINING["plateau"], ckpt_dir=ckpt,
                      ckpt_meta={"name": "cnn", "seed": cfg.seed})
    history = trainer.fit(train, val, epochs=epochs or CNN_TRAINING["epochs"]).history
    _write_json(Path(cfg.artifact_dir) / "history_cnn.json", history)
    return ckpt


def evaluate_models(cfg: Config, names=MODEL_NAMES, mesh=None) -> dict:
    """Chi^2 tables, stellar populations, imputed-vs-natural and leakage for trained emulators."""
    from prism import evaluate as E

    mesh = _mesh(mesh)
    prep = prepare_all(cfg)
    test = {k: np.asarray(v) for k, v in _test_split(prep).items()}
    mean, std = load_stats(prep)
    am = load_apogee_mask(cfg.mask_dir)
    models = {n: load_model(find_checkpoint(cfg, n)) for n in names}
    table = E.chi2_table(models, test, am, mesh, cfg.mask_threshold)
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
        "leakage": {n: E.leakage(models[n], test["labels"][valid], test["bad_mask"][valid], am, std) for n in names},
    }
    E.binned_median_plots({"Teff": phys[valid, 0], "log_g": phys[valid, 1], "M_H": phys[valid, 2],
                           "SNR": test["snr"][valid]}, masked, out_dir)
    E.leakage_plot(summary["leakage"], out_dir)
    _write_json(out_dir / "summary.json", summary)
    for n in names:
        c = summary["chi2"][n]
        print(f"  {n:<15} median chi2 {c['median']:.3f}   masked {c['median_masked']:.3f}")
    return summary


def _mcmc_settings(cfg: Config, method: str):
    from prism.inverse import mcmc as M

    core, elem, joint = ((M.NUTS_CORE, M.NUTS_ELEM, M.NUTS_JOINT) if method == "nuts"
                         else (M.HMC_CORE, M.HMC_ELEM, dataclasses.replace(M.HMC_CORE, n_chains=6)))
    if cfg.mcmc_overrides:
        core, elem, joint = (dataclasses.replace(s, **cfg.mcmc_overrides) for s in (core, elem, joint))
    return core, elem, joint


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
    for start in range(len(results), len(rows_all), batch_stars):
        rows = rows_all[start:start + batch_stars]
        flux = np.asarray(test["flux"][rows], np.float32)
        ivar = np.asarray(test["ivar"][rows], np.float32)
        mu, sd = predict_physical(cnn, flux, mean[:23], std[:23])
        t0 = time.time()
        extra = {}
        if method in ("map", "laplace"):
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
            model = Payne(depth, width, rngs=nnx.Rngs(cfg.seed))
            model.prism_name = "expanded_payne"
            t = Trainer(model, tcfg, mesh, loss_fn=make_forward_loss(suppress), optimizer=make_optimizer(tcfg),
                        monitor="val_spl")
            return fit(t, epochs or 35, trial)
        p1 = trial.suggest_float("penalty_1", 1e-2, 1e2, log=True)
        p2 = trial.suggest_float("penalty_2", 50.0, 2000.0, log=True)
        model = Payne(2, 300, rngs=nnx.Rngs(cfg.seed))
        model.prism_name = "reg_payne"
        t1 = Trainer(model, tcfg, mesh, loss_fn=make_forward_loss(suppress, p1, forbidden),
                     optimizer=make_optimizer(tcfg), monitor="val_spl")
        fit(t1, min(5, epochs or 5))
        t2 = Trainer(model, tcfg, mesh, loss_fn=make_forward_loss(suppress, p2, forbidden),
                     optimizer=make_optimizer(tcfg), monitor="val_spl")
        return fit(t2, epochs or 30, trial)

    s = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
                            sampler=optuna.samplers.TPESampler(seed=cfg.seed))
    s.optimize(objective, n_trials=n_trials)
    out = Path(cfg.artifact_dir) / "optuna" / f"{study}_study.pkl"
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(s, out)
    print(f"Best {study}: {s.best_params} (val SPL {s.best_value:.6g})")
    return s
