# PRISM

PRISM is a research codebase for learning and inverting high-resolution stellar spectra from APOGEE DR17. The core idea is a physics-structured neural emulator that predicts optical depth, separates continuum, molecular and elemental contributions, and uses element-specific spectral windows so that each abundance can only change the flux at its own lines.

The code is a tested **JAX + Flax NNX** package (`prism/`) driven by thin Kaggle notebooks (`notebooks/`). Training and inversion run data-parallel over a 1-D device mesh, so they use both GPUs of Kaggle's **T4 x2** accelerator. The original TensorFlow/Keras notebooks are kept in [`legacy_tf/`](legacy_tf) as a reference.

## Scientific summary

PRISM is trained directly on real APOGEE DR17 spectra (ASPCAP labels as supervision) and predicts flux through a structured optical-depth decomposition:

1. a continuum branch (low-order Chebyshev series),
2. a molecular branch for the blended C/N/O opacity,
3. a thermodynamic-state branch that compresses the global labels into a 16-d state vector,
4. sparse atomic experts, one per element, that see only the state vector and their own abundance and write optical depth only inside that element's line windows.

Flux follows the Beer-Lambert law, `F(lambda) = k(lambda) * exp(-tau(lambda))`.

Labels (always in this order): `TEFF LOGG M_H VMICRO VMACRO VSINI C_FE N_FE O_FE FE_H MG_FE SI_FE CA_FE TI_FE S_FE AL_FE MN_FE NI_FE CR_FE K_FE NA_FE V_FE CO_FE`, plus four engineered inputs: `INV_TEFF = 5040/TEFF`, `vbroad = sqrt(VMACRO^2 + VSINI^2)`, `C_O_diff = C_FE - O_FE`, `LOGPE = 0.5 LOGG + 0.5 M_H`.

> **Abundance convention:** although the columns keep their ASPCAP names, every `*_FE` label is converted from [X/Fe] to **[X/H]** during preprocessing (by adding `FE_H`). The emulators, the CNN warm start, the inverse pipelines and their result files all work in [X/H], and the plots label them that way. Only the ASPCAP `*_ERR` columns used as reference uncertainties are still [X/Fe] errors.

Splits by HDF5 row: train `[0, 120000)`, validation `[120000, 140000)`, test `[140000, 149500)`. The label imputer and the normalisation statistics are fitted on the training split only.

## Pipeline

| Stage | Function (`prism.stages`) | Notebook | Output (`/kaggle/working/prism_artifacts/...`) |
| --- | --- | --- | --- |
| Data cache | `prepare_data` | `01_prepare_data` | `stats.npz`, `imputer.pkl`, `manifest.json` (large arrays in `/tmp/prism_data`) |
| PRISM + ablation | `train_forward(cfg, "prism" / "ablation")` | `02_train_prism` | `checkpoints/<model>/`, `history_<model>.json` |
| Baselines | `train_forward(cfg, "payne" / "expanded_payne" / "reg_payne")` | `03_train_baselines` | same |
| CNN warm start | `train_cnn` | `04_train_cnn` | `checkpoints/cnn/` |
| Evaluation | `evaluate_models` | `05_evaluate` | `evaluation/summary.json`, chi-squared and leakage figures |
| MAP / Laplace | `run_inverse(cfg, "map" / "laplace")` | `06_inverse_map_laplace` | `inverse/<method>_<mode>/results.npz`, `figures/` |
| NUTS / HMC | `run_inverse(cfg, "nuts" / "hmc")` | `07_inverse_mcmc` | same, with per-star R-hat and ESS |
| Hyper-parameters | `run_optuna` | `08_optuna` | `optuna/<study>_study.pkl` |

Inverse results can be re-analysed anywhere with `python -m prism.analysis --results results.npz --save-dir figures`.

## Running on Kaggle (T4 x2)

0. **Publish the code**: the install cell installs this package from GitHub. Commit and push the branch that holds `prism/` and `pyproject.toml`, then set `PRISM_REF` in each notebook's first cell to that branch (default `"main"`, i.e. after merging).
1. **Create a notebook**: on Kaggle, *Create -> New Notebook*, then *File -> Import Notebook* and import one of the files in [`notebooks/`](notebooks) (from GitHub or by upload). Start with `02_train_prism.ipynb`.
2. **Settings** (right-hand panel): *Accelerator* = **GPU T4 x2**, *Internet* = **On** (needed for the install cell).
3. **Add inputs** (*Add Input*): the `aspcapstar-dr17-150kstars` dataset (`apogee_dr17_parallel.h5`) and the `element-masks` dataset (`apogee_mask.npy`, `C.filt`, `N.filt`, `O.filt`). `Config.for_kaggle()` finds them under `/kaggle/input` whatever the mount layout.
4. **Run all.**
   - The first cell installs this package from GitHub at `PRISM_REF` while pinning Kaggle's preinstalled JAX, Flax and Orbax, so their versions are never changed.
   - The `check_env()` cell must report **2 GPUs** with compute capability **7.5**. If it raises, it prints the fix: install `jax[cuda12]` at the already-installed JAX version (the CUDA-13 JAX wheels need NVIDIA driver >= 580), then restart the session.
5. **Save the outputs**: use *Save Version -> Save & Run All* so `/kaggle/working/prism_artifacts` is kept as the notebook's output.
6. **Chain the stages**: notebooks 05-07 need checkpoints from earlier runs. Add the earlier notebooks' outputs as inputs (*Add Input -> Your Work -> Notebook output*): 02 and 03 for evaluation, 02 and 04 for inversion. Their `Config.for_kaggle(checkpoint_inputs="/kaggle/input/**/prism_artifacts/checkpoints")` line picks up every attached checkpoint, however deeply Kaggle mounts it.
7. **Order**: `01 -> 02 -> 03 -> 04 -> 05 -> 06 -> 07` (08 optional; 01 is optional because every notebook rebuilds the cache itself).

If Internet must stay off, attach a copy of this repository as a Kaggle dataset and replace the URL with its path, e.g. `pip install -c /tmp/kaggle-pins.txt /kaggle/input/prism-repo`.

## Local development

```bash
python3 -m venv .venv
printf "jax==0.11.1\njaxlib==0.11.1\nflax==0.12.12\norbax-checkpoint==0.12.8\n" > /tmp/pins.txt  # Kaggle's versions
.venv/bin/pip install -c /tmp/pins.txt -e ".[test]"
.venv/bin/pytest              # fast suite; fakes a 2-device mesh on CPU
.venv/bin/pytest -m slow      # end-to-end stages + converged MCMC (several minutes)
```

`tests/conftest.py` sets `XLA_FLAGS=--xla_force_host_platform_device_count=2`, so every sharding path is exercised on CPU exactly as it runs on the two T4s.

## Package map

| Module | Role |
| --- | --- |
| `prism/config.py` | Label constants, `Config` (paths, splits, hyper-parameters), `Config.for_kaggle()` |
| `prism/env.py` | `check_env()`: versions, devices, driver check, sharded smoke test |
| `prism/mesh.py` | 1-D `('data',)` mesh, batch padding and sharding helpers |
| `prism/data/` | ASPCAP flags + train-only imputation, masks, HDF5 -> numpy cache, prefetching loader |
| `prism/models/` | `Prism`, `PrismAblation`, `Payne`, `HeteroscedasticCNNPredictor`, RegPayne Jacobian penalty, registry |
| `prism/losses.py`, `prism/train.py`, `prism/checkpoint.py` | Masked spectral loss, sharded NNX trainer (early stopping, LR plateau, Orbax checkpoints) |
| `prism/evaluate.py` | Reduced chi-squared (plain and imputation-masked), stellar populations, gradient leakage |
| `prism/inverse/` | Shared objectives and bounds, CNN prior, batched MAP, Laplace, BlackJAX NUTS/HMC |
| `prism/analysis.py` | Result tables, calibration, outlier diagnostics and figures |
| `prism/stages.py` | One function per pipeline stage (what the notebooks call) |

## Data assets in the repository

| Path | Purpose |
| --- | --- |
| `apogee_mask.npy` | `(8575, 27)` line-window weights per input label (> 0 = line pixel) |
| `apogee_inference_mask.npy` | Variant of the mask used by the legacy inverse notebooks |
| `dataset_stats_120k.npz` | Legacy training-set mean/std of the 27 inputs |
| `element masks/`, `element filters/` | Per-element window files on the 7514-pixel gap-free grid |

## Other documents

- [`paper_draft.tex`](paper_draft.tex): draft manuscript.
- [`legacy_tf/`](legacy_tf): the TensorFlow notebooks and scripts this package replaces; to be deleted once the JAX re-runs reproduce comparable numbers.

## License

Apache License 2.0, see [`LICENSE`](LICENSE).
