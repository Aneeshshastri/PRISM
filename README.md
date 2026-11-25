# PRISM

PRISM is a research codebase for learning and inverting high-resolution stellar spectra from APOGEE DR17. The core idea is a physics-structured neural emulator that predicts optical depth, separates continuum, molecular and elemental contributions, and uses element-specific spectral windows so that each abundance can only change the flux at its own lines.

The code is a tested **JAX + Flax NNX** package (`prism/`) driven by thin Kaggle notebooks (`notebooks/`). Training and inversion run data-parallel over a 1-D device mesh, so they use both GPUs of Kaggle's **T4 x2** accelerator. The original TensorFlow/Keras notebooks and the first paper draft are kept in [`legacy/`](legacy) as a reference. Results of the current models are logged in [`RESULTS.md`](RESULTS.md).

## Scientific summary

PRISM is trained directly on real APOGEE DR17 spectra (ASPCAP labels as supervision) and predicts flux through a structured optical-depth decomposition:

1. a continuum branch: one Chebyshev series per detector chip, initialised at the normalised continuum (`k = 1`, `tau ~ 0`),
2. a molecular branch for the blended C/N/O opacity (CN, CO, OH), which may act on every pixel,
3. a thermodynamic-state branch that compresses the global labels into a 16-d state vector,
4. sparse atomic experts, one per element, that see only the state vector and their own abundance and write optical depth only inside that element's line windows.

Flux follows the Beer-Lambert law, `F(lambda) = k(lambda) * exp(-tau(lambda))`. The model has outputs only for the 7514 real pixels; the 1061 chip-gap pixels are returned as zero. The 14 atomic abundances therefore have exactly zero gradient leakage outside their windows. C/N/O are deliberately not confined: cool-giant spectra carry real molecular C/N/O signal outside the C/N/O filter windows (see [`RESULTS.md`](RESULTS.md)).

### Architecture options

Architecture switches live in `Config` and are saved in every checkpoint, so old checkpoints rebuild exactly as trained:

| Field | Default | Effect |
| --- | --- | --- |
| `chip_continuum` | `True` | per-chip continuum with the physical initialisation above |
| `drop_gaps` | `True` | no output units for chip-gap pixels (all forward models) |
| `cno_split` | `False` | C/N/O expert confined to the C/N/O windows; off because it hides real molecular signal (kept for the ablation) |

### Training protocol

All forward models share one protocol: Adam with global-norm clipping, the spectral loss scaled by `loss_scale = 100` (the raw loss is ~2e-5, which left most gradients below Adam's epsilon), one linear warm-up plus cosine decay over the full `epochs` budget, and early stopping on a relative improvement threshold (`rel_min_delta`). `resume=True` continues a forward model from its best checkpoint (weights only, the learning rate re-warms over one epoch), and `epoch_timeout_s` makes a hung epoch dump every thread's stack and exit instead of stalling silently.

Labels (always in this order): `TEFF LOGG M_H VMICRO VMACRO VSINI C_FE N_FE O_FE FE_H MG_FE SI_FE CA_FE TI_FE S_FE AL_FE MN_FE NI_FE CR_FE K_FE NA_FE V_FE CO_FE`, plus four engineered inputs: `INV_TEFF = 5040/TEFF`, `vbroad = sqrt(VMACRO^2 + VSINI^2)`, `C_O_diff = C_FE - O_FE`, `LOGPE = 0.5 LOGG + 0.5 M_H`.

> **Abundance convention:** although the columns keep their ASPCAP names, every `*_FE` label is converted from [X/Fe] to **[X/H]** during preprocessing (by adding `FE_H`). The emulators, the CNN warm start, the inverse pipelines and their result files all work in [X/H], and the plots label them that way. Only the ASPCAP `*_ERR` columns used as reference uncertainties are still [X/Fe] errors.

> **Rotation convention:** DR17 ASPCAP fits `VSINI` only for dwarfs; giants (~89% of stars) have no measured `VSINI` and use `VMACRO` instead. PRISM sets missing `VSINI` to 0 (so `vbroad = VMACRO` for giants) rather than imputing it, and does not count it as an imputed label. The legacy pipeline imputed it, giving giants a regression-guessed rotation of several km/s.

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
| Precision benchmark | `prism.bench.benchmark_training` | `09_benchmark_precision` | `benchmark/precision.json` |

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
7. **Order**: `01 -> 02 -> 03 -> 04 -> 05 -> 06 -> 07` (08 and 09 optional; 01 is optional because every notebook rebuilds the cache itself).

If Internet must stay off, attach a copy of this repository as a Kaggle dataset and replace the URL with its path, e.g. `pip install -c /tmp/kaggle-pins.txt /kaggle/input/prism-repo`.

## Running locally (single GPU, e.g. an RTX 5050 laptop)

The same notebooks run on one local GPU; environment variables replace the Kaggle paths.

```bash
# 1. A Python >= 3.12 env with the SAME JAX/Flax versions as Kaggle (JAX 0.11 needs Python >= 3.12;
#    CUDA-13 wheels need NVIDIA driver >= 580; WSL2 works)
conda create -n prism python=3.13 pip -y && conda activate prism
printf "jax==0.11.1\njaxlib==0.11.1\nflax==0.12.12\norbax-checkpoint==0.12.8\n" > /tmp/pins.txt
pip install -c /tmp/pins.txt "jax[cuda13]" flax orbax-checkpoint optax blackjax jupyterlab ipykernel -e ".[test]"
python -c "from prism.env import check_env; check_env(expected_gpus=1)"

# 2. Point PRISM at your data and keep the ~10 GB cache OFF /tmp if /tmp is a RAM disk (it is on WSL)
export PRISM_INPUT_ROOT=$HOME/apogee              # contains apogee_dr17_parallel.h5 and the element-masks files
export PRISM_CACHE_DIR=$HOME/prism_data
export PRISM_ARTIFACT_DIR=$HOME/prism_artifacts
export PRISM_EXPECTED_GPUS=1
export PRISM_TRAIN_PRECISION=float32              # or bfloat16 / tensorfloat32 (see Precision)
export XLA_PYTHON_CLIENT_PREALLOCATE=false        # laptop GPUs share VRAM with the display: allocate on demand
jupyter lab notebooks/
```

Without Jupyter, run stages directly; `run_pipeline` logs each stage's status and wall time to `$PRISM_ARTIFACT_DIR/run_log.json`, and any `Config` field can be overridden by keyword:

```python
from prism.config import Config
from prism.stages import run_pipeline

cfg = Config.for_kaggle()                      # e.g. Config.for_kaggle(epochs=100, resume=True)
run_pipeline(cfg, ["prepare", "prism", "ablation", "evaluate"])
run_pipeline(cfg, ["prepare", "prism", "evaluate"], smoke=True)   # 1 epoch / few stars: setup check
```

In each notebook skip the first (Kaggle install) cell. Stages 05-07 find checkpoints in `$PRISM_ARTIFACT_DIR` when everything runs on the same machine. Run `09_benchmark_precision` first to measure your GPU.

## Precision

| `train_precision` | Matmuls | Use |
| --- | --- | --- |
| `float32` (default) | true fp32 everywhere | Kaggle T4s, reference results |
| `tensorfloat32` | TF32 tensor cores (10-bit mantissa), fp32 storage | Ampere+ GPUs, faster training |
| `bfloat16` | bf16 compute in the layers, fp32 weights, loss and optimizer | Ampere+ GPUs, fastest training |

- Only training is affected. Checkpoints hold fp32 weights and are always re-loaded as fp32 models, and evaluation plus every inverse method run with **full fp32 matmuls on every GPU**.
- On Ampere and newer GPUs JAX's *default* float32 matmul may silently use TF32, while a T4 has no TF32. PRISM therefore pins the matmul precision explicitly, so the same code gives the same numbers on a T4 and on an RTX card.
- bf16 keeps ~3 significant digits. At random initialisation PRISM's bf16 forward pass differs from fp32 by ~1.4% (relative RMS), several times the ~0.1% flux accuracy the emulator is trained towards. Check `09_benchmark_precision`: compare `val SPL(fp32)` of the bf16 run with the fp32 run before adopting bf16 for production models.

## Package map

| Module | Role |
| --- | --- |
| `prism/config.py` | Label constants, `Config` (paths, splits, hyper-parameters), `Config.for_kaggle()` |
| `prism/env.py` | `check_env()`: versions, devices, driver check, sharded smoke test |
| `prism/mesh.py` | 1-D `('data',)` mesh, batch padding and sharding helpers |
| `prism/data/` | ASPCAP flags + train-only imputation, masks, HDF5 -> numpy cache, prefetching loader |
| `prism/models/` | `Prism`, `PrismAblation`, `Payne`, `HeteroscedasticCNNPredictor`, RegPayne Jacobian penalty, registry |
| `prism/losses.py`, `prism/train.py`, `prism/checkpoint.py` | Masked spectral loss, sharded NNX trainer (early stopping, LR plateau, Orbax checkpoints) |
| `prism/evaluate.py` | Reduced chi-squared (plain, imputation-masked, per spectral region and Teff group), stellar populations, gradient leakage, wrong-sign response fractions |
| `prism/entanglement.py` | Physical-vs-spurious entanglement diagnostics (partial-slope maps, proxy tests) |
| `prism/inverse/` | Shared objectives and bounds, CNN prior, batched MAP, Laplace, BlackJAX NUTS/HMC |
| `prism/analysis.py` | Result tables, calibration, outlier diagnostics and figures |
| `prism/precision.py`, `prism/bench.py` | Training precision policy (fp32 / TF32 / bf16), fp32 inference pinning, precision benchmark |
| `prism/stages.py` | One function per pipeline stage (what the notebooks call) and `run_pipeline` |

## Data assets in the repository

| Path | Purpose |
| --- | --- |
| `apogee_mask.npy` | `(8575, 27)` line-window weights per input label (> 0 = line pixel) |
| `element filters/` | Per-element window files on the 7514-pixel gap-free grid; `CNO filters/{C,N,O}.filt` define the C/N/O windows |

Normalisation statistics and the label imputer are rebuilt from the training split by `prepare_data`; no dataset statistics are stored in the repository.

## Other documents

- [`RESULTS.md`](RESULTS.md): log of experiments and results with the current code.
- [`legacy/paper_draft.tex`](legacy/paper_draft.tex) and [`legacy/results/`](legacy/results): the first manuscript draft and its figures; superseded (they describe the TensorFlow-era model and protocol).
- [`legacy/tf/`](legacy/tf): the TensorFlow notebooks and scripts this package replaces.

## License

Apache License 2.0, see [`LICENSE`](LICENSE).
