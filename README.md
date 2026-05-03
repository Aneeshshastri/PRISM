# PRISM

PRISM is a research repository for learning and inverting high-resolution stellar spectra from APOGEE DR17. The core idea is a physics-constrained neural emulator that predicts optical depth, separates continuum, molecular, and elemental contributions, and uses element-specific spectral masks to reduce cross-talk between abundance labels.

This repository is notebook-first, Kaggle-oriented, and actively experimental. It contains the forward-model training notebooks, baseline model experiments, inverse-inference pipelines, evaluation tooling, mask-generation utilities, and the draft paper that describes the scientific motivation and architecture.

## What This Repository Contains

At a high level, the repo has four layers:

1. Forward spectral emulators that map stellar labels to an 8575-pixel APOGEE spectrum.
2. Baseline models used for comparison against PRISM.
3. Inverse pipelines that recover stellar labels from observed spectra using the trained forward model.
4. Supporting mask files, analysis utilities, and research notes.

The project works with:

- 23 astrophysical labels.
- 8575 APOGEE detector pixels.
- 4 engineered features added to the label vector.
- A train/validation/test split of roughly `120k / 20k / 9.5k` stars.

## Scientific Summary

PRISM is trained directly on real APOGEE DR17 spectra rather than synthetic grids. The forward model uses ASPCAP labels as supervision and predicts flux through a structured optical-depth decomposition:

1. A continuum branch models the pseudo-continuum.
2. A molecular branch handles the coupled C/N/O contribution.
3. A thermodynamic state branch builds a latent state vector.
4. Sparse atomic experts model element-specific absorption only on their allowed wavelength regions.

Flux is reconstructed with the Beer-Lambert law:

```text
F(lambda) = k(lambda) * exp(-tau(lambda))
```

The repository consistently uses these 23 labels:

`TEFF`, `LOGG`, `M_H`, `VMICRO`, `VMACRO`, `VSINI`, `C_FE`, `N_FE`, `O_FE`, `FE_H`, `MG_FE`, `SI_FE`, `CA_FE`, `TI_FE`, `S_FE`, `AL_FE`, `MN_FE`, `NI_FE`, `CR_FE`, `K_FE`, `NA_FE`, `V_FE`, `CO_FE`

The training code also engineers four extra features:

- `INV_TEFF = 5040 / TEFF`
- `vbroad = sqrt(VMACRO^2 + VSINI^2)`
- `C_O_diff = C_FE - O_FE`
- `LOGPE = 0.5 * LOGG + 0.5 * M_H`

## Read This Repo In 5 Minutes

If you only want the shortest accurate mental model of the repository, this is it:

- [`model-trainer.ipynb`](model-trainer.ipynb) is the current working notebook for forward-model research.
- [`model-trainer_backup.ipynb`](model-trainer_backup.ipynb) is the broader baseline zoo containing PRISM-adjacent ablations plus Payne-style baselines.
- [`CNNwarmstart.ipynb`](CNNwarmstart.ipynb) trains a heteroscedastic CNN that predicts labels from spectra and is used to warm-start inverse inference.
- [`Model-tester.ipynb`](Model-tester.ipynb) evaluates trained forward models with reduced chi-squared, leakage analysis, and imputation-aware tests.
- [`backward-model-v3-laplace.ipynb`](backward-model-v3-laplace.ipynb), [`backward-model-nuts.ipynb`](backward-model-nuts.ipynb), and [`backward_model_hmc.py`](backward_model_hmc.py) are three generations of inverse-inference pipelines.
- [`map_results_analysis.py`](map_results_analysis.py) is the cleanest standalone analysis utility in the repo.
- [`paper_draft.tex`](paper_draft.tex) is the best narrative description of the method and experimental framing.

## Repository Map

| Path | Role | Notes |
| --- | --- | --- |
| [`model-trainer.ipynb`](model-trainer.ipynb) | Current forward-model training notebook | Main active notebook; includes masked-loss, gradient-gating, sparse experts, and current PRISM experiments. |
| [`model-trainer_backup.ipynb`](model-trainer_backup.ipynb) | Historical training notebook with baselines | Trains multiple forward models: baseline PRISM variant, CNN generator, Payne, Expanded Payne, and RegPayne. |
| [`optuna-tuner.ipynb`](optuna-tuner.ipynb) | Hyperparameter search notebook | Uses Optuna to tune Expanded Payne and RegPayne penalty settings. |
| [`CNNwarmstart.ipynb`](CNNwarmstart.ipynb) | Spectrum-to-label warm-start model | Trains a heteroscedastic 1D CNN that predicts means and uncertainties for 23 labels. |
| [`Model-tester.ipynb`](Model-tester.ipynb) | Forward-model evaluation notebook | Compares models on chi-squared, leakage, stellar-subpopulation performance, and imputation robustness. |
| [`backward-model.ipynb`](backward-model.ipynb) | Older inverse pipeline | Batched MAP-style inversion plus uncertainty estimation; points to Payne-family checkpoints. |
| [`backward-model-v3-laplace.ipynb`](backward-model-v3-laplace.ipynb) | Fast inverse pipeline | Tier-1 MAP optimization with Laplace/Hessian-based uncertainty approximation. |
| [`backward-model-nuts.ipynb`](backward-model-nuts.ipynb) | Full Bayesian inverse pipeline | Tier-3 NUTS-based posterior inference for core labels and abundances. |
| [`backward_model_hmc.py`](backward_model_hmc.py) | Script version of inverse inference | Fixed-step HMC variant designed for XLA compilation and Kaggle execution. |
| [`map_results_analysis.py`](map_results_analysis.py) | Standalone result analysis | Loads `results.npz`, prints metrics, and saves publication-style figures. |
| [`apogee_mask_loader.py`](apogee_mask_loader.py) | Mask conversion utility | Builds full-grid APOGEE masks from per-element `.mask` files. |
| [`paper_draft.tex`](paper_draft.tex) | Draft manuscript | Most complete prose description of the model and dataset. |
| [`implementation_plan.md.resolved`](implementation_plan.md.resolved) | Research planning note | Documents proposed architecture and training changes. |
| [`walkthrough.md.resolved`](walkthrough.md.resolved) | Implementation note | Explains masked loss, gating, and TFRecord/data-flow changes. |
| [`ANTIGRAVITY.md`](ANTIGRAVITY.md) | Repo-specific agent note | Environment and workflow instructions; not part of the scientific code. |

## Forward Models In The Repo

The repository contains several forward-model families:

### 1. PRISM / PRISM-like structured emulator

Implemented primarily in [`model-trainer.ipynb`](model-trainer.ipynb) and earlier forms in [`model-trainer_backup.ipynb`](model-trainer_backup.ipynb).

Key ideas:

- Predict optical depth instead of flux directly.
- Use a continuum branch plus molecular and atomic branches.
- Use `SparseProjector` layers to restrict each atomic expert to element-specific wavelengths.
- Use Beer-Lambert recombination for final flux.
- Incorporate engineered stellar-physics features into the label vector.

Important repo nuance:

- The notebook contains both `build_prism_emulator_masked(...)` and `build_final_emulator(...)`.
- The newer masked/gated machinery is present, but the execution path in the current notebook still instantiates `build_final_emulator(...)`.
- In practice, this means [`model-trainer.ipynb`](model-trainer.ipynb) is a live research notebook, not a fully cleaned production pipeline.

### 2. Ablation / baseline PRISM variant

The backup trainer and tester refer to a model saved as `SPECTROGRAM_GENERATOR.keras`. In the surrounding experiments, this acts as an ablation-style structured model without the full masked PRISM setup used in the paper comparisons.

### 3. Payne baselines

Defined in [`model-trainer_backup.ipynb`](model-trainer_backup.ipynb):

- `ThePayne.keras`: shallow MLP baseline.
- `Expanded_Payne.keras`: deeper/wider MLP baseline.
- `RegPayne.keras`: Payne baseline trained with an additional disentanglement-style penalty.

`RegPayne` is trained in two phases:

1. A weak-penalty phase to learn spectral structure.
2. A strong-penalty phase to enforce discipline against leakage.

### 4. CNN generator baseline

Also present in [`model-trainer_backup.ipynb`](model-trainer_backup.ipynb) as a label-to-spectrum baseline distinct from the warm-start CNN.

## Inverse Inference Pipelines

The inverse side of the repository recovers stellar labels from observed spectra by repeatedly calling a trained forward model.

### Tier 1: MAP + Laplace

[`backward-model-v3-laplace.ipynb`](backward-model-v3-laplace.ipynb)

- Fastest inverse pipeline.
- Uses batched Adam optimization.
- Fits 9 core physics parameters on the full spectrum first.
- Fits 14 elemental abundances next on element-masked pixels.
- Uses Hessian-based curvature estimates for approximate uncertainties.

### Tier 2: Fixed-step HMC

[`backward_model_hmc.py`](backward_model_hmc.py)

- Most script-like inverse implementation in the repo.
- Replaces NUTS with fixed-step HMC for better XLA compilation.
- Uses the same two-stage FERRE-style decomposition.
- Warm-starts from the CNN predictor and adds Gaussian priors.

### Tier 3: NUTS

[`backward-model-nuts.ipynb`](backward-model-nuts.ipynb)

- Slowest but most Bayesian inverse workflow.
- Uses No-U-Turn Sampling for posterior exploration.
- Intended for more faithful uncertainty estimation than Laplace.

## Evaluation And Diagnostics

[`Model-tester.ipynb`](Model-tester.ipynb) is the main evaluation notebook for forward models. It compares:

- PRISM
- The Payne
- Expanded Payne
- Ablation PRISM
- RegPayne

It includes:

- Reduced chi-squared on the held-out test set.
- Imputed-vs-natural label performance checks.
- Mask-aware chi-squared recalculation.
- Information-leakage analysis using gradients and `apogee_mask.npy`.
- Performance breakdown by stellar population.

[`map_results_analysis.py`](map_results_analysis.py) is the clean standalone analysis tool for inverse results. It:

- Loads `results.npz`.
- Computes bias, MAD, RMSE, uncertainty ratios, and timing summaries.
- Generates one-to-one plots, residual histograms, bias bars, residual-vs-`TEFF` plots, heatmaps, and timing charts.

## Data Assets Included In The Repo

This repository includes several local artifacts that support training or inference:

| Path | Purpose |
| --- | --- |
| [`apogee_mask.npy`](apogee_mask.npy) | Full `8575 x 27` spectral mask array used for element-sensitive pixels. |
| [`apogee_inference_mask.npy`](apogee_inference_mask.npy) | Inference-oriented variant of the APOGEE mask. |
| [`dataset_stats_120k.npz`](dataset_stats_120k.npz) | Training-set mean and standard deviation vectors for the 27-dimensional input space. |
| [`element masks/`](element%20masks) | Per-element `.mask` files on the 7514-pixel gap-removed grid. |
| [`element filters/`](element%20filters) | Per-element `.filt` files, including C/N/O filters. |

`apogee_mask_loader.py` converts the sparse 7514-pixel element files into the full 8575-pixel detector layout by inserting APOGEE chip gaps.

## Typical Workflow

The repository is easiest to understand as one research workflow:

1. Prepare labels, impute missing values, engineer 4 extra features, and write TFRecords.
2. Train a forward emulator in [`model-trainer.ipynb`](model-trainer.ipynb) or a baseline in [`model-trainer_backup.ipynb`](model-trainer_backup.ipynb).
3. Evaluate forward models in [`Model-tester.ipynb`](Model-tester.ipynb).
4. Train the warm-start CNN in [`CNNwarmstart.ipynb`](CNNwarmstart.ipynb).
5. Run inverse inference with MAP, HMC, or NUTS.
6. Analyze inverse results with [`map_results_analysis.py`](map_results_analysis.py).

## Running The Code

This repo is not packaged as a pip-installable project. It is designed mainly for Kaggle notebooks with hard-coded dataset/model paths.

### Expected environment

Most workflows expect:

- Python 3
- TensorFlow
- TensorFlow Probability
- NumPy
- h5py
- scikit-learn
- matplotlib
- SciPy
- joblib
- Optuna

### Important practical notes

- Most notebook paths point to Kaggle datasets or Kaggle model registry artifacts.
- There is no pinned `requirements.txt` or environment file in the repo.
- Several notebooks assume GPU execution.
- Some functionality depends on model files that are not checked into this repository, only referenced by Kaggle paths.

## Which Files Are Canonical

If you are trying to orient yourself quickly, use this priority order:

1. [`paper_draft.tex`](paper_draft.tex) for the scientific story.
2. [`model-trainer.ipynb`](model-trainer.ipynb) for the latest forward-model direction.
3. [`Model-tester.ipynb`](Model-tester.ipynb) for how the author judges model quality.
4. [`CNNwarmstart.ipynb`](CNNwarmstart.ipynb) plus [`backward_model_hmc.py`](backward_model_hmc.py) for the current inverse workflow.
5. [`model-trainer_backup.ipynb`](model-trainer_backup.ipynb) and [`optuna-tuner.ipynb`](optuna-tuner.ipynb) for baseline and hyperparameter-search history.

## Current State Of The Repository

This is a research codebase, not a production library. That matters for how to read it:

- The notebooks are the source of truth.
- Some scripts are extracted or cleaned versions of notebook logic.
- Naming is historically layered: newer and older variants coexist.
- Hard-coded paths and saved-model names reflect experiment history.
- The repo contains both active code and supporting notes for planned or partially integrated changes.

That said, the overall structure is coherent:

- forward modeling
- baseline comparison
- warm-start prediction
- inverse inference
- diagnostic analysis

## Related Documents

- [`paper_draft.tex`](paper_draft.tex): draft manuscript for the method.
- [`walkthrough.md.resolved`](walkthrough.md.resolved): masked-loss and data-flow walkthrough.
- [`implementation_plan.md.resolved`](implementation_plan.md.resolved): architecture-improvement plan.

## License

This repository is licensed under the Apache License 2.0. See [`LICENSE`](LICENSE).
