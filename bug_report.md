# Bug Report: Backward Model Optimization Files

## Critical Bugs

### Bug 1: `backward_model_hmc.py` — Self-import crash (line 903)
```python
from backward_model_hmc import select_stratified_test_sample, load_test_spectra
```
The `__main__` block tries to import `select_stratified_test_sample` **from itself**, which would trigger a recursive import. Then it immediately redefines the function locally, making the import completely dead code. The `load_test_spectra` function doesn't even exist in this file.

**Fix**: Remove the import line entirely. The function is defined inline right after it.

---

### Bug 2: `backward_model_nuts.py` — `cnn_predict_physical` returns (std, mu) swapped (lines 248–259)
```python
def cnn_predict_physical(flux_batch):
    ...
    return np.maximum(std_p, 0.01*CNN_LABEL_STD).astype(np.float32), mu_p.astype(np.float32)
    # NOTE: returns (std, mu) — swapped for compat below
# Fix the order:
_cnn_orig = cnn_predict_physical
def cnn_predict_physical(flux_batch):
    std, mu = _cnn_orig(flux_batch)
    return mu, std
```
This convoluted wrapper-swap pattern was introduced during editing. It works but is fragile, confusing, and wasteful. Should just return `(mu, std)` directly like the other two files.

---

### Bug 3: `backward_model_hmc.py` — `DualAveragingStepSizeAdaptation` may be XLA-incompatible
`DualAveragingStepSizeAdaptation` wrapping HMC is inside a `@tf.function(jit_compile=True)`. While fixed-step HMC itself is XLA-compatible, `DualAveragingStepSizeAdaptation` uses internal counters and conditional updates that *may* cause XLA compilation failures depending on the TFP version. 

**Risk level**: Medium — works in some TFP versions, fails in others. The notebook's original NUTS version used the same adaptation but without `jit_compile`.

**Fix**: If XLA compilation fails at runtime, fall back to removing `jit_compile=True` from `sample_core_compiled` and `sample_element_compiled`, keeping only `reduce_retracing=True`. Added a comment noting this.

---

### Bug 4: `backward_model_map.py` — `tf.Variable` + `tf.keras.optimizers.Adam` inside `@tf.function(jit_compile=True)` (lines 336-388)
Creating a `tf.Variable` and `tf.keras.optimizers.Adam` **inside** a `jit_compile=True` function is problematic:
- `tf.Variable` creation inside XLA is not supported — variables must be created outside the traced function
- Adam optimizer maintains internal state (momentum, velocity) as variables

**Fix**: Move `tf.Variable` and optimizer creation outside the JIT function. Pass them as arguments or use a class-based approach.

---

## Moderate Bugs

### Bug 5: Ivar formula differs from notebook (all three files)
All three files use:
```python
safe_ivar = 1.0 / (1.0 / (safe_ivar + 1e-12) + 1e-5)
```
The original notebook uses:
```python
safe_ivar = 1.0 / (1.0 / (safe_ivar) + 1e-5)
```
The `+ 1e-12` was added as a division-by-zero guard. This is actually a **correct improvement** over the notebook, but changes the effective ivar values slightly. Where `safe_ivar = 0`, the notebook formula gives `0/(0+1e-5) = 0` while the new formula gives `1/(1/1e-12 + 1e-5) ≈ 1e-12`. This is negligible, so this is fine.

**Verdict**: Not a bug — intentional improvement. Keep it.

---

### Bug 6: `backward_model_hmc.py` — Unused `flux` and `ivar` variables (lines 975-976)
```python
flux = real_data[global_indices - config.start]
ivar = real_data_ivar[global_indices - config.start]
```
These are computed but never used — the pipeline is called with `real_data` and `real_data_ivar` directly.

---

### Bug 7: `backward_model_nuts.py` — Notebook used 1000 burnin / 800 adapt
The notebook docstring says `NUM_BURNIN 400 → 1000, NUM_ADAPT 320 → 800`, but the Tier 3 file uses the old values:
```python
NUM_CHAINS_CORE = 8; NUM_RESULTS_CORE = 500; NUM_BURNIN_CORE = 400; NUM_ADAPT_CORE = 320
```
Looking at the notebook's actual code (not just the docstring), it uses `400`/`320` — so this matches. The docstring was aspirational. **Not a bug** in the file, but the docstring in the notebook is misleading.

---

### Bug 8: `backward_model_map.py` — `get_err` uses `_FLAG` names not `_ERR` names for flag detection  
In `get_err()`, the flag checking loop uses `f"{l}_FLAG"` where `l` comes from `config.ERRORS` (which are `TEFF_ERR`, `LOGG_ERR`, etc.). So the flag name would be `TEFF_ERR_FLAG` which doesn't exist in the HDF5 file.

This is actually **inherited from the notebook** and appears to be harmless because the flag lookup simply falls through to the `elif` branch. But it's technically wrong — the flags are never matched. The imputation via percentile fill-in handles it regardless.

**Verdict**: Pre-existing notebook issue; not introduced by the optimization. Same in all three files.
