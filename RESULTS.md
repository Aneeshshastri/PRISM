# Results log

Experiments with the JAX code in `prism/`. Every number is on the APOGEE DR17 test split (HDF5 rows 140000-149500, 9500 stars). Masked chi-squared and region chi-squared use the 8527 stars without an imputed core, C/N/O or Fe label. All runs: seed 42, float32, one RTX 5050. Two runs of the same model and seed differ by about 0.2% in median chi-squared (measured below), so smaller differences are noise. Seed-to-seed variation is not measured yet.

## 2026-11-25: architecture options C, E, A and the new training protocol

### What changed

- **Protocol** (all forward models):
  - spectral loss x100;
  - one warm-up plus cosine decay over 150 epochs, replacing the SGDR restarts, which stopped models at different points in their cycles;
  - relative early-stopping threshold.
- **C** (`chip_continuum`): per-chip Chebyshev continuum, initialised at `k = 1`, `tau ~ 0`.
- **E** (`drop_gaps`): no outputs for the 1061 chip-gap pixels.
- **A** (`cno_split`):
  - The molecular branch is split into a background branch that sees only the global labels, plus a C/N/O expert that writes only inside the union of the C/N/O windows (5198 of 7514 pixels).
  - The C/N/O expert was made narrower so the parameter counts match: 8.58M with A, 8.36M without.

### A vs no A

| PRISM | best val loss | median chi2 | mean chi2 | p95 chi2 |
| --- | --- | --- | --- | --- |
| previous model (no C/E/A, old protocol) | - | 1.858 | 2.418 | 4.445 |
| C+E+A, resumed at epoch 67 after a hang | 0.0016701 | 1.690 | 1.959 | 3.466 |
| C+E+A, fresh run (same seed) | 0.0016700 | 1.690 | 1.955 | 3.462 |
| **C+E** | **0.0016334** | **1.666** | **1.885** | **3.343** |

The two C+E+A rows are the replicate that gives the 0.2% noise floor. They also show that resuming from a weights-only checkpoint costs nothing measurable.

Median chi2 by spectral region:

| Stars | Region | C+E+A | C+E | difference |
| --- | --- | --- | --- | --- |
| all | C/N/O windows only | 1.674 | 1.674 | 0.0% |
| Teff < 4000 giants (899) | C/N/O windows only | 2.671 | 2.642 | -1.1% |
| Teff < 4000 giants | unassigned pixels | 2.352 | 2.094 | -11.0% |
| 4000-4500 K giants (2047) | unassigned pixels | 1.693 | 1.565 | -7.5% |
| 4500-5000 K giants (4880) | unassigned pixels | 1.556 | 1.503 | -3.4% |
| Teff >= 5000 or dwarfs (701) | unassigned pixels | 1.528 | 1.529 | 0.0% |

"Unassigned" means real pixels outside every element window and outside the C/N/O windows (1414 pixels).

**Knockout test.** In the trained C+E model, C/N/O (and C-O) were replaced by their prediction from the other labels (R^2 = 0.95-0.97). This removes their independent information but keeps the part they share with metallicity. Unassigned-pixel chi2 then rises back to the C+E+A level or beyond:

| Stars | C+E | C+E, C/N/O knocked out | C+E+A |
| --- | --- | --- | --- |
| Teff < 4000 giants | 2.096 | 2.460 | 2.354 |
| 4000-4500 K giants | 1.565 | 1.651 | 1.697 |
| 4500-5000 K giants | 1.503 | 1.544 | 1.558 |
| Teff >= 5000 or dwarfs | 1.528 | 1.530 | 1.528 |

The effect grows towards low Teff and vanishes in hot stars. That is the signature of molecular (CN/CO/OH) lines outside the C/N/O filter windows, not of extra capacity or a metallicity proxy. Lowering the filter threshold does not recover these pixels: at 0.0005 the C/N/O union grows by only 7%.

A also broke the atomic experts' response signs. With C/N/O blocked from molecular blends, the Fe, Ni and Si experts absorbed them. Wrong-sign fraction is the fraction of stars whose mean in-window response to the element has the unphysical sign:

| | Fe | Ni | Si | Co | Cr | V |
| --- | --- | --- | --- | --- | --- | --- |
| C+E+A | 0.147 | 0.100 | 0.041 | 0.031 | 0.077 | 0.126 |
| C+E | 0.020 | 0.022 | 0.002 | 0.013 | 0.112 | 0.141 |

**Decision:** keep C and E, drop A (`cno_split = False` by default). B (monotonic experts by construction) was not implemented. After dropping A, only Cr and V show notable wrong-sign fractions, and both have weak lines in the H band.

### PRISM vs its ablation (same architecture, every label reaches every pixel)

Both trained under the new protocol:

| | PRISM | Ablation | gap now | gap under the old protocol |
| --- | --- | --- | --- | --- |
| median chi2 | 1.666 | 1.649 | +1.0% | +8.3% |
| mean chi2 | 1.885 | 1.858 | +1.4% | +22.0% |
| p95 chi2 | 3.343 | 3.277 | +2.0% | +25.4% |
| median chi2, masked | 1.678 | 1.656 | +1.3% | +6.6% |

The remaining gap sits in cool giants:
- Teff < 4000 K giants: +3.8% in the element windows and +3.2% in the C/N/O windows.
- Hotter stars and dwarfs: below 0.6%.
- Subgiants and metal-poor giants: PRISM is slightly better.

| | PRISM | Ablation |
| --- | --- | --- |
| leakage (forbidden / allowed response), 14 atomic elements | 0 | 0.07-0.68 (Fe 0.68, Ni 0.34, Si 0.29) |
| leakage, C / N / O (forbidden = outside the C/N/O windows) | not confined by design | 0.77 / 0.75 / 0.88 |
| wrong-sign fraction Fe / Mg / Co | 0.020 / 0.000 / 0.013 | 0.290 / 0.045 / 0.076 |
| wrong-sign fraction Cr / V | 0.112 / 0.141 | 0.125 / 0.191 |

### Open

- Payne, Expanded Payne, RegPayne and the CNN are not yet retrained under the new protocol. Until they are, compare PRISM only with the ablation.
- Only one seed has been run; the seed-to-seed spread is unknown.
- The inverse stages (MAP, Laplace, NUTS, HMC) have not been rerun with the new emulator.
- The Cr and V wrong-sign responses have not been investigated yet.
