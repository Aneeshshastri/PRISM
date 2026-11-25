"""Physical vs non-physical abundance entanglement diagnostics.

The leakage ratio in ``prism.evaluate`` counts *any* response of the flux to [X/H]
outside X's line windows as bad. Real spectra do have some: electron donors (Mg, Si,
Fe, Na, Al, Ca, K) change the electron pressure, hence H- continuous opacity and other
species' ionisation balance; molecules (MgH, TiO, VO) matter in cool stars. Two
data-only tests separate that from spurious, correlation-driven entanglement:

1. **Residual test** (``residual_test``): PRISM forces zero response outside X's windows.
   If real off-window signal exists, PRISM's noise-normalised residuals there correlate
   with [X/M] = [X/H] - [M/H] at fixed (Teff, logg, [M/H]). The per-pixel partial slopes
   are summarised by a permutation-calibrated significance and by the bias-corrected
   chi^2 that ignoring the off-window response costs for a 0.05 dex change.
2. **Proxy test** (``proxy_test``): galactic chemical evolution correlates abundances,
   so an unconstrained emulator can use X as a stand-in for Y's lines. That leaves a
   fingerprint: X's off-window response concentrates on other elements' windows and
   matches sum_Y beta(Y|X) dF/d[Y/H], with beta taken from the training labels. Physical
   electron-pressure effects are instead broadband (captured by a smooth basis).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from prism.config import ATOMIC_LABELS, CNO_LABELS, CORE_LABELS, FEH_LABELS, LABELS_27, SELECTED_LABELS

DX = 0.05  # dex: abundance change used for the effect size
_RET_COLS = [SELECTED_LABELS.index(lab) for lab in CORE_LABELS + CNO_LABELS + FEH_LABELS]


# ----------------------------------------------------------------- regression core
def design_matrix(teff, logg, mh) -> np.ndarray:
    """Standardised [1, T, g, M, T^2, Tg, g^2, TM] covariates (core-label trends)."""
    t, g, m = (np.asarray(a, np.float64) for a in (teff, logg, mh))
    t, g, m = ((a - a.mean()) / (a.std() or 1.0) for a in (t, g, m))
    return np.stack([np.ones_like(t), t, g, m, t * t, t * g, g * g, t * m], axis=1)


def residualize(z, valid, design, block: int = 1024) -> np.ndarray:
    """Per-pixel weighted least-squares residuals of ``z`` on ``design`` (invalid -> 0)."""
    z = np.asarray(z, np.float64)
    m = np.asarray(valid, np.float64)
    d = np.asarray(design, np.float64)
    out = np.zeros_like(z)
    k = d.shape[1]
    for s in range(0, z.shape[1], block):
        mz, mm = (z[:, s:s + block] * m[:, s:s + block]), m[:, s:s + block]
        gram = np.einsum("np,ni,nj->pij", mm, d, d) + 1e-9 * np.eye(k)
        rhs = np.einsum("np,ni->pi", mz, d)
        coef = np.linalg.solve(gram, rhs[..., None])[..., 0]  # (p, k)
        out[:, s:s + block] = (z[:, s:s + block] - d @ coef.T) * mm
    return out


def _slopes(z_perp, valid, x_perp, n_cov: int):
    m = np.asarray(valid, np.float64)
    sxx = (m * x_perp[:, None] ** 2).sum(0)
    b = (x_perp @ z_perp) / np.maximum(sxx, 1e-12)
    resid = (z_perp - np.outer(x_perp, b)) * m
    dof = np.maximum(m.sum(0) - n_cov - 1, 1.0)
    se = np.sqrt((resid ** 2).sum(0) / dof / np.maximum(sxx, 1e-12))
    return b, se


def _x_perp(x, design):
    coef, *_ = np.linalg.lstsq(design, x, rcond=None)
    return np.asarray(x, np.float64) - design @ coef


def partial_slope_map(z, valid, x, design):
    """Per-pixel slope of ``z`` on ``x`` controlling for ``design`` (Frisch-Waugh), with SE."""
    return _slopes(residualize(z, valid, design), valid, _x_perp(x, design), design.shape[1])


def signal_summary(z, valid, x, design, *, n_perm: int = 20, seed: int = 0, dx: float = DX) -> dict:
    """Significance (permutation-calibrated) and size of the z ~ x dependence over pixels.

    ``T`` = sum over pixels of (b/se)^2; ``perm_z`` = (T - mean_perm) / std_perm with x
    shuffled across stars (pixels share star-level noise, so T is not chi^2-distributed).
    ``delta_chi2_005dex`` = dx^2 * sum(b^2 - se^2): the chi^2 (in noise units) of the
    response a 0.05 dex change produces in these pixels, corrected for estimation noise.
    """
    z_perp = residualize(z, valid, design)
    k = design.shape[1]
    xp = _x_perp(x, design)
    b, se = _slopes(z_perp, valid, xp, k)
    t_obs = float(np.sum((b / se) ** 2))
    rng = np.random.default_rng(seed)
    t_null = []
    for _ in range(n_perm):
        bp, sep = _slopes(z_perp, valid, _x_perp(rng.permutation(np.asarray(x)), design), k)
        t_null.append(float(np.sum((bp / sep) ** 2)))
    t_null = np.asarray(t_null)
    perm_z = float((t_obs - t_null.mean()) / max(t_null.std(), 1e-12)) if n_perm else float("nan")
    return {"n_stars": int(z.shape[0]), "n_pixels": int(z.shape[1]), "T": t_obs, "perm_z": perm_z,
            "delta_chi2_005dex": float(dx ** 2 * np.sum(b ** 2 - se ** 2)),
            "n_pixels_t_gt_5": int(np.sum(np.abs(b / se) > 5))}


# ----------------------------------------------------------------- proxy metrics
def label_betas(x, others: dict, core) -> dict:
    """Partial slope of each other abundance on x at fixed core labels (training labels)."""
    c = np.asarray(core, np.float64)
    d = np.column_stack([np.ones(len(x)), c, np.asarray(x, np.float64)])
    return {name: float(np.linalg.lstsq(d, np.asarray(y, np.float64), rcond=None)[0][-1])
            for name, y in others.items()}


def proxy_metrics(j_x, win_x, j_others: dict, win_others: dict, betas: dict, degree: int = 20) -> dict:
    """Decompose X's off-window response into proxy-like, line-concentrated and broadband parts."""
    from prism.models.layers import chebyshev_basis

    j_x = np.asarray(j_x, np.float64)
    forb = ~np.asarray(win_x, bool)
    energy = float(np.sum(j_x[forb] ** 2))
    if energy <= 0:
        return {"forbidden_energy": 0.0, "window_concentration": float("nan"),
                "proxy_r2": float("nan"), "smooth_fraction": float("nan")}
    union = np.zeros_like(forb)
    for w in win_others.values():
        union |= np.asarray(w, bool)
    in_u = forb & union
    area = in_u.sum() / max(forb.sum(), 1)
    conc = float((np.sum(j_x[in_u] ** 2) / energy) / area) if area > 0 else float("nan")
    pred = sum(betas[name] * np.asarray(j, np.float64) for name, j in j_others.items())
    proxy_r2 = 1.0 - float(np.sum((j_x[forb] - pred[forb]) ** 2)) / energy
    basis = chebyshev_basis(degree, j_x.size).T[forb]  # (n_forb, degree)
    coef, *_ = np.linalg.lstsq(basis, j_x[forb], rcond=None)
    smooth = 1.0 - float(np.sum((j_x[forb] - basis @ coef) ** 2)) / energy
    return {"forbidden_energy": energy, "window_concentration": conc, "proxy_r2": proxy_r2,
            "smooth_fraction": smooth}


# ----------------------------------------------------------------- real-data drivers
def _retained(bad_mask) -> np.ndarray:
    return ~np.asarray(bad_mask, bool)[:, _RET_COLS].any(axis=1)


def residual_test(model, test: dict, mean, std, apogee_mask, mesh, *, elements=ATOMIC_LABELS,
                  n_perm: int = 20, seed: int = 0, var_floor: float = 1e-5, min_valid: int = 200) -> dict:
    """Run test 1 for one emulator on the test split; returns per-element summaries."""
    from prism.evaluate import predict

    bad = np.asarray(test["bad_mask"], bool)
    phys = np.asarray(test["labels"])[:, :23] * std[:23] + mean[:23]
    keep = _retained(bad)
    flux = np.asarray(test["flux"], np.float32)
    ivar = np.asarray(test["ivar"], np.float32)
    pred = predict(model, np.asarray(test["labels"]), mesh)
    with np.errstate(divide="ignore", invalid="ignore"):
        ok = np.isfinite(flux) & np.isfinite(ivar) & (ivar > 0) & (flux > 1e-3)
        w = np.where(ok, 1.0 / np.sqrt(1.0 / np.where(ok, ivar, 1.0) + var_floor), 0.0)
    z_all = np.where(ok, (flux - pred) * w, 0.0).astype(np.float32)
    del pred
    teff, logg, mh = phys[:, 0], phys[:, 1], phys[:, 2]
    cool = (teff < 4000) & (logg < 2.0)
    results = {}
    for lab in elements:
        col = LABELS_27.index(lab)
        sel = SELECTED_LABELS.index(lab)
        stars = keep & ~bad[:, sel]
        x = phys[:, sel] - mh  # [X/M]
        out = {}
        for region, pix in (("forbidden", apogee_mask[:, col] <= 0), ("window", apogee_mask[:, col] > 0)):
            for group, gmask in (("all", stars), ("cool_giants", stars & cool)):
                if gmask.sum() < 50:
                    continue
                v = ok[np.ix_(gmask, pix)]
                cols = v.sum(0) >= min(min_valid, gmask.sum() // 2)
                if cols.sum() == 0:
                    continue
                z = z_all[np.ix_(gmask, pix)][:, cols]
                d = design_matrix(teff[gmask], logg[gmask], mh[gmask])
                out[f"{region}/{group}"] = signal_summary(z, v[:, cols], x[gmask], d, n_perm=n_perm, seed=seed)
        results[lab] = out
        f = out.get("forbidden/all", {})
        print(f"  {lab:<6} forbidden: perm_z {f.get('perm_z', float('nan')):7.1f}  "
              f"dchi2(0.05 dex) {f.get('delta_chi2_005dex', float('nan')):8.3f}  "
              f"pixels |t|>5: {f.get('n_pixels_t_gt_5', 0)}")
    return results


def mean_jacobians(model, labels_norm, std, *, elements=ATOMIC_LABELS, batch: int = 256) -> dict:
    """Mean signed dF/d[X/H] (physical units) over the given stars, per element."""
    import jax
    import jax.numpy as jnp

    from prism.models.functional import make_flux_fn

    fn, state = make_flux_fn(model, jit=False)

    @jax.jit
    def jvp_sum(s, x, col):
        t = jnp.zeros_like(x).at[:, col].set(1.0)
        return jnp.sum(jax.jvp(lambda xx: fn(s, xx), (x,), (t,))[1], axis=0)

    x = np.asarray(labels_norm, np.float32)
    n_pix = int(fn(state, jnp.asarray(x[:1])).shape[-1])
    out = {}
    for lab in elements:
        col = LABELS_27.index(lab)
        tot = np.zeros(n_pix)
        for s in range(0, len(x), batch):
            tot += np.asarray(jvp_sum(state, jnp.asarray(x[s:s + batch]), col))
        out[lab] = tot / len(x) / std[col]
    return out


def proxy_test(models: dict, test: dict, train_phys, train_bad, mean, std, apogee_mask, *,
               elements=ATOMIC_LABELS, n_stars: int = 512, seed: int = 0) -> dict:
    """Run test 2 for each model; returns {model: {element: metrics}} plus label betas."""
    bad = np.asarray(test["bad_mask"], bool)
    keep = np.where(_retained(bad) & ~bad[:, [SELECTED_LABELS.index(e) for e in elements]].any(1))[0]
    rng = np.random.default_rng(seed)
    pick = np.sort(rng.choice(keep, min(n_stars, len(keep)), replace=False))
    labels = np.asarray(test["labels"])[pick]

    tb = np.asarray(train_bad, bool)
    tidx = [SELECTED_LABELS.index(e) for e in elements]
    clean = ~tb[:, tidx].any(1) & ~tb[:, [SELECTED_LABELS.index(c) for c in CORE_LABELS]].any(1)
    tp = np.asarray(train_phys)[clean]
    core = tp[:, [0, 1, 2]]
    betas = {}
    for e in elements:
        xi = tp[:, SELECTED_LABELS.index(e)]
        betas[e] = label_betas(xi, {o: tp[:, SELECTED_LABELS.index(o)] for o in elements if o != e}, core)

    windows = {e: apogee_mask[:, LABELS_27.index(e)] > 0 for e in elements}
    out = {"label_betas": betas}
    for name, model in models.items():
        jac = mean_jacobians(model, labels, std, elements=elements)
        out[name] = {}
        for e in elements:
            others = [o for o in elements if o != e]
            out[name][e] = proxy_metrics(jac[e], windows[e], {o: jac[o] for o in others},
                                         {o: windows[o] for o in others}, betas[e])
        med = {k: float(np.nanmedian([out[name][e][k] for e in elements]))
               for k in ("window_concentration", "proxy_r2", "smooth_fraction")}
        print(f"  {name:<15} median over elements: window concentration {med['window_concentration']:.2f}  "
              f"proxy R^2 {med['proxy_r2']:.2f}  smooth fraction {med['smooth_fraction']:.2f}")
    return out


def write_json(path: Path, obj) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=float))
