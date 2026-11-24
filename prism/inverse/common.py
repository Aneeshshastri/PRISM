"""Shared pieces of the inverse pipelines: features, bounds/bijectors, likelihood,
priors, test-star selection and the results schema.

All abundances are [X/H] (see prism.config).
"""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from prism.config import SELECTED_LABELS
from prism.models.functional import make_flux_fn  # noqa: F401  (re-export)

# Prior box shared by MAP, Laplace and MCMC (audit-fixed notebook bounds; VSINI >= 0).
LOWER = np.array([3000., -0.5, -2.5, 0., 0., 0., -2.5, -2.5, -2.5, -2.5, -2.0, -1.5, -1.5,
                  -2.0, -2.0, -2.0, -2.0, -2.0, -2.5, -2.0, -2.0, -2.0, -2.0], np.float32)
UPPER = np.array([25000., 6.0, 2.0, 6.0, 25.0, 150.0, 3.0, 4.0, 3.0, 3.0, 3.0, 3.0, 3.0,
                  3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 4.0, 5.0, 3.0, 4.0], np.float32)

IVAR_FLOOR_VAR = 1e-5  # extra variance added to 1/ivar


def features_27(theta23):
    """Physical 23 labels -> 27 inputs (same formulas as prism.data.labels.engineer_features)."""
    t = theta23
    eng = jnp.stack([
        5040.0 / (t[..., 0] + 1e-6),
        jnp.sqrt(t[..., 4] ** 2 + t[..., 5] ** 2),
        t[..., 6] - t[..., 8],
        0.5 * t[..., 1] + 0.5 * t[..., 2],
    ], axis=-1)
    return jnp.concatenate([t, eng], axis=-1)


def to_physical(u, lo, hi):
    return lo + (hi - lo) * jax.nn.sigmoid(u)


def to_unconstrained(x, lo, hi, margin: float = 1e-3):
    """Clip strictly inside (lo, hi), then logit; out-of-box values never become NaN."""
    lo = jnp.asarray(lo, jnp.float32)
    hi = jnp.asarray(hi, jnp.float32)
    x = jnp.clip(jnp.asarray(x, jnp.float32), lo + margin, hi - margin)
    p = (x - lo) / (hi - lo)
    return jnp.log(p) - jnp.log1p(-p)


def log_det_jacobian(u, lo, hi):
    """log |d to_physical / du| summed over the last axis."""
    return jnp.sum(jnp.log(jnp.asarray(hi - lo, jnp.float32)) + jax.nn.log_sigmoid(u) + jax.nn.log_sigmoid(-u),
                   axis=-1)


def log_likelihood(model_flux, obs_flux, obs_ivar, pixel_mask=None):
    """Gaussian log-likelihood with a 1e-5 variance floor; bad pixels contribute nothing."""
    obs = jnp.asarray(obs_flux, jnp.float32)
    ivar = jnp.asarray(obs_ivar, jnp.float32)
    ok = jnp.isfinite(obs) & jnp.isfinite(ivar) & (ivar > 0) & (obs > 1e-3) & (obs < 1.3)
    if pixel_mask is not None:
        ok = ok & (jnp.asarray(pixel_mask) > 0)
    obs = jnp.where(ok, obs, 1.0)
    ivar_eff = 1.0 / (1.0 / jnp.where(ok, ivar, 1.0) + IVAR_FLOOR_VAR)
    resid = jnp.where(ok, obs - model_flux, 0.0)
    return -0.5 * jnp.sum(resid ** 2 * ivar_eff, axis=-1)


def gaussian_prior(theta, mu, sigma, weight: float = 1.0):
    return -0.5 * weight * jnp.sum(((theta - mu) / sigma) ** 2, axis=-1)


def select_test_stars(snr, teff, logg, mh, target_n: int | None, seed: int = 42,
                      bins: tuple[int, int, int] = (5, 10, 10)) -> np.ndarray:
    """Local indices of the stars to invert.

    ``None`` -> every valid star. Otherwise one median-SNR star per occupied
    (logg, teff, [M/H]) percentile bin; with more occupied bins than
    ``target_n`` a seeded random subset of bins is used, with fewer the set is
    topped up with random valid stars.
    """
    snr, teff, logg, mh = (np.asarray(a, np.float64) for a in (snr, teff, logg, mh))
    valid = (teff > -5000) & (logg > -5000) & (mh > -5000)
    valid_idx = np.where(valid)[0]
    if target_n is None:
        return valid_idx
    rng = np.random.default_rng(seed)

    def bin_of(values, n_bins):
        edges = np.percentile(values[valid], np.linspace(0, 100, n_bins + 1))
        edges[0] -= 1e-6
        edges[-1] += 1e-6
        return np.clip(np.digitize(values, edges) - 1, 0, n_bins - 1)

    key = (bin_of(logg, bins[0]) * bins[1] + bin_of(teff, bins[1])) * bins[2] + bin_of(mh, bins[2])
    chosen = []
    for b in np.unique(key[valid]):
        idx = valid_idx[key[valid] == b]
        chosen.append(idx[np.argmin(np.abs(snr[idx] - np.median(snr[idx])))])
    chosen = np.array(chosen, dtype=np.int64)
    if len(chosen) > target_n:
        chosen = chosen[np.sort(rng.choice(len(chosen), target_n, replace=False))]
    elif len(chosen) < target_n:
        rest = np.setdiff1d(valid_idx, chosen)
        extra = rng.choice(rest, min(target_n - len(chosen), len(rest)), replace=False)
        chosen = np.concatenate([chosen, extra])
    return np.sort(chosen)


_REQUIRED = ("global_indices", "true_labels", "inferred_labels", "aspcap_errors",
             "true_label_imputed", "wall_seconds")


@dataclass
class InverseResults:
    """Per-star inverse results; ``save`` writes the npz schema prism.analysis reads."""

    global_indices: np.ndarray | None = None
    true_labels: np.ndarray | None = None
    inferred_labels: np.ndarray | None = None
    aspcap_errors: np.ndarray | None = None
    true_label_imputed: np.ndarray | None = None
    wall_seconds: np.ndarray | None = None
    inferred_errors: np.ndarray | None = None
    lower_sigma: np.ndarray | None = None
    upper_sigma: np.ndarray | None = None
    r_hat: np.ndarray | None = None
    ess: np.ndarray | None = None
    acceptance_rates: np.ndarray | None = None
    core_loss: np.ndarray | None = None
    elem_loss: np.ndarray | None = None

    @classmethod
    def empty(cls) -> "InverseResults":
        return cls()

    def __len__(self) -> int:
        return 0 if self.global_indices is None else len(self.global_indices)

    def append(self, other: "InverseResults") -> "InverseResults":
        merged = {}
        for f in dataclasses.fields(self):
            a, b = getattr(self, f.name), getattr(other, f.name)
            merged[f.name] = b if a is None else a if b is None else np.concatenate([a, b])
        return InverseResults(**merged)

    def arrays(self) -> dict[str, np.ndarray]:
        out = {f.name: np.asarray(getattr(self, f.name)) for f in dataclasses.fields(self)
               if getattr(self, f.name) is not None}
        out["label_names"] = np.array(SELECTED_LABELS)
        return out

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".npz.tmp")
        with open(tmp, "wb") as fh:
            np.savez(fh, **self.arrays())
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: Path) -> "InverseResults":
        with np.load(path, allow_pickle=False) as d:
            names = {f.name for f in dataclasses.fields(cls)}
            return cls(**{k: d[k] for k in d.files if k in names})


class ResultsCheckpoint:
    """Resume support: only reuses saved stars if they are a prefix of the expected order."""

    def __init__(self, path: Path, expected_indices: np.ndarray):
        self.path = Path(path)
        self.expected = np.asarray(expected_indices)

    def load(self) -> InverseResults | None:
        if not self.path.exists():
            return None
        res = InverseResults.load(self.path)
        done = np.asarray(res.global_indices)
        if len(done) > len(self.expected) or not np.array_equal(done, self.expected[:len(done)]):
            print(f"  checkpoint {self.path} does not match this star list; starting fresh")
            return None
        return res

    def save(self, results: InverseResults) -> None:
        results.save(self.path)


LO_CORE, HI_CORE = LOWER[:9], UPPER[:9]


class Objectives:
    """Per-star negative log posteriors in unconstrained space (no Jacobian term).

    Every method takes ONE star; vmap over stars. ``fn(state, x)`` is the
    emulator applied to normalised 27-vectors (``make_flux_fn``).
    """

    def __init__(self, fn, state, mean27, std27):
        self.fn = fn
        self.state = state
        self.mean = jnp.asarray(mean27, jnp.float32)
        self.std = jnp.asarray(std27, jnp.float32)
        self.lower = jnp.asarray(LOWER)
        self.upper = jnp.asarray(UPPER)

    def model_flux(self, theta23):
        x = (features_27(theta23) - self.mean) / self.std
        return self.fn(self.state, x[None, :])[0]

    def core(self, u9, abund14, obs, ivar, mu9, sig9, weight=1.0):
        theta = to_physical(u9, LO_CORE, HI_CORE)
        flux = self.model_flux(jnp.concatenate([theta, abund14]))
        return -log_likelihood(flux, obs, ivar) - gaussian_prior(theta, mu9, sig9, weight)

    def element(self, u1, j, full23, obs, ivar, pixel_mask, mu1, sig1, weight):
        theta = to_physical(u1, self.lower[j], self.upper[j])
        flux = self.model_flux(full23.at[j].set(theta[0]))
        return -log_likelihood(flux, obs, ivar, pixel_mask) - gaussian_prior(theta, mu1, sig1, weight)

    def joint(self, u23, obs, ivar, mu23, sig23, weight=1.0):
        theta = to_physical(u23, self.lower, self.upper)
        return -log_likelihood(self.model_flux(theta), obs, ivar) - gaussian_prior(theta, mu23, sig23, weight)
