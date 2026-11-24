"""Building blocks shared by the forward models."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import jax.numpy as jnp
import numpy as np
from flax import nnx

from prism.config import N_PIXELS


class Const(nnx.Variable):
    """Non-trainable array stored in the model state (saved with checkpoints)."""


def chebyshev_basis(degree: int, n_pixels: int) -> np.ndarray:
    x = np.linspace(-1.0, 1.0, n_pixels, dtype=np.float64)
    basis = np.zeros((degree, n_pixels), np.float64)
    if degree > 0:
        basis[0] = 1.0
    if degree > 1:
        basis[1] = x
    for i in range(2, degree):
        basis[i] = 2.0 * x * basis[i - 1] - basis[i - 2]
    return basis.astype(np.float32)


class MLP(nnx.Module):
    def __init__(self, in_features: int, widths: Sequence[int], out_features: int, *,
                 hidden_act: Callable = nnx.swish, out_act: Callable | None = None, rngs: nnx.Rngs):
        dims = [in_features, *widths]
        self.hidden = nnx.List([nnx.Linear(a, b, rngs=rngs) for a, b in zip(dims[:-1], dims[1:])])
        self.out = nnx.Linear(dims[-1], out_features, rngs=rngs)
        self.hidden_act = hidden_act
        self.out_act = out_act

    def __call__(self, x):
        for layer in self.hidden:
            x = self.hidden_act(layer(x))
        x = self.out(x)
        return x if self.out_act is None else self.out_act(x)


class ChebyshevContinuum(nnx.Module):
    """Pseudo-continuum = low-order Chebyshev series whose coefficients an MLP predicts."""

    def __init__(self, in_features: int, *, degree: int = 20, n_pixels: int = N_PIXELS, rngs: nnx.Rngs):
        self.coeffs = MLP(in_features, [64, 32], degree, rngs=rngs)
        self.basis = Const(jnp.asarray(chebyshev_basis(degree, n_pixels)))

    def __call__(self, x):
        return self.coeffs(x) @ self.basis[...]


class SparseProjector(nnx.Module):
    """Scatter an expert's local optical depth into its weighted line window; zero elsewhere."""

    def __init__(self, indices: np.ndarray, weights: np.ndarray, n_pixels: int = N_PIXELS):
        self.indices = Const(jnp.asarray(np.asarray(indices, np.int32)))
        self.weights = Const(jnp.asarray(np.asarray(weights, np.float32)))
        self.n_pixels = n_pixels

    def __call__(self, local_tau):
        out = jnp.zeros(local_tau.shape[:-1] + (self.n_pixels,), local_tau.dtype)
        return out.at[..., self.indices[...]].set(local_tau * self.weights[...])


def beer_lambert(k, tau):
    return k * jnp.exp(-tau)
