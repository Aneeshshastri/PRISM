"""Building blocks shared by the forward models."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import jax.numpy as jnp
import numpy as np
from flax import nnx

from prism.config import N_PIXELS
from prism.data.masks import CHIPS


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


def chip_chebyshev_basis(degree: int, n_pixels: int = N_PIXELS) -> np.ndarray:
    """(3 * degree, n_pixels): an independent Chebyshev series on each detector chip
    (rows ``i*degree:(i+1)*degree`` live on chip i), zero in the gaps."""
    basis = np.zeros((len(CHIPS) * degree, n_pixels), np.float32)
    for i, (a, b) in enumerate(CHIPS):
        basis[i * degree:(i + 1) * degree, a:b] = chebyshev_basis(degree, b - a)
    return basis


def expand_gaps(x):
    """(..., 7514) values on the real pixels -> (..., 8575) with zeros in the chip gaps."""
    lead = x.shape[:-1]
    parts, prev, off = [], 0, 0
    for a, b in CHIPS:
        parts += [jnp.zeros(lead + (a - prev,), x.dtype), x[..., off:off + b - a]]
        off += b - a
        prev = b
    parts.append(jnp.zeros(lead + (N_PIXELS - prev,), x.dtype))
    return jnp.concatenate(parts, axis=-1)


def init_output(mlp: "MLP", *, bias=None, kernel_scale: float | None = None) -> None:
    """Set an MLP's output-layer bias and/or shrink its kernel (start near a chosen solution)."""
    if kernel_scale is not None:
        mlp.out.kernel[...] = mlp.out.kernel[...] * kernel_scale
    if bias is not None:
        mlp.out.bias[...] = jnp.broadcast_to(jnp.asarray(bias, jnp.float32), mlp.out.bias[...].shape)


class MLP(nnx.Module):
    """Dense stack; ``dtype`` is the compute dtype (bf16 for mixed precision), parameters stay fp32.
    The output is always returned in fp32."""

    def __init__(self, in_features: int, widths: Sequence[int], out_features: int, *,
                 hidden_act: Callable = nnx.swish, out_act: Callable | None = None, dtype=jnp.float32,
                 rngs: nnx.Rngs):
        dims = [in_features, *widths]
        lin = lambda a, b: nnx.Linear(a, b, dtype=dtype, param_dtype=jnp.float32, rngs=rngs)  # noqa: E731
        self.hidden = nnx.List([lin(a, b) for a, b in zip(dims[:-1], dims[1:])])
        self.out = lin(dims[-1], out_features)
        self.hidden_act = hidden_act
        self.out_act = out_act

    def __call__(self, x):
        for layer in self.hidden:
            x = self.hidden_act(layer(x))
        x = self.out(x).astype(jnp.float32)
        return x if self.out_act is None else self.out_act(x)


class ChebyshevContinuum(nnx.Module):
    """Pseudo-continuum = low-order Chebyshev series whose coefficients an MLP predicts."""

    def __init__(self, in_features: int, *, degree: int = 20, n_pixels: int = N_PIXELS, dtype=jnp.float32,
                 rngs: nnx.Rngs):
        self.coeffs = MLP(in_features, [64, 32], degree, dtype=dtype, rngs=rngs)
        self.basis = Const(jnp.asarray(chebyshev_basis(degree, n_pixels)))

    def __call__(self, x):
        return self.coeffs(x) @ self.basis[...]  # coefficients are fp32 (MLP output), basis product in fp32


class ChipContinuum(nnx.Module):
    """Pseudo-continuum with one low-order Chebyshev series per detector chip (ASPCAP
    normalises each chip separately); zero in the gaps. Initialised at k = 1."""

    def __init__(self, in_features: int, *, degree: int = 8, dtype=jnp.float32, rngs: nnx.Rngs):
        self.coeffs = MLP(in_features, [64, 32], len(CHIPS) * degree, dtype=dtype, rngs=rngs)
        bias = np.zeros(len(CHIPS) * degree, np.float32)
        bias[::degree] = 1.0  # T0 coefficient of every chip
        init_output(self.coeffs, bias=bias, kernel_scale=0.01)
        self.basis = Const(jnp.asarray(chip_chebyshev_basis(degree)))

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
