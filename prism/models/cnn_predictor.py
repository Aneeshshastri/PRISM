"""Heteroscedastic 1-D CNN: spectrum -> 23 label means + variances (inverse warm start / prior).

Port of CNNwarmstart.ipynb. Targets are ``labels[:, :23]`` from the prepared
cache: those are (physical [X/H] label - train mean) / train std over the same
train-only-imputed labels that ``cnn_label_stats.npz`` used to hold, so the
CNN's normalisation stats are simply ``stats.mean[:23]`` / ``stats.std[:23]``
and no separate stats file is needed.
"""

from __future__ import annotations

import jax.numpy as jnp
from flax import nnx

from prism.config import N_PIXELS, N_RAW_LABELS
from prism.losses import heteroscedastic_nll

# (features, kernel, stride) per conv block — conv -> ReLU -> BatchNorm
_CONVS = ((32, 16, 4), (64, 8, 4), (128, 4, 2), (256, 4, 2))


def clean_flux(flux):
    """Replace non-finite or <= 1e-3 flux with 1.0 (continuum), as in the TF pipeline."""
    flux = jnp.asarray(flux, jnp.float32)
    return jnp.where(jnp.isfinite(flux) & (flux > 1e-3), flux, 1.0)


class HeteroscedasticCNNPredictor(nnx.Module):
    def __init__(self, n_labels: int = N_RAW_LABELS, *, rngs: nnx.Rngs):
        convs, norms, cin = [], [], 1
        for feat, k, s in _CONVS:
            convs.append(nnx.Conv(cin, feat, kernel_size=(k,), strides=(s,), padding="SAME", rngs=rngs))
            norms.append(nnx.BatchNorm(feat, momentum=0.99, epsilon=1e-3, rngs=rngs))  # Keras defaults
            cin = feat
        self.convs = nnx.List(convs)
        self.norms = nnx.List(norms)
        self.dense1 = nnx.Linear(cin, 512, rngs=rngs)
        self.drop1 = nnx.Dropout(0.3, rngs=rngs)
        self.dense2 = nnx.Linear(512, 256, rngs=rngs)
        self.drop2 = nnx.Dropout(0.2, rngs=rngs)
        self.mean_head = nnx.Linear(256, n_labels, rngs=rngs)
        self.var_head = nnx.Linear(256, n_labels, rngs=rngs)

    def __call__(self, flux):
        h = jnp.asarray(flux, jnp.float32).reshape(flux.shape[0], N_PIXELS, 1)
        for conv, norm in zip(self.convs, self.norms):
            h = norm(nnx.relu(conv(h)))
        h = jnp.mean(h, axis=1)  # global average pooling
        h = self.drop1(nnx.relu(self.dense1(h)))
        h = self.drop2(nnx.relu(self.dense2(h)))
        return self.mean_head(h), self.var_head(h)


def make_cnn_loss():
    """Loss for ``Trainer``: heteroscedastic NLL on normalised labels; metric ``mse``."""

    def loss_fn(model, batch, key):
        del key
        mu, raw_var = model(clean_flux(batch["flux"]))
        y = batch["labels"][:, :N_RAW_LABELS]
        rm = batch["row_mask"]
        nll = heteroscedastic_nll(y, mu, raw_var, rm)
        mse = jnp.sum(jnp.mean((y - mu) ** 2, axis=-1) * rm) / jnp.maximum(jnp.sum(rm), 1.0)
        return nll, {"mse": mse}

    return loss_fn
