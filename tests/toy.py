"""Known-answer toy emulator for the inverse tests: flux = 1 + labels_norm @ A."""

import jax.numpy as jnp
import numpy as np
from flax import nnx

from prism.config import ABUND_INDICES, N_PIXELS
from prism.inverse.common import LOWER, UPPER


class LinearToy(nnx.Module):
    def __init__(self, A):
        self.A = nnx.Variable(jnp.asarray(A, jnp.float32))

    def __call__(self, labels, bad_mask=None):
        return 1.0 + labels @ self.A[...]


def make_problem(n_stars=4, seed=0, n_elem_pix=300):
    """Return (model, element_masks, theta_true (n,23), mean27, std27)."""
    rng = np.random.default_rng(seed)
    A = np.zeros((27, N_PIXELS), np.float32)
    A[:9] = rng.normal(0, 0.02, (9, N_PIXELS))
    A[23:] = rng.normal(0, 0.01, (4, N_PIXELS))
    masks = {}
    perm = rng.permutation(N_PIXELS)
    for k, j in enumerate(ABUND_INDICES):
        pix = perm[k * n_elem_pix:(k + 1) * n_elem_pix]
        A[j, pix] = rng.normal(0, 0.05, n_elem_pix)
        m = np.zeros(N_PIXELS, np.float32)
        m[pix] = 1.0
        masks[j] = m
    centre = (LOWER + UPPER) / 2
    span = UPPER - LOWER
    theta = np.empty((n_stars, 23), np.float32)
    theta[:, 0] = rng.uniform(4000, 6500, n_stars)
    theta[:, 1] = rng.uniform(1.0, 4.5, n_stars)
    theta[:, 2] = rng.uniform(-1.0, 0.3, n_stars)
    theta[:, 3:6] = rng.uniform([0.8, 2.0, 1.0], [2.0, 8.0, 10.0], (n_stars, 3))
    theta[:, 6:] = rng.uniform(-0.8, 0.3, (n_stars, 17))
    assert (theta > LOWER).all() and (theta < UPPER).all()
    del centre, span
    mean27 = np.zeros(27, np.float32)
    mean27[0] = 5000.0
    std27 = np.ones(27, np.float32)
    std27[0] = 500.0
    return LinearToy(A), masks, theta, mean27, std27
