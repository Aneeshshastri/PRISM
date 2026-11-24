"""Run an NNX model as a pure function of (state, inputs) — for jit/vmap/jvp over fixed weights."""

from __future__ import annotations

import jax
from flax import nnx

# One pure apply function (and one jitted wrapper) per model *structure*, so repeated
# calls with new weights or new batches reuse compiled code instead of re-tracing.
_APPLY: dict = {}


def _apply_for(graphdef):
    if graphdef not in _APPLY:
        def fn(s, x):
            return nnx.merge(graphdef, s)(x)

        _APPLY[graphdef] = (fn, jax.jit(fn))
    return _APPLY[graphdef]


def make_flux_fn(model: nnx.Module, *, jit: bool = True):
    """Return ``(fn, state)`` with ``fn(state, x) == model(x)`` in eval mode.

    ``fn`` is shared by every model with the same structure (hashable, stable
    identity), so it can be a static argument of compiled inverse functions.
    """
    model.eval()
    graphdef, state = nnx.split(model)
    raw, jitted = _apply_for(graphdef)
    return (jitted if jit else raw), state
