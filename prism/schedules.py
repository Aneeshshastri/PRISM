"""Learning-rate schedules."""

from __future__ import annotations

import jax.numpy as jnp
import optax


def cosine_decay_restarts(init_lr: float, first_decay_steps: int, t_mul: float = 2.0,
                          m_mul: float = 1.0, alpha: float = 0.0) -> optax.Schedule:
    """Exactly Keras ``CosineDecayRestarts`` (SGDR with period growth ``t_mul``)."""

    def schedule(step):
        frac = jnp.asarray(step, jnp.float32) / first_decay_steps
        if t_mul == 1.0:
            i_restart = jnp.floor(frac)
            frac = frac - i_restart
        else:
            i_restart = jnp.floor(jnp.log1p(-frac * (1.0 - t_mul)) / jnp.log(t_mul))
            sum_r = (1.0 - t_mul ** i_restart) / (1.0 - t_mul)
            frac = (frac - sum_r) / t_mul ** i_restart
        cosine = 0.5 * (m_mul ** i_restart) * (1.0 + jnp.cos(jnp.pi * frac))
        return init_lr * ((1.0 - alpha) * cosine + alpha)

    return schedule


def warmup_cosine(peak_lr: float, total_steps: int, warmup_steps: int = 0,
                  final_fraction: float = 1e-3) -> optax.Schedule:
    """Linear warm-up to ``peak_lr`` then a single cosine decay to ``peak_lr * final_fraction``
    at ``total_steps`` (no restarts, so every model trains on the same schedule)."""
    total_steps = max(int(total_steps), int(warmup_steps) + 1)
    return optax.warmup_cosine_decay_schedule(init_value=peak_lr * final_fraction, peak_value=peak_lr,
                                              warmup_steps=int(warmup_steps), decay_steps=total_steps,
                                              end_value=peak_lr * final_fraction)
