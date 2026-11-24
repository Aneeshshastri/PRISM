"""1-D device mesh helpers.

Parameters and optimizer state are replicated; the leading (batch / star) axis
of data is sharded over the ``'data'`` axis. The same code runs on one device,
on Kaggle's T4 x2, or on N fake CPU devices in tests.
"""

from __future__ import annotations

import jax
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

AXIS = "data"


def make_mesh(devices=None) -> Mesh:
    devices = list(jax.devices() if devices is None else devices)
    return Mesh(np.array(devices), (AXIS,))


def replicated(mesh: Mesh) -> NamedSharding:
    return NamedSharding(mesh, P())


def data_sharding(mesh: Mesh) -> NamedSharding:
    return NamedSharding(mesh, P(AXIS))


def n_devices(mesh: Mesh) -> int:
    return int(mesh.devices.size)


def pad_to_multiple(arr: np.ndarray, n: int, axis: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Pad ``arr`` along ``axis`` (repeating its last row) to a multiple of ``n``.

    Returns ``(padded, row_mask)`` where ``row_mask`` is float32 1 for real rows
    and 0 for padding. Padding repeats real data so it stays finite.
    """
    arr = np.asarray(arr)
    size = arr.shape[axis]
    target = -(-size // n) * n if size else n
    row_mask = np.zeros(target, np.float32)
    row_mask[:size] = 1.0
    if target == size:
        return arr, row_mask
    if size == 0:
        raise ValueError("cannot pad an empty array")
    idx = np.concatenate([np.arange(size), np.full(target - size, size - 1)])
    return np.take(arr, idx, axis=axis), row_mask


def shard_batch(batch: dict[str, np.ndarray], mesh: Mesh) -> dict[str, jax.Array]:
    """Place every array's leading axis on the ``'data'`` axis of ``mesh``."""
    nd = n_devices(mesh)
    sharding = data_sharding(mesh)
    out = {}
    for key, value in batch.items():
        value = np.asarray(value)
        if value.ndim == 0 or value.shape[0] % nd:
            raise ValueError(
                f"batch field {key!r} has leading size {value.shape[:1]}, "
                f"not divisible by the {nd} mesh devices; pad it first"
            )
        out[key] = jax.device_put(value, sharding)
    return out


def replicate(tree, mesh: Mesh):
    return jax.device_put(tree, replicated(mesh))
