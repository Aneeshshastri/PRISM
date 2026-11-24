"""Memory-mapped batch iterator over the prepared cache."""

from __future__ import annotations

import queue
import threading
from pathlib import Path

import numpy as np

from prism.mesh import pad_to_multiple

DEFAULT_FIELDS = ("labels", "bad_mask", "flux", "ivar", "weights")


class BatchLoader:
    """Yields dicts of numpy arrays (plus ``row_mask``) for one split.

    Every batch has exactly ``batch_size`` rows; the last one is padded and
    its padding rows have ``row_mask == 0``. Shuffling is seeded with
    ``seed + epoch`` (see :meth:`set_epoch`).
    """

    def __init__(self, prep_dir: Path, split: str, batch_size: int, *, shuffle: bool, seed: int,
                 fields: tuple[str, ...] = DEFAULT_FIELDS, n_devices: int = 1, prefetch: int = 2):
        if batch_size % n_devices:
            raise ValueError(f"batch_size {batch_size} must be a multiple of n_devices {n_devices}")
        self.dir = Path(prep_dir) / split
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.fields = tuple(fields)
        self.prefetch = max(1, prefetch)
        self.epoch = 0
        self.arrays = {f: np.load(self.dir / f"{f}.npy", mmap_mode="r") for f in self.fields}
        self.n = len(next(iter(self.arrays.values())))

    def __len__(self) -> int:
        return -(-self.n // self.batch_size)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _order(self) -> np.ndarray:
        if not self.shuffle:
            return np.arange(self.n)
        return np.random.default_rng(self.seed + self.epoch).permutation(self.n)

    def _batch(self, idx: np.ndarray) -> dict[str, np.ndarray]:
        order = np.argsort(idx)
        inverse = np.empty_like(order)
        inverse[order] = np.arange(len(order))
        sorted_idx = idx[order]
        out = {}
        for name, arr in self.arrays.items():
            rows = np.asarray(arr[sorted_idx])[inverse]
            if rows.dtype == bool:
                rows = rows.astype(np.float32)
            out[name] = rows
        row_mask = np.ones(len(idx), np.float32)
        if len(idx) < self.batch_size:
            for name in out:
                out[name], row_mask = pad_to_multiple(out[name], self.batch_size)
        out["row_mask"] = row_mask
        return out

    def __iter__(self):
        order = self._order()
        chunks = [order[i:i + self.batch_size] for i in range(0, self.n, self.batch_size)]
        q: queue.Queue = queue.Queue(maxsize=self.prefetch)
        sentinel = object()

        def worker():
            try:
                for chunk in chunks:
                    q.put(self._batch(chunk))
            except BaseException as exc:  # noqa: BLE001 - re-raised in consumer
                q.put(exc)
            q.put(sentinel)

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        while True:
            item = q.get()
            if item is sentinel:
                break
            if isinstance(item, BaseException):
                raise item
            yield item
        thread.join()
