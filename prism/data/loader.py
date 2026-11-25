"""Memory-mapped batch iterator over the prepared cache."""

from __future__ import annotations

import os
import queue
import threading
from pathlib import Path

import numpy as np

from prism.mesh import pad_to_multiple

DEFAULT_FIELDS = ("labels", "bad_mask", "flux", "spl_weight", "weights")


def _load_and_drop_cache(path: Path, mm: np.ndarray, n: int) -> np.ndarray:
    """Copy the first ``n`` rows into RAM, then tell the kernel to drop the file's page
    cache so the cached copy doesn't double memory use and push the RAM copy into swap."""
    out = np.array(mm[:n])
    if hasattr(os, "posix_fadvise"):
        fd = os.open(path, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
    return out


class BatchLoader:
    """Yields dicts of numpy arrays (plus ``row_mask``) for one split.

    Every batch has exactly ``batch_size`` rows; the last one is padded and its
    padding rows have ``row_mask == 0``.

    With ``in_memory`` the split is read into RAM once and rows are shuffled
    individually. Otherwise (memmap) the I/O pattern matters: the cache can be far larger than RAM, and random single-row
    reads are ~10x slower than contiguous ones (measured ~18 vs ~180 MB/s on WSL). So
    rows are read only as contiguous ``block_rows`` slices. With ``shuffle`` the
    block order is permuted (seed ``seed + epoch``, see :meth:`set_epoch`), and
    ``window_blocks`` blocks are loaded together and shuffled row-wise, so each batch
    mixes rows from many places in the dataset.
    """

    def __init__(self, prep_dir: Path, split: str, batch_size: int, *, shuffle: bool, seed: int,
                 fields: tuple[str, ...] = DEFAULT_FIELDS, n_devices: int = 1, prefetch: int = 2,
                 limit: int | None = None, block_rows: int = 64, window_blocks: int = 64,
                 in_memory: bool = False):
        if batch_size % n_devices:
            raise ValueError(f"batch_size {batch_size} must be a multiple of n_devices {n_devices}")
        self.dir = Path(prep_dir) / split
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.fields = tuple(fields)
        self.prefetch = max(1, prefetch)
        self.block_rows = max(1, int(block_rows))
        self.window_blocks = max(1, int(window_blocks))
        self.epoch = 0
        self.arrays = {f: np.load(self.dir / f"{f}.npy", mmap_mode="r") for f in self.fields}
        self.n = len(next(iter(self.arrays.values())))
        if limit is not None:  # use only the first `limit` rows (quick runs, benchmarks)
            self.n = min(self.n, int(limit))
        self.in_memory = in_memory
        if in_memory:  # one sequential read up front; every epoch afterwards is RAM-speed
            self.arrays = {f: _load_and_drop_cache(self.dir / f"{f}.npy", a, self.n)
                           for f, a in self.arrays.items()}

    def __len__(self) -> int:
        return -(-self.n // self.batch_size)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _read(self, name: str, start: int, stop: int) -> np.ndarray:
        """One contiguous slice of one field (the only way rows are read)."""
        rows = np.array(self.arrays[name][start:stop])
        return rows.astype(np.float32) if rows.dtype == bool else rows

    def _row_batches(self):
        """In-RAM data: random row gathers are cheap, so shuffle individual rows."""
        rng = np.random.default_rng(self.seed + self.epoch)
        order = rng.permutation(self.n) if self.shuffle else np.arange(self.n)
        for s in range(0, self.n, self.batch_size):
            idx = order[s:s + self.batch_size]
            batch = {}
            for f, a in self.arrays.items():
                rows = a[idx]
                batch[f] = rows.astype(np.float32) if rows.dtype == bool else rows
            row_mask = np.ones(len(idx), np.float32)
            if len(idx) < self.batch_size:
                for f in self.fields:
                    batch[f], row_mask = pad_to_multiple(batch[f], self.batch_size)
            batch["row_mask"] = row_mask
            yield batch

    def _batches(self):
        if self.in_memory:
            yield from self._row_batches()
            return
        rng = np.random.default_rng(self.seed + self.epoch)
        blocks = [(s, min(s + self.block_rows, self.n)) for s in range(0, self.n, self.block_rows)]
        if self.shuffle:
            blocks = [blocks[i] for i in rng.permutation(len(blocks))]
        carry = None
        for w in range(0, len(blocks), self.window_blocks):
            window = blocks[w:w + self.window_blocks]
            data = {f: np.concatenate([self._read(f, s, e) for s, e in window]) for f in self.fields}
            if self.shuffle:
                perm = rng.permutation(len(data[self.fields[0]]))
                data = {f: v[perm] for f, v in data.items()}
            if carry is not None:
                data = {f: np.concatenate([carry[f], data[f]]) for f in self.fields}
            n_rows = len(data[self.fields[0]])
            full = n_rows - n_rows % self.batch_size
            for s in range(0, full, self.batch_size):
                batch = {f: v[s:s + self.batch_size] for f, v in data.items()}
                batch["row_mask"] = np.ones(self.batch_size, np.float32)
                yield batch
            carry = {f: v[full:] for f, v in data.items()} if full < n_rows else None
        if carry is not None:
            batch = {}
            for f, v in carry.items():
                batch[f], row_mask = pad_to_multiple(v, self.batch_size)
            batch["row_mask"] = row_mask
            yield batch

    def __iter__(self):
        q: queue.Queue = queue.Queue(maxsize=self.prefetch)
        sentinel = object()

        def worker():
            try:
                for batch in self._batches():
                    q.put(batch)
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
