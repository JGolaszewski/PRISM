"""Dataset and loader for preprocessed or augmented NDF grids."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


GRID_SIZE = 64
GRID_VALUES = GRID_SIZE * GRID_SIZE


class AugmentedNDFDataset(Dataset[torch.Tensor]):
    """Pack individual 64x64 grids into a reusable memory-mapped cache."""

    def __init__(self, data_dir: str | Path, share_memory: bool = False, grid_size: int = GRID_SIZE) -> None:
        if grid_size < 1:
            raise ValueError("grid_size must be positive")
        self.grid_size = grid_size
        self.grid_values = grid_size * grid_size
        self.data_dir = Path(data_dir)
        self.files = sorted(self.data_dir.glob("*.npy"))
        if not self.files:
            raise FileNotFoundError(f"No .npy NDF grids found in {self.data_dir}")

        cache_dir = self.data_dir.parent / ".cache"
        cache_path = cache_dir / f"{self.data_dir.name}_ndf{grid_size}.npy"
        newest_input_ns = max(path.stat().st_mtime_ns for path in self.files)
        valid_cache = False
        if cache_path.is_file() and cache_path.stat().st_mtime_ns >= newest_input_ns:
            try:
                cache = np.load(cache_path, mmap_mode="r", allow_pickle=False)
                valid_cache = (
                    cache.shape == (len(self.files), grid_size, grid_size)
                    and cache.dtype == np.dtype(np.float32)
                )
                del cache
            except (OSError, ValueError):
                valid_cache = False

        if not valid_cache:
            self._build_cache(cache_path)

        self._cache = np.load(cache_path, mmap_mode="r+", allow_pickle=False)
        flat_cache = self._cache.reshape(len(self.files), self.grid_values)
        self.samples = torch.from_numpy(flat_cache)
        if share_memory:
            self.samples = self.samples.share_memory_()

    def _load_grid(self, path: Path) -> np.ndarray:
        array = np.load(path, allow_pickle=False)
        if array.shape != (self.grid_size, self.grid_size):
            raise ValueError(f"{path}: expected {self.grid_size}x{self.grid_size} grid, got {array.shape}")
        if not np.isfinite(array).all():
            raise ValueError(f"{path}: contains NaN or infinite values")
        if np.any(array < 0.0):
            raise ValueError(f"{path}: contains negative NDF values")
        return array.astype(np.float32, copy=False)

    def _build_cache(self, cache_path: Path) -> None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = cache_path.with_name(cache_path.stem + ".tmp.npy")
        cache = np.lib.format.open_memmap(
            temporary_path,
            mode="w+",
            dtype=np.float32,
            shape=(len(self.files), self.grid_size, self.grid_size),
        )
        worker_count = min(len(self.files), max(4, min(16, os.cpu_count() or 4)))
        try:
            with ThreadPoolExecutor(max_workers=worker_count) as pool:
                for index, array in enumerate(pool.map(self._load_grid, self.files)):
                    cache[index] = array
            cache.flush()
            del cache
            temporary_path.replace(cache_path)
        except Exception:
            del cache
            temporary_path.unlink(missing_ok=True)
            raise

    def __len__(self) -> int:
        return self.samples.shape[0]

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.samples[index]


def get_dataloader(
    data_dir: str | Path,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4,
) -> DataLoader[torch.Tensor]:
    """Create a DataLoader backed by the dataset's shared in-memory cache."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if num_workers < 0:
        raise ValueError("num_workers cannot be negative")
    dataset = AugmentedNDFDataset(data_dir, share_memory=True)
    kwargs: dict[str, object] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = 2
    return DataLoader(**kwargs)  # type: ignore[arg-type]
