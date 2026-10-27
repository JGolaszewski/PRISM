"""Generate physically plausible NDF variations from preprocessed grids."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "dataset" / "grid"
DEFAULT_OUTPUT = REPO_ROOT / "dataset" / "augmented"
GRID_SIZE = 64


def load_materials(input_dir: Path) -> tuple[list[Path], np.ndarray]:
    """Load all valid 64x64 MERL/RGL grids into a compact float32 array."""
    paths = sorted(
        path
        for path in input_dir.glob("*.npy")
        if path.name.startswith(("merl_", "rgl_"))
    )
    if not paths:
        raise FileNotFoundError(f"No merl_*.npy or rgl_*.npy grids found in {input_dir}")

    grids = []
    for path in paths:
        grid = np.load(path, allow_pickle=False)
        if grid.shape != (GRID_SIZE, GRID_SIZE):
            raise ValueError(f"{path}: expected a 64x64 grid, got {grid.shape}")
        if not np.isfinite(grid).all():
            raise ValueError(f"{path}: contains NaN or infinite values")
        if np.any(grid < 0):
            raise ValueError(f"{path}: NDF grid contains negative values")
        grids.append(grid.astype(np.float32, copy=False))
    return paths, np.stack(grids, axis=0)


def _bilinear_sample(grid: np.ndarray, source_x: np.ndarray, source_y: np.ndarray) -> np.ndarray:
    """Sample a square image at floating point pixel coordinates, with zero padding."""
    height, width = grid.shape
    inside = (
        (source_x >= 0.0)
        & (source_x <= width - 1)
        & (source_y >= 0.0)
        & (source_y <= height - 1)
    )

    # Clip only for safe indexing; `inside` keeps out-of-bounds samples zero.
    source_x = np.clip(source_x, 0.0, width - 1.0)
    source_y = np.clip(source_y, 0.0, height - 1.0)
    x0 = np.floor(source_x).astype(np.intp)
    y0 = np.floor(source_y).astype(np.intp)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    wx = source_x - x0
    wy = source_y - y0

    sampled = (
        grid[y0, x0] * (1.0 - wx) * (1.0 - wy)
        + grid[y0, x1] * wx * (1.0 - wy)
        + grid[y1, x0] * (1.0 - wx) * wy
        + grid[y1, x1] * wx * wy
    )
    return np.where(inside, sampled, 0.0)


def _normalize_and_mask(values: np.ndarray, disk_mask: np.ndarray) -> np.ndarray:
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    values = np.maximum(values, 0.0)
    values[~disk_mask] = 0.0
    peak = float(values.max(initial=0.0))
    if peak > 0.0 and np.isfinite(peak):
        values /= peak
    else:
        values.fill(0.0)
    values[~disk_mask] = 0.0
    return values.astype(np.float32)


def augment_one(
    materials: np.ndarray,
    rng: np.random.Generator,
    x: np.ndarray,
    y: np.ndarray,
    disk_mask: np.ndarray,
    *,
    return_indices: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Blend 2-3 materials and optionally return their selected row indices."""
    count = int(rng.integers(2, 4))
    indices = rng.choice(materials.shape[0], size=count, replace=False)
    weights = rng.dirichlet(np.ones(count))
    blended = np.tensordot(weights, materials[indices], axes=(0, 0)).astype(np.float64)

    # Inverse-map a rotated, radially dilated NDF onto the output grid. A scale
    # above one broadens the lobe; the inverse-square factor preserves its
    # approximate area in the 2D projected-normal domain before peak scaling.
    angle = rng.uniform(0.0, 2.0 * np.pi)
    roughness_scale = rng.uniform(0.85, 1.15)
    cosine, sine = np.cos(angle), np.sin(angle)
    source_x = (cosine * x + sine * y) / roughness_scale
    source_y = (-sine * x + cosine * y) / roughness_scale

    # Grid pixel centers span [-1, 1]; convert disk coordinates back to pixels.
    source_col = (source_x + 1.0) * ((GRID_SIZE - 1) / 2.0)
    source_row = (source_y + 1.0) * ((GRID_SIZE - 1) / 2.0)
    warped = _bilinear_sample(blended, source_col, source_row)
    warped *= 1.0 / (roughness_scale**2)

    # Small absolute noise relative to the unit-peak inputs; clamp negatives.
    noise_sigma = rng.uniform(0.001, 0.008)
    warped += rng.normal(0.0, noise_sigma, size=warped.shape)
    warped = np.maximum(warped, 0.0)
    sample = _normalize_and_mask(warped, disk_mask)
    if return_indices:
        return sample, indices
    return sample


def generate_dataset(
    input_dir: Path,
    output_dir: Path,
    num_samples: int = 30_000,
    seed: int | None = None,
    record_lineage: bool = True,
) -> int:
    if num_samples < 1:
        raise ValueError("num_samples must be at least 1")

    paths, materials = load_materials(input_dir)
    if materials.shape[0] < 3:
        raise ValueError("At least three base material grids are required")

    output_dir.mkdir(parents=True, exist_ok=True)
    coords = np.linspace(-1.0, 1.0, GRID_SIZE, dtype=np.float64)
    x, y = np.meshgrid(coords, coords, indexing="xy")
    disk_mask = x * x + y * y <= 1.0
    rng = np.random.default_rng(seed)

    lineage: dict[str, list[str]] = {}
    for sample_index in tqdm(range(num_samples), desc="Augmenting NDFs", unit="sample"):
        if record_lineage:
            sample, source_indices = augment_one(
                materials, rng, x, y, disk_mask, return_indices=True
            )
        else:
            sample = augment_one(materials, rng, x, y, disk_mask)
        path = output_dir / f"aug_sample_{sample_index:05d}.npy"
        np.save(path, sample, allow_pickle=False)
        if record_lineage:
            lineage[path.name] = [paths[int(index)].stem for index in source_indices]

    if record_lineage:
        manifest_path = output_dir / "lineage.json"
        temporary_path = manifest_path.with_suffix(".json.tmp")
        temporary_path.write_text(json.dumps(lineage, indent=2), encoding="utf-8")
        temporary_path.replace(manifest_path)
    return len(paths)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--num-samples", type=int, default=30_000,
        help="Number of augmented grids to generate (default: 30000).",
    )
    parser.add_argument(
        "--input-dir", type=Path, default=DEFAULT_INPUT,
        help="Directory containing the preprocessed 64x64 NDF grids.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT,
        help="Directory for generated .npy files.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    parser.add_argument(
        "--no-lineage", action="store_true",
        help="Skip writing the source-grid mapping sidecar.",
    )
    args = parser.parse_args()

    try:
        material_count = generate_dataset(
            args.input_dir, args.output_dir, args.num_samples, args.seed,
            record_lineage=not args.no_lineage,
        )
    except Exception as exc:
        parser.exit(1, f"Augmentation failed: {exc}\n")
    print(
        f"Generated {args.num_samples} samples from {material_count} materials "
        f"in {args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
