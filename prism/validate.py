"""Evaluate a trained NDF VAE and export metrics and visual diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from prism.dataloader import AugmentedNDFDataset, GRID_SIZE
from prism.model import (
    NDFPolynomialVAE,
    create_legacy_bernstein_basis,
    create_polynomial_basis,
    select_device,
)
from prism.train import DEFAULT_CHECKPOINT_DIR


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "dataset" / "grid"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "eval_latest"
DEFAULT_BATCH_SIZE = 2048
PSNR_DATA_RANGE = 1.0  # Preprocessing peak-normalizes each material to [0, 1].


def _load_model(
    checkpoint_path: Path, device: torch.device
) -> tuple[NDFPolynomialVAE, torch.Tensor, dict[str, object]]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if "model_state_dict" not in checkpoint:
        raise ValueError(f"Checkpoint has no model_state_dict: {checkpoint_path}")

    grid_size = int(checkpoint.get("grid_size", GRID_SIZE))
    polynomial_degree = int(checkpoint.get("polynomial_degree", 4))
    latent_dim = int(checkpoint.get("latent_dim", 8))
    num_poly_terms = int(
        checkpoint.get("num_poly_terms", (polynomial_degree + 1) * (polynomial_degree + 2) // 2)
    )
    if grid_size != GRID_SIZE:
        raise ValueError(f"Expected {GRID_SIZE}x{GRID_SIZE} checkpoint, got {grid_size}x{grid_size}")

    basis_type = checkpoint.get("basis_type")
    is_legacy_bernstein = basis_type is None and num_poly_terms == (polynomial_degree + 1) ** 2
    if basis_type == "tensor_product_bernstein" or is_legacy_bernstein:
        basis = create_legacy_bernstein_basis(grid_size, polynomial_degree).to(device)
        positive_coefficients = True
    else:
        basis = create_polynomial_basis(grid_size, polynomial_degree).to(device)
        positive_coefficients = False
    if basis.shape[0] != num_poly_terms:
        raise ValueError(
            f"Checkpoint expects {num_poly_terms} polynomial terms, but degree "
            f"{polynomial_degree} generates {basis.shape[0]}"
        )
    model = NDFPolynomialVAE(
        input_grid_size=grid_size,
        num_poly_terms=num_poly_terms,
        latent_dim=latent_dim,
        positive_coefficients=positive_coefficients,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, basis, checkpoint


def _evaluate_metrics(
    model: NDFPolynomialVAE,
    basis: torch.Tensor,
    dataset: AugmentedNDFDataset,
    device: torch.device,
    batch_size: int,
) -> dict[str, float | int]:
    absolute_sum = 0.0
    squared_sum = 0.0
    log_absolute_sum = 0.0
    psnr_sum = 0.0
    value_count = 0

    with torch.no_grad():
        for start in tqdm(
            range(0, len(dataset), batch_size),
            desc="Evaluating dataset",
            unit="batch",
        ):
            targets = dataset.samples[start : start + batch_size].to(
                device, non_blocking=True
            )
            predictions, _, _, _ = model(targets, basis)
            targets = targets.reshape(targets.shape[0], -1).float()
            predictions = predictions.float()
            difference = predictions - targets
            absolute_sum += difference.abs().sum().item()
            squared_sum += difference.square().sum().item()
            sample_mse = difference.square().mean(dim=1).clamp_min(1e-24)
            psnr_sum += (10.0 * torch.log10(PSNR_DATA_RANGE**2 / sample_mse)).sum().item()
            log_difference = torch.log1p(predictions) - torch.log1p(targets)
            log_absolute_sum += log_difference.abs().sum().item()
            value_count += targets.numel()

    if value_count == 0:
        raise ValueError("Evaluation dataset is empty")
    mae = absolute_sum / value_count
    mse = squared_sum / value_count
    rmse = float(np.sqrt(mse))
    psnr = psnr_sum / len(dataset)
    return {
        "num_materials": len(dataset),
        "grid_size": GRID_SIZE,
        "mae": mae,
        "rmse": rmse,
        "psnr_db": psnr,
        "psnr_data_range": PSNR_DATA_RANGE,
        "log_scale_l1": log_absolute_sum / value_count,
    }


def _predict_samples(
    model: NDFPolynomialVAE,
    basis: torch.Tensor,
    dataset: AugmentedNDFDataset,
    indices: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sample_indices = torch.as_tensor(indices, dtype=torch.long)
    targets = dataset.samples.index_select(0, sample_indices).to(device)
    with torch.no_grad():
        predictions, _, mu, _ = model(targets, basis)
    return (
        targets.reshape(-1, GRID_SIZE, GRID_SIZE).cpu().numpy(),
        predictions.reshape(-1, GRID_SIZE, GRID_SIZE).float().cpu().numpy(),
        mu.float(),
    )


def _save_comparison_grid(
    output_dir: Path,
    dataset: AugmentedNDFDataset,
    indices: np.ndarray,
    targets: np.ndarray,
    predictions: np.ndarray,
) -> None:
    count = len(indices)
    figure, axes = plt.subplots(
        count, 4, figsize=(16, max(3.2 * count, 5.0)), squeeze=False
    )
    columns = ("Ground truth", "Reconstruction", "Absolute error", "Center row · log1p")
    for column, title in enumerate(columns):
        axes[0, column].set_title(title, fontsize=11)

    for row, (index, target, prediction) in enumerate(zip(indices, targets, predictions)):
        error = np.abs(target - prediction)
        vmax = max(float(target.max()), float(prediction.max()), 1e-8)
        axes[row, 0].imshow(target, origin="lower", extent=(-1, 1, -1, 1), vmin=0, vmax=vmax, cmap="magma")
        axes[row, 1].imshow(prediction, origin="lower", extent=(-1, 1, -1, 1), vmin=0, vmax=vmax, cmap="magma")
        image = axes[row, 2].imshow(error, origin="lower", extent=(-1, 1, -1, 1), cmap="viridis")
        figure.colorbar(image, ax=axes[row, 2], fraction=0.046, pad=0.03)
        axes[row, 3].plot(np.arange(GRID_SIZE), np.log1p(target[GRID_SIZE // 2]), label="GT", linewidth=1.5)
        axes[row, 3].plot(np.arange(GRID_SIZE), np.log1p(prediction[GRID_SIZE // 2]), label="Recon", linewidth=1.5)
        axes[row, 3].set_xlim(0, GRID_SIZE - 1)
        axes[row, 3].grid(alpha=0.25)
        axes[row, 3].legend(loc="best", fontsize=8)
        axes[row, 0].set_ylabel(dataset.files[int(index)].name, fontsize=8)
        for column in range(3):
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
        axes[row, 3].set_xlabel("Grid column")
    figure.suptitle("NDF reconstruction comparison", fontsize=14)
    figure.tight_layout(rect=(0, 0, 1, 0.99))
    figure.savefig(output_dir / "comparison_grid.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def _save_interpolation(
    output_dir: Path,
    model: NDFPolynomialVAE,
    basis: torch.Tensor,
    latent_a: torch.Tensor,
    latent_b: torch.Tensor,
    steps: int,
    device: torch.device,
    name_a: str,
    name_b: str,
) -> None:
    t_values = torch.linspace(0.0, 1.0, steps, device=device)
    with torch.no_grad():
        latent = (1.0 - t_values[:, None]) * latent_a + t_values[:, None] * latent_b
        coefficients = model.decoder(latent)
        decoded = coefficients @ basis
        if not model.positive_coefficients:
            decoded = torch.relu(decoded)
        grids = decoded.reshape(steps, GRID_SIZE, GRID_SIZE).cpu().numpy()

    figure, axes = plt.subplots(
        1, steps, figsize=(2.5 * steps, 3.2), squeeze=False, constrained_layout=True
    )
    vmax = max(float(np.max(grids)), 1e-8)
    image = None
    for index, axis in enumerate(axes[0]):
        image = axis.imshow(grids[index], origin="lower", extent=(-1, 1, -1, 1), vmin=0, vmax=vmax, cmap="magma")
        axis.set_title(f"t = {t_values[index].item():.2f}")
        axis.set_xticks([])
        axis.set_yticks([])
    figure.suptitle(f"Latent interpolation: {name_a} → {name_b}", fontsize=12)
    if image is not None:
        figure.colorbar(
            image,
            ax=axes.ravel().tolist(),
            shrink=0.78,
            pad=0.025,
            label="NDF",
        )
    figure.savefig(output_dir / "latent_interpolation.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def _save_prior_samples(
    output_dir: Path,
    model: NDFPolynomialVAE,
    basis: torch.Tensor,
    device: torch.device,
) -> None:
    with torch.no_grad():
        latent = torch.randn(9, model.latent_dim, device=device)
        coefficients = model.decoder(latent)
        decoded = coefficients @ basis
        if not model.positive_coefficients:
            decoded = torch.relu(decoded)
        grids = decoded.reshape(9, GRID_SIZE, GRID_SIZE).cpu().numpy()
    vmax = max(float(np.max(grids)), 1e-8)
    figure, axes = plt.subplots(3, 3, figsize=(8, 8), constrained_layout=True)
    for index, axis in enumerate(axes.flat):
        image = axis.imshow(grids[index], origin="lower", extent=(-1, 1, -1, 1), vmin=0, vmax=vmax, cmap="magma")
        axis.set_title(f"Prior sample {index + 1}")
        axis.set_xticks([])
        axis.set_yticks([])
    figure.colorbar(
        image,
        ax=axes.ravel().tolist(),
        shrink=0.88,
        pad=0.025,
        label="NDF",
    )
    figure.suptitle("Random generation from N(0, I)", fontsize=13)
    figure.savefig(output_dir / "prior_samples.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def validate(
    checkpoint_path: Path = DEFAULT_CHECKPOINT_DIR / "best_model.pt",
    data_dir: Path = DEFAULT_DATA_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    num_samples: int = 8,
    interp_steps: int = 7,
    batch_size: int = DEFAULT_BATCH_SIZE,
    seed: int = 42,
) -> dict[str, float | int]:
    if num_samples < 1:
        raise ValueError("num_samples must be at least 1")
    if interp_steps < 2:
        raise ValueError("interp_steps must be at least 2")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = select_device()
    dataset = AugmentedNDFDataset(data_dir)
    if len(dataset) < 2:
        raise ValueError("At least two NDF grids are needed for interpolation")
    model, basis, checkpoint = _load_model(checkpoint_path, device)

    metrics = _evaluate_metrics(model, basis, dataset, device, batch_size)
    rng = np.random.default_rng(seed)
    comparison_count = min(num_samples, len(dataset))
    comparison_indices = np.sort(rng.choice(len(dataset), comparison_count, replace=False))
    targets, predictions, latents = _predict_samples(
        model, basis, dataset, comparison_indices, device
    )
    _save_comparison_grid(output_dir, dataset, comparison_indices, targets, predictions)

    interpolation_indices = rng.choice(len(dataset), size=2, replace=False)
    interp_targets, _, interp_latents = _predict_samples(
        model, basis, dataset, interpolation_indices, device
    )
    del interp_targets
    name_a = dataset.files[int(interpolation_indices[0])].stem
    name_b = dataset.files[int(interpolation_indices[1])].stem
    _save_interpolation(
        output_dir,
        model,
        basis,
        interp_latents[0],
        interp_latents[1],
        interp_steps,
        device,
        name_a,
        name_b,
    )
    _save_prior_samples(output_dir, model, basis, device)

    report: dict[str, object] = {
        **metrics,
        "checkpoint": str(checkpoint_path),
        "data_dir": str(data_dir),
        "output_dir": str(output_dir),
        "device": str(device),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "latent_dim": model.latent_dim,
        "polynomial_degree": int(checkpoint.get("polynomial_degree", 4)),
        "basis_type": checkpoint.get("basis_type", "legacy_tensor_product_bernstein"),
        "comparison_samples": comparison_count,
        "interpolation_steps": interp_steps,
        "seed": seed,
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")

    print(f"Evaluation on {metrics['num_materials']} materials using {device}")
    print(
        f"MAE={metrics['mae']:.6f}  RMSE={metrics['rmse']:.6f}  "
        f"PSNR={metrics['psnr_db']:.2f} dB  Log-L1={metrics['log_scale_l1']:.6f}"
    )
    print(f"Saved plots and metrics to {output_dir}")
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_DIR / "best_model.pt")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--interp-steps", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        validate(
            checkpoint_path=args.checkpoint,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            num_samples=args.num_samples,
            interp_steps=args.interp_steps,
            batch_size=args.batch_size,
            seed=args.seed,
        )
    except Exception as exc:
        parser.exit(1, f"Evaluation failed: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
