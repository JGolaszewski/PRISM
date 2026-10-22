"""Train the polynomial VAE on preprocessed or augmented NDF grids."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from prism.dataloader import AugmentedNDFDataset, GRID_SIZE
from prism.model import (
    DEFAULT_POLYNOMIAL_DEGREE,
    NDFPolynomialVAE,
    create_polynomial_basis,
    select_device,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "dataset" / "augmented"
DEFAULT_CHECKPOINT_DIR = REPO_ROOT / "checkpoints"
DEFAULT_BETA = 0.001
DEFAULT_VALIDATION_FRACTION = 0.1
DEFAULT_PEAK_WEIGHT = 15.0


def compute_reconstruction_loss(
    ndf_recon: torch.Tensor,
    target: torch.Tensor,
    peak_weight_scale: float = DEFAULT_PEAK_WEIGHT,
) -> torch.Tensor:
    """Log-space L1 loss weighted by target intensity to emphasize highlights."""
    if peak_weight_scale < 0.0 or not math.isfinite(peak_weight_scale):
        raise ValueError("peak_weight_scale must be finite and nonnegative")
    ndf_recon = ndf_recon.float()
    target = target.float()
    weights = 1.0 + peak_weight_scale * target
    return torch.mean(weights * torch.abs(torch.log1p(ndf_recon) - torch.log1p(target)))


def _losses(
    model: NDFPolynomialVAE,
    batch: torch.Tensor,
    basis: torch.Tensor,
    beta: float,
    peak_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ndf_recon, _, mu, logvar = model(batch, basis)
    targets = batch.reshape(batch.shape[0], -1)
    recon_loss = compute_reconstruction_loss(ndf_recon, targets, peak_weight)
    mu, logvar = mu.float(), logvar.float()
    kl_loss = -0.5 * torch.mean(1.0 + logvar - mu.square() - logvar.exp())
    return recon_loss + beta * kl_loss, recon_loss, kl_loss


def _run_epoch(
    model: NDFPolynomialVAE,
    samples: torch.Tensor,
    indices: torch.Tensor,
    batch_size: int,
    basis: torch.Tensor,
    device: torch.device,
    beta: float,
    peak_weight: float,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    epochs: int,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = torch.zeros(3, dtype=torch.float64, device=device)
    sample_count = int(indices.numel())
    context = torch.enable_grad() if training else torch.no_grad()
    epoch_indices = indices
    if training:
        epoch_indices = indices[torch.randperm(sample_count, device=device)]
    use_amp = device.type == "cuda"

    with context:
        batch_starts = range(0, sample_count, batch_size)
        progress = tqdm(
            batch_starts, desc=f"Epoch {epoch}/{epochs}", unit="batch", leave=False
        )
        for start in progress:
            batch_indices = epoch_indices[start : start + batch_size]
            batch = samples.index_select(0, batch_indices)
            if training:
                optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=use_amp,
            ):
                total_loss, recon_loss, kl_loss = _losses(
                    model, batch, basis, beta, peak_weight
                )
            if training:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

            current_size = batch.shape[0]
            totals += torch.stack(
                (total_loss.detach(), recon_loss.detach(), kl_loss.detach())
            ).to(torch.float64) * current_size

    if sample_count == 0:
        raise ValueError("DataLoader produced no samples")
    means = (totals / sample_count).cpu().tolist()
    return dict(zip(("total", "reconstruction", "kl"), means))


def _checkpoint_state(
    model: NDFPolynomialVAE,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    train_history: list[dict[str, float]],
    validation_history: list[dict[str, float]],
    best_validation_loss: float,
    latent_dim: int,
    beta: float,
    polynomial_degree: int,
    peak_weight: float,
) -> dict[str, Any]:
    return {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss_history": [entry["total"] for entry in train_history],
        "train_loss_history": train_history,
        "validation_loss_history": validation_history,
        "best_validation_loss": best_validation_loss,
        "grid_size": GRID_SIZE,
        "polynomial_degree": polynomial_degree,
        "basis_type": "complete_monomial",
        "latent_dim": latent_dim,
        "num_poly_terms": model.num_poly_terms,
        "beta": beta,
        "peak_weight": peak_weight,
    }


def _save_checkpoint(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".tmp")
    torch.save(state, temporary_path)
    temporary_path.replace(path)


def train_ndf_vae(
    data_dir: Path = DEFAULT_DATA_DIR,
    checkpoint_dir: Path = DEFAULT_CHECKPOINT_DIR,
    epochs: int = 120,
    batch_size: int = 2048,
    learning_rate: float = 1e-3,
    latent_dim: int = 8,
    save_freq: int = 5,
    beta: float = DEFAULT_BETA,
    polynomial_degree: int = DEFAULT_POLYNOMIAL_DEGREE,
    peak_weight: float = DEFAULT_PEAK_WEIGHT,
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
    seed: int | None = None,
) -> NDFPolynomialVAE:
    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if learning_rate <= 0.0 or not math.isfinite(learning_rate):
        raise ValueError("learning rate must be a finite positive number")
    if save_freq < 1:
        raise ValueError("save_freq must be at least 1")
    if polynomial_degree not in (4, 6, 8):
        raise ValueError("polynomial_degree must be one of 4, 6, or 8")
    if peak_weight < 0.0 or not math.isfinite(peak_weight):
        raise ValueError("peak_weight must be finite and nonnegative")
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in [0, 1)")

    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    # Cache once, then keep the compact 500 MB grid tensor on the GPU for all
    # epochs. This avoids per-batch disk, multiprocessing, and PCIe overhead.
    dataset = AugmentedNDFDataset(data_dir)
    sample_count = len(dataset)
    if sample_count < 2:
        raise ValueError("At least two NDF grids are required for training")

    validation_count = (
        max(1, int(round(sample_count * validation_fraction)))
        if validation_fraction > 0.0
        else 0
    )
    validation_count = min(validation_count, sample_count - 1)
    train_count = sample_count - validation_count
    split_seed = seed if seed is not None else torch.seed()
    split_generator = torch.Generator().manual_seed(split_seed)
    permutation = torch.randperm(sample_count, generator=split_generator)
    train_indices_cpu = permutation[validation_count:]
    validation_indices_cpu = permutation[:validation_count]

    device = select_device()
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    elif os.cpu_count():
        torch.set_num_threads(min(os.cpu_count() or 1, 24))

    # Transfer split indices once and batch by indexing the resident tensor.
    train_indices = train_indices_cpu.to(device)
    validation_indices = validation_indices_cpu.to(device)
    samples = dataset.samples.to(device, non_blocking=True)
    print(f"Training on {device} using {train_count} samples")
    if validation_count:
        print(f"Validation set: {validation_count} samples")

    basis = create_polynomial_basis(
        grid_size=GRID_SIZE, degree=polynomial_degree
    ).to(device)
    model = NDFPolynomialVAE(
        input_grid_size=GRID_SIZE,
        num_poly_terms=basis.shape[0],
        latent_dim=latent_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, fused=device.type == "cuda"
    )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    train_history: list[dict[str, float]] = []
    validation_history: list[dict[str, float]] = []
    best_validation_loss = float("inf")

    for epoch in range(1, epochs + 1):
        train_metrics = _run_epoch(
            model,
            samples,
            train_indices,
            batch_size,
            basis,
            device,
            beta,
            peak_weight,
            optimizer,
            epoch,
            epochs,
        )
        train_history.append(train_metrics)
        validation_metrics = (
            _run_epoch(
                model,
                samples,
                validation_indices,
                batch_size,
                basis,
                device,
                beta,
                peak_weight,
                None,
                epoch,
                epochs,
            )
            if validation_count > 0
            else train_metrics
        )
        validation_history.append(validation_metrics)
        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"train={train_metrics['total']:.6f} "
            f"(recon={train_metrics['reconstruction']:.6f}, kl={train_metrics['kl']:.6f}) | "
            f"validation={validation_metrics['total']:.6f}"
        )

        state = _checkpoint_state(
            model,
            optimizer,
            epoch,
            train_history,
            validation_history,
            best_validation_loss,
            latent_dim,
            beta,
            polynomial_degree,
            peak_weight,
        )
        current_validation_loss = validation_metrics["total"]
        if current_validation_loss < best_validation_loss:
            best_validation_loss = current_validation_loss
            state["best_validation_loss"] = best_validation_loss
            _save_checkpoint(checkpoint_dir / "best_model.pt", state)
            print(f"Saved new best model: {checkpoint_dir / 'best_model.pt'}")

        if epoch % save_freq == 0 or epoch == epochs:
            _save_checkpoint(
                checkpoint_dir / f"checkpoint_epoch_{epoch:02d}.pt", state
            )

    return model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--latent-dim", type=int, default=8)
    parser.add_argument("--save-freq", type=int, default=5)
    parser.add_argument("--beta", type=float, default=DEFAULT_BETA)
    parser.add_argument(
        "--poly-degree", type=int, choices=(4, 6, 8),
        default=DEFAULT_POLYNOMIAL_DEGREE,
    )
    parser.add_argument("--peak-weight", type=float, default=DEFAULT_PEAK_WEIGHT)
    parser.add_argument(
        "--validation-fraction", type=float, default=DEFAULT_VALIDATION_FRACTION
    )
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    try:
        train_ndf_vae(
            data_dir=args.data_dir,
            checkpoint_dir=args.checkpoint_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            latent_dim=args.latent_dim,
            save_freq=args.save_freq,
            beta=args.beta,
            polynomial_degree=args.poly_degree,
            peak_weight=args.peak_weight,
            validation_fraction=args.validation_fraction,
            seed=args.seed,
        )
    except Exception as exc:
        parser.exit(1, f"Training failed: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
