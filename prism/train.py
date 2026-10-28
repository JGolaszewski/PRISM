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
    create_grid_zernike_basis,
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


def compute_gradient_matching_loss(
    ndf_recon: torch.Tensor, target: torch.Tensor, grid_size: int = GRID_SIZE
) -> torch.Tensor:
    """Match adjacent log-NDF changes in both grid directions.

    This term penalizes ringing in the predicted profile while preserving
    target edges: it compares reconstruction gradients with target gradients
    instead of smoothing the prediction toward a flat field.
    """
    if ndf_recon.shape != target.shape:
        raise ValueError("ndf_recon and target must have matching shapes")
    if ndf_recon.shape[-1] != grid_size * grid_size:
        raise ValueError(f"expected flattened {grid_size}x{grid_size} grids")
    prediction = torch.log1p(ndf_recon.float().reshape(-1, grid_size, grid_size))
    reference = torch.log1p(target.float().reshape(-1, grid_size, grid_size))
    prediction_dx = prediction[..., :, 1:] - prediction[..., :, :-1]
    reference_dx = reference[..., :, 1:] - reference[..., :, :-1]
    prediction_dy = prediction[..., 1:, :] - prediction[..., :-1, :]
    reference_dy = reference[..., 1:, :] - reference[..., :-1, :]
    return 0.5 * (
        torch.mean(torch.abs(prediction_dx - reference_dx))
        + torch.mean(torch.abs(prediction_dy - reference_dy))
    )


def _losses(
    model: NDFPolynomialVAE,
    batch: torch.Tensor,
    basis: torch.Tensor,
    beta: float,
    peak_weight: float,
    mse_weight: float,
    coefficient_penalty: float,
    coefficient_degree_weights: torch.Tensor,
    gradient_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ndf_recon, coefficients, mu, logvar = model(batch, basis)
    targets = batch.reshape(batch.shape[0], -1)
    recon_loss = compute_reconstruction_loss(ndf_recon, targets, peak_weight)
    if mse_weight:
        recon_loss = recon_loss + mse_weight * torch.mean((ndf_recon - targets).square())
    if coefficient_penalty:
        recon_loss = recon_loss + coefficient_penalty * torch.mean(
            coefficients.square() * coefficient_degree_weights
        )
    if gradient_weight:
        recon_loss = recon_loss + gradient_weight * compute_gradient_matching_loss(
            ndf_recon, targets
        )
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
    mse_weight: float,
    coefficient_penalty: float,
    coefficient_degree_weights: torch.Tensor,
    gradient_weight: float,
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
                    model,
                    batch,
                    basis,
                    beta,
                    peak_weight,
                    mse_weight,
                    coefficient_penalty,
                    coefficient_degree_weights,
                    gradient_weight,
                )
            if training:
                total_loss.backward()
                gradient_clip = 1.0 if peak_weight > 30.0 else 5.0
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)
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
    basis_type: str,
    residual_decoder: bool,
    residual_decoder_version: int,
    grid_size: int,
) -> dict[str, Any]:
    return {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss_history": [entry["total"] for entry in train_history],
        "train_loss_history": train_history,
        "validation_loss_history": validation_history,
        "best_validation_loss": best_validation_loss,
        "grid_size": grid_size,
        "polynomial_degree": polynomial_degree,
        "basis_type": basis_type,
        "latent_dim": latent_dim,
        "num_poly_terms": model.num_poly_terms,
        "positive_coefficients": model.positive_coefficients,
        "residual_decoder": residual_decoder,
        "residual_decoder_version": residual_decoder_version if residual_decoder else 0,
        "beta": beta,
        "peak_weight": peak_weight,
    }


def _save_checkpoint(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".tmp")
    torch.save(state, temporary_path)
    temporary_path.replace(path)


def _initialize_shared_weights(
    model: NDFPolynomialVAE,
    checkpoint_path: Path,
    basis_type: str,
    polynomial_degree: int,
    latent_dim: int,
) -> None:
    """Warm-start shared VAE weights while leaving the target residual branch fresh."""
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Initialization checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    expected = {
        "basis_type": basis_type,
        "polynomial_degree": polynomial_degree,
        "latent_dim": latent_dim,
        "num_poly_terms": model.num_poly_terms,
    }
    for key, expected_value in expected.items():
        actual_value = checkpoint.get(key)
        if key == "basis_type" and actual_value == "complete_monomial":
            actual_value = "monomial"
        if actual_value != expected_value:
            raise ValueError(
                f"Initialization checkpoint {key}={actual_value!r} does not match "
                f"requested {expected_value!r}"
            )

    source_state = checkpoint.get("model_state_dict")
    if not isinstance(source_state, dict):
        raise ValueError(f"Initialization checkpoint has no model_state_dict: {checkpoint_path}")
    shared_state = {
        key: value
        for key, value in source_state.items()
        if not key.startswith("residual_decoder.")
    }
    incompatible = model.load_state_dict(shared_state, strict=False)
    missing_shared = [
        key for key in incompatible.missing_keys
        if not key.startswith("residual_decoder.")
    ]
    unexpected_shared = [
        key for key in incompatible.unexpected_keys
        if not key.startswith("residual_decoder.")
    ]
    if missing_shared or unexpected_shared:
        raise ValueError(
            "Initialization checkpoint shared weights are incompatible: "
            f"missing={missing_shared}, unexpected={unexpected_shared}"
        )


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
    basis_type: str = "zernike",
    scheduler: str = "none",
    beta_schedule: str = "fixed",
    beta_warmup_epochs: int = 10,
    mse_weight: float = 0.0,
    coefficient_penalty: float = 0.0,
    gradient_weight: float = 0.0,
    residual_decoder: bool = False,
    residual_decoder_version: int = 3,
    initialize_from: Path | None = None,
    grid_size: int = GRID_SIZE,
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
    if basis_type not in ("monomial", "zernike"):
        raise ValueError("basis_type must be 'monomial' or 'zernike'")
    if residual_decoder and residual_decoder_version not in (1, 2, 3):
        raise ValueError("residual_decoder_version must be 1, 2, or 3")
    if scheduler not in ("none", "cosine", "plateau"):
        raise ValueError("scheduler must be 'none', 'cosine', or 'plateau'")
    if beta_schedule not in ("fixed", "linear"):
        raise ValueError("beta_schedule must be 'fixed' or 'linear'")
    if beta_warmup_epochs < 1:
        raise ValueError("beta_warmup_epochs must be at least 1")
    if mse_weight < 0.0 or not math.isfinite(mse_weight):
        raise ValueError("mse_weight must be finite and nonnegative")
    if coefficient_penalty < 0.0 or not math.isfinite(coefficient_penalty):
        raise ValueError("coefficient_penalty must be finite and nonnegative")
    if gradient_weight < 0.0 or not math.isfinite(gradient_weight):
        raise ValueError("gradient_weight must be finite and nonnegative")
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
    dataset = AugmentedNDFDataset(data_dir, grid_size=grid_size)
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

    # Construct once before epoch/batch loops and retain on the selected device.
    basis = (
        create_grid_zernike_basis(grid_size, polynomial_degree)
        if basis_type == "zernike"
        else create_polynomial_basis(grid_size=grid_size, degree=polynomial_degree)
    ).to(device)
    coefficient_degree_weights = torch.tensor(
        [
            (degree / max(polynomial_degree, 1)) ** 4
            for degree in range(polynomial_degree + 1)
            for _ in range(degree + 1)
        ],
        dtype=torch.float32,
        device=device,
    )
    model = NDFPolynomialVAE(
        input_grid_size=grid_size,
        num_poly_terms=basis.shape[0],
        latent_dim=latent_dim,
        # Zernike polynomials are signed. Their coefficients must also be
        # signed to allow cancellation; nonnegativity is enforced after basis
        # reconstruction in NDFPolynomialVAE.forward.
        positive_coefficients=False,
        residual_decoder=residual_decoder,
        residual_decoder_version=residual_decoder_version,
    ).to(device)
    if initialize_from is not None:
        _initialize_shared_weights(
            model,
            initialize_from,
            basis_type,
            polynomial_degree,
            latent_dim,
        )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, fused=device.type == "cuda"
    )
    lr_scheduler = None
    if scheduler == "cosine":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=learning_rate * 0.01
        )
    elif scheduler == "plateau":
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=8, min_lr=learning_rate * 0.01
        )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    train_history: list[dict[str, float]] = []
    validation_history: list[dict[str, float]] = []
    best_validation_loss = float("inf")

    for epoch in range(1, epochs + 1):
        current_beta = beta
        if beta_schedule == "linear":
            current_beta *= min(1.0, epoch / beta_warmup_epochs)
        train_metrics = _run_epoch(
            model,
            samples,
            train_indices,
            batch_size,
            basis,
            device,
            current_beta,
            peak_weight,
            mse_weight,
            coefficient_penalty,
            coefficient_degree_weights,
            gradient_weight,
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
                current_beta,
                peak_weight,
                mse_weight,
                coefficient_penalty,
                coefficient_degree_weights,
                gradient_weight,
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
            basis_type,
            residual_decoder,
            residual_decoder_version,
            grid_size,
        )
        state["learning_rate"] = learning_rate
        state["scheduler"] = scheduler
        state["beta_schedule"] = beta_schedule
        state["beta_warmup_epochs"] = beta_warmup_epochs
        state["effective_beta"] = current_beta
        state["mse_weight"] = mse_weight
        state["coefficient_penalty"] = coefficient_penalty
        state["gradient_weight"] = gradient_weight
        state["initialize_from"] = (
            str(initialize_from) if initialize_from is not None else None
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
        if lr_scheduler is not None:
            if scheduler == "plateau":
                lr_scheduler.step(current_validation_loss)
            else:
                lr_scheduler.step()

    return model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--grid-size", type=int, choices=(64, 128), default=GRID_SIZE)
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
    parser.add_argument("--basis-type", choices=("monomial", "zernike"), default="zernike")
    parser.add_argument("--residual-decoder", action="store_true")
    parser.add_argument(
        "--residual-decoder-version", type=int, choices=(1, 2, 3), default=3
    )
    parser.add_argument("--initialize-from", type=Path, default=None)
    parser.add_argument("--peak-weight", type=float, default=DEFAULT_PEAK_WEIGHT)
    parser.add_argument(
        "--validation-fraction", type=float, default=DEFAULT_VALIDATION_FRACTION
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--scheduler", choices=("none", "cosine", "plateau"), default="none")
    parser.add_argument("--beta-schedule", choices=("fixed", "linear"), default="fixed")
    parser.add_argument("--beta-warmup-epochs", type=int, default=10)
    parser.add_argument("--mse-weight", type=float, default=0.0)
    parser.add_argument("--coefficient-penalty", type=float, default=0.0)
    parser.add_argument(
        "--gradient-weight", type=float, default=0.0,
        help="Weight for matching adjacent log-NDF gradients in x and y",
    )
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
            basis_type=args.basis_type,
            scheduler=args.scheduler,
            beta_schedule=args.beta_schedule,
            beta_warmup_epochs=args.beta_warmup_epochs,
            mse_weight=args.mse_weight,
            coefficient_penalty=args.coefficient_penalty,
            gradient_weight=args.gradient_weight,
            residual_decoder=args.residual_decoder,
            residual_decoder_version=args.residual_decoder_version,
            initialize_from=args.initialize_from,
            grid_size=args.grid_size,
        )
    except Exception as exc:
        parser.exit(1, f"Training failed: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
