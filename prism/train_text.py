"""Train a text-embedding to latent predictor against a frozen NDF VAE."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from prism.model import select_device
from prism.text_model import TextToLatent
from prism.train import compute_reconstruction_loss
from prism.validate import _load_model


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_CACHE = REPO_ROOT / "checkpoints" / "text_dataset_grid.npz"
DEFAULT_VAE_CHECKPOINT = REPO_ROOT / "checkpoints" / "best_model.pt"
DEFAULT_CHECKPOINT = REPO_ROOT / "checkpoints" / "best_text_model.pt"


def _load_artifact(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Text dataset artifact not found: {path}")
    with np.load(path, allow_pickle=False) as archive:
        data = {key: np.asarray(archive[key]) for key in archive.files}
    required = ("captions", "embeddings", "latents", "grids", "material_ids", "material_names", "source_paths", "embedding_model")
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"{path} is missing required arrays: {', '.join(missing)}")
    n = len(data["captions"])
    for key in required[:-1]:
        if data[key].ndim == 0 or len(data[key]) != n:
            raise ValueError(f"{key} must have first dimension N={n}")
    if data["embeddings"].ndim != 2 or data["latents"].ndim != 2:
        raise ValueError("embeddings and latents must be rank-2 [N, D/L] arrays")
    if data["grids"].shape not in ((n, 4096), (n, 64, 64)):
        raise ValueError(f"grids must have shape [N,4096] or [N,64,64], got {data['grids'].shape}")
    for key in ("material_ids", "material_names", "source_paths"):
        if data[key].ndim == 0 or len(data[key]) != n:
            raise ValueError(f"{key} must have length N={n}")
    if data["embedding_model"].size != 1:
        raise ValueError("embedding_model must be scalar metadata in the NPZ")
    for key in ("embeddings", "latents", "grids"):
        if not np.isfinite(data[key]).all():
            raise ValueError(f"{key} contains NaN or infinite values")
    if np.any(data["grids"] < 0):
        raise ValueError("grids must contain nonnegative NDF values")
    return data


def _split_by_material(
    material_ids: np.ndarray,
    validation_fraction: float,
    test_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not 0.0 < validation_fraction < 1.0 or not 0.0 < test_fraction < 1.0:
        raise ValueError("validation_fraction and test_fraction must be in (0, 1)")
    if validation_fraction + test_fraction >= 1.0:
        raise ValueError("validation_fraction + test_fraction must be less than 1")
    groups = np.unique(material_ids.astype(str))
    if len(groups) < 3:
        raise ValueError("at least three distinct material_ids are required for train/validation/test splits")
    rng = np.random.default_rng(seed)
    shuffled = groups.copy()
    rng.shuffle(shuffled)
    n_val = min(len(groups) - 2, max(1, round(len(groups) * validation_fraction)))
    n_test = min(len(groups) - n_val - 1, max(1, round(len(groups) * test_fraction)))
    validation_groups = set(shuffled[:n_val])
    test_groups = set(shuffled[n_val : n_val + n_test])
    ids = material_ids.astype(str)
    is_val = np.isin(ids, list(validation_groups))
    is_test = np.isin(ids, list(test_groups))
    return np.flatnonzero(~(is_val | is_test)), np.flatnonzero(is_val), np.flatnonzero(is_test)


def _run_epoch(
    predictor: TextToLatent,
    vae: nn.Module,
    basis: torch.Tensor,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    latent_weight: float,
    reconstruction_weight: float,
    peak_weight: float,
) -> dict[str, float]:
    training = optimizer is not None
    predictor.train(training)
    vae.eval()
    totals = torch.zeros(3, dtype=torch.float64, device=device)
    count = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for embedding, latent_target, grid_target in loader:
            embedding = embedding.to(device, non_blocking=True)
            latent_target = latent_target.to(device, non_blocking=True)
            grid_target = grid_target.to(device, non_blocking=True).reshape(embedding.shape[0], -1)
            if training:
                optimizer.zero_grad(set_to_none=True)
            latent_pred = predictor(embedding)
            latent_loss = torch.mean((latent_pred - latent_target).square())
            # Keep gradients to latent_pred while all VAE weights stay frozen.
            coefficients = vae.decoder(latent_pred)
            basis_kn = basis if basis.shape[0] == coefficients.shape[1] else basis.T
            reconstructed_grid = torch.relu(coefficients @ basis_kn)
            reconstruction_loss = compute_reconstruction_loss(
                reconstructed_grid, grid_target, peak_weight
            )
            loss = latent_weight * latent_loss + reconstruction_weight * reconstruction_loss
            if training:
                loss.backward()
                optimizer.step()
            batch_n = embedding.shape[0]
            totals += torch.stack(
                (loss.detach(), latent_loss.detach(), reconstruction_loss.detach())
            ).to(dtype=torch.float64) * batch_n
            count += batch_n
    if count == 0:
        raise ValueError("empty training or validation loader")
    means = (totals / count).cpu().tolist()
    return {"loss": means[0], "latent_mse": means[1], "reconstruction_loss": means[2]}


def train_text(
    data_cache: Path = DEFAULT_DATA_CACHE,
    vae_checkpoint: Path = DEFAULT_VAE_CHECKPOINT,
    checkpoint_path: Path = DEFAULT_CHECKPOINT,
    epochs: int = 200,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    latent_weight: float = 1.0,
    reconstruction_weight: float = 1.0,
    peak_weight: float = 15.0,
    validation_fraction: float = 0.2,
    test_fraction: float = 0.1,
    seed: int = 42,
    patience: int = 30,
) -> dict[str, Any]:
    if epochs < 1 or batch_size < 1 or patience < 1:
        raise ValueError("epochs, batch_size, and patience must be positive")
    if learning_rate <= 0 or weight_decay < 0:
        raise ValueError("learning_rate must be positive and weight_decay nonnegative")
    if latent_weight < 0 or reconstruction_weight < 0 or latent_weight + reconstruction_weight == 0:
        raise ValueError("loss weights must be nonnegative and not both zero")
    if not math.isfinite(peak_weight) or peak_weight < 0:
        raise ValueError("peak_weight must be finite and nonnegative")

    data = _load_artifact(data_cache)
    train_indices, val_indices, test_indices = _split_by_material(
        data["material_ids"], validation_fraction, test_fraction, seed
    )
    device = select_device()
    vae, basis, vae_meta = _load_model(vae_checkpoint, device)
    latent_dim = int(vae_meta["latent_dim"])
    if data["latents"].shape[1] != latent_dim:
        raise ValueError(
            f"artifact latent width {data['latents'].shape[1]} does not match VAE latent_dim {latent_dim}"
        )
    if data["grids"].shape[1:] not in ((4096,), (64, 64)):
        raise ValueError("expected 64x64 grids")
    for parameter in vae.parameters():
        parameter.requires_grad_(False)
    vae.eval()

    embeddings = torch.from_numpy(data["embeddings"].astype(np.float32, copy=False))
    latents = torch.from_numpy(data["latents"].astype(np.float32, copy=False))
    grids = torch.from_numpy(data["grids"].astype(np.float32, copy=False).reshape(-1, 4096))
    train_set = TensorDataset(embeddings[train_indices], latents[train_indices], grids[train_indices])
    val_set = TensorDataset(embeddings[val_indices], latents[val_indices], grids[val_indices])
    test_set = TensorDataset(embeddings[test_indices], latents[test_indices], grids[test_indices])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    predictor = TextToLatent(embeddings.shape[1], latent_dim).to(device)
    optimizer = torch.optim.AdamW(predictor.parameters(), lr=learning_rate, weight_decay=weight_decay)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    best_epoch = 0
    stale = 0
    history: list[dict[str, Any]] = []
    train_ids = set(data["material_ids"][train_indices].astype(str))
    val_ids = set(data["material_ids"][val_indices].astype(str))
    test_ids = set(data["material_ids"][test_indices].astype(str))
    if train_ids & val_ids or train_ids & test_ids or val_ids & test_ids:
        raise RuntimeError("material-level split leaked material IDs across train/validation/test")
    for epoch in range(1, epochs + 1):
        train_metrics = _run_epoch(
            predictor, vae, basis, train_loader, device, optimizer,
            latent_weight, reconstruction_weight, peak_weight,
        )
        val_metrics = _run_epoch(
            predictor, vae, basis, val_loader, device, None,
            latent_weight, reconstruction_weight, peak_weight,
        )
        row = {"epoch": epoch, "train": train_metrics, "validation": val_metrics}
        history.append(row)
        print(
            f"Epoch {epoch:03d}/{epochs}: train latent MSE={train_metrics['latent_mse']:.6f}, "
            f"heldout latent MSE={val_metrics['latent_mse']:.6f}, "
            f"heldout loss={val_metrics['loss']:.6f}"
        )
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            best_epoch = epoch
            stale = 0
            state: dict[str, Any] = {
                "model_state_dict": predictor.state_dict(),
                "embedding_dim": predictor.embedding_dim,
                "latent_dim": predictor.latent_dim,
                "hidden_dims": [512, 256],
                "activation": "SiLU",
                "data_cache": str(data_cache),
                "embedding_model": str(data["embedding_model"].reshape(-1)[0]),
                "vae_checkpoint": str(vae_checkpoint),
                "vae_basis_type": vae_meta.get("basis_type", "unknown"),
                "epoch": epoch,
                "validation_loss": best_val,
                "validation_latent_mse": val_metrics["latent_mse"],
                "validation_reconstruction_loss": val_metrics["reconstruction_loss"],
                "train_material_count": len(train_ids),
                "validation_material_count": len(val_ids),
                "test_material_count": len(test_ids),
                "train_sample_count": len(train_indices),
                "validation_sample_count": len(val_indices),
                "test_sample_count": len(test_indices),
                "validation_fraction": validation_fraction,
                "test_fraction": test_fraction,
                "seed": seed,
                "loss_weights": {"latent": latent_weight, "reconstruction": reconstruction_weight},
                "peak_weight": peak_weight,
                "optimizer": {"name": "AdamW", "learning_rate": learning_rate, "weight_decay": weight_decay},
            }
            torch.save(state, checkpoint_path)
        else:
            stale += 1
            if stale >= patience:
                print(f"Early stopping at epoch {epoch} (best epoch {best_epoch}).")
                break
    best_state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    predictor.load_state_dict(best_state["model_state_dict"])
    predictor.to(device)
    test_metrics = _run_epoch(
        predictor, vae, basis, test_loader, device, None,
        latent_weight, reconstruction_weight, peak_weight,
    )
    best_state["test_latent_mse"] = test_metrics["latent_mse"]
    best_state["test_reconstruction_loss"] = test_metrics["reconstruction_loss"]
    torch.save(best_state, checkpoint_path)
    result = {
        "checkpoint": str(checkpoint_path),
        "best_epoch": best_epoch,
        "validation_latent_mse": float(best_state["validation_latent_mse"]),
        "test_latent_mse": test_metrics["latent_mse"],
        "test_reconstruction_loss": test_metrics["reconstruction_loss"],
        "train_material_count": len(train_ids),
        "validation_material_count": len(val_ids),
        "test_material_count": len(test_ids),
        "history": history,
    }
    metrics_path = checkpoint_path.with_suffix(".json")
    metrics_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        f"Saved best text predictor to {checkpoint_path}; "
        f"test latent MSE={result['test_latent_mse']:.6f}"
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-cache", type=Path, default=DEFAULT_DATA_CACHE, help="NPZ with text embeddings, VAE latents, NDF grids, and material metadata")
    parser.add_argument("--vae-checkpoint", type=Path, default=DEFAULT_VAE_CHECKPOINT)
    parser.add_argument("--checkpoint", dest="checkpoint_path", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--latent-weight", type=float, default=1.0)
    parser.add_argument("--reconstruction-weight", type=float, default=1.0)
    parser.add_argument("--peak-weight", type=float, default=15.0)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=30)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    train_text(**vars(args))


if __name__ == "__main__":
    main()
