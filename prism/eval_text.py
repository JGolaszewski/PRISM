"""Evaluate a text-to-latent checkpoint on material-grouped heldout samples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from prism.infer_text import (
    CONTRAST_PROMPTS,
    DEFAULT_COMPARISON_OUTPUT,
    _choose_device,
    generate_comparison,
    _load_predictor,
)
from prism.text_dataset import encode_texts, parse_profile_features, parse_prompt_controls
from prism.train import compute_reconstruction_loss
from prism.train_text import (
    DEFAULT_CHECKPOINT,
    DEFAULT_DATA_CACHE,
    DEFAULT_VAE_CHECKPOINT,
    _load_artifact,
    _material_ids_in_rows,
    _split_by_material,
    _split_from_material_groups,
)
from prism.validate import _load_model


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METRICS = REPO_ROOT / "checkpoints" / "best_text_model.json"
LOBE_COMPONENT_THRESHOLD = 0.2


def _qualitative_lobe_metrics(
    ndf: np.ndarray, relative_threshold: float = LOBE_COMPONENT_THRESHOLD
) -> dict[str, float | int]:
    """Report centroid-aware lobe shape and thresholded component concentration.

    Component membership uses 8-connected pixels at or above the configured
    fraction of the valid-disk peak. The reported share is that component's
    intensity mass divided by total intensity mass inside the disk.
    """
    values = np.asarray(ndf, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError(f"expected a square NDF grid, got shape {values.shape}")
    if not 0.0 < relative_threshold <= 1.0:
        raise ValueError("relative_threshold must be in (0, 1]")
    if not np.isfinite(values).all():
        raise ValueError("NDF grid must contain only finite values")

    size = values.shape[0]
    coords = np.linspace(-1.0, 1.0, size, dtype=np.float64)
    x, y = np.meshgrid(coords, coords, indexing="xy")
    disk = x * x + y * y <= 1.0
    weights = np.where(disk, np.maximum(values, 0.0), 0.0)
    total_mass = float(weights.sum())
    peak = float(weights.max(initial=0.0))
    if total_mass <= 1e-12 or peak <= 0.0:
        return {
            "intensity_centroid_x": 0.0,
            "intensity_centroid_y": 0.0,
            "centered_covariance_anisotropy": 0.0,
            "centered_covariance_eigenvalue_ratio": 0.0,
            "largest_component_intensity_share": 0.0,
            "component_count": 0,
            "component_threshold_fraction_of_peak": relative_threshold,
        }

    centroid_x = float((weights * x).sum() / total_mass)
    centroid_y = float((weights * y).sum() / total_mass)
    dx = x - centroid_x
    dy = y - centroid_y
    covariance_xx = float((weights * dx * dx).sum() / total_mass)
    covariance_yy = float((weights * dy * dy).sum() / total_mass)
    covariance_xy = float((weights * dx * dy).sum() / total_mass)
    eigenvalues = np.linalg.eigvalsh(
        np.asarray(
            [[covariance_xx, covariance_xy], [covariance_xy, covariance_yy]],
            dtype=np.float64,
        )
    )
    minor = max(float(eigenvalues[0]), 1e-12)
    major = max(float(eigenvalues[1]), 0.0)
    covariance_anisotropy = (major - minor) / max(major + minor, 1e-12)

    above_threshold = disk & (weights >= peak * relative_threshold)
    visited = np.zeros_like(above_threshold, dtype=np.bool_)
    height, width = above_threshold.shape
    largest_component_mass = 0.0
    component_count = 0
    for row, col in zip(*np.nonzero(above_threshold & ~visited)):
        if visited[row, col]:
            continue
        component_count += 1
        stack = [(int(row), int(col))]
        visited[row, col] = True
        component_mass = 0.0
        while stack:
            current_row, current_col = stack.pop()
            component_mass += float(weights[current_row, current_col])
            for neighbor_row in range(max(0, current_row - 1), min(height, current_row + 2)):
                for neighbor_col in range(max(0, current_col - 1), min(width, current_col + 2)):
                    if (
                        above_threshold[neighbor_row, neighbor_col]
                        and not visited[neighbor_row, neighbor_col]
                    ):
                        visited[neighbor_row, neighbor_col] = True
                        stack.append((neighbor_row, neighbor_col))
        largest_component_mass = max(largest_component_mass, component_mass)

    return {
        "intensity_centroid_x": centroid_x,
        "intensity_centroid_y": centroid_y,
        "centered_covariance_anisotropy": float(covariance_anisotropy),
        "centered_covariance_eigenvalue_ratio": float(major / minor),
        "largest_component_intensity_share": largest_component_mass / total_mass,
        "component_count": component_count,
        "component_threshold_fraction_of_peak": relative_threshold,
    }


def _generate_prompt_grids(
    prompts: tuple[str, ...],
    predictor: torch.nn.Module,
    vae: torch.nn.Module,
    basis: torch.Tensor,
    device: torch.device,
    embedding_model: str,
) -> list[np.ndarray]:
    """Decode prompt grids for additive structural metrics without rendering."""
    embeddings = encode_texts(
        list(prompts), embedding_model, batch_size=min(32, len(prompts)), device=device
    ).to(device)
    profile_features = (
        torch.from_numpy(np.stack([parse_profile_features(prompt) for prompt in prompts])).to(device)
        if getattr(predictor, "profile_dim", 0)
        else None
    )
    control_features = (
        torch.from_numpy(np.stack([parse_prompt_controls(prompt) for prompt in prompts])).to(device)
        if getattr(predictor, "control_dim", 0)
        else None
    )
    with torch.inference_mode():
        predicted = predictor(embeddings, profile_features, control_features)
        decoded = vae.decode_ndf(predicted, basis)
    size = int(getattr(vae, "input_grid_size", round(decoded.shape[-1] ** 0.5)))
    return [
        grid
        for grid in decoded.reshape(len(prompts), size, size).float().cpu().numpy()
    ]


def evaluate_text(
    data_cache: Path = DEFAULT_DATA_CACHE,
    vae_checkpoint: Path = DEFAULT_VAE_CHECKPOINT,
    text_checkpoint: Path = DEFAULT_CHECKPOINT,
    metrics_path: Path = DEFAULT_METRICS,
    batch_size: int = 128,
    device_name: str = "auto",
    peak_weight: float = 15.0,
    prompt_grid: Path | None = DEFAULT_COMPARISON_OUTPUT,
) -> dict[str, Any]:
    """Measure heldout latent and NDF error, then optionally render prompt contrasts."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    data = _load_artifact(data_cache)
    predictor, text_meta = _load_predictor(text_checkpoint, _choose_device(device_name))
    cache_encoder = str(np.asarray(data["embedding_model"]).reshape(-1)[0])
    checkpoint_encoder = str(text_meta.get("embedding_model", cache_encoder))
    if cache_encoder != checkpoint_encoder:
        raise ValueError(
            f"cache encoder {cache_encoder!r} does not match checkpoint encoder {checkpoint_encoder!r}"
        )
    if data["embeddings"].shape[1] != predictor.embedding_dim:
        raise ValueError(
            f"cache embedding width {data['embeddings'].shape[1]} does not match predictor width {predictor.embedding_dim}"
        )
    seed = int(text_meta.get("seed", 42))
    val_fraction = float(text_meta.get("validation_fraction", 0.2))
    test_fraction = float(text_meta.get("test_fraction", 0.1))
    split_strategy = str(text_meta.get("split_strategy", "random"))
    manifest_groups = text_meta.get("split_material_groups")
    if isinstance(manifest_groups, dict):
        train_idx, val_idx, test_idx = _split_from_material_groups(
            data["material_ids"],
            manifest_groups["train"],
            manifest_groups["validation"],
            manifest_groups["test"],
        )
    else:
        train_idx, val_idx, test_idx = _split_by_material(
            data["material_ids"], val_fraction, test_fraction, seed,
            split_strategy, data["latents"],
        )
    if len(test_idx) == 0:
        raise ValueError("material-grouped test split is empty")
    test_material_ids = sorted(_material_ids_in_rows(data["material_ids"], test_idx))
    if len(test_material_ids) != int(text_meta.get("test_material_count", len(test_material_ids))):
        raise ValueError("cache split does not match the material count recorded by the text checkpoint")

    device = _choose_device(device_name)
    vae, basis, vae_meta = _load_model(vae_checkpoint, device)
    if int(text_meta.get("latent_dim", predictor.latent_dim)) != int(vae.latent_dim):
        raise ValueError("text predictor latent width does not match evaluation VAE")
    saved_vae = text_meta.get("vae_checkpoint")
    if saved_vae and Path(str(saved_vae)).resolve() != vae_checkpoint.resolve():
        raise ValueError(
            f"checkpoint was trained against VAE {saved_vae!s}; requested evaluation VAE is {vae_checkpoint}"
        )
    if text_meta.get("vae_basis_type") and text_meta["vae_basis_type"] != vae_meta.get("basis_type"):
        raise ValueError("text predictor and evaluation VAE use different basis types")
    for parameter in vae.parameters():
        parameter.requires_grad_(False)
    vae.eval()
    predictor = predictor.to(device).eval()

    embeddings = torch.from_numpy(data["embeddings"].astype(np.float32, copy=False)[test_idx])
    targets = torch.from_numpy(data["latents"].astype(np.float32, copy=False)[test_idx])
    grids = torch.from_numpy(data["grids"].astype(np.float32, copy=False)[test_idx].reshape(-1, 4096))
    if predictor.profile_dim:
        if "profile_features" in data:
            profile_features = torch.from_numpy(
                data["profile_features"].astype(np.float32, copy=False)[test_idx]
            )
        else:
            profile_features = torch.zeros((len(test_idx), predictor.profile_dim), dtype=torch.float32)
        if profile_features.shape[1] != predictor.profile_dim:
            raise ValueError("cache profile feature width does not match text predictor")
    else:
        profile_features = torch.zeros((len(test_idx), 0), dtype=torch.float32)
    loader = DataLoader(
        TensorDataset(embeddings, targets, grids, profile_features),
        batch_size=batch_size, shuffle=False,
    )
    latent_sq_error = 0.0
    latent_elements = 0
    cos_sum = 0.0
    recon_weighted_sum = 0.0
    log_l1_sum = 0.0
    grid_elements = 0
    with torch.inference_mode():
        for batch_embedding, batch_target, batch_grid, batch_profile_features in loader:
            batch_embedding = batch_embedding.to(device)
            batch_target = batch_target.to(device)
            batch_grid = batch_grid.to(device)
            batch_profile_features = batch_profile_features.to(device)
            predicted = predictor(batch_embedding, batch_profile_features)
            decoded = vae.decode_ndf(predicted, basis).reshape_as(batch_grid)
            latent_sq_error += float((predicted - batch_target).square().sum())
            latent_elements += batch_target.numel()
            cos_sum += float(F.cosine_similarity(predicted, batch_target, dim=1).sum())
            recon_weighted_sum += float(
                compute_reconstruction_loss(decoded, batch_grid, peak_weight) * batch_grid.numel()
            )
            log_l1_sum += float(
                (torch.log1p(decoded.float()) - torch.log1p(batch_grid.float())).abs().sum()
            )
            grid_elements += batch_grid.numel()

    result: dict[str, Any] = {
        "text_checkpoint": str(text_checkpoint),
        "data_cache": str(data_cache),
        "vae_checkpoint": str(vae_checkpoint),
        "embedding_model": cache_encoder,
        "split": "material_grouped_test",
        "split_strategy": split_strategy,
        "seed": seed,
        "train_material_count": len(_material_ids_in_rows(data["material_ids"], train_idx)),
        "validation_material_count": len(_material_ids_in_rows(data["material_ids"], val_idx)),
        "test_material_count": len(test_material_ids),
        "test_sample_count": len(test_idx),
        "heldout_material_ids": test_material_ids,
        "heldout_latent_mse": latent_sq_error / latent_elements,
        "heldout_latent_cosine_similarity": cos_sum / len(test_idx),
        "test_reconstruction_log_l1": log_l1_sum / grid_elements,
        "test_specular_weighted_reconstruction_loss": recon_weighted_sum / grid_elements,
        "specular_peak_weight": peak_weight,
    }
    if prompt_grid is not None:
        _, prompt_shape_metrics = generate_comparison(
            CONTRAST_PROMPTS, prompt_grid, vae_checkpoint, text_checkpoint, device,
            return_metrics=True,
        )
        prompt_grids = _generate_prompt_grids(
            CONTRAST_PROMPTS,
            predictor,
            vae,
            basis,
            device,
            checkpoint_encoder,
        )
        result["qualitative_prompt_grid"] = str(prompt_grid)
        result["qualitative_prompts"] = [
            {
                "prompt": prompt,
                "shape_metrics": shape,
                "lobe_metrics": _qualitative_lobe_metrics(
                    grid, LOBE_COMPONENT_THRESHOLD
                ),
            }
            for prompt, shape, grid in zip(
                CONTRAST_PROMPTS, prompt_shape_metrics, prompt_grids
            )
        ]

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, Any] = {}
    if metrics_path.exists():
        try:
            loaded = json.loads(metrics_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                existing = loaded
        except (json.JSONDecodeError, OSError):
            existing = {}
    existing["heldout_evaluation"] = result
    metrics_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved heldout evaluation to {metrics_path}")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-cache", type=Path, default=DEFAULT_DATA_CACHE)
    parser.add_argument("--vae-checkpoint", type=Path, default=DEFAULT_VAE_CHECKPOINT)
    parser.add_argument("--text-checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps")
    parser.add_argument("--peak-weight", type=float, default=15.0)
    parser.add_argument("--prompt-grid", type=Path, default=DEFAULT_COMPARISON_OUTPUT)
    parser.add_argument("--skip-prompt-grid", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        evaluate_text(
            data_cache=args.data_cache,
            vae_checkpoint=args.vae_checkpoint,
            text_checkpoint=args.text_checkpoint,
            metrics_path=args.metrics,
            batch_size=args.batch_size,
            device_name=args.device,
            peak_weight=args.peak_weight,
            prompt_grid=None if args.skip_prompt_grid else args.prompt_grid,
        )
    except (FileNotFoundError, RuntimeError, ValueError, OSError) as exc:
        raise SystemExit(f"Text evaluation failed: {exc}") from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
