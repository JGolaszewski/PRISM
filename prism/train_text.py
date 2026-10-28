"""Train a text-embedding to latent predictor against a frozen NDF VAE."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
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
DEFAULT_VAE_CHECKPOINT = (
    REPO_ROOT / "checkpoints" / "zernike_residual_v3_warm_plateau42" / "best_model.pt"
)
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
    if data["grids"].ndim not in (2, 3) or (data["grids"].ndim == 3 and data["grids"].shape[1] != data["grids"].shape[2]):
        raise ValueError(f"grids must be flattened or square [N,H,W], got {data['grids'].shape}")
    for key in ("material_ids", "material_names", "source_paths"):
        if data[key].ndim == 0 or len(data[key]) != n:
            raise ValueError(f"{key} must have length N={n}")
    if data["embedding_model"].size != 1:
        raise ValueError("embedding_model must be scalar metadata in the NPZ")
    for key in ("embeddings", "latents", "grids"):
        if not np.isfinite(data[key]).all():
            raise ValueError(f"{key} contains NaN or infinite values")
    if "profile_features" in data:
        if data["profile_features"].shape != (n, 25):
            raise ValueError(f"profile_features must have shape [N,25], got {data['profile_features'].shape}")
        if not np.isfinite(data["profile_features"]).all():
            raise ValueError("profile_features contains NaN or infinite values")
    if "control_features" in data:
        if data["control_features"].shape != (n, 7):
            raise ValueError(f"control_features must have shape [N,7], got {data['control_features'].shape}")
        if not np.isfinite(data["control_features"]).all():
            raise ValueError("control_features contains NaN or infinite values")
    if np.any(data["grids"] < 0):
        raise ValueError("grids must contain nonnegative NDF values")
    return data


def _split_by_material(
    material_ids: np.ndarray,
    validation_fraction: float,
    test_fraction: float,
    seed: int,
    strategy: str = "random",
    latent_targets: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not 0.0 < validation_fraction < 1.0 or not 0.0 < test_fraction < 1.0:
        raise ValueError("validation_fraction and test_fraction must be in (0, 1)")
    if validation_fraction + test_fraction >= 1.0:
        raise ValueError("validation_fraction + test_fraction must be less than 1")
    row_members = [
        tuple(member for member in str(value).split("|") if member)
        for value in material_ids.astype(str)
    ]
    groups = np.unique(np.asarray([member for members in row_members for member in members], dtype=str))
    if len(groups) < 3:
        raise ValueError("at least three distinct material_ids are required for train/validation/test splits")
    rng = np.random.default_rng(seed)
    n_val = min(len(groups) - 2, max(1, round(len(groups) * validation_fraction)))
    n_test = min(len(groups) - n_val - 1, max(1, round(len(groups) * test_fraction)))
    if strategy == "random":
        shuffled = groups.copy()
        rng.shuffle(shuffled)
    elif strategy == "norm_stratified":
        if latent_targets is None or len(latent_targets) != len(material_ids):
            raise ValueError("norm_stratified splitting requires one latent target row per material_id row")
        if not np.isfinite(latent_targets).all():
            raise ValueError("latent targets for norm_stratified splitting must be finite")
        norms_by_group: dict[str, list[float]] = {str(group): [] for group in groups}
        for members, target in zip(row_members, latent_targets):
            if len(members) == 1:
                norms_by_group[members[0]].append(float(np.linalg.norm(target)))
        known = [group for group in groups if norms_by_group[str(group)]]
        missing = [group for group in groups if not norms_by_group[str(group)]]
        known.sort(key=lambda group: (float(np.median(norms_by_group[str(group)])), str(group)))
        bands = [band for band in np.array_split(np.asarray(known, dtype=str), 4) if len(band)]
        if missing:
            bands.append(np.asarray(missing, dtype=str))
        training_groups: set[str] = set()
        validation_groups: set[str] = set()
        test_groups: set[str] = set()
        n_train = len(groups) - n_val - n_test
        proportions = np.asarray([n_train, n_val, n_test], dtype=np.float64) / len(groups)
        for band in bands:
            shuffled_band = band.copy()
            rng.shuffle(shuffled_band)
            raw_counts = len(band) * proportions
            counts = np.floor(raw_counts).astype(int)
            remainder = len(band) - int(counts.sum())
            for bucket in np.argsort(-(raw_counts - counts))[:remainder]:
                counts[bucket] += 1
            train_end = counts[0]
            val_end = train_end + counts[1]
            training_groups.update(map(str, shuffled_band[:train_end]))
            validation_groups.update(map(str, shuffled_band[train_end:val_end]))
            test_groups.update(map(str, shuffled_band[val_end:]))
    else:
        raise ValueError("strategy must be 'random' or 'norm_stratified'")
    if strategy == "random":
        validation_groups = set(map(str, shuffled[:n_val]))
        test_groups = set(map(str, shuffled[n_val : n_val + n_test]))
        training_groups = set(map(str, shuffled[n_val + n_test :]))
    train_indices: list[int] = []
    val_indices: list[int] = []
    test_indices: list[int] = []
    for index, members in enumerate(row_members):
        member_set = set(members)
        if member_set and member_set <= training_groups:
            train_indices.append(index)
        elif member_set and member_set <= validation_groups:
            val_indices.append(index)
        elif member_set and member_set <= test_groups:
            test_indices.append(index)
        # Composite lineage rows spanning partitions are excluded to prevent
        # any base material from appearing in more than one split.
    return np.asarray(train_indices), np.asarray(val_indices), np.asarray(test_indices)


def _split_from_material_groups(
    material_ids: np.ndarray,
    train_material_ids: list[str] | set[str],
    validation_material_ids: list[str] | set[str],
    test_material_ids: list[str] | set[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply a predeclared complete material partition to artifact rows."""
    groups = [set(map(str, values)) for values in (
        train_material_ids, validation_material_ids, test_material_ids
    )]
    if any(not group for group in groups):
        raise ValueError("manifest train, validation, and test material groups must be nonempty")
    if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
        raise ValueError("manifest material groups overlap")
    row_members = [
        set(member for member in str(value).split("|") if member)
        for value in material_ids.astype(str)
    ]
    all_groups = set().union(*row_members)
    if set.union(*groups) != all_groups:
        missing = all_groups - set.union(*groups)
        unknown = set.union(*groups) - all_groups
        raise ValueError(
            f"manifest must cover exactly the artifact material IDs; missing={sorted(missing)[:5]}, "
            f"unknown={sorted(unknown)[:5]}"
        )
    indices: list[list[int]] = [[], [], []]
    for index, members in enumerate(row_members):
        for partition, allowed in enumerate(groups):
            if members and members <= allowed:
                indices[partition].append(index)
                break
    if any(not values for values in indices):
        raise ValueError("manifest material partition produced an empty row split")
    return tuple(np.asarray(values, dtype=np.int64) for values in indices)  # type: ignore[return-value]


def _material_ids_in_rows(material_ids: np.ndarray, indices: np.ndarray) -> set[str]:
    """Expand composite source-material IDs for split counts and leakage checks."""
    return {
        member
        for value in material_ids[indices].astype(str)
        for member in value.split("|")
        if member
    }


def _valid_disk_tv(grid: torch.Tensor, grid_size: int = 64) -> torch.Tensor:
    """Mean total variation over horizontal/vertical pairs inside the disk."""
    values = grid.reshape(-1, grid_size, grid_size).float()
    coords = torch.linspace(-1.0, 1.0, grid_size, device=grid.device)
    y, x = torch.meshgrid(coords, coords, indexing="ij")
    valid = x.square() + y.square() <= 1.0
    horizontal = valid[:, 1:] & valid[:, :-1]
    vertical = valid[1:, :] & valid[:-1, :]
    hdiff = (values[:, :, 1:] - values[:, :, :-1]).abs()
    vdiff = (values[:, 1:, :] - values[:, :-1, :]).abs()
    h_tv = (hdiff * horizontal).sum() / horizontal.sum().clamp_min(1)
    v_tv = (vdiff * vertical).sum() / vertical.sum().clamp_min(1)
    return 0.5 * (h_tv + v_tv)


def _decoded_control_loss(
    grid: torch.Tensor,
    controls: torch.Tensor,
    grid_size: int = 64,
) -> torch.Tensor:
    """Match prompted controls to differentiable statistics of the decoded NDF."""
    values = grid.reshape(-1, grid_size, grid_size).float()
    controls = controls.float()
    coords = torch.linspace(-1.0, 1.0, grid_size, device=grid.device)
    y, x = torch.meshgrid(coords, coords, indexing="ij")
    radius = torch.sqrt(x.square() + y.square())
    disk = radius <= 1.0
    values = values.clamp_min(0.0) * disk
    total = values.sum(dim=(1, 2)).clamp_min(1e-8)
    mean_radius = (values * radius).sum(dim=(1, 2)) / total
    xx = (values * x.square()).sum(dim=(1, 2)) / total
    yy = (values * y.square()).sum(dim=(1, 2)) / total
    xy = (values * x * y).sum(dim=(1, 2)) / total
    delta_x = xx - yy
    delta_y = 2.0 * xy
    trace = (xx + yy).clamp_min(1e-8)
    predicted_anisotropy_vector = torch.stack((delta_y, delta_x), dim=-1) / trace[:, None]
    anisotropy = torch.linalg.vector_norm(predicted_anisotropy_vector, dim=-1)

    rough_mask = controls[:, 4]
    anisotropy_mask = controls[:, 5]
    orientation_mask = controls[:, 6] * (controls[:, 1] >= 0.05)
    rough_loss = (
        nn.functional.smooth_l1_loss(mean_radius, controls[:, 0], reduction="none")
        * rough_mask
    ).sum() / rough_mask.sum().clamp_min(1.0)
    target_axis = controls[:, 2:4]
    target_anisotropy_vector = controls[:, 1:2] * target_axis
    vector_mask = anisotropy_mask * orientation_mask
    vector_loss = nn.functional.smooth_l1_loss(
        predicted_anisotropy_vector,
        target_anisotropy_vector,
        reduction="none",
    ).mean(dim=-1)
    magnitude_loss = nn.functional.smooth_l1_loss(
        anisotropy, controls[:, 1], reduction="none"
    )
    anisotropy_loss = (
        vector_loss * vector_mask + magnitude_loss * (anisotropy_mask - vector_mask)
    ).sum() / anisotropy_mask.sum().clamp_min(1.0)
    return rough_loss + anisotropy_loss


def _run_epoch(
    predictor: TextToLatent,
    vae: nn.Module,
    basis: torch.Tensor,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    latent_weight: float,
    cosine_weight: float,
    reconstruction_weight: float,
    magnitude_weight: float,
    peak_weight: float,
    tv_weight: float,
    control_weight: float,
    grid_size: int,
) -> dict[str, float]:
    training = optimizer is not None
    predictor.train(training)
    vae.eval()
    totals = torch.zeros(8, dtype=torch.float64, device=device)
    count = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for embedding, latent_target, grid_target, profile_features, control_features in loader:
            embedding = embedding.to(device, non_blocking=True)
            latent_target = latent_target.to(device, non_blocking=True)
            grid_target = grid_target.to(device, non_blocking=True).reshape(embedding.shape[0], -1)
            profile_features = profile_features.to(device, non_blocking=True)
            control_features = control_features.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            amp_context = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if device.type == "cuda"
                else nullcontext()
            )
            with amp_context:
                latent_pred = predictor(embedding, profile_features, control_features)
                # Keep gradients to latent_pred while all VAE weights stay frozen.
                reconstructed_grid = (
                    vae.decode_ndf(latent_pred, basis)
                    if reconstruction_weight > 0 or tv_weight > 0 or control_weight > 0
                    else None
                )
            latent_loss = torch.mean((latent_pred.float() - latent_target).square())
            cosine_loss = (1.0 - nn.functional.cosine_similarity(
                latent_pred.float(), latent_target, dim=-1, eps=1e-8
            )).mean()
            magnitude_loss = nn.functional.smooth_l1_loss(
                torch.log1p(torch.linalg.vector_norm(latent_pred.float(), dim=-1)),
                torch.log1p(torch.linalg.vector_norm(latent_target, dim=-1)),
            )
            reconstruction_loss = (
                compute_reconstruction_loss(reconstructed_grid, grid_target, peak_weight)
                if reconstructed_grid is not None
                else latent_loss.new_zeros(())
            )
            tv_loss = (
                _valid_disk_tv(reconstructed_grid, grid_size)
                if reconstructed_grid is not None and tv_weight > 0
                else latent_loss.new_zeros(())
            )
            control_loss = (
                _decoded_control_loss(reconstructed_grid, control_features, grid_size)
                if reconstructed_grid is not None and control_weight > 0 and control_features.shape[-1] >= 7
                else latent_loss.new_zeros(())
            )
            loss = (
                latent_weight * latent_loss
                + cosine_weight * cosine_loss
                + reconstruction_weight * reconstruction_loss
                + magnitude_weight * magnitude_loss
                + tv_weight * tv_loss
                + control_weight * control_loss
            )
            if training:
                loss.backward()
                optimizer.step()
            batch_n = embedding.shape[0]
            totals += torch.stack(
                (
                    loss.detach(), latent_loss.detach(), cosine_loss.detach(),
                    reconstruction_loss.detach(), magnitude_loss.detach(), tv_loss.detach(),
                    control_loss.detach(),
                    torch.linalg.vector_norm(latent_pred - latent_target, dim=-1).mean().detach(),
                )
            ).to(dtype=torch.float64) * batch_n
            count += batch_n
    if count == 0:
        raise ValueError("empty training or validation loader")
    means = (totals / count).cpu().tolist()
    return {
        "loss": means[0],
        "latent_mse": means[1],
        "cosine_loss": means[2],
        "reconstruction_loss": means[3],
        "magnitude_loss": means[4],
        "tv_loss": means[5],
        "control_loss": means[6],
        "latent_rmse": means[7],
    }


def train_text(
    data_cache: Path = DEFAULT_DATA_CACHE,
    additional_data_caches: list[Path] | None = None,
    training_only_data_caches: list[Path] | None = None,
    training_only_sample_limit: int = 5_000,
    vae_checkpoint: Path = DEFAULT_VAE_CHECKPOINT,
    checkpoint_path: Path = DEFAULT_CHECKPOINT,
    epochs: int = 200,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    latent_weight: float = 1.0,
    cosine_weight: float = 0.1,
    reconstruction_weight: float = 1.0,
    magnitude_weight: float = 0.0,
    peak_weight: float = 15.0,
    tv_weight: float = 0.01,
    control_weight: float = 1.0,
    control_validation_weight: float = 0.0,
    validation_fraction: float = 0.2,
    test_fraction: float = 0.1,
    seed: int = 42,
    split_strategy: str = "random",
    patience: int = 30,
    scheduler_t0: int = 20,
    scheduler_t_mult: int = 2,
    minimum_learning_rate: float = 1e-5,
    dropout: float = 0.15,
    hidden_dims: tuple[int, int] = (512, 256),
    selection_metric: str = "latent_mse",
    split_manifest_path: Path | None = None,
    initialize_from: Path | None = None,
    freeze_text_path: bool = False,
    model_seed: int | None = None,
    skip_test_eval: bool = False,
) -> dict[str, Any]:
    if epochs < 1 or batch_size < 1 or patience < 1:
        raise ValueError("epochs, batch_size, and patience must be positive")
    if training_only_sample_limit < 0:
        raise ValueError("training_only_sample_limit must be nonnegative (0 means use all rows)")
    if learning_rate <= 0 or weight_decay < 0:
        raise ValueError("learning_rate must be positive and weight_decay nonnegative")
    if min(latent_weight, cosine_weight, reconstruction_weight, magnitude_weight, tv_weight, control_weight) < 0 or (
        latent_weight + cosine_weight + reconstruction_weight + magnitude_weight + tv_weight + control_weight == 0
    ):
        raise ValueError("loss weights must be nonnegative and not all zero")
    if not math.isfinite(control_validation_weight) or control_validation_weight < 0:
        raise ValueError("control_validation_weight must be finite and nonnegative")
    if scheduler_t0 < 1 or scheduler_t_mult < 1:
        raise ValueError("scheduler_t0 and scheduler_t_mult must be positive")
    if minimum_learning_rate < 0 or minimum_learning_rate > learning_rate:
        raise ValueError("minimum_learning_rate must be between zero and learning_rate")
    if not 0.0 <= dropout < 1.0:
        raise ValueError("dropout must be in [0, 1)")
    if len(hidden_dims) != 2 or min(hidden_dims) < 1:
        raise ValueError("hidden_dims must contain two positive widths")
    if selection_metric not in ("latent_mse", "loss", "reconstruction_loss"):
        raise ValueError(
            "selection_metric must be 'latent_mse', 'loss', or 'reconstruction_loss'"
        )
    if not math.isfinite(peak_weight) or peak_weight < 0:
        raise ValueError("peak_weight must be finite and nonnegative")

    cache_paths = [data_cache, *(additional_data_caches or [])]
    loaded_caches = [_load_artifact(path) for path in cache_paths]
    training_only_paths = list(training_only_data_caches or [])
    training_only_caches = [_load_artifact(path) for path in training_only_paths]
    embedding_models = [str(item["embedding_model"].reshape(-1)[0]) for item in loaded_caches]
    if len(set(embedding_models)) != 1:
        raise ValueError(f"all training caches must use the same embedding model, got {embedding_models}")
    if any(str(item["embedding_model"].reshape(-1)[0]) != embedding_models[0] for item in training_only_caches):
        raise ValueError("training-only caches must use the same embedding model as the named cache")
    for path, item in [*zip(cache_paths, loaded_caches), *zip(training_only_paths, training_only_caches)]:
        if "vae_checkpoint" not in item:
            continue
        cached_vae = Path(str(item["vae_checkpoint"].reshape(-1)[0])).expanduser()
        if not cached_vae.is_absolute():
            cached_vae = REPO_ROOT / cached_vae
        requested_vae = vae_checkpoint.expanduser()
        if not requested_vae.is_absolute():
            requested_vae = REPO_ROOT / requested_vae
        if cached_vae.resolve() != requested_vae.resolve():
            raise ValueError(
                "text cache was encoded with VAE checkpoint "
                f"{cached_vae.resolve()}, but training requested {requested_vae.resolve()}; "
                "rebuild the cache or select the matching VAE checkpoint"
            )
    data: dict[str, np.ndarray] = {}
    row_keys = (
        "captions", "embeddings", "latents", "grids", "material_ids",
        "material_names", "source_paths",
    )
    for key in row_keys:
        data[key] = np.concatenate([item[key] for item in loaded_caches], axis=0)
    profile_dims = [
        item["profile_features"].shape[1] if "profile_features" in item else 0
        for item in loaded_caches
    ]
    profile_dim = max(profile_dims, default=0)
    if profile_dim and any(dim not in (0, profile_dim) for dim in profile_dims):
        raise ValueError(f"incompatible profile feature widths across caches: {profile_dims}")
    data["profile_features"] = np.concatenate([
        item["profile_features"] if "profile_features" in item
        else np.zeros((len(item["captions"]), profile_dim), dtype=np.float32)
        for item in loaded_caches
    ], axis=0) if profile_dim else np.zeros((len(data["captions"]), 0), dtype=np.float32)
    control_dim = max(
        (item["control_features"].shape[1] if "control_features" in item else 0)
        for item in [*loaded_caches, *training_only_caches]
    )
    if control_dim not in (0, 7):
        raise ValueError(f"control feature width must be 7, got {control_dim}")
    data["control_features"] = np.concatenate([
        item["control_features"] if "control_features" in item
        else np.zeros((len(item["captions"]), control_dim), dtype=np.float32)
        for item in loaded_caches
    ], axis=0) if control_dim else np.zeros((len(data["captions"]), 0), dtype=np.float32)
    data["embedding_model"] = loaded_caches[0]["embedding_model"]
    data["vae_checkpoint"] = loaded_caches[0].get("vae_checkpoint", np.asarray(str(vae_checkpoint)))
    initialization_seed = seed if model_seed is None else model_seed
    torch.manual_seed(initialization_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(initialization_seed)
    split_material_groups: dict[str, list[str]] | None = None
    if split_manifest_path is None:
        train_indices, val_indices, test_indices = _split_by_material(
            data["material_ids"], validation_fraction, test_fraction, seed,
            split_strategy, data["latents"],
        )
    else:
        manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
        selected_split = next(
            (item for item in manifest.get("splits", []) if int(item.get("seed", -1)) == seed),
            None,
        )
        if selected_split is None:
            raise ValueError(f"split manifest has no split entry for seed {seed}: {split_manifest_path}")
        split_material_groups = {
            "train": list(map(str, selected_split["train_material_ids"])),
            "validation": list(map(str, selected_split["validation_material_ids"])),
            "test": list(map(str, selected_split["test_material_ids"])),
        }
        train_indices, val_indices, test_indices = _split_from_material_groups(
            data["material_ids"],
            split_material_groups["train"],
            split_material_groups["validation"],
            split_material_groups["test"],
        )
    train_ids = _material_ids_in_rows(data["material_ids"], train_indices)
    val_ids = _material_ids_in_rows(data["material_ids"], val_indices)
    test_ids = _material_ids_in_rows(data["material_ids"], test_indices)
    named_material_ids = set(_material_ids_in_rows(data["material_ids"], np.arange(len(data["material_ids"]))))
    for path, item in zip(training_only_paths, training_only_caches):
        lineage_aware = bool(int(item.get("lineage_aware", np.asarray(0)).reshape(-1)[0]))
        if lineage_aware:
            continue
        overlap = named_material_ids & set(_material_ids_in_rows(item["material_ids"], np.arange(len(item["material_ids"]))))
        if overlap:
            raise ValueError(f"training-only cache {path} overlaps named material IDs: {sorted(overlap)[:5]}")
    device = select_device()
    vae, basis, vae_meta = _load_model(vae_checkpoint, device)
    latent_dim = int(vae_meta["latent_dim"])
    if data["latents"].shape[1] != latent_dim:
        raise ValueError(
            f"artifact latent width {data['latents'].shape[1]} does not match VAE latent_dim {latent_dim}"
        )
    grid_size = vae.input_grid_size
    expected_values = grid_size * grid_size
    if int(np.prod(data["grids"].shape[1:])) != expected_values:
        raise ValueError(f"expected {grid_size}x{grid_size} grids")
    for parameter in vae.parameters():
        parameter.requires_grad_(False)
    vae.eval()

    embeddings = torch.from_numpy(data["embeddings"].astype(np.float32, copy=False))
    latents = torch.from_numpy(data["latents"].astype(np.float32, copy=False))
    grids = torch.from_numpy(data["grids"].astype(np.float32, copy=False).reshape(-1, expected_values))
    profile_features = torch.from_numpy(data["profile_features"].astype(np.float32, copy=False))
    control_features = torch.from_numpy(data["control_features"].astype(np.float32, copy=False))
    training_only_tensors = []
    control_validation_tensors = []
    training_only_used_count = 0
    control_validation_used_count = 0
    for cache_index, item in enumerate(training_only_caches):
        if item["latents"].shape[1] != latent_dim:
            raise ValueError("training-only cache latent width does not match the VAE")
        if item["embeddings"].shape[1] != data["embeddings"].shape[1]:
            raise ValueError("training-only cache embedding width does not match the named cache")
        if int(np.prod(item["grids"].shape[1:])) != expected_values:
            raise ValueError(f"expected {grid_size}x{grid_size} grids in training-only cache")
        aux_profiles = item.get("profile_features")
        if profile_dim:
            if aux_profiles is None:
                aux_profiles = np.zeros((len(item["captions"]), profile_dim), dtype=np.float32)
            elif aux_profiles.shape[1] != profile_dim:
                raise ValueError("training-only cache profile feature width does not match named cache")
        else:
            aux_profiles = np.zeros((len(item["captions"]), 0), dtype=np.float32)
        aux_controls = item.get("control_features")
        if control_dim:
            if aux_controls is None:
                aux_controls = np.zeros((len(item["captions"]), control_dim), dtype=np.float32)
            elif aux_controls.shape[1] != control_dim:
                raise ValueError("training-only cache control feature width does not match named cache")
        else:
            aux_controls = np.zeros((len(item["captions"]), 0), dtype=np.float32)
        row_members = [set(str(value).split("|")) - {""} for value in item["material_ids"]]
        lineage_aware = bool(int(item.get("lineage_aware", np.asarray(0)).reshape(-1)[0]))
        if lineage_aware:
            keep = np.asarray([members and members <= train_ids for members in row_members], dtype=bool)
            if not keep.any():
                raise ValueError("lineage-aware training-only cache has no rows sourced only from training materials")
            validation_keep = np.asarray([members and members <= val_ids for members in row_members], dtype=bool)
            control_validation_keep = validation_keep & (aux_controls[:, 4:7].max(axis=1) > 0)
            control_validation_used_count += int(control_validation_keep.sum())
            if control_validation_keep.any():
                control_validation_tensors.append((
                    torch.from_numpy(item["embeddings"][control_validation_keep].astype(np.float32, copy=False)),
                    torch.from_numpy(item["latents"][control_validation_keep].astype(np.float32, copy=False)),
                    torch.from_numpy(item["grids"][control_validation_keep].astype(np.float32, copy=False).reshape(-1, expected_values)),
                    torch.from_numpy(aux_profiles[control_validation_keep].astype(np.float32, copy=False)),
                    torch.from_numpy(aux_controls[control_validation_keep].astype(np.float32, copy=False)),
                ))
        else:
            overlap = named_material_ids & set(_material_ids_in_rows(item["material_ids"], np.arange(len(item["captions"]))))
            if overlap:
                raise ValueError(f"training-only cache overlaps named material IDs: {sorted(overlap)[:5]}")
            keep = np.ones(len(item["captions"]), dtype=bool)
        kept_indices = np.flatnonzero(keep)
        if training_only_sample_limit and len(kept_indices) > training_only_sample_limit:
            sample_rng = np.random.default_rng(seed + cache_index)
            kept_indices = np.sort(sample_rng.choice(
                kept_indices, size=training_only_sample_limit, replace=False
            ))
            keep = np.zeros(len(item["captions"]), dtype=bool)
            keep[kept_indices] = True
        training_only_used_count += int(keep.sum())
        training_only_tensors.append((
            torch.from_numpy(item["embeddings"][keep].astype(np.float32, copy=False)),
            torch.from_numpy(item["latents"][keep].astype(np.float32, copy=False)),
            torch.from_numpy(item["grids"][keep].astype(np.float32, copy=False).reshape(-1, expected_values)),
            torch.from_numpy(aux_profiles[keep].astype(np.float32, copy=False)),
            torch.from_numpy(aux_controls[keep].astype(np.float32, copy=False)),
        ))
    train_tensors = [embeddings[train_indices], latents[train_indices], grids[train_indices], profile_features[train_indices], control_features[train_indices]]
    if training_only_tensors:
        train_tensors = [
            torch.cat([base, *(aux[index] for aux in training_only_tensors)], dim=0)
            for index, base in enumerate(train_tensors)
        ]
    train_set = TensorDataset(
        *train_tensors
    )
    val_set = TensorDataset(
        embeddings[val_indices], latents[val_indices], grids[val_indices],
        profile_features[val_indices], control_features[val_indices],
    )
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    control_validation_loader = None
    if control_validation_tensors and control_validation_weight > 0 and control_dim >= 7:
        control_val_tensors = [
            torch.cat([cache[index] for cache in control_validation_tensors], dim=0)
            for index in range(5)
        ]
        control_validation_loader = DataLoader(
            TensorDataset(*control_val_tensors), batch_size=batch_size, shuffle=False
        )
    if control_validation_weight > 0 and control_dim >= 7 and control_validation_loader is None:
        raise ValueError("control-validation weight is enabled but no lineage-safe numeric validation rows are available")
    test_loader = None
    if not skip_test_eval:
        test_set = TensorDataset(
            embeddings[test_indices], latents[test_indices], grids[test_indices],
            profile_features[test_indices], control_features[test_indices],
        )
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    predictor = TextToLatent(
        embeddings.shape[1], latent_dim, hidden_dims=hidden_dims,
        dropout=dropout, architecture_version=2, profile_dim=profile_dim,
        profile_projection_dim=64 if profile_dim else 0,
        control_dim=control_dim, control_projection_dim=32 if control_dim else 0,
    ).to(device)
    if initialize_from is not None:
        if not initialize_from.is_file():
            raise FileNotFoundError(f"initial text checkpoint not found: {initialize_from}")
        initial_state = torch.load(initialize_from, map_location="cpu", weights_only=True)
        if int(initial_state.get("embedding_dim", -1)) != embeddings.shape[1]:
            raise ValueError("initial text checkpoint embedding width does not match the cache")
        if int(initial_state.get("latent_dim", -1)) != latent_dim:
            raise ValueError("initial text checkpoint latent width does not match the VAE")
        source_profile_dim = int(initial_state.get("profile_dim", 0))
        source_projection_dim = int(initial_state.get("profile_projection_dim", 0))
        source_control_dim = int(initial_state.get("control_dim", 0))
        source_control_projection_dim = int(initial_state.get("control_projection_dim", 0))
        source_weights = initial_state["model_state_dict"]
        if (
            source_profile_dim == profile_dim
            and source_projection_dim == predictor.profile_projection_dim
            and source_control_dim == control_dim
            and source_control_projection_dim == predictor.control_projection_dim
        ):
            try:
                predictor.load_state_dict(source_weights, strict=True)
            except RuntimeError as exc:
                raise ValueError(
                    "initial text checkpoint architecture does not match the requested model; "
                    "use the same --hidden-dims and --dropout settings"
                ) from exc
        elif source_control_dim == 0 and (
            (source_profile_dim == 0 and source_projection_dim == 0)
            or (
                source_profile_dim == profile_dim
                and source_projection_dim == predictor.profile_projection_dim
            )
        ):
            # Transfer compatible existing input columns and initialize newly
            # added control projections independently.
            target_weights = predictor.state_dict()
            for name, source_tensor in source_weights.items():
                if name not in target_weights:
                    raise ValueError(f"initial text checkpoint has unexpected parameter: {name}")
                target_tensor = target_weights[name]
                if source_tensor.shape == target_tensor.shape:
                    target_tensor.copy_(source_tensor)
                elif (
                    name == "network.0.weight"
                    and source_tensor.ndim == 2
                    and target_tensor.ndim == 2
                    and source_tensor.shape[0] == target_tensor.shape[0]
                    and source_tensor.shape[1] <= target_tensor.shape[1]
                ):
                    target_tensor[:, :source_tensor.shape[1]].copy_(source_tensor)
                else:
                    raise ValueError(
                        f"initial text checkpoint parameter shape does not match: {name} "
                        f"{tuple(source_tensor.shape)} vs {tuple(target_tensor.shape)}"
                    )
            predictor.load_state_dict(target_weights, strict=True)
        else:
            raise ValueError("initial text checkpoint profile architecture does not match the cache")
    if freeze_text_path:
        if profile_dim <= 0 or predictor.profile_projection_dim <= 0:
            raise ValueError("--freeze-text-path requires profile features and a profile projection")
        for parameter in predictor.network.parameters():
            parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(predictor.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=scheduler_t0, T_mult=scheduler_t_mult, eta_min=minimum_learning_rate
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    best_epoch = 0
    stale = 0
    history: list[dict[str, Any]] = []
    if train_ids & val_ids or train_ids & test_ids or val_ids & test_ids:
        raise RuntimeError("material-level split leaked material IDs across train/validation/test")
    for epoch in range(1, epochs + 1):
        train_metrics = _run_epoch(
            predictor, vae, basis, train_loader, device, optimizer,
            latent_weight, cosine_weight, reconstruction_weight, magnitude_weight,
            peak_weight, tv_weight, control_weight, grid_size,
        )
        val_metrics = _run_epoch(
            predictor, vae, basis, val_loader, device, None,
            latent_weight, cosine_weight, reconstruction_weight, magnitude_weight,
            peak_weight, tv_weight, control_weight, grid_size,
        )
        control_val_metrics = None
        if control_validation_loader is not None:
            control_val_metrics = _run_epoch(
                predictor, vae, basis, control_validation_loader, device, None,
                latent_weight, cosine_weight, reconstruction_weight, magnitude_weight,
                peak_weight, tv_weight, control_weight, grid_size,
            )
        scheduler.step(epoch)
        row = {
            "epoch": epoch,
            "train": train_metrics,
            "validation": val_metrics,
            "control_validation": control_val_metrics,
        }
        history.append(row)
        print(
            f"Epoch {epoch:03d}/{epochs}: train latent MSE={train_metrics['latent_mse']:.6f}, "
            f"heldout latent MSE={val_metrics['latent_mse']:.6f}, "
            f"heldout loss={val_metrics['loss']:.6f}"
        )
        selection_value = val_metrics[selection_metric]
        if control_val_metrics is not None:
            selection_value += control_validation_weight * control_val_metrics["control_loss"]
        if selection_value < best_val:
            best_val = selection_value
            best_epoch = epoch
            stale = 0
            state: dict[str, Any] = {
                "model_state_dict": predictor.state_dict(),
                "embedding_dim": predictor.embedding_dim,
                "latent_dim": predictor.latent_dim,
                "profile_dim": predictor.profile_dim,
                "profile_projection_dim": predictor.profile_projection_dim,
                "control_dim": predictor.control_dim,
                "control_projection_dim": predictor.control_projection_dim,
                "hidden_dims": list(predictor.hidden_dims),
                "dropout": predictor.dropout_probability,
                "architecture": {
                    "version": predictor.architecture_version,
                    "hidden_dims": list(predictor.hidden_dims),
                    "dropout": predictor.dropout_probability,
                    "residual": True,
                    "normalization": "LayerNorm",
                },
                "activation": "SiLU",
                "data_cache": str(data_cache),
                "additional_data_caches": [str(path) for path in (additional_data_caches or [])],
                "training_only_data_caches": [str(path) for path in training_only_paths],
                "split_manifest_path": str(split_manifest_path) if split_manifest_path else None,
                "split_material_groups": split_material_groups,
                "embedding_model": str(data["embedding_model"].reshape(-1)[0]),
                "vae_checkpoint": str(vae_checkpoint),
                "vae_basis_type": vae_meta.get("basis_type", "unknown"),
                "epoch": epoch,
                "validation_loss": val_metrics["loss"],
                "selection_metric": selection_metric,
                "selection_value": best_val,
                "control_validation_weight": control_validation_weight,
                "control_validation_loss": (
                    control_val_metrics["control_loss"] if control_val_metrics is not None else None
                ),
                "initialized_from": str(initialize_from) if initialize_from else None,
                "validation_latent_mse": val_metrics["latent_mse"],
                "validation_cosine_loss": val_metrics["cosine_loss"],
                "validation_reconstruction_loss": val_metrics["reconstruction_loss"],
                "train_material_count": len(train_ids),
                "validation_material_count": len(val_ids),
                "test_material_count": len(test_ids),
                "train_sample_count": len(train_indices),
                "training_only_sample_count": training_only_used_count,
                "control_validation_sample_count": control_validation_used_count,
                "validation_sample_count": len(val_indices),
                "test_sample_count": len(test_indices),
                "validation_fraction": validation_fraction,
                "test_fraction": test_fraction,
                "split_strategy": split_strategy,
                "split_version": 1,
                "seed": seed,
                "model_seed": initialization_seed,
                "loss_weights": {
                    "latent": latent_weight,
                    "cosine": cosine_weight,
                    "reconstruction": reconstruction_weight,
                    "magnitude": magnitude_weight,
                    "total_variation": tv_weight,
                "decoded_controls": control_weight,
                },
                "peak_weight": peak_weight,
                "tv_weight": tv_weight,
                "control_weight": control_weight,
                "optimizer": {"name": "AdamW", "learning_rate": learning_rate, "weight_decay": weight_decay},
                "scheduler": {
                    "name": "CosineAnnealingWarmRestarts",
                    "T_0": scheduler_t0,
                    "T_mult": scheduler_t_mult,
                    "eta_min": minimum_learning_rate,
                },
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
    test_metrics = None
    if test_loader is not None:
        test_metrics = _run_epoch(
            predictor, vae, basis, test_loader, device, None,
            latent_weight, cosine_weight, reconstruction_weight, magnitude_weight,
            peak_weight, tv_weight, control_weight, grid_size,
        )
        best_state["test_latent_mse"] = test_metrics["latent_mse"]
        best_state["test_cosine_loss"] = test_metrics["cosine_loss"]
        best_state["test_reconstruction_loss"] = test_metrics["reconstruction_loss"]
        best_state["test_tv_loss"] = test_metrics["tv_loss"]
        best_state["test_control_loss"] = test_metrics["control_loss"]
    best_state["test_evaluated"] = test_metrics is not None
    torch.save(best_state, checkpoint_path)
    result = {
        "checkpoint": str(checkpoint_path),
        "data_caches": [str(path) for path in cache_paths],
        "training_only_data_caches": [str(path) for path in training_only_paths],
        "embedding_model": embedding_models[0],
        "seed": int(seed),
        "split_strategy": split_strategy,
        "model_seed": int(initialization_seed),
        "best_epoch": best_epoch,
        "validation_latent_mse": float(best_state["validation_latent_mse"]),
        "test_latent_mse": test_metrics["latent_mse"] if test_metrics else None,
        "test_reconstruction_loss": test_metrics["reconstruction_loss"] if test_metrics else None,
        "test_cosine_loss": test_metrics["cosine_loss"] if test_metrics else None,
        "test_evaluated": test_metrics is not None,
        "validation_control_loss": best_state.get("control_validation_loss"),
        "control_validation_sample_count": int(best_state.get("control_validation_sample_count", 0)),
        "training_only_sample_count": int(best_state.get("training_only_sample_count", 0)),
        "split_manifest_path": str(split_manifest_path) if split_manifest_path else None,
        "train_material_count": len(train_ids),
        "validation_material_count": len(val_ids),
        "test_material_count": len(test_ids),
        "history": history,
    }
    metrics_path = checkpoint_path.with_suffix(".json")
    metrics_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if test_metrics is None:
        print(f"Saved best text predictor to {checkpoint_path}; reserved test split was not evaluated.")
    else:
        print(
            f"Saved best text predictor to {checkpoint_path}; "
            f"test latent MSE={result['test_latent_mse']:.6f}"
        )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-cache", type=Path, default=DEFAULT_DATA_CACHE, help="NPZ with text embeddings, VAE latents, NDF grids, and material metadata")
    parser.add_argument(
        "--additional-data-cache", dest="additional_data_caches",
        type=Path, nargs="+", default=None,
        help="additional compatible caches to concatenate with --data-cache",
    )
    parser.add_argument(
        "--training-only-data-cache", dest="training_only_data_caches",
        type=Path, nargs="+", default=None,
        help="compatible caches appended to training rows after the named material split",
    )
    parser.add_argument(
        "--training-only-sample-limit", type=int, default=5_000,
        help="maximum rows sampled per training-only cache (0 uses all rows; default 5000)",
    )
    parser.add_argument("--vae-checkpoint", type=Path, default=DEFAULT_VAE_CHECKPOINT)
    parser.add_argument("--checkpoint", dest="checkpoint_path", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--latent-weight", type=float, default=1.0)
    parser.add_argument("--cosine-weight", type=float, default=0.1)
    parser.add_argument("--reconstruction-weight", type=float, default=1.0)
    parser.add_argument(
        "--magnitude-weight", type=float, default=0.0,
        help="auxiliary smooth-L1 loss on log latent-vector magnitude (default: 0)",
    )
    parser.add_argument("--peak-weight", type=float, default=15.0)
    parser.add_argument(
        "--tv-weight", type=float, default=0.01,
        help="decoded-grid total-variation regularization weight (set 0 to disable)",
    )
    parser.add_argument(
        "--control-weight", type=float, default=1.0,
        help="decoded-grid loss matching prompted roughness, anisotropy, and orientation",
    )
    parser.add_argument(
        "--control-validation-weight", type=float, default=0.0,
        help="weight of lineage-safe numeric control validation in checkpoint selection",
    )
    parser.add_argument(
        "--split-manifest", dest="split_manifest_path", type=Path,
        help="JSON manifest with explicit complete train/validation/test material IDs",
    )
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument(
        "--split-strategy", choices=("random", "norm_stratified"), default="random",
        help="grouped split policy; norm_stratified balances target latent magnitudes",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--model-seed", type=int, default=None,
        help="initialization/data-order seed independent of the material split seed",
    )
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--scheduler-t0", type=int, default=20)
    parser.add_argument("--scheduler-t-mult", type=int, default=2)
    parser.add_argument("--minimum-learning-rate", type=float, default=1e-5)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument(
        "--hidden-dims", type=int, nargs=2, metavar=("WIDTH", "BOTTLENECK"),
        default=(512, 256), help="predictor hidden widths (default: 512 256)",
    )
    parser.add_argument(
        "--initialize-from", type=Path,
        help="optional compatible text predictor checkpoint for fine-tuning",
    )
    parser.add_argument(
        "--freeze-text-path", action="store_true",
        help="freeze the text predictor and train only the profile projection",
    )
    parser.add_argument(
        "--selection-metric",
        choices=("latent_mse", "loss", "reconstruction_loss"),
        default="latent_mse",
        help="validation metric for checkpoint selection and early stopping",
    )
    parser.add_argument(
        "--skip-test-eval", action="store_true",
        help="reserve the grouped test split without evaluating it during candidate selection",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    train_text(**vars(args))


if __name__ == "__main__":
    main()
