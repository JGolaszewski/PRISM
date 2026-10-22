"""Build a caption, text-embedding, and frozen-VAE latent training cache.

Run with ``python -m prism.text_dataset``. The resulting per-source NPZ files
contain ``captions``, ``embeddings`` [N,D], ``latents`` [N,L],
``grids`` [N,4096], ``material_ids``, ``material_names``, ``source_paths``,
and ``embedding_model``.

Text embeddings use the Hugging Face Transformers implementation of
``sentence-transformers/all-MiniLM-L6-v2`` (masked mean pooling followed by
L2 normalization).  Install ``transformers`` and ``huggingface_hub`` before
running; the model is downloaded/cached by Hugging Face on first use.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from prism.model import NDFPolynomialVAE


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = REPO_ROOT / "checkpoints" / "best_model.pt"
DEFAULT_GRID_DIR = REPO_ROOT / "dataset" / "grid"
DEFAULT_AUGMENTED_DIR = REPO_ROOT / "dataset" / "augmented"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "checkpoints"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GRID_SIZE = 64



def _material_traits(name: str) -> list[str]:
    """Expand recognizable filename tokens without treating them as labels."""
    words = re.sub(r"[^a-z0-9]+", " ", name.lower()).split()
    traits: list[str] = []
    categories = {
        "metal": {"metal", "metallic", "steel", "aluminium", "aluminum", "bronze", "brass", "copper", "chrome", "iron", "silver", "gold"},
        "plastic": {"plastic", "acrylic", "phenolic", "polymer", "rubber", "vinyl"},
        "fabric": {"fabric", "cloth", "velvet", "silk", "wool", "cotton", "denim"},
        "wood": {"wood", "maple", "oak", "walnut", "cherry", "pine", "mahogany"},
    }
    joined = set(words)
    for category, vocabulary in categories.items():
        if joined & vocabulary:
            traits.append(category)
    if joined & {"rough", "roughness", "matte", "diffuse", "rubber", "fabric", "cloth"}:
        traits.append("rough")
    if joined & {"smooth", "polished", "chrome", "mirror", "glossy", "shiny"}:
        traits.append("smooth")
    if joined & {"anisotropic", "anisotropy", "brushed", "hair", "fiber", "fibers"}:
        traits.append("anisotropic")
    if joined & {"paint", "painted"}:
        traits.append("paint")
    if joined & {"oxidized", "patina", "rust"}:
        traits.append("oxidized")
    return list(dict.fromkeys(traits))


def _grid_descriptors(grid: np.ndarray) -> tuple[str, str, str]:
    """Return cautious peak-width, concentration, and anisotropy descriptors."""
    values = np.asarray(grid, dtype=np.float64)
    peak = float(values.max(initial=0.0))
    if peak <= 0.0:
        return "near-zero intensity", "approximately isotropic", "low peak concentration"
    coords = np.linspace(-1.0, 1.0, GRID_SIZE, dtype=np.float64)
    x, y = np.meshgrid(coords, coords, indexing="xy")
    disk = x * x + y * y <= 1.0
    weights = np.where(disk, np.maximum(values, 0.0), 0.0)
    total = float(weights.sum())
    if total <= 1e-12:
        return "near-zero intensity", "approximately isotropic", "low peak concentration"
    radius = np.sqrt(x * x + y * y)
    mean_radius = float((weights * radius).sum() / total)
    if mean_radius < 0.22:
        width = "narrow low-roughness highlight"
    elif mean_radius < 0.42:
        width = "medium-width roughness profile"
    else:
        width = "broad high-roughness highlight"
    xx = float((weights * x * x).sum() / total)
    yy = float((weights * y * y).sum() / total)
    xy = float((weights * x * y).sum() / total)
    anisotropy = (np.hypot(xx - yy, 2.0 * xy) / max(xx + yy, 1e-12))
    if anisotropy >= 0.22:
        angle = 0.5 * np.arctan2(2.0 * xy, xx - yy)
        angle_deg = abs(float(np.degrees(angle))) % 180.0
        if angle_deg < 22.5 or angle_deg >= 157.5:
            orientation = "horizontal"
        elif 67.5 <= angle_deg < 112.5:
            orientation = "vertical"
        else:
            orientation = "diagonal"
        direction = f"directionally anisotropic with {orientation} orientation"
    else:
        direction = "approximately isotropic"
    central = float(weights[radius <= 0.15].sum() / total)
    concentration = "high central-peak concentration" if central >= 0.55 else (
        "moderate central-peak concentration" if central >= 0.25 else "low central-peak concentration"
    )
    return width, direction, concentration


def caption_for_material(name: str, grid: np.ndarray, source: str) -> str:
    """Create a repeatable natural-language description from name and grid."""
    width, direction, concentration = _grid_descriptors(grid)
    if source == "augmented":
        # augment.py blends random base grids but discards their identities and
        # blend parameters; only describe observable grid traits here.
        return (
            f"synthetic augmented material blend with a {width}, {concentration}, "
            f"and {direction} NDF distribution"
        )
    label = re.sub(r"^(?:merl|rgl)_", "", name.lower())
    label = re.sub(r"[_-]+", " ", label).strip()
    traits = _material_traits(label)
    subject = f"{label} material" if label else "material"
    phrases = []
    if "metal" in traits:
        phrases.append("metallic")
    elif "plastic" in traits:
        phrases.append("plastic")
    elif "fabric" in traits:
        phrases.append("fabric")
    elif "wood" in traits:
        phrases.append("wood")
    if "paint" in traits:
        phrases.append("paint finish")
    if "oxidized" in traits:
        phrases.append("oxidized surface")
    if "rough" in traits:
        phrases.append("rough")
    elif "smooth" in traits:
        phrases.append("smooth glossy")
    if "anisotropic" in traits:
        phrases.append("brushed anisotropic")
    descriptor = " ".join(dict.fromkeys(phrases))
    if descriptor:
        subject = f"{label} {descriptor} material" if label else f"{descriptor} material"
    return f"{subject} with a {width}, {concentration}, and {direction} NDF distribution"


def _material_id(name: str, source: str) -> str:
    if source == "augmented":
        # Augmentation currently discards its randomly blended input indices.
        # Keep each synthetic sample independent rather than invent lineage.
        return "aug:" + name.lower()
    normalized = re.sub(r"^(?:merl|rgl)_", "", name.lower())
    return "material:" + re.sub(r"[^a-z0-9]+", "-", normalized).strip("-")


def _load_grid(path: Path) -> np.ndarray:
    grid = np.load(path, allow_pickle=False)
    if grid.shape != (GRID_SIZE, GRID_SIZE):
        raise ValueError(f"{path}: expected {GRID_SIZE}x{GRID_SIZE}, got {grid.shape}")
    if not np.isfinite(grid).all() or np.any(grid < 0):
        raise ValueError(f"{path}: NDF grid must contain finite nonnegative values")
    return grid.astype(np.float32, copy=False)


def collect_samples(
    source: str = "both",
    grid_dir: Path = DEFAULT_GRID_DIR,
    augmented_dir: Path = DEFAULT_AUGMENTED_DIR,
) -> tuple[list[Path], list[str]]:
    """Collect sorted samples and their source labels."""
    paths: list[Path] = []
    sources: list[str] = []
    if source in ("grid", "both"):
        found = sorted(grid_dir.glob("*.npy"))
        if not found:
            raise FileNotFoundError(f"No .npy grids found in {grid_dir}")
        paths.extend(found)
        sources.extend(["grid"] * len(found))
    if source in ("augmented", "both"):
        found = sorted(augmented_dir.glob("*.npy"))
        if not found:
            raise FileNotFoundError(f"No .npy augmented grids found in {augmented_dir}")
        paths.extend(found)
        sources.extend(["augmented"] * len(found))
    return paths, sources


def load_frozen_vae(checkpoint_path: Path, device: torch.device) -> NDFPolynomialVAE:
    """Instantiate and freeze the VAE described by the training checkpoint."""
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"VAE checkpoint not found: {checkpoint_path}")
    try:
        checkpoint: dict[str, Any] = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
    except TypeError:  # compatibility with older PyTorch versions
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, dict):
        raise ValueError(f"{checkpoint_path}: missing model_state_dict")
    latent_dim = int(checkpoint.get("latent_dim", state["fc_mu.weight"].shape[0]))
    grid_size = int(checkpoint.get("grid_size", GRID_SIZE))
    num_terms = int(checkpoint.get("num_poly_terms", state["decoder.4.weight"].shape[0]))
    first_hidden = int(state["encoder.0.weight"].shape[0])
    second_hidden = int(state["encoder.2.weight"].shape[0])
    model = NDFPolynomialVAE(
        input_grid_size=grid_size,
        num_poly_terms=num_terms,
        hidden_dims=(first_hidden, second_hidden),
        latent_dim=latent_dim,
        positive_coefficients=bool(checkpoint.get("positive_coefficients", False)),
    )
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def encode_texts(
    captions: list[str], model_name: str, batch_size: int, device: torch.device
) -> torch.Tensor:
    """Compute normalized masked-mean text embeddings using Transformers."""
    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Text embedding requires Transformers. Install it with "
            "`python -m pip install transformers` (and ensure Hugging Face model access), "
            "then rerun this command."
        ) from exc
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        encoder = AutoModel.from_pretrained(model_name).to(device).eval()
    except Exception as exc:
        raise RuntimeError(
            f"Could not load embedding model {model_name!r}. Check network/cache access "
            "and the model identifier."
        ) from exc
    all_embeddings: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, len(captions), batch_size):
            tokens = tokenizer(
                captions[start : start + batch_size],
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            ).to(device)
            hidden = encoder(**tokens).last_hidden_state
            mask = tokens["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
            all_embeddings.append(nn.functional.normalize(pooled.float(), p=2, dim=1).cpu())
    return torch.cat(all_embeddings, dim=0)


def build_text_dataset(
    source: str,
    output_path: Path,
    checkpoint_path: Path = DEFAULT_CHECKPOINT,
    grid_dir: Path = DEFAULT_GRID_DIR,
    augmented_dir: Path = DEFAULT_AUGMENTED_DIR,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    batch_size: int = 128,
    device: str | None = None,
) -> dict[str, Any]:
    """Create and save a stable, per-source NPZ artifact for text training."""
    if source not in ("grid", "augmented"):
        raise ValueError("build_text_dataset source must be 'grid' or 'augmented'")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    target_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    paths, sources = collect_samples(source, grid_dir, augmented_dir)
    grids_np = np.stack([_load_grid(path) for path in paths])
    material_names = [path.stem for path in paths]
    captions = [
        caption_for_material(name, grid, sample_source)
        for name, grid, sample_source in zip(material_names, grids_np, sources)
    ]

    vae = load_frozen_vae(checkpoint_path, target_device)
    if vae.input_grid_size != GRID_SIZE:
        raise ValueError(f"VAE expects {vae.input_grid_size} grid size, source grids are {GRID_SIZE}")
    latent_batches: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, len(grids_np), batch_size):
            batch = torch.from_numpy(grids_np[start : start + batch_size]).to(target_device)
            mu, _ = vae.encode(batch)
            latent_batches.append(mu.float().cpu())
    latents = torch.cat(latent_batches, dim=0)
    embeddings = encode_texts(captions, embedding_model, batch_size, target_device)
    artifact: dict[str, Any] = {
        "captions": np.asarray(captions, dtype=np.str_),
        "embeddings": embeddings.numpy().astype(np.float32, copy=False),
        "latents": latents.numpy().astype(np.float32, copy=False),
        "grids": grids_np.reshape(len(paths), -1),
        "material_ids": np.asarray(
            [_material_id(name, sample_source) for name, sample_source in zip(material_names, sources)],
            dtype=np.str_,
        ),
        "material_names": np.asarray(material_names, dtype=np.str_),
        "source_paths": np.asarray([str(path.resolve()) for path in paths], dtype=np.str_),
        "embedding_model": np.asarray(embedding_model, dtype=np.str_),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() != ".npz":
        raise ValueError(f"output path must end in .npz, got {output_path}")
    temporary_path = output_path.with_name(output_path.stem + ".tmp.npz")
    np.savez(temporary_path, **artifact)
    temporary_path.replace(output_path)
    return artifact


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("grid", "augmented", "both"), default="both")
    parser.add_argument("--grid-dir", type=Path, default=DEFAULT_GRID_DIR)
    parser.add_argument("--augmented-dir", type=Path, default=DEFAULT_AUGMENTED_DIR)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default=None, help="Optional torch device, e.g. cpu or cuda")
    args = parser.parse_args()
    try:
        selected_sources = ("grid", "augmented") if args.source == "both" else (args.source,)
        artifacts = [
            build_text_dataset(
                output_path=args.output_dir / f"text_dataset_{selected}.npz",
                checkpoint_path=args.checkpoint,
                source=selected,
                grid_dir=args.grid_dir,
                augmented_dir=args.augmented_dir,
                embedding_model=args.embedding_model,
                batch_size=args.batch_size,
                device=args.device,
            )
            for selected in selected_sources
        ]
    except Exception as exc:
        parser.exit(1, f"Text dataset creation failed: {exc}\n")
    print(
        "Saved " + "; ".join(
            f"{len(artifact['captions'])} examples, embeddings={artifact['embeddings'].shape}, "
            f"latents={artifact['latents'].shape}"
            for artifact in artifacts
        ) + f" under {args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
