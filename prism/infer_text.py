"""Generate and visualize an NDF from a natural-language material prompt."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from prism.model import select_device
from prism.text_model import SimilarityLatentRetriever, TextToLatent
from prism.text_dataset import (
    DEFAULT_EMBEDDING_MODEL,
    encode_texts,
    parse_profile_features,
    parse_prompt_controls,
)
from prism.train_text import (
    DEFAULT_CHECKPOINT,
    DEFAULT_VAE_CHECKPOINT,
)
from prism.validate import _load_model


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "results" / "text_demo.png"
DEFAULT_COMPARISON_OUTPUT = REPO_ROOT / "results" / "text_prompt_comparison.png"
_DIRECTIONAL_READY_CHECKPOINT = (
    REPO_ROOT / "checkpoints" / "experiments" / "text_to_latent_functionality"
    / "directional_inference_correction" / "directional_retrieval.pt"
)
DEFAULT_INFERENCE_CHECKPOINT = (
    _DIRECTIONAL_READY_CHECKPOINT if _DIRECTIONAL_READY_CHECKPOINT.is_file() else DEFAULT_CHECKPOINT
)
CONTRAST_PROMPTS = (
    "shiny mirror-like gold",
    "matte rough velvet",
    "anisotropic brushed aluminum",
)


def _choose_device(name: str) -> torch.device:
    if name == "auto":
        return select_device()
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but this PyTorch installation has no available CUDA device")
    if device.type not in ("cpu", "cuda", "mps"):
        raise ValueError("--device must be auto, cpu, cuda, or mps")
    if device.type == "mps" and not getattr(torch.backends, "mps", None):
        raise RuntimeError("MPS was requested, but this PyTorch installation does not support MPS")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested, but it is not available on this system")
    return device


def _load_predictor(
    path: Path, device: torch.device, expected_embedding_model: str | None = None
) -> tuple[TextToLatent, dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Text predictor checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(f"Text checkpoint has no model_state_dict: {path}")
    embedding_model = str(checkpoint.get("embedding_model", expected_embedding_model))
    if expected_embedding_model is not None and embedding_model != expected_embedding_model:
        raise ValueError(
            "Text checkpoint expects embedding model "
            f"{embedding_model!r}, but the VAE/text pipeline expects {expected_embedding_model!r}"
        )
    embedding_dim = int(checkpoint.get("embedding_dim", 384))
    latent_dim = int(checkpoint.get("latent_dim", 0))
    if embedding_dim < 1 or latent_dim < 1:
        raise ValueError("Text checkpoint must specify positive embedding_dim and latent_dim")
    if checkpoint.get("predictor_type") == "e5_similarity_retrieval":
        predictor = SimilarityLatentRetriever(
            torch.as_tensor(checkpoint["prototype_embeddings"]),
            torch.as_tensor(checkpoint["prototype_latents"]),
            int(checkpoint["neighbors"]),
            float(checkpoint["temperature"]),
            torch.as_tensor(checkpoint["prototype_anisotropy"])
            if "prototype_anisotropy" in checkpoint else None,
            float(checkpoint.get("directional_threshold", 0.30)),
        ).to(device)
        predictor.load_state_dict(checkpoint["model_state_dict"])
        predictor.eval()
        return predictor, checkpoint
    architecture = checkpoint.get("architecture", {})
    if not isinstance(architecture, dict):
        architecture = {}
    predictor = TextToLatent(
        embedding_dim,
        latent_dim,
        hidden_dims=tuple(checkpoint.get("hidden_dims", architecture.get("hidden_dims", (512, 256)))),
        dropout=float(checkpoint.get("dropout", architecture.get("dropout", 0.0))),
        architecture_version=int(architecture.get("version", checkpoint.get("architecture_version", 1))),
        profile_dim=int(checkpoint.get("profile_dim", 0)),
        profile_projection_dim=int(checkpoint.get("profile_projection_dim", 0)),
        control_dim=int(checkpoint.get("control_dim", 0)),
        control_projection_dim=int(checkpoint.get("control_projection_dim", 0)),
    ).to(device)
    predictor.load_state_dict(checkpoint["model_state_dict"])
    predictor.eval()
    return predictor, checkpoint


def _encode_prompt(prompt: str, model_name: str, device: torch.device) -> torch.Tensor:
    """Use the same text preprocessing as the dataset builder."""
    return encode_texts([prompt], model_name, batch_size=1, device=device).to(device)


def _prompt_requests_direction(prompt: str) -> bool:
    lowered = prompt.casefold()
    return any(term in lowered for term in ("anisotropic", "brushed", "elongated", "aligned"))


def _recenter_ndf_batch(ndf: torch.Tensor) -> torch.Tensor:
    """Translate each positive NDF so its in-disk intensity centroid is centered."""
    if ndf.ndim == 2:
        ndf = ndf.unsqueeze(0)
    if ndf.ndim == 4 and ndf.shape[1] == 1:
        ndf = ndf[:, 0]
    if ndf.ndim != 3 or ndf.shape[-1] != ndf.shape[-2]:
        raise ValueError(f"expected [batch, size, size] NDF grids, got {tuple(ndf.shape)}")
    batch, size, _ = ndf.shape
    dtype, device = ndf.dtype, ndf.device
    coords = torch.linspace(-1.0, 1.0, size, dtype=dtype, device=device)
    y, x = torch.meshgrid(coords, coords, indexing="ij")
    disk = x.square() + y.square() <= 1.0
    weights = ndf.clamp_min(0.0) * disk
    mass = weights.sum(dim=(-2, -1)).clamp_min(1e-12)
    center_x = (weights * x).sum(dim=(-2, -1)) / mass
    center_y = (weights * y).sum(dim=(-2, -1)) / mass
    sample_x = x.unsqueeze(0) + center_x[:, None, None]
    sample_y = y.unsqueeze(0) + center_y[:, None, None]
    sample_grid = torch.stack((sample_x, sample_y), dim=-1)
    shifted = torch.nn.functional.grid_sample(
        weights.unsqueeze(1), sample_grid, mode="bilinear", padding_mode="zeros", align_corners=True
    )[:, 0]
    return shifted * disk


def _render_ndf(ndf: np.ndarray, prompt: str, output: Path) -> None:
    grid_size = ndf.shape[0]
    coords = np.linspace(-1.0, 1.0, grid_size, dtype=np.float32)
    hx, hy = np.meshgrid(coords, coords, indexing="xy")
    radius_sq = hx**2 + hy**2
    valid = radius_sq <= 1.0
    safe_ndf = np.where(valid, np.maximum(ndf, 0.0), np.nan)
    values = np.nan_to_num(safe_ndf, nan=0.0)
    peak = float(values.max())
    norm_values = values / peak if peak > 0 else values

    fig = plt.figure(figsize=(13, 6), constrained_layout=True)
    fig.suptitle(f"Prompt: {prompt}", fontsize=12)
    outer = fig.add_gridspec(1, 2, width_ratios=(1.05, 1.35))

    ax_heat = fig.add_subplot(outer[0, 0])
    image = ax_heat.imshow(
        safe_ndf,
        origin="lower",
        extent=(-1.0, 1.0, -1.0, 1.0),
        cmap="magma",
        interpolation="nearest",
    )
    ax_heat.set_title("Predicted NDF")
    ax_heat.set_xlabel("Half-vector x")
    ax_heat.set_ylabel("Half-vector y")
    fig.colorbar(image, ax=ax_heat, shrink=0.82, label="NDF")

    right = outer[0, 1].subgridspec(2, 1, height_ratios=(0.7, 1.3))
    ax_profile = fig.add_subplot(right[0, 0])
    center_row = grid_size // 2
    ax_profile.plot(coords, ndf[center_row], color="#167d9a", linewidth=1.7)
    ax_profile.set_title("Center-row profile")
    ax_profile.set_xlabel("Half-vector x")
    ax_profile.set_ylabel("NDF")
    ax_profile.grid(alpha=0.25)

    ax_surface = fig.add_subplot(right[1, 0], projection="3d")
    # Map each in-disk NDF sample to its direction on the upper hemisphere.
    # Modulating the hemisphere radius by normalized NDF exposes the lobe shape.
    hemisphere_z = np.sqrt(np.clip(1.0 - radius_sq, 0.0, 1.0))
    radial = np.where(valid, norm_values, np.nan)
    x_surface = radial * hx
    y_surface = radial * hy
    z_surface = radial * hemisphere_z
    surface = ax_surface.plot_surface(
        x_surface,
        y_surface,
        z_surface,
        facecolors=plt.get_cmap("magma")(np.clip(norm_values, 0.0, 1.0)),
        linewidth=0,
        antialiased=True,
        shade=False,
    )
    del surface
    ax_surface.set_title("NDF lobe on upper hemisphere")
    ax_surface.set_xlabel("x")
    ax_surface.set_ylabel("y")
    ax_surface.set_zlabel("z")
    ax_surface.set_box_aspect((1, 1, 0.8))
    ax_surface.view_init(elev=28, azim=-55)
    ax_surface.set_xlim(-1, 1)
    ax_surface.set_ylim(-1, 1)
    ax_surface.set_zlim(0, 1)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def _render_comparison(ndfs: list[np.ndarray], prompts: Sequence[str], output: Path) -> Path:
    """Render comparable NDF maps and center-row profiles for prompt contrasts."""
    if len(ndfs) != len(prompts) or not ndfs:
        raise ValueError("ndfs and prompts must have the same nonzero length")
    size = ndfs[0].shape[0]
    if any(ndf.shape != (size, size) for ndf in ndfs):
        raise ValueError("all comparison NDFs must have the same square shape")
    coords = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    fig, axes = plt.subplots(2, len(prompts), figsize=(5 * len(prompts), 8), constrained_layout=True)
    if len(prompts) == 1:
        axes = np.asarray(axes).reshape(2, 1)
    for index, (ndf, prompt) in enumerate(zip(ndfs, prompts)):
        safe = np.where(np.isfinite(ndf), np.maximum(ndf, 0), 0)
        peak = float(safe.max())
        normalized = safe / peak if peak > 0 else safe
        axes[0, index].imshow(
            normalized, origin="lower", extent=(-1, 1, -1, 1), cmap="magma", vmin=0, vmax=1
        )
        axes[0, index].set_title(prompt)
        axes[0, index].set_xlabel("Half-vector x")
        axes[0, index].set_ylabel("Half-vector y")
        axes[1, index].plot(coords, safe[size // 2], color="#167d9a", linewidth=1.7)
        axes[1, index].set_title("Center-row NDF profile")
        axes[1, index].set_xlabel("Half-vector x")
        axes[1, index].set_ylabel("NDF")
        axes[1, index].grid(alpha=0.25)
    fig.suptitle("Text prompt comparison (each heatmap normalized to its own peak)")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def _ndf_shape_metrics(ndf: np.ndarray) -> dict[str, float]:
    """Summarize concentration, width, and directional shape of an NDF lobe."""
    size = ndf.shape[0]
    coords = np.linspace(-1.0, 1.0, size, dtype=np.float64)
    x, y = np.meshgrid(coords, coords, indexing="xy")
    radius = np.sqrt(x * x + y * y)
    values = np.where(radius <= 1.0, np.maximum(np.nan_to_num(ndf, nan=0.0), 0.0), 0.0)
    total = float(values.sum())
    if total <= 1e-12:
        return {"peak": float(values.max()), "central_mass_fraction": 0.0, "mean_radius": 0.0, "anisotropy": 0.0}
    xx = float((values * x * x).sum() / total)
    yy = float((values * y * y).sum() / total)
    xy = float((values * x * y).sum() / total)
    anisotropy = float(np.hypot(xx - yy, 2.0 * xy) / max(xx + yy, 1e-12))
    return {
        "peak": float(values.max()),
        "central_mass_fraction": float(values[radius <= 0.15].sum() / total),
        "mean_radius": float((values * radius).sum() / total),
        "anisotropy": anisotropy,
    }


def generate(prompt: str, output: Path, vae_checkpoint: Path, text_checkpoint: Path, device: torch.device) -> Path:
    if not prompt.strip():
        raise ValueError("Prompt must contain at least one non-whitespace character")
    vae, basis, vae_metadata = _load_model(vae_checkpoint, device)
    predictor, text_metadata = _load_predictor(text_checkpoint, device)
    embedding_model = str(text_metadata.get("embedding_model", DEFAULT_EMBEDDING_MODEL))
    if predictor.latent_dim != vae.latent_dim:
        raise ValueError(
            f"Text model latent width {predictor.latent_dim} does not match VAE width {vae.latent_dim}"
        )
    if "vae_basis_type" in text_metadata and text_metadata["vae_basis_type"] != vae_metadata.get("basis_type"):
        raise ValueError("Text checkpoint and VAE checkpoint use different basis types")

    embedding = _encode_prompt(prompt, embedding_model, device)
    profile_features = (
        torch.from_numpy(parse_profile_features(prompt)).unsqueeze(0).to(device)
        if predictor.profile_dim else None
    )
    control_features = (
        torch.from_numpy(parse_prompt_controls(prompt)).unsqueeze(0).to(device)
        if predictor.control_dim else None
    )
    directional = bool(
        isinstance(predictor, SimilarityLatentRetriever)
        and predictor.prototype_anisotropy is not None
        and _prompt_requests_direction(prompt)
    )
    if directional:
        control_features = torch.ones(1, dtype=torch.bool, device=device)
    with torch.inference_mode():
        latent = predictor(embedding, profile_features, control_features)
        ndf = vae.decode_ndf(latent, basis)
    latent_values = latent[0].float().cpu().tolist()
    print("Predicted latent z=[" + ", ".join(f"{value:.6f}" for value in latent_values) + "]")
    grid_size = int(vae.input_grid_size)
    ndf_grid = ndf.reshape(grid_size, grid_size).float().cpu().numpy()
    if directional:
        ndf_grid = _recenter_ndf_batch(torch.from_numpy(ndf_grid).to(device).unsqueeze(0))[0].cpu().numpy()
    print(f"Decoded NDF shape metrics (raw grid scale): {_ndf_shape_metrics(ndf_grid)}")
    _render_ndf(ndf_grid, prompt, output)
    return output


def generate_comparison(
    prompts: Sequence[str],
    output: Path,
    vae_checkpoint: Path,
    text_checkpoint: Path,
    device: torch.device,
    return_metrics: bool = False,
) -> Path | tuple[Path, list[dict[str, float]]]:
    """Generate a side-by-side heatmap/profile grid for contrasting prompts."""
    if not prompts or any(not prompt.strip() for prompt in prompts):
        raise ValueError("comparison prompts must all contain non-whitespace text")
    vae, basis, vae_metadata = _load_model(vae_checkpoint, device)
    predictor, text_metadata = _load_predictor(text_checkpoint, device)
    if predictor.latent_dim != vae.latent_dim:
        raise ValueError(
            f"Text model latent width {predictor.latent_dim} does not match VAE width {vae.latent_dim}"
        )
    if "vae_basis_type" in text_metadata and text_metadata["vae_basis_type"] != vae_metadata.get("basis_type"):
        raise ValueError("Text checkpoint and VAE checkpoint use different basis types")
    embedding_model = str(text_metadata.get("embedding_model", DEFAULT_EMBEDDING_MODEL))
    embeddings = encode_texts(list(prompts), embedding_model, batch_size=min(32, len(prompts)), device=device).to(device)
    profile_features = (
        torch.from_numpy(np.stack([parse_profile_features(prompt) for prompt in prompts])).to(device)
        if predictor.profile_dim else None
    )
    control_features = (
        torch.from_numpy(np.stack([parse_prompt_controls(prompt) for prompt in prompts])).to(device)
        if predictor.control_dim else None
    )
    directional_flags = [
        bool(_prompt_requests_direction(prompt)) for prompt in prompts
    ] if (
        isinstance(predictor, SimilarityLatentRetriever)
        and predictor.prototype_anisotropy is not None
    ) else [False] * len(prompts)
    if any(directional_flags):
        control_features = torch.as_tensor(directional_flags, dtype=torch.bool, device=device)
    with torch.inference_mode():
        latents = predictor(embeddings, profile_features, control_features)
        decoded = vae.decode_ndf(latents, basis)
    grids = decoded.reshape(len(prompts), int(vae.input_grid_size), int(vae.input_grid_size)).float().cpu().numpy()
    for index, directional in enumerate(directional_flags):
        if directional:
            grids[index] = _recenter_ndf_batch(
                torch.from_numpy(grids[index]).to(device).unsqueeze(0)
            )[0].cpu().numpy()
    shape_metrics = [_ndf_shape_metrics(grid) for grid in grids]
    for prompt, latent in zip(prompts, latents.float().cpu().tolist()):
        print(f"{prompt}: z=[" + ", ".join(f"{value:.6f}" for value in latent) + "]")
    for prompt, metrics in zip(prompts, shape_metrics):
        print(f"{prompt}: NDF shape={metrics}")
    rendered = _render_comparison(list(grids), prompts, output)
    return (rendered, shape_metrics) if return_metrics else rendered


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", help="Material description; omit to enter prompts interactively")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help=f"PNG output path (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--comparison-grid", action="store_true", help="Render the three contrasting material prompts as a comparison grid")
    parser.add_argument("--comparison-output", type=Path, default=DEFAULT_COMPARISON_OUTPUT)
    parser.add_argument("--vae-checkpoint", type=Path, default=DEFAULT_VAE_CHECKPOINT)
    parser.add_argument("--text-checkpoint", type=Path, default=DEFAULT_INFERENCE_CHECKPOINT)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps (default: auto)")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        device = _choose_device(args.device)
        if args.comparison_grid:
            output = generate_comparison(
                CONTRAST_PROMPTS, args.comparison_output, args.vae_checkpoint, args.text_checkpoint, device
            )
            print(f"Saved prompt comparison to {output}")
            return 0
        if args.prompt is not None:
            output = generate(args.prompt, args.output, args.vae_checkpoint, args.text_checkpoint, device)
            print(f"Saved generated NDF to {output}")
            return 0
        while True:
            prompt = input("Material prompt (blank to quit): ").strip()
            if not prompt:
                return 0
            output = generate(prompt, args.output, args.vae_checkpoint, args.text_checkpoint, device)
            print(f"Saved generated NDF to {output}")
    except (FileNotFoundError, RuntimeError, ValueError, OSError) as exc:
        raise SystemExit(f"Text inference failed: {exc}") from exc


if __name__ == "__main__":
    raise SystemExit(main())
