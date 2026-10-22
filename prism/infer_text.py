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
from prism.text_model import TextToLatent
from prism.text_dataset import DEFAULT_EMBEDDING_MODEL, encode_texts
from prism.train_text import DEFAULT_CHECKPOINT, DEFAULT_VAE_CHECKPOINT
from prism.validate import _load_model


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "results" / "text_demo.png"
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
    predictor = TextToLatent(embedding_dim, latent_dim).to(device)
    predictor.load_state_dict(checkpoint["model_state_dict"])
    predictor.eval()
    return predictor, checkpoint


def _encode_prompt(prompt: str, model_name: str, device: torch.device) -> torch.Tensor:
    """Use the same text preprocessing as the dataset builder."""
    return encode_texts([prompt], model_name, batch_size=1, device=device).to(device)


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


def generate(prompt: str, output: Path, vae_checkpoint: Path, text_checkpoint: Path, device: torch.device) -> Path:
    if not prompt.strip():
        raise ValueError("Prompt must contain at least one non-whitespace character")
    vae, basis, vae_metadata = _load_model(vae_checkpoint, device)
    predictor, text_metadata = _load_predictor(text_checkpoint, device)
    embedding_model = str(text_metadata.get("embedding_model", DEFAULT_EMBEDDING_MODEL))
    if embedding_model != DEFAULT_EMBEDDING_MODEL:
        raise ValueError(
            f"Unsupported text embedding model {embedding_model!r}; expected {DEFAULT_EMBEDDING_MODEL!r}"
        )
    if predictor.latent_dim != vae.latent_dim:
        raise ValueError(
            f"Text model latent width {predictor.latent_dim} does not match VAE width {vae.latent_dim}"
        )
    if "vae_basis_type" in text_metadata and text_metadata["vae_basis_type"] != vae_metadata.get("basis_type"):
        raise ValueError("Text checkpoint and VAE checkpoint use different basis types")

    embedding = _encode_prompt(prompt, embedding_model, device)
    with torch.inference_mode():
        latent = predictor(embedding)
        coefficients = vae.decoder(latent)
        ndf = torch.relu(coefficients @ basis)
    latent_values = latent[0].float().cpu().tolist()
    print("Predicted latent z=[" + ", ".join(f"{value:.6f}" for value in latent_values) + "]")
    grid_size = int(vae.input_grid_size)
    ndf_grid = ndf.reshape(grid_size, grid_size).float().cpu().numpy()
    _render_ndf(ndf_grid, prompt, output)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", help="Material description; omit to enter prompts interactively")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help=f"PNG output path (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--vae-checkpoint", type=Path, default=DEFAULT_VAE_CHECKPOINT)
    parser.add_argument("--text-checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps (default: auto)")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        device = _choose_device(args.device)
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
