"""Render a README showcase from the released E5 and PRISM ONNX models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = ROOT / "out" / "prism_models"
DEFAULT_OUTPUT = ROOT / "assets" / "readme" / "prism-model-generated-outputs.png"
PROMPTS = (
    ("shiny mirror-like gold", False),
    ("matte rough velvet", False),
    ("anisotropic brushed aluminum", True),
)
SEQUENCE_LENGTH = 128


def recenter_directional_grid(grid: np.ndarray) -> np.ndarray:
    """Match PRISM's directional centroid recentering postprocess."""
    size = grid.shape[0]
    coords = torch.linspace(-1.0, 1.0, size, dtype=torch.float32)
    y, x = torch.meshgrid(coords, coords, indexing="ij")
    disk = x.square() + y.square() <= 1.0
    weights = torch.from_numpy(grid).clamp_min(0.0) * disk
    mass = weights.sum().clamp_min(1e-12)
    center_x = (weights * x).sum() / mass
    center_y = (weights * y).sum() / mass
    sample_grid = torch.stack((x + center_x, y + center_y), dim=-1).unsqueeze(0)
    shifted = F.grid_sample(
        weights[None, None], sample_grid,
        mode="bilinear", padding_mode="zeros", align_corners=True,
    )[0, 0]
    return (shifted * disk).numpy()


def render(model_dir: Path, output: Path) -> Path:
    tokenizer_dir = model_dir / "tokenizer"
    if not (model_dir / "tokenizer.json").is_file() and (tokenizer_dir / "tokenizer.json").is_file():
        tokenizer_dir = tokenizer_dir
    else:
        tokenizer_dir = model_dir
    required = (
        "e5_encoder.onnx", "e5_encoder.onnx.data",
        "retriever.onnx", "retriever.onnx.data",
        "vae_decoder.onnx", "vae_decoder.onnx.data",
    )
    missing = [name for name in required if not (model_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing model assets in {model_dir}: {', '.join(missing)}")
    if not (tokenizer_dir / "tokenizer.json").is_file():
        raise FileNotFoundError(f"Tokenizer assets not found in {tokenizer_dir}")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
    encoder = ort.InferenceSession(str(model_dir / "e5_encoder.onnx"), providers=["CPUExecutionProvider"])
    retriever = ort.InferenceSession(str(model_dir / "retriever.onnx"), providers=["CPUExecutionProvider"])
    decoder = ort.InferenceSession(str(model_dir / "vae_decoder.onnx"), providers=["CPUExecutionProvider"])

    coords = np.linspace(-1.0, 1.0, 64, dtype=np.float32)
    y, x = np.meshgrid(coords, coords, indexing="ij")
    disk = x * x + y * y <= 1.0
    grids: list[np.ndarray] = []
    for prompt, directional in PROMPTS:
        tokens = tokenizer(
            f"query: {prompt}", padding="max_length", truncation=True,
            max_length=SEQUENCE_LENGTH, return_tensors="np",
        )
        embedding = encoder.run(["normalized_embedding"], {
            "input_ids": tokens["input_ids"].astype(np.int64),
            "attention_mask": tokens["attention_mask"].astype(np.int64),
        })[0]
        latent = retriever.run(["predicted_latent"], {
            "embedding": embedding,
            "directional": np.asarray([directional], dtype=np.bool_),
        })[0]
        grid = decoder.run(["ndf_grid_flat"], {"latent": latent})[0].reshape(64, 64).astype(np.float32)
        if directional:
            grid = recenter_directional_grid(grid)
        grids.append(np.where(disk, np.maximum(grid, 0.0), np.nan))

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "text.color": "#e8edf5",
        "axes.labelcolor": "#bdc7d5", "xtick.color": "#aeb8c6",
        "ytick.color": "#aeb8c6",
    })
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.2), constrained_layout=True, facecolor="#0c1422")
    fig.suptitle("Real PRISM inference outputs", fontsize=19, fontweight="bold", color="#f4f6fa")
    for index, ((prompt, _), grid) in enumerate(zip(PROMPTS, grids)):
        peak = float(np.nanmax(grid))
        normalized = grid / peak if peak > 0 else np.nan_to_num(grid)
        heat = axes[0, index]
        heat.set_facecolor("#05080d")
        heat.imshow(normalized, origin="lower", extent=(-1, 1, -1, 1), cmap="magma", vmin=0, vmax=1, interpolation="nearest")
        heat.set_title(prompt, fontsize=11, pad=10, color="#f0f3f8")
        heat.set_xlabel("Half-vector x")
        if index == 0:
            heat.set_ylabel("Half-vector y")
        heat.set_xticks([-1, 0, 1]); heat.set_yticks([-1, 0, 1])
        for spine in heat.spines.values():
            spine.set_color("#40516a")

        profile_ax = axes[1, index]
        profile_ax.set_facecolor("#111c2d")
        profile = np.nan_to_num(grid[32])
        profile_peak = float(profile.max())
        if profile_peak > 0:
            profile /= profile_peak
        profile_ax.plot(coords, profile, color="#74d7c2", linewidth=2.2)
        profile_ax.fill_between(coords, profile, color="#74d7c2", alpha=0.13)
        profile_ax.set_ylim(bottom=0)
        profile_ax.set_xlabel("Half-vector x")
        if index == 0:
            profile_ax.set_ylabel("Center-row intensity\n(normalized)")
        profile_ax.grid(alpha=0.2, color="#8796aa")
        for spine in profile_ax.spines.values():
            spine.set_color("#40516a")
        profile_ax.tick_params(colors="#aeb8c6")

    fig.text(
        0.5, -0.012,
        "E5 encoder → PRISM retriever → Residual V3 decoder · each heatmap normalized to its own peak",
        ha="center", color="#9eacbf", fontsize=9,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(json.dumps({
        "output": str(output),
        "model_dir": str(model_dir),
        "prompts": [prompt for prompt, _ in PROMPTS],
        "directional_flags": [directional for _, directional in PROMPTS],
        "peak_values": [float(np.nanmax(grid)) for grid in grids],
    }, indent=2))
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    render(args.model_dir, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
