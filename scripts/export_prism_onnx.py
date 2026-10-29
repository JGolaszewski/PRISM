"""Export selected PRISM prompt models to ONNX and compare with PyTorch."""

from __future__ import annotations

import json
import argparse
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / ".venv/Lib/site-packages"))
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import onnxruntime as ort
from transformers import AutoModel, AutoTokenizer

from prism.infer_text import (  # noqa: E402
    _prompt_requests_direction,
    _recenter_ndf_batch,
    _load_predictor,
    DEFAULT_INFERENCE_CHECKPOINT,
)
from prism.text_dataset import parse_profile_features, parse_prompt_controls  # noqa: E402
from prism.text_model import SimilarityLatentRetriever, TextToLatent  # noqa: E402
from prism.validate import _load_model  # noqa: E402

TEXT_CHECKPOINT = DEFAULT_INFERENCE_CHECKPOINT
VAE_CHECKPOINT = ROOT / "checkpoints/zernike_residual_v3_warm_plateau42/best_model.pt"
MODEL_ID = "intfloat/e5-large-v2"
SEQUENCE_LENGTH = 128
PROMPTS = (
    "shiny mirror-like gold",
    "matte rough velvet",
    "anisotropic brushed aluminum",
)


class E5Export(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        return F.normalize(pooled.float(), p=2, dim=1)


class RetrieverExport(nn.Module):
    """Vectorized equivalent of SimilarityLatentRetriever.forward."""
    def __init__(self, predictor: nn.Module) -> None:
        super().__init__()
        self.register_buffer("prototype_embeddings", predictor.prototype_embeddings.detach().float())
        self.register_buffer("prototype_latents", predictor.prototype_latents.detach().float())
        self.register_buffer("prototype_anisotropy", predictor.prototype_anisotropy.detach().float())
        self.neighbors = predictor.neighbors
        self.temperature = predictor.temperature
        self.directional_threshold = predictor.directional_threshold

    def forward(self, embedding: torch.Tensor, directional: torch.Tensor) -> torch.Tensor:
        query = F.normalize(embedding.float(), dim=-1)
        similarity = query @ self.prototype_embeddings.T
        eligible = self.prototype_anisotropy >= self.directional_threshold
        disallowed = directional.float().reshape(-1, 1) * (~eligible).float().reshape(1, -1)
        # -1e9 matches -inf for top-k because this checkpoint has >= k eligible
        # anisotropic prototypes, while avoiding an unsupported ORT Where kernel.
        scores = similarity - disallowed * 1.0e9
        top_scores, indices = scores.topk(self.neighbors, dim=-1)
        weights = torch.softmax((top_scores - top_scores.amax(dim=-1, keepdim=True)) / self.temperature, dim=-1)
        neighbors = self.prototype_latents[indices]
        return (neighbors * weights.unsqueeze(-1)).sum(dim=1)


class TextPredictorExportWithControls(nn.Module):
    """Concrete two-input wrapper for MLPs trained with control features only."""

    def __init__(self, predictor: TextToLatent) -> None:
        super().__init__()
        self.predictor = predictor

    def forward(self, embedding: torch.Tensor, control_features: torch.Tensor) -> torch.Tensor:
        return self.predictor(embedding, control_features=control_features)


class TextPredictorExportWithProfiles(nn.Module):
    """Concrete two-input wrapper for MLPs trained with profile features only."""

    def __init__(self, predictor: TextToLatent) -> None:
        super().__init__()
        self.predictor = predictor

    def forward(self, embedding: torch.Tensor, profile_features: torch.Tensor) -> torch.Tensor:
        return self.predictor(embedding, profile_features=profile_features)


class TextPredictorExportEmbeddingOnly(nn.Module):
    """Concrete one-input wrapper for plain embedding-to-latent MLPs."""

    def __init__(self, predictor: TextToLatent) -> None:
        super().__init__()
        self.predictor = predictor

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.predictor(embedding)


class TextPredictorExportWithProfilesAndControls(nn.Module):
    """Concrete three-input wrapper for MLPs using both feature families."""

    def __init__(self, predictor: TextToLatent) -> None:
        super().__init__()
        self.predictor = predictor

    def forward(
        self, embedding: torch.Tensor, profile_features: torch.Tensor, control_features: torch.Tensor
    ) -> torch.Tensor:
        return self.predictor(embedding, profile_features, control_features)


class DecoderExport(nn.Module):
    def __init__(self, vae: nn.Module, basis: torch.Tensor) -> None:
        super().__init__()
        self.vae = vae
        self.register_buffer("basis", basis)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.vae.decode_ndf(latent, self.basis)


def export_onnx(module: nn.Module, inputs: tuple[torch.Tensor, ...], path: Path,
                input_names: list[str], output_names: list[str]) -> None:
    module.eval()
    torch.onnx.export(
        module, inputs, path,
        input_names=input_names,
        output_names=output_names,
        opset_version=18,
        dynamo=True,
    )


def metrics(reference: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    delta = np.asarray(actual, np.float64) - np.asarray(reference, np.float64)
    return {
        "max_abs": float(np.max(np.abs(delta))),
        "mean_abs": float(np.mean(np.abs(delta))),
        "rmse": float(np.sqrt(np.mean(delta * delta))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vae-checkpoint", type=Path, default=VAE_CHECKPOINT)
    parser.add_argument("--text-checkpoint", type=Path, default=TEXT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "out/prism_models")
    args = parser.parse_args()
    vae_checkpoint, text_checkpoint = args.vae_checkpoint, args.text_checkpoint
    if not text_checkpoint.is_file():
        raise FileNotFoundError(f"Text checkpoint not found: {text_checkpoint}")
    if not vae_checkpoint.is_file():
        raise FileNotFoundError(f"VAE checkpoint not found: {vae_checkpoint}")

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
    text_encoder = AutoModel.from_pretrained(MODEL_ID, local_files_only=True).to(device).eval()
    e5 = E5Export(text_encoder).eval()
    ids_example = torch.zeros((1, SEQUENCE_LENGTH), dtype=torch.long)
    mask_example = torch.zeros((1, SEQUENCE_LENGTH), dtype=torch.long)
    export_onnx(e5, (ids_example, mask_example), out / "e5_encoder.onnx",
                ["input_ids", "attention_mask"], ["normalized_embedding"])
    tokenizer.save_pretrained(out / "tokenizer")

    predictor, text_meta = _load_predictor(text_checkpoint, device)
    embedding_example = torch.zeros((1, predictor.embedding_dim), dtype=torch.float32)
    directional_retrieval = isinstance(predictor, SimilarityLatentRetriever)
    predictor_input_names = ["embedding"]
    predictor_inputs: list[torch.Tensor] = [embedding_example]
    if directional_retrieval:
        predictor_export = RetrieverExport(predictor).eval()
        predictor_input_names.append("directional")
        predictor_inputs.append(torch.zeros((1,), dtype=torch.bool))
    else:
        if not isinstance(predictor, TextToLatent):
            raise TypeError(f"Unsupported text predictor type: {type(predictor).__name__}")
        profile_example = (
            torch.zeros((1, predictor.profile_dim), dtype=torch.float32)
            if predictor.profile_dim else None
        )
        control_example = (
            torch.zeros((1, predictor.control_dim), dtype=torch.float32)
            if predictor.control_dim else None
        )
        if profile_example is not None:
            predictor_input_names.append("profile_features")
            predictor_inputs.append(profile_example)
        if control_example is not None:
            predictor_input_names.append("control_features")
            predictor_inputs.append(control_example)
        if profile_example is not None and control_example is not None:
            predictor_export = TextPredictorExportWithProfilesAndControls(predictor).eval()
        elif profile_example is not None:
            predictor_export = TextPredictorExportWithProfiles(predictor).eval()
        elif control_example is not None:
            predictor_export = TextPredictorExportWithControls(predictor).eval()
        else:
            predictor_export = TextPredictorExportEmbeddingOnly(predictor).eval()
    export_onnx(
        predictor_export, tuple(predictor_inputs), out / "retriever.onnx",
        predictor_input_names, ["predicted_latent"],
    )

    vae, basis, vae_meta = _load_model(vae_checkpoint, device)
    decoder = DecoderExport(vae.eval(), basis).eval()
    latent_example = torch.zeros((1, vae.latent_dim), dtype=torch.float32)
    export_onnx(decoder, (latent_example,), out / "vae_decoder.onnx",
                ["latent"], ["ndf_grid_flat"])

    # CPUExecutionProvider is the portable reference target. Engine deployments
    # can select another ORT Execution Provider after this parity run.
    e5_session = ort.InferenceSession(str(out / "e5_encoder.onnx"), providers=["CPUExecutionProvider"])
    retrieval_session = ort.InferenceSession(str(out / "retriever.onnx"), providers=["CPUExecutionProvider"])
    decoder_session = ort.InferenceSession(str(out / "vae_decoder.onnx"), providers=["CPUExecutionProvider"])
    grid_size = int(vae.input_grid_size)
    comparisons: dict[str, dict[str, dict[str, float]]] = {}

    for prompt in PROMPTS:
        is_directional = _prompt_requests_direction(prompt)
        tokens = tokenizer(
            [f"query: {prompt}"],
            padding="max_length",
            truncation=True,
            max_length=SEQUENCE_LENGTH,
            return_tensors="pt",
        )
        ids = tokens["input_ids"].to(torch.int64)
        mask = tokens["attention_mask"].to(torch.int64)
        with torch.inference_mode():
            py_embedding = e5(ids, mask).cpu().numpy()
        ort_embedding = e5_session.run(None, {
            "input_ids": ids.numpy(),
            "attention_mask": mask.numpy(),
        })[0]

        profile_features = (
            torch.from_numpy(parse_profile_features(prompt)).unsqueeze(0)
            if getattr(predictor, "profile_dim", 0) else None
        )
        control_features = None
        if getattr(predictor, "control_dim", 0):
            if directional_retrieval:
                control_features = torch.tensor([is_directional], dtype=torch.bool)
            else:
                control_features = torch.from_numpy(parse_prompt_controls(prompt)).unsqueeze(0).float()
        with torch.inference_mode():
            py_latent = predictor(
                torch.from_numpy(py_embedding), profile_features, control_features
            ).cpu().numpy()
        predictor_feed = {"embedding": ort_embedding}
        if directional_retrieval:
            predictor_feed["directional"] = control_features.numpy()
        else:
            if profile_features is not None:
                predictor_feed["profile_features"] = profile_features.numpy()
            if control_features is not None:
                predictor_feed["control_features"] = control_features.numpy()
        ort_latent = retrieval_session.run(None, predictor_feed)[0]

        with torch.inference_mode():
            py_grid = decoder(torch.from_numpy(py_latent)).cpu().numpy()
        ort_grid = decoder_session.run(None, {"latent": ort_latent})[0]
        recenter = directional_retrieval and is_directional
        if recenter:
            py_grid = _recenter_ndf_batch(torch.from_numpy(py_grid.reshape(1, grid_size, grid_size)))[0].numpy().reshape(1, -1)
            ort_grid = _recenter_ndf_batch(torch.from_numpy(ort_grid.reshape(1, grid_size, grid_size)))[0].numpy().reshape(1, -1)

        comparisons[prompt] = {
            "directional_correction_applied": recenter,
            "embedding": metrics(py_embedding, ort_embedding),
            "latent": metrics(py_latent, ort_latent),
            "ndf": metrics(py_grid, ort_grid),
        }
        # Keep the exact reference/ORT arrays for downstream C++ smoke tests.
        np.save(out / f"{_slug(prompt)}_python_ndf.npy", py_grid.reshape(grid_size, grid_size).astype(np.float32))
        np.save(out / f"{_slug(prompt)}_onnx_ndf.npy", ort_grid.reshape(grid_size, grid_size).astype(np.float32))

    report = {
        "model_id": MODEL_ID,
        "sequence_length": SEQUENCE_LENGTH,
        "text_checkpoint": str(text_checkpoint),
        "vae_checkpoint": str(vae_checkpoint),
        "text_predictor": ({
            "type": "e5_similarity_retrieval",
            "neighbors": predictor.neighbors,
            "temperature": predictor.temperature,
            "directional_threshold": predictor.directional_threshold,
            "prototype_count": int(predictor.prototype_embeddings.shape[0]),
        } if directional_retrieval else {
            "type": "text_to_latent_mlp",
            "profile_dim": predictor.profile_dim,
            "control_dim": predictor.control_dim,
            "hidden_dims": list(predictor.hidden_dims),
            "architecture_version": predictor.architecture_version,
        }),
        "vae": {
            "latent_dim": vae.latent_dim,
            "grid_size": vae.input_grid_size,
            "basis_type": vae_meta.get("basis_type"),
            "residual_decoder_version": vae_meta.get("residual_decoder_version"),
        },
        "text_checkpoint_metadata": {
            key: text_meta.get(key) for key in (
                "embedding_model", "predictor_type", "neighbors", "temperature",
                "control_dim", "profile_dim", "validation_latent_mse",
            )
        },
        "providers": {"encoder": e5_session.get_providers(), "retriever": retrieval_session.get_providers(), "decoder": decoder_session.get_providers()},
        "comparisons": comparisons,
    }
    (out / "parity_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def _slug(prompt: str) -> str:
    return "_".join("".join(ch.lower() if ch.isalnum() else " " for ch in prompt).split())


if __name__ == "__main__":
    main()
