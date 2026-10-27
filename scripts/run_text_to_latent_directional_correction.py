"""Run one inference-only directional retrieval correction candidate."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from prism.eval_text import _qualitative_lobe_metrics
from prism.infer_text import (
    _choose_device,
    _load_predictor,
    _ndf_shape_metrics,
    _prompt_requests_direction,
    _recenter_ndf_batch,
    _render_comparison,
)
from prism.text_dataset import encode_texts
from prism.text_model import SimilarityLatentRetriever
from prism.train_text import _load_artifact
from prism.validate import _load_model


RUN_DIR = ROOT / "checkpoints/experiments/text_to_latent_functionality"
BASE_CHECKPOINT = RUN_DIR / "split_20261031/retrieval.pt"
VAE_PATH = ROOT / "checkpoints/zernike_residual_v3_warm_plateau42/best_model.pt"
CACHE_PATH = ROOT / "checkpoints/text_dataset_grid.npz"
OUTPUT_DIR = RUN_DIR / "directional_inference_correction"
CANDIDATE_PATH = OUTPUT_DIR / "directional_retrieval.pt"
REPORT_PATH = OUTPUT_DIR / "directional_functionality_report.json"
PROMPTS = (
    "shiny mirror-like gold",
    "polished metallic gold with a tight concentrated highlight",
    "gold with a sharp, compact specular lobe",
    "matte rough velvet",
    "soft matte velvet with a broad diffuse highlight",
    "rough velvet with a wide, low-concentration response",
    "anisotropic brushed aluminum",
    "brushed aluminum with an elongated highlight",
    "metal with a narrow highlight aligned to the brushing direction",
)


def save_candidate() -> dict:
    base = torch.load(BASE_CHECKPOINT, map_location="cpu", weights_only=True)
    if base.get("predictor_type") != "e5_similarity_retrieval":
        raise ValueError("expected the retained E5 similarity retrieval checkpoint")
    if set(base["split_material_groups"]["train"]) != set(base["prototype_material_ids"]):
        raise ValueError("prototype bank is not limited to this split's training materials")
    # Calculate the eligibility flag from source grids belonging only to the
    # checkpoint's train partition, as specified by this bounded scope.
    cache = _load_artifact(CACHE_PATH)
    material_ids = cache["material_ids"].astype(str)
    source_anisotropy = []
    for material_id in base["prototype_material_ids"]:
        rows = np.flatnonzero(material_ids == str(material_id))
        if not len(rows):
            raise ValueError(f"training source NDF is missing for prototype {material_id}")
        source_grid = cache["grids"][rows].astype(np.float32).reshape(-1, 64, 64).mean(axis=0)
        source_anisotropy.append(_ndf_shape_metrics(source_grid)["anisotropy"])
    anisotropy = np.asarray(source_anisotropy, dtype=np.float32)
    if int((anisotropy >= 0.30).sum()) < int(base["neighbors"]):
        raise ValueError("too few anisotropic training prototypes to support k=5 filtering")
    model = SimilarityLatentRetriever(
        torch.as_tensor(base["prototype_embeddings"]),
        torch.as_tensor(base["prototype_latents"]),
        int(base["neighbors"]),
        float(base["temperature"]),
        torch.from_numpy(anisotropy),
        0.30,
    )
    candidate = dict(base)
    candidate.update({
        "model_state_dict": model.state_dict(),
        "prototype_embeddings": model.prototype_embeddings.cpu(),
        "prototype_latents": model.prototype_latents.cpu(),
        "prototype_anisotropy": model.prototype_anisotropy.cpu(),
        "directional_threshold": 0.30,
        "directional_filter": "mask decoded train-only prototype anisotropy < 0.30 for explicit directional prompts",
        "directional_centering": "translate decoded in-disk NDF centroid to origin, zero-fill, reapply unit disk",
        "parent_checkpoint": str(BASE_CHECKPOINT),
        "test_labels_read": False,
    })
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(candidate, CANDIDATE_PATH)
    return {"train_prototype_count": len(anisotropy),
            "directional_train_prototype_count": int((anisotropy >= 0.30).sum()),
            "anisotropy_min": float(anisotropy.min()),
            "anisotropy_max": float(anisotropy.max())}


def evaluate(device: torch.device, vae: torch.nn.Module, basis: torch.Tensor, train_summary: dict) -> dict:
    if REPORT_PATH.exists():
        raise RuntimeError(f"candidate report already exists; refusing a second prompt pass: {REPORT_PATH}")
    predictor, metadata = _load_predictor(CANDIDATE_PATH, device)
    encoder = str(metadata["embedding_model"])
    grids = []
    repeat_checks = {}
    for index, prompt in enumerate(PROMPTS):
        embedding = encode_texts([prompt], encoder, batch_size=1, device=device).to(device)
        directional = _prompt_requests_direction(prompt)
        flags = torch.tensor([directional], dtype=torch.bool, device=device)
        with torch.inference_mode():
            latent = predictor(embedding, control_features=flags)
            grid = vae.decode_ndf(latent, basis).reshape(64, 64).float()
            if directional:
                grid = _recenter_ndf_batch(grid.unsqueeze(0))[0]
        grid_np = grid.cpu().numpy()
        if grid_np.shape != (64, 64) or not np.isfinite(grid_np).all() or float(grid_np.min()) < 0:
            raise ValueError(f"invalid NDF output for prompt: {prompt}")
        grids.append(grid_np)
        if index in (0, 6, 8):
            repeated_embedding = encode_texts([prompt], encoder, batch_size=1, device=device).to(device)
            with torch.inference_mode():
                repeated_latent = predictor(
                    repeated_embedding,
                    control_features=torch.tensor([directional], dtype=torch.bool, device=device),
                )
                repeated_grid = vae.decode_ndf(repeated_latent, basis).reshape(64, 64).float()
                if directional:
                    repeated_grid = _recenter_ndf_batch(repeated_grid.unsqueeze(0))[0]
            repeat_np = repeated_grid.cpu().numpy()
            repeat_checks[prompt] = {
                "max_abs_grid_difference": float(np.max(np.abs(repeat_np - grid_np))),
                "stable_within_1e_6_rtol_1e_7_atol": bool(np.allclose(repeat_np, grid_np, rtol=1e-6, atol=1e-7)),
            }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    grids_np = np.stack(grids)
    np.save(OUTPUT_DIR / "fixed_prompt_ndfs.npy", grids_np)
    _render_comparison(list(grids_np), PROMPTS, OUTPUT_DIR / "fixed_prompt_grid.png")
    records = [{"prompt": prompt,
                "directional_filter_applied": _prompt_requests_direction(prompt),
                "shape_metrics": _ndf_shape_metrics(grid),
                "lobe_metrics": _qualitative_lobe_metrics(grid)}
               for prompt, grid in zip(PROMPTS, grids_np)]
    gold, velvet, aluminum = records[:3], records[3:6], records[6:]
    import statistics
    gates = {
        "gold_median_central_mass_exceeds_velvet": statistics.median(x["shape_metrics"]["central_mass_fraction"] for x in gold) > statistics.median(x["shape_metrics"]["central_mass_fraction"] for x in velvet),
        "gold_median_mean_radius_below_velvet": statistics.median(x["shape_metrics"]["mean_radius"] for x in gold) < statistics.median(x["shape_metrics"]["mean_radius"] for x in velvet),
        "all_aluminum_anisotropy_at_least_0_30": all(x["shape_metrics"]["anisotropy"] >= 0.30 for x in aluminum),
        "all_aluminum_centroid_offset_at_most_0_15": all(np.hypot(x["lobe_metrics"]["intensity_centroid_x"], x["lobe_metrics"]["intensity_centroid_y"]) <= 0.15 for x in aluminum),
        "all_aluminum_component_count_at_most_2": all(x["lobe_metrics"]["component_count"] <= 2 for x in aluminum),
        "repeat_inference_stable": all(x["stable_within_1e_6_rtol_1e_7_atol"] for x in repeat_checks.values()),
        "all_outputs_finite_nonnegative_64x64": bool(grids_np.shape == (9,64,64) and np.isfinite(grids_np).all() and np.min(grids_np) >= 0),
    }
    cli_checks = []
    for suffix in ("a", "b"):
        output = OUTPUT_DIR / f"cli_repeat_{suffix}.png"
        process = subprocess.run(
            [sys.executable, "-m", "prism.infer_text", "--prompt", PROMPTS[6],
             "--output", str(output), "--vae-checkpoint", str(VAE_PATH),
             "--text-checkpoint", str(CANDIDATE_PATH), "--device", str(device)],
            cwd=ROOT, capture_output=True, text=True, timeout=600,
        )
        record = {"returncode": process.returncode, "output": str(output),
                  "file_created": output.is_file(), "stdout": process.stdout[-1500:],
                  "stderr": process.stderr[-1500:],
                  "sha256": hashlib.sha256(output.read_bytes()).hexdigest() if output.is_file() else None}
        cli_checks.append(record)
        if process.returncode != 0 or not output.is_file():
            raise RuntimeError(f"directional candidate CLI failed: {process.stderr}")
    gates["public_cli_repeat_output_identical"] = (
        cli_checks[0]["sha256"] == cli_checks[1]["sha256"]
    )
    report = {
        "goal": "directional text-to-latent inference correction",
        "status": "READY" if all(gates.values()) else "NOT_READY_BOUNDED_ATTEMPT_COMPLETE",
        "candidate_checkpoint": str(CANDIDATE_PATH),
        "parent_checkpoint": str(BASE_CHECKPOINT),
        "vae_checkpoint": str(VAE_PATH),
        "embedding_model": encoder,
        "training_prototype_anisotropy_summary": train_summary,
        "directional_prompt_filter": "Use k=5 and temperature=0.025 over train-only prototypes whose decoded anisotropy is >=0.30.",
        "directional_grid_correction": "Translate in-disk intensity centroid to the origin with bilinear zero-fill and reapply the unit-disk mask.",
        "prompts": records,
        "repeat_checks": repeat_checks,
        "public_cli_repeat_checks": cli_checks,
        "gates": gates,
        "all_functionality_gates_pass": all(gates.values()),
        "latent_mse": {
            "previous_retrieval_mean_diagnostic": 0.5125938852628072,
            "blocking_threshold": None,
            "note": "Latents are unchanged by the inference grid correction; no heldout targets were read in this follow-up.",
        },
        "test_labels_read": False,
        "artifacts": {"ndfs": str(OUTPUT_DIR / "fixed_prompt_ndfs.npy"),
                      "grid": str(OUTPUT_DIR / "fixed_prompt_grid.png")},
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "gates": gates,
                      "report": str(REPORT_PATH)}, indent=2))
    return report


def main() -> int:
    if REPORT_PATH.exists():
        raise RuntimeError(f"candidate report already exists; refusing to retrain or rerun prompts: {REPORT_PATH}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = _choose_device("auto")
    vae, basis, _ = _load_model(VAE_PATH, device)
    summary = save_candidate()
    report = evaluate(device, vae, basis, summary)
    print(f"Candidate checkpoint: {CANDIDATE_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
