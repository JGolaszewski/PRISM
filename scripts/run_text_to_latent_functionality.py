"""Run the bounded fresh-cohort text-to-latent functionality milestone."""

from __future__ import annotations

import hashlib
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from prism.eval_text import _qualitative_lobe_metrics
from prism.infer_text import (
    CONTRAST_PROMPTS,
    _choose_device,
    _ndf_shape_metrics,
    _render_comparison,
    _load_predictor,
)
from prism.text_dataset import encode_texts
from prism.train_text import _load_artifact, _split_from_material_groups, train_text
from prism.validate import _load_model
from prism.text_model import SimilarityLatentRetriever


MANIFEST_PATH = ROOT / "results/text_ndf_prompt_quality/fresh_retraining_split_manifest.json"
DATA_PATH = ROOT / "checkpoints/text_dataset_grid.npz"
SYNTHETIC_PATH = ROOT / "checkpoints/text_synthetic_controls_v2.npz"
VAE_PATH = ROOT / "checkpoints/zernike_residual_v3_warm_plateau42/best_model.pt"
RUN_DIR = ROOT / "checkpoints/experiments/text_to_latent_functionality"
REPORT_PATH = RUN_DIR / "functionality_report.json"
EPOCH_CAP = 60
TRAINING_WALL_CAP_SECONDS = 2 * 60 * 60


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_family(name: str) -> str:
    return "RGL extracted NDF estimate" if name.startswith("rgl_") else "MERL BRDF-derived NDF proxy"


def material_prototypes(
    data: dict[str, np.ndarray], row_indices: np.ndarray
) -> tuple[list[str], np.ndarray, np.ndarray]:
    ids = sorted(set(map(str, data["material_ids"][row_indices])))
    embeddings, latents = [], []
    for material_id in ids:
        rows = row_indices[data["material_ids"][row_indices] == material_id]
        embedding = data["embeddings"][rows].astype(np.float32).mean(axis=0)
        norm = float(np.linalg.norm(embedding))
        if not norm:
            raise ValueError(f"zero text prototype for {material_id}")
        embeddings.append(embedding / norm)
        latents.append(data["latents"][rows].astype(np.float32).mean(axis=0))
    return ids, np.stack(embeddings), np.stack(latents)


def predict_retrieval(
    query: np.ndarray,
    prototype_embeddings: np.ndarray,
    prototype_latents: np.ndarray,
    neighbors: int,
    temperature: float,
) -> np.ndarray:
    similarity = F.normalize(torch.as_tensor(query, dtype=torch.float32), dim=-1) @ torch.as_tensor(
        prototype_embeddings, dtype=torch.float32
    ).T
    scores, indices = similarity.topk(neighbors, dim=-1)
    weights = torch.softmax((scores - scores.amax(dim=-1, keepdim=True)) / temperature, dim=-1)
    return (torch.as_tensor(prototype_latents)[indices] * weights[..., None]).sum(dim=1).numpy()


def select_retrieval(
    data: dict[str, np.ndarray],
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
) -> tuple[dict, list[str], np.ndarray, np.ndarray]:
    ids, prototype_embeddings, prototype_latents = material_prototypes(data, train_indices)
    validation_embeddings = data["embeddings"][validation_indices].astype(np.float32)
    targets = data["latents"][validation_indices].astype(np.float32)
    rows = []
    for neighbors in (1, 3, 5, 8, 12, 16, 24, 32):
        if neighbors > len(ids):
            continue
        for temperature in (0.01, 0.025, 0.05, 0.1, 0.2, 0.5, 1.0):
            predicted = predict_retrieval(
                validation_embeddings, prototype_embeddings, prototype_latents,
                neighbors, temperature,
            )
            rows.append({
                "neighbors": neighbors,
                "temperature": temperature,
                "validation_latent_mse": float(np.mean((predicted - targets) ** 2)),
            })
    chosen = min(rows, key=lambda row: (row["validation_latent_mse"], row["neighbors"], row["temperature"]))
    chosen["validation_search"] = rows
    chosen["validation_material_count"] = len(set(map(str, data["material_ids"][validation_indices])))
    return chosen, ids, prototype_embeddings, prototype_latents


def evaluate_rows(
    data: dict[str, np.ndarray], indices: np.ndarray, predicted_latents: np.ndarray,
    vae: torch.nn.Module, basis: torch.Tensor, device: torch.device,
) -> dict:
    target_latents = data["latents"][indices].astype(np.float32)
    true_grids = data["grids"][indices].astype(np.float32).reshape(-1, 4096)
    with torch.inference_mode():
        decoded = vae.decode_ndf(
            torch.as_tensor(predicted_latents, dtype=torch.float32, device=device), basis
        ).reshape(-1, 4096).float().cpu().numpy()
    metrics = {
        "sample_count": int(len(indices)),
        "material_count": int(len(set(map(str, data["material_ids"][indices])))),
        "latent_mse": float(np.mean((predicted_latents - target_latents) ** 2)),
        "reconstruction_log_l1": float(np.mean(np.abs(np.log1p(decoded) - np.log1p(true_grids)))),
        "finite_decoded_fraction": float(np.isfinite(decoded).mean()),
        "nonnegative_decoded_fraction": float((decoded >= 0).mean()),
        "by_source_family": {},
    }
    families = np.asarray([source_family(str(name)) for name in data["material_names"][indices]])
    for family in sorted(set(families)):
        mask = families == family
        metrics["by_source_family"][family] = {
            "sample_count": int(mask.sum()),
            "material_count": int(len(set(map(str, data["material_ids"][indices][mask])))),
            "latent_mse": float(np.mean((predicted_latents[mask] - target_latents[mask]) ** 2)),
            "reconstruction_log_l1": float(np.mean(np.abs(np.log1p(decoded[mask]) - np.log1p(true_grids[mask])))),
        }
    return metrics


def validate_partitions(manifest: dict, data: dict[str, np.ndarray], synthetic: dict[str, np.ndarray]) -> list[dict]:
    all_ids = set(map(str, data["material_ids"]))
    audits = []
    for split in manifest["splits"]:
        groups = {key: set(map(str, split[f"{key}_material_ids"])) for key in ("train", "validation", "test")}
        if set.union(*groups.values()) != all_ids:
            raise ValueError(f"manifest does not cover canonical materials for seed {split['seed']}")
        if (groups["train"] & groups["validation"] or groups["train"] & groups["test"]
                or groups["validation"] & groups["test"]):
            raise ValueError(f"manifest partitions overlap for seed {split['seed']}")
        indices = _split_from_material_groups(
            data["material_ids"], sorted(groups["train"]), sorted(groups["validation"]), sorted(groups["test"])
        )
        if set(map(str, data["material_ids"][indices[2]])) != groups["test"]:
            raise ValueError(f"test rows differ from the declared test IDs for seed {split['seed']}")
        row_lineage = [set(str(value).split("|")) - {""} for value in synthetic["material_ids"]]
        counts = {
            label: sum(bool(members) and members <= groups[label] for members in row_lineage)
            for label in ("train", "validation", "test")
        }
        crosses = sum(
            bool(members) and not any(members <= group for group in groups.values())
            for members in row_lineage
        )
        if crosses:
            raise ValueError(f"synthetic lineage crosses manifest partitions for seed {split['seed']}")
        audits.append({"seed": int(split["seed"]), "synthetic_lineage_rows_by_partition": counts,
                       "cross_partition_or_unassigned_rows": int(crosses),
                       "synthetic_rows_used_for_training": 0})
    return audits


def prompt_check(
    checkpoint: Path, vae: torch.nn.Module, basis: torch.Tensor, device: torch.device,
    output_dir: Path,
) -> dict:
    predictor, metadata = _load_predictor(checkpoint, device)
    encoder = str(metadata["embedding_model"])
    # Match the public CLI path: encode one prompt per call, so repeat checks do
    # not compare batched-transformer numerics against single-prompt numerics.
    embeddings = torch.cat([
        encode_texts([prompt], encoder, batch_size=1, device=device) for prompt in PROMPTS
    ]).to(device)
    with torch.inference_mode():
        latents = predictor(embeddings)
        decoded = vae.decode_ndf(latents, basis).reshape(len(PROMPTS), 64, 64).float().cpu().numpy()
    if decoded.shape != (9, 64, 64) or not np.isfinite(decoded).all() or float(decoded.min()) < 0:
        raise ValueError("prompt inference did not produce finite, nonnegative 64x64 NDFs")
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "fixed_prompt_ndfs.npy", decoded)
    _render_comparison(list(decoded), PROMPTS, output_dir / "fixed_prompt_grid.png")
    repeated = {}
    with torch.inference_mode():
        for index in (0, 4, 8):
            second_embedding = encode_texts([PROMPTS[index]], encoder, batch_size=1, device=device).to(device)
            second_latent = predictor(second_embedding)
            second = vae.decode_ndf(second_latent, basis).reshape(64, 64).float().cpu().numpy()
            repeated[PROMPTS[index]] = {
                "max_abs_grid_difference": float(np.max(np.abs(second - decoded[index]))),
                "stable_within_1e_6": bool(np.allclose(second, decoded[index], rtol=1e-6, atol=1e-7)),
            }
    records = []
    for prompt, grid in zip(PROMPTS, decoded):
        records.append({"prompt": prompt, "shape_metrics": _ndf_shape_metrics(grid),
                        "lobe_metrics": _qualitative_lobe_metrics(grid)})
    gold = [item["shape_metrics"] for item in records[:3]]
    velvet = [item["shape_metrics"] for item in records[3:6]]
    aluminum = records[6:]
    gates = {
        "gold_median_central_mass_exceeds_velvet": statistics.median(x["central_mass_fraction"] for x in gold) > statistics.median(x["central_mass_fraction"] for x in velvet),
        "gold_median_mean_radius_below_velvet": statistics.median(x["mean_radius"] for x in gold) < statistics.median(x["mean_radius"] for x in velvet),
        "all_aluminum_anisotropy_at_least_0_30": all(x["shape_metrics"]["anisotropy"] >= 0.30 for x in aluminum),
        "all_aluminum_centroid_offset_at_most_0_15": all(np.hypot(x["lobe_metrics"]["intensity_centroid_x"], x["lobe_metrics"]["intensity_centroid_y"]) <= 0.15 for x in aluminum),
        "all_aluminum_component_count_at_most_2": all(x["lobe_metrics"]["component_count"] <= 2 for x in aluminum),
        "repeat_inference_stable": all(x["stable_within_1e_6"] for x in repeated.values()),
        "finite_nonnegative_64x64_output": True,
    }
    return {"checkpoint": str(checkpoint), "prompts": records, "repeat_checks": repeated,
            "gates": gates, "all_functionality_gates_pass": all(gates.values()),
            "artifacts": {"ndfs": str(output_dir / "fixed_prompt_ndfs.npy"),
                          "grid": str(output_dir / "fixed_prompt_grid.png")}}


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


def main() -> int:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    if REPORT_PATH.exists():
        raise RuntimeError(
            f"bounded report already exists at {REPORT_PATH}; refusing to retrain or rescore the fresh test cohort"
        )
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if not manifest.get("predeclared_before_future_training"):
        raise ValueError("fresh split manifest is not marked predeclared")
    hashes = {"canonical": sha256(DATA_PATH), "synthetic": sha256(SYNTHETIC_PATH)}
    if hashes["canonical"] != manifest["canonical_sha256"] or hashes["synthetic"] != manifest["synthetic_sha256"]:
        raise ValueError("cache hash differs from the predeclared manifest")
    data, synthetic = _load_artifact(DATA_PATH), _load_artifact(SYNTHETIC_PATH)
    if str(data["embedding_model"].reshape(-1)[0]) != "intfloat/e5-large-v2":
        raise ValueError("canonical cache is not the planned E5 cache")
    lineage_audit = validate_partitions(manifest, data, synthetic)
    device = _choose_device("auto")
    vae, basis, vae_meta = _load_model(VAE_PATH, device)
    training_started = time.monotonic()
    split_results = []
    selected_candidates = []
    for split in manifest["splits"]:
        seed = int(split["seed"])
        split_dir = RUN_DIR / f"split_{seed}"
        split_dir.mkdir(parents=True, exist_ok=True)
        baseline_checkpoint = split_dir / "baseline.pt"
        log_path = split_dir / "training.log"
        import contextlib
        with log_path.open("w", encoding="utf-8") as log, contextlib.redirect_stdout(log):
            baseline_training = train_text(
                data_cache=DATA_PATH,
                vae_checkpoint=VAE_PATH,
                checkpoint_path=baseline_checkpoint,
                epochs=EPOCH_CAP,
                batch_size=32,
                learning_rate=5e-4,
                weight_decay=1e-3,
                latent_weight=1.0,
                cosine_weight=0.1,
                reconstruction_weight=0.3,
                peak_weight=15.0,
                tv_weight=0.0,
                control_weight=0.0,
                seed=seed,
                model_seed=7,
                patience=15,
                scheduler_t0=20,
                scheduler_t_mult=2,
                minimum_learning_rate=1e-5,
                dropout=0.2,
                hidden_dims=(256, 128),
                selection_metric="latent_mse",
                split_manifest_path=MANIFEST_PATH,
                skip_test_eval=True,
            )
        if int(baseline_training["best_epoch"]) > EPOCH_CAP:
            raise ValueError(f"baseline exceeded epoch cap for seed {seed}")
        train_indices, validation_indices, test_indices = _split_from_material_groups(
            data["material_ids"], split["train_material_ids"],
            split["validation_material_ids"], split["test_material_ids"],
        )
        chosen, prototype_ids, prototype_embeddings, prototype_latents = select_retrieval(
            data, train_indices, validation_indices
        )
        retriever = SimilarityLatentRetriever(
            torch.from_numpy(prototype_embeddings), torch.from_numpy(prototype_latents),
            chosen["neighbors"], chosen["temperature"],
        )
        retrieval_checkpoint = split_dir / "retrieval.pt"
        torch.save({
            "predictor_type": "e5_similarity_retrieval",
            "model_state_dict": retriever.state_dict(),
            "prototype_embeddings": torch.from_numpy(prototype_embeddings),
            "prototype_latents": torch.from_numpy(prototype_latents),
            "prototype_material_ids": prototype_ids,
            "embedding_dim": int(prototype_embeddings.shape[1]),
            "latent_dim": int(prototype_latents.shape[1]),
            "profile_dim": 0,
            "control_dim": 0,
            "neighbors": chosen["neighbors"],
            "temperature": chosen["temperature"],
            "embedding_model": "intfloat/e5-large-v2",
            "vae_checkpoint": str(VAE_PATH),
            "vae_basis_type": vae_meta.get("basis_type", "unknown"),
            "data_cache": str(DATA_PATH),
            "split_manifest_path": str(MANIFEST_PATH),
            "split_material_groups": {key: list(map(str, split[f"{key}_material_ids"]))
                                      for key in ("train", "validation", "test")},
            "seed": seed,
            "selection_metric": "validation_latent_mse",
            "validation_latent_mse": chosen["validation_latent_mse"],
        }, retrieval_checkpoint)
        # Validation has selected both checkpoints/settings. Only now score the reserved test rows.
        baseline, baseline_meta = _load_predictor(baseline_checkpoint, device)
        with torch.inference_mode():
            baseline_test_latents = baseline(torch.as_tensor(data["embeddings"][test_indices], device=device))
        baseline_test = evaluate_rows(data, test_indices, baseline_test_latents.cpu().numpy(), vae, basis, device)
        retrieval_test_latents = predict_retrieval(
            data["embeddings"][test_indices], prototype_embeddings, prototype_latents,
            chosen["neighbors"], chosen["temperature"],
        )
        retrieval_test = evaluate_rows(data, test_indices, retrieval_test_latents, vae, basis, device)
        candidate = {"seed": seed, "checkpoint": retrieval_checkpoint, "validation": chosen,
                     "prototype_material_ids": prototype_ids, "train_indices": train_indices}
        selected_candidates.append(candidate)
        split_results.append({
            "seed": seed,
            "partition_counts": {"train_materials": len(split["train_material_ids"]),
                                 "validation_materials": len(split["validation_material_ids"]),
                                 "test_materials": len(split["test_material_ids"])},
            "test_material_ids": sorted(map(str, split["test_material_ids"])),
            "baseline": {"checkpoint": str(baseline_checkpoint), "best_epoch": int(baseline_meta["epoch"]),
                         "validation_latent_mse": float(baseline_meta["validation_latent_mse"]),
                         "heldout": baseline_test},
            "retrieval": {"checkpoint": str(retrieval_checkpoint), "neighbors": chosen["neighbors"],
                          "temperature": chosen["temperature"],
                          "validation_latent_mse": chosen["validation_latent_mse"],
                          "heldout": retrieval_test},
        })
        if time.monotonic() - training_started > TRAINING_WALL_CAP_SECONDS:
            raise TimeoutError("bounded training wall-time cap exceeded")
        print(f"Completed split {seed}: baseline epoch {baseline_meta['epoch']}, retrieval val MSE {chosen['validation_latent_mse']:.6f}", flush=True)

    selected = min(selected_candidates, key=lambda item: (item["validation"]["validation_latent_mse"], item["seed"]))
    selected_checkpoint = selected["checkpoint"]
    selected_prompt_check = prompt_check(
        selected_checkpoint, vae, basis, device,
        RUN_DIR / f"split_{selected['seed']}" / "selected_candidate_prompts",
    )
    reference_prompt_check = prompt_check(
        ROOT / "checkpoints/best_text_model.pt", vae, basis, device,
        RUN_DIR / "existing_predictor_reference_prompts",
    )
    # Exercise the public CLI with the selected retrieval checkpoint on a repeat prompt.
    cli_checks = []
    for suffix in ("a", "b"):
        output = RUN_DIR / f"cli_repeat_{suffix}.png"
        process = subprocess.run(
            [sys.executable, "-m", "prism.infer_text", "--prompt", PROMPTS[0],
             "--output", str(output), "--vae-checkpoint", str(VAE_PATH),
             "--text-checkpoint", str(selected_checkpoint), "--device", str(device)],
            cwd=ROOT, capture_output=True, text=True, timeout=600,
        )
        cli_checks.append({"output": str(output), "returncode": process.returncode,
                           "file_created": output.is_file(), "stdout": process.stdout[-2000:],
                           "stderr": process.stderr[-2000:],
                           "output_sha256": sha256(output) if output.is_file() else None})
        if process.returncode != 0 or not output.is_file():
            raise RuntimeError(f"text inference CLI failed: {process.stderr}")

    training_wall_seconds = time.monotonic() - training_started
    baseline_mse = [row["baseline"]["heldout"]["latent_mse"] for row in split_results]
    retrieval_mse = [row["retrieval"]["heldout"]["latent_mse"] for row in split_results]
    report = {
        "goal": "text-to-latent functionality with frozen Residual V3 VAE",
        "status": "READY" if selected_prompt_check["all_functionality_gates_pass"] else "NOT_READY_BOUNDED_ATTEMPT_COMPLETE",
        "manifest": str(MANIFEST_PATH),
        "manifest_predeclared": bool(manifest["predeclared_before_future_training"]),
        "cache_sha256": hashes,
        "embedding_model": "intfloat/e5-large-v2 (frozen)",
        "vae_checkpoint": str(VAE_PATH),
        "device": str(device),
        "split_results": split_results,
        "synthetic_lineage_audit": lineage_audit,
        "training_wall_seconds_including_evaluation": training_wall_seconds,
        "training_wall_measurement_note": "Single successful orchestration attempt; a recovery attempt also repeated the deterministic fits after a checkpoint-loading failure.",
        "training_wall_gpu_hours_upper_bound": training_wall_seconds / 3600.0,
        "training_wall_cap_seconds": TRAINING_WALL_CAP_SECONDS,
        "heldout_latent_mse_summary": {
            "baseline_per_split": baseline_mse,
            "retrieval_per_split": retrieval_mse,
            "baseline_mean": float(np.mean(baseline_mse)),
            "retrieval_mean": float(np.mean(retrieval_mse)),
            "blocking_threshold": None,
        },
        "selected_candidate": {"seed_selected_by_validation_only": int(selected["seed"]),
                               "checkpoint": str(selected_checkpoint),
                               "validation_latent_mse": selected["validation"]["validation_latent_mse"]},
        "selected_candidate_prompt_check": selected_prompt_check,
        "existing_predictor_reference_prompt_check": reference_prompt_check,
        "inference_cli_repeat_checks": cli_checks,
        "readiness_criteria": {
            "fresh_splits_complete": len(split_results) == 3,
            "no_cross_partition_synthetic_lineage": all(x["cross_partition_or_unassigned_rows"] == 0 for x in lineage_audit),
            "within_two_gpu_hour_wall_cap": training_wall_seconds <= TRAINING_WALL_CAP_SECONDS,
            "candidate_prompt_functionality": selected_prompt_check["all_functionality_gates_pass"],
            "latent_mse_below_0_10_is_not_a_gate": True,
        },
    }
    report["ready_for_use"] = all(report["readiness_criteria"].values())
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "ready_for_use": report["ready_for_use"],
                      "report": str(REPORT_PATH), "selected_candidate": report["selected_candidate"],
                      "prompt_gates": selected_prompt_check["gates"],
                      "heldout_latent_mse": report["heldout_latent_mse_summary"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
