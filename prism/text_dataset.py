"""Build a caption, text-embedding, and frozen-VAE latent training cache.

Run with ``python -m prism.text_dataset``. The resulting per-source NPZ files
contain ``captions``, ``embeddings`` [N,D], ``latents`` [N,L],
``grids`` [N,4096], ``material_ids``, ``material_names``, ``source_paths``,
and ``embedding_model``. Named materials receive six captions by default, with
an option for up to ten distinct variants;
synthetic augmented samples receive one grounded caption to keep that cache
compact. Caption rows for one source retain the same material ID and VAE target.

Text embeddings default to ``intfloat/e5-large-v2`` (the ``query: `` prefix,
masked mean pooling, and L2 normalization). MPNet/Roberta mean pooling, BGE
CLS pooling, and projected OpenAI CLIP text features are also supported.
Install ``transformers`` before running; models are downloaded/cached by
Hugging Face on first use.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from prism.model import NDFPolynomialVAE


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = REPO_ROOT / "checkpoints" / "zernike_residual_v3_warm_plateau42" / "best_model.pt"
DEFAULT_GRID_DIR = REPO_ROOT / "dataset" / "grid"
DEFAULT_AUGMENTED_DIR = REPO_ROOT / "dataset" / "augmented"
RGL_METADATA_PATH = Path(__file__).with_name("rgl_material_metadata.json")
RGL_METADATA = json.loads(RGL_METADATA_PATH.read_text(encoding="utf-8"))["records"]
DTU_METADATA_PATH = Path(__file__).with_name("dtu_nielsen_material_metadata.json")
DTU_METADATA = json.loads(DTU_METADATA_PATH.read_text(encoding="utf-8"))["records"]
NGAN_METADATA_PATH = Path(__file__).with_name("ngan_anisotropic_material_metadata.json")
NGAN_METADATA = json.loads(NGAN_METADATA_PATH.read_text(encoding="utf-8"))["records"]
URBAN_METADATA_PATH = Path(__file__).with_name("urban_brdf_material_metadata.json")
URBAN_METADATA = json.loads(URBAN_METADATA_PATH.read_text(encoding="utf-8"))["records"]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "checkpoints"
DEFAULT_EMBEDDING_MODEL = "intfloat/e5-large-v2"
GRID_SIZE = 64
GRID_CAPTIONS_PER_SAMPLE = 6
MAX_GRID_CAPTIONS_PER_SAMPLE = 10



def _material_traits(name: str) -> list[str]:
    """Expand recognizable filename tokens without treating them as labels."""
    words = re.sub(r"[^a-z0-9]+", " ", name.lower()).split()
    traits: list[str] = []
    categories = {
        "metal": {"metal", "metallic", "steel", "ss440", "nickel", "tungsten", "alu", "alum", "aluminium", "aluminum", "bronze", "brass", "copper", "chrome", "iron", "silver", "gold"},
        "plastic": {"plastic", "acrylic", "phenolic", "polymer", "rubber", "vinyl", "pvc", "teflon", "delrin", "latex", "polyethylene", "polyurethane", "spectralon"},
        "fabric": {"fabric", "felt", "cloth", "velvet", "silk", "wool", "cotton", "denim"},
        "wood": {"wood", "fruitwood", "maple", "oak", "walnut", "cherry", "pine", "mahogany"},
        "stone": {"stone", "granite", "limestone", "marble", "obsidian", "jasper", "hematite"},
        "concrete": {"concrete"},
        "asphalt": {"asphalt"},
        "pavement": {"pavement"},
        "silica": {"silica"},
        "paper": {"paper"},
        "ceramic": {"ceramic", "alumina", "nitride", "nitrade"},
    }
    joined = set(words)
    numbered_fabric = any(re.fullmatch(r"fabric\d+", word) for word in words)
    if numbered_fabric:
        joined.add("fabric")
    for category, vocabulary in categories.items():
        if joined & vocabulary:
            traits.append(category)
    if joined & {"rough", "roughness", "matte", "diffuse", "rubber", "fabric", "felt", "cloth"} or numbered_fabric:
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


def _primary_material_class(name: str, traits: list[str]) -> str | None:
    """Choose a grounded material class, preferring explicit textile terms."""
    words = re.sub(r"[^a-z0-9]+", " ", name.lower()).split()
    if any(word in {"felt", "fabric", "cloth", "velvet", "silk", "wool", "cotton", "denim"}
           or re.fullmatch(r"fabric\d+", word) for word in words):
        return "fabric"
    return next((kind for kind in (
        "metal", "plastic", "wood", "stone", "concrete", "asphalt", "pavement", "silica", "paper", "ceramic"
    ) if kind in traits), None)


def _label_contains_class(name: str, class_phrase: str) -> bool:
    """Check whether a class word is already explicit in a material label."""
    words = re.sub(r"[^a-z0-9]+", " ", name.lower()).split()
    return class_phrase in words or (
        class_phrase == "fabric"
        and any(re.fullmatch(r"fabric\d+", word) for word in words)
    )


def _grid_descriptor_stats(
    grid: np.ndarray,
) -> tuple[str, str, str, float, float, float, str]:
    """Return categorical and continuous physical descriptors from an NDF."""
    values = np.asarray(grid, dtype=np.float64)
    peak = float(values.max(initial=0.0))
    if peak <= 0.0:
        return "near-zero intensity", "approximately isotropic", "low peak concentration", 0.0, 0.0, 0.0, "unknown"
    coords = np.linspace(-1.0, 1.0, GRID_SIZE, dtype=np.float64)
    x, y = np.meshgrid(coords, coords, indexing="xy")
    disk = x * x + y * y <= 1.0
    weights = np.where(disk, np.maximum(values, 0.0), 0.0)
    total = float(weights.sum())
    if total <= 1e-12:
        return "near-zero intensity", "approximately isotropic", "low peak concentration", 0.0, 0.0, 0.0, "unknown"
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
    return width, direction, concentration, mean_radius, central, anisotropy, orientation if anisotropy >= 0.22 else "isotropic"


def _grid_descriptors(grid: np.ndarray) -> tuple[str, str, str]:
    """Return cautious peak-width, concentration, and anisotropy descriptors."""
    width, direction, concentration, *_ = _grid_descriptor_stats(grid)
    return width, direction, concentration


def _grid_profile_feature_vector(grid: np.ndarray) -> np.ndarray:
    """Return 24 normalized radial/angular NDF profile measurements."""
    values = np.asarray(grid, dtype=np.float64)
    coords = np.linspace(-1.0, 1.0, GRID_SIZE, dtype=np.float64)
    x, y = np.meshgrid(coords, coords, indexing="xy")
    radius = np.sqrt(x * x + y * y)
    disk = radius <= 1.0
    weights = np.where(disk, np.maximum(values, 0.0), 0.0)
    total = float(weights.sum())
    if total <= 1e-12:
        return np.zeros(24, dtype=np.float32)
    normalized = weights / total
    radial_edges = np.linspace(0.0, 1.0, 9)
    radial_bins = [
        float(normalized[(radius >= low) & (radius < high)].sum())
        for low, high in zip(radial_edges[:-1], radial_edges[1:])
    ]
    angle = np.arctan2(y, x)
    angle_edges = np.linspace(-np.pi, np.pi, 9)
    directions = ("SW", "S", "SE", "E", "NE", "N", "NW", "W")
    angle_bins = [
        float(normalized[(angle >= low) & (angle < high)].sum())
        for low, high in zip(angle_edges[:-1], angle_edges[1:])
    ]
    centroid_x = float((normalized * x).sum())
    centroid_y = float((normalized * y).sum())
    peak_index = np.unravel_index(int(np.argmax(weights)), weights.shape)
    peak_x, peak_y = float(x[peak_index]), float(y[peak_index])
    sorted_radius = radius.ravel()
    sorted_weights = weights.ravel()
    order = np.argsort(sorted_radius)
    cumulative = np.cumsum(sorted_weights[order]) / total
    quantiles = [
        float(sorted_radius[order[min(np.searchsorted(cumulative, q), len(order) - 1)]])
        for q in (0.25, 0.50, 0.75, 0.90)
    ]
    return np.asarray(
        radial_bins + angle_bins + [centroid_x, centroid_y, peak_x, peak_y] + quantiles,
        dtype=np.float32,
    )


def _grid_profile_statistics(grid: np.ndarray) -> str:
    """Describe a richer radial/angular NDF profile using measured values."""
    features = _grid_profile_feature_vector(grid)
    radial_text = " ".join(f"{value * 100:.0f}" for value in features[:8])
    angle_text = " ".join(f"{value * 100:.0f}" for value in features[8:16])
    quantile_text = " ".join(f"{value:.2f}" for value in features[20:24])
    return (
        f"R-bin energy % center-edge: {radial_text}. A-bin % SW,S,SE,E,NE,N,NW,W: {angle_text}. "
        f"Center xy: {features[16]:.2f} {features[17]:.2f}. "
        f"Peak xy: {features[18]:.2f} {features[19]:.2f}. Rq25,50,75,90: {quantile_text}."
    )


def parse_profile_features(caption: str) -> np.ndarray:
    """Parse the optional measured NDF profile side channel from a prompt."""
    patterns = (
        r"R-bin energy % center-edge:\s*([0-9.\s-]+)\.",
        r"A-bin % SW,S,SE,E,NE,N,NW,W:\s*([0-9.\s-]+)\.",
        r"Center xy:\s*([0-9.\s-]+)\.",
        r"Peak xy:\s*([0-9.\s-]+)\.",
        r"Rq25,50,75,90:\s*([0-9.\s-]+)\.",
    )
    parsed: list[list[float]] = []
    for pattern in patterns:
        match = re.search(pattern, caption)
        if match is None:
            return np.zeros(25, dtype=np.float32)
        try:
            parsed.append([float(value) for value in match.group(1).split()])
        except ValueError:
            return np.zeros(25, dtype=np.float32)
    if [len(values) for values in parsed] != [8, 8, 2, 2, 4]:
        return np.zeros(25, dtype=np.float32)
    profile = np.asarray(
        [value / 100.0 for value in parsed[0] + parsed[1]]
        + parsed[2] + parsed[3] + parsed[4],
        dtype=np.float32,
    )
    return np.concatenate((profile, np.ones(1, dtype=np.float32)))


def parse_prompt_controls(prompt: str) -> np.ndarray:
    """Parse optional roughness, anisotropy, and axial orientation controls.

    Roughness is the normalized mean radius of the decoded NDF (not a renderer
    microfacet-alpha value); anisotropy is the second-moment ratio; orientation
    is the principal-axis angle in degrees. The seven returned values contain
    roughness, anisotropy, sin(2*angle),
    cos(2*angle), then one presence mask per control. Roughness and
    anisotropy are normalized to [0, 1]; orientation is expressed in degrees
    modulo 180 because an NDF principal axis is unchanged by a half turn.
    """
    number = r"([-+]?(?:\d+(?:\.\d*)?|\.\d+))"
    values = np.zeros(7, dtype=np.float32)
    patterns = (
        ("roughness", rf"\broughness\s*(?:=|:)\s*{number}\s*(%|percent)?(?=\s|[,;.]|$)", 0),
        ("anisotropy", rf"\banisotropy\s*(?:=|:)\s*{number}\s*(%|percent)?(?=\s|[,;.]|$)", 1),
    )
    for name, pattern, index in patterns:
        match = re.search(pattern, prompt, flags=re.IGNORECASE)
        if match is None:
            continue
        value = float(match.group(1))
        if match.group(2):
            value /= 100.0
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1 (or 0% and 100%)")
        values[index] = value
        values[4 + index] = 1.0
    angle_match = re.search(
        rf"\b(?:orientation|angle)\s*(?:=|:)\s*{number}\s*(?:degrees?|deg|°)?(?=\s|[,;.]|$)",
        prompt,
        flags=re.IGNORECASE,
    )
    if angle_match is not None:
        angle = float(angle_match.group(1))
        if not np.isfinite(angle):
            raise ValueError("orientation must be a finite angle in degrees")
        radians = np.deg2rad(angle % 180.0)
        values[2] = np.sin(2.0 * radians)
        values[3] = np.cos(2.0 * radians)
        values[6] = 1.0
    return values


def _material_label(name: str) -> str:
    """Turn a dataset filename into a readable material label."""
    label = re.sub(r"^(?:merl|rgl|dtu|ngan|urban)_", "", name.lower())
    label = re.sub(r"[_-]+", " ", label).strip()
    label = re.sub(r"\b(?:alu|alum|aluminium)\b", "aluminum", label)
    # RGL uses ``aniso`` in names; spell it out for text encoders and prompts.
    return re.sub(r"\baniso\b", "anisotropic", label)


def _catalog_description(name: str) -> str | None:
    """Return verified wording for a measured source material."""
    key = Path(name).stem.lower()
    is_dtu = key.startswith("dtu_")
    is_ngan = key.startswith("ngan_")
    is_urban = key.startswith("urban_")
    key = re.sub(r"^(?:rgl|dtu|ngan|urban)_", "", key)
    records = (
        URBAN_METADATA if is_urban else NGAN_METADATA if is_ngan
        else DTU_METADATA if is_dtu else RGL_METADATA
    )
    record = records.get(key)
    if not record:
        return None
    description = str(record.get("description", "")).strip()
    tags = ", ".join(str(tag) for tag in record.get("tags", []) if str(tag).strip())
    if tags:
        description = f"{description}; catalogue class: {tags}"
    return description or None


def _catalog_label(name: str) -> str | None:
    """Return a concise verified material name when the source catalog has one."""
    key = Path(name).stem.lower()
    if not key.startswith("urban_"):
        return None
    record = URBAN_METADATA.get(re.sub(r"^urban_", "", key))
    if not record:
        return None
    label = str(record.get("display_name", "")).strip()
    return label or None


def captions_for_material(
    name: str,
    grid: np.ndarray,
    source: str,
    include_numeric_stats: bool = False,
    source_materials: list[str] | None = None,
    caption_count: int = GRID_CAPTIONS_PER_SAMPLE,
    include_profile_stats: bool = False,
    include_catalog_metadata: bool = False,
) -> list[str]:
    """Create grounded caption variants, without inventing augmented labels."""
    if not 5 <= caption_count <= MAX_GRID_CAPTIONS_PER_SAMPLE:
        raise ValueError(
            f"caption_count must be between 5 and {MAX_GRID_CAPTIONS_PER_SAMPLE}"
        )
    if include_profile_stats and caption_count != MAX_GRID_CAPTIONS_PER_SAMPLE:
        raise ValueError("include_profile_stats requires caption_count=10")
    width, direction, concentration, mean_radius, central, anisotropy, orientation = (
        _grid_descriptor_stats(grid)
    )
    ndf = f"{width}, {concentration}, and {direction} NDF distribution"
    measured = (
        f"Measured NDF statistics: mean radial extent {mean_radius:.3f}, central mass fraction "
        f"{central:.3f}, anisotropy ratio {anisotropy:.3f}, orientation {orientation}."
    )
    measured_suffix = f". {measured}" if include_numeric_stats else ""
    profile_stats = _grid_profile_statistics(grid) if include_profile_stats else ""
    if source == "augmented":
        # When lineage is present, name the source materials; otherwise keep
        # the description limited to observable grid traits.
        labels: list[str] = []
        for source_name in source_materials or []:
            label = re.sub(r"^(?:merl|rgl|dtu|ngan|urban)_", "", Path(source_name).stem.lower())
            label = re.sub(r"[_-]+", " ", label).strip()
            label = re.sub(r"\b(?:alu|alum|aluminium)\b", "aluminum", label)
            traits = _material_traits(label)
            kind = _primary_material_class(label, traits)
            if kind and not _label_contains_class(label, kind):
                label = f"{label} {kind}"
            if label:
                labels.append(label)
        subject = (
            "blend of " + " and ".join(labels)
            if labels
            else "synthetic material blend"
        )
        caption = f"{subject} with a {ndf}{measured_suffix} {profile_stats}".strip()
        if include_catalog_metadata:
            source_notes = [
                note for note in (_catalog_description(source_name) for source_name in source_materials or [])
                if note
            ]
            if source_notes:
                caption += ". Source material catalogue notes: " + "; ".join(source_notes) + "."
        return [caption]
    label = _material_label(name)
    catalog_label = _catalog_label(name) if include_catalog_metadata else None
    if catalog_label:
        label = catalog_label.lower()
    legacy_label = re.sub(r"^(?:merl|rgl|dtu|ngan|urban)_", "", name.lower())
    legacy_label = re.sub(r"[_-]+", " ", legacy_label).strip()
    legacy_label = re.sub(r"\b(?:alu|alum|aluminium)\b", "aluminum", legacy_label)
    if catalog_label:
        legacy_label = catalog_label.lower()
    traits = _material_traits(legacy_label)
    subject = f"{label} material" if label else "material"
    phrases: list[str] = []
    material_class = _primary_material_class(legacy_label, traits)
    class_phrase = {
        "metal": "metallic", "plastic": "plastic", "fabric": "fabric",
        "wood": "wood", "stone": "stone", "concrete": "concrete",
        "asphalt": "asphalt", "pavement": "pavement", "silica": "silica",
        "paper": "paper", "ceramic": "ceramic",
    }.get(material_class)
    if class_phrase and not _label_contains_class(label, class_phrase):
        phrases.append(class_phrase)
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
    # Keep the original six prompts byte-for-byte reproducible. The richer
    # variants use a normalized, nonduplicated material label below.
    legacy_subject = f"{legacy_label} material" if legacy_label else "material"
    legacy_phrases: list[str] = []
    legacy_traits = _material_traits(legacy_label)
    legacy_class = _primary_material_class(legacy_label, legacy_traits)
    legacy_class_phrase = {
        "metal": "metallic", "plastic": "plastic", "fabric": "fabric",
        "wood": "wood", "stone": "stone", "concrete": "concrete",
        "asphalt": "asphalt", "pavement": "pavement", "silica": "silica",
        "paper": "paper", "ceramic": "ceramic",
    }.get(legacy_class)
    if legacy_class_phrase and not _label_contains_class(legacy_label, legacy_class_phrase):
        legacy_phrases.append(legacy_class_phrase)
    if "paint" in legacy_traits:
        legacy_phrases.append("paint finish")
    if "oxidized" in legacy_traits:
        legacy_phrases.append("oxidized surface")
    if "rough" in legacy_traits:
        legacy_phrases.append("rough")
    elif "smooth" in legacy_traits:
        legacy_phrases.append("smooth glossy")
    if "anisotropic" in legacy_traits:
        legacy_phrases.append("brushed anisotropic")
    legacy_descriptor = " ".join(dict.fromkeys(legacy_phrases))
    if legacy_descriptor:
        legacy_subject = (
            f"{legacy_label} {legacy_descriptor} material"
            if legacy_label else f"{legacy_descriptor} material"
        )
    # Physical wording comes from observed NDF statistics. Material adjectives
    # come only from explicit filename tokens, never guessed from the grid.
    lobe_shape = {
        "narrow low-roughness highlight": "tight, sharp specular lobe",
        "medium-width roughness profile": "moderately spread specular lobe",
        "broad high-roughness highlight": "broad, soft specular lobe",
    }[width]
    highlight_strength = {
        "high central-peak concentration": "a strong central highlight",
        "moderate central-peak concentration": "a moderately concentrated highlight",
        "low central-peak concentration": "a broadly distributed highlight",
    }[concentration]
    shape_description = (
        "approximately isotropic"
        if orientation == "isotropic"
        else f"elongated, with a {orientation} orientation"
    )
    # Captions vary vocabulary and sentence form while preserving the same
    # measured optical facts. This gives prompt paraphrases without making up
    # material properties absent from the source name or NDF.
    variants = [
        f"{legacy_subject} with a {ndf}{measured_suffix}",
        f"A {legacy_subject} showing {ndf}{measured_suffix}",
        f"The measured specular response of {legacy_subject} has {ndf}{measured_suffix}",
        f"Describe {legacy_subject}: the surface produces {ndf}{measured_suffix}",
        f"What does {legacy_subject} look like? Its NDF shows {ndf}{measured_suffix}",
        f"{legacy_label or 'Material'} with {ndf}; a natural-language description of its surface response{measured_suffix}",
        f"For PBR rendering, describe {subject} as having a {ndf}{measured_suffix}",
        f"Light reflecting from {subject} forms a {lobe_shape}, with {highlight_strength}; its response is {shape_description}{measured_suffix}",
        f"How would you describe the light response of {subject}? It has a {lobe_shape}; its distribution is {shape_description}{measured_suffix}",
        f"PBR surface notes for {subject}: {highlight_strength}, a {lobe_shape}, and a response that is {shape_description}{measured_suffix}",
    ]
    if include_profile_stats:
        variants[6:] = [
            f"NDF profile: {profile_stats}",
            f"PBR lobe measurements for {label or 'material'}: {profile_stats}",
            f"Describe the light-response map of {label or 'this surface'}: {profile_stats}",
            f"Measured radial and angular response of {label or 'the material'}: {profile_stats}",
        ]
    selected = variants[:caption_count]
    if include_catalog_metadata:
        catalog_note = _catalog_description(name)
        if catalog_note:
            selected = [f"{caption}. Verified material catalogue description: {catalog_note}." for caption in selected]
    return selected


def caption_for_material(name: str, grid: np.ndarray, source: str) -> str:
    """Backward-compatible first caption for a material sample."""
    return captions_for_material(name, grid, source)[0]


def _material_id(name: str, source: str) -> str:
    if source == "augmented":
        # Augmentation currently discards its randomly blended input indices.
        # Keep each synthetic sample independent rather than invent lineage.
        return "aug:" + name.lower()
    if name.lower().startswith(("dtu_", "ngan_", "urban_")):
        # External examples are distinct samples even when their descriptive
        # labels overlap a MERL material (for example, the generic label "cardboard").
        source_prefix = (
            "urban" if name.lower().startswith("urban_")
            else "ngan" if name.lower().startswith("ngan_") else "dtu"
        )
        normalized = re.sub(r"^(?:dtu|ngan|urban)_", "", name.lower())
        return f"material:{source_prefix}:" + re.sub(r"[^a-z0-9]+", "-", normalized).strip("-")
    normalized = re.sub(r"^(?:merl|rgl|dtu|ngan|urban)_", "", name.lower())
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
        residual_decoder=bool(checkpoint.get("residual_decoder", False)),
        residual_decoder_version=int(
            checkpoint.get(
                "residual_decoder_version",
                1 if checkpoint.get("residual_decoder", False) else 1,
            )
        ),
    )
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def encode_texts(
    captions: list[str],
    model_name: str,
    batch_size: int,
    device: torch.device,
    local_files_only: bool = False,
) -> torch.Tensor:
    """Compute normalized sentence-transformer, BGE, or projected CLIP embeddings."""
    try:
        from transformers import AutoModel, AutoTokenizer, CLIPTextModelWithProjection
    except ImportError as exc:
        raise RuntimeError(
            "Text embedding requires Transformers. Install it with "
            "`python -m pip install transformers` (and ensure Hugging Face model access), "
            "then rerun this command."
        ) from exc
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
        if model_name.startswith("openai/clip-vit-"):
            encoder = CLIPTextModelWithProjection.from_pretrained(
                model_name, local_files_only=local_files_only
            )
        else:
            encoder = AutoModel.from_pretrained(model_name, local_files_only=local_files_only)
        encoder = encoder.to(device).eval()
    except Exception as exc:
        raise RuntimeError(
            f"Could not load embedding model {model_name!r}. Check network/cache access "
            "and the model identifier."
        ) from exc
    all_embeddings: list[torch.Tensor] = []
    is_clip = model_name.startswith("openai/clip-vit-")
    is_bge = model_name.startswith("BAAI/bge-")
    is_e5 = model_name.startswith("intfloat/e5-")
    model_max_length = int(getattr(encoder.config, "max_position_embeddings", 128))
    max_length = min(128, model_max_length)
    with torch.inference_mode():
        for start in range(0, len(captions), batch_size):
            text_batch = captions[start : start + batch_size]
            if is_e5:
                text_batch = [f"query: {caption}" for caption in text_batch]
            tokens = tokenizer(
                text_batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            outputs = encoder(**tokens)
            if is_clip:
                # CLIP's text projection maps the pooled EOS representation
                # into its contrastive space; mean-pooling hidden tokens would
                # discard the representation the model was trained to use.
                pooled = outputs.text_embeds
            elif is_bge:
                # BGE's Sentence-Transformers configuration specifies CLS pooling.
                pooled = outputs.last_hidden_state[:, 0]
            else:
                hidden = outputs.last_hidden_state
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
    include_numeric_stats: bool = False,
    caption_count: int = GRID_CAPTIONS_PER_SAMPLE,
    include_profile_stats: bool = False,
    include_catalog_metadata: bool = False,
) -> dict[str, Any]:
    """Create and save a stable, per-source NPZ artifact for text training."""
    if source not in ("grid", "augmented"):
        raise ValueError("build_text_dataset source must be 'grid' or 'augmented'")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if not 5 <= caption_count <= MAX_GRID_CAPTIONS_PER_SAMPLE:
        raise ValueError(
            f"caption_count must be between 5 and {MAX_GRID_CAPTIONS_PER_SAMPLE}"
        )
    if include_profile_stats and caption_count != MAX_GRID_CAPTIONS_PER_SAMPLE:
        raise ValueError("include_profile_stats requires caption_count=10")
    target_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    paths, sources = collect_samples(source, grid_dir, augmented_dir)
    grids_np = np.stack([_load_grid(path) for path in paths])
    material_names = [path.stem for path in paths]
    lineage_by_name: dict[str, list[str]] = {}
    lineage_path = augmented_dir / "lineage.json"
    if source == "augmented" and lineage_path.is_file():
        try:
            raw_lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"could not read augmented lineage manifest {lineage_path}: {exc}") from exc
        if not isinstance(raw_lineage, dict):
            raise ValueError(f"{lineage_path}: expected a JSON object mapping sample filenames to source materials")
        lineage_by_name = {
            str(sample_name): [str(material) for material in materials]
            for sample_name, materials in raw_lineage.items()
            if isinstance(materials, list) and materials
        }
        missing_lineage = [path.name for path, sample_source in zip(paths, sources)
                           if sample_source == "augmented" and path.name not in lineage_by_name]
        if missing_lineage:
            raise ValueError(
                f"{lineage_path}: missing source materials for {len(missing_lineage)} augmented samples"
            )
    captions_by_sample = [
        captions_for_material(
            name, grid, sample_source, include_numeric_stats,
            lineage_by_name.get(path.name),
            caption_count,
            include_profile_stats,
            include_catalog_metadata,
        )
        for name, grid, sample_source, path in zip(material_names, grids_np, sources, paths)
    ]
    expected_caption_counts = [
        caption_count if sample_source == "grid" else 1
        for sample_source in sources
    ]
    if any(
        len(captions) != expected
        for captions, expected in zip(captions_by_sample, expected_caption_counts)
    ):
        raise ValueError("caption generator returned an unexpected number of variants")
    captions = [caption for sample_captions in captions_by_sample for caption in sample_captions]

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
    row_sample_indices = np.repeat(np.arange(len(paths)), expected_caption_counts)
    artifact: dict[str, Any] = {
        "captions": np.asarray(captions, dtype=np.str_),
        "embeddings": embeddings.numpy().astype(np.float32, copy=False),
        "control_features": np.stack(
            [parse_prompt_controls(caption) for caption in captions]
        ).astype(np.float32, copy=False),
        "latents": latents.numpy()[row_sample_indices].astype(np.float32, copy=False),
        "grids": grids_np[row_sample_indices].reshape(len(captions), -1),
        "material_ids": np.asarray([
            "|".join(sorted({
                _material_id(Path(material).stem, "grid")
                for material in lineage_by_name[paths[index].name]
            }))
            if sources[index] == "augmented" and paths[index].name in lineage_by_name
            else _material_id(material_names[index], sources[index])
            for index in row_sample_indices
        ], dtype=np.str_),
        "material_names": np.asarray([material_names[index] for index in row_sample_indices], dtype=np.str_),
        "source_paths": np.asarray([str(paths[index].resolve()) for index in row_sample_indices], dtype=np.str_),
        "embedding_model": np.asarray(embedding_model, dtype=np.str_),
        "vae_checkpoint": np.asarray(str(checkpoint_path.resolve()), dtype=np.str_),
        "caption_policy": np.asarray(
            f"{caption_count} captions per named grid material; one observed-trait caption per augmented sample"
            + ("; continuous numeric NDF statistics included" if include_numeric_stats else "")
            + ("; detailed radial/angular NDF profile included" if include_profile_stats else "")
            + ("; verified RGL, DTU/Nielsen, Ngan, and urban material metadata included" if include_catalog_metadata else ""),
            dtype=np.str_,
        ),
        "lineage_manifest": np.asarray(
            str(lineage_path.resolve()) if lineage_by_name else "unavailable",
            dtype=np.str_,
        ),
    }
    if include_profile_stats:
        profile_rows: list[np.ndarray] = []
        absent_profile = np.zeros(25, dtype=np.float32)
        for index, sample_source in enumerate(sources):
            measured_profile = np.concatenate((
                _grid_profile_feature_vector(grids_np[index]),
                np.ones(1, dtype=np.float32),
            ))
            count = expected_caption_counts[index]
            if sample_source == "grid":
                profile_rows.extend([absent_profile.copy()] * (count - 4))
                profile_rows.extend([measured_profile.copy()] * 4)
            else:
                profile_rows.append(measured_profile)
        artifact["profile_features"] = np.stack(profile_rows).astype(np.float32, copy=False)
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
    parser.add_argument(
        "--include-numeric-stats", action="store_true",
        help="append measured radius, central-mass, and anisotropy values to captions",
    )
    parser.add_argument(
        "--caption-count", type=int, choices=range(5, MAX_GRID_CAPTIONS_PER_SAMPLE + 1),
        default=GRID_CAPTIONS_PER_SAMPLE,
        help="caption variants per named material (default: 6)",
    )
    parser.add_argument(
        "--include-profile-stats", action="store_true",
        help="add four profile-stat captions; requires --caption-count 10",
    )
    parser.add_argument(
        "--include-catalog-metadata", action="store_true",
        help="append verified RGL, DTU/Nielsen, Ngan, or urban material descriptions and tags",
    )
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
                include_numeric_stats=args.include_numeric_stats,
                caption_count=args.caption_count,
                include_profile_stats=args.include_profile_stats,
                include_catalog_metadata=args.include_catalog_metadata,
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
