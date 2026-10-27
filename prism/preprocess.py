"""Convert measured MERL, RGL, DTU/Nielsen, Ngan, and urban BRDF data to NDF grids."""

from __future__ import annotations

import argparse
import json
import math
import struct
from pathlib import Path
from typing import BinaryIO
from zipfile import ZipFile

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "dataset" / "raw"
DEFAULT_OUTPUT = REPO_ROOT / "dataset" / "grid"

MERL_DIMS = (90, 90, 180)  # theta_h, theta_d, phi_d
MERL_CHANNEL_SCALE = np.array((1 / 1500, 1.15 / 1500, 1.66 / 1500))
MERL_LUMINANCE = np.array((0.2126, 0.7152, 0.0722))
MERL_CHANNELS = 3

_BSDF_DTYPES = {
    1: np.dtype("u1"),
    2: np.dtype("i1"),
    3: np.dtype("<u2"),
    4: np.dtype("<i2"),
    5: np.dtype("<u4"),
    6: np.dtype("<i4"),
    7: np.dtype("<u8"),
    8: np.dtype("<i8"),
    9: np.dtype("<f2"),
    10: np.dtype("<f4"),
    11: np.dtype("<f8"),
}


def make_grid(grid_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return disk coordinates and a boolean unit-disk mask (rows are y)."""
    if grid_size < 2:
        raise ValueError("grid_size must be at least 2")
    coords = np.linspace(-1.0, 1.0, grid_size, dtype=np.float64)
    hx, hy = np.meshgrid(coords, coords, indexing="xy")
    radius = np.hypot(hx, hy)
    return hx, hy, radius <= 1.0


def normalize_ndf(
    values: np.ndarray, mask: np.ndarray, return_peak: bool = False
) -> np.ndarray | tuple[np.ndarray, float]:
    """Clean, unit-peak normalize, and optionally retain the pre-normalized peak."""
    clean = np.nan_to_num(
        np.asarray(values, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0
    )
    clean = np.maximum(clean, 0.0)
    clean[~mask] = 0.0
    peak = float(clean.max(initial=0.0))
    if peak > 0.0 and math.isfinite(peak):
        clean /= peak
    else:
        clean.fill(0.0)
    clean[~mask] = 0.0
    grid = clean.astype(np.float32)
    return (grid, peak) if return_peak else grid


def radial_profile_to_grid(
    profiles: np.ndarray,
    hx: np.ndarray,
    hy: np.ndarray,
    mask: np.ndarray,
    return_peak: bool = False,
) -> np.ndarray | tuple[np.ndarray, float]:
    """Interpolate one/two orthogonal radial NDF cuts over the unit disk."""
    profiles = np.asarray(profiles, dtype=np.float64)
    if profiles.ndim == 1:
        profiles = profiles[np.newaxis, :]
    if profiles.ndim != 2 or profiles.shape[0] not in (1, 2) or profiles.shape[1] < 2:
        raise ValueError(f"expected one or two radial profiles, got shape {profiles.shape}")

    profiles = np.nan_to_num(profiles, nan=0.0, posinf=0.0, neginf=0.0)
    profiles = np.maximum(profiles, 0.0)
    sample_radius = np.linspace(0.0, 1.0, profiles.shape[1])
    radius = np.hypot(hx, hy)
    profile_x = np.interp(radius, sample_radius, profiles[0])
    if profiles.shape[0] == 2:
        profile_y = np.interp(radius, sample_radius, profiles[1])
        angle = np.arctan2(hy, hx)
        # Interpolate the two measured principal-axis cuts around the azimuth.
        values = profile_x * np.cos(angle) ** 2 + profile_y * np.sin(angle) ** 2
    else:
        values = profile_x
    return normalize_ndf(values, mask, return_peak=return_peak)


def merl_retroreflective_profile(path: Path) -> np.ndarray:
    """Read MERL and derive an isotropic NDF proxy from its theta_d=0 slice.

    MERL stores BRDF values, not a standalone NDF. On the theta_d=0 slice,
    convert the BRDF to a distribution-shaped profile with the microfacet
    geometry factor (4 cos(theta_h)^2). Fresnel and masking terms are not
    provided separately by MERL, so this is a normalized NDF proxy.
    """
    raw = path.read_bytes()
    if len(raw) < 12:
        raise ValueError(f"{path}: truncated MERL header")
    dims = struct.unpack("<3i", raw[:12])
    if dims != MERL_DIMS:
        raise ValueError(f"{path}: expected MERL dimensions {MERL_DIMS}, found {dims}")

    value_count = math.prod(dims) * MERL_CHANNELS
    expected_size = 12 + value_count * np.dtype("<f8").itemsize
    if len(raw) != expected_size:
        raise ValueError(f"{path}: expected {expected_size} bytes, found {len(raw)}")

    data = np.frombuffer(raw, dtype="<f8", offset=12).reshape(
        MERL_CHANNELS, *dims
    )
    # phi_d is immaterial at theta_d=0; average the redundant samples.
    retro_slice = data[:, :, 0, :].mean(axis=-1).T
    rgb = retro_slice * MERL_CHANNEL_SCALE
    profile = np.sum(rgb * MERL_LUMINANCE, axis=-1)

    theta_h_index = np.arange(MERL_DIMS[0], dtype=np.float64)
    theta_h = (theta_h_index**2 / (MERL_DIMS[0] ** 2)) * (math.pi / 2.0)
    profile *= 4.0 * np.cos(theta_h) ** 2
    return np.nan_to_num(profile, nan=0.0, posinf=0.0, neginf=0.0)


def read_bsdf_fields(path: Path) -> dict[str, np.ndarray]:
    """Read the public RGL tensor_file container used by .bsdf files."""
    file_size = path.stat().st_size
    with path.open("rb") as stream:
        if stream.read(12) != b"tensor_file\0":
            raise ValueError(f"{path}: unrecognized RGL .bsdf header")
        version = stream.read(2)
        if version != b"\x01\x00":
            raise ValueError(f"{path}: unsupported tensor format version {version!r}")
        count_bytes = stream.read(4)
        if len(count_bytes) != 4:
            raise ValueError(f"{path}: truncated tensor field count")
        field_count = struct.unpack("<I", count_bytes)[0]
        fields: dict[str, np.ndarray] = {}
        for _ in range(field_count):
            name_len = _read_uint(stream, "<H", path)
            name = _read_exact(stream, name_len, path).decode("utf-8")
            ndim = _read_uint(stream, "<H", path)
            dtype_code = _read_uint(stream, "<B", path)
            if dtype_code not in _BSDF_DTYPES:
                raise ValueError(f"{path}: unsupported dtype code {dtype_code} for {name}")
            offset = _read_uint(stream, "<Q", path)
            shape = tuple(_read_uint(stream, "<Q", path) for _ in range(ndim))
            dtype = _BSDF_DTYPES[dtype_code]
            count = math.prod(shape)
            byte_count = count * dtype.itemsize
            if offset < 18 or offset + byte_count > file_size:
                raise ValueError(f"{path}: invalid data range for tensor field {name}")
            fields[name] = np.memmap(path, dtype=dtype, mode="r", offset=offset, shape=shape)
    return fields


def _read_exact(stream: BinaryIO, size: int, path: Path) -> bytes:
    result = stream.read(size)
    if len(result) != size:
        raise ValueError(f"{path}: truncated tensor field table")
    return result


def _read_uint(stream: BinaryIO, fmt: str, path: Path) -> int:
    size = struct.calcsize(fmt)
    return struct.unpack(fmt, _read_exact(stream, size, path))[0]


def rgl_ndf_profiles(path: Path) -> np.ndarray:
    """Load RGL's tabulated NDF field from a .bsdf file."""
    fields = read_bsdf_fields(path)
    if "ndf" not in fields:
        raise ValueError(f"{path}: RGL file has no 'ndf' field")
    profiles = np.asarray(fields["ndf"])
    if profiles.ndim == 2 and profiles.shape[0] == 2 and profiles.shape[1] >= 2:
        return profiles
    if profiles.ndim == 2 and profiles.shape[0] > 2 and profiles.shape[1] > 2:
        return profiles
    if profiles.ndim == 1 and profiles.size >= 2:
        return profiles
    raise ValueError(f"{path}: unsupported RGL ndf shape {profiles.shape}")


def rgl_ndf_to_grid(
    ndf: np.ndarray,
    hx: np.ndarray,
    hy: np.ndarray,
    mask: np.ndarray,
    return_peak: bool = False,
) -> np.ndarray | tuple[np.ndarray, float]:
    """Map RGL radial cuts or a polar NDF table onto the Cartesian disk grid."""
    ndf = np.asarray(ndf, dtype=np.float64)
    if ndf.ndim == 2 and ndf.shape[0] > 2:
        # RGL anisotropic materials store NDF samples in (theta_h, phi_h).
        ndf = np.nan_to_num(ndf, nan=0.0, posinf=0.0, neginf=0.0)
        ndf = np.maximum(ndf, 0.0)
        radius = np.hypot(hx, hy)
        theta = np.arcsin(np.clip(radius, 0.0, 1.0))
        radial_pos = theta * (2.0 / math.pi) * (ndf.shape[0] - 1)
        radial0 = np.floor(radial_pos).astype(np.intp)
        radial1 = np.minimum(radial0 + 1, ndf.shape[0] - 1)
        radial_weight = radial_pos - radial0

        phi_pos = (np.arctan2(hy, hx) % (2.0 * math.pi)) * (ndf.shape[1] / (2.0 * math.pi))
        phi0 = np.floor(phi_pos).astype(np.intp) % ndf.shape[1]
        phi1 = (phi0 + 1) % ndf.shape[1]
        phi_weight = phi_pos - np.floor(phi_pos)

        row0 = ndf[radial0, phi0] * (1.0 - phi_weight) + ndf[radial0, phi1] * phi_weight
        row1 = ndf[radial1, phi0] * (1.0 - phi_weight) + ndf[radial1, phi1] * phi_weight
        values = row0 * (1.0 - radial_weight) + row1 * radial_weight
        return normalize_ndf(values, mask, return_peak=return_peak)
    return radial_profile_to_grid(ndf, hx, hy, mask, return_peak=return_peak)


def ngan_normal_view_ndf_proxy(
    path: Path, hx: np.ndarray, hy: np.ndarray, mask: np.ndarray, return_peak: bool = False
) -> np.ndarray | tuple[np.ndarray, float]:
    """Build an explicitly limited NDF proxy from an Ngan anisotropic BRDF.

    Uses the measured normal-view slice (theta_out=0, phi_diff=0), applies the
    microfacet denominator's 4*cos(theta_in) factor, and retains unknown
    Fresnel and masking terms in the proxy. This slice only covers theta_h up
    to 45 degrees, so the unmeasured outer disk is zero-filled.
    """
    with ZipFile(path) as archive:
        members = [name for name in archive.namelist() if name.lower().endswith(".dat")]
        if len(members) != 1:
            raise ValueError(f"{path}: expected one .dat BRDF member, found {members}")
        raw = archive.read(members[0])
    if len(raw) < 64:
        raise ValueError(f"{path}: truncated Ngan anisotropic header")
    dims = struct.unpack("<4I", raw[:16])
    modes = struct.unpack("<10i", raw[16:56])
    if modes[2] != 1 or modes[6] != 3:
        raise ValueError(f"{path}: expected standard RGB parameterization")
    expected = 64 + math.prod(dims) * 3 * np.dtype("<f4").itemsize
    if len(raw) != expected:
        raise ValueError(f"{path}: expected {expected} bytes, found {len(raw)}")

    data = np.frombuffer(raw, dtype="<f4", offset=64).reshape((3, *dims))
    rgb = np.moveaxis(np.asarray(data[:, :, 0, 0, :], dtype=np.float64), 0, -1)
    luminance = rgb @ np.array((0.2126, 0.7152, 0.0722), dtype=np.float64)
    luminance = np.where(np.isfinite(luminance) & (luminance > 0.0), luminance, 0.0)
    theta_in = np.arange(dims[0], dtype=np.float64) * (math.pi / 2.0) / dims[0]
    luminance *= 4.0 * np.cos(theta_in)[:, None]

    radius = np.hypot(hx, hy)
    phi = np.mod(np.arctan2(hy, hx), 2.0 * math.pi)
    measured_radius = math.sin(math.pi / 4.0)
    theta_query = 2.0 * np.arcsin(np.clip(radius, 0.0, measured_radius))
    theta_position = theta_query / (math.pi / 2.0) * dims[0]
    theta0 = np.floor(theta_position).astype(np.intp).clip(0, dims[0] - 1)
    theta1 = np.minimum(theta0 + 1, dims[0] - 1)
    theta_weight = theta_position - np.floor(theta_position)
    phi_position = phi / (2.0 * math.pi) * dims[3]
    phi0 = np.floor(phi_position).astype(np.intp) % dims[3]
    phi1 = (phi0 + 1) % dims[3]
    phi_weight = phi_position - np.floor(phi_position)

    row0 = luminance[theta0, phi0] * (1.0 - phi_weight) + luminance[theta0, phi1] * phi_weight
    row1 = luminance[theta1, phi0] * (1.0 - phi_weight) + luminance[theta1, phi1] * phi_weight
    values = row0 * (1.0 - theta_weight) + row1 * theta_weight
    values[radius > measured_radius] = 0.0
    return normalize_ndf(values, mask, return_peak=return_peak)


def urban_brdf_to_ndf_proxy(
    record: dict[str, object],
    hx: np.ndarray,
    hy: np.ndarray,
    mask: np.ndarray,
    return_peak: bool = False,
) -> tuple[np.ndarray, float] | tuple[np.ndarray, float, float]:
    """Map an urban BRDF sample table into a limited half-vector NDF proxy.

    The source records view elevation from the surface plane, the light angle
    from the normal, and their plane angle. Convert each pair to a half vector,
    multiply luminance BRDF by the microfacet cosine denominator, and smooth
    sparse measurements onto the unit disk. Unknown Fresnel, masking, and
    diffuse terms remain; azimuth mirroring assumes an isotropic surface.
    """
    try:
        alpha, beta, gamma, q = (
            np.asarray(record[name], dtype=np.float64)
            for name in ("alpha", "beta", "gamma", "q")
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("urban BRDF record must contain alpha, beta, gamma, and q arrays") from exc
    if not (alpha.ndim == beta.ndim == gamma.ndim == q.ndim == 1):
        raise ValueError("urban BRDF angle/value arrays must be one-dimensional")
    if len(alpha) < 16 or not (len(alpha) == len(beta) == len(gamma) == len(q)):
        raise ValueError("urban BRDF angle/value arrays must have matching lengths >= 16")

    theta_out = np.deg2rad(90.0 - alpha)
    theta_in = np.deg2rad(gamma)
    phi_in = np.deg2rad(180.0 + beta)
    view = np.stack(
        (np.sin(theta_out), np.zeros_like(theta_out), np.cos(theta_out)), axis=-1
    )
    light = np.stack(
        (
            np.sin(theta_in) * np.cos(phi_in),
            np.sin(theta_in) * np.sin(phi_in),
            np.cos(theta_in),
        ),
        axis=-1,
    )
    half = view + light
    half_length = np.linalg.norm(half, axis=-1)
    valid = (
        np.isfinite(alpha) & np.isfinite(beta) & np.isfinite(gamma) & np.isfinite(q)
        & (q >= 0.0) & (half_length > 1e-8)
        & (np.cos(theta_in) >= 0.0) & (np.cos(theta_out) >= 0.0)
    )
    half = half[valid] / half_length[valid, None]
    values = q[valid] * 4.0 * np.cos(theta_in[valid]) * np.cos(theta_out[valid])
    valid = half[:, 2] > 0.0
    half, values = half[valid], values[valid]
    if len(values) < 16 or float(values.max(initial=0.0)) <= 0.0:
        raise ValueError("urban BRDF record has too few positive upper-hemisphere samples")

    # beta spans a half turn between planes. Mirror the measurements across
    # the view plane to complete an isotropic azimuthal NDF proxy.
    sample_xy = half[:, :2]
    sample_xy = np.concatenate((sample_xy, sample_xy * np.array((1.0, -1.0))), axis=0)
    sample_values = np.concatenate((values, values), axis=0)
    centers = np.stack((hx[mask], hy[mask]), axis=1)
    accumulated = np.zeros(len(centers), dtype=np.float64)
    support = np.zeros(len(centers), dtype=np.float64)
    sigma = 0.025
    cutoff_squared = (3.0 * sigma) ** 2
    for start in range(0, len(centers), 128):
        center_batch = centers[start : start + 128]
        distance_squared = (
            (center_batch[:, None, :] - sample_xy[None, :, :]) ** 2
        ).sum(axis=-1)
        weights = np.exp(-0.5 * distance_squared / (sigma * sigma))
        weights[distance_squared > cutoff_squared] = 0.0
        stop = start + len(center_batch)
        accumulated[start:stop] = weights @ sample_values
        support[start:stop] = weights.sum(axis=1)

    supported = support > 0.05
    support_fraction = float(supported.sum() / len(supported))
    values_on_disk = np.divide(
        accumulated, support, out=np.zeros_like(accumulated), where=supported
    )
    grid = np.zeros_like(hx, dtype=np.float64)
    grid[mask] = values_on_disk
    normalized = normalize_ndf(grid, mask, return_peak=return_peak)
    if return_peak:
        normalized_grid, raw_peak = normalized
        return normalized_grid, support_fraction, raw_peak
    return normalized, support_fraction


def iter_rgl_files(directory: Path) -> list[Path]:
    """Prefer spectral files; use RGB only when no spectral file exists."""
    files = sorted(directory.glob("*_spec.bsdf"))
    if files:
        return files
    return sorted(directory.glob("*_rgb.bsdf"))


def preprocess(
    input_dir: Path, output_dir: Path, grid_size: int = 64
) -> tuple[int, int, int, int, int]:
    hx, hy, mask = make_grid(grid_size)
    output_dir.mkdir(parents=True, exist_ok=True)
    amplitude_scales: list[dict[str, object]] = []

    merl_dir = input_dir / "merl"
    merl_files = sorted(merl_dir.glob("*.binary")) if merl_dir.is_dir() else []
    for path in merl_files:
        profile = merl_retroreflective_profile(path)
        theta_h_index = np.arange(MERL_DIMS[0], dtype=np.float64)
        theta_h = (theta_h_index**2 / (MERL_DIMS[0] ** 2)) * (math.pi / 2.0)
        sample_radius = np.sin(theta_h)
        values = np.interp(np.hypot(hx, hy), sample_radius, profile)
        result, raw_peak = normalize_ndf(values, mask, return_peak=True)
        np.save(output_dir / f"merl_{path.stem}.npy", result, allow_pickle=False)
        amplitude_scales.append({"grid": f"merl_{path.stem}.npy", "source_family": "MERL", "raw_proxy_peak": raw_peak})

    dtu_dir = input_dir / "dtu_nielsen"
    if not dtu_dir.is_dir() and input_dir.name == "dtu_nielsen":
        dtu_dir = input_dir
    dtu_files = sorted(dtu_dir.glob("*.binary")) if dtu_dir.is_dir() else []
    for path in dtu_files:
        profile = merl_retroreflective_profile(path)
        theta_h_index = np.arange(MERL_DIMS[0], dtype=np.float64)
        theta_h = (theta_h_index**2 / (MERL_DIMS[0] ** 2)) * (math.pi / 2.0)
        values = np.interp(np.hypot(hx, hy), np.sin(theta_h), profile)
        result, raw_peak = normalize_ndf(values, mask, return_peak=True)
        np.save(output_dir / f"dtu_{path.stem}.npy", result, allow_pickle=False)
        amplitude_scales.append({"grid": f"dtu_{path.stem}.npy", "source_family": "DTU/Nielsen", "raw_proxy_peak": raw_peak})

    rgl_dir = input_dir / "rgl"
    rgl_files = iter_rgl_files(rgl_dir) if rgl_dir.is_dir() else []
    for path in rgl_files:
        ndf = rgl_ndf_profiles(path)
        result, raw_peak = rgl_ndf_to_grid(ndf, hx, hy, mask, return_peak=True)
        material_name = path.stem.removesuffix("_spec").removesuffix("_rgb")
        np.save(output_dir / f"rgl_{material_name}.npy", result, allow_pickle=False)
        amplitude_scales.append({"grid": f"rgl_{material_name}.npy", "source_family": "RGL", "raw_proxy_peak": raw_peak})

    ngan_dir = input_dir / "ngan_anisotropic"
    if not ngan_dir.is_dir() and input_dir.name == "ngan_anisotropic":
        ngan_dir = input_dir
    ngan_files = sorted(ngan_dir.glob("*.zip")) if ngan_dir.is_dir() else []
    for path in ngan_files:
        result, raw_peak = ngan_normal_view_ndf_proxy(path, hx, hy, mask, return_peak=True)
        np.save(output_dir / f"ngan_{path.stem}.npy", result, allow_pickle=False)
        amplitude_scales.append({"grid": f"ngan_{path.stem}.npy", "source_family": "Ngan", "raw_proxy_peak": raw_peak})

    urban_dir = input_dir / "urban_brdf"
    if not urban_dir.is_dir() and input_dir.name == "urban_brdf":
        urban_dir = input_dir
    urban_archives = sorted(urban_dir.glob("*.zip")) if urban_dir.is_dir() else []
    urban_count = 0
    urban_quality: list[dict[str, object]] = []
    for path in urban_archives:
        with ZipFile(path) as archive:
            member_names = sorted(
                name for name in archive.namelist()
                if name.lower().endswith(".json")
                and "r-table_" not in Path(name).name.lower()
            )
            for member in member_names:
                record = json.loads(archive.read(member))
                if not all(key in record for key in ("alpha", "beta", "gamma", "q")):
                    continue
                material_name = Path(member).stem.lower()
                try:
                    result, support_fraction, raw_peak = urban_brdf_to_ndf_proxy(
                        record, hx, hy, mask, return_peak=True
                    )
                except ValueError as exc:
                    urban_quality.append({"material": material_name, "skipped": str(exc)})
                    continue
                np.save(output_dir / f"urban_{material_name}.npy", result, allow_pickle=False)
                amplitude_scales.append({
                    "grid": f"urban_{material_name}.npy",
                    "source_family": "urban BRDF proxy",
                    "raw_proxy_peak": raw_peak,
                })
                urban_count += 1
                urban_quality.append({
                    "material": material_name,
                    "measurement_count": len(record["q"]),
                    "support_fraction": support_fraction,
                    "proxy": "BRDF times 4*cos(theta_i)*cos(theta_o), with isotropic azimuth mirroring",
                })
    if urban_archives:
        (output_dir / "urban_proxy_preprocess_metrics.json").write_text(
            json.dumps({"items": urban_quality}, indent=2) + "\n", encoding="utf-8"
        )

    if not merl_files and not rgl_files and not dtu_files and not ngan_files and not urban_count:
        raise FileNotFoundError(
            f"No supported MERL, RGL, DTU/Nielsen, Ngan, or urban BRDF files found under {input_dir}"
        )
    (output_dir / "source_amplitude_scales.json").write_text(
        json.dumps(
            {
                "normalization": "Each saved grid is unit-peak normalized.",
                "raw_proxy_peak_semantics": (
                    "Peak before per-grid normalization; proxy scales may be compared only within "
                    "a source family and are not validated specular-intensity labels."
                ),
                "cross_source_comparable": False,
                "valid_for_amplitude_head": False,
                "items": amplitude_scales,
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    return len(merl_files), len(rgl_files), len(dtu_files), len(ngan_files), urban_count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--grid-size", type=int, default=64, help="Output matrix width and height (default: 64)."
    )
    parser.add_argument(
        "--input-dir", type=Path, default=DEFAULT_INPUT,
        help="Directory containing supported raw BRDF dataset folders.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT,
        help="Directory for normalized .npy grids (default: dataset/grid/).",
    )
    args = parser.parse_args()
    try:
        merl_count, rgl_count, dtu_count, ngan_count, urban_count = preprocess(
            args.input_dir, args.output_dir, args.grid_size
        )
    except Exception as exc:
        parser.exit(1, f"Preprocessing failed: {exc}\n")
    print(
        f"Saved {merl_count} MERL, {rgl_count} RGL, {dtu_count} DTU/Nielsen, "
        f"{ngan_count} Ngan, and {urban_count} urban BRDF grids "
        f"({args.grid_size}x{args.grid_size}, float32) to {args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
