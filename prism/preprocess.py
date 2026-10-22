"""Convert MERL and RGL BRDF files to normalized NDF disk grids."""

from __future__ import annotations

import argparse
import math
import struct
from pathlib import Path
from typing import BinaryIO

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


def normalize_ndf(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Clean, unit-peak normalize, mask, and return a float32 grid."""
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
    return clean.astype(np.float32)


def radial_profile_to_grid(
    profiles: np.ndarray,
    hx: np.ndarray,
    hy: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
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
    return normalize_ndf(values, mask)


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
) -> np.ndarray:
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
        return normalize_ndf(values, mask)
    return radial_profile_to_grid(ndf, hx, hy, mask)


def iter_rgl_files(directory: Path) -> list[Path]:
    """Prefer spectral files; use RGB only when no spectral file exists."""
    files = sorted(directory.glob("*_spec.bsdf"))
    if files:
        return files
    return sorted(directory.glob("*_rgb.bsdf"))


def preprocess(input_dir: Path, output_dir: Path, grid_size: int = 64) -> tuple[int, int]:
    hx, hy, mask = make_grid(grid_size)
    output_dir.mkdir(parents=True, exist_ok=True)

    merl_dir = input_dir / "merl"
    merl_files = sorted(merl_dir.glob("*.binary")) if merl_dir.is_dir() else []
    for path in merl_files:
        profile = merl_retroreflective_profile(path)
        theta_h_index = np.arange(MERL_DIMS[0], dtype=np.float64)
        theta_h = (theta_h_index**2 / (MERL_DIMS[0] ** 2)) * (math.pi / 2.0)
        sample_radius = np.sin(theta_h)
        values = np.interp(np.hypot(hx, hy), sample_radius, profile)
        result = normalize_ndf(values, mask)
        np.save(output_dir / f"merl_{path.stem}.npy", result, allow_pickle=False)

    rgl_dir = input_dir / "rgl"
    rgl_files = iter_rgl_files(rgl_dir) if rgl_dir.is_dir() else []
    for path in rgl_files:
        ndf = rgl_ndf_profiles(path)
        result = rgl_ndf_to_grid(ndf, hx, hy, mask)
        material_name = path.stem.removesuffix("_spec").removesuffix("_rgb")
        np.save(output_dir / f"rgl_{material_name}.npy", result, allow_pickle=False)

    if not merl_files and not rgl_files:
        raise FileNotFoundError(
            f"No MERL .binary or RGL .bsdf files found under {input_dir}"
        )
    return len(merl_files), len(rgl_files)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--grid-size", type=int, default=64, help="Output matrix width and height (default: 64)."
    )
    parser.add_argument(
        "--input-dir", type=Path, default=DEFAULT_INPUT,
        help="Directory containing merl/ and rgl/ raw data folders.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT,
        help="Directory for normalized .npy grids (default: dataset/grid/).",
    )
    args = parser.parse_args()
    try:
        merl_count, rgl_count = preprocess(args.input_dir, args.output_dir, args.grid_size)
    except Exception as exc:
        parser.exit(1, f"Preprocessing failed: {exc}\n")
    print(
        f"Saved {merl_count} MERL and {rgl_count} RGL grids "
        f"({args.grid_size}x{args.grid_size}, float32) to {args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
