"""Polynomial decoder VAE for 64x64 NDF grids."""

from __future__ import annotations

import math

import torch
from torch import nn


DEFAULT_GRID_SIZE = 64
DEFAULT_POLYNOMIAL_DEGREE = 8
DEFAULT_POLYNOMIAL_TERMS = (DEFAULT_POLYNOMIAL_DEGREE + 1) * (DEFAULT_POLYNOMIAL_DEGREE + 2) // 2


def get_num_zernike_terms(max_degree: int) -> int:
    if max_degree < 0:
        raise ValueError("max_degree must be nonnegative")
    return (max_degree + 1) * (max_degree + 2) // 2


def create_zernike_basis(
    h_x: torch.Tensor, h_y: torch.Tensor, max_degree: int = 8
) -> torch.Tensor:
    """Return canonical real Zernike terms as [Num_Points, K]."""
    if max_degree < 0:
        raise ValueError("max_degree must be nonnegative")
    if h_x.shape != h_y.shape or h_x.ndim not in (1, 2):
        raise ValueError("h_x and h_y must have matching 1D or 2D shapes")
    r = torch.sqrt(h_x.square() + h_y.square())
    phi = torch.atan2(h_y, h_x)
    terms = []
    for n in range(max_degree + 1):
        for m in range(-n, n + 1, 2):
            absolute_m = abs(m)
            radial = torch.zeros_like(r)
            for k in range((n - absolute_m) // 2 + 1):
                coefficient = ((-1) ** k * math.factorial(n - k)) / (
                    math.factorial(k)
                    * math.factorial((n + absolute_m) // 2 - k)
                    * math.factorial((n - absolute_m) // 2 - k)
                )
                radial = radial + coefficient * r.pow(n - 2 * k)
            angular = torch.cos(absolute_m * phi) if m >= 0 else torch.sin(absolute_m * phi)
            term = radial * angular
            terms.append(torch.where(r <= 1.0, term, torch.zeros_like(term)).reshape(-1))
    return torch.stack(terms, dim=1)


def create_grid_zernike_basis(
    grid_size: int = DEFAULT_GRID_SIZE, max_degree: int = DEFAULT_POLYNOMIAL_DEGREE
) -> torch.Tensor:
    coords = torch.linspace(-1.0, 1.0, grid_size, dtype=torch.float32)
    hx, hy = torch.meshgrid(coords, coords, indexing="xy")
    return create_zernike_basis(hx, hy, max_degree).T.contiguous()


def create_polynomial_basis(
    grid_size: int = DEFAULT_GRID_SIZE,
    degree: int = DEFAULT_POLYNOMIAL_DEGREE,
) -> torch.Tensor:
    """Create the complete 2D monomial basis on the unit disk.

    Terms contain every ``hx**i * hy**j`` for which ``i + j <= degree``.
    This gives 15, 28, and 45 coefficients at degrees 4, 6, and 8. The basis
    has shape ``[K, grid_size * grid_size]`` and is zero outside the unit disk.
    Coefficients may be signed; the model applies ReLU to the evaluated grid
    to keep the predicted NDF nonnegative.
    """
    if grid_size < 2:
        raise ValueError("grid_size must be at least 2")
    if degree < 1:
        raise ValueError("degree must be at least 1")

    coords = torch.linspace(-1.0, 1.0, grid_size, dtype=torch.float32)
    hx, hy = torch.meshgrid(coords, coords, indexing="xy")
    mask = hx.square() + hy.square() <= 1.0
    terms = []
    for total_degree in range(degree + 1):
        for i in range(total_degree + 1):
            j = total_degree - i
            term = hx.pow(i) * hy.pow(j)
            terms.append(torch.where(mask, term, torch.zeros_like(term)).reshape(-1))
    return torch.stack(terms, dim=0)


def create_legacy_bernstein_basis(
    grid_size: int = DEFAULT_GRID_SIZE,
    degree: int = 4,
) -> torch.Tensor:
    """Build the older tensor-product Bernstein basis for checkpoint loading."""
    if grid_size < 2 or degree < 1:
        raise ValueError("grid_size must be at least 2 and degree at least 1")
    coords = torch.linspace(-1.0, 1.0, grid_size, dtype=torch.float32)
    hx, hy = torch.meshgrid(coords, coords, indexing="xy")
    mask = hx.square() + hy.square() <= 1.0
    u, v = (hx + 1.0) * 0.5, (hy + 1.0) * 0.5
    basis_u = [
        math.comb(degree, i) * u.pow(i) * (1.0 - u).pow(degree - i)
        for i in range(degree + 1)
    ]
    basis_v = [
        math.comb(degree, j) * v.pow(j) * (1.0 - v).pow(degree - j)
        for j in range(degree + 1)
    ]
    terms = [
        torch.where(mask, basis_u[i] * basis_v[j], torch.zeros_like(hx)).reshape(-1)
        for i in range(degree + 1)
        for j in range(degree + 1)
    ]
    return torch.stack(terms)


class _ResidualUNetDecoder(nn.Module):
    """Multi-scale residual corrector with a broad spatial receptive field."""

    def __init__(self, input_channels: int) -> None:
        super().__init__()
        self.enc0 = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, padding=1), nn.SiLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.SiLU(),
        )
        self.enc1 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.SiLU(),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.SiLU(),
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1), nn.SiLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.SiLU(),
        )
        self.dec1 = nn.Sequential(
            nn.Conv2d(192, 64, 3, padding=1), nn.SiLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.SiLU(),
        )
        self.dec0 = nn.Sequential(
            nn.Conv2d(96, 32, 3, padding=1), nn.SiLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.SiLU(),
        )
        self.output = nn.Conv2d(32, 1, 3, padding=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc0 = self.enc0(x)
        enc1 = self.enc1(enc0)
        center = self.bottleneck(self.enc2(enc1))
        up1 = torch.nn.functional.interpolate(
            center, size=enc1.shape[-2:], mode="bilinear", align_corners=False
        )
        dec1 = self.dec1(torch.cat((up1, enc1), dim=1))
        up0 = torch.nn.functional.interpolate(
            dec1, size=enc0.shape[-2:], mode="bilinear", align_corners=False
        )
        dec0 = self.dec0(torch.cat((up0, enc0), dim=1))
        return self.output(dec0)


class NDFPolynomialVAE(nn.Module):
    def __init__(
        self,
        input_grid_size: int = DEFAULT_GRID_SIZE,
        num_poly_terms: int = DEFAULT_POLYNOMIAL_TERMS,
        hidden_dims: tuple[int, int] = (512, 256),
        latent_dim: int = 8,
        positive_coefficients: bool = False,
        residual_decoder: bool = False,
        residual_decoder_version: int = 1,
    ) -> None:
        super().__init__()
        if input_grid_size < 2:
            raise ValueError("input_grid_size must be at least 2")
        if num_poly_terms < 1 or latent_dim < 1:
            raise ValueError("num_poly_terms and latent_dim must be positive")
        if len(hidden_dims) != 2 or min(hidden_dims) < 1:
            raise ValueError("hidden_dims must contain two positive widths")

        self.input_grid_size = input_grid_size
        self.input_dim = input_grid_size * input_grid_size
        self.num_poly_terms = num_poly_terms
        self.latent_dim = latent_dim
        self.positive_coefficients = positive_coefficients
        self.residual_decoder_enabled = residual_decoder
        if residual_decoder and residual_decoder_version not in (1, 2, 3):
            raise ValueError("residual_decoder_version must be 1, 2, or 3")
        self.residual_decoder_version = residual_decoder_version if residual_decoder else 0

        first_hidden, second_hidden = hidden_dims
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, first_hidden),
            nn.SiLU(),
            nn.Linear(first_hidden, second_hidden),
            nn.SiLU(),
        )
        self.fc_mu = nn.Linear(second_hidden, latent_dim)
        self.fc_logvar = nn.Linear(second_hidden, latent_dim)
        decoder_layers: list[nn.Module] = [
            nn.Linear(latent_dim, second_hidden),
            nn.SiLU(),
            nn.Linear(second_hidden, first_hidden),
            nn.SiLU(),
            nn.Linear(first_hidden, num_poly_terms),
        ]
        if positive_coefficients:
            decoder_layers.append(nn.Softplus())
        self.decoder = nn.Sequential(*decoder_layers)
        self.residual_decoder = None
        if residual_decoder:
            if residual_decoder_version == 1:
                # Keep the v1 module names and tensor shapes stable for checkpoints.
                self.residual_decoder = nn.Sequential(
                    nn.Conv2d(latent_dim + 1, 16, kernel_size=3, padding=1),
                    nn.SiLU(),
                    nn.Conv2d(16, 1, kernel_size=3, padding=1),
                )
            elif residual_decoder_version == 2:
                self.residual_decoder = nn.Sequential(
                    nn.Conv2d(latent_dim + 4, 32, kernel_size=3, padding=1),
                    nn.SiLU(),
                    nn.Conv2d(32, 64, kernel_size=3, padding=1),
                    nn.SiLU(),
                    nn.Conv2d(64, 64, kernel_size=3, padding=1),
                    nn.SiLU(),
                    nn.Conv2d(64, 1, kernel_size=3, padding=1),
                )
                # Zero residual at initialization preserves the coefficient decoder exactly.
                nn.init.zeros_(self.residual_decoder[-1].weight)
                nn.init.zeros_(self.residual_decoder[-1].bias)
            else:
                self.residual_decoder = _ResidualUNetDecoder(latent_dim + 4)
            if residual_decoder_version == 1:
                # Zero residual at initialization preserves the coefficient decoder exactly.
                nn.init.zeros_(self.residual_decoder[-1].weight)
                nn.init.zeros_(self.residual_decoder[-1].bias)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x.reshape(x.shape[0], -1)
        if x.shape[1] != self.input_dim:
            raise ValueError(
                f"expected {self.input_dim} grid values, got {x.shape[1]}"
            )
        hidden = self.encoder(x)
        mu = self.fc_mu(hidden).float()
        # Bound exp(0.5 * logvar) in the reparameterization and exp(logvar) in
        # the KL term, especially under mixed-precision training.
        logvar = self.fc_logvar(hidden).float().clamp(-10.0, 10.0)
        return mu, logvar

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode_ndf(self, latent: torch.Tensor, poly_basis: torch.Tensor) -> torch.Tensor:
        """Decode latent vectors to flattened, nonnegative NDF grids."""
        coefficients = self.decoder(latent)
        return self._decode_coefficients(latent, poly_basis, coefficients)

    def _decode_coefficients(
        self, latent: torch.Tensor, poly_basis: torch.Tensor, coefficients: torch.Tensor
    ) -> torch.Tensor:
        if poly_basis.shape not in (
            (self.num_poly_terms, self.input_dim),
            (self.input_dim, self.num_poly_terms),
        ):
            raise ValueError("poly_basis dimensions do not match this model")
        basis_kn = poly_basis if poly_basis.shape[0] == self.num_poly_terms else poly_basis.T
        base = torch.relu(coefficients @ basis_kn)
        if self.residual_decoder is None:
            return base
        basis_mask = (basis_kn.abs().sum(dim=0) > 0).to(base.dtype)
        base_grid = base.reshape(-1, 1, self.input_grid_size, self.input_grid_size)
        mask = basis_mask.reshape(1, 1, self.input_grid_size, self.input_grid_size)
        condition = latent.to(base.dtype)[:, :, None, None].expand(
            -1, -1, self.input_grid_size, self.input_grid_size
        )
        residual_inputs = [base_grid, condition]
        if self.residual_decoder_version in (2, 3):
            coords = torch.linspace(
                -1.0, 1.0, self.input_grid_size, device=base.device, dtype=base.dtype
            )
            y, x = torch.meshgrid(coords, coords, indexing="ij")
            radius = torch.sqrt(x.square() + y.square())
            spatial = torch.stack((x, y, radius), dim=0).unsqueeze(0).expand(
                base.shape[0], -1, -1, -1
            )
            residual_inputs.append(spatial)
        residual = self.residual_decoder(torch.cat(residual_inputs, dim=1))
        if self.residual_decoder_version == 3:
            # Bound the exploratory spatial correction to prevent it from
            # overwhelming the stable coefficient reconstruction.
            residual = 0.1 * torch.tanh(residual)
        return (torch.relu(base_grid + residual) * mask).reshape(-1, self.input_dim)

    def forward(
        self,
        x: torch.Tensor,
        poly_basis: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if poly_basis.shape not in (
            (self.num_poly_terms, self.input_dim),
            (self.input_dim, self.num_poly_terms),
        ):
            raise ValueError(
                "poly_basis must have shape "
                f"({self.num_poly_terms}, {self.input_dim}), got {tuple(poly_basis.shape)}"
            )
        mu, logvar = self.encode(x)
        latent = self.reparameterize(mu, logvar) if self.training else mu
        coeffs = self.decoder(latent)
        ndf_recon = self._decode_coefficients(latent, poly_basis, coeffs)
        return ndf_recon, coeffs, mu, logvar


def select_device() -> torch.device:
    """Select CUDA when available and explain the installed build otherwise."""
    if torch.cuda.is_available():
        return torch.device("cuda")

    print(
        "CUDA is unavailable to this PyTorch install "
        f"(torch={torch.__version__}, CUDA build={torch.version.cuda}, "
        f"CUDA built={torch.backends.cuda.is_built()}, "
        f"visible devices={torch.cuda.device_count()}). Falling back to CPU."
    )
    if torch.version.cuda is None:
        print(
            "This is a CPU-only PyTorch build. Install a CUDA-enabled PyTorch "
            "wheel that matches your Python version and NVIDIA driver."
        )
    return torch.device("cpu")
