"""Polynomial decoder VAE for 64x64 NDF grids."""

from __future__ import annotations

import math

import torch
from torch import nn


DEFAULT_GRID_SIZE = 64
DEFAULT_POLYNOMIAL_DEGREE = 8
DEFAULT_POLYNOMIAL_TERMS = (DEFAULT_POLYNOMIAL_DEGREE + 1) * (DEFAULT_POLYNOMIAL_DEGREE + 2) // 2


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


class NDFPolynomialVAE(nn.Module):
    def __init__(
        self,
        input_grid_size: int = DEFAULT_GRID_SIZE,
        num_poly_terms: int = DEFAULT_POLYNOMIAL_TERMS,
        hidden_dims: tuple[int, int] = (512, 256),
        latent_dim: int = 8,
        positive_coefficients: bool = False,
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

    def forward(
        self,
        x: torch.Tensor,
        poly_basis: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if poly_basis.shape != (self.num_poly_terms, self.input_dim):
            raise ValueError(
                "poly_basis must have shape "
                f"({self.num_poly_terms}, {self.input_dim}), got {tuple(poly_basis.shape)}"
            )
        mu, logvar = self.encode(x)
        latent = self.reparameterize(mu, logvar) if self.training else mu
        coeffs = self.decoder(latent)
        # Basis layout is [K, N], so [B, K] @ [K, N] -> [B, N].
        ndf_recon = torch.matmul(coeffs, poly_basis)
        if not self.positive_coefficients:
            ndf_recon = torch.relu(ndf_recon)
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
