"""Text embedding to frozen-VAE latent predictor."""

from __future__ import annotations

import torch
from torch import nn


class TextToLatent(nn.Module):
    """Map a precomputed text embedding to the VAE's latent space."""

    def __init__(self, embedding_dim: int, latent_dim: int) -> None:
        super().__init__()
        if embedding_dim < 1 or latent_dim < 1:
            raise ValueError("embedding_dim and latent_dim must be positive")
        self.embedding_dim = int(embedding_dim)
        self.latent_dim = int(latent_dim)
        self.network = nn.Sequential(
            nn.Linear(self.embedding_dim, 512),
            nn.SiLU(),
            nn.Linear(512, 256),
            nn.SiLU(),
            nn.Linear(256, self.latent_dim),
        )

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        if embedding.shape[-1] != self.embedding_dim:
            raise ValueError(
                f"expected text embeddings with width {self.embedding_dim}, "
                f"got {embedding.shape[-1]}"
            )
        return self.network(embedding.float())
