"""Text embedding to frozen-VAE latent predictor."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    """Pre-normalized residual MLP block with a regularized update."""

    def __init__(self, width: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.fc1 = nn.Linear(width, width)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(width, width)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        update = self.fc1(self.norm(value))
        update = self.activation(update)
        update = self.dropout(update)
        update = self.fc2(update)
        return self.activation(value + self.dropout(update))


class TextToLatent(nn.Module):
    """Map normalized text embeddings to the frozen VAE's latent space.

    Architecture version 1 is retained for loading the original prototype's
    checkpoints. New models use a LayerNorm input projection, a residual MLP
    block, and dropout before latent prediction.
    """

    def __init__(
        self,
        embedding_dim: int,
        latent_dim: int,
        hidden_dims: tuple[int, int] | list[int] = (512, 256),
        dropout: float = 0.15,
        architecture_version: int = 2,
        profile_dim: int = 0,
        profile_projection_dim: int = 0,
        control_dim: int = 0,
        control_projection_dim: int = 0,
    ) -> None:
        super().__init__()
        if embedding_dim < 1 or latent_dim < 1:
            raise ValueError("embedding_dim and latent_dim must be positive")
        if len(hidden_dims) != 2 or min(hidden_dims) < 1:
            raise ValueError("hidden_dims must contain two positive widths")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if architecture_version not in (1, 2):
            raise ValueError("architecture_version must be 1 or 2")
        if profile_dim < 0:
            raise ValueError("profile_dim must be nonnegative")
        if profile_projection_dim < 0 or (profile_projection_dim and profile_dim < 2):
            raise ValueError("profile_projection_dim requires profile_dim >= 2")
        if control_dim < 0 or control_projection_dim < 0:
            raise ValueError("control dimensions must be nonnegative")
        if control_projection_dim and control_dim < 1:
            raise ValueError("control_projection_dim requires control_dim >= 1")
        self.embedding_dim = int(embedding_dim)
        self.latent_dim = int(latent_dim)
        self.profile_dim = int(profile_dim)
        self.profile_projection_dim = int(profile_projection_dim)
        self.control_dim = int(control_dim)
        self.control_projection_dim = int(control_projection_dim)
        input_dim = self.embedding_dim + (
            self.profile_projection_dim if self.profile_projection_dim else self.profile_dim
        ) + (self.control_projection_dim if self.control_projection_dim else self.control_dim)
        self.profile_projection = (
            nn.Sequential(
                nn.Linear(self.profile_dim - 1, self.profile_projection_dim),
                nn.LayerNorm(self.profile_projection_dim),
                nn.SiLU(),
                nn.Linear(self.profile_projection_dim, self.profile_projection_dim),
                nn.LayerNorm(self.profile_projection_dim),
            )
            if self.profile_projection_dim else None
        )
        self.control_projection = (
            nn.Sequential(
                nn.Linear(self.control_dim, self.control_projection_dim),
                nn.LayerNorm(self.control_projection_dim),
                nn.SiLU(),
            )
            if self.control_projection_dim else None
        )
        self.hidden_dims = (int(hidden_dims[0]), int(hidden_dims[1]))
        self.dropout_probability = float(dropout)
        self.architecture_version = int(architecture_version)

        if architecture_version == 1:
            self.network = nn.Sequential(
                nn.Linear(input_dim, self.hidden_dims[0]),
                nn.SiLU(),
                nn.Linear(self.hidden_dims[0], self.hidden_dims[1]),
                nn.SiLU(),
                nn.Linear(self.hidden_dims[1], self.latent_dim),
            )
        else:
            width, bottleneck = self.hidden_dims
            self.network = nn.Sequential(
                nn.Linear(input_dim, width),
                nn.LayerNorm(width),
                nn.SiLU(),
                ResidualBlock(width, dropout),
                nn.Linear(width, bottleneck),
                nn.LayerNorm(bottleneck),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(bottleneck, self.latent_dim),
            )

    def forward(
        self,
        embedding: torch.Tensor,
        profile_features: torch.Tensor | None = None,
        control_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if embedding.shape[-1] != self.embedding_dim:
            raise ValueError(
                f"expected text embeddings with width {self.embedding_dim}, "
                f"got {embedding.shape[-1]}"
            )
        model_input = embedding.float()
        if self.profile_dim:
            if profile_features is None:
                profile_features = model_input.new_zeros((model_input.shape[0], self.profile_dim))
            if profile_features.shape != (model_input.shape[0], self.profile_dim):
                raise ValueError(
                    f"expected profile features with shape ({model_input.shape[0]}, {self.profile_dim}), "
                    f"got {tuple(profile_features.shape)}"
                )
            profile_features = profile_features.to(model_input).float()
            if self.profile_projection is not None:
                available = profile_features[:, -1:].clamp(0.0, 1.0)
                projected = self.profile_projection(profile_features[:, :-1])
                projected = projected * available / (self.profile_projection_dim**0.5)
                model_input = torch.cat((model_input, projected), dim=-1)
            else:
                model_input = torch.cat((model_input, profile_features), dim=-1)
        elif profile_features is not None and profile_features.shape[-1] != 0:
            raise ValueError("profile features were provided to a model with profile_dim=0")
        if self.control_dim:
            if control_features is None:
                control_features = model_input.new_zeros((model_input.shape[0], self.control_dim))
            if control_features.shape != (model_input.shape[0], self.control_dim):
                raise ValueError(
                    f"expected control features with shape ({model_input.shape[0]}, {self.control_dim}), "
                    f"got {tuple(control_features.shape)}"
                )
            control_features = control_features.to(model_input).float()
            if self.control_projection is not None:
                projected = self.control_projection(control_features)
                if self.control_dim >= 7:
                    available = control_features[:, 4:7].amax(dim=-1, keepdim=True)
                    projected = projected * available / (self.control_projection_dim**0.5)
                model_input = torch.cat((model_input, projected), dim=-1)
            else:
                model_input = torch.cat((model_input, control_features), dim=-1)
        elif control_features is not None and control_features.shape[-1] != 0:
            raise ValueError("control features were provided to a model with control_dim=0")
        return self.network(model_input)


class SimilarityLatentRetriever(nn.Module):
    """Predict a latent by softmax interpolation over material text prototypes."""

    def __init__(
        self,
        prototype_embeddings: torch.Tensor,
        prototype_latents: torch.Tensor,
        neighbors: int,
        temperature: float,
        prototype_anisotropy: torch.Tensor | None = None,
        directional_threshold: float = 0.30,
    ) -> None:
        super().__init__()
        if prototype_embeddings.ndim != 2 or prototype_latents.ndim != 2:
            raise ValueError("retrieval prototypes must be rank-2 embedding and latent matrices")
        if prototype_embeddings.shape[0] != prototype_latents.shape[0] or not len(prototype_embeddings):
            raise ValueError("retrieval embedding and latent prototypes must have equal nonzero row counts")
        if neighbors < 1 or neighbors > len(prototype_embeddings):
            raise ValueError("neighbors must be between one and the prototype count")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.embedding_dim = int(prototype_embeddings.shape[1])
        self.latent_dim = int(prototype_latents.shape[1])
        self.profile_dim = 0
        self.control_dim = 0
        self.neighbors = int(neighbors)
        self.temperature = float(temperature)
        if prototype_anisotropy is not None and prototype_anisotropy.shape != (len(prototype_embeddings),):
            raise ValueError("prototype_anisotropy must have one value per prototype")
        self.directional_threshold = float(directional_threshold)
        self.register_buffer("prototype_embeddings", F.normalize(prototype_embeddings.float(), dim=-1))
        self.register_buffer("prototype_latents", prototype_latents.float())
        self.register_buffer(
            "prototype_anisotropy",
            prototype_anisotropy.float() if prototype_anisotropy is not None else None,
        )

    def forward(
        self,
        embedding: torch.Tensor,
        profile_features: torch.Tensor | None = None,
        control_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if embedding.shape[-1] != self.embedding_dim:
            raise ValueError(
                f"expected text embeddings with width {self.embedding_dim}, got {embedding.shape[-1]}"
            )
        if profile_features is not None and profile_features.shape[-1] != 0:
            raise ValueError("retrieval does not accept profile features")
        query = F.normalize(embedding.float(), dim=-1)
        similarity = query @ self.prototype_embeddings.T
        if control_features is not None:
            if self.prototype_anisotropy is None:
                raise ValueError("directional retrieval requested without prototype anisotropy labels")
            directional = control_features.to(device=embedding.device, dtype=torch.bool).reshape(-1)
            if directional.shape != (embedding.shape[0],):
                raise ValueError("directional retrieval flags must have shape [batch]")
            if directional.any():
                eligible = self.prototype_anisotropy >= self.directional_threshold
                if int(eligible.sum()) < self.neighbors:
                    raise ValueError("too few anisotropic train prototypes for the configured neighbor count")
                similarity = similarity.clone()
                similarity[directional] = similarity[directional].masked_fill(~eligible, -torch.inf)
        scores, indices = similarity.topk(self.neighbors, dim=-1)
        weights = torch.softmax((scores - scores.amax(dim=-1, keepdim=True)) / self.temperature, dim=-1)
        neighbors = self.prototype_latents[indices]
        return (neighbors * weights.unsqueeze(-1)).sum(dim=1)
