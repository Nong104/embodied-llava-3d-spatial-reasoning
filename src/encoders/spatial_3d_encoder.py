import torch
import torch.nn as nn


class Spatial3DEncoder(nn.Module):
    def __init__(
        self,
        num_categories: int,
        category_embed_dim: int = 128,
        hidden_dim: int = 512,
        output_dim: int = 4096,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.category_embedding = nn.Embedding(
            num_embeddings=num_categories,
            embedding_dim=category_embed_dim,
        )

        input_dim = 6 + category_embed_dim

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

        self.layer_norm = nn.LayerNorm(output_dim)

    def forward(self, boxes: torch.Tensor, category_ids: torch.Tensor) -> torch.Tensor:
        if boxes.dim() != 3 or boxes.size(-1) != 6:
            raise ValueError("boxes must have shape [batch_size, num_objects, 6]")

        if category_ids.dim() != 2:
            raise ValueError("category_ids must have shape [batch_size, num_objects]")

        category_embeds = self.category_embedding(category_ids)
        features = torch.cat([boxes, category_embeds], dim=-1)

        embeddings = self.mlp(features)
        embeddings = self.layer_norm(embeddings)

        return embeddings