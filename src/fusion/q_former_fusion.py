import torch
import torch.nn as nn


class QFormerFusion(nn.Module):
    def __init__(
        self,
        input_dim: int = 4096,
        hidden_dim: int = 4096,
        num_queries: int = 32,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.query_tokens = nn.Parameter(
            torch.randn(1, num_queries, hidden_dim) * 0.02
        )

        self.input_projection = nn.Linear(input_dim, hidden_dim)

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        image_tokens: torch.Tensor,
        spatial_tokens: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = image_tokens.size(0)

        memory = torch.cat([image_tokens, spatial_tokens], dim=1)
        memory = self.input_projection(memory)

        queries = self.query_tokens.expand(batch_size, -1, -1)

        attended_tokens, _ = self.cross_attention(
            query=queries,
            key=memory,
            value=memory,
        )

        fused_tokens = self.norm1(queries + attended_tokens)
        fused_tokens = self.norm2(fused_tokens + self.ffn(fused_tokens))

        return fused_tokens