import torch
import torch.nn as nn


class LateFusion(nn.Module):
    """Late fusion strategy (Table 3.1): each modality is encoded
    independently through its own small network, with no cross-modal
    interaction during that encoding step. The two independently-encoded
    token sequences are only combined at the very end, via concatenation
    followed by a shared projection layer.

    This sits between EarlyFusion (no learned processing at all) and
    QFormerFusion (cross-attention throughout) in terms of how much the two
    modalities are allowed to influence each other: here, image tokens and
    spatial tokens cannot attend to one another at any point, which is the
    "Reduced cross-modal interaction" limitation noted in Table 3.1. The
    corresponding advantage is modularity: image_encoder and spatial_encoder
    are independent sub-modules that could be swapped out individually
    without touching the rest of the pipeline.

    Interface-compatible with QFormerFusion: forward(image_tokens,
    spatial_tokens) -> fused_tokens (Section 3.8.3).
    """

    def __init__(self, hidden_dim: int = 4096, dropout: float = 0.1):
        super().__init__()

        # Each modality gets its own independent encoder. Neither of these
        # ever sees the other modality's tokens; that separation is the
        # defining property of "late" fusion.
        self.image_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.spatial_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # The only point where the two modalities meet: a shared linear
        # projection applied after concatenation. This mixes the output
        # representation space but is not attention, so tokens still do not
        # attend across modalities.
        self.merge_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        image_tokens: torch.Tensor,
        spatial_tokens: torch.Tensor,
    ) -> torch.Tensor:
        image_encoded = self.image_encoder(image_tokens)
        spatial_encoded = self.spatial_encoder(spatial_tokens)

        fused_tokens = torch.cat([image_encoded, spatial_encoded], dim=1)
        fused_tokens = self.merge_proj(fused_tokens)
        fused_tokens = self.norm(fused_tokens)

        return fused_tokens