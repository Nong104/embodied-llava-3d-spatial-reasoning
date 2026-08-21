import torch
import torch.nn as nn


class EarlyFusion(nn.Module):
    """Early fusion strategy (Table 3.1): concatenate all modality vectors
    before they are passed to the LLM, with no additional cross-modal
    computation. This is deliberately the simplest of the four candidate
    fusion strategies, so that it can serve as a baseline in the ablation
    study: it isolates how much benefit, if any, the more elaborate fusion
    mechanisms (Q-Former, Late Fusion, MoE) provide over doing nothing more
    than concatenation.

    Interface-compatible with QFormerFusion: forward(image_tokens,
    spatial_tokens) -> fused_tokens, so it can be substituted directly into
    EmbodiedLLaVA without changing any other component (Section 3.8.3).
    """

    def __init__(self, hidden_dim: int = 4096):
        super().__init__()
        # No learned cross-modal mixing by design (that is the point of this
        # strategy). A single LayerNorm is kept so the output scale is
        # comparable to the other three strategies, which all end in a
        # normalisation step; without it, differences in output magnitude
        # alone (rather than the fusion mechanism) could confound the
        # ablation comparison.
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        image_tokens: torch.Tensor,
        spatial_tokens: torch.Tensor,
    ) -> torch.Tensor:
        fused_tokens = torch.cat([image_tokens, spatial_tokens], dim=1)
        fused_tokens = self.norm(fused_tokens)
        return fused_tokens