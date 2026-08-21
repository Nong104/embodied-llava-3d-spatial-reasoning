import torch
import torch.nn as nn
import torch.nn.functional as F


class MoEFusion(nn.Module):
    """Mixture-of-Experts fusion strategy (Table 3.1): tokens are routed to
    a small number of specialised expert sub-networks via a learned gating
    network, rather than every token being processed identically.

    Design (top-2 routing, following the standard sparse MoE formulation):
      1. A gate network scores every token against `num_experts` experts.
      2. Each token is routed to its top-`top_k` highest-scoring experts
         (sparse activation: most experts are skipped for most tokens).
      3. Each selected expert processes only the tokens routed to it, and
         its output is weighted by the (renormalised) gate probability.
      4. An auxiliary load-balancing loss is computed and exposed via
         `self.last_load_balancing_loss`, so the training loop can add it to
         the main loss with a small weight. Without this term, the gate
         tends to collapse onto a small subset of experts, which is exactly
         the "load balancing" difficulty flagged as MoE's limitation in
         Table 3.1.

    Interface-compatible with QFormerFusion: forward(image_tokens,
    spatial_tokens) -> fused_tokens (Section 3.8.3).
    """

    def __init__(
        self,
        hidden_dim: int = 4096,
        num_experts: int = 4,
        top_k: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        self.gate = nn.Linear(hidden_dim, num_experts)

        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for _ in range(num_experts)
            ]
        )

        self.norm = nn.LayerNorm(hidden_dim)

        # Populated on every forward() call; the training script is
        # responsible for adding this to the main loss (e.g. loss + 0.01 *
        # fusion.last_load_balancing_loss). Not folded into forward()'s
        # return value because forward() must keep returning only
        # fused_tokens to stay interface-compatible with the other three
        # strategies (Section 3.8.3).
        self.last_load_balancing_loss = None

    def forward(
        self,
        image_tokens: torch.Tensor,
        spatial_tokens: torch.Tensor,
    ) -> torch.Tensor:
        tokens = torch.cat([image_tokens, spatial_tokens], dim=1)  # [B, T, D]
        batch_size, seq_len, hidden_dim = tokens.shape
        flat_tokens = tokens.reshape(-1, hidden_dim)  # [B*T, D]

        gate_logits = self.gate(flat_tokens)  # [B*T, num_experts]
        gate_probs = F.softmax(gate_logits, dim=-1)

        topk_probs, topk_indices = gate_probs.topk(self.top_k, dim=-1)  # [B*T, top_k]
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)  # renormalise to sum to 1

        output = torch.zeros_like(flat_tokens)
        for expert_id in range(self.num_experts):
            expert_slot_mask = topk_indices == expert_id  # [B*T, top_k]
            token_mask = expert_slot_mask.any(dim=-1)  # [B*T]
            if not token_mask.any():
                continue

            expert_output = self.experts[expert_id](flat_tokens[token_mask])
            weight = (topk_probs * expert_slot_mask.float()).sum(dim=-1)[token_mask]
            output[token_mask] += weight.unsqueeze(-1) * expert_output

        fused_tokens = output.reshape(batch_size, seq_len, hidden_dim)
        fused_tokens = self.norm(fused_tokens)

        self.last_load_balancing_loss = self._load_balancing_loss(gate_probs, topk_indices)

        return fused_tokens

    def _load_balancing_loss(
        self,
        gate_probs: torch.Tensor,
        topk_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Switch-Transformer-style auxiliary loss: num_experts * sum_i(f_i * P_i),
        where f_i is the fraction of tokens routed to expert i (hard
        assignment) and P_i is the average gate probability assigned to
        expert i (soft). Minimising this term pushes both quantities toward
        a uniform 1/num_experts, discouraging the gate from collapsing onto
        a small subset of experts.
        """
        tokens_per_expert = torch.zeros(self.num_experts, device=gate_probs.device)
        for expert_id in range(self.num_experts):
            tokens_per_expert[expert_id] = (topk_indices == expert_id).float().mean()

        avg_gate_prob = gate_probs.mean(dim=0)  # [num_experts]

        return self.num_experts * (tokens_per_expert * avg_gate_prob).sum()