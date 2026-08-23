import torch
import torch.nn as nn

ACTIONS = ["MoveAhead", "RotateLeft", "RotateRight", "LookUp", "LookDown", "Stop"]
ACTION_TO_ID = {a: i for i, a in enumerate(ACTIONS)}


class NavigationHead(nn.Module):
    """Lightweight action-prediction head appended to the shared LLaVA
    backbone (Section 3.6.2). Wraps an existing EmbodiedLLaVA instance,
    reusing its spatial encoder, fusion module, and frozen LLaVA backbone;
    only this small classification head is newly trained for navigation,
    following the same shared-backbone, task-specific-head design used for
    the QA task.
    """

    def __init__(self, embodied_llava, hidden_dim=4096, num_actions=6):
        super().__init__()
        self.embodied_llava = embodied_llava  # reused, not owned
        # LayerNorm added after diagnosing occasional NaN losses: the raw
        # last-token hidden state from the frozen LLaVA backbone was seen
        # ranging roughly -43 to +60, which is large enough to occasionally
        # overflow inside CrossEntropyLoss's softmax. Normalising keeps the
        # scale stable regardless of which token/scene produced it.
        self.norm = nn.LayerNorm(hidden_dim)
        self.action_head = nn.Linear(hidden_dim, num_actions)

    def forward(
        self,
        image_tokens,
        boxes,
        category_ids,
        instruction_input_ids,
        instruction_attention_mask,
        target_action_ids=None,
    ):
        model = self.embodied_llava

        spatial_tokens = model.encode_spatial_tokens(boxes=boxes, category_ids=category_ids)
        fused_tokens = model.fuse_multimodal_tokens(image_tokens=image_tokens, spatial_tokens=spatial_tokens)

        text_embeddings = model.llava_model.get_input_embeddings()(instruction_input_ids)
        fused_tokens = fused_tokens.to(device=text_embeddings.device, dtype=text_embeddings.dtype)

        inputs_embeds = torch.cat([fused_tokens, text_embeddings], dim=1)

        batch_size = instruction_input_ids.size(0)
        num_fused_tokens = fused_tokens.size(1)
        fused_attention_mask = torch.ones(
            batch_size, num_fused_tokens,
            device=instruction_attention_mask.device,
            dtype=instruction_attention_mask.dtype,
        )
        combined_attention_mask = torch.cat([fused_attention_mask, instruction_attention_mask], dim=1)

        outputs = model.llava_model.model(
            inputs_embeds=inputs_embeds,
            attention_mask=combined_attention_mask,
        )

        # last token's hidden state = the "decision" representation.
        # .float() converts from the LLaVA backbone's float16 to float32,
        # matching action_head's dtype (mirrors the same fix used in
        # forward_qa for fused_tokens).
        last_hidden = outputs.last_hidden_state[:, -1, :].float()
        last_hidden = self.norm(last_hidden)
        action_logits = self.action_head(last_hidden)

        loss = None
        if target_action_ids is not None:
            loss = nn.CrossEntropyLoss()(action_logits, target_action_ids)

        return type("NavOutput", (), {"logits": action_logits, "loss": loss})()