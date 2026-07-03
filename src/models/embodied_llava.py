import torch
import torch.nn as nn

from src.encoders.spatial_3d_encoder import Spatial3DEncoder
from src.fusion.q_former_fusion import QFormerFusion


class EmbodiedLLaVA(nn.Module):
    def __init__(
        self,
        llava_model: nn.Module,
        num_categories: int,
        hidden_dim: int = 4096,
        num_fusion_queries: int = 32,
    ):
        super().__init__()

        self.llava_model = llava_model

        self.spatial_encoder = Spatial3DEncoder(
            num_categories=num_categories,
            category_embed_dim=128,
            hidden_dim=512,
            output_dim=hidden_dim,
        )

        self.fusion = QFormerFusion(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_queries=num_fusion_queries,
            num_heads=8,
        )

    def encode_spatial_tokens(
        self,
        boxes: torch.Tensor,
        category_ids: torch.Tensor,
    ) -> torch.Tensor:
        spatial_tokens = self.spatial_encoder(boxes, category_ids)
        return spatial_tokens

    def fuse_multimodal_tokens(
        self,
        image_tokens: torch.Tensor,
        spatial_tokens: torch.Tensor,
    ) -> torch.Tensor:
        fused_tokens = self.fusion(image_tokens, spatial_tokens)
        return fused_tokens

    def forward(
        self,
        image_tokens: torch.Tensor,
        boxes: torch.Tensor,
        category_ids: torch.Tensor,
    ) -> torch.Tensor:
        spatial_tokens = self.encode_spatial_tokens(
            boxes=boxes,
            category_ids=category_ids,
        )

        fused_tokens = self.fuse_multimodal_tokens(
            image_tokens=image_tokens,
            spatial_tokens=spatial_tokens,
        )

        return fused_tokens

    def forward_qa(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        image_tokens: torch.Tensor,
        boxes: torch.Tensor,
        category_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ):
        spatial_tokens = self.encode_spatial_tokens(
            boxes=boxes,
            category_ids=category_ids,
        )

        fused_tokens = self.fuse_multimodal_tokens(
            image_tokens=image_tokens,
            spatial_tokens=spatial_tokens,
        )

        text_embeddings = self.llava_model.get_input_embeddings()(input_ids)

        fused_tokens = fused_tokens.to(
            device=text_embeddings.device,
            dtype=text_embeddings.dtype,
        )

        inputs_embeds = torch.cat(
            [fused_tokens, text_embeddings],
            dim=1,
        )

        batch_size = input_ids.size(0)
        num_fused_tokens = fused_tokens.size(1)

        fused_attention_mask = torch.ones(
            batch_size,
            num_fused_tokens,
            device=attention_mask.device,
            dtype=attention_mask.dtype,
        )

        combined_attention_mask = torch.cat(
            [fused_attention_mask, attention_mask],
            dim=1,
        )

        combined_labels = None
        if labels is not None:
            ignore_fused_labels = torch.full(
                (batch_size, num_fused_tokens),
                -100,
                device=labels.device,
                dtype=labels.dtype,
            )

            combined_labels = torch.cat(
                [ignore_fused_labels, labels],
                dim=1,
            )

        outputs = self.llava_model.model(
            inputs_embeds=inputs_embeds,
            attention_mask=combined_attention_mask,
        )

        logits = self.llava_model.lm_head(outputs.last_hidden_state)

        loss = None
        if combined_labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = combined_labels[:, 1:].contiguous()

            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fn(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

        return type("Output", (), {"logits": logits, "loss": loss})()