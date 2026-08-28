import torch
import torch.nn as nn

from src.encoders.spatial_3d_encoder import Spatial3DEncoder
from src.fusion.q_former_fusion import QFormerFusion
from src.fusion.early_fusion import EarlyFusion
from src.fusion.late_fusion import LateFusion
from src.fusion.moe_fusion import MoEFusion


# Maps the fusion_strategy string used throughout the ablation study
# (Table 3.1 / Section 3.8.3) to the corresponding fusion module class.
# Adding a fifth strategy in the future only requires one new entry here.
FUSION_STRATEGIES = {
    "qformer": QFormerFusion,
    "early": EarlyFusion,
    "late": LateFusion,
    "moe": MoEFusion,
}


class EmbodiedLLaVA(nn.Module):
    def __init__(
        self,
        llava_model: nn.Module,
        num_categories: int,
        hidden_dim: int = 4096,
        fusion_strategy: str = "qformer",
        num_fusion_queries: int = 32,
    ):
        super().__init__()

        if fusion_strategy not in FUSION_STRATEGIES:
            raise ValueError(
                f"Unknown fusion_strategy {fusion_strategy!r}. "
                f"Expected one of {list(FUSION_STRATEGIES)}."
            )

        self.llava_model = llava_model
        self.fusion_strategy = fusion_strategy

        self.spatial_encoder = Spatial3DEncoder(
            num_categories=num_categories,
            category_embed_dim=128,
            hidden_dim=512,
            output_dim=hidden_dim,
        )

        if fusion_strategy == "qformer":
            self.fusion = QFormerFusion(
                input_dim=hidden_dim,
                hidden_dim=hidden_dim,
                num_queries=num_fusion_queries,
                num_heads=8,
            )
        else:
            fusion_cls = FUSION_STRATEGIES[fusion_strategy]
            self.fusion = fusion_cls(hidden_dim=hidden_dim)

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

        # Call the top-level model directly rather than its inner .model
        # submodule. When a LoRA adapter is attached via peft, the attribute
        # name .model is reused internally by peft itself and no longer
        # refers to the original inner submodule, so calling it directly
        # would silently skip the LoRA-modified layers or return a
        # differently-shaped output. Calling the top-level callable works
        # correctly whether or not LoRA is attached, and already returns
        # logits directly without needing a separate lm_head call.
        outputs = self.llava_model(
            inputs_embeds=inputs_embeds,
            attention_mask=combined_attention_mask,
        )

        logits = outputs.logits

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