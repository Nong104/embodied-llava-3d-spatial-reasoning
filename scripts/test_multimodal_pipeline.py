import torch

from src.encoders.spatial_3d_encoder import Spatial3DEncoder
from src.fusion.q_former_fusion import QFormerFusion


def main():
    spatial_encoder = Spatial3DEncoder(
        num_categories=100,
        category_embed_dim=128,
        hidden_dim=512,
        output_dim=4096,
    )

    fusion = QFormerFusion(
        input_dim=4096,
        hidden_dim=4096,
        num_queries=32,
        num_heads=8,
    )

    boxes = torch.randn(2, 10, 6)
    category_ids = torch.randint(0, 100, (2, 10))
    image_tokens = torch.randn(2, 64, 4096)

    spatial_tokens = spatial_encoder(boxes, category_ids)
    fused_tokens = fusion(image_tokens, spatial_tokens)

    print("boxes:", boxes.shape)
    print("category_ids:", category_ids.shape)
    print("image_tokens:", image_tokens.shape)
    print("spatial_tokens:", spatial_tokens.shape)
    print("fused_tokens:", fused_tokens.shape)


if __name__ == "__main__":
    main()