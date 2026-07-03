import torch

from src.fusion.q_former_fusion import QFormerFusion


def main():
    fusion = QFormerFusion(
        input_dim=4096,
        hidden_dim=4096,
        num_queries=32,
        num_heads=8,
    )

    image_tokens = torch.randn(2, 64, 4096)
    spatial_tokens = torch.randn(2, 10, 4096)

    fused_tokens = fusion(image_tokens, spatial_tokens)

    print("image_tokens:", image_tokens.shape)
    print("spatial_tokens:", spatial_tokens.shape)
    print("fused_tokens:", fused_tokens.shape)


if __name__ == "__main__":
    main()