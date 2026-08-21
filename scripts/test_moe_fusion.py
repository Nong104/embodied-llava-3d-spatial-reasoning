import torch

from src.fusion.moe_fusion import MoEFusion


def main():
    fusion = MoEFusion(hidden_dim=4096, num_experts=4, top_k=2)

    image_tokens = torch.randn(2, 64, 4096)
    spatial_tokens = torch.randn(2, 10, 4096)

    fused_tokens = fusion(image_tokens, spatial_tokens)

    print("image_tokens:", image_tokens.shape)
    print("spatial_tokens:", spatial_tokens.shape)
    print("fused_tokens:", fused_tokens.shape)
    print("load_balancing_loss:", fusion.last_load_balancing_loss.item())


if __name__ == "__main__":
    main()