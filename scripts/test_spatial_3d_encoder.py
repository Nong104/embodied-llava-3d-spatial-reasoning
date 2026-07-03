import torch

from src.encoders.spatial_3d_encoder import Spatial3DEncoder


def main():
    encoder = Spatial3DEncoder(
        num_categories=100,
        category_embed_dim=128,
        hidden_dim=512,
        output_dim=4096,
    )

    boxes = torch.randn(2, 10, 6)
    category_ids = torch.randint(0, 100, (2, 10))

    spatial_tokens = encoder(boxes, category_ids)

    print("boxes:", boxes.shape)
    print("category_ids:", category_ids.shape)
    print("spatial_tokens:", spatial_tokens.shape)


if __name__ == "__main__":
    main()