import torch

from src.models.embodied_llava import EmbodiedLLaVA
from src.models.llava_loader import load_llava_4bit


def main():
    processor, llava_model = load_llava_4bit()

    model = EmbodiedLLaVA(
        llava_model=llava_model,
        num_categories=100,
        hidden_dim=4096,
        num_fusion_queries=32,
    )

    image_tokens = torch.randn(2, 64, 4096)
    boxes = torch.randn(2, 10, 6)
    category_ids = torch.randint(0, 100, (2, 10))

    fused_tokens = model(
        image_tokens=image_tokens,
        boxes=boxes,
        category_ids=category_ids,
    )

    print("Loaded real LLaVA backbone.")
    print("LLaVA type:", type(llava_model).__name__)
    print("image_tokens:", image_tokens.shape)
    print("boxes:", boxes.shape)
    print("category_ids:", category_ids.shape)
    print("fused_tokens:", fused_tokens.shape)


if __name__ == "__main__":
    main()