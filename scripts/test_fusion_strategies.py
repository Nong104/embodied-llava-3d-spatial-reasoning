import torch
import torch.nn as nn

from src.models.embodied_llava import EmbodiedLLaVA


class DummyLLaVA(nn.Module):
    def __init__(self):
        super().__init__()


def main():
    for strategy in ["qformer", "early", "late", "moe"]:
        model = EmbodiedLLaVA(
            llava_model=DummyLLaVA(),
            num_categories=100,
            hidden_dim=4096,
            fusion_strategy=strategy,
        )

        image_tokens = torch.randn(2, 64, 4096)
        boxes = torch.randn(2, 10, 6)
        category_ids = torch.randint(0, 100, (2, 10))

        fused_tokens = model(
            image_tokens=image_tokens,
            boxes=boxes,
            category_ids=category_ids,
        )

        print(
            f"{strategy:10} -> fusion class: {type(model.fusion).__name__:15} "
            f"fused_tokens: {tuple(fused_tokens.shape)}"
        )

    print("\nAll four fusion strategies switched correctly.")


if __name__ == "__main__":
    main()