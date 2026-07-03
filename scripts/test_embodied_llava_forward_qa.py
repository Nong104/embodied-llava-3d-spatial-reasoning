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

    device = "cuda"

    model.spatial_encoder.to(device)
    model.fusion.to(device)

    prompt = "Question: What is in the room? Answer:"
    encoded = processor.tokenizer(
        prompt,
        return_tensors="pt",
    )

    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    image_tokens = torch.randn(1, 64, 4096, device=device)
    boxes = torch.randn(1, 10, 6, device=device)
    category_ids = torch.randint(0, 100, (1, 10), device=device)

    with torch.no_grad():
        outputs = model.forward_qa(
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_tokens=image_tokens,
            boxes=boxes,
            category_ids=category_ids,
        )

    print("Forward QA passed.")
    print("logits:", outputs.logits.shape)


if __name__ == "__main__":
    main()