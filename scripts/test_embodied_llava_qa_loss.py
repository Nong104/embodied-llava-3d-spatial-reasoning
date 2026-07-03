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

    question = "Question: What is in the room? Answer:"
    answer = " a chair."

    full_text = question + answer

    encoded_full = processor.tokenizer(
        full_text,
        return_tensors="pt",
    )

    encoded_question = processor.tokenizer(
        question,
        return_tensors="pt",
    )

    input_ids = encoded_full["input_ids"].to(device)
    attention_mask = encoded_full["attention_mask"].to(device)

    labels = input_ids.clone()

    question_length = encoded_question["input_ids"].size(1)
    labels[:, :question_length] = -100

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
            labels=labels,
        )

    print("QA loss test passed.")
    print("logits:", outputs.logits.shape)
    print("loss:", outputs.loss.item())


if __name__ == "__main__":
    main()