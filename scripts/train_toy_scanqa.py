import torch
from torch.utils.data import DataLoader

from src.data.scanqa_dataset import ScanQADataset, scanqa_collate_fn
from src.models.embodied_llava import EmbodiedLLaVA
from src.models.llava_loader import load_llava_4bit


def build_qa_texts(questions, answers):
    prompts = []
    full_texts = []

    for question, answer in zip(questions, answers):
        prompt = f"Question: {question} Answer:"
        full_text = prompt + " " + answer + "."

        prompts.append(prompt)
        full_texts.append(full_text)

    return prompts, full_texts


def freeze_llava(llava_model):
    for param in llava_model.parameters():
        param.requires_grad = False


def main():
    device = "cuda"

    processor, llava_model = load_llava_4bit()
    freeze_llava(llava_model)

    model = EmbodiedLLaVA(
        llava_model=llava_model,
        num_categories=100,
        hidden_dim=4096,
        num_fusion_queries=32,
    )

    model.spatial_encoder.to(device)
    model.fusion.to(device)

    trainable_params = list(model.spatial_encoder.parameters()) + list(model.fusion.parameters())

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=1e-4,
        weight_decay=0.01,
    )

    dataset = ScanQADataset(
        json_path="data/toy_scanqa.json",
        max_objects=8,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=scanqa_collate_fn,
    )

    model.train()

    for step in range(5):
        batch = next(iter(dataloader))

        prompts, full_texts = build_qa_texts(
            batch["questions"],
            batch["answers"],
        )

        encoded_full = processor.tokenizer(
            full_texts,
            return_tensors="pt",
            padding=True,
        )

        encoded_prompts = processor.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
        )

        input_ids = encoded_full["input_ids"].to(device)
        attention_mask = encoded_full["attention_mask"].to(device)

        labels = input_ids.clone()

        for i in range(len(prompts)):
            prompt_length = encoded_prompts["attention_mask"][i].sum().item()
            labels[i, :prompt_length] = -100

        boxes = batch["boxes"].to(device)
        category_ids = batch["category_ids"].to(device)

        image_tokens = torch.randn(
            boxes.size(0),
            64,
            4096,
            device=device,
        )

        optimizer.zero_grad()

        outputs = model.forward_qa(
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_tokens=image_tokens,
            boxes=boxes,
            category_ids=category_ids,
            labels=labels,
        )

        loss = outputs.loss
        loss.backward()
        optimizer.step()

        print(f"step {step + 1}: loss = {loss.item():.4f}")

    print("Toy ScanQA training loop passed.")


if __name__ == "__main__":
    main()