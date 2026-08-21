import torch
from torch.utils.data import DataLoader

from src.data.scanqa_dataset import RealScanQADataset, scanqa_collate_fn
from src.models.embodied_llava import EmbodiedLLaVA
from src.models.llava_loader import load_llava_4bit
from src.models.generation import generate_answer
from src.evaluation.metrics import compute_metrics

FUSION_STRATEGY = "qformer"  # the thesis's default strategy (Section 3.4)
NUM_TRAIN_STEPS = 200
NUM_EVAL_SAMPLES = 8  # how many real validation questions to generate + score at the end
LOG_EVERY = 20


def build_qa_texts(questions, answers, eos_token):
    prompts = []
    full_texts = []
    for question, answer in zip(questions, answers):
        prompt = f"Question: {question} Answer:"
        # EOS appended so labels supervise *where to stop*, not just *what to say*
        # (verified separately that the tokenizer collapses this back into a
        # single eos_token_id rather than splitting it into ordinary text tokens).
        full_text = prompt + " " + answer + "." + eos_token
        prompts.append(prompt)
        full_texts.append(full_text)
    return prompts, full_texts


def main():
    device = "cuda"

    processor, llava_model = load_llava_4bit()
    for param in llava_model.parameters():
        param.requires_grad = False

    eos_token = processor.tokenizer.eos_token
    eos_token_id = processor.tokenizer.eos_token_id

    train_dataset = RealScanQADataset(
        json_path="data/scanqa_raw/ScanQA_v1.0_train.json",
        max_objects=8,
        num_categories=100,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=scanqa_collate_fn,
    )

    val_dataset = RealScanQADataset(
        json_path="data/scanqa_raw/ScanQA_v1.0_val.json",
        max_objects=8,
        num_categories=100,
    )

    model = EmbodiedLLaVA(
        llava_model=llava_model,
        num_categories=100,
        hidden_dim=4096,
        fusion_strategy=FUSION_STRATEGY,
        num_fusion_queries=32,
    )
    model.spatial_encoder.to(device)
    model.fusion.to(device)

    trainable_params = list(model.spatial_encoder.parameters()) + list(model.fusion.parameters())
    print(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")

    optimizer = torch.optim.AdamW(trainable_params, lr=1e-4, weight_decay=0.01)

    model.train()
    train_iter = iter(train_dataloader)

    for step in range(1, NUM_TRAIN_STEPS + 1):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_dataloader)
            batch = next(train_iter)

        prompts, full_texts = build_qa_texts(batch["questions"], batch["answers"], eos_token)

        encoded_full = processor.tokenizer(full_texts, return_tensors="pt", padding=True)
        encoded_prompts = processor.tokenizer(prompts, return_tensors="pt", padding=True)

        input_ids = encoded_full["input_ids"].to(device)
        attention_mask = encoded_full["attention_mask"].to(device)

        labels = input_ids.clone()
        for i in range(len(prompts)):
            prompt_length = encoded_prompts["attention_mask"][i].sum().item()
            labels[i, :prompt_length] = -100

        boxes = batch["boxes"].to(device)
        category_ids = batch["category_ids"].to(device)

        image_tokens = torch.randn(boxes.size(0), 64, 4096, device=device)

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

        if step % LOG_EVERY == 0 or step == 1:
            print(f"step {step:4d}/{NUM_TRAIN_STEPS}: loss = {loss.item():.4f}")

    print("\nTraining loop finished. Generating on real validation questions...\n")

    # --- Generate real answers on held-out validation questions and score them ---
    model.eval()
    predictions, questions_list, references_list = [], [], []

    for i in range(NUM_EVAL_SAMPLES):
        sample = val_dataset[i]
        prompt = f"Question: {sample['question']} Answer:"

        image_tokens = torch.randn(1, 64, 4096, device=device)
        boxes = sample["boxes"].unsqueeze(0).to(device)
        category_ids = sample["category_ids"].unsqueeze(0).to(device)

        answer = generate_answer(
            model,
            processor.tokenizer,
            image_tokens,
            boxes,
            category_ids,
            prompt=prompt,
            max_new_tokens=12,
            eos_token_id=eos_token_id,
        )
        predictions.append(answer.strip())
        questions_list.append(sample["question"])
        references_list.append([sample["answer"]])

        print(f"Q: {sample['question']}")
        print(f"   predicted: {answer.strip()!r:40} reference: {sample['answer']!r}")

    metrics = compute_metrics(predictions, references_list, questions_list)

    print("\n" + "=" * 60)
    print("Metrics on", NUM_EVAL_SAMPLES, "real validation questions")
    print("(spatial content not meaningful yet: boxes are still placeholders)")
    print("=" * 60)
    for key, value in metrics.items():
        print(f"{key:30}: {value}")


if __name__ == "__main__":
    main()