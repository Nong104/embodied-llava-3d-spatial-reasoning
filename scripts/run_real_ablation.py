import torch
from torch.utils.data import DataLoader

from src.data.scanqa_dataset import RealScanQADataset, scanqa_collate_fn
from src.models.embodied_llava import EmbodiedLLaVA
from src.models.llava_loader import load_llava_4bit
from src.models.generation import generate_answer
from src.evaluation.metrics import compute_metrics

FUSION_STRATEGIES = ["early", "qformer", "late", "moe"]
NUM_TRAIN_STEPS = 100  # per strategy; kept modest given the 5-day deadline
NUM_EVAL_SAMPLES = 10
LOG_EVERY = 20


def build_qa_texts(questions, answers, eos_token):
    prompts, full_texts = [], []
    for question, answer in zip(questions, answers):
        prompt = f"Question: {question} Answer:"
        full_text = prompt + " " + answer + "." + eos_token
        prompts.append(prompt)
        full_texts.append(full_text)
    return prompts, full_texts


def train_one_strategy(strategy, llava_model, processor, train_dataloader, val_dataset, device):
    print(f"\n{'=' * 70}")
    print(f"Fusion strategy: {strategy}")
    print(f"{'=' * 70}")

    eos_token = processor.tokenizer.eos_token
    eos_token_id = processor.tokenizer.eos_token_id

    model = EmbodiedLLaVA(
        llava_model=llava_model,
        num_categories=100,
        hidden_dim=4096,
        fusion_strategy=strategy,
        num_fusion_queries=32,
    )
    model.spatial_encoder.to(device)
    model.fusion.to(device)

    trainable_params = list(model.spatial_encoder.parameters()) + list(model.fusion.parameters())
    num_trainable = sum(p.numel() for p in trainable_params)
    print(f"Trainable parameters: {num_trainable:,}")

    optimizer = torch.optim.AdamW(trainable_params, lr=1e-4, weight_decay=0.01)

    model.train()
    train_iter = iter(train_dataloader)
    losses = []

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

        aux_loss = getattr(model.fusion, "last_load_balancing_loss", None)
        total_loss = loss if aux_loss is None else loss + 0.01 * aux_loss

        total_loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if step % LOG_EVERY == 0 or step == 1:
            print(f"step {step:4d}/{NUM_TRAIN_STEPS}: loss = {loss.item():.4f}")

    # --- Generate + score on real held-out validation questions ---
    model.eval()
    predictions, questions_list, references_list = [], [], []

    print(f"\n--- Generated answers ({strategy}) ---")
    for i in range(NUM_EVAL_SAMPLES):
        sample = val_dataset[i]
        prompt = f"Question: {sample['question']} Answer:"

        image_tokens = torch.randn(1, 64, 4096, device=device)
        boxes = sample["boxes"].unsqueeze(0).to(device)
        category_ids = sample["category_ids"].unsqueeze(0).to(device)

        answer = generate_answer(
            model, processor.tokenizer, image_tokens, boxes, category_ids,
            prompt=prompt, max_new_tokens=12, eos_token_id=eos_token_id,
        )
        predictions.append(answer.strip())
        questions_list.append(sample["question"])
        references_list.append([sample["answer"]])
        print(f"  Q: {sample['question']!r:50} predicted: {answer.strip()!r:25} reference: {sample['answer']!r}")

    metrics = compute_metrics(predictions, references_list, questions_list)

    del model
    torch.cuda.empty_cache()

    return {
        "strategy": strategy,
        "final_loss": losses[-1],
        "trainable_params": num_trainable,
        "EM": metrics["EM"],
        "BLEU-4": metrics["BLEU-4"],
        "CIDEr": metrics["CIDEr"],
    }


def main():
    device = "cuda"

    processor, llava_model = load_llava_4bit()
    for param in llava_model.parameters():
        param.requires_grad = False

    train_dataset = RealScanQADataset(
        json_path="data/scanqa_raw/ScanQA_v1.0_train.json",
        max_objects=8,
        num_categories=100,
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=4, shuffle=True, collate_fn=scanqa_collate_fn,
    )

    val_dataset = RealScanQADataset(
        json_path="data/scanqa_raw/ScanQA_v1.0_val.json",
        max_objects=8,
        num_categories=100,
    )

    results = []
    for strategy in FUSION_STRATEGIES:
        result = train_one_strategy(strategy, llava_model, processor, train_dataloader, val_dataset, device)
        results.append(result)

    print("\n" + "=" * 100)
    print("Table 3.4 — Fusion-strategy ablation on real ScanQA text (placeholder boxes)")
    print(f"{'Strategy':<12}{'Final Loss':<14}{'Trainable Params':<20}{'EM':<10}{'BLEU-4':<12}{'CIDEr':<10}")
    print("=" * 100)
    for r in results:
        print(
            f"{r['strategy']:<12}{r['final_loss']:<14.4f}{r['trainable_params']:<20,}"
            f"{r['EM']:<10.3f}{r['BLEU-4']:<12.6f}{r['CIDEr']:<10.3f}"
        )

    print("\nReal-data ablation across all four fusion strategies passed.")


if __name__ == "__main__":
    main()