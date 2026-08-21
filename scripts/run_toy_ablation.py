import torch
from torch.utils.data import DataLoader

from src.data.scanqa_dataset import ScanQADataset, scanqa_collate_fn
from src.models.embodied_llava import EmbodiedLLaVA
from src.models.llava_loader import load_llava_4bit
from src.models.generation import generate_answer
from src.evaluation.metrics import compute_metrics


FUSION_STRATEGIES = ["early", "qformer", "late", "moe"]
NUM_TRAIN_STEPS = 15  # small: this is a pipeline smoke test on 2 toy examples, not a real experiment


def build_qa_texts(questions, answers):
    prompts = []
    full_texts = []
    for question, answer in zip(questions, answers):
        prompt = f"Question: {question} Answer:"
        full_text = prompt + " " + answer + "."
        prompts.append(prompt)
        full_texts.append(full_text)
    return prompts, full_texts


def count_trainable_params(params):
    return sum(p.numel() for p in params)


def train_one_strategy(strategy, llava_model, processor, dataset, dataloader, device):
    print(f"\n{'=' * 60}")
    print(f"Fusion strategy: {strategy}")
    print(f"{'=' * 60}")

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
    num_trainable = count_trainable_params(trainable_params)

    optimizer = torch.optim.AdamW(trainable_params, lr=1e-4, weight_decay=0.01)

    model.train()
    losses = []
    for step in range(NUM_TRAIN_STEPS):
        batch = next(iter(dataloader))

        prompts, full_texts = build_qa_texts(batch["questions"], batch["answers"])

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

        # MoE exposes an auxiliary load-balancing term (Section 3.8.3); the
        # other three strategies simply don't set this attribute, so this
        # stays a no-op for them.
        aux_loss = getattr(model.fusion, "last_load_balancing_loss", None)
        total_loss = loss if aux_loss is None else loss + 0.01 * aux_loss

        total_loss.backward()
        optimizer.step()

        losses.append(loss.item())

    print(f"loss: {losses[0]:.4f} -> {losses[-1]:.4f}  (over {NUM_TRAIN_STEPS} steps)")

    # --- Generate real answers and score them ---
    model.eval()
    predictions, questions_list, references_list = [], [], []

    for i in range(len(dataset)):
        sample = dataset[i]
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
            max_new_tokens=6,
            eos_token_id=processor.tokenizer.eos_token_id,
        )
        predictions.append(answer.strip())
        questions_list.append(sample["question"])
        references_list.append([sample["answer"]])

    metrics = compute_metrics(predictions, references_list, questions_list)

    for q, p, r in zip(questions_list, predictions, [r[0] for r in references_list]):
        print(f"  Q: {q}")
        print(f"     predicted: {p!r:30} reference: {r!r}")

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

    dataset = ScanQADataset(json_path="data/toy_scanqa.json", max_objects=8)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=scanqa_collate_fn)

    results = []
    for strategy in FUSION_STRATEGIES:
        result = train_one_strategy(strategy, llava_model, processor, dataset, dataloader, device)
        results.append(result)
        torch.cuda.empty_cache()  # free memory before the next strategy's model is built

    print("\n" + "=" * 90)
    print(f"{'Strategy':<12}{'Final Loss':<14}{'Trainable Params':<20}{'EM':<10}{'BLEU-4':<12}{'CIDEr':<10}")
    print("=" * 90)
    for r in results:
        print(
            f"{r['strategy']:<12}{r['final_loss']:<14.4f}{r['trainable_params']:<20,}"
            f"{r['EM']:<10.3f}{r['BLEU-4']:<12.6f}{r['CIDEr']:<10.3f}"
        )

    print("\nToy ablation run across all four fusion strategies passed.")


if __name__ == "__main__":
    main()
    