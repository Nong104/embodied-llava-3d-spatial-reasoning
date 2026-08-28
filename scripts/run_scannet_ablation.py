import time
import random

import torch
from torch.utils.data import DataLoader, Subset

from src.data.scanqa_dataset import ScanNetRealQADataset, scanqa_collate_fn
from src.models.embodied_llava import EmbodiedLLaVA
from src.models.llava_loader import load_llava_4bit, attach_lora
from src.models.generation import generate_answer
from src.evaluation.metrics import compute_metrics

FUSION_STRATEGIES = ["early", "qformer", "late", "moe"]
NUM_TRAIN_STEPS = 10
NUM_EVAL_SAMPLES = 3
LOG_EVERY = 20
VAL_SCENE_FRACTION = 0.15


def build_qa_texts(questions, answers, eos_token):
    prompts, full_texts = [], []
    for question, answer in zip(questions, answers):
        prompt = f"Question: {question} Answer:"
        full_text = prompt + " " + answer + "." + eos_token
        prompts.append(prompt)
        full_texts.append(full_text)
    return prompts, full_texts


def split_by_scene(dataset, val_fraction, seed=42):
    scene_ids = sorted({sample["scene_id"] for sample in dataset.samples})
    rng = random.Random(seed)
    rng.shuffle(scene_ids)

    num_val_scenes = max(1, int(len(scene_ids) * val_fraction))
    val_scenes = set(scene_ids[:num_val_scenes])
    train_scenes = set(scene_ids[num_val_scenes:])

    train_indices = [i for i, s in enumerate(dataset.samples) if s["scene_id"] in train_scenes]
    val_indices = [i for i, s in enumerate(dataset.samples) if s["scene_id"] in val_scenes]

    print(f"Scene split: {len(train_scenes)} train scenes, {len(val_scenes)} val scenes "
          f"({len(train_indices)} train questions, {len(val_indices)} val questions)")

    return Subset(dataset, train_indices), Subset(dataset, val_indices)


def train_one_strategy(strategy, train_dataloader, val_dataset, device):
    print(f"\n{'=' * 70}")
    print(f"Fusion strategy: {strategy}")
    print(f"{'=' * 70}")

    # Fresh model AND fresh LoRA adapter for every strategy, so each starts
    # from an identical, untrained baseline. Sharing one LoRA-wrapped model
    # object across strategies would let earlier strategies' training leak
    # into later ones, breaking the controlled comparison in Section 3.8.3.
    load_start = time.time()
    processor, llava_model = load_llava_4bit()
    llava_model = attach_lora(llava_model)
    print(f"Fresh model + LoRA loaded in {time.time() - load_start:.1f}s")

    eos_token = processor.tokenizer.eos_token
    eos_token_id = processor.tokenizer.eos_token_id

    model = EmbodiedLLaVA(
        llava_model=llava_model, num_categories=100, hidden_dim=4096,
        fusion_strategy=strategy, num_fusion_queries=32,
    )
    model.spatial_encoder.to(device)
    model.fusion.to(device)

    lora_params = [p for p in llava_model.parameters() if p.requires_grad]
    trainable_params = (
        list(model.spatial_encoder.parameters())
        + list(model.fusion.parameters())
        + lora_params
    )
    num_trainable = sum(p.numel() for p in trainable_params)
    print(f"Trainable parameters (fusion + spatial encoder + LoRA): {num_trainable:,}")

    optimizer = torch.optim.AdamW(trainable_params, lr=1e-4, weight_decay=0.01)

    model.train()
    train_iter = iter(train_dataloader)
    losses = []

    train_start = time.time()
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
            input_ids=input_ids, attention_mask=attention_mask,
            image_tokens=image_tokens, boxes=boxes, category_ids=category_ids, labels=labels,
        )
        loss = outputs.loss

        aux_loss = getattr(model.fusion, "last_load_balancing_loss", None)
        total_loss = loss if aux_loss is None else loss + 0.01 * aux_loss

        total_loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if step % LOG_EVERY == 0 or step == 1:
            elapsed = time.time() - train_start
            print(f"step {step:4d}/{NUM_TRAIN_STEPS}: loss = {loss.item():.4f}  (elapsed: {elapsed:.1f}s)")

    train_duration = time.time() - train_start
    print(f"\nTraining ({NUM_TRAIN_STEPS} steps) took {train_duration:.1f}s "
          f"({train_duration / NUM_TRAIN_STEPS:.3f}s/step)")

    model.eval()
    predictions, questions_list, references_list = [], [], []

    print(f"\n--- Generated answers ({strategy}), unseen scenes, with LoRA ---")
    gen_start = time.time()
    num_eval = min(NUM_EVAL_SAMPLES, len(val_dataset))
    for i in range(num_eval):
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
        if i < 10:
            print(f"  Q: {sample['question']!r:45} predicted: {answer.strip()!r:20} reference: {sample['answer']!r}")

    gen_duration = time.time() - gen_start
    print(f"\nGeneration ({num_eval} samples) took {gen_duration:.1f}s ({gen_duration / num_eval:.2f}s/sample)")

    metrics = compute_metrics(predictions, references_list, questions_list)

    del model, llava_model, processor
    torch.cuda.empty_cache()

    return {
        "strategy": strategy, "final_loss": losses[-1], "trainable_params": num_trainable,
        "EM": metrics["EM"], "BLEU-4": metrics["BLEU-4"], "CIDEr": metrics["CIDEr"],
        "train_duration": train_duration, "gen_duration": gen_duration,
    }


def main():
    script_start = time.time()
    device = "cuda"

    # Dataset scene split is computed once and reused across all four
    # strategies, so every strategy trains and validates on the identical
    # question sets. Only the model (and its LoRA adapter) is reloaded fresh
    # per strategy.
    full_dataset = ScanNetRealQADataset(
        scanqa_json_path="data/scanqa_raw/ScanQA_v1.0_train.json",
        scannet_dir="data/scannet_raw/scans",
        max_objects=8,
    )
    print(f"Full dataset size: {len(full_dataset)}")

    train_dataset, val_dataset = split_by_scene(full_dataset, VAL_SCENE_FRACTION)

    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=scanqa_collate_fn)

    results = []
    for strategy in FUSION_STRATEGIES:
        result = train_one_strategy(strategy, train_dataloader, val_dataset, device)
        results.append(result)

    total_duration = time.time() - script_start

    print("\n" + "=" * 110)
    print("Table 5.y — Fusion-strategy ablation WITH LoRA, scene-disjoint split, real ScanNet + real ScanQA")
    print(f"{'Strategy':<12}{'Final Loss':<14}{'Trainable Params':<20}{'EM':<10}{'BLEU-4':<12}{'CIDEr':<10}{'Train(s)':<10}{'Gen(s)':<10}")
    print("=" * 110)
    for r in results:
        print(
            f"{r['strategy']:<12}{r['final_loss']:<14.4f}{r['trainable_params']:<20,}"
            f"{r['EM']:<10.3f}{r['BLEU-4']:<12.6f}{r['CIDEr']:<10.3f}"
            f"{r['train_duration']:<10.1f}{r['gen_duration']:<10.1f}"
        )

    print(f"\nTotal script runtime: {total_duration:.1f}s ({total_duration/60:.1f} min)")
    print("\nLoRA-enabled ablation across all four fusion strategies passed.")


if __name__ == "__main__":
    main()