import time
import random

import torch
from torch.utils.data import Subset

from src.data.scanqa_dataset import ScanNetRealQADataset
from src.models.llava_loader import load_llava_4bit
from src.evaluation.metrics import compute_metrics

NUM_EVAL_SAMPLES = 30
VAL_SCENE_FRACTION = 0.15


def split_by_scene(dataset, val_fraction, seed=42):
    scene_ids = sorted({sample["scene_id"] for sample in dataset.samples})
    rng = random.Random(seed)
    rng.shuffle(scene_ids)

    num_val_scenes = max(1, int(len(scene_ids) * val_fraction))
    val_scenes = set(scene_ids[:num_val_scenes])
    val_indices = [i for i, s in enumerate(dataset.samples) if s["scene_id"] in val_scenes]
    return Subset(dataset, val_indices)


def main():
    device = "cuda"
    script_start = time.time()

    processor, llava_model = load_llava_4bit()
    for p in llava_model.parameters():
        p.requires_grad = False
    llava_model.eval()

    eos_token_id = processor.tokenizer.eos_token_id

    full_dataset = ScanNetRealQADataset(
        scanqa_json_path="data/scanqa_raw/ScanQA_v1.0_train.json",
        scannet_dir="data/scannet_raw/scans",
        max_objects=8,
    )
    val_dataset = split_by_scene(full_dataset, VAL_SCENE_FRACTION)

    predictions, questions_list, references_list = [], [], []

    print("--- Zero-shot LLaVA (no fusion module, no LoRA, no spatial encoder) ---")
    num_eval = min(NUM_EVAL_SAMPLES, len(val_dataset))
    for i in range(num_eval):
        sample = val_dataset[i]
        prompt = f"Question: {sample['question']} Answer:"

        inputs = processor.tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            output_ids = llava_model.generate(
                **inputs,
                max_new_tokens=12,
                eos_token_id=eos_token_id,
                do_sample=False,
            )

        generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        answer = processor.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        import re
        match = re.search(r"[.!?]", answer)
        if match:
            answer = answer[: match.start()].strip()

        predictions.append(answer)
        questions_list.append(sample["question"])
        references_list.append([sample["answer"]])
        if i < 10:
            print(f"  Q: {sample['question']!r:45} predicted: {answer!r:20} reference: {sample['answer']!r}")

    metrics = compute_metrics(predictions, references_list, questions_list)
    total_duration = time.time() - script_start

    print("\n" + "=" * 90)
    print(f"RESULT strategy=zeroshot_llava EM={metrics['EM']:.3f} BLEU4={metrics['BLEU-4']:.6f} "
          f"CIDEr={metrics['CIDEr']:.3f} total_s={total_duration:.1f}")
    print("=" * 90)


if __name__ == "__main__":
    main()