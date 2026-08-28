import time
import torch

from src.data.scanqa_dataset import ScanNetRealQADataset, scanqa_collate_fn
from src.models.embodied_llava import EmbodiedLLaVA
from src.models.llava_loader import load_llava_4bit, attach_lora
from torch.utils.data import DataLoader

device = "cuda"

processor, llava_model = load_llava_4bit()
llava_model = attach_lora(llava_model)

model = EmbodiedLLaVA(
    llava_model=llava_model, num_categories=100, hidden_dim=4096,
    fusion_strategy="late", num_fusion_queries=32,
)
model.spatial_encoder.to(device)
model.fusion.to(device)

lora_params = [p for p in llava_model.parameters() if p.requires_grad]
trainable_params = list(model.spatial_encoder.parameters()) + list(model.fusion.parameters()) + lora_params
optimizer = torch.optim.AdamW(trainable_params, lr=1e-4)

full_dataset = ScanNetRealQADataset(
    scanqa_json_path="data/scanqa_raw/ScanQA_v1.0_train.json",
    scannet_dir="data/scannet_raw/scans",
    max_objects=8,
)
dataloader = DataLoader(full_dataset, batch_size=4, shuffle=True, collate_fn=scanqa_collate_fn)

eos_token = processor.tokenizer.eos_token

model.train()
train_iter = iter(dataloader)

for step in range(1, 6):
    batch = next(train_iter)
    prompts = [f"Question: {q} Answer:" for q in batch["questions"]]
    full_texts = [p + " " + a + "." + eos_token for p, a in zip(prompts, batch["answers"])]

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

    start = time.time()
    optimizer.zero_grad()
    outputs = model.forward_qa(
        input_ids=input_ids, attention_mask=attention_mask,
        image_tokens=image_tokens, boxes=boxes, category_ids=category_ids, labels=labels,
    )
    outputs.loss.backward()
    optimizer.step()
    step_time = time.time() - start

    print(f"step {step}: loss={outputs.loss.item():.4f}  time={step_time:.2f}s")

print("\n单独测试late策略（全新环境，非累积状态）完成")