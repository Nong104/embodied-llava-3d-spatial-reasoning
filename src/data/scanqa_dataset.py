import json
from pathlib import Path

import torch
from torch.utils.data import Dataset


class ScanQADataset(Dataset):
    def __init__(
        self,
        json_path: str,
        max_objects: int = 32,
    ):
        self.json_path = Path(json_path)
        self.max_objects = max_objects

        with self.json_path.open("r", encoding="utf-8") as f:
            self.samples = json.load(f)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]

        question = sample["question"]
        answer = sample["answers"][0]

        objects = sample["objects"][: self.max_objects]

        boxes = []
        category_ids = []

        for obj in objects:
            boxes.append(obj["bbox"])
            category_ids.append(obj["category_id"])

        num_objects = len(boxes)

        while len(boxes) < self.max_objects:
            boxes.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            category_ids.append(0)

        boxes = torch.tensor(boxes, dtype=torch.float32)
        category_ids = torch.tensor(category_ids, dtype=torch.long)

        object_mask = torch.zeros(self.max_objects, dtype=torch.long)
        object_mask[:num_objects] = 1

        return {
            "scene_id": sample["scene_id"],
            "question": question,
            "answer": answer,
            "boxes": boxes,
            "category_ids": category_ids,
            "object_mask": object_mask,
        }


def scanqa_collate_fn(batch):
    scene_ids = [item["scene_id"] for item in batch]
    questions = [item["question"] for item in batch]
    answers = [item["answer"] for item in batch]

    boxes = torch.stack([item["boxes"] for item in batch], dim=0)
    category_ids = torch.stack([item["category_ids"] for item in batch], dim=0)
    object_mask = torch.stack([item["object_mask"] for item in batch], dim=0)

    return {
        "scene_ids": scene_ids,
        "questions": questions,
        "answers": answers,
        "boxes": boxes,
        "category_ids": category_ids,
        "object_mask": object_mask,
    }