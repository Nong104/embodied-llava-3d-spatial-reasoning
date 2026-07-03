from torch.utils.data import DataLoader

from src.data.scanqa_dataset import ScanQADataset, scanqa_collate_fn


def main():
    dataset = ScanQADataset(
        json_path="data/toy_scanqa.json",
        max_objects=8,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=scanqa_collate_fn,
    )

    batch = next(iter(dataloader))

    print("scene_ids:", batch["scene_ids"])
    print("questions:", batch["questions"])
    print("answers:", batch["answers"])
    print("boxes:", batch["boxes"].shape)
    print("category_ids:", batch["category_ids"].shape)
    print("object_mask:", batch["object_mask"].shape)


if __name__ == "__main__":
    main()