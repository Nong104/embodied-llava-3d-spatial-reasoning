from src.data.scanqa_dataset import ScanQADataset


def main():
    dataset = ScanQADataset(
        json_path="data/toy_scanqa.json",
        max_objects=8,
    )

    print("dataset size:", len(dataset))

    sample = dataset[0]

    print("scene_id:", sample["scene_id"])
    print("question:", sample["question"])
    print("answer:", sample["answer"])
    print("boxes:", sample["boxes"].shape)
    print("category_ids:", sample["category_ids"].shape)
    print("object_mask:", sample["object_mask"])


if __name__ == "__main__":
    main()