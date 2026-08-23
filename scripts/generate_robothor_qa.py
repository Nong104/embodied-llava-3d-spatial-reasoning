import json
import math
import random

from ai2thor.controller import Controller

SCENES = [f"FloorPlan_Train{i}_{j}" for i in range(1, 13) for j in range(1, 6)]  # up to 60 training scenes
MAX_OBJECTS_PER_SCENE = 8
ABSENT_TYPE_CANDIDATES = ["Elephant", "Dinosaur", "Spaceship"]  # never appear in AI2-THOR, safe "no" answers


def euclidean_distance(center_a, center_b):
    return math.sqrt(
        (center_a["x"] - center_b["x"]) ** 2
        + (center_a["y"] - center_b["y"]) ** 2
        + (center_a["z"] - center_b["z"]) ** 2
    )


def generate_qa_for_scene(scene_id, objects, max_objects=MAX_OBJECTS_PER_SCENE, rng=None):
    rng = rng or random
    objects = objects[:max_objects]
    if len(objects) < 2:
        return []

    boxes, category_ids, type_names = [], [], []
    for obj in objects:
        c = obj["axisAlignedBoundingBox"]["center"]
        s = obj["axisAlignedBoundingBox"]["size"]
        boxes.append([c["x"], c["y"], c["z"], s["x"], s["y"], s["z"]])
        category_ids.append(hash(obj["objectType"]) % 100)
        type_names.append(obj["objectType"])

    samples = []

    present_type = rng.choice(type_names)
    absent_type = rng.choice(ABSENT_TYPE_CANDIDATES)
    for obj_type, answer in [(present_type, "yes"), (absent_type, "no")]:
        samples.append({
            "scene_id": scene_id,
            "question": f"Is there a {obj_type.lower()} in the room?",
            "answers": [answer],
            "boxes": boxes, "category_ids": category_ids,
        })

    counted_type = rng.choice(type_names)
    count = type_names.count(counted_type)
    samples.append({
        "scene_id": scene_id,
        "question": f"How many {counted_type.lower()}s are there?",
        "answers": [str(count)],
        "boxes": boxes, "category_ids": category_ids,
    })

    idx = rng.randrange(len(objects))
    ref_center = objects[idx]["axisAlignedBoundingBox"]["center"]
    best_j, best_dist = None, float("inf")
    for j, obj in enumerate(objects):
        if j == idx:
            continue
        d = euclidean_distance(ref_center, obj["axisAlignedBoundingBox"]["center"])
        if d < best_dist:
            best_dist, best_j = d, j
    if best_j is not None:
        samples.append({
            "scene_id": scene_id,
            "question": f"What is closest to the {type_names[idx].lower()}?",
            "answers": [type_names[best_j].lower()],
            "boxes": boxes, "category_ids": category_ids,
        })

    return samples


def main():
    controller = Controller(
        agentMode="locobot", gridSize=0.25, rotateStepDegrees=90,
        visibilityDistance=1.5, width=300, height=300, fieldOfView=60,
    )

    rng = random.Random(42)
    all_samples = []

    for scene in SCENES:
        try:
            event = controller.reset(scene=scene)
        except Exception as e:
            print(f"скip {scene}: {e}")
            continue

        objects = event.metadata["objects"]
        samples = generate_qa_for_scene(scene, objects, rng=rng)
        all_samples.extend(samples)
        print(f"{scene}: {len(objects)} objects -> {len(samples)} QA samples")

    controller.stop()

    output_path = "data/robothor_qa/robothor_qa_train.json"
    import os
    os.makedirs("data/robothor_qa", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_samples, f, indent=2)

    print(f"\nTotal QA samples generated: {len(all_samples)}")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()