import random
import math
import torch

from ai2thor.controller import Controller

from src.models.llava_loader import load_llava_4bit
from src.models.embodied_llava import EmbodiedLLaVA
from src.navigation.navigation_head import NavigationHead, ACTIONS
from src.navigation.path_to_actions import path_to_actions

SCENES = ["FloorPlan_Train1_1", "FloorPlan_Train1_2", "FloorPlan_Train1_3", "FloorPlan_Train1_4", "FloorPlan_Train1_5"]
NUM_EVAL_EPISODES = 5
MAX_STEPS = 20
SUCCESS_THRESHOLD = 1.0  # meters, matches thesis Table 3.3's SR definition
NUM_CATEGORIES = 100


def object_type_to_category_id(object_type):
    return hash(object_type) % NUM_CATEGORIES


def get_scene_boxes(event, max_objects=8):
    objects = event.metadata["objects"][:max_objects]
    boxes, category_ids = [], []
    for obj in objects:
        c = obj["axisAlignedBoundingBox"]["center"]
        s = obj["axisAlignedBoundingBox"]["size"]
        boxes.append([c["x"], c["y"], c["z"], s["x"], s["y"], s["z"]])
        category_ids.append(object_type_to_category_id(obj["objectType"]))
    while len(boxes) < max_objects:
        boxes.append([0.0] * 6)
        category_ids.append(0)
    return torch.tensor(boxes, dtype=torch.float32), torch.tensor(category_ids, dtype=torch.long)


def distance(pos_a, pos_b):
    return math.sqrt((pos_a["x"] - pos_b["x"]) ** 2 + (pos_a["z"] - pos_b["z"]) ** 2)


def path_length(corners):
    total = 0.0
    for i in range(len(corners) - 1):
        total += math.dist(corners[i], corners[i + 1])
    return total


def main():
    device = "cuda"
    processor, llava_model = load_llava_4bit()
    for p in llava_model.parameters():
        p.requires_grad = False

    embodied = EmbodiedLLaVA(
        llava_model=llava_model, num_categories=NUM_CATEGORIES, hidden_dim=4096,
        fusion_strategy="qformer", num_fusion_queries=32,
    )
    embodied.spatial_encoder.to(device)
    embodied.fusion.to(device)
    nav_head = NavigationHead(embodied, hidden_dim=4096, num_actions=6).to(device)

    # NOTE: this currently evaluates a freshly-initialized head (no checkpoint
    # loading yet). Once training is scaled up and we save a checkpoint,
    # load its state_dict here before eval.

    controller = Controller(
        agentMode="locobot", gridSize=0.25, rotateStepDegrees=90,
        visibilityDistance=1.5, width=300, height=300, fieldOfView=60,
    )

    nav_head.eval()
    successes, spls = [], []

    for episode in range(1, NUM_EVAL_EPISODES + 1):
        scene = random.choice(SCENES)
        event = controller.reset(scene=scene)
        objects = event.metadata["objects"]
        if not objects:
            continue
        target_obj = random.choice(objects)
        target_pos = target_obj["position"]

        path_event = controller.step(action="GetShortestPath", objectId=target_obj["objectId"])
        if not path_event.metadata["lastActionSuccess"]:
            print(f"episode {episode}: path planning failed, skipping")
            continue
        corners_raw = path_event.metadata["actionReturn"]["corners"]
        corners = [[p["x"], p["y"], p["z"]] for p in corners_raw]
        shortest_len = path_length(corners)

        instruction_text = f"Navigate to: {target_obj['objectType']}"
        encoded = processor.tokenizer(instruction_text, return_tensors="pt")
        instruction_input_ids = encoded["input_ids"].to(device)
        instruction_attention_mask = encoded["attention_mask"].to(device)

        traveled = 0.0
        prev_pos = event.metadata["agent"]["position"]

        with torch.no_grad():
            for step in range(MAX_STEPS):
                current_event = controller.last_event
                boxes, category_ids = get_scene_boxes(current_event)
                boxes = boxes.unsqueeze(0).to(device)
                category_ids = category_ids.unsqueeze(0).to(device)
                image_tokens = encode_image(current_event.frame, processor, llava_model, device)

                out = nav_head(image_tokens, boxes, category_ids, instruction_input_ids, instruction_attention_mask)
                action_id = out.logits.argmax(dim=-1).item()
                action = ACTIONS[action_id]

                if action == "Stop":
                    break

                controller.step(action=action)
                new_pos = controller.last_event.metadata["agent"]["position"]
                traveled += distance(prev_pos, new_pos)
                prev_pos = new_pos

        final_pos = controller.last_event.metadata["agent"]["position"]
        final_dist = distance(final_pos, target_pos)
        success = final_dist <= SUCCESS_THRESHOLD

        spl = 0.0
        if success:
            spl = shortest_len / max(traveled, shortest_len) if traveled > 0 else 1.0

        successes.append(1 if success else 0)
        spls.append(spl)

        print(
            f"episode {episode:3d}/{NUM_EVAL_EPISODES}: scene={scene} target={target_obj['objectType']} "
            f"success={success} final_dist={final_dist:.2f}m traveled={traveled:.2f}m spl={spl:.3f}"
        )

    controller.stop()

    sr = sum(successes) / len(successes) if successes else 0.0
    mean_spl = sum(spls) / len(spls) if spls else 0.0
    print("\n" + "=" * 60)
    print(f"Success Rate (SR):  {sr:.3f}  ({sum(successes)}/{len(successes)})")
    print(f"Mean SPL:           {mean_spl:.3f}")
    print("=" * 60)


if __name__ == "__main__":
    main()