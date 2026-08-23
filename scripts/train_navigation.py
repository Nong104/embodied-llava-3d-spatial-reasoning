import random
import torch

from ai2thor.controller import Controller

from src.models.llava_loader import load_llava_4bit
from src.models.embodied_llava import EmbodiedLLaVA
from src.navigation.navigation_head import NavigationHead, ACTION_TO_ID
from src.navigation.path_to_actions import path_to_actions

SCENES = ["FloorPlan_Train1_1", "FloorPlan_Train1_2", "FloorPlan_Train1_3", "FloorPlan_Train1_4", "FloorPlan_Train1_5"]
NUM_EPISODES = 100
MAX_STEPS_PER_EPISODE = 15
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

    trainable_params = (
        list(embodied.spatial_encoder.parameters())
        + list(embodied.fusion.parameters())
        + list(nav_head.action_head.parameters())
        + list(nav_head.norm.parameters())
    )
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-5)

    controller = Controller(
        agentMode="locobot", gridSize=0.25, rotateStepDegrees=90,
        visibilityDistance=1.5, width=300, height=300, fieldOfView=60,
    )

    nav_head.train()
    episode_losses = []
    total_steps_attempted = 0
    total_steps_skipped = 0

    for episode in range(1, NUM_EPISODES + 1):
        scene = random.choice(SCENES)
        event = controller.reset(scene=scene)
        objects = event.metadata["objects"]
        if not objects:
            continue
        target_obj = random.choice(objects)

        path_event = controller.step(action="GetShortestPath", objectId=target_obj["objectId"])
        if not path_event.metadata["lastActionSuccess"]:
            print(f"episode {episode}: path planning failed, skipping")
            continue

        corners_raw = path_event.metadata["actionReturn"]["corners"]
        corners = [[p["x"], p["y"], p["z"]] for p in corners_raw]
        start_rotation = event.metadata["agent"]["rotation"]["y"]
        expert_actions = path_to_actions(corners, start_rotation_y=start_rotation)

        instruction_text = f"Navigate to: {target_obj['objectType']}"
        encoded = processor.tokenizer(instruction_text, return_tensors="pt")
        instruction_input_ids = encoded["input_ids"].to(device)
        instruction_attention_mask = encoded["attention_mask"].to(device)

        step_losses = []
        for action in expert_actions[:MAX_STEPS_PER_EPISODE]:
            total_steps_attempted += 1
            current_event = controller.last_event
            boxes, category_ids = get_scene_boxes(current_event)
            boxes = boxes.unsqueeze(0).to(device)
            category_ids = category_ids.unsqueeze(0).to(device)
            image_tokens = encode_image(current_event.frame, processor, llava_model, device)

            target_action_id = torch.tensor([ACTION_TO_ID[action]], device=device)

            optimizer.zero_grad()
            out = nav_head(
                image_tokens, boxes, category_ids,
                instruction_input_ids, instruction_attention_mask,
                target_action_ids=target_action_id,
            )

            if torch.isnan(out.loss):
                total_steps_skipped += 1
                controller.step(action=action)
                continue

            out.loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                total_steps_skipped += 1
                optimizer.zero_grad()
                controller.step(action=action)
                continue

            # snapshot params before stepping, so a step that corrupts any
            # parameter into nan/inf can be rolled back immediately instead
            # of silently poisoning all future steps
            param_backup = [p.detach().clone() for p in trainable_params]
            optimizer.step()

            params_ok = all(torch.isfinite(p).all() for p in trainable_params)
            if not params_ok:
                total_steps_skipped += 1
                with torch.no_grad():
                    for p, backup in zip(trainable_params, param_backup):
                        p.copy_(backup)
                optimizer.zero_grad()
                controller.step(action=action)
                continue

            step_losses.append(out.loss.item())

            controller.step(action=action)  # teacher forcing: follow the EXPERT action

        if step_losses:
            avg_loss = sum(step_losses) / len(step_losses)
            episode_losses.append(avg_loss)
            print(
                f"episode {episode:3d}/{NUM_EPISODES}: scene={scene} target={target_obj['objectType']} "
                f"steps_used={len(step_losses)} avg_loss={avg_loss:.4f}"
            )
        else:
            print(f"episode {episode:3d}/{NUM_EPISODES}: all steps skipped (nan)")

    controller.stop()
    print("\nImitation learning smoke test finished.")
    print("episode avg losses:", episode_losses)
    print(f"total steps attempted: {total_steps_attempted}, skipped (nan): {total_steps_skipped} "
          f"({100 * total_steps_skipped / max(total_steps_attempted, 1):.1f}%)")


if __name__ == "__main__":
    main()