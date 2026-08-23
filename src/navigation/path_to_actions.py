import math


def path_to_actions(corners, start_rotation_y=0.0, grid_size=0.25, rotate_step=90):
    """Convert a GetShortestPath corner-point list into a discrete action
    sequence matching the thesis's action space (Section 3.6.2):
    MoveAhead, RotateLeft, RotateRight, Stop.

    For each segment between consecutive corners, first rotate the agent
    (in rotate_step-degree increments) to face that segment's direction,
    then move forward the number of grid_size steps needed to cover it.
    """
    actions = []
    current_rotation = start_rotation_y % 360

    for i in range(len(corners) - 1):
        x0, _, z0 = corners[i]
        x1, _, z1 = corners[i + 1]

        dx = x1 - x0
        dz = z1 - z0
        segment_length = math.hypot(dx, dz)
        if segment_length < 1e-6:
            continue  # duplicate/zero-length segment, nothing to do

        # AI2-THOR yaw convention: 0=+z (north), 90=+x (east), 180=-z, 270=-x
        target_rotation = math.degrees(math.atan2(dx, dz)) % 360

        # Rotate to face the segment direction (shortest angular direction)
        angle_diff = (target_rotation - current_rotation + 180) % 360 - 180
        num_turns = round(abs(angle_diff) / rotate_step)
        turn_action = "RotateRight" if angle_diff > 0 else "RotateLeft"
        actions.extend([turn_action] * num_turns)
        current_rotation = (current_rotation + num_turns * rotate_step * (1 if angle_diff > 0 else -1)) % 360

        # Move forward enough grid steps to cover the segment
        num_moves = round(segment_length / grid_size)
        actions.extend(["MoveAhead"] * num_moves)

    actions.append("Stop")
    return actions


if __name__ == "__main__":
    # A simple L-shaped path: start facing north (0 deg), go 1m east, then 1m north.
    corners = [
        [0.0, 0.9, 0.0],
        [1.0, 0.9, 0.0],
        [1.0, 0.9, 1.0],
    ]
    actions = path_to_actions(corners, start_rotation_y=0.0, grid_size=0.25, rotate_step=90)
    print("actions:", actions)
    print("total actions:", len(actions))

    assert actions[-1] == "Stop"
    assert actions.count("MoveAhead") == 8
    print("path_to_actions logic test passed.")