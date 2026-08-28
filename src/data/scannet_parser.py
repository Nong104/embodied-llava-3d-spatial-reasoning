import json

import numpy as np
from plyfile import PlyData


def rgb_to_color_name(r, g, b):
    """Map an average RGB value to the closest of a small set of common
    color names, so the extracted color can be inserted into natural
    language answers the same way ScanQA reference answers describe color.
    """
    colors = {
        "black": (0, 0, 0), "white": (255, 255, 255), "gray": (128, 128, 128),
        "red": (200, 30, 30), "green": (30, 150, 30), "blue": (30, 30, 200),
        "yellow": (220, 220, 30), "brown": (140, 90, 50), "orange": (230, 140, 30),
        "beige": (220, 200, 170), "teal": (30, 150, 150), "purple": (130, 30, 160),
        "pink": (230, 150, 180),
    }
    best_name, best_dist = "gray", float("inf")
    for name, (cr, cg, cb) in colors.items():
        dist = (r - cr) ** 2 + (g - cg) ** 2 + (b - cb) ** 2
        if dist < best_dist:
            best_dist, best_name = dist, name
    return best_name


def extract_scannet_objects(scene_dir, scene_id, max_objects=8):
    """Parse a ScanNet scene's raw files (the .ply mesh, .segs.json
    segmentation, and .aggregation.json object labels) into a list of
    real object labels, bounding boxes, and average colors.

    Returns a list of dicts: {'label': str, 'bbox': [...], 'color_name': str},
    where bbox matches Spatial3DEncoder's expected input, and color_name is
    the nearest common color name to the object's real average vertex RGB,
    parsed from the same mesh file that already provided the coordinates.
    """
    ply_path = f"{scene_dir}/{scene_id}_vh_clean_2.ply"
    segs_path = f"{scene_dir}/{scene_id}_vh_clean_2.0.010000.segs.json"
    agg_path = f"{scene_dir}/{scene_id}.aggregation.json"

    plydata = PlyData.read(ply_path)
    verts = plydata["vertex"]
    xyz = np.stack([verts["x"], verts["y"], verts["z"]], axis=1)
    rgb = np.stack([verts["red"], verts["green"], verts["blue"]], axis=1).astype(np.float32)

    with open(segs_path) as f:
        seg_indices = np.array(json.load(f)["segIndices"])

    with open(agg_path) as f:
        seg_groups = json.load(f)["segGroups"]

    objects = []
    for group in seg_groups[:max_objects]:
        label = group["label"]
        segment_ids = set(group["segments"])
        vertex_mask = np.isin(seg_indices, list(segment_ids))
        obj_verts = xyz[vertex_mask]
        obj_rgb = rgb[vertex_mask]
        if len(obj_verts) == 0:
            continue
        vmin = obj_verts.min(axis=0)
        vmax = obj_verts.max(axis=0)
        center = (vmin + vmax) / 2
        size = vmax - vmin
        bbox = [
            float(center[0]), float(center[1]), float(center[2]),
            float(size[0]), float(size[1]), float(size[2]),
        ]
        avg_color = obj_rgb.mean(axis=0)
        color_name = rgb_to_color_name(*avg_color)
        objects.append({"label": label, "bbox": bbox, "color_name": color_name})

    return objects