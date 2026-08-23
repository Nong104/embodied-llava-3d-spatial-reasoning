import json

import numpy as np
from plyfile import PlyData


def extract_scannet_objects(scene_dir, scene_id, max_objects=8):
    """Parse a ScanNet scene's raw files (the .ply mesh, .segs.json
    segmentation, and .aggregation.json object labels) into a list of
    real object labels and bounding boxes.

    Returns a list of dicts: {'label': str, 'bbox': [cx, cy, cz, sx, sy, sz]},
    where the bbox format matches Spatial3DEncoder's expected input and the
    'boxes' field already used throughout this project.
    """
    ply_path = f"{scene_dir}/{scene_id}_vh_clean_2.ply"
    segs_path = f"{scene_dir}/{scene_id}_vh_clean_2.0.010000.segs.json"
    agg_path = f"{scene_dir}/{scene_id}.aggregation.json"

    plydata = PlyData.read(ply_path)
    verts = plydata["vertex"]
    xyz = np.stack([verts["x"], verts["y"], verts["z"]], axis=1)

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
        objects.append({"label": label, "bbox": bbox})

    return objects