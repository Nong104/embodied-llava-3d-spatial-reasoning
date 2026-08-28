import sys
sys.path.insert(0, ".")

import importlib.util
spec = importlib.util.spec_from_file_location("download_scannet", "download-scannet.py")
download_scannet = importlib.util.module_from_spec(spec)
spec.loader.exec_module(download_scannet)

FILETYPES = [".aggregation.json", "_vh_clean_2.ply", "_vh_clean_2.0.010000.segs.json"]
OUT_DIR = "data/scannet_raw"

with open("scenes_subset.txt") as f:
    scene_ids = [line.strip() for line in f if line.strip()]

print(f"Downloading {len(scene_ids)} scenes x {len(FILETYPES)} file types...")

failed_scenes = []
for scene_id in scene_ids:
    out_dir = f"{OUT_DIR}/scans/{scene_id}"
    print(f"--- {scene_id} ---")
    try:
        download_scannet.download_scan(scene_id, out_dir, FILETYPES, use_v1_sens=True, skip_existing=True)
    except Exception as e:
        print(f"  FAILED: {scene_id}: {e}")
        failed_scenes.append(scene_id)

print("Done.")
print(f"Failed scenes ({len(failed_scenes)}):", failed_scenes)