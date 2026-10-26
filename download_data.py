"""
download_data.py - Standalone downloader for authentic PlantVillage tomato leaf images.
LeafLens: Plant Disease Screening Using CNN and Transfer Learning
"""

import os
import sys
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

TARGET_CLASSES = [
    "Tomato___healthy",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Bacterial_spot",
    "Tomato___Leaf_Mold",
]

GITHUB_API_DIR_URL = "https://api.github.com/repos/spMohanty/PlantVillage-Dataset/contents/raw/color"
GITHUB_RAW_BASE_URL = "https://raw.githubusercontent.com/spMohanty/PlantVillage-Dataset/master/raw/color"


def download_single_file(args):
    raw_url, dest_path = args
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
        return True
    headers = {"User-Agent": "LeafLens-Downloader/1.0"}
    try:
        req = urllib.request.Request(raw_url, headers=headers)
        with urllib.request.urlopen(req, timeout=15) as resp:
            content = resp.read()
            with open(dest_path, "wb") as f:
                f.write(content)
        return True
    except Exception as e:
        return False


def download_class_images(class_name: str, output_dir: str, max_images: int = 120):
    class_dir = os.path.join(output_dir, class_name)
    os.makedirs(class_dir, exist_ok=True)

    existing = [f for f in os.listdir(class_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if len(existing) >= max_images:
        print(f"[LeafLens] '{class_name}': {len(existing)} images already present.")
        return len(existing)

    headers = {"User-Agent": "LeafLens-Downloader/1.0"}
    api_url = f"{GITHUB_API_DIR_URL}/{class_name}"

    try:
        req = urllib.request.Request(api_url, headers=headers)
        with urllib.request.urlopen(req, timeout=20) as resp:
            entries = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        print(f"[WARN] Error fetching list for {class_name}: {e}")
        return len(existing)

    image_entries = [e for e in entries if e.get("name", "").lower().endswith((".jpg", ".jpeg", ".png"))][:max_images]
    print(f"[LeafLens] Fetching {len(image_entries)} images for '{class_name}'...")

    tasks = []
    for item in image_entries:
        fname = item["name"]
        raw_url = item.get("download_url") or f"{GITHUB_RAW_BASE_URL}/{class_name}/{fname}"
        dest_path = os.path.join(class_dir, fname)
        tasks.append((raw_url, dest_path))

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(download_single_file, tasks))

    successes = sum(1 for r in results if r)
    print(f"[LeafLens] Finished '{class_name}': {successes}/{len(tasks)} images saved.")
    return successes


def main():
    dest_dir = sys.argv[1] if len(sys.argv) > 1 else "data/plantvillage_subset"
    max_img = int(sys.argv[2]) if len(sys.argv) > 2 else 120

    print("=" * 60)
    print(" LeafLens: Downloading Authentic PlantVillage Dataset Subset")
    print("=" * 60)
    print(f"Destination: {os.path.abspath(dest_dir)}")
    print(f"Max images per class: {max_img}")
    print(f"Target classes: {TARGET_CLASSES}")

    start_time = time.time()
    total_downloaded = 0
    for cname in TARGET_CLASSES:
        count = download_class_images(cname, dest_dir, max_images=max_img)
        total_downloaded += count

    elapsed = time.time() - start_time
    print(f"\n[LeafLens] Download complete! {total_downloaded} total images in {elapsed:.1f}s.")


if __name__ == "__main__":
    main()
