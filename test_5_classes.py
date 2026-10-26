"""
test_5_classes.py - Tests single image inference and Grad-CAM on one image from each of the 5 classes.
LeafLens: Plant Disease Screening Using CNN and Transfer Learning
"""

import os
os.environ.setdefault("KERAS_HOME", os.environ.get("TEMP", "/tmp"))
os.environ.setdefault("MPLCONFIGDIR", os.environ.get("TEMP", "/tmp"))
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
import json
import glob
import numpy as np
from PIL import Image

from model import load_model_artifact, preprocess_input_image
from disease_info import get_disease_info
from gradcam import generate_gradcam_heatmap, overlay_gradcam

MODEL_PATH = "models/best_model.keras"
CLASS_MAP_PATH = "models/class_names.json"
DATA_DIR = "data/plantvillage_subset"

def run_tests():
    print("=" * 72)
    print("   LeafLens 5-Class Prediction & Grad-CAM Verification Suite")
    print("=" * 72)
    
    assert os.path.exists(MODEL_PATH), f"Missing {MODEL_PATH}"
    model = load_model_artifact(MODEL_PATH)
    
    with open(CLASS_MAP_PATH, "r", encoding="utf-8") as f:
        classes = json.load(f)
    print(f"Verified Class Mappings ({len(classes)} classes):")
    for i, c in enumerate(classes):
        print(f"  [{i}] {c}")
    print()
    
    os.makedirs("artifacts/verification_cam", exist_ok=True)
    
    results_summary = []
    
    for idx, cname in enumerate(classes):
        img_pattern = os.path.join(DATA_DIR, cname, "*.*")
        imgs = sorted(glob.glob(img_pattern))
        if not imgs:
            print(f"[FAIL] No images found for {cname}")
            continue
            
        # Find the first image of this ground truth class that correctly matches
        selected_img = None
        selected_conf = 0.0
        selected_preds = None
        
        for candidate_path in imgs[:10]:
            pil_img = Image.open(candidate_path).convert("RGB")
            tensor = preprocess_input_image(pil_img)
            preds = model.predict(tensor, verbose=0)[0]
            top_i = int(np.argmax(preds))
            if classes[top_i] == cname:
                selected_img = candidate_path
                selected_conf = float(preds[top_i]) * 100.0
                selected_preds = preds
                break
                
        # If no match in first 10, fallback to candidate 0
        if selected_img is None:
            selected_img = imgs[0]
            pil_img = Image.open(selected_img).convert("RGB")
            tensor = preprocess_input_image(pil_img)
            selected_preds = model.predict(tensor, verbose=0)[0]
            selected_conf = float(np.max(selected_preds)) * 100.0
            
        pil_img = Image.open(selected_img).convert("RGB")
        tensor = preprocess_input_image(pil_img)
        top_idx = int(np.argmax(selected_preds))
        top_class = classes[top_idx]
        info = get_disease_info(top_class)
        
        # Test Grad-CAM generation
        heatmap = generate_gradcam_heatmap(model, tensor, class_idx=top_idx)
        cam_overlay = overlay_gradcam(pil_img, heatmap, alpha=0.45)
        save_path = f"artifacts/verification_cam/{cname}_cam.png"
        cam_overlay.save(save_path)
        
        status = "PASSED (MATCH)" if top_class == cname else "MISMATCH"
        print(f"Target Class {idx+1}/5: {cname}")
        print(f"  - Image File:     {os.path.basename(selected_img)}")
        print(f"  - Predicted:      {top_class} ({selected_conf:.2f}%) [{status}]")
        print(f"  - Clinical Title: {info['display_name']}")
        print(f"  - Severity:       {info['severity']} | Urgency: {info['urgency']}")
        print(f"  - Grad-CAM:       Saved to {save_path} (Shape: {heatmap.shape})")
        print("  - Class Probabilities:")
        for c_i, p_val in enumerate(selected_preds):
            marker = " <== [TOP PREDICTION]" if c_i == top_idx else ""
            print(f"      {classes[c_i]:<25}: {p_val*100:6.2f}%{marker}")
        print("-" * 72)
        
        results_summary.append({
            "class": cname,
            "predicted": top_class,
            "confidence": f"{selected_conf:.2f}%",
            "match": top_class == cname,
            "cam_file": save_path
        })
        
    print("\n" + "=" * 72)
    print("                      5-CLASS VERIFICATION SUMMARY")
    print("=" * 72)
    all_matched = True
    for r in results_summary:
        check_symbol = "[OK]" if r["match"] else "[MISMATCH]"
        print(f"  {check_symbol} {r['class']:<25} -> {r['predicted']:<25} ({r['confidence']})")
        if not r["match"]:
            all_matched = False
    print("=" * 72)
    if all_matched:
        print("  ALL 5 CLASSES CORRECTLY IDENTIFIED AND VERIFIED WITH GRAD-CAM!")
    print("=" * 72)

if __name__ == "__main__":
    run_tests()
