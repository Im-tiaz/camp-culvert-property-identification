"""
============================================================================
CAMP Culvert ML Pipeline - UNIFIED INFERENCE
============================================================================
Runs ALL enabled property models (one by one) over the same set of culvert
images and writes ONE combined CSV: one row per culvert, prediction +
confidence columns for every property.

Expected image folder structure (produced by download_all_images.py):

    IMAGES_ROOT/
      ├─ G-06-0269/            <- folder name = Culvert ID
      │    ├─ photo_1.jpg
      │    └─ photo_2.jpg
      ├─ OBJECTID_1234/        <- fallback naming also fine
      │    └─ photo_1.jpg
      ...

Also supports a FLAT folder of images named  {culvertid}_{n}.jpg
(the naming Hemanth's inference notebook assumed).

USAGE:
    python run_inference.py --images /path/to/all_images
    python run_inference.py --images /path/to/all_images --models silting,corrosion
    python run_inference.py --images /path/to/all_images --per-image

Output:
    outputs/camp_predictions_v{N}_{date}.csv          (per-culvert, aggregated)
    outputs/camp_predictions_per_image_v{N}_{date}.csv (optional, --per-image)
============================================================================
"""

import argparse
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime
from glob import glob
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from model_registry import MODEL_REGISTRY, OUTPUTS_DIR

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# ===========================================================================
# IMAGE DISCOVERY
# ===========================================================================

def discover_images(images_root: str):
    """
    Map culvert_id -> list of image paths.

    Mode 1 (preferred): subfolder per culvert  ->  folder name is the ID.
    Mode 2 (flat):      filenames like {culvertid}_{n}.jpg -> split on first '_'.
    """
    images_root = os.path.abspath(images_root)
    if not os.path.isdir(images_root):
        sys.exit(f"ERROR: images folder not found: {images_root}")

    culvert_images = defaultdict(list)

    subdirs = [d for d in sorted(os.listdir(images_root))
               if os.path.isdir(os.path.join(images_root, d))]

    if subdirs:  # Mode 1: one folder per culvert
        for d in subdirs:
            for root, _, files in os.walk(os.path.join(images_root, d)):
                for f in sorted(files):
                    if f.lower().endswith(IMG_EXTENSIONS):
                        culvert_images[d].append(os.path.join(root, f))
    else:        # Mode 2: flat folder
        for f in sorted(os.listdir(images_root)):
            if f.lower().endswith(IMG_EXTENSIONS):
                culvert_id = Path(f).stem.split("_", 1)[0]
                culvert_images[culvert_id].append(os.path.join(images_root, f))

    # drop culverts with zero images
    culvert_images = {k: v for k, v in culvert_images.items() if v}
    return culvert_images


# ===========================================================================
# MODEL WRAPPERS
# Each wrapper exposes: predict(image_path) -> dict
# ===========================================================================

class TorchScriptClassifier:
    """Hemanth-style EfficientNet TorchScript classifier."""

    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device
        self.model = torch.jit.load(cfg["model_path"], map_location=device)
        self.model.eval()
        size = cfg.get("input_size", 224)
        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        self.classes = cfg["classes"]
        # optional per-class probability thresholds (raises precision):
        # a class can only be PREDICTED if its probability >= its threshold.
        # Classes without a threshold are always allowed (e.g. the safe class).
        self.min_conf = cfg.get("min_confidence", {})

    @torch.no_grad()
    def predict(self, image_path):
        img = Image.open(image_path).convert("RGB")
        x = self.transform(img).unsqueeze(0).to(self.device)
        probs = torch.softmax(self.model(x), dim=1)[0]

        if self.min_conf:
            # disallow any class whose prob is below its threshold, then argmax
            # over what's left. For binary this is the real decision-threshold
            # knob: argmax alone is always >=0.5, so gating the flagged class's
            # probability is what actually trades recall for precision.
            allowed = probs.clone()
            for idx, label in self.classes.items():
                thr = self.min_conf.get(label)
                if thr is not None and float(probs[idx]) < thr:
                    allowed[idx] = -1.0
            if float(allowed.max()) < 0:      # everything gated out (rare)
                allowed = probs
            idx = int(torch.argmax(allowed))
            return {
                "prediction": self.classes[idx],
                "confidence": round(float(probs[idx]), 4),
            }

        conf, idx = torch.max(probs, dim=0)
        return {
            "prediction": self.classes[int(idx)],
            "confidence": round(float(conf), 4),
        }


class YoloClassifier:
    """Adam-style YOLO image classification model (material)."""

    def __init__(self, cfg, device):
        from ultralytics import YOLO  # lazy import
        self.model = YOLO(cfg["model_path"])
        self.device = device
        # class names come from the weights unless overridden in registry
        self.classes = cfg["classes"] or self.model.names

    def predict(self, image_path):
        res = self.model.predict(image_path, verbose=False, device=self.device)[0]
        idx = int(res.probs.top1)
        conf = float(res.probs.top1conf)
        return {
            "prediction": self.classes[idx],
            "confidence": round(conf, 4),
        }


class YoloDetector:
    """Adam-style YOLO object detection model (shape + count)."""

    def __init__(self, cfg, device):
        from ultralytics import YOLO  # lazy import
        self.model = YOLO(cfg["model_path"])
        self.device = device
        self.conf_thresh = cfg.get("confidence_threshold", 0.6)
        self.classes = cfg["classes"] or self.model.names

    def predict(self, image_path):
        res = self.model.predict(image_path, conf=self.conf_thresh,
                                 verbose=False, device=self.device)[0]
        n = len(res.boxes)
        if n == 0:
            return {"prediction": "NoDetection", "confidence": 0.0, "count": 0}
        # shape = most common detected class in this image
        cls_ids = [int(c) for c in res.boxes.cls.tolist()]
        confs = [float(c) for c in res.boxes.conf.tolist()]
        top_cls = Counter(cls_ids).most_common(1)[0][0]
        avg_conf = sum(confs) / len(confs)
        return {
            "prediction": self.classes[top_cls],
            "confidence": round(avg_conf, 4),
            "count": n,
        }


LOADERS = {
    "torchscript": TorchScriptClassifier,
    "yolo_classify": YoloClassifier,
    "yolo_detect": YoloDetector,
}


# ===========================================================================
# AGGREGATION  (multiple photos of the same culvert -> one value)
# ===========================================================================

def aggregate(preds, cfg):
    """
    preds: list of per-image prediction dicts for one culvert.
    Returns dict of aggregated columns for the CSV.
    """
    prop = cfg["property"]
    valid = [p for p in preds if p["prediction"] != "NoDetection"]
    pool = valid if valid else preds

    out = {}

    if cfg["aggregation"] == "worst_case":
        order = cfg["severity_order"]
        worst = max(pool, key=lambda p: order.index(p["prediction"])
                    if p["prediction"] in order else -1)
        out[f"{prop}_Prediction"] = worst["prediction"]
        out[f"{prop}_Confidence"] = worst["confidence"]

    elif cfg["aggregation"] == "majority":
        votes = Counter(p["prediction"] for p in pool)
        top_label, _ = votes.most_common(1)[0]
        # confidence = max confidence among images that voted for the winner
        conf = max(p["confidence"] for p in pool if p["prediction"] == top_label)
        out[f"{prop}_Prediction"] = top_label
        out[f"{prop}_Confidence"] = round(conf, 4)

    # extra: detector also reports culvert count (max across photos —
    # the photo with the clearest view shows the most barrels)
    if any("count" in p for p in preds):
        out["NumberOfCulverts_Prediction"] = max(p.get("count", 0) for p in preds)

    out[f"{prop}_NumImagesUsed"] = len(preds)
    return out


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    ap = argparse.ArgumentParser(description="CAMP unified culvert inference")
    ap.add_argument("--images", required=True, help="Root folder of culvert images")
    ap.add_argument("--models", default=None,
                    help="Comma-separated registry keys to run (default: all enabled)")
    ap.add_argument("--per-image", action="store_true",
                    help="Also write a per-image (non-aggregated) CSV")
    ap.add_argument("--output-dir", default=OUTPUTS_DIR)
    ap.add_argument("--threshold", nargs="*", default=[],
                    help="Override flagged-class confidence per task, e.g. "
                         "--threshold silting=0.7 corrosion=0.6 . Higher = fewer "
                         "false alarms (more precision), at some cost to recall.")
    ap.add_argument("--shape-conf", type=float, default=None,
                    help="Override the shape/count YOLO detector confidence "
                         "(default 0.6). Lower = detects fainter openings, "
                         "raising counts (recovers 'predicted 0' misses) at some "
                         "risk of over-counting. Try 0.3-0.4 to fix under-counting.")
    args = ap.parse_args()

    if args.shape_conf is not None:
        if "shape_count" in MODEL_REGISTRY:
            MODEL_REGISTRY["shape_count"]["confidence_threshold"] = args.shape_conf
            print(f"Shape/count detector confidence override: {args.shape_conf}")

    # apply --threshold overrides onto the registry's min_confidence.
    # The value is applied to the most-severe (flagged) class of that task.
    for item in args.threshold:
        if "=" not in item:
            sys.exit(f"--threshold expects task=value, got '{item}'")
        task, val = item.split("=", 1)
        if task not in MODEL_REGISTRY:
            sys.exit(f"--threshold: unknown task '{task}'")
        cfg = MODEL_REGISTRY[task]
        order = cfg.get("severity_order")
        if not order:
            sys.exit(f"--threshold: task '{task}' has no severity_order to gate")
        flagged = order[-1]   # most severe class = the one we gate
        cfg.setdefault("min_confidence", {})[flagged] = float(val)
        print(f"Threshold override: {task} '{flagged}' >= {float(val)}")

    # device
    if torch.cuda.is_available():
        device = "cuda"
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Device: {device}")

    # which models to run
    selected = (args.models.split(",") if args.models
                else [k for k, v in MODEL_REGISTRY.items() if v.get("enabled")])
    print(f"Models to run: {selected}")

    # discover images
    culvert_images = discover_images(args.images)
    n_imgs = sum(len(v) for v in culvert_images.values())
    print(f"Found {len(culvert_images)} culverts, {n_imgs} images\n")
    if not culvert_images:
        sys.exit("No images found - check the folder path/structure.")

    # results[culvert_id] = {col: value}, per_image_rows = list of dicts
    results = defaultdict(dict)
    per_image_rows = []

    # ---- run models ONE BY ONE (sequential, as requested) ----
    for key in selected:
        if key not in MODEL_REGISTRY:
            print(f"!! '{key}' not in registry - skipping"); continue
        cfg = MODEL_REGISTRY[key]
        if not os.path.exists(cfg["model_path"]):
            print(f"!! {key}: model file missing ({cfg['model_path']}) - skipping")
            continue

        print(f"=== Running model: {key} ({cfg['property']}) ===")
        model = LOADERS[cfg["loader"]](cfg, device)

        for culvert_id, paths in tqdm(culvert_images.items(), unit="culvert"):
            preds = []
            for p in paths:
                try:
                    pred = model.predict(p)
                except Exception as e:
                    pred = {"prediction": "ERROR", "confidence": 0.0}
                    tqdm.write(f"  error on {os.path.basename(p)}: {e}")
                preds.append(pred)
                per_image_rows.append({
                    "CulvertID": culvert_id,
                    "Image": os.path.basename(p),
                    "Property": cfg["property"],
                    "Prediction": pred["prediction"],
                    "Confidence": pred["confidence"],
                    **({"Count": pred["count"]} if "count" in pred else {}),
                })
            ok = [p for p in preds if p["prediction"] != "ERROR"]
            if ok:
                results[culvert_id].update(aggregate(ok, cfg))
            else:
                results[culvert_id][f"{cfg['property']}_Prediction"] = "ERROR"

        # free model memory before loading the next one
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
        print()

    # ---- write combined CSV ----
    os.makedirs(args.output_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y_%m_%d")
    version = len(glob(os.path.join(args.output_dir, "camp_predictions_v*.csv"))) + 1

    rows = []
    for culvert_id, cols in results.items():
        rows.append({"CulvertID": culvert_id,
                     "NumImages": len(culvert_images[culvert_id]),
                     **cols})
    df = pd.DataFrame(rows).sort_values("CulvertID")

    out_csv = os.path.join(args.output_dir, f"camp_predictions_v{version}_{stamp}.csv")
    df.to_csv(out_csv, index=False)
    print(f"Saved combined results -> {out_csv}")
    print(f"  {len(df)} culverts x {len(df.columns)} columns")
    print(df.head(10).to_string(index=False))

    if args.per_image:
        out2 = os.path.join(args.output_dir,
                            f"camp_predictions_per_image_v{version}_{stamp}.csv")
        pd.DataFrame(per_image_rows).to_csv(out2, index=False)
        print(f"Saved per-image results -> {out2}")


if __name__ == "__main__":
    main()
