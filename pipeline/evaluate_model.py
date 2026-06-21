"""
============================================================================
CAMP Culvert ML Pipeline - EVALUATE MODEL
============================================================================
Proper evaluation of a trained classifier on a dataset's test split:
  - overall accuracy
  - per-class precision, recall, F1
  - confusion matrix (counts + row-normalized %)
  - macro / weighted F1

Use this to compare experiments. "Accuracy" alone hides whether you got
better at the MINORITY class (the one you actually care about - major
silting, corrosion present). Recall on that class is the real metric.

USAGE:
  python evaluate_model.py ^
    --model "C:\\...\\models\\silting_binary_model_field_2026_06_17.pt" ^
    --data  "C:\\...\\datasets\\silting_binary" ^
    --split test

Optional:
  --classmap "C:\\...\\models\\silting_binary_classmap_field_2026_06_17.json"
      (maps folder index -> readable name in the report)
============================================================================
"""

import argparse
import json
import os
from collections import Counter, defaultdict

import torch
from PIL import Image
from torchvision import transforms

INPUT_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="TorchScript .pt model")
    ap.add_argument("--data", required=True, help="dataset root (has the split folder)")
    ap.add_argument("--split", default="test", help="split folder to evaluate")
    ap.add_argument("--classmap", default=None, help="optional classmap JSON")
    ap.add_argument("--input-size", type=int, default=INPUT_SIZE)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    split_dir = os.path.join(args.data, args.split)
    if not os.path.isdir(split_dir):
        # tolerate valid/val naming
        alt = os.path.join(args.data, "val")
        split_dir = alt if args.split == "valid" and os.path.isdir(alt) else split_dir
    if not os.path.isdir(split_dir):
        raise SystemExit(f"Split folder not found: {split_dir}")

    class_folders = sorted([d for d in os.listdir(split_dir)
                            if os.path.isdir(os.path.join(split_dir, d))])
    n_classes = len(class_folders)
    idx_of = {c: i for i, c in enumerate(class_folders)}

    # readable names
    names = {i: c for i, c in enumerate(class_folders)}
    if args.classmap and os.path.exists(args.classmap):
        cm = json.load(open(args.classmap))
        # classmap is {index: folder_name}; we want readable label per index
        # try to also load CLASS_MEANINGS.txt if present
    meanings_path = os.path.join(args.data, "CLASS_MEANINGS.txt")
    if os.path.exists(meanings_path):
        for line in open(meanings_path):
            line = line.strip()
            if line.startswith("folder '"):
                # e.g. folder '1' = Major (>=60%)
                try:
                    fld = line.split("'")[1]
                    meaning = line.split("=", 1)[1].strip()
                    if fld in idx_of:
                        names[idx_of[fld]] = f"{fld}:{meaning}"
                except Exception:
                    pass

    model = torch.jit.load(args.model, map_location=device)
    model.eval()
    tf = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    # confusion[true][pred]
    confusion = [[0] * n_classes for _ in range(n_classes)]
    n_total = 0

    for cls in class_folders:
        t = idx_of[cls]
        folder = os.path.join(split_dir, cls)
        files = [f for f in os.listdir(folder) if f.lower().endswith(IMG_EXT)]
        for f in files:
            img = Image.open(os.path.join(folder, f)).convert("RGB")
            x = tf(img).unsqueeze(0).to(device)
            with torch.no_grad():
                p = int(model(x).argmax(1).item())
            confusion[t][p] += 1
            n_total += 1

    # ---- metrics ----
    correct = sum(confusion[i][i] for i in range(n_classes))
    acc = correct / max(n_total, 1)

    print(f"\nEvaluated {n_total} images from '{args.split}' split")
    print(f"Overall accuracy: {acc*100:.2f}%\n")

    # confusion matrix (counts)
    header = "true\\pred " + " ".join(f"{names[j][:10]:>11}" for j in range(n_classes))
    print("Confusion matrix (counts):")
    print(header)
    for i in range(n_classes):
        row = " ".join(f"{confusion[i][j]:>11d}" for j in range(n_classes))
        print(f"{names[i][:10]:>9} {row}")

    # row-normalized (recall view)
    print("\nConfusion matrix (row %, = recall per true class):")
    print(header)
    for i in range(n_classes):
        rt = sum(confusion[i]) or 1
        row = " ".join(f"{100*confusion[i][j]/rt:>10.1f}%" for j in range(n_classes))
        print(f"{names[i][:10]:>9} {row}")

    # per-class precision / recall / f1
    print("\nPer-class metrics:")
    print(f"{'class':>22} {'precision':>10} {'recall':>10} {'f1':>8} {'support':>9}")
    macro_f1 = 0.0
    weighted_f1 = 0.0
    for i in range(n_classes):
        tp = confusion[i][i]
        fp = sum(confusion[r][i] for r in range(n_classes)) - tp
        fn = sum(confusion[i]) - tp
        support = sum(confusion[i])
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        macro_f1 += f1
        weighted_f1 += f1 * support
        print(f"{names[i][:22]:>22} {prec:>10.3f} {rec:>10.3f} {f1:>8.3f} {support:>9d}")
    macro_f1 /= n_classes
    weighted_f1 /= max(n_total, 1)
    print(f"\nMacro F1:    {macro_f1:.3f}   (treats all classes equally)")
    print(f"Weighted F1: {weighted_f1:.3f}   (weighted by class size)")
    print("\nFor imbalanced problems, watch RECALL on the minority class -")
    print("that's the fraction of real positives the model actually catches.")


if __name__ == "__main__":
    main()
