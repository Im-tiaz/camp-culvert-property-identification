"""
============================================================================
CAMP Culvert ML Pipeline - THRESHOLD SWEEP
============================================================================
For a binary defect model (silting_binary, corrosion), sweep the decision
threshold on the FLAGGED class and report precision / recall / F1 / counts
at each value, measured on a labeled split (default: test).

Why: for a 2-class model the argmax is always >=0.5 confident, so raising
the threshold on the flagged class is the real precision/recall knob. This
tool shows the whole tradeoff curve so you can pick the operating point:
  - want to catch as many real defects as possible -> favor recall (low thr)
  - want few false alarms for reviewers           -> favor precision (high thr)
  - balance                                        -> highest F1

USAGE:
  python threshold_sweep.py ^
    --model "C:\\...\\models\\silting_binary_model_unfrozen_2026_06_17.pt" ^
    --data  "C:\\...\\datasets\\silting_binary" ^
    --flagged-index 1            (which folder index is the "defect" class)

Optional:
  --split test            (default)
  --thresholds 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9   (override the grid)
  --input-size 224
============================================================================
"""

import argparse
import os

import torch
from PIL import Image
from torchvision import transforms

INPUT_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def load_meanings(data_dir, idx_of):
    """Read CLASS_MEANINGS.txt if present -> {index: readable label}."""
    names = {i: c for c, i in idx_of.items()}
    path = os.path.join(data_dir, "CLASS_MEANINGS.txt")
    if os.path.exists(path):
        for line in open(path):
            line = line.strip()
            if line.startswith("folder '"):
                try:
                    fld = line.split("'")[1]
                    meaning = line.split("=", 1)[1].strip()
                    if fld in idx_of:
                        names[idx_of[fld]] = f"{fld}:{meaning}"
                except Exception:
                    pass
    return names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="TorchScript .pt model")
    ap.add_argument("--data", required=True, help="dataset root (has the split folder)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--flagged-index", type=int, default=1,
                    help="folder index of the defect/positive class (default 1)")
    ap.add_argument("--thresholds", type=float, nargs="*",
                    default=[0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90])
    ap.add_argument("--input-size", type=int, default=INPUT_SIZE)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    split_dir = os.path.join(args.data, args.split)
    if not os.path.isdir(split_dir):
        raise SystemExit(f"Split folder not found: {split_dir}")

    class_folders = sorted([d for d in os.listdir(split_dir)
                            if os.path.isdir(os.path.join(split_dir, d))])
    if len(class_folders) != 2:
        raise SystemExit(f"This sweep is for BINARY models; found "
                         f"{len(class_folders)} classes: {class_folders}")
    idx_of = {c: i for i, c in enumerate(class_folders)}
    names = load_meanings(args.data, idx_of)
    flag = args.flagged_index
    safe = 1 - flag
    print(f"Flagged (positive) class : {names[flag]}")
    print(f"Safe (negative) class    : {names[safe]}\n")

    model = torch.jit.load(args.model, map_location=device)
    model.eval()
    tf = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    # collect (true_label, prob_of_flagged_class) for every test image ONCE,
    # then we can evaluate any number of thresholds cheaply.
    records = []   # (true_idx, p_flagged)
    for cls in class_folders:
        t = idx_of[cls]
        folder = os.path.join(split_dir, cls)
        for f in os.listdir(folder):
            if not f.lower().endswith(IMG_EXT):
                continue
            img = Image.open(os.path.join(folder, f)).convert("RGB")
            x = tf(img).unsqueeze(0).to(device)
            with torch.no_grad():
                probs = torch.softmax(model(x), dim=1)[0]
            records.append((t, float(probs[flag])))

    n_pos = sum(1 for t, _ in records if t == flag)
    n_neg = len(records) - n_pos
    print(f"Evaluated {len(records)} images "
          f"({n_pos} {names[flag]}, {n_neg} {names[safe]})\n")

    # sweep
    header = (f"{'thresh':>7} {'precision':>10} {'recall':>9} {'f1':>8} "
              f"{'TP':>5} {'FP':>5} {'FN':>5} {'flagged':>8}")
    print("Sweep on the FLAGGED class (predict flagged iff p_flagged >= thresh):")
    print(header)
    print("-" * len(header))
    best = None
    for thr in args.thresholds:
        tp = fp = fn = 0
        for true_idx, p in records:
            pred_flag = p >= thr
            if pred_flag and true_idx == flag:
                tp += 1
            elif pred_flag and true_idx != flag:
                fp += 1
            elif (not pred_flag) and true_idx == flag:
                fn += 1
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        flagged = tp + fp
        marker = ""
        if best is None or f1 > best[1]:
            best = (thr, f1)
        print(f"{thr:>7.2f} {prec:>10.3f} {rec:>9.3f} {f1:>8.3f} "
              f"{tp:>5d} {fp:>5d} {fn:>5d} {flagged:>8d}{marker}")

    print("-" * len(header))
    print(f"Best F1 at threshold {best[0]:.2f} (F1={best[1]:.3f})")
    print("\nReading the table:")
    print("  precision = of the culverts flagged, fraction truly positive")
    print("  recall    = of the truly positive culverts, fraction caught")
    print("  FP        = false alarms a reviewer must rule out")
    print("  FN        = real defects missed")
    print("\nPick the threshold whose precision/recall balance fits your QC")
    print("workflow, then set it in model_registry.py min_confidence (or pass")
    print("--threshold to run_inference.py).")


if __name__ == "__main__":
    main()
