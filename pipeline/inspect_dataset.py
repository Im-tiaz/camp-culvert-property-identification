"""
============================================================================
CAMP Culvert ML Pipeline - DATASET INSPECTOR
============================================================================
Reports the structure, image counts, and class balance of a training
dataset (silting or corrosion) before retraining. Run this first so we
know exactly what we're training on.

USAGE:
    python inspect_dataset.py --data "C:\\path\\to\\silting"
    python inspect_dataset.py --data "C:\\path\\to\\corrosion"
============================================================================
"""

import argparse
import os

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def count_images(folder):
    n = 0
    for root, _, files in os.walk(folder):
        n += sum(1 for f in files if f.lower().endswith(IMG_EXT))
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to dataset root")
    args = ap.parse_args()

    root = os.path.abspath(args.data)
    print(f"Dataset root: {root}\n")

    splits = [s for s in ("train", "valid", "val", "test")
              if os.path.isdir(os.path.join(root, s))]
    if not splits:
        print("ERROR: no train/valid/test subfolders found.")
        print(f"Contents: {os.listdir(root)}")
        return

    grand_total = 0
    all_classes = set()
    per_split = {}

    for split in splits:
        split_dir = os.path.join(root, split)
        classes = sorted([d for d in os.listdir(split_dir)
                          if os.path.isdir(os.path.join(split_dir, d))])
        all_classes.update(classes)
        counts = {c: count_images(os.path.join(split_dir, c)) for c in classes}
        per_split[split] = counts
        total = sum(counts.values())
        grand_total += total
        print(f"[{split}]  {total} images")
        for c in classes:
            bar = "#" * int(40 * counts[c] / max(total, 1))
            print(f"    class '{c}': {counts[c]:5d}  {bar}")
        print()

    n_classes = len(all_classes)
    task = ("SILTING (3-class)" if n_classes == 3
            else "CORROSION (2-class)" if n_classes == 2
            else f"UNKNOWN ({n_classes} classes)")
    print(f"Detected task : {task}")
    print(f"Classes       : {sorted(all_classes)}")
    print(f"Total images  : {grand_total}")

    # imbalance warning on the training split
    train_key = "train" if "train" in per_split else splits[0]
    tc = per_split[train_key]
    if tc:
        mx, mn = max(tc.values()), min(tc.values())
        ratio = mx / max(mn, 1)
        print(f"\nTrain class balance: max/min ratio = {ratio:.1f}x")
        if ratio > 2.0:
            print("  WARNING: classes are imbalanced (>2x). retrain.py will")
            print("  apply class weighting to compensate.")
        else:
            print("  Balance looks OK.")


if __name__ == "__main__":
    main()
