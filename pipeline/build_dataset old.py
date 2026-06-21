"""
============================================================================
CAMP Culvert ML Pipeline - DATASET BUILDER
============================================================================
Builds a train/valid/test image-classification dataset from YOUR field
images, labeled by the inspector values already in nmdot_table_fixed.csv.

Why: models trained on the inherited (curated) datasets collapse on real
field photos. Training on field images labeled by inspectors fixes the
distribution mismatch - test accuracy then reflects deployment accuracy.

This is the reusable Phase-2 tool: point --field at any inventory column,
give it a label-grouping map, and it builds a dataset for that property.

How the join works (same as the rest of the pipeline):
  inventory "Culvert ID"  ->  safe_folder_name()  ->  all_images/<that folder>/

USAGE (silting):
  python build_dataset.py ^
    --table "C:\\...\\nmdot_table_fixed.csv" ^
    --images "C:\\...\\all_images" ^
    --field "Silting" ^
    --out "C:\\...\\datasets\\silting_field" ^
    --map silting

USAGE (corrosion):
  python build_dataset.py --table ... --images ... ^
    --field "Corrosion" --out ...\\datasets\\corrosion_field --map corrosion

USAGE (any future property): pass --map none and edit GROUPINGS, or use
--map passthrough to make one class folder per distinct raw value.
============================================================================
"""

import argparse
import os
import random
import shutil
from collections import Counter, defaultdict

import pandas as pd

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
COL_ID = "Culvert ID"
COL_OBJECTID = "OBJECTID"


# ---------------------------------------------------------------------------
# Label groupings: raw inspector value -> class folder name.
# Folder names use 0/1/2... so the retrained model's class order is explicit
# and matches Hemanth's convention.
# ---------------------------------------------------------------------------
GROUPINGS = {
    # Silting -> 3 levels (paper definitions: low<10, med 10-60, high>60)
    "silting": {
        "Clean": "0", "Minor Silting (>%10)": "0",
        "10% to 30% Silted": "1", "30% to 60% Silted": "1",
        "60% to 90% Silted": "2", ">90% Silted": "2",
    },
    # Corrosion -> binary (0 = none/minor, 1 = corrosion)
    "corrosion": {
        "None Evident": "0", "Minor (Rusting on Inside OR Outside)": "0",
        "Moderate (Rusting on Inside AND Outside)": "1", "Major": "1",
        # "Not Known" intentionally omitted -> those records are skipped
    },
}

# human-readable meaning of each folder index, written into a README per dataset
MEANINGS = {
    "silting": {"0": "Low (<10%)", "1": "Medium (10-60%)", "2": "High (>60%)"},
    "corrosion": {"0": "None/Minor", "1": "Corrosion"},
}


def safe_folder_name(name: str) -> str:
    """EXACT copy of download_all_images.py logic - keeps the join consistent."""
    for ch in r'\/:*?"<>|':
        name = name.replace(ch, "_")
    return name.strip()


def folder_key(rec):
    cid = str(rec.get(COL_ID, "")).strip()
    oid = str(rec.get(COL_OBJECTID, "")).strip()
    if not cid or cid.lower() == "nan":
        return f"OBJECTID_{oid}"
    return safe_folder_name(cid)


def list_images(folder):
    out = []
    if os.path.isdir(folder):
        for root, _, files in os.walk(folder):
            for f in files:
                if f.lower().endswith(IMG_EXT):
                    out.append(os.path.join(root, f))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True)
    ap.add_argument("--images", required=True, help="all_images root")
    ap.add_argument("--field", required=True, help="inventory column to use as label")
    ap.add_argument("--out", required=True, help="output dataset folder")
    ap.add_argument("--map", default="passthrough",
                    help="grouping: silting | corrosion | passthrough")
    ap.add_argument("--splits", default="0.7,0.15,0.15",
                    help="train,valid,test fractions")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--copy-mode", default="copy", choices=["copy", "symlink"])
    args = ap.parse_args()

    random.seed(args.seed)
    tr, va, te = [float(x) for x in args.splits.split(",")]
    assert abs(tr + va + te - 1.0) < 1e-6, "splits must sum to 1.0"

    table = pd.read_csv(args.table, dtype=str, low_memory=False)
    if args.field not in table.columns:
        raise SystemExit(f"Column '{args.field}' not found. "
                         f"Available: {list(table.columns)}")

    grouping = GROUPINGS.get(args.map)  # None for passthrough

    # ---- assign each culvert to a class, collect its images ----
    # group images by culvert FIRST so we split by culvert (no leakage:
    # all photos of one culvert stay in the same split)
    class_to_culverts = defaultdict(list)   # class -> [(culvert_key, [img paths])]
    skipped_no_label = skipped_no_image = skipped_unmapped = 0
    seen_keys = set()

    for _, rec in table.iterrows():
        key = folder_key(rec)
        if key in seen_keys:      # avoid duplicate inventory rows
            continue
        seen_keys.add(key)

        raw = rec.get(args.field)
        if pd.isna(raw) or str(raw).strip() == "":
            skipped_no_label += 1
            continue
        raw = str(raw).strip()

        if grouping is not None:
            cls = grouping.get(raw)
            if cls is None:
                skipped_unmapped += 1
                continue
        else:
            cls = safe_folder_name(raw)   # passthrough: one folder per value

        imgs = list_images(os.path.join(args.images, key))
        if not imgs:
            skipped_no_image += 1
            continue

        class_to_culverts[cls].append((key, imgs))

    if not class_to_culverts:
        raise SystemExit("No labeled culverts with images found - check paths/field.")

    # ---- split per class, by culvert ----
    for split in ("train", "valid", "test"):
        for cls in class_to_culverts:
            os.makedirs(os.path.join(args.out, split, cls), exist_ok=True)

    summary = defaultdict(lambda: Counter())
    img_counts = defaultdict(lambda: Counter())

    for cls, culverts in class_to_culverts.items():
        random.shuffle(culverts)
        n = len(culverts)
        n_tr = int(n * tr)
        n_va = int(n * va)
        split_assign = ([("train", c) for c in culverts[:n_tr]]
                        + [("valid", c) for c in culverts[n_tr:n_tr + n_va]]
                        + [("test", c) for c in culverts[n_tr + n_va:]])

        for split, (key, imgs) in split_assign:
            summary[split][cls] += 1
            for src in imgs:
                fname = f"{key}__{os.path.basename(src)}"
                dst = os.path.join(args.out, split, cls, fname)
                if args.copy_mode == "copy":
                    shutil.copy2(src, dst)
                else:
                    try:
                        os.symlink(os.path.abspath(src), dst)
                    except OSError:
                        shutil.copy2(src, dst)   # Windows without symlink priv
                img_counts[split][cls] += 1

    # ---- report ----
    print(f"\nDataset built at: {args.out}")
    print(f"Field: '{args.field}'  | grouping: {args.map}")
    print(f"Skipped - no label: {skipped_no_label}, "
          f"unmapped value: {skipped_unmapped}, no image: {skipped_no_image}")
    print("\nCulverts per split/class (images in parentheses):")
    for split in ("train", "valid", "test"):
        parts = [f"{cls}={summary[split][cls]}({img_counts[split][cls]} img)"
                 for cls in sorted(class_to_culverts)]
        print(f"  {split:5s}: " + ", ".join(parts))

    # write a README documenting class meanings
    meaning = MEANINGS.get(args.map, {})
    with open(os.path.join(args.out, "CLASS_MEANINGS.txt"), "w") as f:
        f.write(f"Field: {args.field}\nGrouping: {args.map}\n\n")
        for cls in sorted(class_to_culverts):
            f.write(f"  folder '{cls}' = {meaning.get(cls, cls)}\n")
    print(f"\nWrote class meanings -> {os.path.join(args.out, 'CLASS_MEANINGS.txt')}")
    print("\nNext: python retrain.py --data <this folder> --task "
          f"{args.map if args.map in ('silting','corrosion') else '<task>'}")


if __name__ == "__main__":
    main()
