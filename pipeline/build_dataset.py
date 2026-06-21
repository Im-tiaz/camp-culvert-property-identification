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
    # Silting binary, cut at 60% -> matches handbook "major silting (>60%)".
    #   0 = clean/minor (anything < 60%), 1 = major (>= 60%)
    "silting_binary": {
        "Clean": "0", "Minor Silting (>%10)": "0",
        "10% to 30% Silted": "0", "30% to 60% Silted": "0",
        "60% to 90% Silted": "1", ">90% Silted": "1",
    },
    # Silting binary, cut at 10% -> "any notable sediment?" (more balanced)
    #   0 = clean (<10%), 1 = silted (>= 10%)
    "silting_binary10": {
        "Clean": "0", "Minor Silting (>%10)": "0",
        "10% to 30% Silted": "1", "30% to 60% Silted": "1",
        "60% to 90% Silted": "1", ">90% Silted": "1",
    },
    # Corrosion -> binary (0 = none/minor, 1 = corrosion)
    "corrosion": {
        "None Evident": "0", "Minor (Rusting on Inside OR Outside)": "0",
        "Moderate (Rusting on Inside AND Outside)": "1", "Major": "1",
        # "Not Known" intentionally omitted -> those records are skipped
    },
    # End Section Type -> 5 consolidated visual classes.
    # Dropped (no coherent visual class): MDI/SDI/CDI (a drop-inlet, not an
    # end section), Unknown, Other -> omitted so they are skipped.
    "end_section": {
        "No End Section": "0",
        "Metal End Section": "1",
        "Concrete End Section": "1",          # metal/concrete flared -> same look class
        "Headwall": "2",
        "Headwall with Wingwalls": "2",
        "Concrete Slope Blanket": "3",
        "Concrete Slope Blanket With Safety Grate": "3",
        "CBC With Apron": "4",
        "CBC Without Apron": "4",
    },
    # End Section, 4-class: CBC dropped (only ~48 imgs, destabilized training).
    # CBC values are simply absent -> those culverts are skipped (unmapped),
    # giving a cleaner, more reliable model on the four common types.
    "end_section_4class": {
        "No End Section": "0",
        "Metal End Section": "1",
        "Concrete End Section": "1",
        "Headwall": "2",
        "Headwall with Wingwalls": "2",
        "Concrete Slope Blanket": "3",
        "Concrete Slope Blanket With Safety Grate": "3",
        # CBC With/Without Apron intentionally omitted
    },
    # Scour -> binary. Outlet-bed erosion. 4 severity bands collapse to
    #   0 = little/no scour, 1 = any scour (minor/major/severe).
    # Trained on ALL of a culvert's photos (outlet cannot be reliably isolated:
    # only ~9% of records have an Inspection Direction). Evaluate per-culvert.
    "scour": {
        "Little or no Scour (< ft)": "0",
        "Minor Scour (1 to 3 ft)": "1",
        "Major Scour (3 to 8 ft)": "1",
        "Severe Scour (>8 ft)": "1",
    },
    # Erosion Control -> binary present/absent. Per inspector guidance, a BLANK
    # value means "none" (see BLANK_AS_CLASS below), as does "Not Evident".
    # Any actual control structure -> present. ("Other" treated as present,
    # since it still denotes some control in this field.)
    "erosion_control": {
        "Not Evident": "0",
        "Loose Rip-Rap": "1",
        "Wire-Enclosed Rip-Rap Pad": "1",
        "Concrete Structure": "1",
        "Gabions": "1",
        "Grouted Rip-Rap": "1",
        "Other": "1",
    },
    # Channel Condition -> 3 classes. Weeds/Debris and the two Vegetated values
    # are merged (they intermatch heavily). Channel Degrading kept per request
    # but is very thin (~16) -> may need dropping if it destabilizes training.
    # Whole-channel property: all photos are appropriate (no outlet issue).
    "channel_condition": {
        "Good": "0",
        "Weeds and/or Debris": "1",
        "Dry/Heavily Vegetated": "1",
        "Swampy/Heavily Vegetated": "1",
        "Channel Degrading": "2",
    },
    # Fallback: drop the 16-example Degrading class -> clean, well-balanced
    # 2-class (Good 782 vs Weeds/Vegetated 936, ~1.2x). Use if Degrading
    # destabilizes the 3-class model (very likely, given 58x imbalance).
    "channel_condition_2class": {
        "Good": "0",
        "Weeds and/or Debris": "1",
        "Dry/Heavily Vegetated": "1",
        "Swampy/Heavily Vegetated": "1",
        # Channel Degrading omitted
    },
    # Physical Damage -> 4 severity tiers. Blank means no damage (BLANK_AS_CLASS).
    # Metal and concrete vocabularies are mapped onto one severity scale.
    # "Other" (ambiguous) is omitted -> skipped.
    "physical_damage": {
        "Minor Damage (Metal)": "1",
        "Spalling, No Exposed Rebar (Concrete)": "1",
        "Moderate Damage (Metal)": "2",
        "Circular Concrete Pipe Damage": "2",
        "Spalling and Cracks on Headwall/Aprons": "2",
        "Heavy Damage (Metal)": "3",
        "Severe Spalling, Exposed Rebar (Concrete)": "3",
        "Severe Cracks on Concrete (>1/4 in.)": "3",
    },
    # Fallback: 3 tiers (None / Minor / Major), collapsing Moderate+Severe.
    # Use if the Severe class (~120) proves too thin in the 4-tier model.
    # Binary: any damage vs none. The severity tiers (minor/moderate/severe)
    # do not separate visually in field photos, but damaged-vs-undamaged does
    # (None reached F1 0.74 in the 4-tier model). This is the deployable form.
    "physical_damage_binary": {
        "Minor Damage (Metal)": "1",
        "Spalling, No Exposed Rebar (Concrete)": "1",
        "Moderate Damage (Metal)": "1",
        "Circular Concrete Pipe Damage": "1",
        "Spalling and Cracks on Headwall/Aprons": "1",
        "Heavy Damage (Metal)": "1",
        "Severe Spalling, Exposed Rebar (Concrete)": "1",
        "Severe Cracks on Concrete (>1/4 in.)": "1",
    },
    # Channel Type -> 5 classes. Dry Arroyo and Roadside/Median Ditch are MERGED:
    # they look identical (a ditch is distinguished by road-crossing location,
    # not appearance), so keeping them separate would be unlearnable. The small
    # classes (Running Water, Irrigation, Concrete-Lined) are kept because they
    # are visually distinctive, per inspector guidance. "Other" dropped.
    "channel_type": {
        "Dry Arroyo/Ephemeral": "0",
        "Roadside/Median Ditch": "0",
        "No Channel Evident": "1",
        "Running Water": "2",
        "Irrigation": "3",
        "Concrete/Asphalt Lined": "4",
    },
    # Fallback: drop Concrete/Asphalt Lined (only 8 examples -> unmeasurable,
    # CBC-like). Keeps the four classes with enough data to train/evaluate.
    "channel_type_4class": {
        "Dry Arroyo/Ephemeral": "0",
        "Roadside/Median Ditch": "0",
        "No Channel Evident": "1",
        "Running Water": "2",
        "Irrigation": "3",
    },
}

# Groupings where a BLANK / missing field value is a real class (not skipped).
# Erosion control: blank means the inspector recorded no control -> class "0".
BLANK_AS_CLASS = {
    "erosion_control": "0",
    "physical_damage": "0",
    "physical_damage_3class": "0",
    "physical_damage_binary": "0",
}

# human-readable meaning of each folder index, written into a README per dataset
MEANINGS = {
    "silting": {"0": "Low (<10%)", "1": "Medium (10-60%)", "2": "High (>60%)"},
    "silting_binary": {"0": "Clean/Minor (<60%)", "1": "Major (>=60%)"},
    "silting_binary10": {"0": "Clean (<10%)", "1": "Silted (>=10%)"},
    "corrosion": {"0": "None/Minor", "1": "Corrosion"},
    "end_section": {"0": "None/Projecting", "1": "Metal/Concrete Flared",
                    "2": "Headwall", "3": "Slope Blanket", "4": "CBC"},
    "end_section_4class": {"0": "None/Projecting", "1": "Metal/Concrete Flared",
                           "2": "Headwall", "3": "Slope Blanket"},
    "scour": {"0": "Little/No Scour", "1": "Scour Present"},
    "erosion_control": {"0": "None", "1": "Control Present"},
    "channel_condition": {"0": "Good", "1": "Weeds/Debris/Vegetated",
                          "2": "Degrading"},
    "channel_condition_2class": {"0": "Good", "1": "Weeds/Debris/Vegetated"},
    "physical_damage": {"0": "None", "1": "Minor", "2": "Moderate", "3": "Severe"},
    "physical_damage_3class": {"0": "None", "1": "Minor", "2": "Major"},
    "physical_damage_binary": {"0": "No Damage", "1": "Damage"},
    "channel_type": {"0": "Dry/Ephemeral", "1": "No Channel", "2": "Running Water",
                     "3": "Irrigation", "4": "Concrete/Asphalt Lined"},
    "channel_type_4class": {"0": "Dry/Ephemeral", "1": "No Channel",
                            "2": "Running Water", "3": "Irrigation"},
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
    ap.add_argument("--field", required=True,
                    help="inventory column(s) for the label. Comma-separate two "
                         "columns to combine them (e.g. inlet+outlet end section): "
                         "'Outlet End Section Type,Inlet End Section Type'")
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
    fields = [f.strip() for f in args.field.split(",")]
    missing = [f for f in fields if f not in table.columns]
    if missing:
        raise SystemExit(f"Column(s) not found: {missing}. "
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

        # Resolve label: try each field in order, take the first that maps
        # to a class (for end-section, inlet and outlet describe the same
        # structure type, so either is a valid label for this culvert).
        cls = None
        any_value = False
        for fld in fields:
            raw = rec.get(fld)
            if pd.isna(raw) or str(raw).strip() == "":
                continue
            any_value = True
            raw = str(raw).strip()
            if grouping is not None:
                mapped = grouping.get(raw)
                if mapped is not None:
                    cls = mapped
                    break
            else:
                cls = safe_folder_name(raw)
                break

        if cls is None:
            # For some properties a blank field is itself a class (e.g. erosion
            # control: no value recorded -> "none"). Apply that here.
            blank_cls = BLANK_AS_CLASS.get(args.map)
            if blank_cls is not None and not any_value:
                cls = blank_cls
            elif any_value:
                skipped_unmapped += 1
                continue
            else:
                skipped_no_label += 1
                continue

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
