"""
============================================================================
CAMP Culvert ML Pipeline - MERGE RESULTS
============================================================================
Joins, on "Culvert ID" (the project primary key):

  1. nmdot_table_fixed.csv          (inspector-entered inventory values)
  2. camp_predictions_*.csv         (ML model predictions, from run_inference.py)
  3. dimension_check_*.csv          (rule-based span/rise flags, optional)

and adds AGREE/DISAGREE comparison columns between inspector values and
model predictions -> one master QC CSV.

NOTE on the join key: run_inference.py uses image FOLDER names as CulvertID.
Those folders were created by download_all_images.py using
safe_folder_name("Culvert ID") with OBJECTID_x fallback. This script applies
the same transformation to the table's Culvert ID so the join always matches.

USAGE:
    python merge_results.py --table nmdot_table_fixed.csv ^
                            --predictions ..\\outputs\\camp_predictions_v1_2026_06_11.csv ^
                            --dimensions  ..\\outputs\\dimension_check_2026_06_11.csv
============================================================================
"""

import argparse
import os
from datetime import datetime

import pandas as pd

COL_ID = "Culvert ID"
COL_OBJECTID = "OBJECTID"

# inventory column -> (prediction column, value mapping inspector->model labels)
COMPARISONS = {
    "Material": {
        "pred_col": "Material_Prediction",
        "map": {"concrete": "Concrete",
                "corrugated metal": "Corrugated Metal",
                "metal": "Corrugated Metal"},
    },
    "Culvert Shape": {
        "pred_col": "Shape_Prediction",
        "map": {"circular": "Circular", "box": "Box"},
    },
    "Number of Culverts": {
        "pred_col": "NumberOfCulverts_Prediction",
        "numeric": True,
    },
    # Silting: inventory categories -> binary model labels (cut at 60% = "major")
    "Silting": {
        "pred_col": "Silting_Prediction",
        "map": {"clean": "Clean/Minor (<60%)",
                "minor silting (>%10)": "Clean/Minor (<60%)",
                "10% to 30% silted": "Clean/Minor (<60%)",
                "30% to 60% silted": "Clean/Minor (<60%)",
                "60% to 90% silted": "Major (>=60%)",
                ">90% silted": "Major (>=60%)"},
    },
    # Corrosion: inventory categories -> binary model labels. "Not Known" is
    # intentionally absent -> scored UNSUPPORTED_VALUE (no ground truth).
    "Corrosion": {
        "pred_col": "Corrosion_Prediction",
        "map": {"none evident": "None/Minor",
                "minor (rusting on inside or outside)": "None/Minor",
                "moderate (rusting on inside and outside)": "Corrosion",
                "major": "Corrosion"},
    },
    # End Section: the model was trained on inlet AND outlet labels combined,
    # so a prediction is correct if it matches EITHER end. This spec compares
    # against both columns; AGREE if the prediction equals either mapped value.
    "EndSection": {
        "pred_col": "EndSection_Prediction",
        "either_cols": ["Inlet End Section Type", "Outlet End Section Type"],
        "flag_name": "EndSection_QC",
        "map": {"no end section": "None/Projecting",
                "metal end section": "Metal/Concrete Flared",
                "concrete end section": "Metal/Concrete Flared",
                "headwall": "Headwall",
                "headwall with wingwalls": "Headwall",
                "concrete slope blanket": "Slope Blanket",
                "concrete slope blanket with safety grate": "Slope Blanket"},
                # CBC removed: 4-class model can't predict it, so inventory-CBC
                # culverts score UNSUPPORTED_VALUE rather than forcing a mismatch.
    },
    # Physical Damage: binary screen. Blank inventory = no damage. Severity
    # categories all map to "Damage". "Other" is left unmapped -> UNSUPPORTED.
    "Physical Damage": {
        "pred_col": "PhysicalDamage_Prediction",
        "blank_maps_to": "No Damage",
        "map": {"minor damage (metal)": "Damage",
                "moderate damage (metal)": "Damage",
                "heavy damage (metal)": "Damage",
                "spalling, no exposed rebar (concrete)": "Damage",
                "circular concrete pipe damage": "Damage",
                "spalling and cracks on headwall/aprons": "Damage",
                "severe spalling, exposed rebar (concrete)": "Damage",
                "severe cracks on concrete (>1/4 in.)": "Damage"},
    },
    # Channel Condition: 2-class. "Channel Degrading" was dropped from the model
    # (too few examples) -> left unmapped, scores UNSUPPORTED_VALUE.
    "Channel Condition": {
        "pred_col": "ChannelCondition_Prediction",
        "map": {"good": "Good",
                "weeds and/or debris": "Weeds/Debris/Vegetated",
                "dry/heavily vegetated": "Weeds/Debris/Vegetated",
                "swampy/heavily vegetated": "Weeds/Debris/Vegetated"},
    },
    # Erosion Control: binary presence. Blank inventory = none (per inspector
    # guidance). Any control structure -> present.
    "Erosion Control": {
        "pred_col": "ErosionControl_Prediction",
        "blank_maps_to": "No Control",
        "map": {"not evident": "No Control",
                "loose rip-rap": "Control Present",
                "wire-enclosed rip-rap pad": "Control Present",
                "concrete structure": "Control Present",
                "gabions": "Control Present",
                "grouted rip-rap": "Control Present",
                "other": "Control Present"},
    },
    # Channel Type: 4-class. Dry Arroyo + Roadside/Median Ditch merged. Concrete/
    # Asphalt Lined and "Other" left unmapped -> UNSUPPORTED_VALUE (Concrete-Lined
    # was dropped from the model; Other was never a class).
    "Channel Type": {
        "pred_col": "ChannelType_Prediction",
        "map": {"dry arroyo/ephemeral": "Dry/Ephemeral",
                "roadside/median ditch": "Dry/Ephemeral",
                "no channel evident": "No Channel",
                "running water": "Running Water",
                "irrigation": "Irrigation"},
    },
}


def safe_folder_name(name: str) -> str:
    """EXACT copy of download_all_images.py logic - keeps the join consistent."""
    for ch in r'\/:*?"<>|':
        name = name.replace(ch, "_")
    return name.strip()


def folder_key_from_table(rec) -> str:
    cid = str(rec.get(COL_ID, "")).strip()
    oid = str(rec.get(COL_OBJECTID, "")).strip()
    if not cid or cid.lower() == "nan":
        return f"OBJECTID_{oid}"
    return safe_folder_name(cid)


def main():
    ap = argparse.ArgumentParser(description="Merge CAMP QC results on Culvert ID")
    ap.add_argument("--table", required=True, help="nmdot_table_fixed.csv")
    ap.add_argument("--predictions", required=True, help="camp_predictions_*.csv")
    ap.add_argument("--dimensions", default=None, help="dimension_check_*.csv (optional)")
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()

    table = pd.read_csv(args.table, dtype=str, low_memory=False)
    preds = pd.read_csv(args.predictions, keep_default_na=False, na_values=[""])
    # keep_default_na=False: do NOT let pandas turn label strings like "None"
    # into NaN. Only genuinely empty cells become NaN. Numeric comparisons
    # (count) still work via float() with try/except.
    print(f"Inventory: {len(table)} records | Predictions: {len(preds)} culverts")

    # ---- build the join key on both sides ----
    table["_JoinKey"] = table.apply(folder_key_from_table, axis=1)
    preds["_JoinKey"] = preds["CulvertID"].astype(str).str.strip()

    merged = table.merge(preds.drop(columns=["CulvertID"]),
                         on="_JoinKey", how="left")

    n_matched = merged["NumImages"].notna().sum() if "NumImages" in merged else 0
    print(f"Matched predictions for {n_matched}/{len(table)} inventory records")

    # ---- optional: dimension check flags ----
    if args.dimensions:
        dims = pd.read_csv(args.dimensions, dtype=str)
        dims["_JoinKey"] = dims["Culvert ID"].astype(str).map(safe_folder_name)
        keep = ["_JoinKey", "Dimension_Flag", "Dimension_Detail", "Closest_Standard"]
        merged = merged.merge(dims[keep].drop_duplicates("_JoinKey"),
                              on="_JoinKey", how="left")
        print("Dimension flags merged.")

    # ---- AGREE / DISAGREE comparisons ----
    for inv_col, spec in COMPARISONS.items():
        pred_col = spec["pred_col"]
        if pred_col not in merged.columns:
            continue
        either_cols = spec.get("either_cols")
        if either_cols:
            # need at least one of the inventory columns present
            cols_present = [c for c in either_cols if c in merged.columns]
            if not cols_present:
                continue
        elif inv_col not in merged.columns:
            continue
        flag_col = spec.get("flag_name", f"{inv_col.replace(' ', '')}_QC")

        def compare(row, spec=spec, inv_col=inv_col, pred_col=pred_col,
                    either_cols=either_cols):
            pred = row.get(pred_col)
            if pd.isna(pred) or str(pred) in ("", "NoDetection", "ERROR"):
                return "NO_COMPARISON"

            # numeric comparison (culvert count)
            if spec.get("numeric"):
                inv = row.get(inv_col)
                if pd.isna(inv):
                    return "NO_COMPARISON"
                try:
                    return "AGREE" if float(inv) == float(pred) else "DISAGREE"
                except ValueError:
                    return "NO_COMPARISON"

            # either-match comparison (end section: inlet OR outlet)
            if either_cols:
                mapped_vals = []
                for c in either_cols:
                    v = row.get(c)
                    if pd.isna(v) or str(v).strip() == "":
                        continue
                    m = spec["map"].get(str(v).strip().lower())
                    if m is not None:
                        mapped_vals.append(m)
                if not mapped_vals:
                    return "UNSUPPORTED_VALUE"   # both ends Unknown/Other/blank
                return "AGREE" if str(pred) in mapped_vals else "DISAGREE"

            # single-column mapped comparison
            inv = row.get(inv_col)
            if pd.isna(inv) or str(inv).strip() == "":
                # For some properties a blank inventory value is a real class
                # (erosion control / physical damage: blank = none/no-damage).
                blank_to = spec.get("blank_maps_to")
                if blank_to is not None:
                    return "AGREE" if blank_to == str(pred) else "DISAGREE"
                return "NO_COMPARISON"
            inv_mapped = spec["map"].get(str(inv).strip().lower())
            if inv_mapped is None:
                return "UNSUPPORTED_VALUE"   # e.g. elliptical shape, plastic material
            return "AGREE" if inv_mapped == str(pred) else "DISAGREE"

        merged[flag_col] = merged.apply(compare, axis=1)
        vc = merged[flag_col].value_counts()
        print(f"  {flag_col}: " + ", ".join(f"{k}={v}" for k, v in vc.items()))

    merged.drop(columns=["_JoinKey"], inplace=True)

    out_dir = args.output_dir or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y_%m_%d")
    out_csv = os.path.join(out_dir, f"camp_master_qc_{stamp}.csv")
    merged.to_csv(out_csv, index=False)
    print(f"\nSaved master QC file -> {out_csv}")

    # ---------------------------------------------------------------
    # COMPACT REVIEW FILE: only the columns a QC reviewer needs,
    # inspector value and model prediction side by side.
    # ---------------------------------------------------------------
    compact_cols = [
        # identity
        "Culvert ID", "OBJECTID", "District", "County", "Route Type",
        "Route Number", "Mile Marker", "NumImages",
        # material: inspector vs model
        "Material", "Material_Prediction", "Material_Confidence", "Material_QC",
        # shape: inspector vs model
        "Culvert Shape", "Shape_Prediction", "Shape_Confidence", "CulvertShape_QC",
        # count: inspector vs model
        "Number of Culverts", "NumberOfCulverts_Prediction", "NumberofCulverts_QC",
        # silting / corrosion: inspector vs model (now with QC flags)
        "Silting", "Silting_Prediction", "Silting_Confidence", "Silting_QC",
        "Corrosion", "Corrosion_Prediction", "Corrosion_Confidence", "Corrosion_QC",
        # end section: inspector (both ends) vs model, either-match QC
        "Inlet End Section Type", "Outlet End Section Type",
        "EndSection_Prediction", "EndSection_Confidence", "EndSection_QC",
        # dimensions
        "Span (inches)", "Rise (inches)",
        "Dimension_Flag", "Dimension_Detail", "Closest_Standard",
    ]
    available = [c for c in compact_cols if c in merged.columns]
    compact = merged[available]
    out_csv2 = os.path.join(out_dir, f"camp_qc_review_{stamp}.csv")
    compact.to_csv(out_csv2, index=False)
    print(f"Saved COMPACT review file -> {out_csv2}")
    print(f"  ({len(available)} columns instead of {len(merged.columns)})")


if __name__ == "__main__":
    main()
