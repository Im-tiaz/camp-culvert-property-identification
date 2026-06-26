"""
============================================================================
CAMP Culvert ML Pipeline - DIMENSION CHECK (rule-based, no ML)
============================================================================
Validates Span / Rise values recorded in nmdot_table_fixed.csv against the
legal/standard manufactured dimensions for:

  - Reinforced Concrete Pipe (RCP)        : standard diameters
  - Corrugated Steel Circular Pipe (CMP)  : standard diameters
  - Corrugated Steel Pipe-Arch            : standard span x rise pairs

Primary key: "Culvert ID"

Flags raised per record:
  OK                  measurement matches a standard size (within tolerance)
  INVALID_DIMENSION   measurement does not match any standard size
  SPAN_RISE_MISMATCH  circular culvert where span != rise
  MISSING_DATA        span/rise/shape/material missing -> cannot check
  NOT_CHECKED         shape/material combination has no standard list
                      (box, elliptical, plastic, wood, ...)

USAGE:
    python dimension_check.py --table nmdot_table_fixed.csv
    python dimension_check.py --table nmdot_table_fixed.csv --tolerance 1.0
============================================================================
"""

import argparse
import os
from datetime import datetime

import numpy as np
import pandas as pd

# ===========================================================================
# STANDARD DIMENSION TABLES  (inches)
# ===========================================================================

# --- Reinforced Concrete Pipe: inner diameter -> (outer diameter, wall) ---
RCP_STANDARDS = {
    12: {"OD": 16.0,  "wall": 2.0},
    15: {"OD": 19.5,  "wall": 2.25},
    18: {"OD": 23.0,  "wall": 2.5},
    24: {"OD": 30.0,  "wall": 3.0},
    30: {"OD": 37.0,  "wall": 3.5},
    36: {"OD": 44.0,  "wall": 4.0},
    42: {"OD": 52.5,  "wall": 5.25},
    48: {"OD": 59.5,  "wall": 5.75},
    54: {"OD": 66.5,  "wall": 6.25},
    60: {"OD": 73.5,  "wall": 6.75},
}
RCP_DIAMETERS = sorted(RCP_STANDARDS.keys())

# --- Corrugated Steel Circular Pipe: diameter -> end area (sq ft) ---
CMP_STANDARDS = {
    15: 1.1, 18: 1.6, 21: 2.2, 24: 2.9, 30: 4.5,
    36: 6.5, 42: 8.9, 48: 11.6, 54: 14.7, 60: 18.1,
}
CMP_DIAMETERS = sorted(CMP_STANDARDS.keys())

# --- Corrugated Steel Pipe-Arch: (span, rise) standard pairs ---
# includes the small arch equivalents from the circular table (17x13 .. 24x18)
ARCH_STANDARDS = [
    (17, 13), (21, 15), (24, 18),            # equivalents of 15", 18", 21"
    (28, 20), (35, 24), (42, 29), (49, 33),  # 24" .. 42"
    (57, 38), (64, 43), (71, 47), (73, 55),  # 48" .. 66"
    (81, 59), (87, 63), (95, 67), (103, 71), # 72" .. 90"
    (112, 75), (117, 79), (128, 83),         # 96" .. 108"
    (137, 87), (142, 91),                    # 114", 120"
]

# ===========================================================================
# COLUMN NAMES in nmdot_table_fixed.csv (display names after fix_nmdot_csv.py)
# ===========================================================================
COL_ID = "Culvert ID"
COL_SHAPE = "Culvert Shape"
COL_MATERIAL = "Material"
COL_SPAN = "Span (inches)"
COL_RISE = "Rise (inches)"
COL_OBJECTID = "OBJECTID"
COL_COUNT = "Number of Culverts"
COL_MBWIDTH = "Width of Multi-Barrel Culverts (ft)"
COL_HACC = "Horizontal Accuracy (m)"
GPS_THRESHOLD_M = 1.5   # hard requirement: horizontal accuracy must be <= 1.5 m


def to_num(v):
    try:
        f = float(str(v).strip())
        return f if np.isfinite(f) else None
    except (ValueError, TypeError):
        return None


def closest(value, options):
    """Return (closest_standard, abs_difference)."""
    arr = np.asarray(options, dtype=float)
    i = int(np.argmin(np.abs(arr - value)))
    return options[i], abs(float(arr[i]) - value)


def closest_pair(span, rise, pairs):
    """Closest (span, rise) standard pair by max axis error."""
    best, best_err = None, float("inf")
    for s, r in pairs:
        err = max(abs(s - span), abs(r - rise))
        if err < best_err:
            best, best_err = (s, r), err
    return best, best_err


def norm(s):
    return str(s).strip().lower() if pd.notna(s) else ""


def check_record(shape, material, span, rise, tol):
    """
    Returns (flag, detail, closest_standard_str)
    """
    shape_n, mat_n = norm(shape), norm(material)

    # ---- missing data ----
    if span is None and rise is None:
        return "MISSING_DATA", "Span and Rise both missing", ""

    is_circular = "circular" in shape_n
    is_arch = "arch" in shape_n
    is_concrete = "concrete" in mat_n
    is_metal = ("metal" in mat_n) or ("steel" in mat_n) or ("cmp" in mat_n)

    # ---- circular culverts: diameter check ----
    if is_circular:
        diam = span if span is not None else rise
        if diam is None:
            return "MISSING_DATA", "No diameter recorded", ""

        # span and rise should agree for a circle
        if span is not None and rise is not None and abs(span - rise) > tol:
            return ("SPAN_RISE_MISMATCH",
                    f"Circular but span={span:g} != rise={rise:g}", "")

        if is_concrete:
            # special check: did the inspector measure the OUTSIDE diameter?
            # A value far from every standard INNER diameter but matching a
            # standard RCP OUTER diameter strongly suggests an OD measurement.
            id_std, id_err = closest(diam, RCP_DIAMETERS)
            od_hits = [(d, s["OD"]) for d, s in RCP_STANDARDS.items()
                       if abs(s["OD"] - diam) <= tol]
            if id_err > tol and od_hits:
                d_in, od = od_hits[0]
                return ("POSSIBLE_OD_MEASURED",
                        f'{diam:g}" matches the OUTSIDE diameter of a {d_in}" RCP '
                        f'(OD {od}") - true inner diameter is likely {d_in}"',
                        f'RCP {d_in}" (OD {od}")')
            std, err = id_std, id_err
            label = f'RCP {std}" (OD {RCP_STANDARDS[std]["OD"]}", wall {RCP_STANDARDS[std]["wall"]}")'
        elif is_metal:
            std, err = closest(diam, CMP_DIAMETERS)
            label = f'CMP {std}" (end area {CMP_STANDARDS[std]} sq ft)'
        else:
            # unknown material: accept if it matches EITHER standard list
            std, err = closest(diam, sorted(set(RCP_DIAMETERS + CMP_DIAMETERS)))
            label = f'{std}" (material unknown)'

        if err <= tol:
            return "OK", f'Diameter {diam:g}" matches standard {std}"', label
        return ("INVALID_DIMENSION",
                f'Diameter {diam:g}" is not a standard size (closest: {std}", off by {err:g}")',
                label)

    # ---- arch culverts: span x rise pair check ----
    if is_arch:
        if span is None or rise is None:
            return "MISSING_DATA", "Arch needs both span and rise", ""
        if not is_metal and mat_n:
            # arch standards provided are for corrugated steel only
            if is_concrete:
                return "NOT_CHECKED", "Concrete arch: no standard table provided", ""
        (s, r), err = closest_pair(span, rise, ARCH_STANDARDS)
        label = f'Pipe-Arch {s}" x {r}"'
        if err <= tol:
            return "OK", f'{span:g}" x {rise:g}" matches standard {s}" x {r}"', label
        return ("INVALID_DIMENSION",
                f'{span:g}" x {rise:g}" not a standard pair (closest: {s}" x {r}", off by {err:g}")',
                label)

    # ---- everything else: box, elliptical, other, blank ----
    return "NOT_CHECKED", f"No standard table for shape='{shape}'", ""


def main():
    ap = argparse.ArgumentParser(description="CAMP dimension validity check")
    ap.add_argument("--table", required=True, help="Path to nmdot_table_fixed.csv")
    ap.add_argument("--tolerance", type=float, default=1.0,
                    help="Allowed deviation in inches (default 1.0)")
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.table, dtype=str, low_memory=False)
    print(f"Loaded {len(df)} records from {args.table}")

    missing_cols = [c for c in (COL_ID, COL_SHAPE, COL_MATERIAL, COL_SPAN, COL_RISE)
                    if c not in df.columns]
    if missing_cols:
        print(f"WARNING: columns not found: {missing_cols}")
        print(f"Available columns: {list(df.columns)}")

    rows = []
    for _, rec in df.iterrows():
        cid = str(rec.get(COL_ID, "")).strip()
        oid = str(rec.get(COL_OBJECTID, "")).strip()
        if not cid or cid.lower() == "nan":
            cid = f"OBJECTID_{oid}"

        span = to_num(rec.get(COL_SPAN))
        rise = to_num(rec.get(COL_RISE))
        shape = rec.get(COL_SHAPE, "")
        material = rec.get(COL_MATERIAL, "")

        flag, detail, std = check_record(shape, material, span, rise, args.tolerance)

        # --- Rule: multi-barrel culvert (count > 2) should record a width ---
        count_v = to_num(rec.get(COL_COUNT))
        width_v = to_num(rec.get(COL_MBWIDTH))
        if count_v is not None and count_v > 2:
            mb_flag = "OK" if (width_v is not None and width_v > 0) else "MISSING_WIDTH"
        else:
            mb_flag = "NOT_APPLICABLE"

        # --- Rule: horizontal GPS accuracy must meet the 1.5 m standard ---
        hacc_v = to_num(rec.get(COL_HACC))
        if hacc_v is None:
            gps_flag = "NO_GPS_DATA"
        elif hacc_v > GPS_THRESHOLD_M:
            gps_flag = "EXCEEDS_1.5M"
        else:
            gps_flag = "OK"

        rows.append({
            "Culvert ID": cid,
            "OBJECTID": oid,
            "Culvert Shape": shape,
            "Material": material,
            "Span (inches)": span,
            "Rise (inches)": rise,
            "Dimension_Flag": flag,
            "Dimension_Detail": detail,
            "Closest_Standard": std,
            "MultiBarrelWidth_Flag": mb_flag,
            "GPSAccuracy_Flag": gps_flag,
        })

    out = pd.DataFrame(rows)

    out_dir = args.output_dir or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y_%m_%d")
    out_csv = os.path.join(out_dir, f"dimension_check_{stamp}.csv")
    out.to_csv(out_csv, index=False)

    print(f"\nSaved -> {out_csv}\n")
    print("Flag summary:")
    print(out["Dimension_Flag"].value_counts().to_string())
    n_bad = (out["Dimension_Flag"].isin(
        ["INVALID_DIMENSION", "SPAN_RISE_MISMATCH", "POSSIBLE_OD_MEASURED"])).sum()
    print(f"\n{n_bad} records flagged for QC review.")
    print("\nMulti-barrel width flag summary:")
    print(out["MultiBarrelWidth_Flag"].value_counts().to_string())
    print("\nGPS accuracy flag summary (1.5 m standard):")
    print(out["GPSAccuracy_Flag"].value_counts().to_string())
    n_gps = (out["GPSAccuracy_Flag"] == "EXCEEDS_1.5M").sum()
    pct = 100.0 * n_gps / max(len(out), 1)
    print(f"  {n_gps} of {len(out)} culverts ({pct:.0f}%) exceed the 1.5 m "
          f"horizontal-accuracy standard.")


if __name__ == "__main__":
    main()
