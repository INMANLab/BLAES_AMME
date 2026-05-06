#!/usr/bin/env python
"""
Build AMME-timing-filtered combined encoding CSVs.

For AMME timing patients (amyg045-amyg072) the original combined encoding
CSVs collapse Before stim (stimulation=1), During stim (=2), and After stim
(=3) all into trial_type='stim'. This script rebuilds the AMME portion of
each combined CSV after dropping Before stim and During stim trials, so
trial_type='stim' for AMME patients only reflects After stim (=3).

For non-AMME patients (BLAES), rows are passed through unchanged.

Outputs:
  outputs/csvs/combined_encoding_<modality>_amme_timing_filtered_mlmr_input.csv
"""

import glob
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/martinahollearn/Library/CloudStorage/Box-Box/InmanLab/"
            "BLAES_data/dissertation/AMME_BLAES")
CSV_DIR = ROOT / "outputs" / "csvs"

AMME_SOURCE_DIR = Path("/Users/martinahollearn/Library/CloudStorage/Box-Box/"
                       "InmanLab/AMME_Data_Emory/AMME_Data/"
                       "LFP_analyses_Martina/Results_CSVOutput/Phase1")
PAC_SOURCE_DIR  = Path("/Users/martinahollearn/Library/CloudStorage/Box-Box/"
                       "InmanLab/BLAES_data/dissertation/LFP_analyses/"
                       "Results_CSVOutput")

AMME_PATIENTS = [f"amyg{n:03d}" for n in
                 (45, 46, 48, 54, 57, 59, 61, 66, 72)]

# Stimulation codes to KEEP for AMME (drop Before=1 and During=2)
KEEP_STIMULATION = {0, 3}


def is_pnas_or_same_region(region):
    s = str(region)
    if "PNAS" in s.upper():
        return True
    if "_" in s:
        parts = s.split("_")
        if len(parts) == 2 and parts[0] == parts[1]:
            return True
    return False


def normalize_region(region):
    """For pair regions, sort tokens alphabetically (CA_DG, not DG_CA).
    ER -> EC normalization to match pipeline."""
    s = str(region)
    parts = [p.strip() for p in s.split("_") if p.strip()]
    parts = ["EC" if p == "ER" else p for p in parts]
    if len(parts) == 2:
        parts = sorted(parts)
    return "_".join(parts)


def filter_amme_source(df, measure_name):
    """Apply the AMME-timing filter to a per-patient source DF.
    Returns trial-level DF with cols Measure, Patient, Region, trial_type,
    yes_or_no, diff_Freq_*."""
    if "stimulation" not in df.columns or "test_yes_or_no" not in df.columns:
        return None

    df = df.copy()
    df["stimulation"] = pd.to_numeric(df["stimulation"], errors="coerce")
    df = df[df["stimulation"].isin(KEEP_STIMULATION)]
    df = df[df["test_yes_or_no"].isin(["yes", "no"])]
    if df.empty:
        return None

    df["yes_or_no"] = df["test_yes_or_no"]
    df["trial_type"] = np.where(df["stimulation"] == 0, "nostim", "stim")
    df["Region"] = df["Region"].map(normalize_region)
    df = df[~df["Region"].map(is_pnas_or_same_region)]

    diff_cols = sorted([c for c in df.columns if c.startswith("diff_Freq_")],
                       key=lambda c: float(c.replace("diff_Freq_", "")))
    if not diff_cols:
        return None

    df["Measure"] = measure_name
    out = df[["Measure", "Patient", "Region", "trial_type", "yes_or_no"]
             + diff_cols].copy()
    return out


def collect_amme_filtered(modality):
    """Read the per-patient AMME source files for a modality, filter, return
    one stacked DF."""
    if modality == "power":
        pattern = "Data_JoeChannelsamyg*phase1MeasurePower.csv"
        source_dir = AMME_SOURCE_DIR
        measure = "Power"
    elif modality == "coherence":
        pattern = "Data_JoeChannelsamyg*phase1MeasureCoherency.csv"
        source_dir = AMME_SOURCE_DIR
        measure = "Coherence"
    elif modality == "pac":
        pattern = "Data_amyg*phase1MeasurePAC_TG.csv"
        source_dir = PAC_SOURCE_DIR
        measure = "PAC"
    else:
        raise ValueError(modality)

    files = sorted(glob.glob(str(source_dir / pattern)))
    pieces = []
    for fp in files:
        df = pd.read_csv(fp)
        if "Patient" not in df.columns:
            continue
        patient = str(df["Patient"].iloc[0])
        if patient not in AMME_PATIENTS:
            continue
        out = filter_amme_source(df, measure)
        if out is None or out.empty:
            print(f"  {modality}: skipped {Path(fp).name} (no usable rows)")
            continue
        n_keep = (out["trial_type"] == "stim").sum()
        n_drop = len(df) - len(out)
        print(f"  {modality}: {patient} -> kept {len(out)} rows "
              f"({n_keep} after-stim + {len(out)-n_keep} nostim), "
              f"dropped {n_drop} rows from source")
        pieces.append(out)

    if not pieces:
        return pd.DataFrame()
    return pd.concat(pieces, ignore_index=True)


def build_filtered_combined(modality):
    src = CSV_DIR / f"combined_encoding_{modality}_all_mlmr_input.csv"
    print(f"\n=== {modality} ===")
    print(f"  Reading existing combined: {src.name}")
    combined = pd.read_csv(src)
    n_before = len(combined)
    n_amme_rows_before = combined["Patient"].isin(AMME_PATIENTS).sum()
    print(f"  combined rows: {n_before:,}; AMME rows: {n_amme_rows_before:,}")

    # Drop existing AMME rows
    non_amme = combined[~combined["Patient"].isin(AMME_PATIENTS)].copy()

    # Build replacement AMME rows
    amme_filtered = collect_amme_filtered(modality)
    if amme_filtered.empty:
        print("  No AMME source data found; output will lack AMME patients.")
    else:
        amme_filtered = amme_filtered[non_amme.columns.intersection(
            amme_filtered.columns)].reindex(columns=non_amme.columns)

    # Concat
    out_df = pd.concat([non_amme, amme_filtered], ignore_index=True)
    out_df = out_df.sort_values(
        ["Patient", "Region", "trial_type", "yes_or_no"]
    ).reset_index(drop=True)

    out_path = CSV_DIR / (f"combined_encoding_{modality}"
                          f"_amme_timing_filtered_mlmr_input.csv")
    out_df.to_csv(out_path, index=False)
    n_after = len(out_df)
    n_amme_rows_after = out_df["Patient"].isin(AMME_PATIENTS).sum()
    print(f"  rows: {n_before:,} -> {n_after:,} | "
          f"AMME rows: {n_amme_rows_before:,} -> {n_amme_rows_after:,}")
    print(f"  saved {out_path.name}")


if __name__ == "__main__":
    for m in ("power", "coherence", "pac"):
        build_filtered_combined(m)
    print("\n===== AMME timing filter build complete =====")
