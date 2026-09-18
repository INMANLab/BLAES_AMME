#!/usr/bin/env python
"""Build one-sec-stim filtered combined CSVs for AMME Timing AND
AMME Duration patients, across encoding (phase1) AND retrieval (phase3),
for all 3 modalities (power, coherence, pac).

This sits OUTSIDE the basic combiner code (no compiler scripts are
modified). It re-aggregates raw rows for the AMME patients with the
correct stimulation-column filter, then merges them with the existing
non-AMME rows from the existing combined CSVs.

Filter rules (per user):
  - AMME Timing (amyg045, 046, 048, 054, 057, 059, 061, 066, 072):
      keep stimulation in {0, 3}.
      stimulation == 0 -> trial_type='nostim'
      stimulation == 3 -> trial_type='stim'  (one-sec After-stim)
      Drop stimulation in {1, 2} (Before-stim, During-stim).
  - AMME Duration (amyg030, 033, 034, 037):
      keep stimulation in {0, 1}.
      stimulation == 0 -> trial_type='nostim'
      stimulation == 1 -> trial_type='stim'  (one-sec stim)
      Drop stimulation == 3 (3-sec stim).
  - All other patients: rows passed through from existing combined CSV.

Encoding source CSV (already AMME-Timing-filtered for the 9 patients):
  outputs/csvs/combined_encoding_<modality>_amme_timing_filtered_mlmr_input.csv
  -> we drop AMME Duration rows from it and re-add them filtered.

Retrieval source CSV (no AMME filter applied yet):
  outputs/csvs/combined_retrieval_<modality>_all_mlmr_input.csv
  -> we drop ALL AMME (Timing + Duration) rows from it and re-add them
     filtered.

Outputs:
  outputs/csvs/onesec/combined_encoding_<modality>_onesec_mlmr_input.csv
  outputs/csvs/onesec/combined_retrieval_<modality>_onesec_mlmr_input.csv
"""

import glob
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/martinahollearn/Library/CloudStorage/Box-Box/InmanLab/"
            "BLAES_data/dissertation/AMME_BLAES")
CSV_DIR = ROOT / "outputs" / "csvs"
OUT_DIR = CSV_DIR / "onesec"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RAW_DIR = Path("/Users/martinahollearn/Library/CloudStorage/Box-Box/InmanLab/"
               "BLAES_data/dissertation/LFP_analyses/Results_CSVOutput")

AMME_TIMING = [f"amyg{n:03d}"
               for n in (45, 46, 48, 54, 57, 59, 61, 66, 72)]
AMME_DURATION = ["amyg030", "amyg033", "amyg034", "amyg037"]
AMME_ALL = AMME_TIMING + AMME_DURATION

KEEP_TIMING = {0, 3}
KEEP_DURATION = {0, 1}

MODALITY_MEASURE = {"power": "Power", "coherence": "Coherence", "pac": "PAC"}


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
    s = str(region)
    parts = [p.strip() for p in s.split("_") if p.strip()]
    parts = ["EC" if p == "ER" else p for p in parts]
    if len(parts) == 2:
        parts = sorted(parts)
    return "_".join(parts)


# Phase3 (retrieval) trial_type strings -> our normalized {nostim, stim} labels.
# Anything not in these maps (e.g., "new", "Before stim", "During stim",
# "3s stim") is dropped.
P3_TIMING_TRIAL_MAP = {"nostim": "nostim", "After stim": "stim"}
P3_DURATION_TRIAL_MAP = {"nostim": "nostim", "1s stim": "stim"}


def filter_amme_source_phase1(df, measure_name, patient):
    """Phase1 (encoding): filter on numeric `stimulation` column."""
    if "stimulation" not in df.columns or "test_yes_or_no" not in df.columns:
        return None
    df = df.copy()
    df["stimulation"] = pd.to_numeric(df["stimulation"], errors="coerce")
    if patient in AMME_TIMING:
        df = df[df["stimulation"].isin(KEEP_TIMING)]
        df["trial_type"] = np.where(df["stimulation"] == 0, "nostim", "stim")
    elif patient in AMME_DURATION:
        df = df[df["stimulation"].isin(KEEP_DURATION)]
        df["trial_type"] = np.where(df["stimulation"] == 0, "nostim", "stim")
    else:
        return None
    df = df[df["test_yes_or_no"].isin(["yes", "no"])]
    if df.empty:
        return None
    df["yes_or_no"] = df["test_yes_or_no"]
    return _finalize(df, measure_name)


def filter_amme_source_phase3(df, measure_name, patient):
    """Phase3 (retrieval): filter on string `trial_type` column.

    AMME phase3 files have no numeric `stimulation` column. Trial types are:
      AMME Timing:    {nostim, Before stim, After stim, During stim, new}
      AMME Duration:  {nostim, 1s stim, 3s stim, new}
    Memory response is in `yes_or_no` (not `test_yes_or_no`). 'new' = foil.
    """
    if "trial_type" not in df.columns or "yes_or_no" not in df.columns:
        return None
    df = df.copy()
    if patient in AMME_TIMING:
        trial_map = P3_TIMING_TRIAL_MAP
    elif patient in AMME_DURATION:
        trial_map = P3_DURATION_TRIAL_MAP
    else:
        return None
    df = df[df["trial_type"].isin(trial_map)]
    if df.empty:
        return None
    df["trial_type"] = df["trial_type"].map(trial_map)
    df = df[df["yes_or_no"].isin(["yes", "no"])]
    if df.empty:
        return None
    return _finalize(df, measure_name)


def _finalize(df, measure_name):
    df["Region"] = df["Region"].map(normalize_region)
    df = df[~df["Region"].map(is_pnas_or_same_region)]
    diff_cols = sorted(
        [c for c in df.columns if c.startswith("diff_Freq_")],
        key=lambda c: float(c.replace("diff_Freq_", "")),
    )
    if not diff_cols:
        # PAC phase3 only has pre/post; compute diff = post - pre.
        pre_cols = sorted(
            [c for c in df.columns if c.startswith("pre_Freq_")],
            key=lambda c: float(c.replace("pre_Freq_", "")),
        )
        post_cols = sorted(
            [c for c in df.columns if c.startswith("post_Freq_")],
            key=lambda c: float(c.replace("post_Freq_", "")),
        )
        if not pre_cols or not post_cols:
            return None
        shared = sorted(
            set(c.replace("pre_Freq_", "") for c in pre_cols)
            & set(c.replace("post_Freq_", "") for c in post_cols),
            key=float,
        )
        if not shared:
            return None
        diff_cols = []
        for f in shared:
            diff_col = f"diff_Freq_{f}"
            df[diff_col] = df[f"post_Freq_{f}"] - df[f"pre_Freq_{f}"]
            diff_cols.append(diff_col)
    df["Measure"] = measure_name
    return df[["Measure", "Patient", "Region", "trial_type",
               "yes_or_no"] + diff_cols].copy()


def raw_pattern(modality, phase, patient):
    """Return the expected raw filename for one (patient, phase, modality)."""
    if modality == "power":
        return RAW_DIR / f"Data_MartinaChannelsamyg{patient[-3:]}phase{phase}MeasurePower.csv"
    if modality == "coherence":
        return RAW_DIR / f"Data_MartinaChannelsamyg{patient[-3:]}phase{phase}MeasureCoherency.csv"
    if modality == "pac":
        return RAW_DIR / f"Data_amyg{patient[-3:]}phase{phase}MeasurePAC_TG.csv"
    raise ValueError(modality)


def collect_amme_filtered(modality, phase, patients):
    """Read raw files for the given AMME patients, filter, return stacked DF."""
    measure = MODALITY_MEASURE[modality]
    filter_fn = (filter_amme_source_phase1 if phase == 1
                 else filter_amme_source_phase3)
    pieces = []
    for patient in patients:
        fp = raw_pattern(modality, phase, patient)
        if not fp.exists():
            print(f"  {modality} phase{phase}: SKIP {patient} - file not found ({fp.name})")
            continue
        df = pd.read_csv(fp)
        if "Patient" not in df.columns:
            print(f"  {modality} phase{phase}: SKIP {patient} - no Patient column")
            continue
        out = filter_fn(df, measure, patient)
        if out is None or out.empty:
            print(f"  {modality} phase{phase}: SKIP {patient} - no usable rows")
            continue
        n_stim = (out["trial_type"] == "stim").sum()
        n_nostim = (out["trial_type"] == "nostim").sum()
        print(f"  {modality} phase{phase}: {patient} -> kept {len(out):,} rows "
              f"(stim={n_stim:,}, nostim={n_nostim:,})")
        pieces.append(out)
    if not pieces:
        return pd.DataFrame()
    return pd.concat(pieces, ignore_index=True)


def build_encoding_onesec(modality):
    """Encoding onesec = existing AMME-Timing-filtered CSV
    minus AMME Duration rows + correctly re-aggregated AMME Duration rows."""
    src = CSV_DIR / f"combined_encoding_{modality}_amme_timing_filtered_mlmr_input.csv"
    print(f"\n=== ENCODING {modality} ===")
    print(f"  Source: {src.name}")
    base = pd.read_csv(src)
    n_before = len(base)
    n_dur_before = base["Patient"].isin(AMME_DURATION).sum()
    print(f"  base rows: {n_before:,} | AMME Duration rows in base: {n_dur_before:,}")

    # Drop AMME Duration rows from base; AMME Timing already correctly filtered.
    keep = base[~base["Patient"].isin(AMME_DURATION)].copy()

    # Re-aggregate AMME Duration patients from raw with correct filter.
    dur_filtered = collect_amme_filtered(modality, phase=1,
                                         patients=AMME_DURATION)
    if not dur_filtered.empty:
        dur_filtered = dur_filtered.reindex(columns=keep.columns)

    out = pd.concat([keep, dur_filtered], ignore_index=True)
    out = out.sort_values(
        ["Patient", "Region", "trial_type", "yes_or_no"]
    ).reset_index(drop=True)

    out_path = OUT_DIR / f"combined_encoding_{modality}_onesec_mlmr_input.csv"
    out.to_csv(out_path, index=False)
    n_after = len(out)
    n_dur_after = out["Patient"].isin(AMME_DURATION).sum()
    n_tim_after = out["Patient"].isin(AMME_TIMING).sum()
    print(f"  rows: {n_before:,} -> {n_after:,} | "
          f"AMME Timing rows: {n_tim_after:,} | "
          f"AMME Duration rows: {n_dur_before:,} -> {n_dur_after:,}")
    print(f"  saved {out_path.name}")


def build_retrieval_onesec(modality):
    """Retrieval onesec = existing _all_ CSV minus all AMME rows
    + correctly re-aggregated AMME (Timing + Duration) rows."""
    src = CSV_DIR / f"combined_retrieval_{modality}_all_mlmr_input.csv"
    print(f"\n=== RETRIEVAL {modality} ===")
    print(f"  Source: {src.name}")
    base = pd.read_csv(src)
    n_before = len(base)
    n_amme_before = base["Patient"].isin(AMME_ALL).sum()
    print(f"  base rows: {n_before:,} | AMME rows in base: {n_amme_before:,}")

    # Drop ALL AMME rows from base (retrieval combiner had no filter at all).
    keep = base[~base["Patient"].isin(AMME_ALL)].copy()

    # Re-aggregate AMME Timing + Duration patients from raw with correct filter.
    amme_filtered = collect_amme_filtered(modality, phase=3,
                                          patients=AMME_ALL)
    if not amme_filtered.empty:
        amme_filtered = amme_filtered.reindex(columns=keep.columns)

    out = pd.concat([keep, amme_filtered], ignore_index=True)
    out = out.sort_values(
        ["Patient", "Region", "trial_type", "yes_or_no"]
    ).reset_index(drop=True)

    out_path = OUT_DIR / f"combined_retrieval_{modality}_onesec_mlmr_input.csv"
    out.to_csv(out_path, index=False)
    n_after = len(out)
    n_amme_after = out["Patient"].isin(AMME_ALL).sum()
    print(f"  rows: {n_before:,} -> {n_after:,} | "
          f"AMME rows: {n_amme_before:,} -> {n_amme_after:,}")
    print(f"  saved {out_path.name}")


if __name__ == "__main__":
    for m in ("power", "coherence", "pac"):
        build_encoding_onesec(m)
    for m in ("power", "coherence", "pac"):
        build_retrieval_onesec(m)
    print("\n===== one-sec filter build complete =====")
