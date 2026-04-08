#!/usr/bin/env python3
"""Build trial-level PAC MLM input CSVs from raw PAC_TG files.

PAC_TG = theta-phase x gamma-amplitude coupling.
Amplitude frequencies in the raw data range from 30-100 Hz.

Encoding bands:
  - Theta phase x Slow gamma amplitude (30-50 Hz)
  - Theta phase x HFA amplitude (55-100 Hz)

Retrieval bands:
  - Theta phase x Slow gamma amplitude (30-50 Hz) only
"""

import glob
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = Path(__file__).resolve().parent
DATA_PATH = Path(
    "/Users/martinahollearn/Library/CloudStorage/Box-Box/InmanLab/"
    "BLAES_data/dissertation/LFP_analyses/Results_CSVOutput"
)
CSV_OUT = BASE / "outputs" / "csvs"

ENCODING_FILES = sorted(glob.glob(str(DATA_PATH / "*phase1*PAC_TG*.csv")))
RETRIEVAL_FILES = sorted(glob.glob(str(DATA_PATH / "*phase3*PAC_TG*.csv")))

ENCODING_EXCLUSIONS = {"BLA": {"amyg066"}}


def is_same_region(region):
    parts = str(region).split("_")
    return len(parts) == 2 and parts[0] == parts[1]

def has_excluded_token(region):
    return "PNAS" in str(region).upper()

def normalize_region_label(region):
    parts = [p.strip() for p in str(region).split("_") if p.strip()]
    for i, p in enumerate(parts):
        if p == "ER":
            parts[i] = "EC"
    if len(parts) == 2:
        parts = sorted(parts)
    return "_".join(parts)

def sorted_freq_cols(df, prefix):
    cols = [c for c in df.columns if c.startswith(prefix)]
    return sorted(cols, key=lambda c: float(c.replace(prefix, "")))


# ---------------------------------------------------------------------------
# BLAES retrieval helpers (mirroring the existing retrieval PAC loader)
# ---------------------------------------------------------------------------
def normalize_stimulation_blaes(x):
    if pd.isna(x):
        return None
    if isinstance(x, str):
        x = x.strip().lower()
    if x in {0, "0", 0.0, False, "false", "nostim", "no_stim", "no stim"}:
        return "nostim"
    if x in {1, "1", 1.0, True, "true", "stim"}:
        return "stim"
    return None

def normalize_memory_condition_blaes(x):
    if pd.isna(x):
        return None
    s = str(x).strip().lower()
    if s in {"targ", "old"}:
        return "old"
    if s in {"new", "foil", "lure"}:
        return "new"
    return None

def normalize_response_blaes(x):
    if pd.isna(x):
        return None
    s = str(x).strip().lower()
    if s in {"old", "yes"}:
        return "old"
    if s in {"new", "no"}:
        return "new"
    old_codes = {"78", "66", "37"}
    new_codes = {"67", "86", "39"}
    if s in old_codes:
        return "old"
    if s in new_codes:
        return "new"
    try:
        n = int(float(s))
        if str(n) in old_codes:
            return "old"
        if str(n) in new_codes:
            return "new"
    except (ValueError, TypeError):
        pass
    return None

def normalize_trial_type_amme(ttype):
    if isinstance(ttype, str):
        if ttype.lower() == "nostim":
            return "nostim"
        elif "stim" in ttype.lower():
            return "stim"
    return None


# ---------------------------------------------------------------------------
# Generic PAC file loader
# ---------------------------------------------------------------------------
def load_pac_files(file_list, phase_name, exclusions=None):
    """Load raw PAC CSVs, return a trial-level DataFrame."""
    frames = []
    is_encoding = phase_name.lower() == "encoding"

    for pac_file in file_list:
        df = pd.read_csv(pac_file)

        if "Patient" not in df.columns or "Region" not in df.columns:
            continue

        df = df.copy()

        # --- Determine trial_type and yes_or_no based on phase + cohort ---
        if is_encoding:
            # Encoding: stimulation + ret_response (BLAES) or test_yes_or_no (AMME)
            if "stimulation" not in df.columns:
                continue
            df["stimulation"] = pd.to_numeric(df["stimulation"], errors="coerce")
            df = df[df["stimulation"].isin([0, 1])].copy()

            if "ret_response" in df.columns:
                memory_map = {"Old": "yes", "New": "no"}
                df["yes_or_no"] = df["ret_response"].map(memory_map)
            elif "test_yes_or_no" in df.columns:
                df["yes_or_no"] = df["test_yes_or_no"].copy()
                if "test_trial_type" in df.columns:
                    df["test_trial_type"] = df["test_trial_type"].apply(normalize_trial_type_amme)
            else:
                continue

            df = df[df["yes_or_no"].isin(["yes", "no"])].copy()
            if df.empty:
                continue
            df["trial_type"] = np.where(df["stimulation"] == 1, "stim", "nostim")

        else:
            # Retrieval: more complex logic
            if "stimulation" in df.columns and "response" in df.columns:
                # BLAES retrieval
                if "trial_type" in df.columns:
                    df["trial_type_raw"] = df["trial_type"]
                df["trial_type"] = df["stimulation"].apply(normalize_stimulation_blaes)
                df = df[df["trial_type"].notnull()].copy()
                if df.empty or df["trial_type"].nunique() < 2:
                    continue

                # Add memory condition
                response_col = "response"
                if "trial_type_raw" not in df.columns and "trial_type" in df.columns:
                    df["trial_type_raw"] = df.get("trial_type_raw", df["trial_type"])

                # For BLAES retrieval: trial_type_raw has Old/New, response has the response
                raw_col = "trial_type_raw" if "trial_type_raw" in df.columns else None
                if raw_col:
                    df["memory_condition"] = df[raw_col].apply(normalize_memory_condition_blaes)
                    df["normalized_response"] = df[response_col].apply(normalize_response_blaes)
                    conditions = [
                        (df["memory_condition"] == "old") & (df["normalized_response"] == "old"),
                        (df["memory_condition"] == "old") & (df["normalized_response"] == "new"),
                    ]
                    values = ["yes", "no"]
                    df["yes_or_no"] = np.select(conditions, values, default=np.nan)
                    df = df[df["yes_or_no"].isin(["yes", "no"])].copy()
                else:
                    continue

            elif "test_yes_or_no" in df.columns:
                # AMME retrieval
                df["yes_or_no"] = df["test_yes_or_no"].copy()
                if "test_trial_type" in df.columns:
                    df["trial_type"] = df["test_trial_type"].apply(normalize_trial_type_amme)
                elif "stimulation" in df.columns:
                    df["trial_type"] = df["stimulation"].apply(normalize_stimulation_blaes)
                else:
                    continue
                df = df[df["trial_type"].notnull()].copy()
                df = df[df["yes_or_no"].isin(["yes", "no"])].copy()
                if df.empty:
                    continue
            else:
                continue

        if df.empty:
            continue

        # Filter regions
        df = df[~df["Region"].astype(str).apply(is_same_region)].copy()
        df = df[~df["Region"].astype(str).apply(has_excluded_token)].copy()

        # Apply exclusions
        if exclusions and not df.empty and "Region" in df.columns:
            for region_token, excluded_patients in exclusions.items():
                mask = (
                    df["Region"].astype(str).str.contains(region_token, na=False)
                    & df["Patient"].astype(str).isin(excluded_patients)
                )
                df = df[~mask].copy()

        if df.empty:
            continue

        # Compute diff columns
        pre_cols = sorted_freq_cols(df, "pre_Freq_")
        post_cols = sorted_freq_cols(df, "post_Freq_")
        if not pre_cols or not post_cols:
            continue

        pre_freqs = {float(c.replace("pre_Freq_", "")) for c in pre_cols}
        post_freqs = {float(c.replace("post_Freq_", "")) for c in post_cols}
        shared = sorted(pre_freqs & post_freqs)
        if not shared:
            continue

        diff_cols = []
        for freq in shared:
            freq_str = str(int(freq)) if float(freq).is_integer() else str(freq)
            pre_c = f"pre_Freq_{freq_str}"
            post_c = f"post_Freq_{freq_str}"
            diff_c = f"diff_Freq_{freq_str}"
            if pre_c in df.columns and post_c in df.columns:
                df[diff_c] = pd.to_numeric(df[post_c], errors="coerce") - pd.to_numeric(df[pre_c], errors="coerce")
                diff_cols.append(diff_c)

        if not diff_cols:
            continue

        # Normalize region label
        df["Region"] = df["Region"].apply(normalize_region_label)

        # Build export
        export = df[["Patient", "Region", "trial_type", "yes_or_no"] + diff_cols].copy()
        export.insert(0, "Measure", "PAC")
        export = export.dropna(subset=diff_cols, how="all")
        frames.append(export)

    if not frames:
        return pd.DataFrame()

    result = pd.concat(frames, ignore_index=True)
    result = result.sort_values(["Measure", "Patient", "Region", "trial_type", "yes_or_no"]).reset_index(drop=True)
    return result


def main():
    CSV_OUT.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Building PAC MLM input CSVs")
    print("=" * 60)

    # Encoding PAC
    print(f"\nEncoding: {len(ENCODING_FILES)} files")
    enc_df = load_pac_files(ENCODING_FILES, "Encoding", exclusions=ENCODING_EXCLUSIONS)
    if not enc_df.empty:
        path = CSV_OUT / "combined_encoding_pac_all_mlmr_input.csv"
        enc_df.to_csv(path, index=False)
        print(f"  Exported: {path.name} ({len(enc_df)} rows, "
              f"{enc_df['Patient'].nunique()} patients, "
              f"{enc_df['Region'].nunique()} regions)")
    else:
        print("  WARNING: No encoding PAC data found!")

    # Retrieval PAC
    print(f"\nRetrieval: {len(RETRIEVAL_FILES)} files")
    ret_df = load_pac_files(RETRIEVAL_FILES, "Retrieval")
    if not ret_df.empty:
        path = CSV_OUT / "combined_retrieval_pac_all_mlmr_input.csv"
        ret_df.to_csv(path, index=False)
        print(f"  Exported: {path.name} ({len(ret_df)} rows, "
              f"{ret_df['Patient'].nunique()} patients, "
              f"{ret_df['Region'].nunique()} regions)")
    else:
        print("  WARNING: No retrieval PAC data found!")

    print("\nDone.")


if __name__ == "__main__":
    main()
