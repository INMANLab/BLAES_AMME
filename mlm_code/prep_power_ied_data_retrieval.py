#!/usr/bin/env python
"""
Prepare RETRIEVAL trial-level power data merged with IED timing windows.

Mirror of prep_power_ied_data.py but for the retrieval phase (phase3):
  - IED timing windows at retrieval are BeforeImg and DuringImg only (no
    during-stim / after-image: there is no stimulation during retrieval).
  - The stim analog is prior_stim = whether the item was stimulated at STUDY
    (S vs NS); coded stim = 1 for S.
  - Memory = remembered (1) vs forgotten (0) for each retrieved OLD item.

Power is per-region (BLA, HPC). Extracted from raw phase3 source files with
real trial IDs (BLAES: trialIdx; AMME: number), band-averaged, and merged with
the retrieval IED timing trials on (patient, trial).

Outputs (in IED/ied_timing_memory/):
  retrieval_power_all.csv         - all retrieval power trials (from MLMR CSV)
  retrieval_power_ied_merged.csv  - IED old-item trials merged with power
"""

import os
import glob
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
OUTPUT_DIR = os.path.join(REPO_ROOT, 'IED', 'ied_timing_memory')

MLMR_CSV = os.path.join(REPO_ROOT, 'OUTPUTS', 'csvs',
    'combined_retrieval_power_all_mlmr_input.csv')
IED_TRIALS_CSV = os.path.join(OUTPUT_DIR, 'retrieval_timing_memory_trials.csv')

BLAES_DIR = ('/Users/martinahollearn/Library/CloudStorage/Box-Box/InmanLab/'
             'BLAES_data/dissertation/LFP_analyses/Results_CSVOutput')
AMME_DIR = ('/Users/martinahollearn/Library/CloudStorage/Box-Box/InmanLab/'
            'AMME_Data_Emory/AMME_Data/LFP_analyses_Martina/Results_CSVOutput')

THETA = (4, 8)
SLOW_GAMMA = (30, 55)
REGIONS = ['BLA', 'HPC']


def band_average(df, freq_cols, freqs):
    for name, (lo, hi) in [('theta', THETA), ('slow_gamma', SLOW_GAMMA)]:
        mask = (freqs >= lo) & (freqs <= hi)
        band_cols = np.array(freq_cols)[mask]
        if len(band_cols) > 0:
            df[f'pow_{name}'] = df[band_cols].astype(float).mean(axis=1)
    return df


def prep_power_all():
    """Band-average retrieval power for all trials from the MLMR CSV."""
    print("=" * 60)
    print("PART 1: All Retrieval Power Trials (MLMR CSV)")
    print("=" * 60)
    df = pd.read_csv(MLMR_CSV)
    freq_cols = sorted([c for c in df.columns if c.startswith('diff_Freq_')])
    freqs = np.array([float(c.replace('diff_Freq_', '')) for c in freq_cols])

    df = df[df['Region'].isin(REGIONS)].copy()
    df = band_average(df, freq_cols, freqs)
    df['memory'] = (df['yes_or_no'] == 'yes').astype(int)
    # trial_type stim/nostim at retrieval encodes prior_stim (stimulated at study)
    df['stim'] = (df['trial_type'] == 'stim').astype(int)
    df = df.rename(columns={'Patient': 'patient_id', 'Region': 'region'})

    out = df[['patient_id', 'region', 'stim', 'memory',
              'pow_theta', 'pow_slow_gamma']].copy()
    out_path = os.path.join(OUTPUT_DIR, 'retrieval_power_all.csv')
    out.to_csv(out_path, index=False)
    for r in REGIONS:
        sub = out[out['region'] == r]
        print(f"  {r}: {len(sub)} trials, {sub['patient_id'].nunique()} patients")
    print(f"  Total: {len(out)} -> {out_path}")
    return out


def load_raw_phase3_power():
    """Extract per-region phase3 power with real trial IDs from raw files."""
    print("\n" + "=" * 60)
    print("PART 2: Raw Phase3 Power with Trial IDs")
    print("=" * 60)
    rows = []

    # BLAES: trial id column is 'trialIdx'
    blaes_files = sorted(glob.glob(os.path.join(
        BLAES_DIR, 'Data_*phase3MeasurePower.csv')))
    blaes_files = [f for f in blaes_files if 'AllPatients' not in f]
    print(f"\n  BLAES phase3 power files: {len(blaes_files)}")
    for f in blaes_files:
        df = pd.read_csv(f)
        if 'trialIdx' not in df.columns or 'Region' not in df.columns:
            continue
        patient = df['Patient'].iloc[0]
        freq_cols = sorted([c for c in df.columns if c.startswith('diff_Freq_')])
        if not freq_cols:
            continue
        freqs = np.array([float(c.replace('diff_Freq_', '')) for c in freq_cols])
        for region in REGIONS:
            sub = df[df['Region'] == region].copy()
            if sub.empty:
                continue
            sub = band_average(sub, freq_cols, freqs)
            for _, row in sub.iterrows():
                rows.append({'patient_id': patient, 'region': region,
                             'trial': int(row['trialIdx']),
                             'pow_theta': row.get('pow_theta', np.nan),
                             'pow_slow_gamma': row.get('pow_slow_gamma', np.nan)})

    # AMME: trial id column is 'number'
    amme_files = sorted(glob.glob(os.path.join(
        AMME_DIR, '*phase3MeasurePower.csv')))
    amme_files = [f for f in amme_files
                  if 'AllPatients' not in f and 'JoeChanels' not in f]
    print(f"  AMME phase3 power files: {len(amme_files)}")
    for f in amme_files:
        df = pd.read_csv(f)
        if 'number' not in df.columns or 'Region' not in df.columns:
            continue
        patient = df['Patient'].iloc[0]
        freq_cols = sorted([c for c in df.columns if c.startswith('diff_Freq_')])
        if not freq_cols:
            continue
        freqs = np.array([float(c.replace('diff_Freq_', '')) for c in freq_cols])
        for region in REGIONS:
            sub = df[df['Region'] == region].copy()
            if sub.empty:
                continue
            sub = band_average(sub, freq_cols, freqs)
            for _, row in sub.iterrows():
                rows.append({'patient_id': patient, 'region': region,
                             'trial': int(row['number']),
                             'pow_theta': row.get('pow_theta', np.nan),
                             'pow_slow_gamma': row.get('pow_slow_gamma', np.nan)})

    power_df = pd.DataFrame(rows)
    print(f"\n  Raw phase3 power extracted: {len(power_df)} rows, "
          f"{power_df['patient_id'].nunique()} patients")
    return power_df


def merge_power_ied(power_df):
    """Merge phase3 power with retrieval IED timing trials."""
    print("\n" + "=" * 60)
    print("MERGING Retrieval Power + IED Timing")
    print("=" * 60)
    ied = pd.read_csv(IED_TRIALS_CSV)
    ied = ied.rename(columns={'Patient': 'patient_id', 'Trial': 'trial'})
    ied['ied_before_image'] = (ied['BeforeImg'] == 'Y').astype(int)
    ied['ied_during_image'] = (ied['DuringImg'] == 'Y').astype(int)
    ied['stim'] = (ied['prior_stim'] == 'S').astype(int)

    merged = power_df.merge(
        ied[['patient_id', 'trial', 'memory', 'prior_stim', 'stim',
             'ied_before_image', 'ied_during_image']],
        on=['patient_id', 'trial'], how='inner')

    print(f"\n  Merged total: {len(merged)} trial-region rows")
    for r in REGIONS:
        sub = merged[merged['region'] == r]
        if len(sub):
            print(f"  {r}: {len(sub)} trials, {sub['patient_id'].nunique()} "
                  f"patients | mem {sub['memory'].mean()*100:.1f}% | "
                  f"before {int(sub['ied_before_image'].sum())} "
                  f"during {int(sub['ied_during_image'].sum())} "
                  f"priorS {int(sub['stim'].sum())}")

    out_path = os.path.join(OUTPUT_DIR, 'retrieval_power_ied_merged.csv')
    keep = ['patient_id', 'region', 'trial', 'memory', 'prior_stim', 'stim',
            'pow_theta', 'pow_slow_gamma', 'ied_before_image', 'ied_during_image']
    merged[keep].to_csv(out_path, index=False)
    print(f"\n  Saved -> {out_path}")
    return merged


def main():
    prep_power_all()
    power_raw = load_raw_phase3_power()
    merge_power_ied(power_raw)
    print("\n===== RETRIEVAL PREP DONE =====")


if __name__ == '__main__':
    main()
