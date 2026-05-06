#!/usr/bin/env python
"""
Prepare trial-level power data merged with IED timing windows.

Power is per-region (BLA, HPC, CA, DG), not per-pair like coherence.
Extracts from raw source files with real trial IDs, band-averages,
and merges with IED data.

Outputs (in outputs/ied_timing_memory/):
  power_all.csv          - all encoding power trials (from MLMR CSV)
  power_ied_merged.csv   - IED trials merged with power (trial-ID matched)
"""

import os
import glob
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'outputs', 'ied_timing_memory')
os.makedirs(OUTPUT_DIR, exist_ok=True)

MLMR_CSV = os.path.join(SCRIPT_DIR, 'outputs', 'csvs',
    'combined_encoding_power_all_mlmr_input.csv')

IED_CSV = os.path.join(SCRIPT_DIR, 'IED',
    'AMMEBLAES_IEDs_trial_level_dissertation_study_usethis_cleaned_with_memory.csv')

BLAES_DIR = '/Users/martinahollearn/Library/CloudStorage/Box-Box/InmanLab/BLAES_data/dissertation/LFP_analyses/Results_CSVOutput'
AMME_DIR = '/Users/martinahollearn/Library/CloudStorage/Box-Box/InmanLab/AMME_Data_Emory/AMME_Data/LFP_analyses_Martina/Results_CSVOutput/Phase1'

THETA = (4, 8)
SLOW_GAMMA = (30, 55)
HFA = (70, 100)

REGIONS = ['BLA', 'HPC', 'CA', 'DG']
ENC_TIMING = ['BeforeImgITI', 'DuringImg', 'AfterImgITI', 'DuringStim']


def band_average(df, freq_cols, freqs):
    """Add band-averaged power columns."""
    for name, (lo, hi) in [('theta', THETA), ('slow_gamma', SLOW_GAMMA), ('hfa', HFA)]:
        mask = (freqs >= lo) & (freqs <= hi)
        band_cols = np.array(freq_cols)[mask]
        if len(band_cols) > 0:
            df[f'pow_{name}'] = df[band_cols].astype(float).mean(axis=1)
    return df


def prep_power_all():
    """Band-average power for all trials from MLMR CSV."""
    print("=" * 60)
    print("PART 1: All Power Trials (MLMR CSV)")
    print("=" * 60)

    df = pd.read_csv(MLMR_CSV)
    freq_cols = sorted([c for c in df.columns if c.startswith('diff_Freq_')])
    freqs = np.array([float(c.replace('diff_Freq_', '')) for c in freq_cols])

    df_filt = df[df['Region'].isin(REGIONS)].copy()
    df_filt = band_average(df_filt, freq_cols, freqs)

    df_filt['memory'] = (df_filt['yes_or_no'] == 'yes').astype(int)
    df_filt['stim'] = (df_filt['trial_type'] == 'stim').astype(int)
    df_filt = df_filt.rename(columns={'Patient': 'patient_id', 'Region': 'region'})

    keep = ['patient_id', 'region', 'stim', 'memory',
            'pow_theta', 'pow_slow_gamma', 'pow_hfa']
    out = df_filt[keep].copy()

    out_path = os.path.join(OUTPUT_DIR, 'power_all.csv')
    out.to_csv(out_path, index=False)

    for r in REGIONS:
        sub = out[out['region'] == r]
        print(f"  {r}: {len(sub)} trials, {sub['patient_id'].nunique()} patients")
    print(f"  Total: {len(out)} -> {out_path}")
    return out


def load_raw_power():
    """Extract ALL trial-level power with real trial IDs from raw files."""
    print("\n" + "=" * 60)
    print("PART 2: Raw Power with Trial IDs")
    print("=" * 60)

    rows = []

    # BLAES files
    blaes_files = sorted(glob.glob(os.path.join(BLAES_DIR, 'Data_*phase1MeasurePower.csv')))
    # Exclude AllPatients combined files
    blaes_files = [f for f in blaes_files if 'AllPatients' not in f]
    print(f"\n  BLAES power files: {len(blaes_files)}")

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
                rows.append({
                    'patient_id': patient,
                    'region': region,
                    'trial': int(row['trialIdx']),
                    'pow_theta': row.get('pow_theta', np.nan),
                    'pow_slow_gamma': row.get('pow_slow_gamma', np.nan),
                    'pow_hfa': row.get('pow_hfa', np.nan),
                })

    # AMME files
    amme_files = sorted(glob.glob(os.path.join(AMME_DIR, '*MeasurePower.csv')))
    print(f"  AMME power files: {len(amme_files)}")

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
                rows.append({
                    'patient_id': patient,
                    'region': region,
                    'trial': int(row['number']),
                    'pow_theta': row.get('pow_theta', np.nan),
                    'pow_slow_gamma': row.get('pow_slow_gamma', np.nan),
                    'pow_hfa': row.get('pow_hfa', np.nan),
                })

    power_df = pd.DataFrame(rows)
    print(f"\n  Raw power extracted: {len(power_df)} rows, "
          f"{power_df['patient_id'].nunique()} patients")
    for r in REGIONS:
        sub = power_df[power_df['region'] == r]
        if len(sub) > 0:
            print(f"    {r}: {len(sub)} trials, {sub['patient_id'].nunique()} patients")
    return power_df


def prep_ied_trial_data():
    """Prepare IED trial-level data with timing windows."""
    df = pd.read_csv(IED_CSV)
    dm = df[df['MemoryOutcome'].notna()].copy()
    dm = dm[dm['StimCond'].isin(['S', 'NS'])]

    agg_dict = {c: lambda x, c=c: 1 if (x == 'Y').any() else 0
                for c in ENC_TIMING}
    trial = dm.groupby(['Patient', 'Trial', 'MemoryOutcome', 'StimCond']).agg(
        agg_dict
    ).reset_index()

    # Spread metrics
    spread_agg = dm.groupby(['Patient', 'Trial']).agg(
        n_regions=('Region', 'nunique'),
    ).reset_index()

    channel_counts = []
    for (pat, tr), grp in dm.groupby(['Patient', 'Trial']):
        all_contacts = set()
        for cn in grp['ChannelName']:
            contacts = str(cn).split('_')
            all_contacts.update(contacts)
        channel_counts.append({'Patient': pat, 'Trial': tr,
                                'n_channels': len(all_contacts)})
    channel_df = pd.DataFrame(channel_counts)

    trial = trial.merge(spread_agg, on=['Patient', 'Trial'], how='left')
    trial = trial.merge(channel_df, on=['Patient', 'Trial'], how='left')

    trial = trial.rename(columns={
        'Patient': 'patient_id', 'Trial': 'trial',
        'MemoryOutcome': 'memory_str', 'StimCond': 'stim_str',
        'BeforeImgITI': 'ied_before_image', 'DuringImg': 'ied_during_image',
        'AfterImgITI': 'ied_after_image', 'DuringStim': 'ied_during_stim',
    })
    trial['memory'] = (trial['memory_str'] == 'remembered').astype(int)
    trial['stim'] = (trial['stim_str'] == 'S').astype(int)

    print(f"\n  IED trial data: {len(trial)} trials, "
          f"{trial['patient_id'].nunique()} patients")
    return trial


def merge_power_ied(power_df, ied_df):
    """Merge power (with trial IDs) and IED timing windows."""
    print("\n" + "=" * 60)
    print("MERGING Power + IED Data")
    print("=" * 60)

    merged = power_df.merge(
        ied_df,
        on=['patient_id', 'trial'],
        how='inner'
    )

    print(f"\n  Merged total: {len(merged)} trial-region rows")
    for r in REGIONS:
        sub = merged[merged['region'] == r]
        if len(sub) > 0:
            print(f"  {r}: {len(sub)} trials, {sub['patient_id'].nunique()} patients")

    out_path = os.path.join(OUTPUT_DIR, 'power_ied_merged.csv')
    keep_cols = ['patient_id', 'region', 'trial', 'memory', 'stim',
                 'pow_theta', 'pow_slow_gamma', 'pow_hfa',
                 'ied_before_image', 'ied_during_image',
                 'ied_after_image', 'ied_during_stim',
                 'n_regions', 'n_channels']
    merged[keep_cols].to_csv(out_path, index=False)
    print(f"\n  Saved -> {out_path}")
    return merged


def main():
    power_all = prep_power_all()
    power_raw = load_raw_power()
    ied_df = prep_ied_trial_data()
    merged = merge_power_ied(power_raw, ied_df)
    print("\n===== PREP DONE =====")


if __name__ == '__main__':
    main()
