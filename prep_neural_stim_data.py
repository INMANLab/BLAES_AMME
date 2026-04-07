#!/usr/bin/env python
"""
Prepare data for Neural and Stim Effects MLM report.

Reads raw coherence and PAC source files (with real trial IDs) and merges
with IED trial-level data to create datasets for:
  1. Coherence/PAC -> Memory (all BLA-HPC trials)
  2. Coherence/PAC x IED Timing Windows -> Memory (IED trials merged)
  3. During-stim subset

Outputs (all in outputs/ied_timing_memory/):
  neural_stim_coh_all.csv          - all BLA-HPC coherence trials (band-averaged)
  neural_stim_coh_ied_merged.csv   - IED trials merged with BLA-HPC coherence
  neural_stim_pac_all.csv          - all BLA-HPC PAC trials (band-averaged)
  neural_stim_pac_ied_merged.csv   - IED trials merged with BLA-HPC PAC
  neural_stim_descriptives.csv     - per-patient summary of merged data
"""

import os
import glob
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'outputs', 'ied_timing_memory')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Existing MLMR CSV (all trials, no trial IDs)
MLMR_CSV = os.path.join(SCRIPT_DIR, 'outputs', 'csvs',
    'combined_encoding_coherence_all_mlmr_input.csv')

# IED encoding data (IED trials only, with timing windows and memory)
IED_CSV = os.path.join(SCRIPT_DIR, 'IED',
    'AMMEBLAES_IEDs_trial_level_dissertation_study_usethis_cleaned_with_memory.csv')

# Raw coherence source paths
BLAES_COH_DIR = '/Users/martinahollearn/Library/CloudStorage/Box-Box/InmanLab/BLAES_data/dissertation/LFP_analyses/Results_CSVOutput'
AMME_COH_DIR = '/Users/martinahollearn/Library/CloudStorage/Box-Box/InmanLab/AMME_Data_Emory/AMME_Data/LFP_analyses_Martina/Results_CSVOutput/Phase1'

# Raw PAC source paths (same directories, PAC_TG files)
AMME_PAC_DIR = '/Users/martinahollearn/Library/CloudStorage/Box-Box/InmanLab/AMME_Data_Emory/AMME_Data/LFP_analyses_Martina/Results_CSVOutput'

# PAC MLMR CSV (all trials, no trial IDs - for all-trial PAC models)
PAC_MLMR_CSV = os.path.join(SCRIPT_DIR, 'outputs', 'csvs',
    'combined_encoding_pac_all_mlmr_input.csv')

# Frequency bands
THETA = (4, 8)
SLOW_GAMMA = (30, 55)
HFA = (70, 100)

BLA_HPC_PAIRS = ['BLA_HPC', 'BLA_CA', 'BLA_DG']

ENC_TIMING = ['BeforeImgITI', 'DuringImg', 'AfterImgITI', 'DuringStim']


def get_diff_freq_cols(df):
    """Get diff_Freq_* columns and their numeric values."""
    cols = [c for c in df.columns if c.startswith('diff_Freq_')]
    freqs = np.array([float(c.replace('diff_Freq_', '')) for c in cols])
    return cols, freqs


def band_average(df, freq_cols, freqs, prefix='coh'):
    """Add band-averaged columns with given prefix."""
    for name, (lo, hi) in [('theta', THETA), ('slow_gamma', SLOW_GAMMA), ('hfa', HFA)]:
        mask = (freqs >= lo) & (freqs <= hi)
        band_cols = np.array(freq_cols)[mask]
        if len(band_cols) > 0:
            df[f'{prefix}_{name}'] = df[band_cols].astype(float).mean(axis=1)
    return df


def get_pre_post_freq_cols(df):
    """Get pre_Freq_* and post_Freq_* columns."""
    pre_cols = sorted([c for c in df.columns if c.startswith('pre_Freq_')])
    post_cols = sorted([c for c in df.columns if c.startswith('post_Freq_')])
    if pre_cols:
        freqs = np.array([float(c.replace('pre_Freq_', '')) for c in pre_cols])
    else:
        freqs = np.array([])
    return pre_cols, post_cols, freqs


# ═══════════════════════════════════════════════════════════════════════════
# PART 1: All BLA-HPC coherence trials (from existing MLMR export)
# ═══════════════════════════════════════════════════════════════════════════

def prep_coherence_all():
    """Band-average coherence for all BLA-HPC trials from MLMR CSV."""
    print("=" * 60)
    print("PART 1: All BLA-HPC Coherence Trials")
    print("=" * 60)

    df = pd.read_csv(MLMR_CSV)
    freq_cols, freqs = get_diff_freq_cols(df)

    # Filter to BLA target pairs
    df_filt = df[df['Region'].isin(BLA_HPC_PAIRS)].copy()
    df_filt = band_average(df_filt, freq_cols, freqs)

    df_filt['memory'] = (df_filt['yes_or_no'] == 'yes').astype(int)
    df_filt['stim'] = (df_filt['trial_type'] == 'stim').astype(int)
    df_filt = df_filt.rename(columns={'Patient': 'patient_id', 'Region': 'region_pair'})

    keep = ['patient_id', 'region_pair', 'stim', 'memory',
            'coh_theta', 'coh_slow_gamma', 'coh_hfa']
    out = df_filt[keep].copy()

    out_path = os.path.join(OUTPUT_DIR, 'neural_stim_coh_all.csv')
    out.to_csv(out_path, index=False)

    bla_hpc = out[out['region_pair'] == 'BLA_HPC']
    print(f"  All BLA-HPC trials: {len(bla_hpc)} trials, "
          f"{bla_hpc['patient_id'].nunique()} patients")
    print(f"  Total (all pairs): {len(out)} trials -> {out_path}")
    return out


# ═══════════════════════════════════════════════════════════════════════════
# PART 2: Raw coherence with trial IDs, merged with IED timing windows
# ═══════════════════════════════════════════════════════════════════════════

def load_raw_coherence_with_trial_ids():
    """Read raw coherence source files and extract trial-level band-averaged
    coherence with real trial IDs for BLA-HPC pairs."""

    print("\n" + "=" * 60)
    print("PART 2: Raw Coherence with Trial IDs")
    print("=" * 60)

    rows = []

    # BLAES files: trialIdx is the trial ID
    blaes_files = sorted(glob.glob(
        os.path.join(BLAES_COH_DIR, 'Data_*phase1MeasureCoherency.csv')))
    print(f"\n  BLAES coherence files: {len(blaes_files)}")

    for f in blaes_files:
        df = pd.read_csv(f)
        if 'trialIdx' not in df.columns or 'Region' not in df.columns:
            continue

        patient = df['Patient'].iloc[0]
        freq_cols, freqs = get_diff_freq_cols(df)
        if not freq_cols:
            continue

        for pair in BLA_HPC_PAIRS:
            sub = df[df['Region'] == pair].copy()
            if sub.empty:
                continue

            sub = band_average(sub, freq_cols, freqs)
            for _, row in sub.iterrows():
                rows.append({
                    'patient_id': patient,
                    'region_pair': pair,
                    'trial': int(row['trialIdx']),
                    'coh_theta': row['coh_theta'],
                    'coh_slow_gamma': row['coh_slow_gamma'],
                    'coh_hfa': row['coh_hfa'],
                })

    # AMME files: number is the trial ID
    amme_files = sorted(glob.glob(
        os.path.join(AMME_COH_DIR, '*Coherency*.csv')))
    print(f"  AMME coherence files: {len(amme_files)}")

    for f in amme_files:
        df = pd.read_csv(f)
        if 'number' not in df.columns or 'Region' not in df.columns:
            continue

        patient = df['Patient'].iloc[0]
        freq_cols, freqs = get_diff_freq_cols(df)
        if not freq_cols:
            continue

        for pair in BLA_HPC_PAIRS:
            sub = df[df['Region'] == pair].copy()
            if sub.empty:
                continue

            sub = band_average(sub, freq_cols, freqs)
            for _, row in sub.iterrows():
                rows.append({
                    'patient_id': patient,
                    'region_pair': pair,
                    'trial': int(row['number']),
                    'coh_theta': row['coh_theta'],
                    'coh_slow_gamma': row['coh_slow_gamma'],
                    'coh_hfa': row['coh_hfa'],
                })

    coh_df = pd.DataFrame(rows)
    print(f"\n  Raw coherence extracted: {len(coh_df)} rows, "
          f"{coh_df['patient_id'].nunique()} patients")
    return coh_df


def prep_ied_trial_data():
    """Prepare IED trial-level data with timing windows."""
    df = pd.read_csv(IED_CSV)
    dm = df[df['MemoryOutcome'].notna()].copy()
    dm = dm[dm['StimCond'].isin(['S', 'NS'])]

    # Collapse to trial level
    agg_dict = {c: lambda x, c=c: 1 if (x == 'Y').any() else 0
                for c in ENC_TIMING}

    trial = dm.groupby(['Patient', 'Trial', 'MemoryOutcome', 'StimCond']).agg(
        agg_dict
    ).reset_index()

    # Spread metrics per trial
    spread_agg = dm.groupby(['Patient', 'Trial']).agg(
        n_leads=('Region', 'count'),
        n_regions=('Region', 'nunique'),
        region_spread=('RegionSpread', 'first'),
    ).reset_index()

    # Channel count per trial
    channel_counts = []
    for (pat, tr), grp in dm.groupby(['Patient', 'Trial']):
        all_contacts = set()
        for cn in grp['ChannelName']:
            contacts = str(cn).split('_')
            all_contacts.update(contacts)
        channel_counts.append({
            'Patient': pat, 'Trial': tr,
            'n_channels': len(all_contacts),
        })
    channel_df = pd.DataFrame(channel_counts)

    trial = trial.merge(spread_agg, on=['Patient', 'Trial'], how='left')
    trial = trial.merge(channel_df, on=['Patient', 'Trial'], how='left')

    trial = trial.rename(columns={
        'Patient': 'patient_id',
        'Trial': 'trial',
        'MemoryOutcome': 'memory_str',
        'StimCond': 'stim_str',
        'BeforeImgITI': 'ied_before_image',
        'DuringImg': 'ied_during_image',
        'AfterImgITI': 'ied_after_image',
        'DuringStim': 'ied_during_stim',
    })

    trial['memory'] = (trial['memory_str'] == 'remembered').astype(int)
    trial['stim'] = (trial['stim_str'] == 'S').astype(int)
    trial['cross_region'] = (trial['n_regions'] > 1).astype(int)

    print(f"\n  IED trial data: {len(trial)} trials, "
          f"{trial['patient_id'].nunique()} patients")
    return trial


def merge_coherence_ied(coh_df, ied_df):
    """Merge coherence (with trial IDs) and IED timing windows."""
    print("\n" + "=" * 60)
    print("MERGING Coherence + IED Data")
    print("=" * 60)

    # Inner merge: only trials with BOTH coherence and IED data
    merged = coh_df.merge(
        ied_df,
        on=['patient_id', 'trial'],
        how='inner',
        suffixes=('_coh', '_ied')
    )

    # Use IED memory/stim (authoritative for IED trials)
    # Drop duplicate columns if any
    if 'memory_coh' in merged.columns:
        merged = merged.drop(columns=['memory_coh'])
        merged = merged.rename(columns={'memory_ied': 'memory'})
    if 'stim_coh' in merged.columns:
        merged = merged.drop(columns=['stim_coh'])
        merged = merged.rename(columns={'stim_ied': 'stim'})

    # Focus on BLA_HPC
    bla_hpc = merged[merged['region_pair'] == 'BLA_HPC']

    print(f"\n  Merged (all pairs): {len(merged)} trial-pair rows")
    print(f"  BLA_HPC merged: {len(bla_hpc)} trials, "
          f"{bla_hpc['patient_id'].nunique()} patients")

    # Per-patient summary
    print("\n  Per-patient BLA_HPC trial counts:")
    for p in sorted(bla_hpc['patient_id'].unique()):
        sub = bla_hpc[bla_hpc['patient_id'] == p]
        n_rem = sub['memory'].sum()
        n_stim = sub['stim'].sum()
        print(f"    {p}: {len(sub)} trials, "
              f"{n_rem} remembered ({n_rem/len(sub)*100:.0f}%), "
              f"{n_stim} stim")

    out_path = os.path.join(OUTPUT_DIR, 'neural_stim_coh_ied_merged.csv')
    keep_cols = ['patient_id', 'region_pair', 'trial', 'memory', 'stim',
                 'coh_theta', 'coh_slow_gamma', 'coh_hfa',
                 'ied_before_image', 'ied_during_image',
                 'ied_after_image', 'ied_during_stim',
                 'n_leads', 'n_regions', 'n_channels', 'cross_region']
    merged[keep_cols].to_csv(out_path, index=False)
    print(f"\n  Saved -> {out_path}")

    # Descriptives
    desc_rows = []
    for p in sorted(bla_hpc['patient_id'].unique()):
        sub = bla_hpc[bla_hpc['patient_id'] == p]
        desc_rows.append({
            'patient_id': p,
            'n_trials': len(sub),
            'n_remembered': int(sub['memory'].sum()),
            'pct_remembered': round(sub['memory'].mean() * 100, 1),
            'n_stim': int(sub['stim'].sum()),
            'n_before': int(sub['ied_before_image'].sum()),
            'n_during': int(sub['ied_during_image'].sum()),
            'n_after': int(sub['ied_after_image'].sum()),
            'n_stim_window': int(sub['ied_during_stim'].sum()),
            'mean_coh_theta': round(sub['coh_theta'].mean(), 4),
            'mean_coh_slow_gamma': round(sub['coh_slow_gamma'].mean(), 4),
            'mean_coh_hfa': round(sub['coh_hfa'].mean(), 4),
            'mean_regions': round(sub['n_regions'].mean(), 2),
            'mean_channels': round(sub['n_channels'].mean(), 2),
        })

    desc = pd.DataFrame(desc_rows)
    desc_path = os.path.join(OUTPUT_DIR, 'neural_stim_descriptives.csv')
    desc.to_csv(desc_path, index=False)
    print(f"  Saved descriptives -> {desc_path}")

    return merged


# =====================================================================
# PAC: All BLA-HPC PAC trials (from MLMR export)
# =====================================================================

def prep_pac_all():
    """Band-average PAC for all BLA-HPC trials from PAC MLMR CSV."""
    print("\n" + "=" * 60)
    print("PAC: All BLA-HPC PAC Trials (MLMR Export)")
    print("=" * 60)

    if not os.path.exists(PAC_MLMR_CSV):
        print(f"  PAC MLMR CSV not found: {PAC_MLMR_CSV}")
        return None

    df = pd.read_csv(PAC_MLMR_CSV)
    # PAC CSV has: Measure, Patient, Region, trial_type, yes_or_no, diff_Freq_*
    freq_cols = sorted([c for c in df.columns if c.startswith('diff_Freq_')])
    if not freq_cols:
        print("  No diff_Freq columns in PAC CSV, computing from pre/post")
        pre_cols, post_cols, freqs = get_pre_post_freq_cols(df)
        if not pre_cols:
            print("  No frequency columns found in PAC MLMR CSV")
            return None
        for pre, post in zip(pre_cols, post_cols):
            diff_name = pre.replace('pre_', 'diff_')
            df[diff_name] = df[post].astype(float) - df[pre].astype(float)
        freq_cols = sorted([c for c in df.columns if c.startswith('diff_Freq_')])

    freqs = np.array([float(c.replace('diff_Freq_', '')) for c in freq_cols])

    df_filt = df[df['Region'].isin(BLA_HPC_PAIRS)].copy()
    df_filt = band_average(df_filt, freq_cols, freqs, prefix='pac')

    df_filt['memory'] = (df_filt['yes_or_no'] == 'yes').astype(int)
    df_filt['stim'] = (df_filt['trial_type'] == 'stim').astype(int)
    df_filt = df_filt.rename(columns={'Patient': 'patient_id', 'Region': 'region_pair'})

    keep = ['patient_id', 'region_pair', 'stim', 'memory']
    for b in ['slow_gamma', 'hfa']:
        col = f'pac_{b}'
        if col in df_filt.columns:
            keep.append(col)
    out = df_filt[keep].copy()

    out_path = os.path.join(OUTPUT_DIR, 'neural_stim_pac_all.csv')
    out.to_csv(out_path, index=False)

    bla_hpc = out[out['region_pair'] == 'BLA_HPC']
    print(f"  All BLA-HPC PAC trials: {len(bla_hpc)} trials, "
          f"{bla_hpc['patient_id'].nunique()} patients")
    print(f"  Total (all pairs): {len(out)} trials -> {out_path}")
    return out


# =====================================================================
# PAC: Raw files with trial IDs
# =====================================================================

def load_raw_pac_with_trial_ids():
    """Read raw PAC source files and extract trial-level band-averaged
    PAC with real trial IDs for BLA-HPC pairs."""

    print("\n" + "=" * 60)
    print("PAC: Raw Files with Trial IDs")
    print("=" * 60)

    rows = []

    # BLAES files: trialIdx is the trial ID
    blaes_files = sorted(glob.glob(
        os.path.join(BLAES_COH_DIR, 'Data_*phase1MeasurePAC_TG.csv')))
    print(f"\n  BLAES PAC files: {len(blaes_files)}")

    for f in blaes_files:
        df = pd.read_csv(f)
        if 'trialIdx' not in df.columns or 'Region' not in df.columns:
            continue

        patient = df['Patient'].iloc[0]
        pre_cols, post_cols, freqs = get_pre_post_freq_cols(df)
        if not pre_cols:
            continue

        # Compute diff = post - pre
        for pre, post in zip(pre_cols, post_cols):
            diff_name = pre.replace('pre_', 'diff_')
            df[diff_name] = df[post].astype(float) - df[pre].astype(float)
        diff_cols = sorted([c for c in df.columns if c.startswith('diff_Freq_')])

        for pair in BLA_HPC_PAIRS:
            sub = df[df['Region'] == pair].copy()
            if sub.empty:
                continue

            sub = band_average(sub, diff_cols, freqs, prefix='pac')
            for _, row in sub.iterrows():
                r = {
                    'patient_id': patient,
                    'region_pair': pair,
                    'trial': int(row['trialIdx']),
                }
                for b in ['slow_gamma', 'hfa']:
                    col = f'pac_{b}'
                    if col in sub.columns:
                        r[col] = row[col]
                rows.append(r)

    # AMME files: number is the trial ID
    # AMME PAC phase1 files are in the parent Results_CSVOutput dir
    amme_files = sorted(glob.glob(
        os.path.join(AMME_PAC_DIR, 'Data_*phase1MeasurePAC_TG.csv')))
    print(f"  AMME PAC phase1 files: {len(amme_files)}")

    for f in amme_files:
        df = pd.read_csv(f)
        if 'number' not in df.columns or 'Region' not in df.columns:
            continue

        patient = df['Patient'].iloc[0]
        pre_cols, post_cols, freqs = get_pre_post_freq_cols(df)
        if not pre_cols:
            continue

        for pre, post in zip(pre_cols, post_cols):
            diff_name = pre.replace('pre_', 'diff_')
            df[diff_name] = df[post].astype(float) - df[pre].astype(float)
        diff_cols = sorted([c for c in df.columns if c.startswith('diff_Freq_')])

        for pair in BLA_HPC_PAIRS:
            sub = df[df['Region'] == pair].copy()
            if sub.empty:
                continue

            sub = band_average(sub, diff_cols, freqs, prefix='pac')
            for _, row in sub.iterrows():
                r = {
                    'patient_id': patient,
                    'region_pair': pair,
                    'trial': int(row['number']),
                }
                for b in ['slow_gamma', 'hfa']:
                    col = f'pac_{b}'
                    if col in sub.columns:
                        r[col] = row[col]
                rows.append(r)

    pac_df = pd.DataFrame(rows)
    print(f"\n  Raw PAC extracted: {len(pac_df)} rows, "
          f"{pac_df['patient_id'].nunique()} patients")
    return pac_df


def merge_pac_ied(pac_df, ied_df):
    """Merge PAC (with trial IDs) and IED timing windows."""
    print("\n" + "=" * 60)
    print("MERGING PAC + IED Data")
    print("=" * 60)

    merged = pac_df.merge(
        ied_df,
        on=['patient_id', 'trial'],
        how='inner',
        suffixes=('_pac', '_ied')
    )

    if 'memory_pac' in merged.columns:
        merged = merged.drop(columns=['memory_pac'])
        merged = merged.rename(columns={'memory_ied': 'memory'})
    if 'stim_pac' in merged.columns:
        merged = merged.drop(columns=['stim_pac'])
        merged = merged.rename(columns={'stim_ied': 'stim'})

    bla_hpc = merged[merged['region_pair'] == 'BLA_HPC']

    print(f"\n  Merged PAC (all pairs): {len(merged)} trial-pair rows")
    print(f"  BLA_HPC merged: {len(bla_hpc)} trials, "
          f"{bla_hpc['patient_id'].nunique()} patients")

    out_path = os.path.join(OUTPUT_DIR, 'neural_stim_pac_ied_merged.csv')
    keep_cols = ['patient_id', 'region_pair', 'trial', 'memory', 'stim',
                 'ied_before_image', 'ied_during_image',
                 'ied_after_image', 'ied_during_stim',
                 'n_leads', 'n_regions', 'n_channels', 'cross_region']
    for b in ['slow_gamma', 'hfa']:
        col = f'pac_{b}'
        if col in merged.columns:
            keep_cols.append(col)
    merged[keep_cols].to_csv(out_path, index=False)
    print(f"  Saved -> {out_path}")

    return merged


def main():
    # Part 1: All coherence trials
    coh_all = prep_coherence_all()

    # Part 2: Raw coherence with trial IDs
    coh_raw = load_raw_coherence_with_trial_ids()

    # Part 3: IED trial data with timing windows
    ied_df = prep_ied_trial_data()

    # Part 4: Merge coherence + IED
    merged_coh = merge_coherence_ied(coh_raw, ied_df)

    # Part 5: All PAC trials
    pac_all = prep_pac_all()

    # Part 6: Raw PAC with trial IDs
    pac_raw = load_raw_pac_with_trial_ids()

    # Part 7: Merge PAC + IED
    if pac_raw is not None and len(pac_raw) > 0:
        merged_pac = merge_pac_ied(pac_raw, ied_df)
    else:
        merged_pac = None

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    bla_all = coh_all[coh_all['region_pair'] == 'BLA_HPC']
    bla_merged = merged_coh[merged_coh['region_pair'] == 'BLA_HPC']
    print(f"  All BLA-HPC coherence trials: {len(bla_all)} "
          f"({bla_all['patient_id'].nunique()} patients)")
    print(f"  IED+Coherence merged BLA-HPC: {len(bla_merged)} "
          f"({bla_merged['patient_id'].nunique()} patients)")
    if pac_all is not None:
        pac_bla = pac_all[pac_all['region_pair'] == 'BLA_HPC']
        print(f"  All BLA-HPC PAC trials: {len(pac_bla)} "
              f"({pac_bla['patient_id'].nunique()} patients)")
    if merged_pac is not None:
        pac_bla_m = merged_pac[merged_pac['region_pair'] == 'BLA_HPC']
        print(f"  IED+PAC merged BLA-HPC: {len(pac_bla_m)} "
              f"({pac_bla_m['patient_id'].nunique()} patients)")
    print("\n===== PREP DONE =====")


if __name__ == '__main__':
    main()
