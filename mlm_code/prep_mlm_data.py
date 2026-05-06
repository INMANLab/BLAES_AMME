#!/usr/bin/env python
"""
Prepare trial-level CSV for mixed-effects logistic regression in R.
Outputs: outputs/ied_timing_memory/mlm_encoding_trials.csv
         outputs/ied_timing_memory/mlm_retrieval_patient_freq.csv
"""

import os
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'outputs', 'ied_timing_memory')
os.makedirs(OUTPUT_DIR, exist_ok=True)

ENCODING_CSV = os.path.join(SCRIPT_DIR, 'IED',
    'AMMEBLAES_IEDs_trial_level_dissertation_study_usethis_cleaned_with_memory.csv')
RETRIEVAL_CSV = os.path.join(SCRIPT_DIR, 'IED',
    'AMMEBLAES_IEDs_trial_level_dissertation_test_usethis_cleaned_with_memory.csv')


def prep_encoding():
    """Collapse encoding data to trial level with binary outcomes."""
    df = pd.read_csv(ENCODING_CSV)
    dm = df[df['MemoryOutcome'].notna()].copy()
    # Keep only S / NS stim conditions
    dm = dm[dm['StimCond'].isin(['S', 'NS'])]

    trial = dm.groupby(['Patient', 'Trial', 'MemoryOutcome', 'StimCond']).agg({
        'DuringImg': lambda x: 1 if (x == 'Y').any() else 0,
        'DuringStim': lambda x: 1 if (x == 'Y').any() else 0,
        'BeforeImgITI': lambda x: 1 if (x == 'Y').any() else 0,
        'AfterImgITI': lambda x: 1 if (x == 'Y').any() else 0,
    }).reset_index()

    # Rename for R friendliness
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

    # Binary coding
    trial['memory'] = (trial['memory_str'] == 'remembered').astype(int)
    trial['stim'] = (trial['stim_str'] == 'S').astype(int)

    # Any encoding IED (any window)
    trial['encoding_ied'] = (
        (trial['ied_before_image'] == 1) | (trial['ied_during_image'] == 1) |
        (trial['ied_after_image'] == 1) | (trial['ied_during_stim'] == 1)
    ).astype(int)

    # Long-format encoding window variable for the window-specific model
    # Each trial can have IEDs in multiple windows, so we keep per-window columns
    # But also create a "primary window" for visualization
    # For the interaction model, we use the individual window columns directly

    return trial


BLAES_DIR = os.path.join(SCRIPT_DIR, '..', 'LFP_analyses', 'Results_CSVOutput')
AMME_DIR = os.path.join(SCRIPT_DIR, '..', '..', '..', 'AMME_Data_Emory',
                        'AMME_Data', 'LFP_analyses_Martina', 'Results_CSVOutput')


def count_total_retrieval_trials(patients):
    """Count total retrieval trials per patient from phase3 power CSV files."""
    pat_trials = {}
    for pat in patients:
        is_amme = pat.lower().startswith('amyg')
        if is_amme:
            f = os.path.join(AMME_DIR, f'Data_{pat}phase3MeasurePower.csv')
            trial_col = 'number'
        else:
            f = os.path.join(BLAES_DIR, f'Data_{pat}phase3MeasurePower.csv')
            trial_col = 'trialIdx'
        if not os.path.exists(f):
            continue
        df = pd.read_csv(f, usecols=[trial_col, 'Region'])
        first_region = df['Region'].iloc[0]
        sub = df[df['Region'] == first_region]
        pat_trials[pat] = sub[trial_col].nunique()
    return pat_trials


def prep_retrieval_freq(patients):
    """Compute patient-level retrieval IED frequency relative to total trials.

    numerator   = number of retrieval trials with any IED (from IED dataset)
    denominator = total retrieval trials (from phase3 power CSV files)
    patients not in IED dataset get 0 IED trials
    """
    # Total retrieval trials from power files
    total_ret = count_total_retrieval_trials(patients)

    # IED trial counts from retrieval IED dataset
    df = pd.read_csv(RETRIEVAL_CSV)
    dm = df[df['MemoryOutcome'].isin(['remembered', 'forgotten'])].copy()

    # Collapse to trial level
    trial = dm.groupby(['Patient', 'Trial', 'MemoryOutcome']).agg({
        'BeforeImgITI': lambda x: 1 if (x == 'Y').any() else 0,
        'DuringImgITI': lambda x: 1 if (x == 'Y').any() else 0,
    }).reset_index()

    trial['any_retrieval_ied'] = (
        (trial['BeforeImgITI'] == 1) | (trial['DuringImgITI'] == 1)
    ).astype(int)

    ied_counts = trial.groupby('Patient')['any_retrieval_ied'].sum().to_dict()

    # Build patient-level frequency for ALL patients
    rows = []
    for pat in patients:
        n_total = total_ret.get(pat, np.nan)
        n_ied = ied_counts.get(pat, 0)
        freq = n_ied / n_total if pd.notna(n_total) and n_total > 0 else np.nan
        rows.append({
            'patient_id': pat,
            'n_retrieval_trials_total': n_total,
            'n_retrieval_ied': n_ied,
            'retrieval_ied_freq': freq,
        })

    return pd.DataFrame(rows)


def main():
    enc = prep_encoding()
    patients = sorted(enc['patient_id'].unique())
    ret_freq = prep_retrieval_freq(patients)

    # Merge retrieval IED frequency onto encoding trials
    enc = enc.merge(ret_freq[['patient_id', 'retrieval_ied_freq']], on='patient_id', how='left')

    # Save
    out_enc = os.path.join(OUTPUT_DIR, 'mlm_encoding_trials.csv')
    enc.to_csv(out_enc, index=False)
    print(f'Saved {out_enc}')
    print(f'  {len(enc)} trials, {enc["patient_id"].nunique()} patients')
    print(f'  {enc["memory"].sum()} remembered, {(1 - enc["memory"]).sum():.0f} forgotten')
    print(f'  Patients with retrieval IED freq: {enc["retrieval_ied_freq"].notna().sum()} trials '
          f'({enc.loc[enc["retrieval_ied_freq"].notna(), "patient_id"].nunique()} patients)')

    out_ret = os.path.join(OUTPUT_DIR, 'mlm_retrieval_patient_freq.csv')
    ret_freq.to_csv(out_ret, index=False)
    print(f'Saved {out_ret}')


if __name__ == '__main__':
    main()
