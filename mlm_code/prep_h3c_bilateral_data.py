#!/usr/bin/env python
"""
Prepare data for Hypothesis 3c bilateral IED analyses.
Selects patients with bilateral electrode coverage and classifies
trial-level IED laterality (L, R, Bilateral, None).

Outputs:
  outputs/ied_timing_memory/h3c_encoding_bilateral.csv
  outputs/ied_timing_memory/h3c_retrieval_bilateral.csv
  outputs/ied_timing_memory/h3c_encoding_descriptives.csv
  outputs/ied_timing_memory/h3c_retrieval_descriptives.csv
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

ENC_TIMING = ['BeforeImgITI', 'DuringImg', 'AfterImgITI', 'DuringStim']
RET_TIMING = ['BeforeImgITI', 'DuringImgITI']


def classify_hemi(vals):
    """Classify hemisphere from a set of L/R values."""
    clean = set(vals.dropna()) - {'D'}
    if 'L' in clean and 'R' in clean:
        return 'Bilateral'
    elif 'L' in clean:
        return 'L'
    elif 'R' in clean:
        return 'R'
    return 'None'


def find_bilateral_patients(df):
    """Find patients who have BOTH L and R hemisphere entries (bilateral coverage)."""
    pat_hemis = df.groupby('Patient')['Hemisphere'].apply(
        lambda x: set(x.dropna()) - {'D'}
    )
    bilateral = [p for p, h in pat_hemis.items() if 'L' in h and 'R' in h]
    return sorted(bilateral)


def prep_encoding():
    """Prepare encoding data for bilateral patients."""
    df = pd.read_csv(ENCODING_CSV)
    dm = df[df['MemoryOutcome'].notna()].copy()
    dm = dm[dm['StimCond'].isin(['S', 'NS'])]

    # Find bilateral patients
    bilateral_pats = find_bilateral_patients(dm)
    print(f"Encoding bilateral patients ({len(bilateral_pats)}): {bilateral_pats}")

    # ALL patients for the full laterality model
    all_pats = sorted(dm['Patient'].unique())

    # ── Trial-level hemisphere classification ──
    hemi_map = dm.groupby(['Patient', 'Trial']).agg(
        trial_hemisphere=('Hemisphere', classify_hemi)
    ).reset_index()

    # ── Collapse timing windows to trial level ──
    agg_dict = {c: lambda x, c=c: 1 if (x == 'Y').any() else 0 for c in ENC_TIMING}
    trial = dm.groupby(['Patient', 'Trial', 'MemoryOutcome', 'StimCond']).agg(
        agg_dict
    ).reset_index()

    trial = trial.merge(hemi_map, on=['Patient', 'Trial'], how='left')

    # Rename
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
    trial['any_ied'] = (
        (trial['ied_before_image'] == 1) | (trial['ied_during_image'] == 1) |
        (trial['ied_after_image'] == 1) | (trial['ied_during_stim'] == 1)
    ).astype(int)

    # Binary: is this trial bilateral IED?
    trial['bilateral_ied'] = (trial['trial_hemisphere'] == 'Bilateral').astype(int)
    # Categorical hemisphere (for full model)
    trial['hemi_L'] = (trial['trial_hemisphere'] == 'L').astype(int)
    trial['hemi_R'] = (trial['trial_hemisphere'] == 'R').astype(int)
    trial['hemi_bilateral'] = trial['bilateral_ied']

    # Save full dataset (all patients) for the full laterality model
    out_all = os.path.join(OUTPUT_DIR, 'h3c_encoding_all.csv')
    trial.to_csv(out_all, index=False)
    print(f"  Saved all-patient encoding: {len(trial)} trials, "
          f"{trial['patient_id'].nunique()} patients -> {out_all}")

    # Save bilateral-coverage subset
    trial_bi = trial[trial['patient_id'].isin(bilateral_pats)].copy()
    out_bi = os.path.join(OUTPUT_DIR, 'h3c_encoding_bilateral.csv')
    trial_bi.to_csv(out_bi, index=False)
    print(f"  Saved bilateral subset: {len(trial_bi)} trials, "
          f"{trial_bi['patient_id'].nunique()} patients -> {out_bi}")

    # ── Descriptives ──
    desc_rows = []
    for pat in bilateral_pats:
        sub = trial_bi[trial_bi['patient_id'] == pat]
        n = len(sub)
        n_rem = sub['memory'].sum()
        n_stim = sub['stim'].sum()
        n_any_ied = sub['any_ied'].sum()
        n_bi = sub['bilateral_ied'].sum()
        n_l = sub['hemi_L'].sum()
        n_r = sub['hemi_R'].sum()
        for col_name, col_label in [('ied_before_image', 'Before Image'),
                                     ('ied_during_image', 'During Image'),
                                     ('ied_after_image', 'After Image'),
                                     ('ied_during_stim', 'During Stim')]:
            n_win = sub[col_name].sum()
            desc_rows.append({
                'patient_id': pat, 'n_trials': n, 'n_remembered': n_rem,
                'n_stim': n_stim, 'n_any_ied': n_any_ied,
                'n_bilateral': n_bi, 'n_left': n_l, 'n_right': n_r,
                'window': col_label, 'n_window_ied': n_win,
            })

    desc = pd.DataFrame(desc_rows)
    out_desc = os.path.join(OUTPUT_DIR, 'h3c_encoding_descriptives.csv')
    desc.to_csv(out_desc, index=False)
    print(f"  Saved descriptives -> {out_desc}")

    return trial, trial_bi, bilateral_pats


def prep_retrieval():
    """Prepare retrieval data for bilateral patients."""
    df = pd.read_csv(RETRIEVAL_CSV)
    dm = df[df['MemoryOutcome'].isin(['remembered', 'forgotten'])].copy()

    bilateral_pats = find_bilateral_patients(dm)
    print(f"\nRetrieval bilateral patients ({len(bilateral_pats)}): {bilateral_pats}")

    # Trial-level hemisphere
    hemi_map = dm.groupby(['Patient', 'Trial']).agg(
        trial_hemisphere=('Hemisphere', classify_hemi)
    ).reset_index()

    agg_dict = {c: lambda x, c=c: 1 if (x == 'Y').any() else 0 for c in RET_TIMING}
    trial = dm.groupby(['Patient', 'Trial', 'MemoryOutcome']).agg(
        agg_dict
    ).reset_index()

    trial = trial.merge(hemi_map, on=['Patient', 'Trial'], how='left')

    trial = trial.rename(columns={
        'Patient': 'patient_id',
        'Trial': 'trial',
        'MemoryOutcome': 'memory_str',
        'BeforeImgITI': 'ied_before_image',
        'DuringImgITI': 'ied_during_image',
    })

    trial['memory'] = (trial['memory_str'] == 'remembered').astype(int)
    trial['any_ied'] = (
        (trial['ied_before_image'] == 1) | (trial['ied_during_image'] == 1)
    ).astype(int)
    trial['bilateral_ied'] = (trial['trial_hemisphere'] == 'Bilateral').astype(int)
    trial['hemi_L'] = (trial['trial_hemisphere'] == 'L').astype(int)
    trial['hemi_R'] = (trial['trial_hemisphere'] == 'R').astype(int)
    trial['hemi_bilateral'] = trial['bilateral_ied']

    # Save all patients
    out_all = os.path.join(OUTPUT_DIR, 'h3c_retrieval_all.csv')
    trial.to_csv(out_all, index=False)
    print(f"  Saved all-patient retrieval: {len(trial)} trials, "
          f"{trial['patient_id'].nunique()} patients -> {out_all}")

    # Save bilateral subset
    trial_bi = trial[trial['patient_id'].isin(bilateral_pats)].copy()
    out_bi = os.path.join(OUTPUT_DIR, 'h3c_retrieval_bilateral.csv')
    trial_bi.to_csv(out_bi, index=False)
    print(f"  Saved bilateral subset: {len(trial_bi)} trials, "
          f"{trial_bi['patient_id'].nunique()} patients -> {out_bi}")

    # Descriptives
    desc_rows = []
    for pat in bilateral_pats:
        sub = trial_bi[trial_bi['patient_id'] == pat]
        n = len(sub)
        n_rem = sub['memory'].sum()
        n_any_ied = sub['any_ied'].sum()
        n_bi = sub['bilateral_ied'].sum()
        n_l = sub['hemi_L'].sum()
        n_r = sub['hemi_R'].sum()
        for col_name, col_label in [('ied_before_image', 'Before Image'),
                                     ('ied_during_image', 'During Image')]:
            n_win = sub[col_name].sum()
            desc_rows.append({
                'patient_id': pat, 'n_trials': n, 'n_remembered': n_rem,
                'n_any_ied': n_any_ied,
                'n_bilateral': n_bi, 'n_left': n_l, 'n_right': n_r,
                'window': col_label, 'n_window_ied': n_win,
            })

    desc = pd.DataFrame(desc_rows)
    out_desc = os.path.join(OUTPUT_DIR, 'h3c_retrieval_descriptives.csv')
    desc.to_csv(out_desc, index=False)
    print(f"  Saved descriptives -> {out_desc}")

    return trial, trial_bi, bilateral_pats


def main():
    print("=" * 60)
    print("ENCODING PHASE")
    print("=" * 60)
    enc_all, enc_bi, enc_bi_pats = prep_encoding()

    print("\n" + "=" * 60)
    print("RETRIEVAL PHASE")
    print("=" * 60)
    ret_all, ret_bi, ret_bi_pats = prep_retrieval()


if __name__ == '__main__':
    main()
