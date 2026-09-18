#!/usr/bin/env python
"""
Prepare retrieval-phase trial-level data for the IED timing x memory GLMMs.
==========================================================================
Builds one row per retrieved OLD item (remembered / forgotten) with:
  - the two retrieval timing windows (Before Image, During Image), collapsed
    to trial level (Y if an IED occurred in that window on any channel),
  - the item's PRIOR STIMULATION condition (S / NS), i.e. whether the item was
    stimulated when it was studied at encoding.

Prior stimulation is recovered per item from the raw phase-3 (test) power files,
mirroring the Trial -> raw-trial join used to build MemoryOutcome in
trace_ied_memory_test.py:
  - BLAES (BJH/UIC): the test file carries a `stimulation` flag (1 = the item
    was stimulated at study -> S, 0 -> NS), keyed by `trialIdx`.
  - AMME  (amyg):    the test file's `trial_type` is stim / nostim / new, keyed
    by `number`  (stim -> S, nostim -> NS, new -> excluded as a foil).

Output:
  IED/ied_timing_memory/retrieval_timing_memory_trials.csv
"""

import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))            # .../AMME_BLAES/IED
RET_CSV = os.path.join(SCRIPT_DIR,
                       'AMMEBLAES_IEDs_trial_level_dissertation_test_usethis_cleaned_with_memory.csv')
OUT_DIR = os.path.join(SCRIPT_DIR, 'ied_timing_memory')
OUT_CSV = os.path.join(OUT_DIR, 'retrieval_timing_memory_trials.csv')
os.makedirs(OUT_DIR, exist_ok=True)

# Raw phase-3 (test) power files. AMME amyg files live in the BLAES dir under a
# "MartinaChannels" prefix; BLAES files use the plain patient name.
BLAES_CSV_DIR = os.path.join(SCRIPT_DIR, '..', '..', 'LFP_analyses', 'Results_CSVOutput')


def _phase3_path(patient):
    is_amme = patient.lower().startswith('amyg')
    candidates = []
    if is_amme:
        candidates.append(os.path.join(BLAES_CSV_DIR,
                                       f'Data_MartinaChannels{patient}phase3MeasurePower.csv'))
        candidates.append(os.path.join(SCRIPT_DIR, '..', '..', '..', 'AMME_Data_Emory',
                                       'AMME_Data', 'LFP_analyses_Martina', 'Results_CSVOutput',
                                       f'Data_{patient}phase3MeasurePower.csv'))
    else:
        candidates.append(os.path.join(BLAES_CSV_DIR, f'Data_{patient}phase3MeasurePower.csv'))
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def build_prior_stim_map(patient):
    """{trial_id -> 'S'/'NS'} for OLD items, from the raw phase-3 test file."""
    p3_file = _phase3_path(patient)
    if p3_file is None:
        return None
    p3 = pd.read_csv(p3_file)
    # one row per trial (first channel)
    if 'ChanName' in p3.columns:
        p3 = p3[p3['ChanName'] == p3['ChanName'].iloc[0]].copy()
    is_amme = patient.lower().startswith('amyg')
    trial_col = 'number' if 'number' in p3.columns else 'trialIdx'
    p3 = p3.sort_values(trial_col).reset_index(drop=True)

    stim_map = {}
    for _, row in p3.iterrows():
        tid = int(row[trial_col])
        if is_amme:
            raw_tt = str(row.get('trial_type', '')).strip().lower()
            if raw_tt == 'stim':
                stim_map[tid] = 'S'
            elif raw_tt == 'nostim':
                stim_map[tid] = 'NS'
            # 'new' / foils -> not an old item, leave unmapped
        else:
            # BLAES: stimulation flag only meaningful for old (Targ) items
            raw_tt = str(row.get('trial_type', '')).strip().lower()
            if raw_tt in ('targ', 'old', 'stim', 'nostim'):
                stim = row.get('stimulation')
                if pd.notna(stim):
                    stim_map[tid] = 'S' if int(stim) == 1 else 'NS'
    return stim_map


def main():
    ied = pd.read_csv(RET_CSV)
    old = ied[ied['MemoryOutcome'].isin(['remembered', 'forgotten'])].copy()
    print(f'Retrieval IED rows (old items): {len(old)} across {old["Patient"].nunique()} patients')

    # Collapse channels -> trial level: IED present (Y) if any channel had a Y.
    def any_y(x):
        return 'Y' if (x == 'Y').any() else 'N'

    trial = old.groupby(['Patient', 'Trial', 'MemoryOutcome']).agg(
        BeforeImg=('BeforeImgITI', any_y),
        DuringImg=('DuringImgITI', any_y),
    ).reset_index()

    # Attach prior stimulation per (patient, trial)
    stim_maps = {p: build_prior_stim_map(p) for p in sorted(trial['Patient'].unique())}
    missing = [p for p, m in stim_maps.items() if m is None]
    if missing:
        print(f'  WARNING: no phase-3 file for: {missing}')

    def get_stim(row):
        m = stim_maps.get(row['Patient'])
        if not m:
            return np.nan
        return m.get(int(row['Trial']), np.nan)

    trial['prior_stim'] = trial.apply(get_stim, axis=1)

    n_before = len(trial)
    unmapped = trial[trial['prior_stim'].isna()]
    if len(unmapped):
        print(f'  {len(unmapped)} old-item trials had no prior_stim match '
              f'(dropped); per patient: '
              f'{unmapped.groupby("Patient")["Trial"].nunique().to_dict()}')
    trial = trial[trial['prior_stim'].notna()].copy()

    trial['memory'] = (trial['MemoryOutcome'] == 'remembered').astype(int)

    trial = trial[['Patient', 'Trial', 'MemoryOutcome', 'memory',
                   'prior_stim', 'BeforeImg', 'DuringImg']]
    trial.to_csv(OUT_CSV, index=False)

    print(f'\nSaved {OUT_CSV}')
    print(f'Trials: {len(trial)} (from {n_before} old-item trials), '
          f'patients: {trial["Patient"].nunique()}')
    print(f"  remembered={int(trial['memory'].sum())}, "
          f"forgotten={int((1 - trial['memory']).sum())}")
    print('\nprior_stim x memory:')
    print(pd.crosstab(trial['prior_stim'], trial['MemoryOutcome']))
    print('\nBeforeImg IED present x memory:')
    print(pd.crosstab(trial['BeforeImg'], trial['MemoryOutcome']))
    print('\nDuringImg IED present x memory:')
    print(pd.crosstab(trial['DuringImg'], trial['MemoryOutcome']))
    print('\nprior_stim x DuringImg:')
    print(pd.crosstab(trial['prior_stim'], trial['DuringImg']))


if __name__ == '__main__':
    main()
