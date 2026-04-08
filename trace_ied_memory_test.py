#!/usr/bin/env python
"""
Trace IED Test Trials to Memory Outcome
========================================
For each trial in the IED test CSV, determine whether the patient
correctly recalled the item (hit) or missed it, based on phase 3 data.

For old items (Targ/stim/nostim):
  response='old'/code in (78,66,37) or yes_or_no='yes' → remembered (hit)
  response='new'/code in (67,86,39) or yes_or_no='no'  → forgotten (miss)

For new items (foils): labeled as 'new_item' (not memory failures).

Outputs:
  IED/AMMEBLAES_IEDs_trial_level_dissertation_test_usethis_cleaned_with_memory.csv
"""

import os
import glob
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IED_CSV = os.path.join(SCRIPT_DIR, 'IED',
                       'AMMEBLAES_IEDs_trial_level_dissertation_test_usethis_cleaned.csv')
OUTPUT_CSV = os.path.join(SCRIPT_DIR, 'IED',
                          'AMMEBLAES_IEDs_trial_level_dissertation_test_usethis_cleaned_with_memory.csv')

BLAES_CSV_DIR = os.path.join(SCRIPT_DIR, '..', 'LFP_analyses', 'Results_CSVOutput')
AMME_CSV_DIR = os.path.join(SCRIPT_DIR, '..', '..', '..', 'AMME_Data_Emory',
                            'AMME_Data', 'LFP_analyses_Martina', 'Results_CSVOutput')


def normalize_response_blaes(val):
    if pd.isna(val):
        return None
    s = str(val).strip().lower()
    if s in ('yes', 'old'):
        return 'old'
    if s in ('no', 'new'):
        return 'new'
    try:
        code = int(float(val))
    except (ValueError, TypeError):
        return None
    if code in (78, 66, 37):
        return 'old'
    if code in (67, 86, 39):
        return 'new'
    return None


def normalize_trial_type(val):
    if pd.isna(val):
        return None
    s = str(val).strip().lower()
    if s in ('old', 'targ', 'stim', 'nostim'):
        return 'old'
    if s in ('new', 'foil', 'lure'):
        return 'new'
    return None


def build_blaes_test_memory_map(patient):
    """Map phase 3 trial number → memory outcome for BLAES patient."""
    p3_file = os.path.join(BLAES_CSV_DIR, f'Data_{patient}phase3MeasurePower.csv')
    if not os.path.exists(p3_file):
        return None, "missing phase 3 power file"

    p3 = pd.read_csv(p3_file)
    first_chan = p3['ChanName'].iloc[0]
    p3 = p3[p3['ChanName'] == first_chan].copy()
    p3 = p3.sort_values('trialIdx').reset_index(drop=True)

    p3['trial_type_norm'] = p3['trial_type'].apply(normalize_trial_type)
    p3['norm_response'] = p3['response'].apply(normalize_response_blaes)

    memory_map = {}
    stim_map = {}
    for _, row in p3.iterrows():
        tidx = int(row['trialIdx'])
        tt = row['trial_type_norm']

        # Get stim condition if available
        if 'stim_cond' in p3.columns:
            stim_map[tidx] = row['stim_cond']
        elif 'trial_type' in p3.columns:
            raw_tt = str(row['trial_type']).strip().lower()
            if raw_tt in ('stim',):
                stim_map[tidx] = 'S'
            elif raw_tt in ('nostim',):
                stim_map[tidx] = 'NS'

        if tt == 'new':
            memory_map[tidx] = 'new_item'
        elif tt == 'old':
            resp = row['norm_response']
            if resp == 'old':
                memory_map[tidx] = 'remembered'
            elif resp == 'new':
                memory_map[tidx] = 'forgotten'
            else:
                memory_map[tidx] = None
        else:
            memory_map[tidx] = None

    n_old = sum(1 for v in memory_map.values() if v in ('remembered', 'forgotten'))
    n_new = sum(1 for v in memory_map.values() if v == 'new_item')
    return memory_map, f"old={n_old}, new_items={n_new}, total={len(p3)}"


def build_amme_test_memory_map(patient):
    """Map phase 3 trial number → memory outcome for AMME patient."""
    p3_file = os.path.join(AMME_CSV_DIR, f'Data_{patient}phase3MeasurePower.csv')
    if not os.path.exists(p3_file):
        return None, "missing phase 3 power file"

    p3 = pd.read_csv(p3_file)
    if 'Region' in p3.columns:
        p3 = p3[p3['Region'] == p3['Region'].iloc[0]].copy()
    elif 'ChanName' in p3.columns:
        p3 = p3[p3['ChanName'] == p3['ChanName'].iloc[0]].copy()

    trial_col = 'number' if 'number' in p3.columns else 'trialIdx'
    p3 = p3.sort_values(trial_col).reset_index(drop=True)

    memory_map = {}
    stim_map = {}
    for _, row in p3.iterrows():
        num = int(row[trial_col])
        raw_tt = str(row['trial_type']).strip().lower()
        tt = normalize_trial_type(raw_tt)

        # Stim condition from trial_type for AMME
        if raw_tt == 'stim':
            stim_map[num] = 'S'
        elif raw_tt == 'nostim':
            stim_map[num] = 'NS'

        if tt == 'new':
            memory_map[num] = 'new_item'
        elif tt == 'old':
            yes_no = row.get('yes_or_no')
            if pd.notna(yes_no):
                s = str(yes_no).strip().lower()
                if s == 'yes':
                    memory_map[num] = 'remembered'
                elif s == 'no':
                    memory_map[num] = 'forgotten'
                else:
                    memory_map[num] = None
            else:
                memory_map[num] = None
        else:
            memory_map[num] = None

    n_old = sum(1 for v in memory_map.values() if v in ('remembered', 'forgotten'))
    n_new = sum(1 for v in memory_map.values() if v == 'new_item')
    return memory_map, f"old={n_old}, new_items={n_new}, total={len(p3)}"


def main():
    ied = pd.read_csv(IED_CSV)
    print(f"Loaded IED test CSV: {len(ied)} rows, {ied['Patient'].nunique()} patients")

    patients = sorted(ied['Patient'].unique())
    all_memory_maps = {}

    for pat in patients:
        is_amme = pat.lower().startswith('amyg')
        if is_amme:
            mem_map, flag = build_amme_test_memory_map(pat)
        else:
            mem_map, flag = build_blaes_test_memory_map(pat)

        if mem_map is None:
            print(f"  {pat}: SKIPPED - {flag}")
        else:
            all_memory_maps[pat] = mem_map
            n_rem = sum(1 for v in mem_map.values() if v == 'remembered')
            n_forg = sum(1 for v in mem_map.values() if v == 'forgotten')
            n_new = sum(1 for v in mem_map.values() if v == 'new_item')
            print(f"  {pat}: remembered={n_rem}, forgotten={n_forg}, new_items={n_new} | {flag}")

    def get_memory(row):
        pat = row['Patient']
        trial = row['Trial']
        mem_map = all_memory_maps.get(pat)
        if mem_map is None:
            return np.nan
        return mem_map.get(int(trial), np.nan)

    ied['MemoryOutcome'] = ied.apply(get_memory, axis=1)

    # Summary
    print(f"\n--- Summary ---")
    print(f"Total IED test rows: {len(ied)}")
    print(f"Memory outcome distribution:")
    print(ied['MemoryOutcome'].value_counts(dropna=False))

    # Per-patient trial summary
    print(f"\n--- Per-patient summary ---")
    for pat in patients:
        sub = ied[ied['Patient'] == pat]
        n_trials = sub['Trial'].nunique()
        n_rem = sub[sub['MemoryOutcome'] == 'remembered']['Trial'].nunique()
        n_forg = sub[sub['MemoryOutcome'] == 'forgotten']['Trial'].nunique()
        n_new = sub[sub['MemoryOutcome'] == 'new_item']['Trial'].nunique()
        n_na = sub[sub['MemoryOutcome'].isna()]['Trial'].nunique()
        print(f"  {pat}: {n_trials} trials -> rem={n_rem}, forg={n_forg}, new_item={n_new}, NA={n_na}")

    ied.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved: {OUTPUT_CSV}")


if __name__ == '__main__':
    main()
