#!/usr/bin/env python
"""
Trace IED Trials to Subsequent Memory Outcome
==============================================
For each trial in the IED CSV, determine whether it was later
remembered or forgotten.

Strategy (from to_combine scripts):
  BLAES subjects — PAC_TG phase 1 files have `ret_response` pre-merged
      from MATLAB: 'Old' → remembered, 'New' → forgotten.
      Power phase 1 files provide ALL encoding trials for sequential
      trial numbering; PAC file provides memory labels via trialIdx join.
  AMME subjects — Power phase 1 files have `test_yes_or_no` pre-merged:
      'yes' → remembered, 'no' → forgotten.

For subjects where not all encoding items were retested at retrieval,
the non-retested trials receive NaN (expected for experiments that only
test a subset of encoding items).

Outputs:
  IED/AMMEBLAES_IEDs_trial_level_dissertation_study_usethis_cleaned_with_memory.csv
"""

import os
import glob
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IED_CSV = os.path.join(SCRIPT_DIR, 'IED',
                       'AMMEBLAES_IEDs_trial_level_dissertation_study_usethis_cleaned.csv')
OUTPUT_CSV = os.path.join(SCRIPT_DIR, 'IED',
                          'AMMEBLAES_IEDs_trial_level_dissertation_study_usethis_cleaned_with_memory.csv')

BLAES_CSV_DIR = os.path.join(SCRIPT_DIR, '..', 'LFP_analyses', 'Results_CSVOutput')
AMME_CSV_DIR = os.path.join(SCRIPT_DIR, '..', '..', '..', 'AMME_Data_Emory',
                            'AMME_Data', 'LFP_analyses_Martina', 'Results_CSVOutput')
AMME_PHASE1_DIR = os.path.join(AMME_CSV_DIR, 'Phase1')

# Memory labels from to_combine/BLAES_Group_PAC_analyses_from_matlab_encoding.py
MEMORY_LABELS = {'Old': 'remembered', 'New': 'forgotten'}


# ---------------------------------------------------------------------------
# Response normalization helpers
# ---------------------------------------------------------------------------
def normalize_response_blaes(val):
    """Normalize BLAES phase 3 response to 'old' or 'new'."""
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
    """Return 'old' or 'new' (foil/lure/new)."""
    if pd.isna(val):
        return None
    s = str(val).strip().lower()
    if s in ('old', 'targ'):
        return 'old'
    if s in ('new', 'foil', 'lure'):
        return 'new'
    return None


def normalize_image_name(x):
    """Normalize image path to just the lowercase filename."""
    if pd.isna(x):
        return None
    x = str(x).strip().replace("\\", "/")
    return os.path.basename(x).lower()


# ---------------------------------------------------------------------------
# Build memory lookup for BLAES subjects
# ---------------------------------------------------------------------------
def build_blaes_memory_map(patient):
    """Return dict: encoding_trial_number (1-based sequential) -> 'remembered'/'forgotten'/None.

    Uses PAC_TG phase 1 (which has ret_response pre-merged from MATLAB)
    linked to Power phase 1 (which has all encoding trials) via trialIdx.
    Falls back to stimulus_code matching if PAC file unavailable.
    """
    p1_power_file = os.path.join(BLAES_CSV_DIR, f'Data_{patient}phase1MeasurePower.csv')
    p1_pac_file = os.path.join(BLAES_CSV_DIR, f'Data_{patient}phase1MeasurePAC_TG.csv')
    p3_file = os.path.join(BLAES_CSV_DIR, f'Data_{patient}phase3MeasurePower.csv')

    if not os.path.exists(p1_power_file):
        return None, "missing phase 1 power file"

    # Load ALL encoding trials from Power phase 1 (one channel)
    p1_pow = pd.read_csv(p1_power_file)
    first_chan = p1_pow['ChanName'].iloc[0]
    p1_pow = p1_pow[p1_pow['ChanName'] == first_chan].copy()
    p1_pow['stimulus_code'] = pd.to_numeric(p1_pow['stimulus_code'], errors='coerce')
    p1_pow = p1_pow.sort_values('trialIdx').reset_index(drop=True)
    p1_pow['enc_trial_num'] = range(1, len(p1_pow) + 1)
    n_encoding = len(p1_pow)

    # --- Strategy 1: Use PAC_TG phase 1 (has ret_response) ---
    if os.path.exists(p1_pac_file):
        p1_pac = pd.read_csv(p1_pac_file)
        if 'ret_response' in p1_pac.columns:
            first_chan_pac = p1_pac['ChanName'].iloc[0]
            p1_pac = p1_pac[p1_pac['ChanName'] == first_chan_pac].copy()
            p1_pac['memory'] = p1_pac['ret_response'].map(MEMORY_LABELS)

            # Build trialIdx -> memory from PAC
            pac_trialidx_to_memory = {}
            for _, row in p1_pac.iterrows():
                tidx = row['trialIdx']
                pac_trialidx_to_memory[tidx] = row['memory']

            # Map: enc_trial_num -> trialIdx -> memory
            memory_map = {}
            n_matched = 0
            for _, row in p1_pow.iterrows():
                trial_num = row['enc_trial_num']
                tidx = row['trialIdx']
                mem = pac_trialidx_to_memory.get(tidx)
                memory_map[trial_num] = mem
                if mem is not None:
                    n_matched += 1

            flag = None
            if n_matched < n_encoding:
                flag = (f"PAC-matched {n_matched}/{n_encoding} encoding trials "
                        f"(PAC has {len(p1_pac)} retested items)")
            return memory_map, flag

    # --- Strategy 2: Fall back to stimulus_code matching via phase 3 ---
    if not os.path.exists(p3_file):
        return None, "missing both PAC phase 1 and phase 3 power files"

    p3 = pd.read_csv(p3_file)
    first_chan3 = p3['ChanName'].iloc[0]
    p3 = p3[p3['ChanName'] == first_chan3].copy()
    p3['stimulus_code'] = pd.to_numeric(p3['stimulus_code'], errors='coerce')
    p3['trial_type_norm'] = p3['trial_type'].apply(normalize_trial_type)
    p3['norm_response'] = p3['response'].apply(normalize_response_blaes)

    # Old/Targ items: response indicates remembered/forgotten
    p3_old = p3[p3['trial_type_norm'] == 'old'].copy()
    p3_old['memory'] = p3_old['norm_response'].map({'old': 'remembered', 'new': 'forgotten'})

    n_old_p3 = len(p3_old)

    # Build stimulus_code -> memory
    stim_to_memory = {}
    for _, row in p3_old.iterrows():
        sc = row['stimulus_code']
        if pd.notna(sc):
            stim_to_memory[int(sc)] = row['memory']

    # Map encoding trials
    memory_map = {}
    n_matched = 0
    for _, row in p1_pow.iterrows():
        sc = row['stimulus_code']
        trial_num = row['enc_trial_num']
        if pd.notna(sc):
            mem = stim_to_memory.get(int(sc))
            memory_map[trial_num] = mem
            if mem is not None:
                n_matched += 1

    flag = None
    if n_matched < n_encoding:
        flag = (f"stim_code-matched {n_matched}/{n_encoding} encoding trials "
                f"(phase 3 has {n_old_p3} old items)")
    return memory_map, flag


# ---------------------------------------------------------------------------
# Build memory lookup for AMME subjects
# ---------------------------------------------------------------------------
def build_amme_memory_map(patient):
    """Return dict: encoding_trial_number -> 'remembered'/'forgotten'/None.

    AMME phase 1 Power files have test_yes_or_no pre-merged from MATLAB.
    Falls back to phase 3 matching by image name if needed.
    Also checks PAC_TG files in the main AMME CSV dir.
    """
    # Find phase 1 Power file
    p1_pattern = os.path.join(AMME_PHASE1_DIR, f'*{patient}*phase1*Power*.csv')
    p1_files = glob.glob(p1_pattern)
    if not p1_files:
        return None, "missing AMME phase 1 power file"

    p1_file = p1_files[0]
    p1 = pd.read_csv(p1_file)

    # Get one channel
    if 'ChanName' in p1.columns:
        first_chan = p1['ChanName'].iloc[0]
        p1 = p1[p1['ChanName'] == first_chan].copy()
    elif 'ChanOrder' in p1.columns:
        first_order = p1['ChanOrder'].iloc[0]
        p1 = p1[p1['ChanOrder'] == first_order].copy()

    trial_col = 'number' if 'number' in p1.columns else 'trialIdx'
    p1 = p1.sort_values(trial_col).reset_index(drop=True)
    n_encoding = len(p1)

    # --- Check for PAC_TG phase 1 in main AMME dir (has ret_response for some) ---
    pac_pattern = os.path.join(AMME_CSV_DIR, f'*{patient}*phase1*PAC*.csv')
    pac_files = glob.glob(pac_pattern)
    if pac_files:
        pac = pd.read_csv(pac_files[0])
        if 'ret_response' in pac.columns:
            # Use ret_response from PAC (same logic as BLAES)
            if 'ChanName' in pac.columns:
                pac = pac[pac['ChanName'] == pac['ChanName'].iloc[0]].copy()
            elif 'Region' in pac.columns:
                pac = pac[pac['Region'] == pac['Region'].iloc[0]].copy()
            pac['memory'] = pac['ret_response'].map(MEMORY_LABELS)
            pac_trial_col = 'number' if 'number' in pac.columns else 'trialIdx'
            pac_num_to_memory = {}
            for _, row in pac.iterrows():
                pac_num_to_memory[int(row[pac_trial_col])] = row['memory']

            memory_map = {}
            for _, row in p1.iterrows():
                num = int(row[trial_col])
                memory_map[num] = pac_num_to_memory.get(num)
            return memory_map, None

    # --- Strategy: Use test_yes_or_no from phase 1 Power ---
    if 'test_yes_or_no' in p1.columns:
        memory_map = {}
        for _, row in p1.iterrows():
            num = int(row[trial_col])
            yes_no = row.get('test_yes_or_no')
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

        n_matched = sum(1 for v in memory_map.values() if v is not None)
        flag = None
        if n_matched < n_encoding:
            flag = f"test_yes_or_no matched {n_matched}/{n_encoding} encoding trials"
        return memory_map, flag

    # --- Fallback: Match via phase 3 by image name ---
    p3_file = os.path.join(AMME_CSV_DIR, f'Data_{patient}phase3MeasurePower.csv')
    if not os.path.exists(p3_file):
        return None, "missing test_yes_or_no in phase 1 and no phase 3 file"

    p3 = pd.read_csv(p3_file)
    if 'Region' in p3.columns:
        first_reg = p3['Region'].iloc[0]
        p3 = p3[p3['Region'] == first_reg].copy()

    img_col_p1 = 'full_im_name' if 'full_im_name' in p1.columns else 'imagename'
    img_col_p3 = 'full_im_name' if 'full_im_name' in p3.columns else 'imagename'

    # For AMME phase 3: trial_type 'stim'/'nostim' = old items, 'new' = foils
    p3_old = p3[~p3['trial_type'].str.lower().isin(['new'])].copy()
    img_to_memory = {}
    for _, row in p3_old.iterrows():
        img = normalize_image_name(row.get(img_col_p3))
        yes_no = row.get('yes_or_no')
        if img and pd.notna(yes_no):
            s = str(yes_no).strip().lower()
            if s == 'yes':
                img_to_memory[img] = 'remembered'
            elif s == 'no':
                img_to_memory[img] = 'forgotten'

    memory_map = {}
    for _, row in p1.iterrows():
        num = int(row[trial_col])
        img = normalize_image_name(row.get(img_col_p1))
        if img:
            memory_map[num] = img_to_memory.get(img)

    n_matched = sum(1 for v in memory_map.values() if v is not None)
    flag = None
    if n_matched < n_encoding:
        flag = f"image-matched {n_matched}/{n_encoding} encoding trials"
    return memory_map, flag


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ied = pd.read_csv(IED_CSV)
    print(f"Loaded IED CSV: {len(ied)} rows, {len(ied['Patient'].unique())} patients")

    patients = sorted(ied['Patient'].unique())

    # Build memory maps for all patients
    all_memory_maps = {}
    flags = []

    for pat in patients:
        is_amme = pat.lower().startswith('amyg')
        if is_amme:
            mem_map, flag = build_amme_memory_map(pat)
        else:
            mem_map, flag = build_blaes_memory_map(pat)

        if mem_map is None:
            print(f"  {pat}: SKIPPED - {flag}")
            flags.append({'Patient': pat, 'Flag': flag})
        else:
            all_memory_maps[pat] = mem_map
            n_rem = sum(1 for v in mem_map.values() if v == 'remembered')
            n_forg = sum(1 for v in mem_map.values() if v == 'forgotten')
            n_none = sum(1 for v in mem_map.values() if v is None)
            status = f"remembered={n_rem}, forgotten={n_forg}, no_test_data={n_none}"
            if flag:
                status += f" | {flag}"
                flags.append({'Patient': pat, 'Flag': flag})
            print(f"  {pat}: {status}")

    # Map memory outcome to IED rows
    def get_memory(row):
        pat = row['Patient']
        trial = row['Trial']
        mem_map = all_memory_maps.get(pat)
        if mem_map is None:
            return np.nan
        return mem_map.get(int(trial), np.nan)

    ied['MemoryOutcome'] = ied.apply(get_memory, axis=1)

    # Summary
    print("\n--- Summary ---")
    matched = ied['MemoryOutcome'].notna().sum()
    total = len(ied)
    print(f"Total IED rows: {total}")
    print(f"Matched to memory: {matched} ({matched/total*100:.1f}%)")
    print(f"Unmatched: {total - matched}")
    print(f"\nMemory outcome distribution:")
    print(ied['MemoryOutcome'].value_counts(dropna=False))

    # Per-patient summary
    print("\n--- Per-patient summary ---")
    for pat in patients:
        sub = ied[ied['Patient'] == pat]
        n_trials = len(sub['Trial'].unique())
        n_rem = len(sub[sub['MemoryOutcome'] == 'remembered']['Trial'].unique())
        n_forg = len(sub[sub['MemoryOutcome'] == 'forgotten']['Trial'].unique())
        n_na = len(sub[sub['MemoryOutcome'].isna()]['Trial'].unique())
        print(f"  {pat}: {n_trials} trials -> rem={n_rem}, forg={n_forg}, NA={n_na}")

    if flags:
        print("\n--- Flags ---")
        for f in flags:
            print(f"  {f['Patient']}: {f['Flag']}")

    # Save
    ied.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved: {OUTPUT_CSV}")


if __name__ == '__main__':
    main()
