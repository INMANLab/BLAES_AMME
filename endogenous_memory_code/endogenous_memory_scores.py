#!/usr/bin/env python
"""
Compute no-stim endogenous memory scores from combined power MLMR exports.

Logic
-----
1. Use only no-stim trials from the existing trial-level baseline-corrected power CSVs.
2. Collapse each trial's spectrum into a band mean (Theta, Slow gamma, HFA).
3. Within each patient x region x phase x band:
   - mean remembered no-stim band power
   - mean forgotten no-stim band power
   - raw endogenous memory effect = remembered - forgotten
   - standardized endogenous memory score = (remembered - forgotten) / SD(all no-stim trials)
4. Build a patient-level composite per phase x band by averaging region scores with
   harmonic-mean trial-count weights so regions with more balanced remembered and
   forgotten evidence contribute more.

Positive values mean higher baseline-corrected power on no-stim remembered trials.
"""

import os
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_ROOT = os.path.join(SCRIPT_DIR, 'outputs')
CSV_OUTPUT_DIR = os.path.join(OUTPUT_ROOT, 'csvs')

# Theta and Slow gamma match the combined power scripts.
# HFA is added here explicitly for endogenous scoring. The current spectra end at ~100 Hz,
# so HFA is defined as 70-100 Hz to avoid overlap with slow gamma and the 60 Hz range.
BAND_RANGES = {
    'Theta': (4, 8),
    'Slow gamma': (30, 55),
    'HFA': (70, 100),
}

MIN_TRIALS_PER_MEMORY = 3

PHASE_GROUP_FILES = {
    ('encoding', 'all'): os.path.join(CSV_OUTPUT_DIR, 'combined_encoding_power_all_mlmr_input.csv'),
    ('encoding', 'blaes'): os.path.join(CSV_OUTPUT_DIR, 'combined_encoding_power_blaes_mlmr_input.csv'),
    ('encoding', 'amme'): os.path.join(CSV_OUTPUT_DIR, 'combined_encoding_power_amme_mlmr_input.csv'),
    ('retrieval', 'all'): os.path.join(CSV_OUTPUT_DIR, 'combined_retrieval_power_all_mlmr_input.csv'),
    ('retrieval', 'blaes'): os.path.join(CSV_OUTPUT_DIR, 'combined_retrieval_power_blaes_mlmr_input.csv'),
    ('retrieval', 'amme'): os.path.join(CSV_OUTPUT_DIR, 'combined_retrieval_power_amme_mlmr_input.csv'),
}

REGION_SCORE_CSV = os.path.join(CSV_OUTPUT_DIR, 'endogenous_memory_power_region_scores.csv')
PATIENT_COMPOSITE_CSV = os.path.join(CSV_OUTPUT_DIR, 'endogenous_memory_power_patient_composites.csv')


def sorted_freq_cols(df: pd.DataFrame, prefix: str) -> List[str]:
    cols = [c for c in df.columns if c.startswith(prefix)]
    return sorted(cols, key=lambda x: float(x.split(prefix)[1]))


def freq_vals(cols: Iterable[str], prefix: str) -> np.ndarray:
    return np.array([float(c.split(prefix)[1]) for c in cols], dtype=float)


def harmonic_mean(a: int, b: int) -> float:
    if a <= 0 or b <= 0:
        return np.nan
    return 2.0 / ((1.0 / a) + (1.0 / b))


def pooled_sd(values_a: np.ndarray, values_b: np.ndarray) -> float:
    n_a = len(values_a)
    n_b = len(values_b)
    if n_a < 2 or n_b < 2:
        return np.nan
    var_a = np.var(values_a, ddof=1)
    var_b = np.var(values_b, ddof=1)
    denom = n_a + n_b - 2
    if denom <= 0:
        return np.nan
    pooled_var = (((n_a - 1) * var_a) + ((n_b - 1) * var_b)) / denom
    if pooled_var <= 0:
        return np.nan
    return float(np.sqrt(pooled_var))


def hedges_g(raw_diff: float, pooled_sd_value: float, n_a: int, n_b: int) -> float:
    if not np.isfinite(raw_diff) or not np.isfinite(pooled_sd_value) or pooled_sd_value <= 0:
        return np.nan
    d_value = raw_diff / pooled_sd_value
    correction = 1.0
    total_n = n_a + n_b
    if total_n > 3:
        correction = 1.0 - (3.0 / ((4.0 * total_n) - 9.0))
    return float(d_value * correction)


def load_trial_band_rows(csv_path: str, phase: str, group: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df[df['trial_type'] == 'nostim'].copy()
    df = df[df['yes_or_no'].isin(['yes', 'no'])].copy()
    if df.empty:
        return pd.DataFrame()

    freq_cols = sorted_freq_cols(df, 'diff_Freq_')
    freqs = freq_vals(freq_cols, 'diff_Freq_')

    rows = []
    memory_map = {'yes': 'remembered', 'no': 'forgotten'}
    base = df[['Patient', 'Region', 'yes_or_no']].copy()
    base['memory_cond'] = base['yes_or_no'].map(memory_map)

    for band, (lo, hi) in BAND_RANGES.items():
        mask = (freqs >= lo) & (freqs <= hi)
        if not np.any(mask):
            continue
        band_df = base.copy()
        band_df['phase'] = phase
        band_df['group'] = group
        band_df['band'] = band
        band_df['band_power'] = df.loc[:, np.array(freq_cols)[mask]].mean(axis=1).to_numpy(dtype=float)
        rows.append(band_df[['group', 'phase', 'Patient', 'Region', 'band', 'memory_cond', 'band_power']])

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def compute_region_scores(trial_band_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ['group', 'phase', 'Patient', 'Region', 'band']
    for keys, sub in trial_band_df.groupby(group_cols, sort=True):
        remembered = sub.loc[sub['memory_cond'] == 'remembered', 'band_power'].to_numpy(dtype=float)
        forgotten = sub.loc[sub['memory_cond'] == 'forgotten', 'band_power'].to_numpy(dtype=float)
        n_rem = len(remembered)
        n_forg = len(forgotten)
        if n_rem == 0 or n_forg == 0:
            continue

        mean_rem = float(np.mean(remembered))
        mean_forg = float(np.mean(forgotten))
        raw_diff = mean_rem - mean_forg

        all_vals = np.concatenate([remembered, forgotten]).astype(float)
        sd_all = float(np.std(all_vals, ddof=1)) if len(all_vals) > 1 else np.nan
        standardized = raw_diff / sd_all if np.isfinite(sd_all) and sd_all > 0 else np.nan

        pooled = pooled_sd(remembered, forgotten)
        g_value = hedges_g(raw_diff, pooled, n_rem, n_forg)
        n_eff = harmonic_mean(n_rem, n_forg)
        eligible = bool(
            n_rem >= MIN_TRIALS_PER_MEMORY
            and n_forg >= MIN_TRIALS_PER_MEMORY
            and np.isfinite(standardized)
        )

        rows.append({
            'group': keys[0],
            'phase': keys[1],
            'Patient': keys[2],
            'Region': keys[3],
            'band': keys[4],
            'n_remembered_nostim_trials': n_rem,
            'n_forgotten_nostim_trials': n_forg,
            'n_total_nostim_trials': n_rem + n_forg,
            'mean_remembered_band_power': mean_rem,
            'mean_forgotten_band_power': mean_forg,
            'rem_minus_forg_raw': raw_diff,
            'sd_all_nostim_band_power': sd_all,
            'standardized_endogenous_memory_score': standardized,
            'pooled_sd_rem_forg': pooled,
            'hedges_g_rem_minus_forg': g_value,
            'harmonic_mean_trial_weight': n_eff,
            'eligible_for_patient_composite': eligible,
        })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    finite = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(finite):
        return np.nan
    return float(np.average(values[finite], weights=weights[finite]))


def compute_patient_composites(region_scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ['group', 'phase', 'Patient', 'band']
    eligible_df = region_scores[region_scores['eligible_for_patient_composite']].copy()
    for keys, sub in eligible_df.groupby(group_cols, sort=True):
        weights = sub['harmonic_mean_trial_weight'].to_numpy(dtype=float)
        raw_values = sub['rem_minus_forg_raw'].to_numpy(dtype=float)
        standardized_values = sub['standardized_endogenous_memory_score'].to_numpy(dtype=float)
        hedges_values = sub['hedges_g_rem_minus_forg'].to_numpy(dtype=float)

        rows.append({
            'group': keys[0],
            'phase': keys[1],
            'Patient': keys[2],
            'band': keys[3],
            'n_regions_used': int(len(sub)),
            'regions_used': ', '.join(sorted(sub['Region'].astype(str).unique())),
            'sum_remembered_nostim_trials': int(sub['n_remembered_nostim_trials'].sum()),
            'sum_forgotten_nostim_trials': int(sub['n_forgotten_nostim_trials'].sum()),
            'sum_harmonic_trial_weight': float(np.nansum(weights)),
            'composite_rem_minus_forg_raw': weighted_mean(raw_values, weights),
            'composite_standardized_endogenous_memory_score': weighted_mean(standardized_values, weights),
            'composite_hedges_g_rem_minus_forg': weighted_mean(hedges_values, weights),
        })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def main():
    trial_rows = []
    for (phase, group), csv_path in PHASE_GROUP_FILES.items():
        if not os.path.exists(csv_path):
            print(f'Skipping missing file: {csv_path}')
            continue
        print(f'Loading {phase} {group}: {csv_path}')
        trial_band_df = load_trial_band_rows(csv_path, phase=phase, group=group)
        if not trial_band_df.empty:
            trial_rows.append(trial_band_df)

    if not trial_rows:
        raise RuntimeError('No endogenous power trial rows were loaded.')

    trial_band_df = pd.concat(trial_rows, ignore_index=True)
    region_scores = compute_region_scores(trial_band_df)
    patient_composites = compute_patient_composites(region_scores)

    region_scores.to_csv(REGION_SCORE_CSV, index=False)
    patient_composites.to_csv(PATIENT_COMPOSITE_CSV, index=False)

    print(f'Wrote region scores: {REGION_SCORE_CSV}')
    print(f'Wrote patient composites: {PATIENT_COMPOSITE_CSV}')
    print(f'Region-score rows: {len(region_scores)}')
    print(f'Patient-composite rows: {len(patient_composites)}')


if __name__ == '__main__':
    main()
