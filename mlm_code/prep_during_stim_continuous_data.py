#!/Users/martinahollearn/anaconda3/bin/python
"""
Prep the trial set for the continuous-modulation version of the During-Stimulation
forgetting analysis.

This reproduces EXACTLY the trial set used by the categorical responder model in
build_during_stim_responder_inferential_report.py (stimulated, IED-positive
encoding trials, from patients who have >=1 During-Stimulation-window IED trial),
but instead of a 4-level responder factor it attaches each patient's CONTINUOUS
memory-modulation score (avg_stim_dprime_diff). The companion R script then fits

    forgotten ~ during_stim * dprime_diff_c + (1 | Patient)

so we can ask whether the continuous d' difference moderates the During-Stim
forgetting effect (vs. the categorical responder split).

Output: OUTPUTS/ied_timing_memory/during_stim_continuous_trials.csv
"""

from __future__ import annotations

import os
import pandas as pd

ROOT = '/Users/martinahollearn/Library/CloudStorage/Box-Box/InmanLab/BLAES_data/dissertation/AMME_BLAES'
ENC_CSV = os.path.join(
    ROOT, 'IED', 'AMMEBLAES_IEDs_trial_level_dissertation_study_usethis_cleaned_with_memory.csv'
)
BEH_CSV = os.path.join(
    ROOT, '..', 'AMMEBLAES_includedpts_firstsession_behavioral.csv'
)
OUT_DIR = os.path.join(ROOT, 'OUTPUTS', 'ied_timing_memory')
OUT_CSV = os.path.join(OUT_DIR, 'during_stim_continuous_trials.csv')

TIMING_COLS = ['BeforeImgITI', 'DuringImg', 'AfterImgITI', 'DuringStim']


def any_yes(series: pd.Series) -> str:
    return 'Y' if (series == 'Y').any() else 'N'


def dprime_lookup() -> pd.Series:
    """Patient -> avg_stim_dprime_diff, with the BJH032<-BJH033 behavioral alias.

    BJH032 is absent from the behavioral CSV; it shares BJH033's behavior (same
    alias the behavioral figure pipeline applies)."""
    beh = pd.read_csv(BEH_CSV)
    d = pd.to_numeric(beh.set_index('Patient')['avg_stim_dprime_diff'], errors='coerce')
    if 'BJH032' not in d.index and 'BJH033' in d.index:
        d.loc['BJH032'] = d.loc['BJH033']
    return d


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    ied = pd.read_csv(ENC_CSV)
    ied = ied[ied['MemoryOutcome'].isin(['remembered', 'forgotten'])].copy()

    trial = (
        ied.groupby(['Patient', 'Trial', 'MemoryOutcome', 'StimCond'])[TIMING_COLS]
        .agg(any_yes)
        .reset_index()
    )
    trial['forgotten'] = (trial['MemoryOutcome'] == 'forgotten').astype(int)
    trial['during_stim'] = (trial['DuringStim'] == 'Y').astype(int)

    # Patients that contribute at least one During-Stim-window IED trial (matches
    # the categorical responder analysis cohort definition).
    stim_window_patients = [
        p for p in sorted(trial['Patient'].dropna().unique())
        if ((trial['Patient'] == p) & (trial['during_stim'] == 1)).sum() > 0
    ]

    dprime = dprime_lookup()
    trial['dprime_diff'] = trial['Patient'].map(dprime)

    df = trial[
        (trial['Patient'].isin(stim_window_patients)) &
        (trial['StimCond'] == 'S') &
        (trial['dprime_diff'].notna())
    ].copy()

    df = df[['Patient', 'Trial', 'forgotten', 'during_stim', 'dprime_diff']]
    df.to_csv(OUT_CSV, index=False)

    print(f'Wrote {OUT_CSV}')
    print(f'  trials: {len(df)} | patients: {df["Patient"].nunique()} '
          f'| forgotten: {int(df["forgotten"].sum())} '
          f'| during_stim trials: {int(df["during_stim"].sum())}')
    print(f'  d\' diff range: {df["dprime_diff"].min():+.3f} to {df["dprime_diff"].max():+.3f}, '
          f'mean {df["dprime_diff"].mean():+.3f}, SD {df["dprime_diff"].std():.3f}')


if __name__ == '__main__':
    main()
