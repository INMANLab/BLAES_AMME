#!/usr/bin/env python
"""Grouped bar chart: original vs onesec-filtered retrieval trial counts.

For every retrieval subject in the remembered/forgotten figure, plot two
adjacent bars:
  - Original : the subject's total retrieval trial count, taken verbatim from
               OUTPUTS/remembered_forgotten_by_subject.csv (same counts that
               drive OUTPUTS/remembered_forgotten_by_subject.png).
  - Filtered : the trial count after the one-second-stim retrieval filter.
               This is identical to Original for every patient EXCEPT AMME
               Timing and AMME Duration patients, whose non-onesec stim trials
               are removed.

Onesec retrieval filter (mirrors permutation/build_onesec_filtered_csvs.py and
onesecstim_code/onesecstim_trials.py):
  - AMME Timing   : keep trial_type in {nostim, "After stim"}; drop Before/During.
  - AMME Duration : keep trial_type in {nostim, "1s stim"};     drop "3s stim".
  - All others (BLAES + non-onesec AMME): unchanged.

Filtered is derived as Original minus the dropped stim trials counted from the
raw phase3 files (dedup on the `number` trial-id col, yes_or_no in {yes, no}),
so Original always matches the published figure and Filtered <= Original.

Output: OUTPUTS/onesec_trial_filter_counts.{png,csv}
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
OUTPUT_DIR = PROJECT_ROOT / 'OUTPUTS'
COUNTS_CSV = OUTPUT_DIR / 'remembered_forgotten_by_subject.csv'
RAW_DIR = Path(
    '/Users/martinahollearn/Library/CloudStorage/Box-Box/InmanLab/'
    'BLAES_data/dissertation/LFP_analyses/Results_CSVOutput'
)

AMME_TIMING = [f'amyg{n:03d}' for n in (45, 46, 48, 54, 57, 59, 61, 66, 72)]
AMME_DURATION = ['amyg030', 'amyg033', 'amyg034', 'amyg037']

# trial_type values KEPT by the onesec retrieval filter, per cohort.
ONESEC_KEEP = {
    'timing': {'nostim', 'After stim'},
    'duration': {'nostim', '1s stim'},
}

ORIGINAL_COLOR = '#C99700'  # darker yellow
FILTERED_COLOR = '#D1410C'  # blood orange

ID_COL = 'number'


def cohort_of(patient: str) -> str | None:
    if patient in AMME_TIMING:
        return 'timing'
    if patient in AMME_DURATION:
        return 'duration'
    return None


def count_dropped(patient: str, cohort: str) -> int:
    """Unique old-item retrieval trials removed by the onesec filter."""
    fp = RAW_DIR / f'Data_MartinaChannels{patient}phase3MeasurePower.csv'
    df = pd.read_csv(fp, usecols=[ID_COL, 'trial_type', 'yes_or_no'])
    trials = df.drop_duplicates(subset=[ID_COL])
    trials = trials[trials['yes_or_no'].isin(['yes', 'no'])]
    old = trials[trials['trial_type'] != 'new']
    kept = old[old['trial_type'].isin(ONESEC_KEEP[cohort])]
    return int(len(old) - len(kept))


def build_counts() -> pd.DataFrame:
    counts = pd.read_csv(COUNTS_CSV)
    counts['Original'] = pd.to_numeric(counts['Total'], errors='coerce').astype(int)

    filtered = []
    cohorts = []
    for _, row in counts.iterrows():
        patient = str(row['Subject'])
        coh = cohort_of(patient)
        if coh is None:
            filtered.append(int(row['Original']))
            cohorts.append('Unchanged')
        else:
            dropped = count_dropped(patient, coh)
            filtered.append(int(row['Original']) - dropped)
            cohorts.append('AMME Timing' if coh == 'timing' else 'AMME Duration')

    counts['Filtered'] = filtered
    counts['Cohort'] = cohorts
    counts['Dropped'] = counts['Original'] - counts['Filtered']
    return counts[['Subject', 'Cohort', 'Original', 'Filtered', 'Dropped']]


def plot_counts(df: pd.DataFrame, out_path: Path) -> None:
    x = np.arange(len(df))
    width = 0.42
    fig_width = max(20, len(df) * 0.62)
    fig, ax = plt.subplots(figsize=(fig_width, 9))

    ax.bar(x - width / 2, df['Original'], width,
           color=ORIGINAL_COLOR, label='Original (all retrieval trials)')
    ax.bar(x + width / 2, df['Filtered'], width,
           color=FILTERED_COLOR, label='Filtered (onesec-stim only)')

    # Mark how many trials the onesec filter removed for the affected patients.
    changed = df['Dropped'] > 0
    y_head = float(df['Original'].max())
    for idx, (_, row) in enumerate(df.iterrows()):
        if row['Dropped'] > 0:
            ax.text(idx, float(row['Original']) + y_head * 0.015,
                    f"-{int(row['Dropped'])}", ha='center', va='bottom',
                    fontsize=9, fontweight='bold', color='#B22222')

    # Color the x labels of the patients whose counts changed.
    ax.set_xticks(x)
    ax.set_xticklabels(df['Subject'], rotation=90, fontsize=10, fontweight='bold')
    for tick_label, is_changed in zip(ax.get_xticklabels(), changed):
        if is_changed:
            tick_label.set_color('#B22222')

    ax.set_xlabel('Subject', fontsize=20)
    ax.set_ylabel('Retrieval Trials', fontsize=20)
    ax.set_title('Original vs Onesec-Filtered Retrieval Trials by Subject',
                 fontsize=24, fontweight='bold')
    ax.tick_params(axis='y', labelsize=14)
    ax.legend(loc='upper left', fontsize=16, frameon=True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(0, float(df['Original'].max()) * 1.10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = build_counts()

    csv_path = OUTPUT_DIR / 'onesec_trial_filter_counts.csv'
    fig_path = OUTPUT_DIR / 'onesec_trial_filter_counts.png'
    df.to_csv(csv_path, index=False)
    plot_counts(df, fig_path)

    changed = df[df['Dropped'] > 0]
    print(df.to_string(index=False))
    print(f'\nSubjects with changed counts: {len(changed)} '
          f'(should be {len(AMME_TIMING) + len(AMME_DURATION)} AMME timing+duration)')
    print(f'Wrote {csv_path}')
    print(f'Wrote {fig_path}')


if __name__ == '__main__':
    main()
