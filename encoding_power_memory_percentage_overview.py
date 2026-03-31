#!/usr/bin/env python
"""Create remembered/forgotten percentage overview figures for encoding power."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from balanced_memory_trials import (
    MIN_TRIALS_PER_CONDITION,
    get_nostim_trial_counts,
    summarize_balanced_subjects,
)
from combined_encoding_power import (
    build_all_encoding_data,
    load_amme_encoding,
    load_blaes_encoding,
)
from endogenous_memory import filter_pnas_regions
from permutation_testing_unbalanced_memory_conditions import (
    OUTPUT_ROOT as PERMUTATION_OUTPUT_ROOT,
    compute_subject_selection,
    ensure_dir,
    load_power_encoding_trials,
    plot_memory_percent_across_subjects,
    plot_memory_percent_by_subject,
)


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = SCRIPT_DIR / 'outputs'
SUMMARY_PATH = OUTPUT_ROOT / 'encoding_power_memory_percentage_overview_summary.txt'
CSV_PATH = OUTPUT_ROOT / 'encoding_power_memory_percentage_overview_subjects.csv'


def get_balanced_memory_subjects() -> set[str]:
    blaes_enc = load_blaes_encoding()
    amme_enc = load_amme_encoding()
    all_enc = filter_pnas_regions(build_all_encoding_data(blaes_enc, amme_enc))
    trial_counts = get_nostim_trial_counts(all_enc, measure='power')
    summary = summarize_balanced_subjects(trial_counts, min_trials=MIN_TRIALS_PER_CONDITION)
    return set(summary['included'])


def add_percentage_columns(selection_df: pd.DataFrame) -> pd.DataFrame:
    plot_df = selection_df.copy()
    totals = plot_df['remembered'] + plot_df['forgotten']
    plot_df['total_trials'] = totals
    plot_df['remembered_pct'] = np.where(totals > 0, plot_df['remembered'] / totals * 100.0, np.nan)
    plot_df['forgotten_pct'] = np.where(totals > 0, plot_df['forgotten'] / totals * 100.0, np.nan)
    return plot_df


def write_summary(plot_df: pd.DataFrame, out_path: Path):
    all_count = len(plot_df)
    balanced_count = int(plot_df['in_balanced_memory_trials'].sum())
    permutation_count = int(plot_df['in_permutation_unbalanced'].sum())
    other_count = int((plot_df['membership'] == 'other').sum())

    lines = [
        'Encoding Power Remembered/Forgotten Percentage Overview',
        '=' * 80,
        f'All subjects in encoding power trial counts: {all_count}',
        f'Remaining subjects in balanced memory trials: {balanced_count}',
        f'Subjects removed from balanced memory trials and included in permutation testing: {permutation_count}',
        f'Other subjects not in either subset: {other_count}',
        '',
        'Membership by subject:',
    ]
    for _, row in plot_df.sort_values(['membership', 'forgotten_pct', 'Patient']).iterrows():
        lines.append(
            f"  {row['Patient']}: membership={row['membership']}, remembered={int(row['remembered'])}, "
            f"forgotten={int(row['forgotten'])}, remembered_pct={row['remembered_pct']:.1f}, "
            f"forgotten_pct={row['forgotten_pct']:.1f}"
        )
    out_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def render_subset(selection_df: pd.DataFrame, stem: str, title_prefix: str):
    if selection_df.empty:
        return
    plot_memory_percent_across_subjects(
        selection_df,
        OUTPUT_ROOT / f'{stem}_across_subjects.png',
        title_prefix,
    )
    plot_memory_percent_by_subject(
        selection_df,
        OUTPUT_ROOT / f'{stem}_by_subject.png',
        title_prefix,
    )


def main():
    print('=' * 80)
    print('Encoding Power Memory Percentage Overview')
    print('=' * 80)

    ensure_dir(OUTPUT_ROOT)
    ensure_dir(PERMUTATION_OUTPUT_ROOT)

    power_df, _ = load_power_encoding_trials()
    selection_df = add_percentage_columns(compute_subject_selection(power_df))
    balanced_subjects = get_balanced_memory_subjects()

    selection_df['in_balanced_memory_trials'] = selection_df['Patient'].isin(balanced_subjects)
    selection_df['in_permutation_unbalanced'] = (
        selection_df['selected_for_permutation'] & ~selection_df['in_balanced_memory_trials']
    )
    selection_df['membership'] = np.select(
        [
            selection_df['in_permutation_unbalanced'],
            selection_df['in_balanced_memory_trials'],
        ],
        [
            'permutation_unbalanced',
            'balanced_memory_trials',
        ],
        default='other',
    )

    selection_df = selection_df.sort_values(['forgotten_pct', 'Patient']).reset_index(drop=True)
    selection_df.to_csv(CSV_PATH, index=False)
    write_summary(selection_df, SUMMARY_PATH)

    render_subset(
        selection_df,
        'encoding_power_remembered_forgotten_percentage_all_subjects',
        'Encoding Power Remembered vs Forgotten %',
    )
    render_subset(
        selection_df[selection_df['membership'] == 'permutation_unbalanced'].copy(),
        'encoding_power_remembered_forgotten_percentage_permutation_subjects',
        'Encoding Power Remembered vs Forgotten %: Permutation Subjects',
    )
    render_subset(
        selection_df[selection_df['membership'] == 'balanced_memory_trials'].copy(),
        'encoding_power_remembered_forgotten_percentage_balanced_subjects',
        'Encoding Power Remembered vs Forgotten %: Balanced Memory Subjects',
    )

    print(f'Wrote summary CSV: {CSV_PATH}')
    print(f'Wrote summary TXT: {SUMMARY_PATH}')
    print(f'Figures saved in: {OUTPUT_ROOT}')


if __name__ == '__main__':
    main()
