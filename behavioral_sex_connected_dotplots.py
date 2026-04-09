#!/usr/bin/env python
"""Create sex-split connected dot plots for full and balanced behavioral subjects."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import ttest_rel

from behavioral_figures_balanced_trials import (
    VALID_SEX,
    apply_balanced_filter,
    load_behavior_csv,
    load_trial_counts,
)


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / 'outputs' / 'behavioral_figures_by_sex'
SUMMARY_CSV = OUTPUT_DIR / 'sex_connected_dotplot_summary.csv'

POINT_COLORS = {'nostim': 'blue', 'stim': 'red'}
SEX_ORDER = ['female', 'male']
SEX_TITLES = {'female': 'Women', 'male': 'Men'}


def prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out['sex'] = out['sex'].astype(str).str.strip().str.lower()
    out['avg_stim'] = pd.to_numeric(out['avg_stim'], errors='coerce')
    out['nostim'] = pd.to_numeric(out['nostim'], errors='coerce')
    out = out[out['sex'].isin(VALID_SEX)].copy()
    out = out.dropna(subset=['avg_stim', 'nostim']).copy()
    return out


def build_summary_rows(df: pd.DataFrame, dataset_label: str) -> list[dict]:
    rows = []
    for sex in SEX_ORDER:
        sub = df[df['sex'] == sex].copy()
        if sub.empty:
            rows.append({
                'dataset': dataset_label,
                'sex': sex,
                'n_subjects': 0,
                'mean_nostim': None,
                'mean_stim': None,
                'mean_delta_stim_minus_nostim': None,
                'paired_t': None,
                'paired_p': None,
            })
            continue

        t_stat, p_val = ttest_rel(sub['avg_stim'], sub['nostim'])
        rows.append({
            'dataset': dataset_label,
            'sex': sex,
            'n_subjects': int(len(sub)),
            'mean_nostim': float(sub['nostim'].mean()),
            'mean_stim': float(sub['avg_stim'].mean()),
            'mean_delta_stim_minus_nostim': float((sub['avg_stim'] - sub['nostim']).mean()),
            'paired_t': float(t_stat),
            'paired_p': float(p_val),
        })
    return rows


def plot_sex_subfigures(df: pd.DataFrame, dataset_label: str, out_name: str) -> list[dict]:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 7.2), sharey=True)
    summary_rows = []

    for ax, sex in zip(axes, SEX_ORDER):
        sub = df[df['sex'] == sex].copy().sort_values('Patient')
        summary_rows.extend(build_summary_rows(df, dataset_label)[SEX_ORDER.index(sex):SEX_ORDER.index(sex)+1])

        if sub.empty:
            ax.text(0.5, 0.5, 'No subjects', transform=ax.transAxes, ha='center', va='center', fontsize=14)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(['nostim', 'stim'], fontsize=14, fontweight='bold')
            continue

        t_stat, p_val = ttest_rel(sub['avg_stim'], sub['nostim'])
        for _, row in sub.iterrows():
            ax.plot([0, 1], [row['nostim'], row['avg_stim']], color='black', linewidth=1.25, alpha=0.45, zorder=1)
            ax.scatter(0, row['nostim'], color=POINT_COLORS['nostim'], s=72, zorder=3)
            ax.scatter(1, row['avg_stim'], color=POINT_COLORS['stim'], s=72, zorder=3)

        ax.set_title(
            f"{SEX_TITLES[sex]} (N={len(sub)})\n"
            f"mean nostim={sub['nostim'].mean():.2f}, mean stim={sub['avg_stim'].mean():.2f}",
            fontsize=14,
            pad=12,
        )
        ax.text(
            0.5,
            0.90,
            f"t = {t_stat:.3f}\np = {p_val:.3f}\nΔ stim-nostim = {(sub['avg_stim'] - sub['nostim']).mean():.3f}",
            transform=ax.transAxes,
            ha='center',
            va='top',
            fontsize=11,
        )
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['nostim', 'stim'], fontsize=14, fontweight='bold')
        ax.tick_params(axis='y', labelsize=12)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    axes[0].set_ylabel('Dprime value', fontsize=16, fontweight='bold')
    fig.suptitle(
        f'AMME & BLAES stim vs. nostim dprime by sex\n{dataset_label}',
        fontsize=18,
        y=0.985,
    )
    fig.subplots_adjust(top=0.76, wspace=0.20, bottom=0.10)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / out_name, dpi=300, bbox_inches='tight', pad_inches=0.35)
    plt.close(fig)
    return summary_rows


def main() -> None:
    behavior_df = load_behavior_csv()
    counts_df = load_trial_counts()
    _, balanced_df = apply_balanced_filter(behavior_df, counts_df)

    full_df = prepare_dataset(behavior_df)
    balanced_plot_df = prepare_dataset(balanced_df)

    summary_rows = []
    summary_rows.extend(plot_sex_subfigures(
        full_df,
        dataset_label=f'All behavioral subjects (N={len(full_df)})',
        out_name='AMMEBLAES_connected_dotplot_by_sex_all_subjects.png',
    ))
    summary_rows.extend(plot_sex_subfigures(
        balanced_plot_df,
        dataset_label=f'Balanced-trials subjects (N={len(balanced_plot_df)})',
        out_name='AMMEBLAES_connected_dotplot_by_sex_balanced_trials.png',
    ))

    pd.DataFrame(summary_rows).to_csv(SUMMARY_CSV, index=False)
    print(f'All-subject rows plotted: {len(full_df)}')
    print(f'Balanced-subject rows plotted: {len(balanced_plot_df)}')
    print(f'Output directory: {OUTPUT_DIR}')
    print(f'Wrote summary CSV: {SUMMARY_CSV}')


if __name__ == '__main__':
    main()
