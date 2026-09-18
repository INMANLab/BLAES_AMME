#!/usr/bin/env python
"""Responder-status swarm plots for the IED encoding (n=33) and retrieval (n=18) subsets.

Responder categories use the GLOBAL full-cohort definition (same quartile/median
thresholds as the main behavioral figure), so each patient keeps the same
Strong/Moderate/Non/Anti label across every figure. Here we simply subset that
labelled cohort to the patients present in the IED encoding and retrieval datasets.

Outputs (in behavioral/outputs/behavioral_figures_reformatted/):
    - AMMEBLAES_responder_status_ied_encoding_retrieval_swarmplot.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from behavioral_figures_reformatted import (
    OUTPUT_DIR,
    RESPONDER_ORDER,
    RESPONDER_PALETTE,
    add_responder_groups,
    load_behavior_csv,
)

IED_DIR = Path(__file__).resolve().parent.parent / 'IED'
ENCODING_IED_CSV = IED_DIR / 'AMMEBLAES_IEDs_trial_level_dissertation_study_usethis_cleaned_with_memory.csv'
RETRIEVAL_TRIALS_CSV = IED_DIR / 'ied_timing_memory' / 'retrieval_timing_memory_trials.csv'


def ied_patients(csv_path: Path) -> list[str]:
    import pandas as pd
    return sorted(pd.read_csv(csv_path)['Patient'].unique())


def add_bjh032_alias(df):
    """BJH032 is absent from the behavioral CSV; it shares BJH033's behavior.

    The main figure script applies this alias only when writing the responder
    CSV. The IED patient lists reference BJH032, so mirror the alias into the
    labelled dataframe itself before subsetting.
    """
    if 'BJH032' in set(df['Patient']) or 'BJH033' not in set(df['Patient']):
        return df
    import pandas as pd
    alias = df[df['Patient'] == 'BJH033'].copy()
    alias['Patient'] = 'BJH032'
    return pd.concat([df, alias], ignore_index=True)


def _swarm_axis(ax, plot_df, subtitle: str) -> None:
    sns.swarmplot(
        ax=ax,
        x='delay_group',
        y='avg_stim_dprime_diff',
        data=plot_df,
        hue='Responder status',
        hue_order=RESPONDER_ORDER,
        palette=RESPONDER_PALETTE,
        size=11,
    )
    ax.axhline(0, linestyle='--', color='black', linewidth=1.5, zorder=0)
    ax.set_title(subtitle, fontsize=17)
    ax.set_xlabel('')
    ax.set_xticks([])
    ax.tick_params(axis='x', length=0)
    ax.tick_params(axis='y', labelsize=14)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    if ax.get_legend() is not None:
        ax.get_legend().remove()


def plot_combined(df, enc_patients, ret_patients, out_name: str) -> None:
    def subset(patients):
        sub = df[df['Patient'].isin(patients) & df['avg_stim_dprime_diff'].notna()].copy()
        sub['delay_group'] = 'all'
        return sub

    enc_df = subset(enc_patients)
    ret_df = subset(ret_patients)

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 7.6), sharey=True)
    _swarm_axis(axes[0], enc_df, f'Encoding (N = {len(enc_df)})')
    _swarm_axis(axes[1], ret_df, f'Retrieval (N = {len(ret_df)})')

    axes[0].set_ylabel('Dprime difference (stimulated - not stimulated)', fontsize=16)
    axes[1].set_ylabel('')

    handles = [plt.Line2D([0], [0], marker='o', linestyle='', markersize=11,
                          markerfacecolor=RESPONDER_PALETTE[s], markeredgecolor='none', label=s)
               for s in RESPONDER_ORDER]
    fig.legend(handles=handles, title='Responder status', loc='upper left',
               bbox_to_anchor=(0.78, 0.93), frameon=False, fontsize=12,
               title_fontsize=14)

    fig.subplots_adjust(left=0.11, right=0.80, top=0.93, bottom=0.05, wspace=0.08)
    fig.savefig(OUTPUT_DIR / out_name, dpi=300, bbox_inches='tight', pad_inches=0.2)
    plt.close(fig)


def main() -> None:
    sns.set_style('white')
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_behavior_csv()
    df = add_responder_groups(df)
    df = add_bjh032_alias(df)

    enc_patients = ied_patients(ENCODING_IED_CSV)
    ret_patients = ied_patients(RETRIEVAL_TRIALS_CSV)
    print(f'IED encoding patients: {len(enc_patients)}')
    print(f'IED retrieval patients: {len(ret_patients)}')

    plot_combined(df, enc_patients, ret_patients,
                  'AMMEBLAES_responder_status_ied_encoding_retrieval_swarmplot.png')

    print(f'Output directory: {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
