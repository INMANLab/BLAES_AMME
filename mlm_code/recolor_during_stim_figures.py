#!/Users/martinahollearn/anaconda3/bin/python
"""
Regenerate the During-Stimulation responder figures using the responder-category
palette from the behavioral swarm plot
(AMMEBLAES_responder_status_ied_encoding_retrieval_swarmplot.png), with updated
labels/legends.

Data come from compute_summaries() in build_during_stim_responder_report.py; the
plotting is reimplemented here so the existing PDF-report pipeline is untouched.

Changes vs. the original figures:
  - subplot titles (subtitles) removed
  - all axis labels bold
  - Forgetting panel: y-axis = "subsequent forgetting rate (%) of trials with
    post-encoding stimulation"; legend title "Stimulated only trials" with
    entries "During Stim time window" (solid) and "all other time windows"
    (hatched)
  - Burden panel: y-axis = "Mean # of IEDs in the During Stim time window per
    patient"

Outputs (to OUTPUTS/updated_IED_figures/):
  - during_stim_burden_by_responder_group.png   (forgetting + DS-IED burden)
  - during_stim_forgetting_boost_by_patient.png (per-patient boost)
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import seaborn as sns

import build_during_stim_responder_report as rpt

ROOT = rpt.ROOT
OUT_DIR = os.path.join(ROOT, 'OUTPUTS', 'updated_IED_figures')

# Responder palette used in the swarm figure.
SWARM_PALETTE = {
    'Strong responders':   '#5B1A78',
    'Moderate responders': '#B21D6B',
    'Non-responders':      '#F04646',
    'Anti-responders':     '#F79B62',
}
RESP_ORDER = rpt.RESP_ORDER

# The current location of the responder-status table (module default is stale).
RESP_CSV = os.path.join(ROOT, 'OUTPUTS', 'csvs', 'AMMEBLAES_responder_status.csv')


def plot_boost_by_patient(summary, out_path: str) -> None:
    df = summary.dropna(subset=['DS_forg_boost']).copy()
    df = df.sort_values('avg_stim_dprime_diff').reset_index(drop=True)
    colors = [SWARM_PALETTE.get(g, '#888') for g in df['responder_status']]

    sns.set_style('ticks')
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(range(len(df)), df['DS_forg_boost'],
           color=colors, edgecolor='black', linewidth=0.5)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xticks(range(len(df)))
    xlabels = [f"{p}\n({d:+.2f})" for p, d in
               zip(df['Patient'], df['avg_stim_dprime_diff'])]
    ax.set_xticklabels(xlabels, rotation=60, ha='right', fontsize=9)
    ax.set_ylabel('Forgetting boost of IED presence in the\n'
                  'During Stimulation time window (pp)',
                  fontsize=11, fontweight='bold')

    handles = [Patch(facecolor=SWARM_PALETTE[g], edgecolor='black', linewidth=0.5,
                     label=g)
               for g in RESP_ORDER if g in df['responder_status'].values]
    ax.legend(handles=handles, loc='upper right', fontsize=9, frameon=False)

    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Saved {out_path}')


def plot_burden_by_group(grp, out_path: str) -> None:
    sns.set_style('ticks')
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    g_order = [g for g in RESP_ORDER if g in grp['responder_status'].values]
    grp2 = grp.set_index('responder_status').reindex(g_order).reset_index()
    colors = [SWARM_PALETTE[g] for g in grp2['responder_status']]

    # Left: pooled %forg During-Stim vs all other windows
    x = np.arange(len(grp2))
    w = 0.35
    axes[0].bar(x - w/2, grp2['pct_forg_DS'], width=w,
                color=colors, edgecolor='black', linewidth=0.5)
    axes[0].bar(x + w/2, grp2['pct_forg_noDS'], width=w,
                color=colors, edgecolor='black', linewidth=0.5,
                alpha=0.5, hatch='//')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([g.replace(' responders', '') for g in grp2['responder_status']],
                            fontsize=10)
    axes[0].set_ylabel('subsequent forgetting rate (%) of trials\n'
                       'with post-encoding stimulation',
                       fontsize=11, fontweight='bold')
    legend_handles = [
        Patch(facecolor='#888888', edgecolor='black', linewidth=0.5,
              label='During Stim time window'),
        Patch(facecolor='#888888', edgecolor='black', linewidth=0.5,
              alpha=0.5, hatch='//', label='all other time windows'),
    ]
    axes[0].legend(handles=legend_handles, title='Stimulated only trials',
                   fontsize=9, frameon=False, loc='upper right')

    # Right: DS-IED burden
    axes[1].bar(x, grp2['mean_n_S_DS_trials'],
                color=colors, edgecolor='black', linewidth=0.5)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([g.replace(' responders', '') for g in grp2['responder_status']],
                            fontsize=10)
    axes[1].set_ylabel('Mean # of IEDs in the During Stim\ntime window per patient',
                       fontsize=11, fontweight='bold')
    for i, n in enumerate(grp2['mean_n_S_DS_trials']):
        axes[1].text(i, n + 0.15, f'{n:.1f}', ha='center', va='bottom', fontsize=9)

    sns.despine()
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Saved {out_path}')


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    rpt.RESP_CSV = RESP_CSV  # point the data loader at the current table

    summary, grp, *_ = rpt.compute_summaries()

    boost_path = os.path.join(OUT_DIR, 'during_stim_forgetting_boost_by_patient.png')
    burden_path = os.path.join(OUT_DIR, 'during_stim_burden_by_responder_group.png')

    plot_boost_by_patient(summary, boost_path)
    plot_burden_by_group(grp, burden_path)

    print(f'Wrote:\n  {boost_path}\n  {burden_path}')


if __name__ == '__main__':
    main()
