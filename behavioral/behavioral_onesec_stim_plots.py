#!/usr/bin/env python
"""
Two behavioral figures restricted to the 1-second-stim condition:

  1. Connected dot plot: nostim vs. onesec_stim d' (one patient per pair
     of connected dots).
  2. Strip plot: onesec_stim_dprime_diff (onesec_stim - nostim), one dot
     per patient, with a reference line at 0.

Input:  dissertation/AMMEBLAES_includedpts_firstsession_behavioral.csv
Output: outputs/behavioral_onesec_stim/
          onesec_stim_nostim_vs_stim_connected.png
          onesec_stim_dprime_diff_strip.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_rel

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(os.path.dirname(SCRIPT_DIR),
                        'AMMEBLAES_includedpts_firstsession_behavioral.csv')
OUT_DIR = os.path.join(SCRIPT_DIR, 'outputs', 'behavioral_onesec_stim')
os.makedirs(OUT_DIR, exist_ok=True)

NOSTIM_COLOR = '#1f77b4'  # blue
STIM_COLOR   = '#d62728'  # red


def _fmt_p(p):
    if p < 0.001:
        return 'p < .001'
    return f'p = {p:.3f}'


def connected_dot_plot(df, out_path):
    sns.set_style('ticks')
    sns.set_context('talk', font_scale=0.85)

    t_stat, p_val = ttest_rel(df['onesec_stim'], df['nostim'])

    fig, ax = plt.subplots(figsize=(3.5, 4.5))

    x0 = np.zeros(len(df))
    x1 = np.ones(len(df))

    for _, r in df.iterrows():
        ax.plot([0, 1], [r['nostim'], r['onesec_stim']],
                color='black', linewidth=0.6, alpha=0.6, zorder=1)

    ax.scatter(x0, df['nostim'], s=55, color=NOSTIM_COLOR,
               edgecolor='black', linewidth=0.4, zorder=2)
    ax.scatter(x1, df['onesec_stim'], s=55, color=STIM_COLOR,
               edgecolor='black', linewidth=0.4, zorder=2)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(['nostim', 'onesec_stim'])
    ax.set_xlim(-0.35, 1.35)
    ax.set_ylabel("D' value")
    ax.set_title(
        f"nostim vs. 1 s stim  (n = {len(df)})\n"
        f"paired t({len(df) - 1}) = {t_stat:.2f}, {_fmt_p(p_val)}",
        fontsize=11, fontweight='bold')

    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Saved {out_path}  |  paired t({len(df) - 1}) = {t_stat:.3f}, '
          f'p = {p_val:.4g}')


def swarm_plot_diff(df, out_path):
    sns.set_style('ticks')
    sns.set_context('talk', font_scale=0.85)

    fig, ax = plt.subplots(figsize=(3.5, 4.5))

    ax.axhline(0, color='black', linestyle='--', linewidth=0.8, zorder=1)
    sns.swarmplot(y=df['onesec_stim_dprime_diff'].values, ax=ax,
                  color='black', size=5.5, linewidth=0.4,
                  edgecolor='black')

    ax.set_xticks([])
    ax.set_xlabel('')
    ax.set_ylabel("D' difference (1 s stim - nostim)")
    ax.set_title(f"onesec_stim d' diff  (n = {len(df)})",
                 fontsize=12, fontweight='bold')

    sns.despine(ax=ax, bottom=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Saved {out_path}')


def main():
    df = pd.read_csv(CSV_PATH)
    paired = df[['Patient', 'nostim', 'onesec_stim',
                 'onesec_stim_dprime_diff']].dropna(
                     subset=['nostim', 'onesec_stim']).reset_index(drop=True)
    print(f'Loaded {len(df)} rows; {len(paired)} with complete '
          f'nostim & onesec_stim.')

    connected_dot_plot(paired,
        os.path.join(OUT_DIR, 'onesec_stim_nostim_vs_stim_connected.png'))

    strip_df = df[['Patient', 'onesec_stim_dprime_diff']].dropna(
        subset=['onesec_stim_dprime_diff']).reset_index(drop=True)
    swarm_plot_diff(strip_df,
        os.path.join(OUT_DIR, 'onesec_stim_dprime_diff_swarm.png'))


if __name__ == '__main__':
    main()
