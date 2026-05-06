#!/usr/bin/env python
"""
Figure: Power x IED-during-stim interaction on encoding memory.

Three-panel plot (Theta, Slow Gamma, HFA) showing GLMM-predicted probability
of remembering as a function of power, split by whether an IED occurred during
the stimulation window. Supports both BLA and HPC via --region flag.

Usage:
    python plot_bla_power_ied_interaction.py            # default: BLA
    python plot_bla_power_ied_interaction.py --region HPC
    python plot_bla_power_ied_interaction.py --region BLA
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / 'outputs' / 'ied_timing_memory'
FIG_DIR = DATA_DIR
MERGED_CSV = DATA_DIR / 'power_ied_merged.csv'

BANDS = [
    ('theta', 'Theta (4–8 Hz)'),
    ('slow_gamma', 'Slow Gamma (30–55 Hz)'),
    ('hfa', 'HFA (70–100 Hz)'),
]

REGION_LABELS = {'BLA': 'Amygdala', 'HPC': 'Hippocampus'}

IED_COLORS = {0: '#2CA9A0', 1: '#E075A8'}
IED_LABELS = {0: 'No IED during stim', 1: 'IED during stim'}


def load_model_coefs(region: str, band: str) -> pd.DataFrame | None:
    path = DATA_DIR / f'pow_TI_{region}_{band}_odds_ratios.csv'
    if not path.exists():
        return None
    return pd.read_csv(path)


def get_coef(df: pd.DataFrame, term: str) -> float:
    row = df[df['term'] == term]
    if len(row) == 0:
        return 0.0
    return np.log(row.iloc[0]['estimate'])


def predict_prob(intercept, b_pow, b_ied, b_interaction, power_vals, ied_val):
    logit = intercept + b_pow * power_vals + b_ied * ied_val + b_interaction * power_vals * ied_val
    return expit(logit)


def make_figure(region: str = 'BLA'):
    region_label = REGION_LABELS.get(region, region)

    merged = pd.read_csv(MERGED_CSV)
    data = merged[merged['region'] == region].copy()
    pc = data.groupby('patient_id').size()
    keep = pc[pc >= 5].index
    data = data[data['patient_id'].isin(keep)].copy()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)

    for ax, (band, band_label) in zip(axes, BANDS):
        pow_col = f'pow_{band}'

        data[f'{pow_col}_z'] = (data[pow_col] - data[pow_col].mean()) / data[pow_col].std()

        coefs_df = load_model_coefs(region, band)
        if coefs_df is None:
            ax.text(0.5, 0.5, 'Model not found', transform=ax.transAxes, ha='center')
            continue

        intercept = get_coef(coefs_df, '(Intercept)')
        b_pow = get_coef(coefs_df, f'pow_{band}_z')
        b_ied_ds = get_coef(coefs_df, 'ied_during_stim')
        interaction_term = f'pow_{band}_z:ied_during_stim'
        b_inter = get_coef(coefs_df, interaction_term)

        power_range = np.linspace(data[f'{pow_col}_z'].quantile(0.02),
                                  data[f'{pow_col}_z'].quantile(0.98), 200)

        for ied_val in [0, 1]:
            prob = predict_prob(intercept, b_pow, b_ied_ds, b_inter,
                                power_range, ied_val)
            ax.plot(power_range, prob, color=IED_COLORS[ied_val],
                    linewidth=2.5, label=IED_LABELS[ied_val])

        sub_no = data[data['ied_during_stim'] == 0]
        sub_yes = data[data['ied_during_stim'] == 1]
        for sub, ied_val in [(sub_no, 0), (sub_yes, 1)]:
            jitter_y = sub['memory'] + np.random.default_rng(42).uniform(
                -0.02, 0.02, size=len(sub))
            ax.scatter(sub[f'{pow_col}_z'], jitter_y,
                       color=IED_COLORS[ied_val], alpha=0.12, s=12,
                       edgecolors='none', zorder=1)

        or_val = coefs_df[coefs_df['term'] == interaction_term].iloc[0]
        p_val = or_val['p.value']
        if p_val < .001:
            sig_label = '***'
        elif p_val < .01:
            sig_label = '**'
        elif p_val < .05:
            sig_label = '*'
        else:
            sig_label = 'n.s.'

        x_lo = power_range[0] + (power_range[-1] - power_range[0]) * 0.15
        x_hi = power_range[0] + (power_range[-1] - power_range[0]) * 0.85
        y_bar = 1.02
        ax.plot([x_lo, x_lo, x_hi, x_hi],
                [y_bar - 0.02, y_bar, y_bar, y_bar - 0.02],
                color='black', linewidth=1.2, clip_on=False)
        label_size = 13 if sig_label != 'n.s.' else 9
        ax.text((x_lo + x_hi) / 2, y_bar + 0.01, sig_label,
                ha='center', va='bottom', fontsize=label_size,
                fontweight='bold', color='black', clip_on=False)

        ax.set_title(band_label, fontsize=12, fontweight='bold')
        ax.set_xlabel(f'{region_label} Power (z-scored)', fontsize=10)
        ax.set_ylim(-0.06, 1.15)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])

    axes[0].set_ylabel('P(Remembered)', fontsize=10)

    n_no_ied = int((data['ied_during_stim'] == 0).sum())
    n_ied = int((data['ied_during_stim'] == 1).sum())
    handles = [
        plt.Line2D([0], [0], color=IED_COLORS[0], linewidth=2.5),
        plt.Line2D([0], [0], color=IED_COLORS[1], linewidth=2.5),
        plt.Line2D([0], [0], marker='o', color='grey', alpha=0.3,
                   linestyle='None', markersize=4),
    ]
    labels = [
        f'{IED_LABELS[0]} (n = {n_no_ied})',
        f'{IED_LABELS[1]} (n = {n_ied})',
        'Individual trials',
    ]
    fig.legend(handles, labels, loc='upper center', ncol=3, fontsize=10,
               frameon=False, bbox_to_anchor=(0.5, 1.02))

    fig.suptitle(f'{region_label} Power \u00d7 IED-During-Stim Interaction on Encoding Memory',
                 fontsize=13, fontweight='bold', y=1.08)

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    tag = region.lower()
    out_png = FIG_DIR / f'{tag}_power_ied_duringstim_interaction.png'
    out_pdf = FIG_DIR / f'{tag}_power_ied_duringstim_interaction.pdf'
    fig.savefig(out_png, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(out_pdf, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Saved -> {out_png}')
    print(f'Saved -> {out_pdf}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--region', default='BLA', choices=['BLA', 'HPC'])
    args = parser.parse_args()
    make_figure(args.region)
