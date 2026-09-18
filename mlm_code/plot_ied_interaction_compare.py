#!/usr/bin/env python
"""
Comparison renderer for the Power x IED-during-stim interaction figure.

Generates two alternative trial-data overlays so they can be compared side by
side, for both BLA and HPC:

  style "binned" : raw 0/1 dots replaced by empirical P(remembered) per power
                   bin (quantiles within each IED group) with binomial SE error
                   bars. Points share the model's y-scale, so they show whether
                   the data tracks the predicted curve. Fewer bins for the small
                   IED group so the pink points stay visible.

  style "rug"    : individual trials kept but made legible -- marginal rug of
                   tick marks (remembered along the top, forgotten along the
                   bottom), one offset row per IED group, higher alpha. The
                   model lines stay unobscured in the middle.

Outputs (PNG) -> <repo>/OUTPUTS/updated_IED_figures/

Usage:
    python plot_ied_interaction_compare.py            # all regions, both styles
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DATA_DIR = REPO_ROOT / 'IED' / 'ied_timing_memory'
MERGED_CSV = DATA_DIR / 'power_ied_merged.csv'
OUT_DIR = REPO_ROOT / 'OUTPUTS' / 'updated_IED_figures'

BANDS = [
    ('theta', 'Theta (4–8 Hz)'),
    ('slow_gamma', 'Slow Gamma (30–55 Hz)'),
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
    logit = (intercept + b_pow * power_vals + b_ied * ied_val
             + b_interaction * power_vals * ied_val)
    return expit(logit)


def sig_label(p_val: float) -> tuple[str, int]:
    if p_val < .001:
        return '***', 13
    if p_val < .01:
        return '**', 13
    if p_val < .05:
        return '*', 13
    return 'n.s.', 9


def n_bins_for(n: int) -> int:
    """Deciles for big groups, quartiles for small ones (keeps bins populated)."""
    return int(np.clip(n // 40, 4, 10))


def draw_sig_bracket(ax, power_range, p_val, y_bar=1.02):
    txt, size = sig_label(p_val)
    x_lo = power_range[0] + (power_range[-1] - power_range[0]) * 0.15
    x_hi = power_range[0] + (power_range[-1] - power_range[0]) * 0.85
    ax.plot([x_lo, x_lo, x_hi, x_hi],
            [y_bar - 0.02, y_bar, y_bar, y_bar - 0.02],
            color='black', linewidth=1.2, clip_on=False)
    ax.text((x_lo + x_hi) / 2, y_bar + 0.01, txt,
            ha='center', va='bottom', fontsize=size,
            fontweight='bold', color='black', clip_on=False)


def overlay_binned(ax, data, pow_col):
    """Empirical P(remembered) per power-quantile bin, with binomial SE."""
    for ied_val in [0, 1]:
        sub = data[data['ied_during_stim'] == ied_val]
        if sub.empty:
            continue
        nb = n_bins_for(len(sub))
        bins = pd.qcut(sub[pow_col], nb, duplicates='drop')
        agg = sub.groupby(bins, observed=True).agg(
            x=(pow_col, 'mean'),
            p=('memory', 'mean'),
            n=('memory', 'size'),
        )
        agg['se'] = np.sqrt(agg['p'] * (1 - agg['p']) / agg['n'])
        ax.errorbar(agg['x'], agg['p'], yerr=agg['se'],
                    fmt='o', color=IED_COLORS[ied_val], ms=7,
                    mec='white', mew=0.8, capsize=3, lw=1.2,
                    alpha=0.95, zorder=5)


def overlay_rug(ax, data, pow_col):
    """Marginal rug: remembered trials along the top, forgotten along the
    bottom; one offset row per IED group."""
    # (ied_val, memory) -> y level for the tick row
    rows = {
        (0, 1): 1.05, (1, 1): 1.10,   # remembered: top
        (0, 0): -0.01, (1, 0): -0.06,  # forgotten: bottom
    }
    for (ied_val, mem), y in rows.items():
        sub = data[(data['ied_during_stim'] == ied_val) & (data['memory'] == mem)]
        if sub.empty:
            continue
        x = sub[pow_col].to_numpy()
        ax.plot(x, np.full_like(x, y, dtype=float),
                marker='|', linestyle='None', markersize=9,
                markeredgewidth=1.0, color=IED_COLORS[ied_val],
                alpha=0.45, clip_on=False, zorder=2)


def make_figure(region: str, style: str):
    region_label = REGION_LABELS.get(region, region)

    merged = pd.read_csv(MERGED_CSV)
    data = merged[merged['region'] == region].copy()
    pc = data.groupby('patient_id').size()
    keep = pc[pc >= 5].index
    data = data[data['patient_id'].isin(keep)].copy()

    fig, axes = plt.subplots(1, len(BANDS), figsize=(4.7 * len(BANDS), 4.5),
                             sharey=True)
    axes = np.atleast_1d(axes)

    for ax, (band, band_label) in zip(axes, BANDS):
        pow_col = f'pow_{band}'
        zcol = f'{pow_col}_z'
        data[zcol] = (data[pow_col] - data[pow_col].mean()) / data[pow_col].std()

        coefs_df = load_model_coefs(region, band)
        if coefs_df is None:
            ax.text(0.5, 0.5, 'Model not found', transform=ax.transAxes,
                    ha='center')
            continue

        intercept = get_coef(coefs_df, '(Intercept)')
        b_pow = get_coef(coefs_df, f'{pow_col}_z')
        b_ied_ds = get_coef(coefs_df, 'ied_during_stim')
        interaction_term = f'{pow_col}_z:ied_during_stim'
        b_inter = get_coef(coefs_df, interaction_term)

        power_range = np.linspace(data[zcol].quantile(0.02),
                                  data[zcol].quantile(0.98), 200)
        for ied_val in [0, 1]:
            prob = predict_prob(intercept, b_pow, b_ied_ds, b_inter,
                                power_range, ied_val)
            ax.plot(power_range, prob, color=IED_COLORS[ied_val],
                    linewidth=2.5, label=IED_LABELS[ied_val], zorder=4)

        if style == 'binned':
            overlay_binned(ax, data, zcol)
        else:
            overlay_rug(ax, data, zcol)

        p_val = coefs_df[coefs_df['term'] == interaction_term].iloc[0]['p.value']
        draw_sig_bracket(ax, power_range, p_val)

        ax.set_title(band_label, fontsize=12, fontweight='bold')
        ax.set_xlabel(f'{region_label} Power (z-scored)', fontsize=10)
        if style == 'rug':
            ax.set_ylim(-0.12, 1.18)
        else:
            ax.set_ylim(-0.06, 1.15)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])

    axes[0].set_ylabel('P(Remembered)', fontsize=10)

    n_no_ied = int((data['ied_during_stim'] == 0).sum())
    n_ied = int((data['ied_during_stim'] == 1).sum())
    if style == 'binned':
        third = plt.Line2D([0], [0], marker='o', color='grey', alpha=0.6,
                           linestyle='None', markersize=6)
        third_label = 'Binned P(remembered) ± SE'
    else:
        third = plt.Line2D([0], [0], marker='|', color='grey', alpha=0.6,
                           linestyle='None', markersize=9, markeredgewidth=1.0)
        third_label = 'Individual trials (rug)'
    handles = [
        plt.Line2D([0], [0], color=IED_COLORS[0], linewidth=2.5),
        plt.Line2D([0], [0], color=IED_COLORS[1], linewidth=2.5),
        third,
    ]
    labels = [
        f'{IED_LABELS[0]} (n = {n_no_ied})',
        f'{IED_LABELS[1]} (n = {n_ied})',
        third_label,
    ]
    fig.legend(handles, labels, loc='upper center', ncol=3, fontsize=10,
               frameon=False, bbox_to_anchor=(0.5, 1.02))

    fig.suptitle(
        f'{region_label} Power × IED-During-Stim Interaction on Encoding Memory',
        fontsize=13, fontweight='bold', y=1.08)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = region.lower()
    stem = f'{tag}_power_ied_duringstim_interaction_{style}'
    out_png = OUT_DIR / f'{stem}.png'
    fig.savefig(out_png, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Saved -> {out_png}')


def main():
    for region in ['BLA', 'HPC']:
        for style in ['binned', 'rug']:
            make_figure(region, style)


if __name__ == '__main__':
    main()
