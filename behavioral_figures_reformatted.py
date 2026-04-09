#!/usr/bin/env python
"""Rebuild the original behavioral figures with safer layout and spacing."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import ttest_ind, ttest_rel


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / 'outputs' / 'behavioral_figures_reformatted'
RESPONDER_CSV = OUTPUT_DIR / 'AMMEBLAES_responder_status.csv'

RESPONDER_ORDER = [
    'Strong responders',
    'Moderate responders',
    'Non-responders',
    'Anti-responders',
]
RESPONDER_PALETTE = {
    'Strong responders': '#5B1A78',
    'Moderate responders': '#B21D6B',
    'Non-responders': '#F04646',
    'Anti-responders': '#F79B62',
}
SWARM_HEIGHT = 8.2
SWARM_ASPECT = 0.78
SWARM_TITLE_Y = 0.985
SWARM_STATS_Y = 0.855
SWARM_TOP = 0.74


def load_behavior_csv() -> pd.DataFrame:
    candidates = [
        SCRIPT_DIR.parent / 'AMMEBLAES_includedpts_firstsession_behavioral.csv',
        SCRIPT_DIR / 'behavioral figures' / 'AMMEBLAES_includedpts_firstsession_behavioral.csv',
    ]
    for path in candidates:
        if path.exists():
            return pd.read_csv(path).copy()
    raise FileNotFoundError('Could not locate behavioral summary CSV')


def add_responder_groups(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    diff = pd.to_numeric(out['avg_stim_dprime_diff'], errors='coerce')
    median = diff.median()
    lower_quartile = np.percentile(diff.dropna(), 25)
    upper_quartile = np.percentile(diff.dropna(), 75)

    def assign_group(value: float) -> str:
        if value <= lower_quartile:
            return 'Anti-responders'
        if value <= median:
            return 'Non-responders'
        if value <= upper_quartile:
            return 'Moderate responders'
        return 'Strong responders'

    out['Responder status'] = diff.apply(assign_group)
    responder_df = out[['Patient', 'avg_stim_dprime_diff', 'Responder status']].copy()
    responder_alias_df = responder_df[responder_df['Patient'] == 'BJH033'].copy()
    responder_alias_df['Patient'] = 'BJH032'
    responder_df = pd.concat([responder_df, responder_alias_df], ignore_index=True)
    responder_df.sort_values('Patient').to_csv(RESPONDER_CSV, index=False)
    return out


def finalize_swarm_plot(grid, out_path: Path, legend_title: str, legend_labels: dict | None = None,
                        legend_anchor: tuple[float, float] = (1.22, 0.75), stats_text: str | None = None) -> None:
    ax = grid.ax
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.axhline(0, linestyle='--', color='black', linewidth=1.5, zorder=0)
    ax.set_xlabel('')
    ax.set_xticks([])
    ax.set_xlim(-0.18, 0.32)
    ax.tick_params(axis='y', labelsize=15)
    ax.tick_params(axis='x', length=0)

    if stats_text:
        grid.fig.text(0.5, SWARM_STATS_Y, stats_text, ha='center', va='center', fontsize=15)

    grid.fig.subplots_adjust(left=0.18, right=0.77, top=SWARM_TOP, bottom=0.09)

    legend = grid._legend
    if legend is not None:
        legend.set_bbox_to_anchor(legend_anchor)
        legend.set_frame_on(False)
        if legend_labels:
            for text, value in zip(legend.texts, legend_labels):
                text.set_text(legend_labels[value])
                text.set_fontsize(13)
        else:
            for text in legend.texts:
                text.set_fontsize(13)
        legend.set_title(legend_title, prop={'size': 16})

    grid.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0.35)
    plt.close(grid.fig)


def make_swarm(data: pd.DataFrame, hue: str | None, hue_order: list | None,
               palette, title: str, out_name: str, legend_title: str | None = None,
               legend_labels: dict | None = None, stats_text: str | None = None,
               legend_anchor: tuple[float, float] = (1.22, 0.75)) -> None:
    kwargs = dict(
        x='delay_group',
        y='avg_stim_dprime_diff',
        data=data,
        kind='swarm',
        s=150,
        height=SWARM_HEIGHT,
        aspect=SWARM_ASPECT,
    )
    if hue is None:
        kwargs['color'] = 'black'
    else:
        kwargs['hue'] = hue
        kwargs['hue_order'] = hue_order
        kwargs['palette'] = palette

    grid = sns.catplot(**kwargs)
    grid.ax.set_ylabel('Dprime difference (stimulated - not stimulated)', fontsize=18)
    grid.ax.set_title('')
    grid.fig.suptitle(title, fontsize=18, y=SWARM_TITLE_Y)
    finalize_swarm_plot(
        grid,
        OUTPUT_DIR / out_name,
        legend_title=legend_title or '',
        legend_labels=legend_labels,
        legend_anchor=legend_anchor,
        stats_text=stats_text,
    )


def plot_connected_dotplot(df: pd.DataFrame) -> None:
    valid = df[['avg_stim', 'nostim']].apply(pd.to_numeric, errors='coerce').dropna()
    t_stat, p_value = ttest_rel(valid['avg_stim'], valid['nostim'])

    fig, ax = plt.subplots(figsize=(6.0, 8.2))
    for stim_value, nostim_value in zip(valid['avg_stim'], valid['nostim']):
        ax.plot(['nostim', 'stim'], [nostim_value, stim_value], color='black', linewidth=1.4, alpha=0.55, zorder=1)

    ax.scatter(['nostim'] * len(valid), valid['nostim'], color='blue', s=120, zorder=3)
    ax.scatter(['stim'] * len(valid), valid['avg_stim'], color='red', s=120, zorder=3)
    ax.set_ylabel('Dprime value', fontsize=18, fontweight='bold', labelpad=8)
    ax.set_xlabel('')
    fig.suptitle('AMME & BLAES stim vs. nostim dprime at one-day delay', fontsize=18, y=0.985)
    fig.text(0.5, 0.845, f't = {t_stat:.3f}\np = {p_value:.3f}', ha='center', va='center', fontsize=14)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['nostim', 'stim'], fontsize=18, fontweight='bold')
    ax.tick_params(axis='y', labelsize=14)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    fig.subplots_adjust(top=0.73, bottom=0.08)
    fig.savefig(OUTPUT_DIR / 'AMME_BLAES_Stim_NoStim_connected_dotplot.png', dpi=300, bbox_inches='tight', pad_inches=0.35)
    plt.close(fig)


def plot_connected_dotplot_responder(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 8.2))
    x_nostim = 0
    x_stim = 1.5
    for _, row in df.iterrows():
        color = RESPONDER_PALETTE[row['Responder status']]
        ax.plot([x_nostim, x_stim], [row['nostim'], row['avg_stim']], color=color, linewidth=1.4, alpha=0.7, zorder=1)
        ax.scatter(x_nostim, row['nostim'], color=color, s=120, zorder=3)
        ax.scatter(x_stim, row['avg_stim'], color=color, s=120, zorder=3)

    ax.set_ylabel('Dprime value', fontsize=18, fontweight='bold', labelpad=8)
    ax.set_xlabel('')
    fig.suptitle('AMME & BLAES stim vs. nostim dprime at one-day delay', fontsize=18, y=0.985)
    ax.set_xticks([x_nostim, x_stim])
    ax.set_xticklabels(['nostim', 'stim'], fontsize=18, fontweight='bold')
    ax.set_xlim(-0.35, x_stim + 0.35)
    ax.tick_params(axis='y', labelsize=14)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label=group, markerfacecolor=RESPONDER_PALETTE[group], markersize=11)
        for group in RESPONDER_ORDER
    ]
    ax.legend(
        handles=legend_elements,
        title='Responder status',
        frameon=False,
        fontsize=13,
        title_fontsize=16,
        loc='upper left',
        bbox_to_anchor=(1.02, 1.0),
    )
    fig.subplots_adjust(top=0.83, right=0.77, bottom=0.08)
    fig.savefig(
        OUTPUT_DIR / 'AMME_BLAES_Stim_NoStim_connected_dotplot_colored_by_responder.png',
        dpi=300,
        bbox_inches='tight',
        pad_inches=0.35,
    )
    plt.close(fig)


def plot_dprime_diff_swarm(df: pd.DataFrame) -> None:
    plot_df = df[df['avg_stim_dprime_diff'].notna()].copy()
    plot_df['delay_group'] = 'One-day delay'
    make_swarm(
        plot_df,
        hue=None,
        hue_order=None,
        palette=None,
        title=f'AMME & BLAES dprime difference at one-day delay, N={len(plot_df)}',
        out_name='AMMEBLAES_dprime_diff_swarmplot.png',
    )


def plot_responder_swarm(df: pd.DataFrame) -> None:
    plot_df = df[df['avg_stim_dprime_diff'].notna()].copy()
    plot_df['delay_group'] = 'One-day delay'
    make_swarm(
        plot_df,
        hue='Responder status',
        hue_order=RESPONDER_ORDER,
        palette=RESPONDER_PALETTE,
        title=f'AMME & BLAES dprime difference at one-day delay, N={len(plot_df)}',
        out_name='AMMEBLAES_responder_status_swarmplot.png',
        legend_title='Responder status',
        legend_anchor=(1.12, 0.75),
    )


def plot_sex_swarm(df: pd.DataFrame) -> None:
    plot_df = df.copy()
    plot_df['sex'] = plot_df['sex'].astype(str).str.strip().str.lower()
    plot_df = plot_df[plot_df['sex'].isin(['male', 'female']) & plot_df['avg_stim_dprime_diff'].notna()].copy()
    plot_df['delay_group'] = 'One-day delay'

    male = plot_df.loc[plot_df['sex'] == 'male', 'avg_stim_dprime_diff']
    female = plot_df.loc[plot_df['sex'] == 'female', 'avg_stim_dprime_diff']
    t_stat, p_val = ttest_ind(male, female, equal_var=False)
    legend_labels = {'male': f'male (N = {len(male)})', 'female': f'female (N = {len(female)})'}
    palette = {'male': '#191273', 'female': '#AA2CAC'}
    make_swarm(
        plot_df,
        hue='sex',
        hue_order=['male', 'female'],
        palette=palette,
        title=f'AMME & BLAES dprime difference by sex at one-day delay, N={len(plot_df)}',
        out_name='AMMEBLAES_sex_swarmplot_stats.png',
        legend_title='Sex',
        legend_labels=legend_labels,
        stats_text=f't = {t_stat:.3f}, p = {p_val:.3f}',
        legend_anchor=(1.12, 0.75),
    )


def plot_age_swarm(df: pd.DataFrame) -> None:
    plot_df = df.copy()
    plot_df['age'] = pd.to_numeric(plot_df['age'], errors='coerce')
    age_clean = plot_df['age'].dropna()
    age_mean = age_clean.mean()
    age_std = age_clean.std()
    lower_bound = age_mean - 0.5 * age_std
    upper_bound = age_mean + 0.5 * age_std

    def assign_age_group(age):
        if pd.isna(age):
            return np.nan
        if age <= lower_bound:
            return 'Youngest'
        if age <= age_mean:
            return 'Younger-middle'
        if age <= upper_bound:
            return 'Older-middle'
        return 'Oldest'

    plot_df['age_group'] = plot_df['age'].apply(assign_age_group)
    plot_df = plot_df.dropna(subset=['age_group', 'avg_stim_dprime_diff']).copy()
    plot_df['delay_group'] = 'One-day delay'
    hue_order = ['Youngest', 'Younger-middle', 'Older-middle', 'Oldest']
    palette = {
        'Youngest': '#C6DBEF',
        'Younger-middle': '#6BAED6',
        'Older-middle': '#2171B5',
        'Oldest': '#08306B',
    }
    age_ranges = plot_df.groupby('age_group')['age'].agg(['min', 'max'])
    legend_labels = {}
    for group in hue_order:
        if group in age_ranges.index:
            legend_labels[group] = f"{group} [{int(np.floor(age_ranges.loc[group, 'min']))}-{int(np.ceil(age_ranges.loc[group, 'max']))}]"
        else:
            legend_labels[group] = group

    make_swarm(
        plot_df,
        hue='age_group',
        hue_order=hue_order,
        palette=palette,
        title=f'AMME & BLAES dprime difference by age group at one-day delay, N={len(plot_df)}',
        out_name='AMMEBLAES_age_group_swarmplot.png',
        legend_title='Age group',
        legend_labels=legend_labels,
        legend_anchor=(1.28, 0.75),
    )


def plot_hemisphere_swarm(df: pd.DataFrame) -> None:
    plot_df = df.copy()
    plot_df['stim_hemisphere'] = plot_df['stim_hemisphere'].astype(str).str.strip().str.upper()
    plot_df = plot_df[plot_df['stim_hemisphere'].isin(['L', 'R']) & plot_df['avg_stim_dprime_diff'].notna()].copy()
    plot_df['delay_group'] = 'One-day delay'
    left = plot_df.loc[plot_df['stim_hemisphere'] == 'L', 'avg_stim_dprime_diff']
    right = plot_df.loc[plot_df['stim_hemisphere'] == 'R', 'avg_stim_dprime_diff']
    t_stat, p_val = ttest_ind(left, right, equal_var=False)
    legend_labels = {'L': f'Left (N = {len(left)})', 'R': f'Right (N = {len(right)})'}
    palette = {'L': '#2A788B', 'R': '#322CAC'}
    make_swarm(
        plot_df,
        hue='stim_hemisphere',
        hue_order=['L', 'R'],
        palette=palette,
        title=f'AMME & BLAES stim hemisphere at one-day delay, N={len(plot_df)}',
        out_name='AMMEBLAES_hemisphere_swarmplot_stats.png',
        legend_title='Stim Hemisphere',
        legend_labels=legend_labels,
        stats_text=f't = {t_stat:.3f}, p = {p_val:.3f}',
        legend_anchor=(1.28, 0.75),
    )


def plot_stim_intensity_swarm(df: pd.DataFrame) -> None:
    plot_df = df.copy()
    plot_df['stim_DB'] = pd.to_numeric(plot_df['stim_DB'], errors='coerce')
    plot_df = plot_df[plot_df['stim_DB'].isin([0.5, 1.0]) & plot_df['avg_stim_dprime_diff'].notna()].copy()
    plot_df['delay_group'] = 'One-day delay'
    low = plot_df.loc[plot_df['stim_DB'] == 0.5, 'avg_stim_dprime_diff']
    high = plot_df.loc[plot_df['stim_DB'] == 1.0, 'avg_stim_dprime_diff']
    t_stat, p_val = ttest_ind(low, high, equal_var=False)
    legend_labels = {0.5: f'0.5 mA (N = {len(low)})', 1.0: f'1.0 mA (N = {len(high)})'}
    palette = {0.5: '#D81B60', 1.0: '#6A1B9A'}
    make_swarm(
        plot_df,
        hue='stim_DB',
        hue_order=[0.5, 1.0],
        palette=palette,
        title=f'AMME & BLAES dprime difference by stim intensity at one-day delay, N={len(plot_df)}',
        out_name='AMMEBLAES_stim_intensity_swarmplot.png',
        legend_title='Stim Intensity',
        legend_labels=legend_labels,
        stats_text=f"Welch's t = {t_stat:.3f}, p = {p_val:.3f}",
        legend_anchor=(1.28, 0.75),
    )


def plot_stim_outside_swarm(df: pd.DataFrame) -> None:
    plot_df = df.copy()
    plot_df['stim_outside_amyg'] = plot_df['stim_outside_amyg'].astype(str).str.strip().str.upper()
    plot_df = plot_df[plot_df['stim_outside_amyg'].isin(['Y', 'N']) & plot_df['avg_stim_dprime_diff'].notna()].copy()
    plot_df['delay_group'] = 'One-day delay'
    outside_yes = plot_df.loc[plot_df['stim_outside_amyg'] == 'Y', 'avg_stim_dprime_diff']
    outside_no = plot_df.loc[plot_df['stim_outside_amyg'] == 'N', 'avg_stim_dprime_diff']
    t_stat, p_val = ttest_ind(outside_yes, outside_no, equal_var=False)
    legend_labels = {'Y': f'Y (N = {len(outside_yes)})', 'N': f'N (N = {len(outside_no)})'}
    palette = {'Y': '#000000', 'N': '#808080'}
    make_swarm(
        plot_df,
        hue='stim_outside_amyg',
        hue_order=['Y', 'N'],
        palette=palette,
        title=f'AMME & BLAES dprime difference by stim outside amygdala at one-day delay, N={len(plot_df)}',
        out_name='AMMEBLAES_stim_outside_amyg_swarmplot.png',
        legend_title='Stim outside amygdala',
        legend_labels=legend_labels,
        stats_text=f"Welch's t = {t_stat:.3f}, p = {p_val:.3f}",
        legend_anchor=(1.30, 0.75),
    )


def plot_ied_frequency_swarm(df: pd.DataFrame) -> None:
    plot_df = df.copy()
    plot_df['IED_freq'] = pd.to_numeric(plot_df['IED_freq'], errors='coerce')
    plot_df = plot_df[plot_df['IED_freq'].isin([0, 1, 2, 3, 4]) & plot_df['avg_stim_dprime_diff'].notna()].copy()
    plot_df['delay_group'] = 'One-day delay'
    freq_map = {0: 'none', 1: 'rare', 2: 'occasional', 3: 'frequent', 4: 'constant'}
    hue_order = ['none', 'rare', 'occasional', 'frequent', 'constant']
    plot_df['IED_freq_label'] = plot_df['IED_freq'].map(freq_map)
    palette = dict(zip(hue_order, sns.color_palette('mako', 5)))
    make_swarm(
        plot_df,
        hue='IED_freq_label',
        hue_order=hue_order,
        palette=palette,
        title=f'AMME & BLAES dprime difference by IED frequency at one-day delay, N={len(plot_df)}',
        out_name='AMMEBLAES_IED_frequency_swarmplot.png',
        legend_title='IED frequency',
        legend_anchor=(1.24, 0.75),
    )


def main() -> None:
    sns.set_style('white')
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_behavior_csv()
    df = add_responder_groups(df)

    plot_connected_dotplot(df)
    plot_dprime_diff_swarm(df)
    plot_responder_swarm(df)
    plot_connected_dotplot_responder(df)
    plot_sex_swarm(df)
    plot_age_swarm(df)
    plot_hemisphere_swarm(df)
    plot_stim_intensity_swarm(df)
    plot_stim_outside_swarm(df)
    plot_ied_frequency_swarm(df)

    print(f'Behavior rows: {len(df)}')
    print(f'Output directory: {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
