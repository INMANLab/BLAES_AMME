#!/usr/bin/env python
"""Build balanced-trials behavioral plots from the AMME+BLAES subject table."""

from __future__ import annotations

from io import StringIO
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr, ttest_ind, ttest_rel


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / 'outputs' / 'behavioral_figures_balanced_trials'
COUNTS_CSV = SCRIPT_DIR / 'outputs' / 'remembered_forgotten_by_subject.csv'
SUBSET_CSV = OUTPUT_DIR / 'balanced_behavioral_subjects.csv'
SUMMARY_TXT = OUTPUT_DIR / 'balanced_behavioral_subjects_summary.txt'
RESPONDER_CSV = OUTPUT_DIR / 'balanced_responder_status.csv'
MANUAL_COUNTS_CSV = OUTPUT_DIR / 'manual_subject_trial_counts_from_logs.csv'

MIN_TRIALS_PER_CONDITION = 10
BALANCED_ALIAS_MAP = {'BJH032': 'BJH033'}
VALID_SEX = ['male', 'female']

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
SWARM_TITLE_Y = 0.985
SWARM_STATS_Y = 0.855
SWARM_TOP = 0.74

MANUAL_COUNT_LOGS = {
    'amyg001': Path('/Users/martinahollearn/Library/CloudStorage/Box-Box/InmanLab/AMME_Data_Emory/AMME_Data/amyg001/amyg001_day2.log'),
    'amyg002': Path('/Users/martinahollearn/Library/CloudStorage/Box-Box/InmanLab/AMME_Data_Emory/AMME_Data/amyg002/amyg002_day2.log'),
    'amyg015': Path('/Users/martinahollearn/Library/CloudStorage/Box-Box/InmanLab/AMME_Data_Emory/AMME_Data/amyg015/amyg015_day2.log'),
    'amyg016': Path('/Users/martinahollearn/Library/CloudStorage/Box-Box/InmanLab/AMME_Data_Emory/AMME_Data/amyg016/amyg016_day2.log'),
    'amyg042': Path('/Users/martinahollearn/Library/CloudStorage/Box-Box/InmanLab/AMME_Data_Emory/AMME_Data/amyg042/amyg042_Ramyg_day2.log'),
    'amyg047': Path('/Users/martinahollearn/Library/CloudStorage/Box-Box/InmanLab/AMME_Data_Emory/AMME_Data/amyg047/amyg047_Lamyg_day2.log'),
    'amyg053': Path('/Users/martinahollearn/Library/CloudStorage/Box-Box/InmanLab/AMME_Data_Emory/AMME_Data/amyg053/amyg053_Ramyg_day2.log'),
    'amyg060': Path('/Users/martinahollearn/Library/CloudStorage/Box-Box/InmanLab/AMME_Data_Emory/AMME_Data/amyg060/amyg060_Lamyg_day2.log'),
}


def load_behavior_csv() -> pd.DataFrame:
    candidates = [
        SCRIPT_DIR.parent / 'AMMEBLAES_includedpts_firstsession_behavioral.csv',
        SCRIPT_DIR / 'behavioral figures' / 'AMMEBLAES_includedpts_firstsession_behavioral.csv',
    ]
    for path in candidates:
        if path.exists():
            df = pd.read_csv(path).copy()
            df['Patient'] = df['Patient'].astype(str).str.strip()
            return df
    raise FileNotFoundError(
        'Could not locate AMMEBLAES_includedpts_firstsession_behavioral.csv '
        f'in: {[str(path) for path in candidates]}'
    )


def load_trial_counts() -> pd.DataFrame:
    if not COUNTS_CSV.exists():
        raise FileNotFoundError(f'Missing remembered/forgotten subject counts: {COUNTS_CSV}')

    counts = pd.read_csv(COUNTS_CSV).copy()
    required = {'Subject', 'Remembered', 'Forgotten'}
    missing = required.difference(counts.columns)
    if missing:
        raise ValueError(f'Missing columns in {COUNTS_CSV}: {sorted(missing)}')

    counts['Subject'] = counts['Subject'].astype(str).str.strip()
    counts['Patient'] = counts['Subject'].replace(BALANCED_ALIAS_MAP)
    counts['Remembered'] = pd.to_numeric(counts['Remembered'], errors='coerce')
    counts['Forgotten'] = pd.to_numeric(counts['Forgotten'], errors='coerce')
    counts['count_source'] = 'existing_subject_count_csv'
    counts['ValidOldTrials'] = counts['Remembered'] + counts['Forgotten']
    counts['DroppedMissingInfo'] = np.nan
    counts['LogPath'] = pd.NA
    counts['balanced_keep'] = (
        (counts['Remembered'] >= MIN_TRIALS_PER_CONDITION) &
        (counts['Forgotten'] >= MIN_TRIALS_PER_CONDITION)
    )
    manual_counts = build_manual_log_counts()
    missing_manual = manual_counts[~manual_counts['Patient'].isin(counts['Patient'])].copy()
    return pd.concat([counts, missing_manual], ignore_index=True, sort=False)


def parse_day2_log_counts(log_path: Path) -> dict:
    if not log_path.exists():
        raise FileNotFoundError(f'Missing manual day-2 log file: {log_path}')

    lines = log_path.read_text(errors='ignore').splitlines()
    header_idx = next((i for i, line in enumerate(lines) if line.startswith('TRIAL\t')), None)
    if header_idx is None:
        raise ValueError(f'Could not find tabular day-2 header in {log_path}')

    df = pd.read_csv(StringIO('\n'.join(lines[header_idx:])), sep='\t')
    df.columns = [str(column).strip() for column in df.columns]
    if 'CONDITION' not in df.columns or 'YES/NO' not in df.columns:
        raise ValueError(f'Required columns missing in {log_path}')

    df['CONDITION'] = df['CONDITION'].astype(str).str.strip().str.lower()
    df['YES/NO'] = df['YES/NO'].astype(str).str.strip().str.lower()

    old_trials = df[df['CONDITION'] != 'new'].copy()
    valid_trials = old_trials[old_trials['YES/NO'].isin(['yes', 'no'])].copy()

    remembered = int((valid_trials['YES/NO'] == 'yes').sum())
    forgotten = int((valid_trials['YES/NO'] == 'no').sum())
    dropped_missing = int(len(old_trials) - len(valid_trials))

    return {
        'Remembered': remembered,
        'Forgotten': forgotten,
        'ValidOldTrials': remembered + forgotten,
        'DroppedMissingInfo': dropped_missing,
        'LogPath': str(log_path),
    }


def build_manual_log_counts() -> pd.DataFrame:
    rows = []
    for patient, log_path in MANUAL_COUNT_LOGS.items():
        parsed = parse_day2_log_counts(log_path)
        rows.append({
            'Subject': patient,
            'Patient': patient,
            'count_source': 'manual_day2_log',
            **parsed,
        })

    manual_df = pd.DataFrame(rows).sort_values('Patient').reset_index(drop=True)
    manual_df['balanced_keep'] = (
        (manual_df['Remembered'] >= MIN_TRIALS_PER_CONDITION) &
        (manual_df['Forgotten'] >= MIN_TRIALS_PER_CONDITION)
    )
    manual_df.to_csv(MANUAL_COUNTS_CSV, index=False)
    return manual_df


def apply_balanced_filter(behavior_df: pd.DataFrame, counts_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    count_cols = [
        'Patient', 'Subject', 'Remembered', 'Forgotten', 'ValidOldTrials',
        'DroppedMissingInfo', 'LogPath', 'count_source', 'balanced_keep',
    ]
    merged = behavior_df.merge(counts_df[count_cols], on='Patient', how='left')

    merged['count_status'] = np.where(
        merged['balanced_keep'].eq(True),
        'kept',
        np.where(
            merged['balanced_keep'].eq(False),
            'excluded_low_trials',
            'excluded_missing_counts',
        ),
    )
    balanced = merged[merged['count_status'] == 'kept'].copy()
    balanced = balanced.sort_values('Patient').reset_index(drop=True)
    return merged, balanced


def save_subset_artifacts(full_df: pd.DataFrame, balanced_df: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    export_cols = [
        'Patient', 'Study', 'Memory_delay', 'nostim', 'avg_stim', 'avg_stim_dprime_diff',
        'stim_hemisphere', 'stim_DB', 'IED_freq', 'sex',
        'Remembered', 'Forgotten', 'ValidOldTrials', 'DroppedMissingInfo', 'count_source', 'count_status',
    ]
    balanced_df[export_cols].to_csv(SUBSET_CSV, index=False)

    missing_counts = full_df[full_df['count_status'] == 'excluded_missing_counts']['Patient'].tolist()
    low_trial_df = full_df[full_df['count_status'] == 'excluded_low_trials'][
        ['Patient', 'Remembered', 'Forgotten', 'count_source']
    ].drop_duplicates().sort_values('Patient')
    manual_count_df = full_df[full_df['count_source'] == 'manual_day2_log'][
        ['Patient', 'Remembered', 'Forgotten', 'ValidOldTrials', 'DroppedMissingInfo', 'LogPath']
    ].drop_duplicates().sort_values('Patient')

    lines = [
        'Balanced Behavioral Subjects Summary',
        '=' * 80,
        f'Total behavioral rows: {len(full_df)}',
        f'Included balanced subjects: {len(balanced_df)}',
        f'Criterion: Remembered >= {MIN_TRIALS_PER_CONDITION} and Forgotten >= {MIN_TRIALS_PER_CONDITION}',
        '',
        f'Subjects supplemented from original day-2 log files: {len(manual_count_df)}',
    ]
    if manual_count_df.empty:
        lines.append('  None')
    else:
        for row in manual_count_df.itertuples(index=False):
            log_name = Path(row.LogPath).name
            lines.append(
                f'  {row.Patient}: remembered={int(row.Remembered)}, forgotten={int(row.Forgotten)}, '
                f'valid_old_trials={int(row.ValidOldTrials)}, dropped_missing_info={int(row.DroppedMissingInfo)}, '
                f'log={log_name}'
            )

    lines.extend([
        '',
        f'Excluded for low trial counts: {len(low_trial_df)}',
    ])
    if low_trial_df.empty:
        lines.append('  None')
    else:
        for row in low_trial_df.itertuples(index=False):
            lines.append(
                f'  {row.Patient}: remembered={int(row.Remembered)}, forgotten={int(row.Forgotten)}, '
                f'source={row.count_source}'
            )

    lines.extend([
        '',
        f'Excluded because no subject count row was available: {len(missing_counts)}',
    ])
    if not missing_counts:
        lines.append('  None')
    else:
        for patient in missing_counts:
            lines.append(f'  {patient}')

    SUMMARY_TXT.write_text('\n'.join(lines) + '\n', encoding='utf-8')


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
    responder_df = out[['Patient', 'avg_stim_dprime_diff', 'Responder status']].sort_values('Patient')
    responder_df.to_csv(RESPONDER_CSV, index=False)
    return out


def finalize_swarm_plot(grid, out_path: Path, legend_title: str, legend_labels: dict | None = None,
                        legend_anchor: tuple[float, float] = (1.30, 0.75), stats_text: str | None = None) -> None:
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
        grid.fig.text(0.5, SWARM_STATS_Y, stats_text, ha='center', va='center', fontsize=16)

    grid.fig.subplots_adjust(left=0.18, right=0.78, top=SWARM_TOP)

    legend = grid._legend
    if legend is not None:
        legend.set_bbox_to_anchor(legend_anchor)
        legend.set_frame_on(False)
        if legend_labels:
            for text, value in zip(legend.texts, legend_labels):
                text.set_text(legend_labels[value])
                text.set_fontsize(14)
        else:
            for text in legend.texts:
                text.set_fontsize(14)
        legend.set_title(legend_title, prop={'size': 18})

    grid.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0.35)
    plt.close(grid.fig)


def plot_dprime_diff_swarm(df: pd.DataFrame) -> None:
    plot_df = df[df['avg_stim_dprime_diff'].notna()].copy()
    plot_df['delay_group'] = 'One-day delay'

    grid = sns.catplot(
        x='delay_group',
        y='avg_stim_dprime_diff',
        data=plot_df,
        kind='swarm',
        color='black',
        s=180,
        height=10,
        aspect=0.55,
    )
    grid.ax.set_ylabel('Dprime difference (stimulated - not stimulated)', fontsize=20)
    grid.ax.set_title('')
    grid.fig.suptitle(
        f'AMME & BLAES dprime difference at one-day delay\nBalanced-trials subjects (N={len(plot_df)})',
        fontsize=20,
        y=SWARM_TITLE_Y,
    )
    finalize_swarm_plot(
        grid,
        OUTPUT_DIR / 'AMMEBLAES_dprime_diff_swarmplot_balanced_trials.png',
        legend_title='',
    )


def plot_connected_dotplot(df: pd.DataFrame) -> None:
    stim = pd.to_numeric(df['avg_stim'], errors='coerce')
    nostim = pd.to_numeric(df['nostim'], errors='coerce')
    valid = pd.DataFrame({'stim': stim, 'nostim': nostim}).dropna()

    t_stat, p_value = ttest_rel(valid['stim'], valid['nostim'])
    fig, ax = plt.subplots(figsize=(5.5, 10))

    for stim_value, nostim_value in zip(valid['stim'], valid['nostim']):
        ax.plot(['nostim', 'stim'], [nostim_value, stim_value], color='black', linewidth=1.5, alpha=0.6, zorder=1)

    ax.scatter(['nostim'] * len(valid), valid['nostim'], color='blue', s=150, zorder=3)
    ax.scatter(['stim'] * len(valid), valid['stim'], color='red', s=150, zorder=3)

    ax.set_ylabel('Dprime value', fontsize=20, fontweight='bold', labelpad=8)
    ax.set_xlabel('')
    fig.suptitle(
        f'AMME & BLAES stim vs. nostim dprime at one-day delay\nBalanced-trials subjects (N={len(valid)})',
        fontsize=18,
        fontweight='normal',
        y=0.985,
    )
    fig.text(0.5, 0.845, f't = {t_stat:.3f}\np = {p_value:.3f}', ha='center', va='center', fontsize=14)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['nostim', 'stim'], fontsize=20, fontweight='bold')
    ax.tick_params(axis='y', labelsize=15)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    fig.subplots_adjust(top=0.73)
    fig.savefig(
        OUTPUT_DIR / 'AMME_BLAES_Stim_NoStim_connected_dotplot_balanced_trials.png',
        dpi=300,
        bbox_inches='tight',
        pad_inches=0.35,
    )
    plt.close(fig)


def plot_responder_swarm(df: pd.DataFrame) -> None:
    grid = sns.catplot(
        x='Memory_delay',
        y='avg_stim_dprime_diff',
        hue='Responder status',
        hue_order=RESPONDER_ORDER,
        data=df,
        kind='swarm',
        s=180,
        palette=RESPONDER_PALETTE,
        height=10,
        aspect=0.55,
    )
    grid.ax.set_ylabel('Dprime difference (stimulated - not stimulated)', fontsize=20)
    grid.ax.set_title('')
    grid.fig.suptitle(
        f'AMME & BLAES dprime difference at one-day delay\nBalanced-trials subjects (N={len(df)})',
        fontsize=20,
        y=SWARM_TITLE_Y,
    )
    finalize_swarm_plot(
        grid,
        OUTPUT_DIR / 'AMMEBLAES_responder_status_swarmplot_balanced_trials.png',
        legend_title='Responder status',
    )


def plot_sex_swarm(df: pd.DataFrame) -> None:
    plot_df = df.copy()
    plot_df['sex'] = plot_df['sex'].astype(str).str.strip().str.lower()
    plot_df = plot_df[plot_df['sex'].isin(VALID_SEX)].copy()
    plot_df['delay_group'] = 'One-day delay'

    male = plot_df.loc[plot_df['sex'] == 'male', 'avg_stim_dprime_diff']
    female = plot_df.loc[plot_df['sex'] == 'female', 'avg_stim_dprime_diff']
    t_stat, p_val = ttest_ind(male, female, equal_var=False)

    legend_labels = {
        'male': f'male (N = {len(male)})',
        'female': f'female (N = {len(female)})',
    }
    palette = {'male': '#191273', 'female': '#AA2CAC'}

    grid = sns.catplot(
        x='delay_group',
        y='avg_stim_dprime_diff',
        hue='sex',
        hue_order=VALID_SEX,
        data=plot_df,
        kind='swarm',
        s=180,
        palette=palette,
        height=10,
        aspect=0.55,
    )
    grid.ax.set_ylabel('Dprime difference (stimulated - not stimulated)', fontsize=20)
    grid.ax.set_title('')
    grid.fig.suptitle(
        f'AMME & BLAES dprime difference by sex at one-day delay\nBalanced-trials subjects (N={len(plot_df)})',
        fontsize=20,
        y=SWARM_TITLE_Y,
    )
    finalize_swarm_plot(
        grid,
        OUTPUT_DIR / 'AMMEBLAES_sex_swarmplot_stats_balanced_trials.png',
        legend_title='Sex',
        legend_labels=legend_labels,
        stats_text=f't = {t_stat:.3f}, p = {p_val:.3f}',
    )


def plot_age_swarm(df: pd.DataFrame) -> None:
    plot_df = df.copy()
    plot_df['age'] = pd.to_numeric(plot_df['age'], errors='coerce')
    plot_df = plot_df[plot_df['avg_stim_dprime_diff'].notna()].copy()

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
    plot_df = plot_df.dropna(subset=['age_group', 'age']).copy()
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
            min_age = int(np.floor(age_ranges.loc[group, 'min']))
            max_age = int(np.ceil(age_ranges.loc[group, 'max']))
            legend_labels[group] = f'{group} [{min_age}-{max_age}]'
        else:
            legend_labels[group] = group

    r_value, p_value = pearsonr(plot_df['age'], plot_df['avg_stim_dprime_diff'])

    grid = sns.catplot(
        x='delay_group',
        y='avg_stim_dprime_diff',
        hue='age_group',
        hue_order=hue_order,
        data=plot_df,
        kind='swarm',
        s=180,
        palette=palette,
        height=10,
        aspect=0.55,
    )
    grid.ax.set_ylabel('Dprime difference (stimulated - not stimulated)', fontsize=20)
    grid.ax.set_title('')
    grid.fig.suptitle(
        f'AMME & BLAES dprime difference by age group at one-day delay\nBalanced-trials subjects (N={len(plot_df)})',
        fontsize=20,
        y=SWARM_TITLE_Y,
    )
    finalize_swarm_plot(
        grid,
        OUTPUT_DIR / 'AMMEBLAES_age_group_swarmplot_balanced_trials.png',
        legend_title='Age group',
        legend_labels=legend_labels,
        legend_anchor=(1.30, 0.75),
        stats_text=f'r = {r_value:.3f}, p = {p_value:.3f}',
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

    grid = sns.catplot(
        x='delay_group',
        y='avg_stim_dprime_diff',
        hue='stim_hemisphere',
        hue_order=['L', 'R'],
        data=plot_df,
        kind='swarm',
        s=180,
        palette=palette,
        height=10,
        aspect=0.55,
    )
    grid.ax.set_ylabel('Dprime difference (stimulated - not stimulated)', fontsize=20)
    grid.ax.set_title('')
    grid.fig.suptitle(
        f'AMME & BLAES stim hemisphere at one-day delay\nBalanced-trials subjects (N={len(plot_df)})',
        fontsize=20,
        y=SWARM_TITLE_Y,
    )
    finalize_swarm_plot(
        grid,
        OUTPUT_DIR / 'AMMEBLAES_hemisphere_swarmplot_stats_balanced_trials.png',
        legend_title='Stim Hemisphere',
        legend_labels=legend_labels,
        stats_text=f't = {t_stat:.3f}, p = {p_val:.3f}',
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

    grid = sns.catplot(
        x='delay_group',
        y='avg_stim_dprime_diff',
        hue='stim_DB',
        hue_order=[0.5, 1.0],
        data=plot_df,
        kind='swarm',
        s=180,
        palette=palette,
        height=10,
        aspect=0.55,
    )
    grid.ax.set_ylabel('Dprime difference (stimulated - not stimulated)', fontsize=20)
    grid.ax.set_title('')
    grid.fig.suptitle(
        f'AMME & BLAES dprime difference by stim intensity at one-day delay\nBalanced-trials subjects (N={len(plot_df)})',
        fontsize=20,
        y=SWARM_TITLE_Y,
    )
    finalize_swarm_plot(
        grid,
        OUTPUT_DIR / 'AMMEBLAES_stim_intensity_swarmplot_balanced_trials.png',
        legend_title='Stim Intensity',
        legend_labels=legend_labels,
        stats_text=f"Welch's t = {t_stat:.3f}, p = {p_val:.3f}",
    )


def plot_ied_frequency_swarm(df: pd.DataFrame) -> None:
    plot_df = df.copy()
    plot_df['IED_freq'] = pd.to_numeric(plot_df['IED_freq'], errors='coerce')
    plot_df = plot_df[plot_df['IED_freq'].isin([0, 1, 2, 3, 4]) & plot_df['avg_stim_dprime_diff'].notna()].copy()
    plot_df['delay_group'] = 'One-day delay'

    freq_map = {
        0: 'none',
        1: 'rare',
        2: 'occasional',
        3: 'frequent',
        4: 'constant',
    }
    hue_order = ['none', 'rare', 'occasional', 'frequent', 'constant']
    plot_df['IED_freq_label'] = plot_df['IED_freq'].map(freq_map)
    palette = dict(zip(hue_order, sns.color_palette('mako', 5)))

    grid = sns.catplot(
        x='delay_group',
        y='avg_stim_dprime_diff',
        hue='IED_freq_label',
        hue_order=hue_order,
        data=plot_df,
        kind='swarm',
        s=180,
        palette=palette,
        height=10,
        aspect=0.55,
    )
    grid.ax.set_ylabel('Dprime difference (stimulated - not stimulated)', fontsize=20)
    grid.ax.set_title('')
    grid.fig.suptitle(
        f'AMME & BLAES dprime difference by IED frequency at one-day delay\nBalanced-trials subjects (N={len(plot_df)})',
        fontsize=20,
        y=SWARM_TITLE_Y,
    )
    finalize_swarm_plot(
        grid,
        OUTPUT_DIR / 'AMMEBLAES_IED_frequency_swarmplot_balanced_trials.png',
        legend_title='IED frequency',
    )


def main() -> None:
    sns.set_style('white')
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    behavior_df = load_behavior_csv()
    counts_df = load_trial_counts()
    full_df, balanced_df = apply_balanced_filter(behavior_df, counts_df)
    balanced_df = add_responder_groups(balanced_df)
    save_subset_artifacts(full_df, balanced_df)

    print(f'Behavior rows: {len(full_df)}')
    print(f'Balanced-trials rows: {len(balanced_df)}')
    print(f'Output directory: {OUTPUT_DIR}')

    plot_dprime_diff_swarm(balanced_df)
    plot_connected_dotplot(balanced_df)
    plot_responder_swarm(balanced_df)
    plot_sex_swarm(balanced_df)
    plot_age_swarm(balanced_df)
    plot_hemisphere_swarm(balanced_df)
    plot_stim_intensity_swarm(balanced_df)
    plot_ied_frequency_swarm(balanced_df)

    print(f'Wrote summary: {SUMMARY_TXT}')
    print(f'Wrote subset CSV: {SUBSET_CSV}')
    print(f'Wrote responder CSV: {RESPONDER_CSV}')


if __name__ == '__main__':
    main()
