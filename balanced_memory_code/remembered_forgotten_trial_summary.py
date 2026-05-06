#!/usr/bin/env python
"""Rebuild remembered/forgotten trial summary figures from the subject count CSV."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from balanced_memory_trials import MIN_TRIALS_PER_CONDITION


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / 'outputs'
BALANCED_OUTPUT_DIR = OUTPUT_DIR / 'balanced_memory_trials'
COUNTS_CSV = OUTPUT_DIR / 'remembered_forgotten_by_subject.csv'
BY_SUBJECT_FIG = OUTPUT_DIR / 'remembered_forgotten_by_subject.png'
AVG_COUNTS_FIG = OUTPUT_DIR / 'avg_remembered_forgotten.png'
AVG_PERCENT_FIG = OUTPUT_DIR / 'avg_remembered_forgotten_percentage.png'

REMEMBERED_COLOR = '#7B2D8E'
FORGOTTEN_COLOR = '#DAA520'


def load_counts(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f'Missing subject count CSV: {csv_path}')

    df = pd.read_csv(csv_path).copy()
    required_cols = {'Subject', 'Remembered', 'Forgotten'}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f'Missing required columns in {csv_path}: {sorted(missing)}')

    if 'Total' not in df.columns:
        df['Total'] = df['Remembered'] + df['Forgotten']

    df['Remembered'] = pd.to_numeric(df['Remembered'], errors='coerce')
    df['Forgotten'] = pd.to_numeric(df['Forgotten'], errors='coerce')
    df['Total'] = pd.to_numeric(df['Total'], errors='coerce')
    df = df.dropna(subset=['Subject', 'Remembered', 'Forgotten', 'Total']).copy()

    df['remembered_pct'] = np.where(df['Total'] > 0, df['Remembered'] / df['Total'] * 100.0, np.nan)
    df['forgotten_pct'] = np.where(df['Total'] > 0, df['Forgotten'] / df['Total'] * 100.0, np.nan)
    return df


def plot_by_subject(df: pd.DataFrame, out_path: Path) -> None:
    x = np.arange(len(df))
    fig_width = max(18, len(df) * 0.42)
    fig, ax = plt.subplots(figsize=(fig_width, 8))

    ax.bar(x, df['Remembered'], color=REMEMBERED_COLOR, label='Remembered')
    ax.bar(x, df['Forgotten'], bottom=df['Remembered'], color=FORGOTTEN_COLOR, label='Forgotten')

    for idx, (_, row) in enumerate(df.iterrows()):
        total = float(row['Remembered'] + row['Forgotten'])
        ax.text(
            idx,
            total + 2.0,
            f"{int(row['Forgotten'])}",
            ha='center',
            va='bottom',
            fontsize=9,
            fontweight='bold',
        )

    ax.set_xticks(x)
    ax.set_xticklabels(df['Subject'], rotation=90, fontsize=10)
    ax.set_xlabel('Subject', fontsize=20)
    ax.set_ylabel('Number of Trials', fontsize=20)
    ax.set_title('Remembered vs Forgotten Trials by Subject', fontsize=24, fontweight='bold')
    ax.tick_params(axis='y', labelsize=15)
    ax.legend(loc='upper left', fontsize=17, frameon=True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(0, float((df['Remembered'] + df['Forgotten']).max()) + 16.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_average_counts(df: pd.DataFrame, out_path: Path) -> None:
    categories = [
        ('Remembered', REMEMBERED_COLOR),
        ('Forgotten', FORGOTTEN_COLOR),
    ]
    rng = np.random.default_rng(29)
    fig, ax = plt.subplots(figsize=(7.5, 7.5))

    for idx, (column, color) in enumerate(categories):
        values = df[column].to_numpy(dtype=float)
        mean_value = float(np.nanmean(values))
        sem_value = float(pd.Series(values).sem()) if len(values) > 1 else 0.0

        ax.bar(
            idx,
            mean_value,
            yerr=sem_value,
            color=color,
            edgecolor='black',
            linewidth=1.2,
            width=0.5,
            capsize=4,
            zorder=1,
        )
        jitter = rng.uniform(-0.08, 0.08, len(values))
        ax.scatter(
            np.full(len(values), idx) + jitter,
            values,
            color='black',
            alpha=0.35,
            s=34,
            zorder=2,
        )
        ax.text(
            idx,
            mean_value + sem_value + 2.0,
            f'{mean_value:.1f}',
            ha='center',
            va='bottom',
            fontsize=20,
            fontweight='bold',
        )

    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels([label for label, _ in categories], fontsize=20)
    ax.set_ylabel('Mean Number of Trials', fontsize=21)
    ax.set_title('Average Remembered vs Forgotten\nAcross Subjects (±SEM)', fontsize=24, fontweight='bold')
    ax.tick_params(axis='y', labelsize=15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_average_percentages(df: pd.DataFrame, out_path: Path) -> None:
    categories = [
        ('remembered_pct', 'Remembered', REMEMBERED_COLOR),
        ('forgotten_pct', 'Forgotten', FORGOTTEN_COLOR),
    ]
    rng = np.random.default_rng(31)
    fig, ax = plt.subplots(figsize=(7.5, 7.5))

    for idx, (column, label, color) in enumerate(categories):
        values = df[column].to_numpy(dtype=float)
        mean_value = float(np.nanmean(values))
        sem_value = float(pd.Series(values).sem()) if len(values) > 1 else 0.0

        ax.bar(
            idx,
            mean_value,
            yerr=sem_value,
            color=color,
            edgecolor='black',
            linewidth=1.2,
            width=0.5,
            capsize=4,
            zorder=1,
        )
        jitter = rng.uniform(-0.08, 0.08, len(values))
        ax.scatter(
            np.full(len(values), idx) + jitter,
            values,
            color='black',
            alpha=0.35,
            s=34,
            zorder=2,
        )
        ax.text(
            idx,
            mean_value + sem_value + 1.8,
            f'{mean_value:.1f}%',
            ha='center',
            va='bottom',
            fontsize=20,
            fontweight='bold',
        )

    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels([label for _, label, _ in categories], fontsize=20)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Mean % of Each Subject's Total Trials", fontsize=21)
    ax.set_title(
        'Average % Remembered vs Forgotten\nAcross Subjects (±SEM)',
        fontsize=24,
        fontweight='bold',
    )
    ax.tick_params(axis='y', labelsize=15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def get_balanced_subset(df: pd.DataFrame) -> pd.DataFrame:
    return df[
        (df['Remembered'] >= MIN_TRIALS_PER_CONDITION) &
        (df['Forgotten'] >= MIN_TRIALS_PER_CONDITION)
    ].copy()


def write_subset_csv(df: pd.DataFrame, out_path: Path) -> None:
    export_df = df.copy()
    export_df['Pct_Remembered'] = export_df['remembered_pct'].round(1)
    export_df['Pct_Forgotten'] = export_df['forgotten_pct'].round(1)
    export_df = export_df[
        ['Subject', 'Forgotten', 'Remembered', 'Total', 'Pct_Remembered', 'Pct_Forgotten']
    ]
    export_df.to_csv(out_path, index=False)


def render_summary_set(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_subset_csv(df, out_dir / 'remembered_forgotten_by_subject.csv')
    plot_by_subject(df, out_dir / 'remembered_forgotten_by_subject.png')
    plot_average_counts(df, out_dir / 'avg_remembered_forgotten.png')
    plot_average_percentages(df, out_dir / 'avg_remembered_forgotten_percentage.png')


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_counts(COUNTS_CSV)
    balanced_df = get_balanced_subset(df)

    render_summary_set(df, OUTPUT_DIR)
    render_summary_set(balanced_df, BALANCED_OUTPUT_DIR)

    print(f'Wrote {BY_SUBJECT_FIG}')
    print(f'Wrote {AVG_COUNTS_FIG}')
    print(f'Wrote {AVG_PERCENT_FIG}')
    print(f"Wrote {BALANCED_OUTPUT_DIR / 'remembered_forgotten_by_subject.png'}")
    print(f"Wrote {BALANCED_OUTPUT_DIR / 'avg_remembered_forgotten.png'}")
    print(f"Wrote {BALANCED_OUTPUT_DIR / 'avg_remembered_forgotten_percentage.png'}")


if __name__ == '__main__':
    main()
