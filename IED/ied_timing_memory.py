#!/usr/bin/env python
"""
IED Timing vs Memory — Encoding & Retrieval Phases
====================================================
Visualize whether the timing of an IED during encoding or retrieval
interfered with subsequent memory.

Encoding-phase timing windows: Before Image, During Image, After Image, During Stim
Retrieval-phase timing windows: Before Image, During Image

Outputs in outputs/ied_timing_memory/:
  Encoding:
    - encoding_ied_timing_proportion.png
    - encoding_ied_timing_by_subject.png
    - encoding_ied_timing_stim_split.png
    - encoding_ied_timing_stacked.png
  Retrieval:
    - retrieval_ied_timing_proportion.png
    - retrieval_ied_timing_by_subject.png
    - retrieval_ied_timing_stacked.png
  Stats:
    - encoding_ied_timing_stats.csv
    - retrieval_ied_timing_stats.csv
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ENCODING_CSV = os.path.join(SCRIPT_DIR,
                            'AMMEBLAES_IEDs_trial_level_dissertation_study_usethis_cleaned_with_memory.csv')
RETRIEVAL_CSV = os.path.join(SCRIPT_DIR,
                             'AMMEBLAES_IEDs_trial_level_dissertation_test_usethis_cleaned_with_memory.csv')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'outputs', 'ied_timing_memory')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Colorblind-friendly palette from seaborn
# Using "colorblind" palette: blue, orange, green, red, purple, brown...
# IED Present = orange-ish, IED Absent = blue-ish (high contrast for CVD)
# ---------------------------------------------------------------------------
CB_PALETTE = sns.color_palette('colorblind')
IED_PRESENT_COLOR = CB_PALETTE[1]   # orange
IED_ABSENT_COLOR = CB_PALETTE[0]    # blue
REMEMBERED_COLOR = CB_PALETTE[2]    # green
FORGOTTEN_COLOR = CB_PALETTE[4]     # purple
STIM_COLOR = CB_PALETTE[3]          # red
NOSTIM_COLOR = CB_PALETTE[0]        # blue

sns.set_style('ticks')
sns.set_context('talk', font_scale=0.9)

# Encoding timing columns & labels
ENC_TIMING_COLS = ['BeforeImgITI', 'DuringImg', 'AfterImgITI', 'DuringStim']
ENC_TIMING_LABELS = {
    'BeforeImgITI': 'Before Image',
    'DuringImg': 'During Image',
    'AfterImgITI': 'After Image',
    'DuringStim': 'During Stim',
}

# Retrieval timing columns & labels
RET_TIMING_COLS = ['BeforeImgITI', 'DuringImgITI']
RET_TIMING_LABELS = {
    'BeforeImgITI': 'Before Image',
    'DuringImgITI': 'During Image',
}


# ========================== DATA LOADING ==========================

def load_encoding_trials():
    """Load encoding-phase IED data, collapse to trial level."""
    df = pd.read_csv(ENCODING_CSV)
    dm = df[df['MemoryOutcome'].notna()].copy()

    trial_level = dm.groupby(['Patient', 'Trial', 'MemoryOutcome', 'StimCond']).agg({
        'DuringImg': lambda x: 'Y' if (x == 'Y').any() else 'N',
        'DuringStim': lambda x: 'Y' if (x == 'Y').any() else 'N',
        'BeforeImgITI': lambda x: 'Y' if (x == 'Y').any() else 'N',
        'AfterImgITI': lambda x: 'Y' if (x == 'Y').any() else 'N',
    }).reset_index()

    return trial_level


def load_retrieval_trials():
    """Load retrieval-phase IED data, collapse to trial level."""
    df = pd.read_csv(RETRIEVAL_CSV)
    # Keep only old items (remembered / forgotten), exclude new_item
    dm = df[df['MemoryOutcome'].isin(['remembered', 'forgotten'])].copy()

    trial_level = dm.groupby(['Patient', 'Trial', 'MemoryOutcome']).agg({
        'BeforeImgITI': lambda x: 'Y' if (x == 'Y').any() else 'N',
        'DuringImgITI': lambda x: 'Y' if (x == 'Y').any() else 'N',
    }).reset_index()

    return trial_level


# ========================== STATISTICS ==========================

def compute_stats(trial_level, timing_cols, timing_labels):
    """Compute % remembered for each timing window (present vs absent)."""
    rows = []
    for col in timing_cols:
        label = timing_labels[col]
        for present in ['Y', 'N']:
            sub = trial_level[trial_level[col] == present]
            n_total = len(sub)
            n_rem = (sub['MemoryOutcome'] == 'remembered').sum()
            n_forg = (sub['MemoryOutcome'] == 'forgotten').sum()
            pct_rem = n_rem / n_total * 100 if n_total > 0 else np.nan
            rows.append({
                'Timing': label,
                'IED_Present': 'Yes' if present == 'Y' else 'No',
                'N_Remembered': n_rem,
                'N_Forgotten': n_forg,
                'N_Total': n_total,
                'Pct_Remembered': round(pct_rem, 1),
            })

        ct = pd.crosstab(trial_level[col], trial_level['MemoryOutcome'])
        if ct.shape == (2, 2):
            chi2, p, dof, expected = chi2_contingency(ct)
            rows[-1]['Chi2'] = round(chi2, 2)
            rows[-1]['p_value'] = round(p, 4)
            rows[-2]['Chi2'] = round(chi2, 2)
            rows[-2]['p_value'] = round(p, 4)

    return pd.DataFrame(rows)


# ========================== SHARED PLOT HELPERS ==========================

def _sig_label(p):
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    return 'n.s.'


def plot_proportion_bars(trial_level, timing_cols, timing_labels, phase_label, filename):
    """Bar chart: % remembered when IED present vs absent in each timing window."""
    fig, ax = plt.subplots(figsize=(max(8, len(timing_cols) * 2.5), 7))

    x_positions = np.arange(len(timing_cols))
    bar_width = 0.35

    pct_present, pct_absent = [], []
    n_present, n_absent = [], []

    for col in timing_cols:
        for val, pct_list, n_list in [('Y', pct_present, n_present),
                                       ('N', pct_absent, n_absent)]:
            sub = trial_level[trial_level[col] == val]
            n = len(sub)
            n_rem = (sub['MemoryOutcome'] == 'remembered').sum()
            pct = n_rem / n * 100 if n > 0 else 0
            pct_list.append(pct)
            n_list.append(n)

    bars1 = ax.bar(x_positions - bar_width / 2, pct_present, bar_width,
                   label='IED Present', color=IED_PRESENT_COLOR,
                   edgecolor='black', linewidth=0.8)
    bars2 = ax.bar(x_positions + bar_width / 2, pct_absent, bar_width,
                   label='IED Absent', color=IED_ABSENT_COLOR,
                   edgecolor='black', linewidth=0.8)

    for bars, pcts, ns in [(bars1, pct_present, n_present),
                            (bars2, pct_absent, n_absent)]:
        for bar, pct, n in zip(bars, pcts, ns):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                    f'{pct:.1f}%\n(n={n})', ha='center', va='bottom',
                    fontsize=9, fontweight='bold')

    for i, col in enumerate(timing_cols):
        ct = pd.crosstab(trial_level[col], trial_level['MemoryOutcome'])
        if ct.shape == (2, 2):
            _, p, _, _ = chi2_contingency(ct)
            y_max = max(pct_present[i], pct_absent[i]) + 12
            ax.text(i, y_max, _sig_label(p), ha='center', va='bottom',
                    fontsize=12, fontweight='bold')

    ax.set_xticks(x_positions)
    ax.set_xticklabels([timing_labels[c] for c in timing_cols], fontsize=14)
    ax.set_ylabel('% Trials Remembered', fontsize=16)
    ax.set_title(f'{phase_label}: IED Timing vs Memory', fontsize=18, fontweight='bold')
    ax.set_ylim(0, 105)
    ax.legend(fontsize=13, frameon=True, loc='upper right')
    sns.despine(ax=ax)
    ax.tick_params(axis='y', labelsize=12)
    fig.tight_layout()

    out = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out}')


def plot_by_subject(trial_level, timing_cols, timing_labels, phase_label, filename):
    """Per-subject % remembered when IED present/absent, with group mean."""
    fig, axes = plt.subplots(1, len(timing_cols),
                             figsize=(max(10, len(timing_cols) * 5), 7), sharey=True)
    if len(timing_cols) == 1:
        axes = [axes]
    rng = np.random.default_rng(42)

    for ax, col in zip(axes, timing_cols):
        label = timing_labels[col]

        for bar_x, present_val, color, bar_label in [
            (0, 'Y', IED_PRESENT_COLOR, 'IED\nPresent'),
            (1, 'N', IED_ABSENT_COLOR, 'IED\nAbsent'),
        ]:
            patient_pcts = []
            for pat in sorted(trial_level['Patient'].unique()):
                sub = trial_level[(trial_level['Patient'] == pat) &
                                  (trial_level[col] == present_val)]
                if len(sub) >= 2:
                    pct = (sub['MemoryOutcome'] == 'remembered').mean() * 100
                    patient_pcts.append(pct)

            if patient_pcts:
                mean_val = np.mean(patient_pcts)
                sem_val = (np.std(patient_pcts, ddof=1) / np.sqrt(len(patient_pcts))
                           if len(patient_pcts) > 1 else 0)

                ax.bar(bar_x, mean_val, yerr=sem_val, color=color,
                       edgecolor='black', linewidth=1, width=0.5, capsize=5,
                       alpha=0.7, zorder=1)
                jitter = rng.uniform(-0.1, 0.1, len(patient_pcts))
                ax.scatter(np.full(len(patient_pcts), bar_x) + jitter, patient_pcts,
                           color='black', alpha=0.4, s=30, zorder=2)
                ax.text(bar_x, mean_val + sem_val + 3,
                        f'{mean_val:.1f}%\n(n={len(patient_pcts)} pts)',
                        ha='center', va='bottom', fontsize=10, fontweight='bold')

        ax.set_xticks([0, 1])
        ax.set_xticklabels(['IED\nPresent', 'IED\nAbsent'], fontsize=11)
        ax.set_title(label, fontsize=14, fontweight='bold')
        ax.set_ylim(0, 115)
        sns.despine(ax=ax)

    axes[0].set_ylabel('% Trials Remembered', fontsize=14)
    fig.suptitle(f'{phase_label}: IED Timing vs Memory by Subject\n(mean ± SEM, dots = patients)',
                 fontsize=16, fontweight='bold', y=1.02)
    fig.tight_layout()

    out = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out}')


def plot_stacked_memory(trial_level, timing_cols, timing_labels, phase_label, filename):
    """Stacked bar: remembered vs forgotten counts for each timing window."""
    REM_C, FORG_C = '#FFD43B', '#9C7AC9'  # match the other IED rem/forgotten figures
    fig, ax = plt.subplots(figsize=(max(6.5, len(timing_cols) * 2.1), 6.5))
    x = np.arange(len(timing_cols))
    bar_width = 0.35

    for offset, present_val, ied_label in [(-bar_width / 2, 'Y', 'IED Present'),
                                            (bar_width / 2, 'N', 'IED Absent')]:
        rem_counts, forg_counts = [], []
        for col in timing_cols:
            sub = trial_level[trial_level[col] == present_val]
            rem_counts.append((sub['MemoryOutcome'] == 'remembered').sum())
            forg_counts.append((sub['MemoryOutcome'] == 'forgotten').sum())

        ax.bar(x + offset, rem_counts, bar_width,
               color=REM_C, edgecolor='black', linewidth=0.6,
               label='Remembered' if offset < 0 else '')
        ax.bar(x + offset, forg_counts, bar_width, bottom=rem_counts,
               color=FORG_C, edgecolor='black', linewidth=0.6,
               label='Forgotten' if offset < 0 else '')

    for i in range(len(timing_cols)):
        ax.text(x[i] - bar_width / 2, -0.02, 'Present', ha='center', va='top',
                fontsize=10, fontweight='bold', rotation=90,
                transform=ax.get_xaxis_transform())
        ax.text(x[i] + bar_width / 2, -0.02, 'Absent', ha='center', va='top',
                fontsize=10, fontweight='bold', rotation=90,
                transform=ax.get_xaxis_transform())

    ax.set_xticks(x)
    ax.set_xticklabels([timing_labels[c] for c in timing_cols], fontsize=15, fontweight='bold')
    ax.tick_params(axis='x', pad=52)
    ax.set_ylabel('Number of Trials', fontsize=16, fontweight='bold')
    ax.set_title(phase_label.replace(' Phase', ''), fontsize=19, fontweight='bold')
    ax.legend(frameon=True, prop={'weight': 'bold', 'size': 14})
    sns.despine(ax=ax)
    plt.setp(ax.get_yticklabels(), fontsize=13, fontweight='bold')
    fig.tight_layout()

    out = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(out, dpi=300, bbox_inches='tight')
    # also write into the consolidated updated_IED_figures folder
    updated_dir = os.path.join(SCRIPT_DIR, '..', 'OUTPUTS', 'updated_IED_figures')
    os.makedirs(updated_dir, exist_ok=True)
    fig.savefig(os.path.join(updated_dir, filename), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out}')


def plot_stim_split(trial_level, timing_cols, timing_labels, phase_label, filename):
    """Stim vs no-stim breakdown for each timing window (encoding only)."""
    fig, axes = plt.subplots(1, len(timing_cols), figsize=(18, 7), sharey=True)
    bar_width = 0.35

    stim_labels = {'S': 'Stim', 'NS': 'No-Stim'}
    stim_colors = {'S': STIM_COLOR, 'NS': NOSTIM_COLOR}

    for ax, col in zip(axes, timing_cols):
        label = timing_labels[col]
        x = np.arange(2)

        for si, (stim_val, stim_label) in enumerate(stim_labels.items()):
            pcts, ns = [], []
            for present in ['Y', 'N']:
                sub = trial_level[(trial_level[col] == present) &
                                  (trial_level['StimCond'] == stim_val)]
                n = len(sub)
                n_rem = (sub['MemoryOutcome'] == 'remembered').sum()
                pct = n_rem / n * 100 if n > 0 else 0
                pcts.append(pct)
                ns.append(n)

            offset = -bar_width / 2 + si * bar_width
            bars = ax.bar(x + offset, pcts, bar_width,
                          label=stim_label if col == timing_cols[0] else '',
                          color=stim_colors[stim_val], edgecolor='black',
                          linewidth=0.6, alpha=0.8)
            for bar, pct, n in zip(bars, pcts, ns):
                if n > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 1,
                            f'{pct:.0f}%\n({n})', ha='center', va='bottom',
                            fontsize=7.5)

        ax.set_xticks(x)
        ax.set_xticklabels(['IED Present', 'IED Absent'], fontsize=10)
        ax.set_title(label, fontsize=13, fontweight='bold')
        ax.set_ylim(0, 110)
        sns.despine(ax=ax)

    axes[0].set_ylabel('% Trials Remembered', fontsize=14)
    fig.legend(*axes[0].get_legend_handles_labels(), loc='upper right',
               fontsize=12, frameon=True)
    fig.suptitle(f'{phase_label}: IED Timing vs Memory — Stim vs No-Stim',
                 fontsize=16, fontweight='bold', y=1.02)
    fig.tight_layout()

    out = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out}')


# ========================== MAIN ==========================

def main():
    # --- Encoding Phase ---
    print('=' * 60)
    print('ENCODING PHASE')
    print('=' * 60)
    enc = load_encoding_trials()
    print(f"Encoding trials: {len(enc)} trials, "
          f"{enc['Patient'].nunique()} patients, "
          f"{(enc['MemoryOutcome']=='remembered').sum()} remembered, "
          f"{(enc['MemoryOutcome']=='forgotten').sum()} forgotten")

    enc_stats = compute_stats(enc, ENC_TIMING_COLS, ENC_TIMING_LABELS)
    enc_stats_path = os.path.join(OUTPUT_DIR, 'encoding_ied_timing_stats.csv')
    enc_stats.to_csv(enc_stats_path, index=False)
    print(f'Saved {enc_stats_path}')
    print(enc_stats.to_string(index=False))
    print()

    plot_proportion_bars(enc, ENC_TIMING_COLS, ENC_TIMING_LABELS,
                         'Encoding Phase', 'encoding_ied_timing_proportion.png')
    plot_by_subject(enc, ENC_TIMING_COLS, ENC_TIMING_LABELS,
                    'Encoding Phase', 'encoding_ied_timing_by_subject.png')
    plot_stim_split(enc, ENC_TIMING_COLS, ENC_TIMING_LABELS,
                    'Encoding Phase', 'encoding_ied_timing_stim_split.png')
    plot_stacked_memory(enc, ENC_TIMING_COLS, ENC_TIMING_LABELS,
                        'Encoding Phase', 'encoding_ied_timing_stacked.png')

    # --- Retrieval Phase ---
    print()
    print('=' * 60)
    print('RETRIEVAL PHASE')
    print('=' * 60)
    ret = load_retrieval_trials()
    print(f"Retrieval trials: {len(ret)} trials, "
          f"{ret['Patient'].nunique()} patients, "
          f"{(ret['MemoryOutcome']=='remembered').sum()} remembered, "
          f"{(ret['MemoryOutcome']=='forgotten').sum()} forgotten")

    ret_stats = compute_stats(ret, RET_TIMING_COLS, RET_TIMING_LABELS)
    ret_stats_path = os.path.join(OUTPUT_DIR, 'retrieval_ied_timing_stats.csv')
    ret_stats.to_csv(ret_stats_path, index=False)
    print(f'Saved {ret_stats_path}')
    print(ret_stats.to_string(index=False))
    print()

    plot_proportion_bars(ret, RET_TIMING_COLS, RET_TIMING_LABELS,
                         'Retrieval Phase', 'retrieval_ied_timing_proportion.png')
    plot_by_subject(ret, RET_TIMING_COLS, RET_TIMING_LABELS,
                    'Retrieval Phase', 'retrieval_ied_timing_by_subject.png')
    plot_stacked_memory(ret, RET_TIMING_COLS, RET_TIMING_LABELS,
                        'Retrieval Phase', 'retrieval_ied_timing_stacked.png')


if __name__ == '__main__':
    main()
