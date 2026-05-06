#!/usr/bin/env python
"""
IED Gray vs White Matter and Weighted IED Rate vs Memory
— Encoding & Retrieval Phases
=========================================================
1) Does gray vs white matter IED location affect memory?
2) Does the weighted IED rate predict remembering or forgetting?

Outputs in outputs/ied_timing_memory/:
  Encoding:
    - encoding_graymatter_memory.png
    - encoding_graymatter_by_subject.png
    - encoding_weighted_rate_memory.png
    - encoding_weighted_rate_violin.png
  Retrieval:
    - retrieval_graymatter_memory.png
    - retrieval_graymatter_by_subject.png
    - retrieval_weighted_rate_memory.png
    - retrieval_weighted_rate_violin.png
  Stats:
    - encoding_graymatter_weighted_stats.csv
    - retrieval_graymatter_weighted_stats.csv
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, mannwhitneyu, spearmanr, pointbiserialr

warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ENCODING_CSV = os.path.join(SCRIPT_DIR, 'IED',
                            'AMMEBLAES_IEDs_trial_level_dissertation_study_usethis_cleaned_with_memory.csv')
RETRIEVAL_CSV = os.path.join(SCRIPT_DIR, 'IED',
                             'AMMEBLAES_IEDs_trial_level_dissertation_test_usethis_cleaned_with_memory.csv')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'outputs', 'ied_timing_memory')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Colorblind-friendly palette from seaborn
# ---------------------------------------------------------------------------
CB_PALETTE = sns.color_palette('colorblind')
REMEMBERED_COLOR = CB_PALETTE[2]   # green
FORGOTTEN_COLOR = CB_PALETTE[4]    # purple

# Tissue type colors — distinct, colorblind-safe
TISSUE_COLORS = {
    'Gray': CB_PALETTE[0],        # blue
    'White': CB_PALETTE[1],       # orange
    'Both': CB_PALETTE[3],        # red
    'Borderline': CB_PALETTE[7],  # gray
}

# Rate bin colors — sequential from seaborn colorblind
RATE_BIN_COLORS = [CB_PALETTE[0], CB_PALETTE[2], CB_PALETTE[1], CB_PALETTE[3]]

sns.set_style('ticks')
sns.set_context('talk', font_scale=0.9)


# ========================== DATA LOADING ==========================

def _classify_tissue(gm_series):
    vals = set(gm_series.dropna())
    has_g = 'G' in vals
    has_w = 'W' in vals
    has_b = 'B' in vals
    if has_g and has_w:
        return 'Both'
    elif has_g:
        return 'Gray'
    elif has_w:
        return 'White'
    elif has_b:
        return 'Borderline'
    else:
        return 'Unknown'


def load_encoding_trials():
    df = pd.read_csv(ENCODING_CSV)
    dm = df[df['MemoryOutcome'].notna()].copy()
    dm['WeightedIEDRate_num'] = pd.to_numeric(dm['WeightedIEDRate'], errors='coerce')

    trial_level = dm.groupby(['Patient', 'Trial', 'MemoryOutcome', 'StimCond']).agg({
        'GrayMatter': _classify_tissue,
        'WeightedIEDRate_num': 'mean',
        'DuringImg': lambda x: 'Y' if (x == 'Y').any() else 'N',
        'DuringStim': lambda x: 'Y' if (x == 'Y').any() else 'N',
        'BeforeImgITI': lambda x: 'Y' if (x == 'Y').any() else 'N',
        'AfterImgITI': lambda x: 'Y' if (x == 'Y').any() else 'N',
    }).reset_index()
    trial_level.rename(columns={'GrayMatter': 'TissueType'}, inplace=True)
    return trial_level


def load_retrieval_trials():
    df = pd.read_csv(RETRIEVAL_CSV)
    dm = df[df['MemoryOutcome'].isin(['remembered', 'forgotten'])].copy()
    dm['WeightedIEDRate_num'] = pd.to_numeric(dm['WeightedIEDRate'], errors='coerce')

    trial_level = dm.groupby(['Patient', 'Trial', 'MemoryOutcome']).agg({
        'GrayMatter': _classify_tissue,
        'WeightedIEDRate_num': 'mean',
        'BeforeImgITI': lambda x: 'Y' if (x == 'Y').any() else 'N',
        'DuringImgITI': lambda x: 'Y' if (x == 'Y').any() else 'N',
    }).reset_index()
    trial_level.rename(columns={'GrayMatter': 'TissueType'}, inplace=True)
    return trial_level


# ========================== GRAY MATTER PLOTS ==========================

def plot_graymatter(trial_level, phase_label, filename):
    """Bar chart: % remembered by tissue type."""
    fig, ax = plt.subplots(figsize=(9, 7))

    categories = ['Gray', 'White', 'Both', 'Borderline']
    pcts, ns, valid_cats = [], [], []

    for cat in categories:
        sub = trial_level[trial_level['TissueType'] == cat]
        n = len(sub)
        if n < 3:
            continue
        n_rem = (sub['MemoryOutcome'] == 'remembered').sum()
        pcts.append(n_rem / n * 100)
        ns.append(n)
        valid_cats.append(cat)

    x = np.arange(len(valid_cats))
    bars = ax.bar(x, pcts, color=[TISSUE_COLORS[c] for c in valid_cats],
                  edgecolor='black', linewidth=1, width=0.5)

    for bar, pct, n in zip(bars, pcts, ns):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f'{pct:.1f}%\n(n={n})', ha='center', va='bottom',
                fontsize=11, fontweight='bold')

    gw = trial_level[trial_level['TissueType'].isin(['Gray', 'White'])]
    if gw['TissueType'].nunique() == 2:
        ct = pd.crosstab(gw['TissueType'], gw['MemoryOutcome'])
        if ct.shape == (2, 2):
            chi2, p, _, _ = chi2_contingency(ct)
            sig = '***' if p < .001 else '**' if p < .01 else '*' if p < .05 else 'n.s.'
            ax.text(0.97, 0.95,
                    f'Gray vs White: \u03c7\u00b2={chi2:.2f}, p={p:.4f} ({sig})',
                    transform=ax.transAxes, ha='right', va='top', fontsize=11,
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow',
                              edgecolor='gray'))

    ax.set_xticks(x)
    ax.set_xticklabels([f'{c} Matter' if c in ('Gray', 'White') else c
                        for c in valid_cats], fontsize=13)
    ax.set_ylabel('% Trials Remembered', fontsize=15)
    ax.set_title(f'{phase_label}: IED Tissue Location vs Memory',
                 fontsize=17, fontweight='bold')
    ax.set_ylim(0, 105)
    sns.despine(ax=ax)
    ax.tick_params(axis='y', labelsize=12)
    fig.tight_layout()

    out = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out}')


def plot_graymatter_by_subject(trial_level, phase_label, filename):
    """Per-subject % remembered in Gray vs White matter IED trials."""
    fig, ax = plt.subplots(figsize=(9, 7))
    rng = np.random.default_rng(42)

    categories = ['Gray', 'White']
    bar_width = 0.5

    for xi, cat in enumerate(categories):
        patient_pcts = []
        for pat in trial_level['Patient'].unique():
            sub = trial_level[(trial_level['Patient'] == pat) &
                              (trial_level['TissueType'] == cat)]
            if len(sub) >= 2:
                pct = (sub['MemoryOutcome'] == 'remembered').mean() * 100
                patient_pcts.append(pct)

        if not patient_pcts:
            continue

        mean_val = np.mean(patient_pcts)
        sem_val = (np.std(patient_pcts, ddof=1) / np.sqrt(len(patient_pcts))
                   if len(patient_pcts) > 1 else 0)

        ax.bar(xi, mean_val, yerr=sem_val, color=TISSUE_COLORS[cat],
               edgecolor='black', linewidth=1, width=bar_width, capsize=5,
               alpha=0.8, zorder=1)
        jitter = rng.uniform(-0.1, 0.1, len(patient_pcts))
        ax.scatter(np.full(len(patient_pcts), xi) + jitter, patient_pcts,
                   color='black', alpha=0.4, s=30, zorder=2)
        ax.text(xi, mean_val + sem_val + 3,
                f'{mean_val:.1f}%\n(n={len(patient_pcts)} pts)',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels([f'{c} Matter' for c in categories], fontsize=14)
    ax.set_ylabel('% Trials Remembered', fontsize=15)
    ax.set_title(f'{phase_label}: IED Tissue Type vs Memory by Subject\n'
                 f'(mean \u00b1 SEM, dots = patients)',
                 fontsize=16, fontweight='bold')
    ax.set_ylim(0, 115)
    sns.despine(ax=ax)
    ax.tick_params(axis='y', labelsize=12)
    fig.tight_layout()

    out = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out}')


# ========================== WEIGHTED RATE PLOTS ==========================

def plot_weighted_rate_bars(trial_level, phase_label, filename):
    """% remembered binned by weighted IED rate."""
    tl = trial_level[trial_level['WeightedIEDRate_num'].notna()].copy()
    if len(tl) < 5:
        print(f'  Skipping {filename}: too few trials with weighted rate')
        return

    tl['rate_bin'] = pd.cut(tl['WeightedIEDRate_num'],
                            bins=[0, 1, 2, 3, 100],
                            labels=['1', '1-2', '2-3', '3+'],
                            include_lowest=True)

    fig, ax = plt.subplots(figsize=(9, 7))

    bins = ['1', '1-2', '2-3', '3+']
    pcts, ns = [], []

    for b in bins:
        sub = tl[tl['rate_bin'] == b]
        n = len(sub)
        n_rem = (sub['MemoryOutcome'] == 'remembered').sum()
        pct = n_rem / n * 100 if n > 0 else 0
        pcts.append(pct)
        ns.append(n)

    x = np.arange(len(bins))
    bars = ax.bar(x, pcts, color=RATE_BIN_COLORS, edgecolor='black',
                  linewidth=1, width=0.55)

    for bar, pct, n in zip(bars, pcts, ns):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f'{pct:.1f}%\n(n={n})', ha='center', va='bottom',
                fontsize=11, fontweight='bold')

    rem_bin = (tl['MemoryOutcome'] == 'remembered').astype(int)
    rho, p_sp = spearmanr(tl['WeightedIEDRate_num'], rem_bin)
    rpb, p_rpb = pointbiserialr(rem_bin, tl['WeightedIEDRate_num'])

    ax.text(0.97, 0.95,
            f'Spearman \u03c1 = {rho:.3f}, p = {p_sp:.4f}\n'
            f'Point-biserial r = {rpb:.3f}, p = {p_rpb:.4f}',
            transform=ax.transAxes, ha='right', va='top', fontsize=10,
            bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow',
                      edgecolor='gray'))

    ax.set_xticks(x)
    ax.set_xticklabels(bins, fontsize=13)
    ax.set_xlabel('Weighted IED Rate (trial average across channels)', fontsize=14)
    ax.set_ylabel('% Trials Remembered', fontsize=15)
    ax.set_title(f'{phase_label}: Weighted IED Rate vs Memory',
                 fontsize=17, fontweight='bold')
    ax.set_ylim(0, 105)
    sns.despine(ax=ax)
    ax.tick_params(axis='y', labelsize=12)
    fig.tight_layout()

    out = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out}')


def plot_weighted_rate_violin(trial_level, phase_label, filename):
    """Violin plot of weighted IED rate for remembered vs forgotten trials."""
    tl = trial_level[trial_level['WeightedIEDRate_num'].notna()].copy()
    if len(tl) < 5:
        print(f'  Skipping {filename}: too few trials with weighted rate')
        return

    fig, ax = plt.subplots(figsize=(8, 7))
    rng = np.random.default_rng(42)

    categories = ['remembered', 'forgotten']
    colors = [REMEMBERED_COLOR, FORGOTTEN_COLOR]
    labels = ['Remembered', 'Forgotten']

    # Collect data for violin plot
    all_vals = [tl[tl['MemoryOutcome'] == cat]['WeightedIEDRate_num'].values
                for cat in categories]

    # Draw violins
    parts = ax.violinplot(all_vals, positions=[0, 1], widths=0.7,
                          showmeans=True, showmedians=True, showextrema=False)

    # Color the violin bodies
    for body, color in zip(parts['bodies'], colors):
        body.set_facecolor(color)
        body.set_edgecolor('black')
        body.set_alpha(0.6)
    parts['cmeans'].set_color('black')
    parts['cmeans'].set_linewidth(1.5)
    parts['cmeans'].set_linestyle('--')
    parts['cmedians'].set_color('black')
    parts['cmedians'].set_linewidth(2)

    # Overlay jittered points
    for xi, (vals, color) in enumerate(zip(all_vals, colors)):
        jitter = rng.uniform(-0.12, 0.12, len(vals))
        ax.scatter(np.full(len(vals), xi) + jitter, vals,
                   color=color, alpha=0.4, s=20, zorder=2, edgecolors='none')

    # Build x-axis labels with stats underneath
    tick_labels = []
    for cat_label, vals in zip(labels, all_vals):
        tick_labels.append(f'{cat_label}\n'
                           f'median={np.median(vals):.2f}, '
                           f'mean={np.mean(vals):.2f}\n'
                           f'(n={len(vals)})')

    # Stats annotation in upper right
    rem_vals = tl[tl['MemoryOutcome'] == 'remembered']['WeightedIEDRate_num']
    forg_vals = tl[tl['MemoryOutcome'] == 'forgotten']['WeightedIEDRate_num']
    if len(rem_vals) > 0 and len(forg_vals) > 0:
        u_stat, p_mw = mannwhitneyu(rem_vals, forg_vals, alternative='two-sided')
        ax.text(0.97, 0.97,
                f'Mann-Whitney U = {u_stat:.0f}\np = {p_mw:.4f}',
                transform=ax.transAxes, ha='right', va='top', fontsize=11,
                bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow',
                          edgecolor='gray'))

    ax.set_xticks([0, 1])
    ax.set_xticklabels(tick_labels, fontsize=11)
    ax.set_ylabel('Weighted IED Rate', fontsize=15)
    ax.set_title(f'{phase_label}: Weighted IED Rate Distribution\n'
                 f'Remembered vs Forgotten Trials',
                 fontsize=16, fontweight='bold')
    sns.despine(ax=ax)
    ax.tick_params(axis='y', labelsize=12)
    fig.tight_layout()

    out = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out}')


# ========================== STATS ==========================

def save_stats(trial_level, phase_label, filename):
    rows = []

    for cat in ['Gray', 'White', 'Both', 'Borderline', 'Unknown']:
        sub = trial_level[trial_level['TissueType'] == cat]
        n = len(sub)
        if n == 0:
            continue
        n_rem = (sub['MemoryOutcome'] == 'remembered').sum()
        rows.append({
            'Analysis': 'Tissue Type',
            'Category': cat,
            'N_Remembered': n_rem,
            'N_Forgotten': n - n_rem,
            'N_Total': n,
            'Pct_Remembered': round(n_rem / n * 100, 1),
        })

    gw = trial_level[trial_level['TissueType'].isin(['Gray', 'White'])]
    if gw['TissueType'].nunique() == 2:
        ct = pd.crosstab(gw['TissueType'], gw['MemoryOutcome'])
        if ct.shape == (2, 2):
            chi2, p, _, _ = chi2_contingency(ct)
            rows.append({
                'Analysis': 'Tissue Type',
                'Category': f'Gray vs White Chi2={chi2:.2f}, p={p:.4f}',
                'N_Total': len(gw),
            })

    tl_rate = trial_level[trial_level['WeightedIEDRate_num'].notna()]
    for mem in ['remembered', 'forgotten']:
        vals = tl_rate[tl_rate['MemoryOutcome'] == mem]['WeightedIEDRate_num']
        if len(vals) > 0:
            rows.append({
                'Analysis': 'Weighted IED Rate',
                'Category': mem,
                'N_Total': len(vals),
                'Pct_Remembered': round(vals.mean(), 3),
            })

    rem_vals = tl_rate[tl_rate['MemoryOutcome'] == 'remembered']['WeightedIEDRate_num']
    forg_vals = tl_rate[tl_rate['MemoryOutcome'] == 'forgotten']['WeightedIEDRate_num']
    if len(rem_vals) > 0 and len(forg_vals) > 0:
        u, p_mw = mannwhitneyu(rem_vals, forg_vals, alternative='two-sided')
        rho, p_sp = spearmanr(tl_rate['WeightedIEDRate_num'],
                              (tl_rate['MemoryOutcome'] == 'remembered').astype(int))
        rows.append({
            'Analysis': 'Weighted IED Rate',
            'Category': f'Mann-Whitney U={u:.0f}, p={p_mw:.4f}; Spearman rho={rho:.3f}, p={p_sp:.4f}',
            'N_Total': len(tl_rate),
        })

    tl_rate_binned = tl_rate.copy()
    tl_rate_binned['rate_bin'] = pd.cut(tl_rate_binned['WeightedIEDRate_num'],
                                        bins=[0, 1, 2, 3, 100],
                                        labels=['1', '1-2', '2-3', '3+'],
                                        include_lowest=True)
    for b in ['1', '1-2', '2-3', '3+']:
        sub = tl_rate_binned[tl_rate_binned['rate_bin'] == b]
        n = len(sub)
        if n == 0:
            continue
        n_rem = (sub['MemoryOutcome'] == 'remembered').sum()
        rows.append({
            'Analysis': 'Weighted Rate Bin',
            'Category': b,
            'N_Remembered': n_rem,
            'N_Forgotten': n - n_rem,
            'N_Total': n,
            'Pct_Remembered': round(n_rem / n * 100, 1),
        })

    stats_df = pd.DataFrame(rows)
    out = os.path.join(OUTPUT_DIR, filename)
    stats_df.to_csv(out, index=False)
    print(f'Saved {out}')
    print(stats_df.to_string(index=False))


# ========================== MAIN ==========================

def main():
    # --- Encoding Phase ---
    print('=' * 60)
    print('ENCODING PHASE')
    print('=' * 60)
    enc = load_encoding_trials()
    print(f"Encoding: {len(enc)} trials, {enc['Patient'].nunique()} patients")
    print(f"  TissueType: {enc['TissueType'].value_counts().to_dict()}")
    print(f"  WeightedIEDRate available: {enc['WeightedIEDRate_num'].notna().sum()}")
    print()

    save_stats(enc, 'Encoding Phase', 'encoding_graymatter_weighted_stats.csv')
    print()
    plot_graymatter(enc, 'Encoding Phase', 'encoding_graymatter_memory.png')
    plot_graymatter_by_subject(enc, 'Encoding Phase', 'encoding_graymatter_by_subject.png')
    plot_weighted_rate_bars(enc, 'Encoding Phase', 'encoding_weighted_rate_memory.png')
    plot_weighted_rate_violin(enc, 'Encoding Phase', 'encoding_weighted_rate_violin.png')

    # --- Retrieval Phase ---
    print()
    print('=' * 60)
    print('RETRIEVAL PHASE')
    print('=' * 60)
    ret = load_retrieval_trials()
    print(f"Retrieval: {len(ret)} trials, {ret['Patient'].nunique()} patients")
    print(f"  TissueType: {ret['TissueType'].value_counts().to_dict()}")
    print(f"  WeightedIEDRate available: {ret['WeightedIEDRate_num'].notna().sum()}")
    print()

    save_stats(ret, 'Retrieval Phase', 'retrieval_graymatter_weighted_stats.csv')
    print()
    plot_graymatter(ret, 'Retrieval Phase', 'retrieval_graymatter_memory.png')
    plot_graymatter_by_subject(ret, 'Retrieval Phase', 'retrieval_graymatter_by_subject.png')
    plot_weighted_rate_bars(ret, 'Retrieval Phase', 'retrieval_weighted_rate_memory.png')
    plot_weighted_rate_violin(ret, 'Retrieval Phase', 'retrieval_weighted_rate_violin.png')


if __name__ == '__main__':
    main()
