#!/usr/bin/env python
"""
Quon et al. (2021) Replication — RETRIEVAL Phase
=================================================
Same analyses as ied_quon_replication.py but for the retrieval (test) phase.
IED features during retrieval × memory performance × neural dynamics.

Outputs → outputs/ied_Quon_paper/retrieval/
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.stats import (mannwhitneyu, spearmanr, pointbiserialr)
from itertools import combinations

warnings.filterwarnings('ignore')

# ===================== PATHS =====================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IED_RET_CSV = os.path.join(SCRIPT_DIR, 'IED',
    'AMMEBLAES_IEDs_trial_level_dissertation_test_usethis_cleaned_with_memory.csv')
POWER_CSV = os.path.join(SCRIPT_DIR, 'outputs', 'csvs',
    'combined_retrieval_power_all_mlmr_input.csv')
COH_CSV = os.path.join(SCRIPT_DIR, 'outputs', 'csvs',
    'combined_retrieval_coherence_all_mlmr_input.csv')
PAC_AMME_CSV = os.path.join(SCRIPT_DIR, 'outputs', 'stim_effect_composite_scores',
    'retrieval_pac', 'amme', 'stim_composite_pac_retrieval.csv')
PAC_BLAES_CSV = os.path.join(SCRIPT_DIR, 'outputs', 'stim_effect_composite_scores',
    'retrieval_pac', 'blaes', 'stim_composite_pac_retrieval.csv')

OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'outputs', 'ied_Quon_paper', 'retrieval')
os.makedirs(OUTPUT_DIR, exist_ok=True)

PHASE_LABEL = 'Retrieval'

# Patients excluded from all retrieval analyses (outlier-leverage sensitivity;
# these two patients together contributed ~33% of retrieval IED rows and
# drove the pooled Channel-Spread positive effect disproportionately).
RETRIEVAL_EXCLUDE = {'BJH042', 'UIC202306'}

# ===================== STYLE =====================
CB = sns.color_palette('colorblind')
sns.set_style('ticks')
sns.set_context('talk', font_scale=0.85)

REM_COLOR = CB[2]
FORG_COLOR = CB[4]
SIG_COLOR = '#C44E52'
NS_COLOR = '#555555'
GM_COLOR = CB[0]
WM_COLOR = CB[1]

BANDS_ALL = {
    'Delta (2-4 Hz)': (2, 4),
    'Theta (4-8 Hz)': (4, 8),
    'Alpha (8-12 Hz)': (8, 12),
    'Beta (12-30 Hz)': (12, 30),
    'Slow Gamma (30-55 Hz)': (30, 55),
    'HFA (70-100 Hz)': (70, 100),
}

IED_FEATURES = {
    'mean_rate': 'Mean IED Rate',
    'mean_ch_spread': 'Mean Channel Spread',
    'mean_reg_spread': 'Mean Region Spread',
    'pct_white': '% White Matter',
    'pct_left': '% Left Hemisphere',
    'pct_during_img': '% During Image/ITI',
}

MIN_N_CORR = 8


# ===================== REGION MAPPING =====================
def classify_mtl_region(region_str):
    r = str(region_str).lower()
    if 'amygdala' in r and 'hippocampus' not in r:
        return 'Amygdala'
    elif 'hippocampus' in r and 'amygdala' not in r:
        return 'Hippocampus'
    elif 'parahippocampal' in r:
        return 'Parahippocampal'
    elif 'entorhinal' in r:
        return 'Entorhinal'
    elif 'amygdala' in r and 'hippocampus' in r:
        return 'Amygdala+HPC'
    elif 'temporal' in r:
        return 'Temporal'
    elif 'fusiform' in r:
        return 'Fusiform'
    elif 'frontal' in r or 'orbitofrontal' in r:
        return 'Frontal'
    elif 'parietal' in r or 'precuneus' in r:
        return 'Parietal'
    elif 'insula' in r:
        return 'Insula'
    elif 'cingulate' in r:
        return 'Cingulate'
    else:
        return 'Other'


def is_mtl(region_str):
    cat = classify_mtl_region(region_str)
    return cat in ['Amygdala', 'Hippocampus', 'Parahippocampal', 'Entorhinal', 'Amygdala+HPC']


def classify_tissue(gm):
    gm = str(gm).upper().strip()
    if gm == 'G': return 'Gray'
    elif gm == 'W': return 'White'
    elif gm == 'B': return 'Both'
    return np.nan


# ===================== DATA LOADING =====================
def load_ied_retrieval():
    """Load retrieval IED data — filter to remembered/forgotten only (exclude new_item).

    Also drops high-leverage outlier patients listed in RETRIEVAL_EXCLUDE from
    all retrieval analyses.
    """
    df = pd.read_csv(IED_RET_CSV)
    dm = df[df['MemoryOutcome'].isin(['remembered', 'forgotten'])].copy()

    n_before = len(dm)
    pts_before = dm['Patient'].nunique()
    dm = dm[~dm['Patient'].isin(RETRIEVAL_EXCLUDE)].copy()
    print(f'  Excluded patients {sorted(RETRIEVAL_EXCLUDE)}: '
          f'{n_before - len(dm)} rows dropped ({pts_before} -> '
          f'{dm["Patient"].nunique()} patients)')

    dm['memory_binary'] = (dm['MemoryOutcome'] == 'remembered').astype(int)
    dm['mtl_region'] = dm['Region'].apply(classify_mtl_region)
    dm['is_mtl'] = dm['Region'].apply(is_mtl)
    dm['tissue'] = dm['GrayMatter'].apply(classify_tissue)

    dm['rate_bin'] = pd.cut(dm['Rate(within trials)'],
                            bins=[0, 1, 2, 4, 8],
                            labels=['1', '2', '3-4', '5-8'])

    # Retrieval has DuringImgITI instead of DuringImg
    if 'DuringImgITI' in dm.columns:
        dm['DuringImg'] = dm['DuringImgITI']
    elif 'DuringImg' not in dm.columns:
        dm['DuringImg'] = 'N'

    return dm


def load_patient_ied_summary(dm):
    rows = []
    for pat, pdf in dm.groupby('Patient'):
        n_ieds = len(pdf)
        n_trials = pdf['Trial'].nunique()
        mean_rate = pdf['Rate(within trials)'].mean()
        max_rate = pdf['Rate(within trials)'].max()
        mean_ch_spread = pdf['ChannelSpread'].mean()
        mean_reg_spread = pdf['RegionSpread'].mean()
        pct_gray = (pdf['GrayMatter'] == 'G').mean()
        pct_white = (pdf['GrayMatter'] == 'W').mean()
        pct_left = (pdf['Hemisphere'] == 'L').mean()
        pct_mtl = pdf['is_mtl'].mean()
        pct_during_img = (pdf['DuringImg'] == 'Y').mean() if 'DuringImg' in pdf.columns else 0
        pct_remembered = pdf['memory_binary'].mean()

        mtl_sub = pdf[pdf['is_mtl']]
        pct_amyg = (mtl_sub['mtl_region'] == 'Amygdala').mean() if len(mtl_sub) > 0 else 0
        pct_hpc = (mtl_sub['mtl_region'] == 'Hippocampus').mean() if len(mtl_sub) > 0 else 0

        rows.append({
            'Patient': pat, 'n_ieds': n_ieds, 'n_trials': n_trials,
            'mean_rate': mean_rate, 'max_rate': max_rate,
            'mean_ch_spread': mean_ch_spread, 'mean_reg_spread': mean_reg_spread,
            'pct_gray': pct_gray, 'pct_white': pct_white,
            'pct_left': pct_left, 'pct_mtl': pct_mtl,
            'pct_during_img': pct_during_img,
            'pct_remembered': pct_remembered,
        })
    return pd.DataFrame(rows)


def load_neural_patient_summary():
    pwr = pd.read_csv(POWER_CSV)
    freq_cols = [c for c in pwr.columns if c.startswith('diff_Freq_')]
    freqs = np.array([float(c.replace('diff_Freq_', '')) for c in freq_cols])
    ns = pwr[pwr['trial_type'] == 'nostim'].copy()

    rows = []
    for (pat, region), gdf in ns.groupby(['Patient', 'Region']):
        rem = gdf[gdf['yes_or_no'] == 'yes']
        forg = gdf[gdf['yes_or_no'] == 'no']
        if len(rem) == 0 or len(forg) == 0:
            continue
        row = {'Patient': pat, 'Region': region, 'n_rem': len(rem), 'n_forg': len(forg)}
        for band_name, (flo, fhi) in BANDS_ALL.items():
            mask = (freqs >= flo) & (freqs <= fhi)
            band_cols = [freq_cols[i] for i in range(len(freq_cols)) if mask[i]]
            if not band_cols:
                continue
            row[f'power_SME_{band_name}'] = rem[band_cols].values.mean() - forg[band_cols].values.mean()
        rows.append(row)
    return pd.DataFrame(rows)


def load_coherence_patient_summary():
    coh = pd.read_csv(COH_CSV)
    freq_cols = [c for c in coh.columns if c.startswith('diff_Freq_')]
    freqs = np.array([float(c.replace('diff_Freq_', '')) for c in freq_cols])
    ns = coh[coh['trial_type'] == 'nostim'].copy()

    rows = []
    for (pat, region), gdf in ns.groupby(['Patient', 'Region']):
        rem = gdf[gdf['yes_or_no'] == 'yes']
        forg = gdf[gdf['yes_or_no'] == 'no']
        if len(rem) == 0 or len(forg) == 0:
            continue
        row = {'Patient': pat, 'Region': region}
        for band_name, (flo, fhi) in BANDS_ALL.items():
            mask = (freqs >= flo) & (freqs <= fhi)
            band_cols = [freq_cols[i] for i in range(len(freq_cols)) if mask[i]]
            if not band_cols:
                continue
            row[f'coh_SME_{band_name}'] = rem[band_cols].values.mean() - forg[band_cols].values.mean()
        rows.append(row)
    return pd.DataFrame(rows)


def load_pac_patient_summary():
    dfs = []
    for csv_path in [PAC_AMME_CSV, PAC_BLAES_CSV]:
        if os.path.exists(csv_path):
            dfs.append(pd.read_csv(csv_path))
    if not dfs:
        return pd.DataFrame()
    pac = pd.concat(dfs, ignore_index=True)
    mtl_kw = ['BLA', 'HPC', 'CA', 'DG', 'PRC', 'EC', 'ALLHPC', 'MTL']
    def is_mtl_pair(r):
        parts = r.split('_')
        return len(parts) == 2 and all(p in mtl_kw for p in parts)
    pac_mtl = pac[pac['Region'].apply(is_mtl_pair)].copy()

    rows = []
    for pat, pdf in pac_mtl.groupby('Patient'):
        row = {'Patient': pat}
        for band in ['Slow gamma', 'HFA']:
            bdf = pdf[pdf['Band'] == band]
            if len(bdf) == 0:
                continue
            nr = bdf['nostim_rem'].dropna()
            nf = bdf['nostim_forg'].dropna()
            row[f'pac_SME_{band}'] = (nr.mean() - nf.mean()) if len(nr) > 0 and len(nf) > 0 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


# ===================== STATISTICAL HELPERS =====================
def compute_effect_size_memory(feature_series, memory_series):
    mask = feature_series.notna() & memory_series.notna()
    n = int(mask.sum())
    if n < 10:
        return np.nan, np.nan, n, np.nan
    r, p = pointbiserialr(memory_series[mask], feature_series[mask])
    se = np.sqrt(max(0.0, 1.0 - r * r) / max(1, n - 2))
    return r, p, n, se


def _se_propdiff(y_grp, n_grp):
    n1, n2 = len(y_grp), len(n_grp)
    if n1 < 2 or n2 < 2:
        return np.nan
    p1, p2 = y_grp.mean(), n_grp.mean()
    return float(np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2))


def _sig_star(p):
    if p < 0.001: return '***'
    elif p < 0.01: return '**'
    elif p < 0.05: return '*'
    return ''


def remembered_by_group(dm, groupcol):
    results = []
    for name, gdf in dm.groupby(groupcol):
        n = len(gdf)
        if n < 5:
            continue
        p_rem = gdf['memory_binary'].mean()
        se = np.sqrt(p_rem * (1 - p_rem) / n)
        results.append({
            'group': name, 'prop_remembered': p_rem,
            'se': se, 'n': n,
            'ci_lo': max(0, p_rem - 1.96 * se),
            'ci_hi': min(1, p_rem + 1.96 * se),
        })
    return pd.DataFrame(results)


# ===================== HEATMAP / SCATTER HELPERS =====================
def _corr_heatmap(rho_df, p_df, n_df, row_label, col_label,
                  title, filepath, figsize=None):
    if rho_df.empty:
        return
    # Drop rows and columns that are entirely NaN (no data)
    keep_rows = ~rho_df.isna().all(axis=1)
    keep_cols = ~rho_df.isna().all(axis=0)
    rho_df = rho_df.loc[keep_rows, keep_cols]
    p_df = p_df.loc[keep_rows, keep_cols]
    n_df = n_df.loc[keep_rows, keep_cols]
    if rho_df.empty:
        return
    if figsize is None:
        figsize = (max(8, len(rho_df.columns)*1.6), max(4, len(rho_df)*0.7))
    fig, ax = plt.subplots(figsize=figsize)
    mask = rho_df.isna()
    sns.heatmap(rho_df, annot=False, cmap='RdBu_r', center=0,
                vmin=-1, vmax=1, mask=mask, linewidths=0.5,
                linecolor='white', ax=ax, cbar_kws={'label': 'Spearman ρ'})
    for i in range(len(rho_df)):
        for j in range(len(rho_df.columns)):
            rho = rho_df.iat[i, j]
            p = p_df.iat[i, j]
            n = n_df.iat[i, j]
            if pd.isna(rho):
                continue
            star = _sig_star(p)
            txt = f'{rho:.2f}{star}\nn={int(n)}'
            color = 'white' if abs(rho) > 0.55 else 'black'
            ax.text(j + 0.5, i + 0.5, txt, ha='center', va='center',
                    fontsize=7.5, color=color, fontweight='bold' if star else 'normal')
    ax.set_xlabel(col_label, fontsize=11)
    ax.set_ylabel(row_label, fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def _scatter_grid(merged, sme_cols, ied_feat, ied_label,
                  sme_label, title, filepath, color=None):
    n_bands = len(sme_cols)
    fig, axes = plt.subplots(1, n_bands, figsize=(4.5 * n_bands, 4.5))
    if n_bands == 1:
        axes = [axes]
    for idx, (sme_col, band_name) in enumerate(sme_cols):
        ax = axes[idx]
        sub = merged[[ied_feat, sme_col]].dropna()
        if len(sub) < MIN_N_CORR:
            ax.set_visible(False)
            continue
        x = sub[ied_feat].values
        y = sub[sme_col].values
        c = color if color else CB[0]
        ax.scatter(x, y, alpha=0.6, s=50, c=c, edgecolors='black', linewidth=0.3)
        r, p = spearmanr(x, y)
        if len(x) > 2:
            z = np.polyfit(x, y, 1)
            x_line = np.linspace(x.min(), x.max(), 100)
            ax.plot(x_line, np.polyval(z, x_line), color='black', linestyle='--', linewidth=1.5)
        ax.axhline(0, color='gold', linewidth=1, alpha=0.7)
        ax.set_xlabel(ied_label, fontsize=10)
        ax.set_ylabel(f'{sme_label}\n(Rem − Forg)', fontsize=10)
        star = _sig_star(p)
        tc = SIG_COLOR if p < 0.05 else 'black'
        ax.set_title(f'{band_name}\nρ={r:.3f}, p={p:.3f} {star}',
                     fontsize=11, color=tc, fontweight='bold' if star else 'normal')
        sns.despine(ax=ax)
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


# ===================== FIGURE 2: IED FEATURES & MEMORY =====================
def figure2_ied_features_memory(dm):
    print(f'\n=== FIGURE 2: {PHASE_LABEL} IED Features & Memory ===')
    features_results = []
    stats_rows = []

    # IED Rate
    r, p, n, se = compute_effect_size_memory(dm['Rate(within trials)'], dm['memory_binary'])
    features_results.append({'Feature': 'IED Rate\n(IEDs/trial)', 'beta': r, 'p': p, 'se': se})
    stats_rows.append({'Feature': 'IED Rate', 'r_or_beta': r, 'se': se,
                       'p_value': p, 'test': 'point-biserial', 'n_yes': n, 'n_no': np.nan})
    print(f'  IED Rate: r={r:+.4f} +/- {se:.4f}, p={p:.4g}, n={n}')

    # Hemisphere
    hem_data = dm[dm['Hemisphere'].isin(['L', 'R'])].copy()
    left_rem = hem_data[hem_data['Hemisphere'] == 'L']['memory_binary']
    right_rem = hem_data[hem_data['Hemisphere'] == 'R']['memory_binary']
    if len(left_rem) > 5 and len(right_rem) > 5:
        diff = right_rem.mean() - left_rem.mean()
        _, p = mannwhitneyu(right_rem, left_rem, alternative='two-sided')
        se = _se_propdiff(right_rem, left_rem)
        features_results.append({'Feature': 'IED Hemisphere\nRight -\nLeft (ref)',
                                 'beta': diff, 'p': p, 'se': se})
        stats_rows.append({'Feature': 'Hemisphere (R-L)', 'r_or_beta': diff, 'se': se,
                           'p_value': p, 'test': 'Mann-Whitney',
                           'n_yes': int(len(right_rem)), 'n_no': int(len(left_rem))})
        print(f'  Hemisphere (R-L): diff={diff:+.4f} +/- {se:.4f}, p={p:.4g}')

    # MTL vs non-MTL
    mtl_rem = dm[dm['is_mtl']]['memory_binary']
    nonmtl_rem = dm[~dm['is_mtl']]['memory_binary']
    if len(mtl_rem) > 5 and len(nonmtl_rem) > 5:
        diff = nonmtl_rem.mean() - mtl_rem.mean()
        _, p = mannwhitneyu(nonmtl_rem, mtl_rem, alternative='two-sided')
        se = _se_propdiff(nonmtl_rem, mtl_rem)
        features_results.append({'Feature': 'Region\nnon-MTL -\nMTL (ref)',
                                 'beta': diff, 'p': p, 'se': se})
        stats_rows.append({'Feature': 'Region (nonMTL-MTL)', 'r_or_beta': diff, 'se': se,
                           'p_value': p, 'test': 'Mann-Whitney',
                           'n_yes': int(len(nonmtl_rem)), 'n_no': int(len(mtl_rem))})
        print(f'  Region (nonMTL-MTL): diff={diff:+.4f} +/- {se:.4f}, p={p:.4g}')

    # Gray vs White
    tissue_data = dm[dm['tissue'].isin(['Gray', 'White'])].copy()
    gray_rem = tissue_data[tissue_data['tissue'] == 'Gray']['memory_binary']
    white_rem = tissue_data[tissue_data['tissue'] == 'White']['memory_binary']
    if len(gray_rem) > 5 and len(white_rem) > 5:
        diff = white_rem.mean() - gray_rem.mean()
        _, p = mannwhitneyu(white_rem, gray_rem, alternative='two-sided')
        se = _se_propdiff(white_rem, gray_rem)
        features_results.append({'Feature': 'White Matter\nPropagation\nYes - No (ref)',
                                 'beta': diff, 'p': p, 'se': se})
        stats_rows.append({'Feature': 'White Matter (W-G)', 'r_or_beta': diff, 'se': se,
                           'p_value': p, 'test': 'Mann-Whitney',
                           'n_yes': int(len(white_rem)), 'n_no': int(len(gray_rem))})
        print(f'  White Matter (W-G): diff={diff:+.4f} +/- {se:.4f}, p={p:.4g}')

    # Channel Spread
    r, p, n, se = compute_effect_size_memory(dm['ChannelSpread'], dm['memory_binary'])
    features_results.append({'Feature': 'Channel\nSpread', 'beta': r, 'p': p, 'se': se})
    stats_rows.append({'Feature': 'Channel Spread', 'r_or_beta': r, 'se': se,
                       'p_value': p, 'test': 'point-biserial', 'n_yes': n, 'n_no': np.nan})
    print(f'  Channel Spread: r={r:+.4f} +/- {se:.4f}, p={p:.4g}, n={n}')

    # Region Spread
    r, p, n, se = compute_effect_size_memory(dm['RegionSpread'], dm['memory_binary'])
    features_results.append({'Feature': 'Region\nSpread', 'beta': r, 'p': p, 'se': se})
    stats_rows.append({'Feature': 'Region Spread', 'r_or_beta': r, 'se': se,
                       'p_value': p, 'test': 'point-biserial', 'n_yes': n, 'n_no': np.nan})
    print(f'  Region Spread: r={r:+.4f} +/- {se:.4f}, p={p:.4g}, n={n}')

    # Option C: retrieval has only two timing windows (BeforeImgITI, DuringImgITI).
    # "Only" means: this window is Y and the other window is N.
    RET_WINDOWS = ['BeforeImgITI', 'DuringImgITI']

    def add_ret_window_only(this_col, plot_label, stats_label):
        if this_col not in dm.columns:
            return
        others = [c for c in RET_WINDOWS if c != this_col and c in dm.columns]
        only_mask = (dm[this_col] == 'Y')
        for c in others:
            only_mask &= (dm[c] == 'N')
        yes_rem = dm.loc[only_mask, 'memory_binary']
        no_rem  = dm.loc[~only_mask, 'memory_binary']
        if len(yes_rem) <= 5 or len(no_rem) <= 5:
            return
        diff = yes_rem.mean() - no_rem.mean()
        _, p = mannwhitneyu(yes_rem, no_rem, alternative='two-sided')
        se = _se_propdiff(yes_rem, no_rem)
        features_results.append({'Feature': plot_label, 'beta': diff, 'p': p, 'se': se})
        stats_rows.append({'Feature': stats_label, 'r_or_beta': diff, 'se': se,
                           'p_value': p, 'test': 'Mann-Whitney (only vs others)',
                           'n_yes': int(len(yes_rem)), 'n_no': int(len(no_rem))})
        print(f'  {stats_label}: diff={diff:+.4f} +/- {se:.4f}, '
              f'p={p:.4g}, n_only={len(yes_rem)}, n_other={len(no_rem)}')

    add_ret_window_only('BeforeImgITI',
        'Before Image\nOnly -\nOther (ref)', 'Before Image (Only)')
    add_ret_window_only('DuringImgITI',
        'During Image/ITI\nOnly -\nOther (ref)', 'During Image/ITI (Only)')

    fr = pd.DataFrame(features_results)

    # ---- PLOT ----
    fig = plt.figure(figsize=(18, 14))
    gs = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.35)

    # Panel A
    ax_a = fig.add_subplot(gs[0, :])
    bar_colors = [SIG_COLOR if row['p'] < 0.05 else NS_COLOR for _, row in fr.iterrows()]
    ax_a.bar(range(len(fr)), fr['beta'], color=bar_colors, edgecolor='black', linewidth=0.5)
    ax_a.errorbar(range(len(fr)), fr['beta'], yerr=fr['se'].astype(float),
                  fmt='none', ecolor='black', capsize=4, linewidth=1.2)
    ax_a.axhline(0, color='black', linestyle='--', linewidth=0.8)
    ax_a.set_xticks(range(len(fr)))
    ax_a.set_xticklabels(fr['Feature'], fontsize=10)
    ax_a.set_ylabel('Effect Size (r or prop. difference) +/- 1 SE', fontsize=12)
    ax_a.set_title(f'(A) {PHASE_LABEL}: IED Features Associated with Memory'
                   f' (excl. {", ".join(sorted(RETRIEVAL_EXCLUDE))})',
                   fontsize=14, fontweight='bold')
    for i, row in fr.iterrows():
        star = _sig_star(row['p'])
        if star:
            se_i = row['se'] if pd.notna(row['se']) else 0
            y = row['beta']
            y_pos = y + se_i + 0.005 if y >= 0 else y - se_i - 0.01
            ax_a.text(i, y_pos, star, ha='center',
                     va='bottom' if y >= 0 else 'top',
                     fontsize=14, fontweight='bold', color=SIG_COLOR)
    sns.despine(ax=ax_a)

    # Panel B: Rate × Tissue
    ax_b = fig.add_subplot(gs[1, 0])
    tissue_rates = dm[dm['tissue'].isin(['Gray', 'White'])].copy()
    if len(tissue_rates) > 20:
        median_rate = tissue_rates['Rate(within trials)'].median()
        tissue_rates['rate_group'] = np.where(
            tissue_rates['Rate(within trials)'] <= median_rate, 'Low Rate', 'High Rate')
        interaction = tissue_rates.groupby(['rate_group', 'tissue'])['memory_binary'].agg(
            ['mean', 'count', 'sem']).reset_index()
        for tissue, color, marker in [('Gray', GM_COLOR, 'o'), ('White', WM_COLOR, 's')]:
            sub = interaction[interaction['tissue'] == tissue].sort_values('rate_group')
            if len(sub) > 0:
                ax_b.errorbar(sub['rate_group'], sub['mean'], yerr=1.96*sub['sem'],
                             color=color, marker=marker, markersize=10, linewidth=2,
                             label=f'{tissue} Matter', capsize=5)
        ax_b.set_ylabel('Proportion Remembered', fontsize=12)
        ax_b.set_xlabel('IED Rate Group', fontsize=12)
        ax_b.set_title(f'(B) {PHASE_LABEL}: IED Rate × Tissue Type', fontsize=14, fontweight='bold')
        ax_b.legend(fontsize=10)
        ax_b.set_ylim(0, 1)
        sns.despine(ax=ax_b)

    # Panel C: Pairwise MTL regions
    ax_c = fig.add_subplot(gs[1, 1])
    mtl_regions = ['Amygdala', 'Hippocampus', 'Parahippocampal', 'Entorhinal']
    mtl_data = dm[dm['mtl_region'].isin(mtl_regions)].copy()
    if len(mtl_data) > 20:
        pair_results = []
        for r1, r2 in combinations(mtl_regions, 2):
            g1 = mtl_data[mtl_data['mtl_region'] == r1]['memory_binary']
            g2 = mtl_data[mtl_data['mtl_region'] == r2]['memory_binary']
            if len(g1) < 5 or len(g2) < 5:
                continue
            diff = g1.mean() - g2.mean()
            se = np.sqrt(g1.var()/len(g1) + g2.var()/len(g2))
            _, p = mannwhitneyu(g1, g2, alternative='two-sided')
            pair_results.append({
                'pair': f'{r1[:4]} - {r2[:4]} (ref)',
                'diff': diff, 'se': se, 'p': p,
                'ci_lo': diff - 1.96*se, 'ci_hi': diff + 1.96*se,
            })
            stats_rows.append({'Feature': f'Region: {r1} vs {r2}',
                'r_or_beta': diff, 'p_value': p, 'test': 'Mann-Whitney'})
            print(f'  {r1} vs {r2}: diff={diff:.4f}, p={p:.4g}')

        pr = pd.DataFrame(pair_results)
        if len(pr) > 0:
            y_pos = range(len(pr))
            colors_c = [SIG_COLOR if row['p'] < 0.05 else NS_COLOR for _, row in pr.iterrows()]
            ax_c.errorbar(pr['diff'], y_pos, xerr=[pr['diff']-pr['ci_lo'], pr['ci_hi']-pr['diff']],
                         fmt='none', ecolor='gray', capsize=4, linewidth=1.5)
            ax_c.scatter(pr['diff'], y_pos, c=colors_c, s=80, zorder=5, edgecolors='black', linewidth=0.5)
            ax_c.axvline(0, color='black', linestyle='--', linewidth=0.8)
            ax_c.set_yticks(list(y_pos))
            ax_c.set_yticklabels(pr['pair'], fontsize=10)
            ax_c.set_xlabel('Δ Proportion Remembered (± 95% CI)', fontsize=11)
            ax_c.set_title(f'(C) {PHASE_LABEL}: Pairwise MTL Region Comparisons', fontsize=14, fontweight='bold')
            for i, row in pr.iterrows():
                if row['p'] < 0.05:
                    ax_c.text(row['ci_hi'] + 0.01, i, '*', fontsize=14,
                             fontweight='bold', color=SIG_COLOR, va='center')
            sns.despine(ax=ax_c)

    plt.savefig(os.path.join(OUTPUT_DIR, 'figure2_retrieval_ied_features_memory.png'),
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    stats_df = pd.DataFrame(stats_rows)
    stats_df.to_csv(os.path.join(OUTPUT_DIR, 'figure2_retrieval_stats.csv'), index=False)
    print(f'  Saved figure2_retrieval_ied_features_memory.png')

    # Supplementary region breakdown
    fig, ax = plt.subplots(figsize=(10, 6))
    region_stats = remembered_by_group(mtl_data, 'mtl_region')
    if len(region_stats) > 0:
        region_stats = region_stats.sort_values('prop_remembered')
        bars = ax.barh(range(len(region_stats)), region_stats['prop_remembered'],
                       xerr=1.96*region_stats['se'], color=CB[0], edgecolor='black',
                       linewidth=0.5, capsize=4)
        ax.set_yticks(range(len(region_stats)))
        ax.set_yticklabels(region_stats['group'])
        ax.set_xlabel('Proportion Remembered (± 95% CI)')
        ax.set_title(f'{PHASE_LABEL}: Memory Performance by MTL Region of IED')
        for i, row in region_stats.reset_index().iterrows():
            ax.text(row['prop_remembered'] + row['se']*1.96 + 0.02, i,
                    f"n={row['n']}", va='center', fontsize=9)
    sns.despine(ax=ax)
    plt.savefig(os.path.join(OUTPUT_DIR, 'figure2_retrieval_supp_region_breakdown.png'),
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    return fr, stats_df


# ===================== FIGURE 3: NEURAL × IED × MEMORY =====================

def figure3_power_by_region(pat_ied, pwr_summary):
    print(f'\n--- {PHASE_LABEL}: POWER by Region × IED Features ---')
    bands = list(BANDS_ALL.keys())
    regions = sorted(pwr_summary['Region'].unique())
    all_stats = []

    for region in regions:
        reg_pwr = pwr_summary[pwr_summary['Region'] == region].copy()
        sme_cols = [f'power_SME_{b}' for b in bands]
        available = [c for c in sme_cols if c in reg_pwr.columns]
        if not available:
            continue
        merged = reg_pwr[['Patient'] + available].merge(pat_ied, on='Patient', how='inner')
        n_pts = len(merged)
        if n_pts < MIN_N_CORR:
            print(f'  {region}: skipping (n={n_pts})')
            continue

        rho_data, p_data, n_data = {}, {}, {}
        for feat, feat_label in IED_FEATURES.items():
            rho_row, p_row, n_row = {}, {}, {}
            for sme_col in available:
                band_name = sme_col.replace('power_SME_', '')
                sub = merged[[feat, sme_col]].dropna()
                if len(sub) < MIN_N_CORR:
                    rho_row[band_name] = np.nan; p_row[band_name] = np.nan; n_row[band_name] = 0
                else:
                    r, p = spearmanr(sub[feat], sub[sme_col])
                    rho_row[band_name] = r; p_row[band_name] = p; n_row[band_name] = len(sub)
                    all_stats.append({
                        'Measure': 'Power', 'Region': region,
                        'Band': band_name, 'IED_Feature': feat,
                        'IED_Feature_Label': feat_label,
                        'rho': r, 'p_value': p, 'n': len(sub)})
            rho_data[feat_label] = rho_row; p_data[feat_label] = p_row; n_data[feat_label] = n_row

        _corr_heatmap(pd.DataFrame(rho_data).T, pd.DataFrame(p_data).T, pd.DataFrame(n_data).T,
                      'IED Feature', 'Frequency Band',
                      f'{PHASE_LABEL} Power SME × IED Features — {region} (n={n_pts})',
                      os.path.join(OUTPUT_DIR, f'fig3_power_heatmap_{region}.png'))
        print(f'  {region}: heatmap saved (n={n_pts})')

        sme_pairs = [(c, c.replace('power_SME_', '')) for c in available]
        _scatter_grid(merged, sme_pairs, 'mean_rate', 'Mean IED Rate',
                      f'{region} Power SME',
                      f'{PHASE_LABEL}: IED Rate × Power SME — {region} (n={n_pts})',
                      os.path.join(OUTPUT_DIR, f'fig3_power_scatter_{region}.png'), color=CB[0])

    # Summary heatmaps
    for feat, feat_label in IED_FEATURES.items():
        rho_s, p_s, n_s = {}, {}, {}
        for region in regions:
            reg_pwr = pwr_summary[pwr_summary['Region'] == region]
            rho_row, p_row, n_row = {}, {}, {}
            for band in bands:
                sme_col = f'power_SME_{band}'
                if sme_col not in reg_pwr.columns:
                    rho_row[band] = np.nan; p_row[band] = np.nan; n_row[band] = 0
                    continue
                merged = reg_pwr[['Patient', sme_col]].merge(
                    pat_ied[['Patient', feat]], on='Patient', how='inner').dropna()
                if len(merged) < MIN_N_CORR:
                    rho_row[band] = np.nan; p_row[band] = np.nan; n_row[band] = 0
                else:
                    r, p = spearmanr(merged[feat], merged[sme_col])
                    rho_row[band] = r; p_row[band] = p; n_row[band] = len(merged)
            rho_s[region] = rho_row; p_s[region] = p_row; n_s[region] = n_row
        _corr_heatmap(pd.DataFrame(rho_s).T, pd.DataFrame(p_s).T, pd.DataFrame(n_s).T,
                      'Brain Region', 'Frequency Band',
                      f'{PHASE_LABEL} Power SME × {feat_label} — All Regions',
                      os.path.join(OUTPUT_DIR, f'fig3_power_heatmap_ALL_regions_{feat}.png'),
                      figsize=(14, 6))

    return pd.DataFrame(all_stats)


def figure3_coherence_by_pair(pat_ied, coh_summary):
    print(f'\n--- {PHASE_LABEL}: COHERENCE by Region Pair × IED Features ---')
    bands = list(BANDS_ALL.keys())
    pairs = sorted(coh_summary['Region'].unique())
    all_stats = []

    pair_counts = coh_summary.groupby('Region')['Patient'].nunique()
    good_pairs = pair_counts[pair_counts >= MIN_N_CORR].index.tolist()
    print(f'  {len(good_pairs)}/{len(pairs)} pairs have n >= {MIN_N_CORR}')

    for pair in good_pairs:
        pair_coh = coh_summary[coh_summary['Region'] == pair].copy()
        sme_cols = [f'coh_SME_{b}' for b in bands]
        available = [c for c in sme_cols if c in pair_coh.columns]
        if not available:
            continue
        merged = pair_coh[['Patient'] + available].merge(pat_ied, on='Patient', how='inner')
        n_pts = len(merged)
        if n_pts < MIN_N_CORR:
            continue

        rho_data, p_data, n_data = {}, {}, {}
        for feat, feat_label in IED_FEATURES.items():
            rho_row, p_row, n_row = {}, {}, {}
            for sme_col in available:
                band_name = sme_col.replace('coh_SME_', '')
                sub = merged[[feat, sme_col]].dropna()
                if len(sub) < MIN_N_CORR:
                    rho_row[band_name] = np.nan; p_row[band_name] = np.nan; n_row[band_name] = 0
                else:
                    r, p = spearmanr(sub[feat], sub[sme_col])
                    rho_row[band_name] = r; p_row[band_name] = p; n_row[band_name] = len(sub)
                    all_stats.append({
                        'Measure': 'Coherence', 'Region': pair,
                        'Band': band_name, 'IED_Feature': feat,
                        'IED_Feature_Label': feat_label,
                        'rho': r, 'p_value': p, 'n': len(sub)})
            rho_data[feat_label] = rho_row; p_data[feat_label] = p_row; n_data[feat_label] = n_row

        _corr_heatmap(pd.DataFrame(rho_data).T, pd.DataFrame(p_data).T, pd.DataFrame(n_data).T,
                      'IED Feature', 'Frequency Band',
                      f'{PHASE_LABEL} Coherence SME × IED Features — {pair} (n={n_pts})',
                      os.path.join(OUTPUT_DIR, f'fig3_coh_heatmap_{pair}.png'))

        sme_pairs_list = [(c, c.replace('coh_SME_', '')) for c in available]
        _scatter_grid(merged, sme_pairs_list, 'mean_rate', 'Mean IED Rate',
                      f'{pair} Coh SME',
                      f'{PHASE_LABEL}: IED Rate × Coherence SME — {pair} (n={n_pts})',
                      os.path.join(OUTPUT_DIR, f'fig3_coh_scatter_{pair}.png'), color=CB[3])

    print(f'  Saved per-pair heatmaps & scatters for {len(good_pairs)} pairs')

    # Summary heatmaps
    for feat, feat_label in IED_FEATURES.items():
        rho_s, p_s, n_s = {}, {}, {}
        for pair in good_pairs:
            pair_coh = coh_summary[coh_summary['Region'] == pair]
            rho_row, p_row, n_row = {}, {}, {}
            for band in bands:
                sme_col = f'coh_SME_{band}'
                if sme_col not in pair_coh.columns:
                    rho_row[band] = np.nan; p_row[band] = np.nan; n_row[band] = 0
                    continue
                merged = pair_coh[['Patient', sme_col]].merge(
                    pat_ied[['Patient', feat]], on='Patient', how='inner').dropna()
                if len(merged) < MIN_N_CORR:
                    rho_row[band] = np.nan; p_row[band] = np.nan; n_row[band] = 0
                else:
                    r, p = spearmanr(merged[feat], merged[sme_col])
                    rho_row[band] = r; p_row[band] = p; n_row[band] = len(merged)
            rho_s[pair] = rho_row; p_s[pair] = p_row; n_s[pair] = n_row
        _corr_heatmap(pd.DataFrame(rho_s).T, pd.DataFrame(p_s).T, pd.DataFrame(n_s).T,
                      'Region Pair', 'Frequency Band',
                      f'{PHASE_LABEL} Coherence SME × {feat_label} — All Pairs',
                      os.path.join(OUTPUT_DIR, f'fig3_coh_heatmap_ALL_pairs_{feat}.png'),
                      figsize=(14, max(6, len(good_pairs)*0.6)))

    return pd.DataFrame(all_stats)


def figure3_pac(pat_ied, pac_summary):
    print(f'\n--- {PHASE_LABEL}: PAC × IED Features ---')
    if pac_summary.empty:
        print('  No PAC data')
        return pd.DataFrame()
    all_stats = []
    pac_sme_cols = [c for c in pac_summary.columns if c.startswith('pac_SME_')]
    if not pac_sme_cols:
        return pd.DataFrame()

    merged = pac_summary.merge(pat_ied, on='Patient', how='inner')
    n_pts = len(merged)
    if n_pts < MIN_N_CORR:
        print(f'  Skipping PAC (n={n_pts})')
        return pd.DataFrame()

    rho_data, p_data, n_data = {}, {}, {}
    for feat, feat_label in IED_FEATURES.items():
        rho_row, p_row, n_row = {}, {}, {}
        for sme_col in pac_sme_cols:
            band_name = sme_col.replace('pac_SME_', '')
            sub = merged[[feat, sme_col]].dropna()
            if len(sub) < MIN_N_CORR:
                rho_row[band_name] = np.nan; p_row[band_name] = np.nan; n_row[band_name] = 0
            else:
                r, p = spearmanr(sub[feat], sub[sme_col])
                rho_row[band_name] = r; p_row[band_name] = p; n_row[band_name] = len(sub)
                all_stats.append({
                    'Measure': 'PAC', 'Region': 'MTL_composite',
                    'Band': band_name, 'IED_Feature': feat,
                    'IED_Feature_Label': feat_label,
                    'rho': r, 'p_value': p, 'n': len(sub)})
        rho_data[feat_label] = rho_row; p_data[feat_label] = p_row; n_data[feat_label] = n_row

    _corr_heatmap(pd.DataFrame(rho_data).T, pd.DataFrame(p_data).T, pd.DataFrame(n_data).T,
                  'IED Feature', 'PAC Band',
                  f'{PHASE_LABEL} PAC SME × IED Features (n={n_pts})',
                  os.path.join(OUTPUT_DIR, 'fig3_pac_heatmap.png'))

    sme_pairs_list = [(c, c.replace('pac_SME_', '')) for c in pac_sme_cols]
    _scatter_grid(merged, sme_pairs_list, 'mean_rate', 'Mean IED Rate',
                  'PAC SME',
                  f'{PHASE_LABEL}: IED Rate × PAC SME (n={n_pts})',
                  os.path.join(OUTPUT_DIR, 'fig3_pac_scatter_rate.png'), color=CB[5])

    print(f'  PAC heatmap & scatter saved (n={n_pts})')
    return pd.DataFrame(all_stats)


# ===================== DESCRIPTIVE TABLE =====================
def save_descriptive_table(dm):
    rows = []
    rows.append(('Patients, n', dm['Patient'].nunique()))
    rows.append(('Total IED detections (with memory)', len(dm)))
    rows.append(('Trials with IEDs, n', dm.groupby(['Patient','Trial']).ngroups))
    rows.append(('', ''))
    rows.append(('Memory outcome, n (%)', ''))
    n_rem = (dm['MemoryOutcome']=='remembered').sum()
    n_forg = (dm['MemoryOutcome']=='forgotten').sum()
    rows.append(('  Remembered', f"{n_rem} ({n_rem/len(dm)*100:.1f}%)"))
    rows.append(('  Forgotten', f"{n_forg} ({n_forg/len(dm)*100:.1f}%)"))
    rows.append(('', ''))
    rows.append(('IED Rate (IEDs/trial)', ''))
    rows.append(('  Mean (SD)', f"{dm['Rate(within trials)'].mean():.2f} (±{dm['Rate(within trials)'].std():.2f})"))
    rows.append(('Channel Spread, M (SD)', f"{dm['ChannelSpread'].mean():.2f} (±{dm['ChannelSpread'].std():.2f})"))
    rows.append(('Region Spread, M (SD)', f"{dm['RegionSpread'].mean():.2f} (±{dm['RegionSpread'].std():.2f})"))
    rows.append(('', ''))
    rows.append(('Hemisphere, n (%)', ''))
    for h in ['L', 'R']:
        n = (dm['Hemisphere'] == h).sum()
        rows.append((f'  {h}', f"{n} ({n/len(dm)*100:.1f}%)"))
    rows.append(('', ''))
    rows.append(('Gray/White Matter, n (%)', ''))
    for gm, label in [('G', 'Gray'), ('W', 'White'), ('B', 'Both')]:
        n = (dm['GrayMatter'] == gm).sum()
        rows.append((f'  {label}', f"{n} ({n/len(dm)*100:.1f}%)"))
    rows.append(('', ''))
    rows.append(('MTL Region, n (%)', ''))
    for reg in ['Amygdala', 'Hippocampus', 'Parahippocampal', 'Entorhinal']:
        sub = dm[dm['mtl_region'] == reg]
        rows.append((f'  {reg}', f"{len(sub)} ({len(sub)/len(dm)*100:.1f}%)"))

    table = pd.DataFrame(rows, columns=['Characteristic', 'Value'])
    table.to_csv(os.path.join(OUTPUT_DIR, 'table1_retrieval_descriptive_stats.csv'), index=False)
    print('  Saved table1_retrieval_descriptive_stats.csv')


# ===================== MAIN =====================
def main():
    print('=' * 70)
    print(f'Quon et al. Replication — {PHASE_LABEL} Phase')
    print('=' * 70)

    dm = load_ied_retrieval()
    print(f'Loaded {len(dm)} IED detections from {dm["Patient"].nunique()} patients')
    print(f'  Remembered: {(dm["MemoryOutcome"]=="remembered").sum()}, '
          f'Forgotten: {(dm["MemoryOutcome"]=="forgotten").sum()}')

    save_descriptive_table(dm)
    figure2_ied_features_memory(dm)

    # Neural analysis
    pat_ied = load_patient_ied_summary(dm)
    print(f'\n  Loading {PHASE_LABEL.lower()} power data...')
    pwr_summary = load_neural_patient_summary()
    print(f'  Loading {PHASE_LABEL.lower()} coherence data...')
    coh_summary = load_coherence_patient_summary()
    print(f'  Loading {PHASE_LABEL.lower()} PAC data...')
    pac_summary = load_pac_patient_summary()

    pwr_stats = figure3_power_by_region(pat_ied, pwr_summary)
    coh_stats = figure3_coherence_by_pair(pat_ied, coh_summary)
    pac_stats = figure3_pac(pat_ied, pac_summary)

    all_stats = pd.concat([pwr_stats, coh_stats, pac_stats], ignore_index=True)
    all_stats.to_csv(os.path.join(OUTPUT_DIR, 'figure3_retrieval_stats.csv'), index=False)

    sig = all_stats[all_stats['p_value'] < 0.05]
    print(f'\n  Total correlations tested: {len(all_stats)}')
    print(f'  Significant at p<.05: {len(sig)}')
    if len(sig) > 0:
        print(f'\n  ---- Significant findings (p < .05) ----')
        for _, row in sig.iterrows():
            print(f'    {row["Measure"]} | {row["Region"]} | {row["Band"]} | '
                  f'{row["IED_Feature_Label"]}: ρ={row["rho"]:.3f}, p={row["p_value"]:.4f}, n={row["n"]}')

    print(f'\n{"="*70}')
    print(f'All {PHASE_LABEL} outputs saved to: {OUTPUT_DIR}')
    print('=' * 70)


if __name__ == '__main__':
    main()
