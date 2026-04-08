#!/usr/bin/env python
"""
Hypothesis 3d: IED Spread, Brain Coverage, Neural Dynamics, & Memory Modulation
================================================================================
Do IEDs with larger spread and MTL coverage (BLA, HPC, PRC) during encoding
cause more interruptions in functional neural dynamics (coherence, PAC),
leading to more extreme memory modulation (avg_stim_dprime_diff)?

Key MTL regions in IED data:
  BLA = "Amygdala"
  HPC = "Hippocampus" (includes CA, DG subfields)
  PRC = "Parahippocampal" (perirhinal cortex area)

Outputs in outputs/ied_spread_neural_memory/:
  Figures & stats CSVs
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, pearsonr, mannwhitneyu, kruskal

warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IED_CSV = os.path.join(SCRIPT_DIR, 'IED',
                       'AMMEBLAES_IEDs_trial_level_dissertation_study_usethis_cleaned_with_memory.csv')
BEH_CSV = os.path.join(SCRIPT_DIR, 'behavioral figures',
                       'AMMEBLAES_includedpts_firstsession_behavioral.csv')
COH_CSV = os.path.join(SCRIPT_DIR, 'outputs', 'csvs',
                       'combined_encoding_coherence_amme_mlmr_input.csv')
PAC_CSV = os.path.join(SCRIPT_DIR, 'outputs', 'stim_effect_composite_scores',
                       'encoding_pac', 'amme', 'stim_composite_pac_encoding.csv')

OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'outputs', 'ied_spread_neural_memory')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Colorblind-friendly palette
CB = sns.color_palette('colorblind')
sns.set_style('ticks')
sns.set_context('talk', font_scale=0.9)

# ---------------------------------------------------------------------------
# Region mapping: IED Region → MTL category
# ---------------------------------------------------------------------------
def _mtl_flags(region_str):
    """Return set of MTL categories covered by this region string."""
    r = str(region_str).lower()
    flags = set()
    if 'amygdala' in r:
        flags.add('BLA')
    if any(k in r for k in ['hippocampus', ' ca,', ' ca ', ' dg,', ' dg ']):
        flags.add('HPC')
    if 'parahippocampal' in r:
        flags.add('PRC')
    if 'entorhinal' in r:
        flags.add('EC')
    return flags


# ========================== DATA LOADING ==========================

def load_patient_ied_features():
    """Compute per-patient IED spread and MTL coverage metrics."""
    df = pd.read_csv(IED_CSV)
    dm = df[df['MemoryOutcome'].notna()].copy()

    rows = []
    for pat, pdf in dm.groupby('Patient'):
        # Spread metrics
        mean_region_spread = pdf['RegionSpread'].mean()
        max_region_spread = pdf['RegionSpread'].max()
        mean_channel_spread = pdf['ChannelSpread'].mean()
        max_channel_spread = pdf['ChannelSpread'].max()

        # MTL region coverage
        all_mtl = set()
        for reg in pdf['Region'].unique():
            all_mtl.update(_mtl_flags(reg))

        has_bla = 'BLA' in all_mtl
        has_hpc = 'HPC' in all_mtl
        has_prc = 'PRC' in all_mtl
        has_ec = 'EC' in all_mtl
        n_mtl_regions = sum([has_bla, has_hpc, has_prc])  # core 3
        n_mtl_regions_with_ec = n_mtl_regions + int(has_ec)

        # Total distinct regions
        n_distinct_regions = pdf['Region'].nunique()

        # Trial counts
        n_trials = len(pdf.groupby(['Trial']))

        rows.append({
            'Patient': pat,
            'mean_region_spread': round(mean_region_spread, 3),
            'max_region_spread': max_region_spread,
            'mean_channel_spread': round(mean_channel_spread, 3),
            'max_channel_spread': max_channel_spread,
            'has_BLA': has_bla,
            'has_HPC': has_hpc,
            'has_PRC': has_prc,
            'has_EC': has_ec,
            'n_mtl_regions': n_mtl_regions,
            'n_mtl_with_ec': n_mtl_regions_with_ec,
            'n_distinct_regions': n_distinct_regions,
            'n_ied_trials': n_trials,
        })

    return pd.DataFrame(rows)


def load_behavioral():
    """Load avg_stim_dprime_diff per patient."""
    beh = pd.read_csv(BEH_CSV)
    return beh[['Patient', 'avg_stim_dprime_diff']].copy()


def load_coherence_summary():
    """
    Compute per-patient mean encoding coherence for key MTL region pairs.
    Returns mean coherence across theta (4-8 Hz), slow gamma (30-55 Hz),
    and HFA (70-100 Hz) bands for each patient.
    """
    coh = pd.read_csv(COH_CSV)

    # Extract frequency columns
    freq_cols = [c for c in coh.columns if c.startswith('diff_Freq_')]
    freqs = np.array([float(c.replace('diff_Freq_', '')) for c in freq_cols])

    # Band definitions
    bands = {
        'theta': (4, 8),
        'slow_gamma': (30, 55),
        'hfa': (70, 100),
    }

    # Key MTL region pairs
    mtl_pairs = ['BLA_CA', 'BLA_DG', 'BLA_HPC', 'BLA_PRC',
                 'CA_HPC', 'CA_PRC', 'DG_HPC', 'DG_PRC',
                 'HPC_PRC', 'BLA_EC', 'CA_DG', 'EC_HPC', 'EC_PRC']

    coh_mtl = coh[coh['Region'].isin(mtl_pairs)].copy()

    rows = []
    for pat, pdf in coh_mtl.groupby('Patient'):
        row = {'Patient': pat}

        for band_name, (flo, fhi) in bands.items():
            mask = (freqs >= flo) & (freqs <= fhi)
            band_cols = [freq_cols[i] for i in range(len(freq_cols)) if mask[i]]
            if not band_cols:
                continue

            # Average absolute coherence difference across trials, regions, freqs
            vals = pdf[band_cols].values
            # Stim effect: compare stim vs nostim rows
            stim_rows = pdf[pdf['trial_type'] == 'stim'][band_cols].values
            nostim_rows = pdf[pdf['trial_type'] == 'nostim'][band_cols].values

            row[f'coh_{band_name}_stim_mean'] = np.nanmean(stim_rows) if len(stim_rows) > 0 else np.nan
            row[f'coh_{band_name}_nostim_mean'] = np.nanmean(nostim_rows) if len(nostim_rows) > 0 else np.nan
            row[f'coh_{band_name}_stim_effect'] = (
                np.nanmean(stim_rows) - np.nanmean(nostim_rows)
                if len(stim_rows) > 0 and len(nostim_rows) > 0 else np.nan)
            row[f'coh_{band_name}_abs_diff'] = np.nanmean(np.abs(vals)) if len(vals) > 0 else np.nan

        rows.append(row)

    return pd.DataFrame(rows)


def load_pac_summary():
    """
    Compute per-patient mean PAC stim effect for key MTL region pairs.
    Stim effect = stim_rem - nostim_rem (enhancement for remembered items).
    Also compute stim × memory interaction.
    """
    pac = pd.read_csv(PAC_CSV)

    # Key MTL pairs (including composite regions)
    mtl_keywords = ['BLA', 'HPC', 'CA', 'DG', 'PRC', 'EC', 'ALLHPC', 'MTL']
    def is_mtl_pair(region):
        parts = region.split('_')
        return len(parts) == 2 and all(p in mtl_keywords for p in parts)

    pac_mtl = pac[pac['Region'].apply(is_mtl_pair)].copy()

    rows = []
    for pat, pdf in pac_mtl.groupby('Patient'):
        row = {'Patient': pat}
        for band in ['Slow gamma', 'HFA']:
            bdf = pdf[pdf['Band'] == band]
            if len(bdf) == 0:
                continue
            band_key = band.lower().replace(' ', '_')

            # Stim effect on remembered
            stim_rem_effect = (bdf['stim_rem'] - bdf['nostim_rem']).mean()
            # Stim effect on forgotten
            stim_forg_effect = (bdf['stim_forg'] - bdf['nostim_forg']).mean()
            # Stim × memory interaction
            interaction = stim_rem_effect - stim_forg_effect

            row[f'pac_{band_key}_stim_rem'] = round(stim_rem_effect, 6)
            row[f'pac_{band_key}_stim_forg'] = round(stim_forg_effect, 6)
            row[f'pac_{band_key}_interaction'] = round(interaction, 6)

        rows.append(row)

    return pd.DataFrame(rows)


# ========================== MERGE ==========================

def build_merged_dataset():
    """Merge IED features, behavioral, coherence, and PAC data."""
    ied = load_patient_ied_features()
    beh = load_behavioral()
    coh = load_coherence_summary()
    pac = load_pac_summary()

    merged = ied.merge(beh, on='Patient', how='inner')
    merged['abs_dprime_diff'] = merged['avg_stim_dprime_diff'].abs()

    # Add coherence (inner or left join — some patients won't have it)
    if len(coh) > 0:
        merged = merged.merge(coh, on='Patient', how='left')
    if len(pac) > 0:
        merged = merged.merge(pac, on='Patient', how='left')

    return merged


# ========================== HELPER ==========================

def _corr_text(x, y, method='spearman'):
    """Return formatted correlation string."""
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    if len(x) < 5:
        return 'n < 5', np.nan, np.nan
    if method == 'spearman':
        r, p = spearmanr(x, y)
        return f'\u03c1 = {r:.3f}, p = {p:.4f} (n={len(x)})', r, p
    else:
        r, p = pearsonr(x, y)
        return f'r = {r:.3f}, p = {p:.4f} (n={len(x)})', r, p


def _pretty_band(band):
    """Convert internal band names to display labels."""
    mapping = {'slow_gamma': 'Slow Gamma', 'hfa': 'HFA', 'theta': 'Theta'}
    return mapping.get(band, band)


def _sig_marker(p):
    if np.isnan(p):
        return ''
    if p < .001:
        return ' ***'
    elif p < .01:
        return ' **'
    elif p < .05:
        return ' *'
    return ''


# ========================== PLOTS ==========================

def plot_spread_vs_dprime(merged):
    """Scatter: IED spread metrics vs memory modulation (dprime diff)."""
    spread_vars = [
        ('mean_region_spread', 'Mean Region Spread'),
        ('max_region_spread', 'Max Region Spread'),
        ('mean_channel_spread', 'Mean Channel Spread'),
        ('max_channel_spread', 'Max Channel Spread'),
        ('n_distinct_regions', 'N Distinct IED Regions'),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    axes = axes.flatten()

    stat_rows = []
    for i, (var, label) in enumerate(spread_vars):
        ax = axes[i]
        x = merged[var].values.astype(float)
        y = merged['avg_stim_dprime_diff'].values.astype(float)

        ax.scatter(x, y, c=[CB[0]], s=50, alpha=0.7, edgecolors='black', linewidth=0.5)

        # Fit line
        mask = ~(np.isnan(x) | np.isnan(y))
        if mask.sum() > 2:
            z = np.polyfit(x[mask], y[mask], 1)
            xline = np.linspace(np.nanmin(x), np.nanmax(x), 100)
            ax.plot(xline, np.polyval(z, xline), '--', color=CB[1], linewidth=2)

        txt, r, p = _corr_text(x, y)
        ax.set_title(f'{label}{_sig_marker(p)}', fontsize=13, fontweight='bold')
        ax.text(0.95, 0.95, txt, transform=ax.transAxes, fontsize=10,
                va='top', ha='right',
                bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='gray'))
        ax.set_xlabel(label, fontsize=11)
        ax.set_ylabel('Avg Stim d\' Diff', fontsize=11)
        sns.despine(ax=ax)

        stat_rows.append({'Predictor': label, 'rho': r, 'p_value': p,
                          'n': mask.sum()})

    # Also abs dprime
    ax = axes[5]
    x = merged['mean_region_spread'].values.astype(float)
    y = merged['abs_dprime_diff'].values.astype(float)
    ax.scatter(x, y, c=[CB[3]], s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() > 2:
        z = np.polyfit(x[mask], y[mask], 1)
        xline = np.linspace(np.nanmin(x), np.nanmax(x), 100)
        ax.plot(xline, np.polyval(z, xline), '--', color=CB[1], linewidth=2)
    txt, r, p = _corr_text(x, y)
    ax.set_title(f'Mean Region Spread vs |d\' Diff|{_sig_marker(p)}',
                 fontsize=13, fontweight='bold')
    ax.text(0.95, 0.95, txt, transform=ax.transAxes, fontsize=10,
            va='top', ha='right',
            bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='gray'))
    ax.set_xlabel('Mean Region Spread', fontsize=11)
    ax.set_ylabel('|Avg Stim d\' Diff|', fontsize=11)
    sns.despine(ax=ax)
    stat_rows.append({'Predictor': 'Mean Region Spread vs |dprime|',
                      'rho': r, 'p_value': p, 'n': mask.sum()})

    fig.suptitle('Hypothesis 3d: IED Spread vs Memory Modulation',
                 fontsize=16, fontweight='bold', y=1.02)
    fig.tight_layout()

    out = os.path.join(OUTPUT_DIR, 'ied_spread_vs_dprime.png')
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out}')

    stats_df = pd.DataFrame(stat_rows)
    stats_out = os.path.join(OUTPUT_DIR, 'ied_spread_vs_dprime_stats.csv')
    stats_df.to_csv(stats_out, index=False)
    print(f'Saved {stats_out}')
    return stats_df


def plot_mtl_coverage_vs_dprime(merged):
    """Bar/scatter: number of MTL regions with IEDs vs dprime diff."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))

    # Panel 1: N MTL regions (BLA, HPC, PRC) vs dprime diff
    ax = axes[0]
    rng = np.random.default_rng(42)
    stat_rows = []

    for n_mtl in sorted(merged['n_mtl_regions'].unique()):
        sub = merged[merged['n_mtl_regions'] == n_mtl]
        vals = sub['avg_stim_dprime_diff'].values
        color = CB[min(n_mtl, len(CB) - 1)]
        mean_val = np.mean(vals)
        sem_val = np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0

        ax.bar(n_mtl, mean_val, yerr=sem_val, color=color, edgecolor='black',
               linewidth=1, width=0.6, capsize=5, alpha=0.7, zorder=1)
        jitter = rng.uniform(-0.15, 0.15, len(vals))
        ax.scatter(np.full(len(vals), n_mtl) + jitter, vals,
                   color='black', alpha=0.4, s=30, zorder=2)

    x_vals = merged['n_mtl_regions'].values.astype(float)
    y_vals = merged['avg_stim_dprime_diff'].values.astype(float)
    txt, r, p = _corr_text(x_vals, y_vals)
    ax.set_xlabel('N MTL Regions with IEDs\n(BLA, HPC, PRC)', fontsize=12)
    ax.set_ylabel('Avg Stim d\' Diff', fontsize=12)
    ax.set_title(f'MTL Coverage vs Memory Modulation{_sig_marker(p)}',
                 fontsize=13, fontweight='bold')
    ax.text(0.95, 0.95, txt, transform=ax.transAxes, fontsize=10, va='top', ha='right',
            bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='gray'))
    sns.despine(ax=ax)
    stat_rows.append({'Analysis': 'n_mtl_regions vs dprime', 'rho': r, 'p': p})

    # Panel 2: same but absolute dprime
    ax = axes[1]
    for n_mtl in sorted(merged['n_mtl_regions'].unique()):
        sub = merged[merged['n_mtl_regions'] == n_mtl]
        vals = sub['abs_dprime_diff'].values
        color = CB[min(n_mtl, len(CB) - 1)]
        mean_val = np.mean(vals)
        sem_val = np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0

        ax.bar(n_mtl, mean_val, yerr=sem_val, color=color, edgecolor='black',
               linewidth=1, width=0.6, capsize=5, alpha=0.7, zorder=1)
        jitter = rng.uniform(-0.15, 0.15, len(vals))
        ax.scatter(np.full(len(vals), n_mtl) + jitter, vals,
                   color='black', alpha=0.4, s=30, zorder=2)

    y_abs = merged['abs_dprime_diff'].values.astype(float)
    txt, r, p = _corr_text(x_vals, y_abs)
    ax.set_xlabel('N MTL Regions with IEDs\n(BLA, HPC, PRC)', fontsize=12)
    ax.set_ylabel('|Avg Stim d\' Diff|', fontsize=12)
    ax.set_title(f'MTL Coverage vs |Memory Modulation|{_sig_marker(p)}',
                 fontsize=13, fontweight='bold')
    ax.text(0.95, 0.95, txt, transform=ax.transAxes, fontsize=10, va='top', ha='right',
            bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='gray'))
    sns.despine(ax=ax)
    stat_rows.append({'Analysis': 'n_mtl_regions vs |dprime|', 'rho': r, 'p': p})

    # Panel 3: individual region flags
    ax = axes[2]
    regions = ['has_BLA', 'has_HPC', 'has_PRC']
    region_labels = ['BLA', 'HPC', 'PRC']
    x_pos = np.arange(len(regions))
    bar_width = 0.35

    for offset, (has_val, label, color) in enumerate([
        (True, 'IED Present', CB[1]),
        (False, 'IED Absent', CB[0]),
    ]):
        means, sems, ns = [], [], []
        for reg in regions:
            sub = merged[merged[reg] == has_val]
            vals = sub['avg_stim_dprime_diff'].values
            means.append(np.mean(vals) if len(vals) > 0 else 0)
            sems.append(np.std(vals, ddof=1) / np.sqrt(len(vals))
                        if len(vals) > 1 else 0)
            ns.append(len(vals))

        off = -bar_width / 2 + offset * bar_width
        bars = ax.bar(x_pos + off, means, bar_width, yerr=sems,
                      color=color, edgecolor='black', linewidth=0.8,
                      capsize=4, alpha=0.8, label=label)
        for bar, m, n in zip(bars, means, ns):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f'n={n}', ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(region_labels, fontsize=13)
    ax.set_ylabel('Avg Stim d\' Diff', fontsize=12)
    ax.set_title('Region-Specific IED Effects', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, frameon=True)
    ax.axhline(0, color='gray', linewidth=0.8, linestyle='-')
    sns.despine(ax=ax)

    fig.suptitle('Hypothesis 3d: MTL IED Coverage vs Memory Modulation',
                 fontsize=16, fontweight='bold', y=1.02)
    fig.tight_layout()

    out = os.path.join(OUTPUT_DIR, 'mtl_coverage_vs_dprime.png')
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out}')

    # Region-specific Mann-Whitney tests
    for reg, rlabel in zip(regions, region_labels):
        present = merged[merged[reg] == True]['avg_stim_dprime_diff'].values
        absent = merged[merged[reg] == False]['avg_stim_dprime_diff'].values
        if len(present) >= 2 and len(absent) >= 2:
            u, p = mannwhitneyu(present, absent, alternative='two-sided')
            stat_rows.append({
                'Analysis': f'{rlabel} present vs absent (dprime)',
                'rho': u, 'p': p,
            })

    stats_df = pd.DataFrame(stat_rows)
    stats_out = os.path.join(OUTPUT_DIR, 'mtl_coverage_vs_dprime_stats.csv')
    stats_df.to_csv(stats_out, index=False)
    print(f'Saved {stats_out}')
    return stats_df


def plot_spread_vs_neural(merged):
    """Scatter: IED spread vs coherence/PAC metrics (neural dynamics disruption)."""
    # Identify available neural columns
    coh_cols = [c for c in merged.columns if c.startswith('coh_') and c.endswith('_stim_effect')]
    pac_cols = [c for c in merged.columns if c.startswith('pac_') and c.endswith('_interaction')]

    neural_vars = []
    for c in coh_cols:
        band = c.replace('coh_', '').replace('_stim_effect', '')
        neural_vars.append((c, f'Coherence {_pretty_band(band)}\n(stim effect)'))
    for c in pac_cols:
        band = c.replace('pac_', '').replace('_interaction', '')
        neural_vars.append((c, f'PAC {_pretty_band(band)}\n(stim\u00d7memory)'))

    if not neural_vars:
        print('  No neural dynamics variables available.')
        return

    spread_vars = [
        ('mean_region_spread', 'Mean Region Spread'),
        ('n_mtl_regions', 'N MTL Regions'),
    ]

    n_neural = len(neural_vars)
    n_spread = len(spread_vars)
    fig, axes = plt.subplots(n_spread, n_neural,
                             figsize=(5 * n_neural, 5 * n_spread),
                             squeeze=False)

    stat_rows = []
    for si, (svar, slabel) in enumerate(spread_vars):
        for ni, (nvar, nlabel) in enumerate(neural_vars):
            ax = axes[si, ni]
            x = merged[svar].values.astype(float)
            y = merged[nvar].values.astype(float)
            mask = ~(np.isnan(x) | np.isnan(y))

            if mask.sum() < 5:
                ax.text(0.5, 0.5, f'n={mask.sum()}\n(insufficient)',
                        transform=ax.transAxes, ha='center', va='center')
                ax.set_title(f'{slabel} vs {nlabel}', fontsize=10)
                sns.despine(ax=ax)
                continue

            ax.scatter(x[mask], y[mask], c=[CB[2]], s=50, alpha=0.7,
                       edgecolors='black', linewidth=0.5)
            z = np.polyfit(x[mask], y[mask], 1)
            xline = np.linspace(np.nanmin(x[mask]), np.nanmax(x[mask]), 100)
            ax.plot(xline, np.polyval(z, xline), '--', color=CB[3], linewidth=2)

            txt, r, p = _corr_text(x, y)
            ax.text(0.95, 0.95, txt, transform=ax.transAxes, fontsize=9,
                    va='top', ha='right',
                    bbox=dict(boxstyle='round', facecolor='lightyellow',
                              edgecolor='gray'))
            ax.set_title(f'{slabel} vs {nlabel}{_sig_marker(p)}',
                         fontsize=10, fontweight='bold')
            ax.set_xlabel(slabel, fontsize=9)
            ax.set_ylabel(nlabel, fontsize=9)
            sns.despine(ax=ax)

            stat_rows.append({
                'Predictor': slabel, 'Outcome': nlabel,
                'rho': r, 'p_value': p, 'n': mask.sum(),
            })

    fig.suptitle('IED Spread vs Neural Dynamics (Encoding Coherence & PAC)',
                 fontsize=15, fontweight='bold', y=1.02)
    fig.tight_layout()

    out = os.path.join(OUTPUT_DIR, 'ied_spread_vs_neural.png')
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out}')

    stats_df = pd.DataFrame(stat_rows)
    stats_out = os.path.join(OUTPUT_DIR, 'ied_spread_vs_neural_stats.csv')
    stats_df.to_csv(stats_out, index=False)
    print(f'Saved {stats_out}')
    return stats_df


def plot_neural_vs_dprime(merged):
    """Scatter: coherence/PAC stim effects vs memory modulation."""
    coh_cols = [c for c in merged.columns if c.startswith('coh_') and c.endswith('_stim_effect')]
    pac_cols = [c for c in merged.columns if c.startswith('pac_') and c.endswith('_interaction')]

    neural_vars = []
    for c in coh_cols:
        band = c.replace('coh_', '').replace('_stim_effect', '')
        neural_vars.append((c, f'Coherence {_pretty_band(band)} stim effect'))
    for c in pac_cols:
        band = c.replace('pac_', '').replace('_interaction', '')
        neural_vars.append((c, f'PAC {_pretty_band(band)} stim\u00d7memory'))

    if not neural_vars:
        print('  No neural dynamics variables for dprime correlation.')
        return

    ncols = min(3, len(neural_vars))
    nrows = (len(neural_vars) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows),
                             squeeze=False)
    axes_flat = axes.flatten()

    stat_rows = []
    for i, (nvar, nlabel) in enumerate(neural_vars):
        ax = axes_flat[i]
        x = merged[nvar].values.astype(float)
        y = merged['avg_stim_dprime_diff'].values.astype(float)
        mask = ~(np.isnan(x) | np.isnan(y))

        if mask.sum() < 5:
            ax.text(0.5, 0.5, f'n={mask.sum()}\n(insufficient)',
                    transform=ax.transAxes, ha='center', va='center')
            ax.set_title(nlabel, fontsize=10)
            sns.despine(ax=ax)
            continue

        ax.scatter(x[mask], y[mask], c=[CB[4]], s=50, alpha=0.7,
                   edgecolors='black', linewidth=0.5)
        z = np.polyfit(x[mask], y[mask], 1)
        xline = np.linspace(np.nanmin(x[mask]), np.nanmax(x[mask]), 100)
        ax.plot(xline, np.polyval(z, xline), '--', color=CB[1], linewidth=2)

        txt, r, p = _corr_text(x, y)
        ax.text(0.95, 0.95, txt, transform=ax.transAxes, fontsize=9, va='top',
                ha='right',
                bbox=dict(boxstyle='round', facecolor='lightyellow',
                          edgecolor='gray'))
        ax.set_title(f'{nlabel}{_sig_marker(p)}', fontsize=10, fontweight='bold')
        ax.set_xlabel(nlabel, fontsize=9)
        ax.set_ylabel('Avg Stim d\' Diff', fontsize=9)
        sns.despine(ax=ax)

        stat_rows.append({
            'Predictor': nlabel, 'Outcome': 'avg_stim_dprime_diff',
            'rho': r, 'p_value': p, 'n': mask.sum(),
        })

    # Hide unused axes
    for j in range(len(neural_vars), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle('Neural Dynamics vs Memory Modulation',
                 fontsize=15, fontweight='bold', y=1.02)
    fig.tight_layout()

    out = os.path.join(OUTPUT_DIR, 'neural_vs_dprime.png')
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out}')

    stats_df = pd.DataFrame(stat_rows)
    stats_out = os.path.join(OUTPUT_DIR, 'neural_vs_dprime_stats.csv')
    stats_df.to_csv(stats_out, index=False)
    print(f'Saved {stats_out}')
    return stats_df


def plot_full_path_summary(merged):
    """Summary heatmap of all pairwise correlations for hypothesis 3d path."""
    key_vars = [
        ('mean_region_spread', 'Mean Region\nSpread'),
        ('max_region_spread', 'Max Region\nSpread'),
        ('n_mtl_regions', 'N MTL\nRegions'),
        ('n_distinct_regions', 'N Distinct\nRegions'),
    ]

    # Add neural vars if present
    for c in merged.columns:
        if c.startswith('coh_') and c.endswith('_stim_effect'):
            band = c.replace('coh_', '').replace('_stim_effect', '')
            key_vars.append((c, f'Coh {_pretty_band(band)}\nStim Effect'))
        if c.startswith('pac_') and c.endswith('_interaction'):
            band = c.replace('pac_', '').replace('_interaction', '')
            key_vars.append((c, f'PAC {_pretty_band(band)}\nInteraction'))

    key_vars.append(('avg_stim_dprime_diff', 'Avg Stim\nd\' Diff'))
    key_vars.append(('abs_dprime_diff', '|d\' Diff|'))

    var_names = [v[0] for v in key_vars]
    var_labels = [v[1] for v in key_vars]

    n = len(var_names)
    corr_matrix = np.full((n, n), np.nan)
    p_matrix = np.full((n, n), np.nan)

    for i in range(n):
        for j in range(n):
            if i == j:
                corr_matrix[i, j] = 1.0
                p_matrix[i, j] = 0.0
                continue
            x = merged[var_names[i]].values.astype(float)
            y = merged[var_names[j]].values.astype(float)
            mask = ~(np.isnan(x) | np.isnan(y))
            if mask.sum() >= 5:
                r, p = spearmanr(x[mask], y[mask])
                corr_matrix[i, j] = r
                p_matrix[i, j] = p

    fig, ax = plt.subplots(figsize=(max(10, n * 1.2), max(8, n * 1.0)))

    cmap = sns.diverging_palette(250, 15, s=75, l=40, as_cmap=True)
    im = ax.imshow(corr_matrix, cmap=cmap, vmin=-1, vmax=1, aspect='auto')

    for i in range(n):
        for j in range(n):
            if not np.isnan(corr_matrix[i, j]) and i != j:
                sig = _sig_marker(p_matrix[i, j])
                ax.text(j, i, f'{corr_matrix[i,j]:.2f}{sig}',
                        ha='center', va='center', fontsize=9,
                        fontweight='bold')

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(var_labels, fontsize=11, fontweight='bold', rotation=45, ha='right')
    ax.set_yticklabels(var_labels, fontsize=11, fontweight='bold')

    plt.colorbar(im, ax=ax, label='Spearman \u03c1', shrink=0.8)
    ax.set_title('Hypothesis 3d: Full Correlation Matrix\n'
                 'IED Spread \u2192 Neural Dynamics \u2192 Memory Modulation',
                 fontsize=14, fontweight='bold')
    fig.tight_layout()

    out = os.path.join(OUTPUT_DIR, 'hypothesis3d_correlation_matrix.png')
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out}')


# ========================== MAIN ==========================

def main():
    print('Building merged dataset...')
    merged = build_merged_dataset()
    print(f'  Merged: {len(merged)} patients')
    print(f'  With coherence data: {merged.filter(like="coh_").notna().any(axis=1).sum()}')
    print(f'  With PAC data: {merged.filter(like="pac_").notna().any(axis=1).sum()}')
    print()

    # Save merged dataset
    merged_out = os.path.join(OUTPUT_DIR, 'merged_ied_neural_memory.csv')
    merged.to_csv(merged_out, index=False)
    print(f'Saved {merged_out}')
    print()

    # Print summary
    print('Patient-level IED features:')
    print(merged[['Patient', 'mean_region_spread', 'max_region_spread',
                   'n_mtl_regions', 'has_BLA', 'has_HPC', 'has_PRC',
                   'avg_stim_dprime_diff']].to_string(index=False))
    print()

    # Plots
    print('\n--- IED Spread vs Memory Modulation ---')
    spread_stats = plot_spread_vs_dprime(merged)
    print(spread_stats.to_string(index=False))
    print()

    print('\n--- MTL Coverage vs Memory Modulation ---')
    mtl_stats = plot_mtl_coverage_vs_dprime(merged)
    print(mtl_stats.to_string(index=False))
    print()

    print('\n--- IED Spread vs Neural Dynamics ---')
    plot_spread_vs_neural(merged)
    print()

    print('\n--- Neural Dynamics vs Memory Modulation ---')
    plot_neural_vs_dprime(merged)
    print()

    print('\n--- Full Path Correlation Matrix ---')
    plot_full_path_summary(merged)
    print()

    print('Done.')


if __name__ == '__main__':
    main()
