#!/usr/bin/env python
"""
Prepare data for Hypothesis 3d — IED spread, neural dynamics, and memory.

Outputs (all in outputs/ied_timing_memory/):
  h3d_spread_trials.csv         – trial-level spread metrics + memory + stim
  h3d_spread_descriptives.csv   – per-patient spread summary
  h3d_coherence_trials.csv      – trial-level band-averaged coherence for BLA/HPC
  h3d_pac_patient.csv           – patient-level PAC metrics for BLA/HPC
  h3d_eligible_subjects.csv     – subjects with spread + neural data
  Figures: h3d_spread_distribution.png, h3d_spread_by_memory.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'outputs', 'ied_timing_memory')
os.makedirs(OUTPUT_DIR, exist_ok=True)

ENCODING_CSV = os.path.join(SCRIPT_DIR, 'IED',
    'AMMEBLAES_IEDs_trial_level_dissertation_study_usethis_cleaned_with_memory.csv')
COHERENCE_CSV = os.path.join(SCRIPT_DIR, 'outputs', 'csvs',
    'combined_encoding_coherence_amme_mlmr_input.csv')
PAC_CSV = os.path.join(SCRIPT_DIR, 'outputs',
    'stim_effect_composite_scores', 'encoding_pac', 'amme',
    'stim_composite_pac_encoding.csv')

# Frequency bands (Hz)
THETA = (4, 8)
SLOW_GAMMA = (30, 55)
HFA = (70, 100)

# Three BLA coherence pairs of interest
BLA_TARGET_PAIRS = ['BLA_HPC', 'BLA_CA', 'BLA_DG']

# Three BLA PAC pairs of interest
PAC_TARGET_PAIRS = ['BLA_HPC', 'BLA_CA', 'BLA_DG']


def get_freq_cols(df):
    """Get frequency column names and their numeric values."""
    freq_cols = [c for c in df.columns if c.startswith('diff_Freq_')]
    freqs = np.array([float(c.replace('diff_Freq_', '')) for c in freq_cols])
    return freq_cols, freqs


def band_average(row, freq_cols, freqs, lo, hi):
    """Average frequency values within a band."""
    mask = (freqs >= lo) & (freqs <= hi)
    vals = row[np.array(freq_cols)[mask]].values.astype(float)
    return np.nanmean(vals)


# ═══════════════════════════════════════════════════════════════════════════
# PART 1: Trial-level IED spread metrics
# ═══════════════════════════════════════════════════════════════════════════

def prep_spread_trials():
    """Compute trial-level spread metrics from encoding IED data."""
    df = pd.read_csv(ENCODING_CSV)
    dm = df[df['MemoryOutcome'].notna()].copy()
    dm = dm[dm['StimCond'].isin(['S', 'NS'])]

    trials = []
    for (pat, trial), grp in dm.groupby(['Patient', 'Trial']):
        stim_str = grp['StimCond'].iloc[0]
        mem_str = grp['MemoryOutcome'].iloc[0]

        # Lead count = number of rows (distinct channel-region detections)
        n_leads = len(grp)

        # Channel count = total unique electrode contacts across all ChannelNames
        all_contacts = set()
        for cn in grp['ChannelName']:
            contacts = str(cn).split('_')
            all_contacts.update(contacts)
        n_channels = len(all_contacts)

        # Region count
        regions = set(grp['Region'].dropna())
        n_regions = len(regions)
        region_spread = grp['RegionSpread'].iloc[0]  # pre-computed

        # Within-region spread: max channels in any single region
        within_region = grp.groupby('Region').apply(
            lambda r: sum(len(str(cn).split('_')) for cn in r['ChannelName'])
        )
        max_within_region = within_region.max() if len(within_region) > 0 else 0
        mean_within_region = within_region.mean() if len(within_region) > 0 else 0

        # Across-region spread (= n_regions, binary: >1 region = cross-region)
        cross_region = 1 if n_regions > 1 else 0

        # MTL region involvement
        has_bla = int(any('Amygdala' in r for r in regions))
        has_hpc = int(any(r in ['Hippocampus', 'CA', 'DG'] or 'Hippocampus' in r
                         for r in regions))
        has_prc = int(any('Parahippocampal' in r for r in regions))
        has_ec = int(any('Entorhinal' in r for r in regions))

        # Timing windows
        ied_before = int((grp['BeforeImgITI'] == 'Y').any())
        ied_during = int((grp['DuringImg'] == 'Y').any())
        ied_after = int((grp['AfterImgITI'] == 'Y').any())
        ied_stim = int((grp['DuringStim'] == 'Y').any())

        trials.append({
            'patient_id': pat,
            'trial': trial,
            'memory': 1 if mem_str == 'remembered' else 0,
            'stim': 1 if stim_str == 'S' else 0,
            'n_leads': n_leads,
            'n_channels': n_channels,
            'n_regions': n_regions,
            'region_spread': region_spread,
            'max_within_region': max_within_region,
            'mean_within_region': round(mean_within_region, 2),
            'cross_region': cross_region,
            'has_bla': has_bla,
            'has_hpc': has_hpc,
            'has_prc': has_prc,
            'has_ec': has_ec,
            'ied_before': ied_before,
            'ied_during': ied_during,
            'ied_after': ied_after,
            'ied_stim': ied_stim,
        })

    trial_df = pd.DataFrame(trials)
    out = os.path.join(OUTPUT_DIR, 'h3d_spread_trials.csv')
    trial_df.to_csv(out, index=False)
    print(f"Saved spread trials: {len(trial_df)} trials, "
          f"{trial_df['patient_id'].nunique()} patients -> {out}")
    return trial_df


def make_spread_descriptives(trial_df):
    """Per-patient spread summary table."""
    desc = trial_df.groupby('patient_id').agg(
        n_trials=('trial', 'count'),
        n_remembered=('memory', 'sum'),
        pct_remembered=('memory', 'mean'),
        mean_leads=('n_leads', 'mean'),
        max_leads=('n_leads', 'max'),
        mean_channels=('n_channels', 'mean'),
        max_channels=('n_channels', 'max'),
        mean_regions=('n_regions', 'mean'),
        max_regions=('n_regions', 'max'),
        pct_cross_region=('cross_region', 'mean'),
        pct_bla=('has_bla', 'mean'),
        pct_hpc=('has_hpc', 'mean'),
        max_within_region=('max_within_region', 'max'),
    ).reset_index()
    desc['pct_remembered'] = (desc['pct_remembered'] * 100).round(1)
    desc['pct_cross_region'] = (desc['pct_cross_region'] * 100).round(1)
    desc['pct_bla'] = (desc['pct_bla'] * 100).round(1)
    desc['pct_hpc'] = (desc['pct_hpc'] * 100).round(1)
    desc['mean_leads'] = desc['mean_leads'].round(2)
    desc['mean_channels'] = desc['mean_channels'].round(2)
    desc['mean_regions'] = desc['mean_regions'].round(2)

    out = os.path.join(OUTPUT_DIR, 'h3d_spread_descriptives.csv')
    desc.to_csv(out, index=False)
    print(f"Saved spread descriptives: {len(desc)} patients -> {out}")
    return desc


def plot_spread_descriptives(trial_df):
    """Figures: spread distributions and spread × memory."""

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # 1. Distribution of leads per trial
    axes[0].hist(trial_df['n_leads'], bins=range(1, trial_df['n_leads'].max() + 2),
                 edgecolor='black', alpha=0.7, color='steelblue')
    axes[0].set_xlabel('Leads per Trial')
    axes[0].set_ylabel('Number of Trials')
    axes[0].set_title('IED Lead Spread Distribution')
    axes[0].axvline(trial_df['n_leads'].mean(), color='red', linestyle='--',
                    label=f"Mean={trial_df['n_leads'].mean():.1f}")
    axes[0].legend()

    # 2. Distribution of channels per trial
    axes[1].hist(trial_df['n_channels'], bins=range(1, trial_df['n_channels'].max() + 2),
                 edgecolor='black', alpha=0.7, color='darkorange')
    axes[1].set_xlabel('Channels (Contacts) per Trial')
    axes[1].set_ylabel('Number of Trials')
    axes[1].set_title('IED Channel Spread Distribution')
    axes[1].axvline(trial_df['n_channels'].mean(), color='red', linestyle='--',
                    label=f"Mean={trial_df['n_channels'].mean():.1f}")
    axes[1].legend()

    # 3. Distribution of regions per trial
    axes[2].hist(trial_df['n_regions'], bins=range(1, trial_df['n_regions'].max() + 2),
                 edgecolor='black', alpha=0.7, color='seagreen')
    axes[2].set_xlabel('Regions per Trial')
    axes[2].set_ylabel('Number of Trials')
    axes[2].set_title('IED Region Spread Distribution')
    axes[2].axvline(trial_df['n_regions'].mean(), color='red', linestyle='--',
                    label=f"Mean={trial_df['n_regions'].mean():.1f}")
    axes[2].legend()

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'h3d_spread_distribution.png')
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Saved -> {out}")

    # Spread × Memory
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    for ax, col, label, color in [
        (axes[0], 'n_leads', 'Leads per Trial', 'steelblue'),
        (axes[1], 'n_channels', 'Channels per Trial', 'darkorange'),
        (axes[2], 'n_regions', 'Regions per Trial', 'seagreen'),
    ]:
        rem = trial_df[trial_df['memory'] == 1][col]
        forg = trial_df[trial_df['memory'] == 0][col]
        ax.boxplot([rem, forg], labels=['Remembered', 'Forgotten'],
                   patch_artist=True,
                   boxprops=dict(facecolor=color, alpha=0.5))
        ax.set_ylabel(label)
        ax.set_title(f'{label} by Memory')
        # Add means
        ax.scatter([1, 2], [rem.mean(), forg.mean()], color='red',
                   zorder=5, s=50, marker='D', label='Mean')
        ax.legend()

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'h3d_spread_by_memory.png')
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Saved -> {out}")

    # Cross-region vs within-region by memory
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # Cross-region: memory rate for cross-region vs single-region IEDs
    cross = trial_df[trial_df['cross_region'] == 1]
    single = trial_df[trial_df['cross_region'] == 0]
    rates = [cross['memory'].mean() * 100, single['memory'].mean() * 100]
    counts = [len(cross), len(single)]
    bars = axes[0].bar(['Cross-Region\nIED', 'Single-Region\nIED'], rates,
                       color=['coral', 'lightblue'], edgecolor='black')
    for bar, r, n in zip(bars, rates, counts):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                     f'{r:.1f}%\n(n={n})', ha='center', va='bottom', fontsize=10)
    axes[0].set_ylabel('% Remembered')
    axes[0].set_title('Memory by Cross-Region IED Spread')
    axes[0].set_ylim(0, max(rates) + 15)

    # BLA/HPC involvement
    bla = trial_df[trial_df['has_bla'] == 1]
    hpc = trial_df[trial_df['has_hpc'] == 1]
    neither = trial_df[(trial_df['has_bla'] == 0) & (trial_df['has_hpc'] == 0)]
    cats = ['BLA IED', 'HPC IED', 'Neither']
    mem_rates = [bla['memory'].mean()*100, hpc['memory'].mean()*100,
                 neither['memory'].mean()*100]
    ns = [len(bla), len(hpc), len(neither)]
    bars = axes[1].bar(cats, mem_rates,
                       color=['salmon', 'mediumpurple', 'lightgray'],
                       edgecolor='black')
    for bar, r, n in zip(bars, mem_rates, ns):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                     f'{r:.1f}%\n(n={n})', ha='center', va='bottom', fontsize=10)
    axes[1].set_ylabel('% Remembered')
    axes[1].set_title('Memory by MTL Region Involvement')
    axes[1].set_ylim(0, max(mem_rates) + 15)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'h3d_spread_region_memory.png')
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Saved -> {out}")


# ═══════════════════════════════════════════════════════════════════════════
# PART 2: Trial-level coherence band averages for BLA/HPC
# ═══════════════════════════════════════════════════════════════════════════

def prep_coherence_trials():
    """Band-average coherence per trial for BLA_HPC, BLA_CA, BLA_DG pairs."""
    coh = pd.read_csv(COHERENCE_CSV)
    freq_cols, freqs = get_freq_cols(coh)

    # Filter to the 3 BLA target pairs
    coh_filt = coh[coh['Region'].isin(BLA_TARGET_PAIRS)].copy()

    # Band-average
    for band_name, (lo, hi) in [('theta', THETA), ('slow_gamma', SLOW_GAMMA), ('hfa', HFA)]:
        mask = (freqs >= lo) & (freqs <= hi)
        band_cols = np.array(freq_cols)[mask]
        coh_filt[f'coh_{band_name}'] = coh_filt[band_cols].astype(float).mean(axis=1)

    coh_filt['memory'] = (coh_filt['yes_or_no'] == 'yes').astype(int)
    coh_filt['stim'] = (coh_filt['trial_type'] == 'stim').astype(int)
    coh_filt = coh_filt.rename(columns={'Patient': 'patient_id', 'Region': 'region_pair'})
    coh_filt['trial_idx'] = coh_filt.groupby(
        ['patient_id', 'region_pair', 'stim', 'memory']).cumcount()

    keep_cols = ['patient_id', 'region_pair', 'stim', 'memory',
                 'trial_idx', 'coh_theta', 'coh_slow_gamma', 'coh_hfa']
    out = os.path.join(OUTPUT_DIR, 'h3d_coherence_bla_trials.csv')
    coh_filt[keep_cols].to_csv(out, index=False)

    for pair in BLA_TARGET_PAIRS:
        sub = coh_filt[coh_filt['region_pair'] == pair]
        print(f"  {pair}: {len(sub)} trials, {sub['patient_id'].nunique()} patients")

    print(f"Saved BLA coherence: {len(coh_filt)} total rows -> {out}")
    return coh_filt


# ═══════════════════════════════════════════════════════════════════════════
# PART 3: Patient-level PAC metrics for BLA/HPC
# ═══════════════════════════════════════════════════════════════════════════

def prep_pac_patient():
    """Compute patient-level PAC stim effects per BLA pair (BLA_HPC, BLA_CA, BLA_DG)."""
    pac = pd.read_csv(PAC_CSV)
    pac_filt = pac[pac['Region'].isin(PAC_TARGET_PAIRS)].copy()
    print(f"\nPAC BLA pairs: {len(pac_filt)} rows, "
          f"{pac_filt['Patient'].nunique()} patients")

    pac_filt['stim_effect_rem'] = pac_filt['stim_rem'] - pac_filt['nostim_rem']
    pac_filt['stim_effect_forg'] = pac_filt['stim_forg'] - pac_filt['nostim_forg']
    pac_filt['stim_x_memory'] = pac_filt['stim_effect_rem'] - pac_filt['stim_effect_forg']

    # Save per-pair, per-patient, per-band for R to use directly
    rows = []
    for (pat, pair, band), grp in pac_filt.groupby(['Patient', 'Region', 'Band']):
        rows.append({
            'patient_id': pat,
            'region_pair': pair,
            'band': band,
            'pac_stim_rem': grp['stim_effect_rem'].mean(),
            'pac_stim_forg': grp['stim_effect_forg'].mean(),
            'pac_interaction': grp['stim_x_memory'].mean(),
        })

    pat_df = pd.DataFrame(rows)
    out = os.path.join(OUTPUT_DIR, 'h3d_pac_bla_patient.csv')
    pat_df.to_csv(out, index=False)

    for pair in PAC_TARGET_PAIRS:
        sub = pat_df[pat_df['region_pair'] == pair]
        print(f"  {pair}: {sub['patient_id'].nunique()} patients")

    print(f"Saved PAC BLA patient metrics: {len(pat_df)} rows -> {out}")
    return pat_df


# ═══════════════════════════════════════════════════════════════════════════
# PART 4: Identify eligible subjects
# ═══════════════════════════════════════════════════════════════════════════

def identify_eligible(trial_df, coh_df, pac_df):
    """Find subjects with IED spread + coherence/PAC data."""
    spread_pats = set(trial_df[trial_df['n_regions'] > 1]['patient_id'].unique())
    spread_pats |= set(trial_df[trial_df['n_channels'] > 1]['patient_id'].unique())

    coh_pats = set(coh_df['patient_id'].unique()) if coh_df is not None and len(coh_df) > 0 else set()
    pac_pats = set(pac_df['patient_id'].unique()) if pac_df is not None and len(pac_df) > 0 else set()

    eligible_coh = sorted(spread_pats & coh_pats)
    eligible_pac = sorted(spread_pats & pac_pats)
    eligible_any = sorted(spread_pats & (coh_pats | pac_pats))

    print(f"\nPatients with IED spread: {len(spread_pats)}")
    print(f"Eligible (spread + coherence): {len(eligible_coh)} — {eligible_coh}")
    print(f"Eligible (spread + PAC): {len(eligible_pac)} — {eligible_pac}")

    elig_df = pd.DataFrame({
        'patient_id': sorted(spread_pats),
        'has_coherence': [p in coh_pats for p in sorted(spread_pats)],
        'has_pac': [p in pac_pats for p in sorted(spread_pats)],
    })
    out = os.path.join(OUTPUT_DIR, 'h3d_eligible_subjects.csv')
    elig_df.to_csv(out, index=False)
    print(f"Saved eligible subjects -> {out}")

    return eligible_coh, eligible_pac


def main():
    print("=" * 70)
    print("HYPOTHESIS 3d: IED SPREAD, NEURAL DYNAMICS, AND MEMORY")
    print("=" * 70)

    # Part 1: Spread metrics
    print("\n--- Part 1: Trial-level IED spread ---")
    trial_df = prep_spread_trials()
    desc = make_spread_descriptives(trial_df)
    plot_spread_descriptives(trial_df)

    # Part 2: Coherence
    print("\n--- Part 2: Coherence band averages ---")
    try:
        coh_df = prep_coherence_trials()
    except Exception as e:
        print(f"Coherence prep failed: {e}")
        coh_df = None

    # Part 3: PAC
    print("\n--- Part 3: PAC patient metrics ---")
    try:
        pac_df = prep_pac_patient()
    except Exception as e:
        print(f"PAC prep failed: {e}")
        pac_df = None

    # Part 4: Eligible subjects
    print("\n--- Part 4: Eligible subjects ---")
    identify_eligible(trial_df, coh_df, pac_df)

    print("\n===== PREP DONE =====")


if __name__ == '__main__':
    main()
