#!/usr/bin/env python
"""
Combined Endogenous Memory x Stimulation Figures
=================================================
For each measure (Power, Coherence, PAC), phase (Encoding, Retrieval),
and each region/pair, generates a side-by-side figure:
  Left panel  = Remembered Trials:  endogenous (purple) + stim (red)
  Right panel = Forgotten Trials:   endogenous (gold)   + stim (red)

All baseline-corrected, All (BLAES + AMME combined).
Output organized into:
  encoding/         {power, coherence, pac}
  retrieval/        {power, coherence, pac}
  balanced_encoding/  {power, coherence, pac}
  balanced_retrieval/ {power, coherence, pac}
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from textwrap import fill

warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

# Colors matching existing endogenous memory palette
COLOR_REM = '#7B2D8E'   # purple
COLOR_FORG = '#DAA520'  # gold

OUTPUT_ROOT = os.path.join(SCRIPT_DIR, 'outputs', 'endogenous_vs_stim')


# =========================================================================
#  UTILITY FUNCTIONS
# =========================================================================

def filter_pnas_regions(data):
    filtered = {}
    for key, value in data.items():
        if isinstance(value, dict) and value:
            first_val = next(iter(value.values()), None)
            if isinstance(first_val, dict):
                filtered[key] = {
                    roi: sd for roi, sd in value.items()
                    if 'PNAS' not in str(roi)
                }
            else:
                filtered[key] = value
        else:
            filtered[key] = value
    return filtered


def merge_dicts(*dicts):
    merged = {}
    for d in dicts:
        if not d:
            continue
        for roi, sd in d.items():
            merged.setdefault(roi, {}).update(sd)
    return merged


def merge_freqs(a, b):
    if a is not None and b is not None:
        return a if np.array_equal(a, b) else a
    return a if a is not None else b


def filter_subject_dicts(data, keep_subjects):
    """Filter any {region: {subject: values}} mappings to only keep_subjects."""
    if keep_subjects is None:
        return data
    filtered = {}
    for key, value in data.items():
        if isinstance(value, dict) and value:
            first_val = next(iter(value.values()), None)
            if isinstance(first_val, dict):
                filtered[key] = {
                    roi: {subj: vec for subj, vec in sd.items() if subj in keep_subjects}
                    for roi, sd in value.items()
                }
                filtered[key] = {k: v for k, v in filtered[key].items() if v}
            else:
                filtered[key] = value
        else:
            filtered[key] = value
    return filtered


def merge_coherence_data(blaes, amme):
    """Merge BLAES and AMME coherence data dicts."""
    return filter_pnas_regions({
        'freqs_post': merge_freqs(blaes.get('freqs_post'), amme.get('freqs_post')),
        'freqs_diff': merge_freqs(blaes.get('freqs_diff'), amme.get('freqs_diff')),
        'bc_nostim_rem': merge_dicts(blaes.get('bc_nostim_rem', {}), amme.get('bc_nostim_rem', {})),
        'bc_nostim_forg': merge_dicts(blaes.get('bc_nostim_forg', {}), amme.get('bc_nostim_forg', {})),
        'bc_stim_rem': merge_dicts(blaes.get('bc_stim_rem', {}), amme.get('bc_stim_rem', {})),
        'bc_stim_forg': merge_dicts(blaes.get('bc_stim_forg', {}), amme.get('bc_stim_forg', {})),
        'mlmr_export_frames': blaes.get('mlmr_export_frames', []) + amme.get('mlmr_export_frames', []),
    })


# =========================================================================
#  SEX METADATA
# =========================================================================

BEHAVIORAL_CSV = os.path.join(
    SCRIPT_DIR, 'behavioral figures',
    'AMMEBLAES_includedpts_firstsession_behavioral.csv',
)


def load_sex_map():
    """Return {patient_id: 'male'|'female'} from the behavioral CSV."""
    df = pd.read_csv(BEHAVIORAL_CSV)
    df['sex'] = df['sex'].astype(str).str.strip().str.lower()
    return dict(zip(df['Patient'], df['sex']))


def split_by_sex(subject_dict, sex_map):
    """Split a {subject: vector} dict into male and female subsets."""
    male = {s: v for s, v in subject_dict.items() if sex_map.get(s) == 'male'}
    female = {s: v for s, v in subject_dict.items() if sex_map.get(s) == 'female'}
    return male, female


# =========================================================================
#  SEX-STRATIFIED QUADRANT PLOTTING
# =========================================================================

def plot_sex_quadrant_for_roi(roi, freqs, nostim_rem, stim_rem,
                               nostim_forg, stim_forg, sex_map,
                               out_dir, measure_label, ylabel,
                               title_prefix='All', phase_label='Retrieval'):
    """2x2 quadrant: rows = Men/Women, cols = Remembered/Forgotten."""
    nr_m, nr_f = split_by_sex(nostim_rem, sex_map)
    sr_m, sr_f = split_by_sex(stim_rem, sex_map)
    nf_m, nf_f = split_by_sex(nostim_forg, sex_map)
    sf_m, sf_f = split_by_sex(stim_forg, sex_map)

    # Need at least some data in at least one sex
    if not any([nr_m, nr_f, nf_m, nf_f]):
        return

    fig, axes = plt.subplots(2, 2, figsize=(18, 13), sharey=True, sharex=True)

    # (row, col): (sex_label, nostim_data, stim_data, endo_color, mem_label)
    panels = [
        (0, 0, 'Men',   nr_m, sr_m, COLOR_REM,  'Remembered'),
        (0, 1, 'Men',   nf_m, sf_m, COLOR_FORG, 'Forgotten'),
        (1, 0, 'Women', nr_f, sr_f, COLOR_REM,  'Remembered'),
        (1, 1, 'Women', nf_f, sf_f, COLOR_FORG, 'Forgotten'),
    ]

    for r, c, sex_label, nostim_data, stim_data, endo_color, mem_label in panels:
        ax = axes[r, c]

        if nostim_data:
            mat = np.array(list(nostim_data.values()), dtype=np.float64)
            n = mat.shape[0]
            mean = mat.mean(0)
            se = mat.std(0) / np.sqrt(n)
            ax.plot(freqs, mean, color=endo_color, linewidth=2.5,
                    label=f'Endogenous / NoStim ({n})')
            ax.fill_between(freqs, mean - se, mean + se, alpha=0.2, color=endo_color)

        if stim_data:
            mat = np.array(list(stim_data.values()), dtype=np.float64)
            n = mat.shape[0]
            mean = mat.mean(0)
            se = mat.std(0) / np.sqrt(n)
            ax.plot(freqs, mean, color='red', linewidth=2.5,
                    label=f'Stim ({n})')
            ax.fill_between(freqs, mean - se, mean + se, alpha=0.15, color='red')

        ax.axhline(0, color='lightgray', linestyle='--', linewidth=1)
        ax.set_title(f'{sex_label} — {mem_label}', fontsize=16, fontweight='bold')
        ax.tick_params(axis='both', labelsize=13)
        ax.legend(fontsize=11, loc='best')

        fmin, fmax = freqs.min(), freqs.max()
        if fmin >= 25:
            ax.set_xlim(fmin, fmax)
        else:
            ax.set_xlim(max(fmin - 2, 0), fmax + 2)

    axes[1, 0].set_xlabel('Frequency (Hz)', fontsize=16, fontweight='bold')
    axes[1, 1].set_xlabel('Frequency (Hz)', fontsize=16, fontweight='bold')
    axes[0, 0].set_ylabel(ylabel, fontsize=15, fontweight='bold')
    axes[1, 0].set_ylabel(ylabel, fontsize=15, fontweight='bold')

    fig.suptitle(
        f'{title_prefix} {phase_label} {roi}: Endogenous Memory x Stimulation by Sex ({measure_label})',
        fontsize=17, fontweight='bold',
    )

    fig.text(0.5, 0.01,
             fill(f'Method: Each panel shows one sex x memory outcome for {roi}. '
                  f'Colored line = endogenous (NoStim); red = stimulation. '
                  f'Lines = patient means, shading = SEM.', width=140),
             ha='center', va='bottom', fontsize=9, color='dimgray')

    plt.tight_layout(rect=[0, 0.04, 1, 0.95])
    safe_roi = roi.replace('/', '_').replace(' ', '_')
    phase_tag = phase_label.lower()
    fpath = os.path.join(out_dir,
                         f'{safe_roi}_endogenous_memory_x_stim_by_sex_{phase_tag}.png')
    plt.savefig(fpath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return fpath


def generate_all_rois_by_sex(data, bc_nostim_rem_key, bc_nostim_forg_key,
                              bc_stim_rem_key, bc_stim_forg_key,
                              sex_map, out_dir, measure_label, ylabel,
                              title_prefix='All', phase_label='Retrieval'):
    """Generate sex-stratified quadrant figures for all ROIs/pairs."""
    freqs = data['freqs_diff']
    if freqs is None:
        print(f"  WARNING: No freqs_diff for {measure_label}, skipping.")
        return

    nostim_rem_all = data.get(bc_nostim_rem_key, {})
    nostim_forg_all = data.get(bc_nostim_forg_key, {})
    stim_rem_all = data.get(bc_stim_rem_key, {})
    stim_forg_all = data.get(bc_stim_forg_key, {})

    all_rois = sorted(set(nostim_rem_all.keys()) | set(nostim_forg_all.keys())
                       | set(stim_rem_all.keys()) | set(stim_forg_all.keys()))

    os.makedirs(out_dir, exist_ok=True)
    count = 0
    for roi in all_rois:
        if 'PNAS' in roi:
            continue
        fpath = plot_sex_quadrant_for_roi(
            roi, freqs,
            nostim_rem_all.get(roi, {}),
            stim_rem_all.get(roi, {}),
            nostim_forg_all.get(roi, {}),
            stim_forg_all.get(roi, {}),
            sex_map, out_dir, measure_label, ylabel,
            title_prefix=title_prefix,
            phase_label=phase_label,
        )
        if fpath:
            count += 1
            print(f"    Saved: {os.path.basename(fpath)}")

    print(f"  {measure_label}: {count} sex-stratified figures.")


def run_sex_phase(phase_label, power_data, load_coherence_fn, load_pac_fn,
                  sex_map, out_base, title_prefix='All'):
    """Run sex-stratified quadrant plots for all 3 measures in one phase."""

    print(f"\n--- {phase_label.upper()} POWER (by sex) ---")
    generate_all_rois_by_sex(
        power_data,
        'bc_nostim_rem', 'bc_nostim_forg', 'bc_stim_rem', 'bc_stim_forg',
        sex_map, os.path.join(out_base, 'power'),
        'Power', 'Baseline-Corrected Power (dB)',
        title_prefix=title_prefix, phase_label=phase_label,
    )

    print(f"\n--- {phase_label.upper()} COHERENCE (by sex) ---")
    coh_data = load_coherence_fn()
    generate_all_rois_by_sex(
        coh_data,
        'bc_nostim_rem', 'bc_nostim_forg', 'bc_stim_rem', 'bc_stim_forg',
        sex_map, os.path.join(out_base, 'coherence'),
        'Coherence', 'Baseline-Corrected Coherency (dB)',
        title_prefix=title_prefix, phase_label=phase_label,
    )

    print(f"\n--- {phase_label.upper()} PAC (by sex) ---")
    pac_data = load_pac_fn()
    generate_all_rois_by_sex(
        pac_data,
        'diff_nostim_rem', 'diff_nostim_forg', 'diff_stim_rem', 'diff_stim_forg',
        sex_map, os.path.join(out_base, 'pac'),
        'PAC', 'Baseline-Corrected PAC',
        title_prefix=title_prefix, phase_label=phase_label,
    )


def run_balanced_sex_phase(phase_label, power_data, load_coherence_fn, load_pac_fn,
                           sex_map, out_base):
    """Run balanced + sex-stratified quadrant plots for one phase."""
    from balanced_memory_trials import (
        get_nostim_trial_counts,
        get_balanced_subjects,
    )

    print(f"\n  Determining balanced subjects ({phase_label.lower()})...")
    trial_counts = get_nostim_trial_counts(power_data, measure='power')
    balanced_subjects = get_balanced_subjects(trial_counts)
    if balanced_subjects is None:
        print(f"  ERROR: No balanced subjects for {phase_label}. Skipping.")
        return

    bal_power = filter_subject_dicts(power_data, balanced_subjects)

    print(f"\n--- BALANCED {phase_label.upper()} POWER (by sex) ---")
    generate_all_rois_by_sex(
        bal_power,
        'bc_nostim_rem', 'bc_nostim_forg', 'bc_stim_rem', 'bc_stim_forg',
        sex_map, os.path.join(out_base, 'power'),
        'Power', 'Baseline-Corrected Power (dB)',
        title_prefix='All Balanced', phase_label=phase_label,
    )

    print(f"\n--- BALANCED {phase_label.upper()} COHERENCE (by sex) ---")
    coh_data = load_coherence_fn()
    bal_coh = filter_subject_dicts(coh_data, balanced_subjects)
    generate_all_rois_by_sex(
        bal_coh,
        'bc_nostim_rem', 'bc_nostim_forg', 'bc_stim_rem', 'bc_stim_forg',
        sex_map, os.path.join(out_base, 'coherence'),
        'Coherence', 'Baseline-Corrected Coherency (dB)',
        title_prefix='All Balanced', phase_label=phase_label,
    )

    print(f"\n--- BALANCED {phase_label.upper()} PAC (by sex) ---")
    pac_data = load_pac_fn()
    bal_pac = filter_subject_dicts(pac_data, balanced_subjects)
    generate_all_rois_by_sex(
        bal_pac,
        'diff_nostim_rem', 'diff_nostim_forg', 'diff_stim_rem', 'diff_stim_forg',
        sex_map, os.path.join(out_base, 'pac'),
        'PAC', 'Baseline-Corrected PAC',
        title_prefix='All Balanced', phase_label=phase_label,
    )


# =========================================================================
#  PLOTTING
# =========================================================================

def plot_endogenous_vs_stim_for_roi(roi, freqs, nostim_rem, stim_rem,
                                     nostim_forg, stim_forg,
                                     out_dir, measure_label, ylabel,
                                     title_prefix='All', phase_label='Retrieval'):
    """Generate the side-by-side Remembered | Forgotten figure for one ROI."""
    if not nostim_rem and not nostim_forg:
        return

    fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharey=True)

    panels = [
        ('Remembered', nostim_rem, stim_rem, COLOR_REM),
        ('Forgotten',  nostim_forg, stim_forg, COLOR_FORG),
    ]

    for ax, (mem_label, nostim_data, stim_data, endo_color) in zip(axes, panels):
        if nostim_data:
            mat = np.array(list(nostim_data.values()), dtype=np.float64)
            n = mat.shape[0]
            mean = mat.mean(0)
            se = mat.std(0) / np.sqrt(n)
            ax.plot(freqs, mean, color=endo_color, linewidth=2.5,
                    label=f'Endogenous / NoStim ({n})')
            ax.fill_between(freqs, mean - se, mean + se, alpha=0.2, color=endo_color)

        if stim_data:
            mat = np.array(list(stim_data.values()), dtype=np.float64)
            n = mat.shape[0]
            mean = mat.mean(0)
            se = mat.std(0) / np.sqrt(n)
            ax.plot(freqs, mean, color='red', linewidth=2.5,
                    label=f'Stim ({n})')
            ax.fill_between(freqs, mean - se, mean + se, alpha=0.15, color='red')

        ax.axhline(0, color='lightgray', linestyle='--', linewidth=1)
        ax.set_title(f'{mem_label} Trials', fontsize=18, fontweight='bold')
        ax.set_xlabel('Frequency (Hz)', fontsize=16, fontweight='bold')
        fmin, fmax = freqs.min(), freqs.max()
        if fmin >= 25:
            ax.set_xlim(fmin, fmax)
        else:
            ax.set_xlim(max(fmin - 2, 0), fmax + 2)
        ax.tick_params(axis='both', labelsize=14)
        ax.legend(fontsize=13, loc='best')

    axes[0].set_ylabel(ylabel, fontsize=16, fontweight='bold')

    fig.suptitle(f'{title_prefix} {phase_label} {roi}: Endogenous Memory x Stimulation ({measure_label})',
                 fontsize=17, fontweight='bold')

    fig.text(0.5, 0.01,
             fill(f'Method: Each panel shows one memory outcome for {roi}. '
                  f'The colored line is the endogenous (NoStim) {measure_label.lower()} signature; '
                  f'the red line is the stimulation condition. '
                  f'Lines = patient means, shading = SEM.', width=140),
             ha='center', va='bottom', fontsize=9, color='dimgray')

    plt.tight_layout(rect=[0, 0.05, 1, 0.94])
    safe_roi = roi.replace('/', '_').replace(' ', '_')
    phase_tag = phase_label.lower()
    fpath = os.path.join(out_dir, f'{safe_roi}_endogenous_memory_x_stim_{phase_tag}.png')
    plt.savefig(fpath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return fpath


def generate_all_rois(data, bc_nostim_rem_key, bc_nostim_forg_key,
                      bc_stim_rem_key, bc_stim_forg_key,
                      out_dir, measure_label, ylabel,
                      title_prefix='All', phase_label='Retrieval'):
    """Generate figures for all ROIs/pairs found in the data."""
    freqs = data['freqs_diff']
    if freqs is None:
        print(f"  WARNING: No freqs_diff for {measure_label}, skipping.")
        return

    nostim_rem_all = data.get(bc_nostim_rem_key, {})
    nostim_forg_all = data.get(bc_nostim_forg_key, {})
    stim_rem_all = data.get(bc_stim_rem_key, {})
    stim_forg_all = data.get(bc_stim_forg_key, {})

    all_rois = sorted(set(nostim_rem_all.keys()) | set(nostim_forg_all.keys())
                       | set(stim_rem_all.keys()) | set(stim_forg_all.keys()))

    os.makedirs(out_dir, exist_ok=True)
    count = 0
    for roi in all_rois:
        if 'PNAS' in roi:
            continue
        fpath = plot_endogenous_vs_stim_for_roi(
            roi, freqs,
            nostim_rem_all.get(roi, {}),
            stim_rem_all.get(roi, {}),
            nostim_forg_all.get(roi, {}),
            stim_forg_all.get(roi, {}),
            out_dir, measure_label, ylabel,
            title_prefix=title_prefix,
            phase_label=phase_label,
        )
        if fpath:
            count += 1
            print(f"    Saved: {os.path.basename(fpath)}")

    print(f"  {measure_label}: {count} figures for {len(all_rois)} regions/pairs.")


# =========================================================================
#  DATA LOADING — RETRIEVAL
# =========================================================================

def load_retrieval_power():
    from combined_retrieval_power import (
        load_blaes_retrieval, load_amme_retrieval,
        build_all_retrieval_data,
    )
    print("\n  Loading retrieval power data...")
    blaes = filter_pnas_regions(load_blaes_retrieval())
    amme = filter_pnas_regions(load_amme_retrieval())
    return filter_pnas_regions(build_all_retrieval_data(blaes, amme))


def load_retrieval_coherence():
    from combined_retrieval_coherence import (
        load_blaes_retrieval as load_blaes_coh,
        load_amme_retrieval as load_amme_coh,
    )
    print("\n  Loading retrieval coherence data...")
    blaes = filter_pnas_regions(load_blaes_coh())
    amme = filter_pnas_regions(load_amme_coh())
    return merge_coherence_data(blaes, amme)


def load_retrieval_pac():
    from combined_retrieval_pac import load_grouped_retrieval_pac_data
    print("\n  Loading retrieval PAC data...")
    grouped = load_grouped_retrieval_pac_data()
    return grouped['all']


# =========================================================================
#  DATA LOADING — ENCODING
# =========================================================================

def load_encoding_power():
    from combined_encoding_power import (
        load_blaes_encoding, load_amme_encoding,
        build_all_encoding_data,
    )
    print("\n  Loading encoding power data...")
    blaes = filter_pnas_regions(load_blaes_encoding())
    amme = filter_pnas_regions(load_amme_encoding())
    return filter_pnas_regions(build_all_encoding_data(blaes, amme))


def load_encoding_coherence():
    from combined_encoding_coherence import (
        load_blaes_encoding as load_blaes_coh,
        load_amme_encoding as load_amme_coh,
    )
    print("\n  Loading encoding coherence data...")
    blaes = filter_pnas_regions(load_blaes_coh())
    amme = filter_pnas_regions(load_amme_coh())
    return merge_coherence_data(blaes, amme)


def load_encoding_pac():
    from combined_encoding_pac import load_grouped_encoding_pac_data
    print("\n  Loading encoding PAC data...")
    grouped = load_grouped_encoding_pac_data()
    return grouped['all']


# =========================================================================
#  GENERATE ALL MEASURES FOR ONE PHASE
# =========================================================================

def run_phase(phase_label, load_power_fn, load_coherence_fn, load_pac_fn,
              out_base, title_prefix='All'):
    """Run all three measures for a given phase (Encoding or Retrieval)."""

    print(f"\n--- {phase_label.upper()} POWER ---")
    power_data = load_power_fn()
    generate_all_rois(
        power_data,
        bc_nostim_rem_key='bc_nostim_rem',
        bc_nostim_forg_key='bc_nostim_forg',
        bc_stim_rem_key='bc_stim_rem',
        bc_stim_forg_key='bc_stim_forg',
        out_dir=os.path.join(out_base, 'power'),
        measure_label='Power',
        ylabel='Baseline-Corrected Power (dB)',
        title_prefix=title_prefix,
        phase_label=phase_label,
    )

    print(f"\n--- {phase_label.upper()} COHERENCE ---")
    coh_data = load_coherence_fn()
    generate_all_rois(
        coh_data,
        bc_nostim_rem_key='bc_nostim_rem',
        bc_nostim_forg_key='bc_nostim_forg',
        bc_stim_rem_key='bc_stim_rem',
        bc_stim_forg_key='bc_stim_forg',
        out_dir=os.path.join(out_base, 'coherence'),
        measure_label='Coherence',
        ylabel='Baseline-Corrected Coherency (dB)',
        title_prefix=title_prefix,
        phase_label=phase_label,
    )

    print(f"\n--- {phase_label.upper()} PAC ---")
    pac_data = load_pac_fn()
    generate_all_rois(
        pac_data,
        bc_nostim_rem_key='diff_nostim_rem',
        bc_nostim_forg_key='diff_nostim_forg',
        bc_stim_rem_key='diff_stim_rem',
        bc_stim_forg_key='diff_stim_forg',
        out_dir=os.path.join(out_base, 'pac'),
        measure_label='PAC',
        ylabel='Baseline-Corrected PAC',
        title_prefix=title_prefix,
        phase_label=phase_label,
    )

    return power_data


def run_balanced_phase(phase_label, power_data, load_coherence_fn, load_pac_fn,
                       out_base):
    """Run balanced-memory version of all three measures for one phase."""
    from balanced_memory_trials import (
        get_nostim_trial_counts,
        get_balanced_subjects,
    )

    print(f"\n  Determining balanced subjects ({phase_label.lower()})...")
    trial_counts = get_nostim_trial_counts(power_data, measure='power')
    balanced_subjects = get_balanced_subjects(trial_counts)
    if balanced_subjects is None:
        print(f"  ERROR: No balanced subjects for {phase_label}. Skipping.")
        return

    title_pfx = 'All Balanced'

    print(f"\n--- BALANCED {phase_label.upper()} POWER ---")
    bal_power = filter_subject_dicts(power_data, balanced_subjects)
    generate_all_rois(
        bal_power,
        bc_nostim_rem_key='bc_nostim_rem',
        bc_nostim_forg_key='bc_nostim_forg',
        bc_stim_rem_key='bc_stim_rem',
        bc_stim_forg_key='bc_stim_forg',
        out_dir=os.path.join(out_base, 'power'),
        measure_label='Power',
        ylabel='Baseline-Corrected Power (dB)',
        title_prefix=title_pfx,
        phase_label=phase_label,
    )

    print(f"\n--- BALANCED {phase_label.upper()} COHERENCE ---")
    coh_data = load_coherence_fn()
    bal_coh = filter_subject_dicts(coh_data, balanced_subjects)
    generate_all_rois(
        bal_coh,
        bc_nostim_rem_key='bc_nostim_rem',
        bc_nostim_forg_key='bc_nostim_forg',
        bc_stim_rem_key='bc_stim_rem',
        bc_stim_forg_key='bc_stim_forg',
        out_dir=os.path.join(out_base, 'coherence'),
        measure_label='Coherence',
        ylabel='Baseline-Corrected Coherency (dB)',
        title_prefix=title_pfx,
        phase_label=phase_label,
    )

    print(f"\n--- BALANCED {phase_label.upper()} PAC ---")
    pac_data = load_pac_fn()
    bal_pac = filter_subject_dicts(pac_data, balanced_subjects)
    generate_all_rois(
        bal_pac,
        bc_nostim_rem_key='diff_nostim_rem',
        bc_nostim_forg_key='diff_nostim_forg',
        bc_stim_rem_key='diff_stim_rem',
        bc_stim_forg_key='diff_stim_forg',
        out_dir=os.path.join(out_base, 'pac'),
        measure_label='PAC',
        ylabel='Baseline-Corrected PAC',
        title_prefix=title_pfx,
        phase_label=phase_label,
    )


# =========================================================================
#  MAIN
# =========================================================================

def main():
    print("=" * 60)
    print("Endogenous Memory x Stimulation — All Measures & Phases")
    print("=" * 60)

    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    # === RETRIEVAL ===
    print("\n" + "=" * 60)
    print("RETRIEVAL")
    print("=" * 60)
    ret_power = run_phase(
        'Retrieval',
        load_retrieval_power, load_retrieval_coherence, load_retrieval_pac,
        os.path.join(OUTPUT_ROOT, 'retrieval'),
    )

    # === ENCODING ===
    print("\n" + "=" * 60)
    print("ENCODING")
    print("=" * 60)
    enc_power = run_phase(
        'Encoding',
        load_encoding_power, load_encoding_coherence, load_encoding_pac,
        os.path.join(OUTPUT_ROOT, 'encoding'),
    )

    # === BALANCED RETRIEVAL ===
    print("\n" + "=" * 60)
    print("BALANCED RETRIEVAL")
    print("=" * 60)
    run_balanced_phase(
        'Retrieval', ret_power,
        load_retrieval_coherence, load_retrieval_pac,
        os.path.join(OUTPUT_ROOT, 'balanced_retrieval'),
    )

    # === BALANCED ENCODING ===
    print("\n" + "=" * 60)
    print("BALANCED ENCODING")
    print("=" * 60)
    run_balanced_phase(
        'Encoding', enc_power,
        load_encoding_coherence, load_encoding_pac,
        os.path.join(OUTPUT_ROOT, 'balanced_encoding'),
    )

    # =================================================================
    # SEX-STRATIFIED ANALYSES
    # =================================================================
    print("\n\n" + "#" * 60)
    print("SEX-STRATIFIED ANALYSES")
    print("#" * 60)

    sex_map = load_sex_map()
    print(f"  Loaded sex metadata: {sum(1 for v in sex_map.values() if v=='male')} male, "
          f"{sum(1 for v in sex_map.values() if v=='female')} female")

    # === SEX: RETRIEVAL ===
    print("\n" + "=" * 60)
    print("SEX — RETRIEVAL")
    print("=" * 60)
    run_sex_phase('Retrieval', ret_power,
                  load_retrieval_coherence, load_retrieval_pac,
                  sex_map, os.path.join(OUTPUT_ROOT, 'retrieval_by_sex'))

    # === SEX: ENCODING ===
    print("\n" + "=" * 60)
    print("SEX — ENCODING")
    print("=" * 60)
    run_sex_phase('Encoding', enc_power,
                  load_encoding_coherence, load_encoding_pac,
                  sex_map, os.path.join(OUTPUT_ROOT, 'encoding_by_sex'))

    # === SEX: BALANCED RETRIEVAL ===
    print("\n" + "=" * 60)
    print("SEX — BALANCED RETRIEVAL")
    print("=" * 60)
    run_balanced_sex_phase('Retrieval', ret_power,
                           load_retrieval_coherence, load_retrieval_pac,
                           sex_map, os.path.join(OUTPUT_ROOT, 'balanced_retrieval_by_sex'))

    # === SEX: BALANCED ENCODING ===
    print("\n" + "=" * 60)
    print("SEX — BALANCED ENCODING")
    print("=" * 60)
    run_balanced_sex_phase('Encoding', enc_power,
                           load_encoding_coherence, load_encoding_pac,
                           sex_map, os.path.join(OUTPUT_ROOT, 'balanced_encoding_by_sex'))

    print("\n" + "=" * 60)
    print(f"Done! All figures saved to: {OUTPUT_ROOT}")
    print("=" * 60)


if __name__ == '__main__':
    main()
