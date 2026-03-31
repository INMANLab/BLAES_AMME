#!/usr/bin/env python
"""
Endogenous Memory Analysis
==========================
Analyzes remembered vs forgotten trials using ONLY no-stim (endogenous) trials
across power, coherence, and PAC for both encoding and retrieval phases.

This script reuses the data loading functions from the existing analysis scripts
and generates a parallel set of figures focused on the endogenous memory contrast.
"""

import os
import sys
import shutil
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

OUTPUT_ROOT = os.path.join(SCRIPT_DIR, 'outputs', 'endogenous_memory')
CSV_OUTPUT_DIR = os.path.join(OUTPUT_ROOT, 'csvs')
RESPONDER_STATUS_CSV = os.path.join(SCRIPT_DIR, 'outputs', 'csvs', 'AMMEBLAES_responder_status.csv')

POWER_RANGES = {'Theta': (4, 8), 'Slow gamma': (30, 55)}
PAC_BANDS = {'Slow gamma': (30, 55)}

ROI_COLORS_SPEC = {
    'BLA': 'teal', 'ALLHPC': '#5B4B8A', 'MTL': '#2E8B57',
    'PRC': '#E61C59', 'EC': '#8A2BE2', 'PHG': 'blue',
    'CA': 'black', 'DG': 'skyblue', 'HPC': 'orange',
}
ROI_COLORS_BAR = {
    'BLA': '#008080', 'ALLHPC': '#5B4B8A', 'MTL': '#2E8B57',
    'CA': '#00CED1', 'DG': '#87CEEB', 'HPC': '#FFA500',
    'EC': '#800080', 'PHG': '#0000FF', 'PRC': '#FFC0CB',
}
MEMORY_COLORS = {'Remembered': '#7B2D8E', 'Forgotten': '#DAA520'}
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
    'Unknown': '#7F7F7F',
}
_RESPONDER_STATUS_MAP = None


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def reset_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)
    return path


def finalize_figure(fig=None):
    plt.close(fig)


def band_filename(filename, band):
    root, ext = os.path.splitext(filename)
    return f'{root}_{band.lower().replace(" ", "_")}{ext}'


def merge_dicts(*dicts):
    merged = {}
    for d in dicts:
        if not d:
            continue
        for roi, sd in d.items():
            merged.setdefault(roi, {}).update(sd)
    return merged


def sorted_freq_cols(df, prefix):
    cols = [c for c in df.columns if c.startswith(prefix)]
    return sorted(cols, key=lambda x: float(x.split(prefix)[1]))


def freq_vals(cols, prefix):
    return np.array([float(c.split(prefix)[1]) for c in cols], dtype=float)


def set_freq_xlim(ax, freqs):
    """Set x-axis limits to the observed frequency range."""
    fmin, fmax = freqs.min(), freqs.max()
    if fmin >= 25:
        ax.set_xlim(fmin, fmax)
        return
    pad = (fmax - fmin) * 0.03
    ax.set_xlim(max(fmin - pad, fmin - 5), fmax + pad)


def collect_unique_legend_items(axes):
    handles, labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        for hh, ll in zip(h, l):
            if ll and ll not in labels:
                handles.append(hh)
                labels.append(ll)
    return handles, labels


def make_subject_color_map(subjects):
    ordered = sorted(subjects)
    palette = sns.color_palette('husl', len(ordered))
    return {s: c for s, c in zip(ordered, palette)}


def get_responder_status_map():
    global _RESPONDER_STATUS_MAP
    if _RESPONDER_STATUS_MAP is not None:
        return _RESPONDER_STATUS_MAP
    if not os.path.exists(RESPONDER_STATUS_CSV):
        _RESPONDER_STATUS_MAP = {}
        return _RESPONDER_STATUS_MAP
    status_df = pd.read_csv(RESPONDER_STATUS_CSV)
    status_column = None
    for candidate in ['Responder status', 'Responder_status', 'quartile']:
        if candidate in status_df.columns:
            status_column = candidate
            break
    if status_column is None or 'Patient' not in status_df.columns:
        _RESPONDER_STATUS_MAP = {}
        return _RESPONDER_STATUS_MAP
    status_df = status_df[['Patient', status_column]].dropna(subset=['Patient']).copy()
    status_df['Patient'] = status_df['Patient'].astype(str)
    _RESPONDER_STATUS_MAP = dict(zip(status_df['Patient'], status_df[status_column]))
    return _RESPONDER_STATUS_MAP


def get_anchor_region(roi_name):
    return str(roi_name).split('_', 1)[0]


def remove_matching_files(out_dir, prefixes):
    if not os.path.isdir(out_dir):
        return
    for filename in os.listdir(out_dir):
        if any(filename.startswith(prefix) for prefix in prefixes):
            os.remove(os.path.join(out_dir, filename))


def plot_anchor_subset_bargraph(
    subset_df,
    out_dir,
    label,
    phase_label,
    band,
    anchor,
    measure_label,
    responder_colored=False,
):
    if subset_df.empty:
        return

    roi_order = sorted(subset_df['Region'].unique())
    if not roi_order:
        return

    fig_width = max(10.5, 1.25 * len(roi_order) + 3.0)
    fig, ax = plt.subplots(figsize=(fig_width, 8.2))
    x = np.arange(len(roi_order))
    summary = subset_df.groupby('Region')['rem_minus_forg'].agg(['mean', 'sem']).reindex(roi_order)
    gray_values = np.linspace(0.85, 0.45, len(roi_order))
    bar_colors = {roi: matplotlib.colors.to_hex((g, g, g)) for roi, g in zip(roi_order, gray_values)}
    ax.bar(
        x,
        summary['mean'].to_numpy(),
        yerr=summary['sem'].fillna(0).to_numpy(),
        color=[bar_colors[r] for r in roi_order],
        edgecolor='#4D4D4D',
        linewidth=1.2,
        width=0.72,
        capsize=3,
        ecolor='#4D4D4D',
        zorder=1,
    )

    rng = np.random.default_rng(7)
    plot_df = subset_df.copy()
    if responder_colored:
        plot_df['Responder status'] = plot_df['Patient'].map(get_responder_status_map()).fillna('Unknown')

    for idx, roi in enumerate(roi_order):
        roi_df = plot_df[plot_df['Region'] == roi]
        jitter = rng.uniform(-0.16, 0.16, len(roi_df))
        if responder_colored:
            colors = [RESPONDER_PALETTE.get(status, RESPONDER_PALETTE['Unknown']) for status in roi_df['Responder status']]
            ax.scatter(
                np.full(len(roi_df), idx, dtype=float) + jitter,
                roi_df['rem_minus_forg'],
                c=colors,
                s=52,
                alpha=0.85,
                edgecolors='none',
                zorder=3,
            )
        else:
            ax.scatter(
                np.full(len(roi_df), idx, dtype=float) + jitter,
                roi_df['rem_minus_forg'],
                c='black',
                s=42,
                alpha=0.55,
                zorder=3,
            )

    ax.axhline(0, color='#4D4D4D', linewidth=1.2, zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels(roi_order, fontsize=16, fontweight='bold', rotation=45, ha='right')
    ax.tick_params(axis='y', labelsize=15, width=2, length=6)
    ax.set_ylabel(
        f'Remembered − Forgotten\n(Baseline-Corrected {measure_label})',
        fontsize=16,
        fontweight='bold',
    )
    ax.set_title(band, fontsize=20, fontweight='bold')
    fig.suptitle(
        f'{label} {phase_label} Endogenous Memory Effect (NoStim) — {anchor} Connections',
        fontsize=22,
        fontweight='bold',
        y=0.98,
    )
    caption = (
        'Method: For each patient and region pair, baseline-corrected spectra are averaged across '
        'remembered and forgotten NoStim trials separately, band means are computed, then '
        'Remembered − Forgotten is taken. Bars = mean ± SEM, dots = individual patients.'
    )
    if responder_colored:
        caption += ' Dots are colored by responder-status CSV.'
    fig.text(0.015, 0.015, caption, ha='left', va='bottom', fontsize=9, wrap=True)

    legend_handles = []
    if responder_colored:
        for status in RESPONDER_ORDER:
            if status in plot_df['Responder status'].values:
                legend_handles.append(
                    Line2D(
                        [0],
                        [0],
                        marker='o',
                        linestyle='',
                        markersize=9,
                        markerfacecolor=RESPONDER_PALETTE[status],
                        markeredgecolor='none',
                        label=status,
                    )
                )
        if 'Unknown' in plot_df['Responder status'].values:
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    marker='o',
                    linestyle='',
                    markersize=9,
                    markerfacecolor=RESPONDER_PALETTE['Unknown'],
                    markeredgecolor='none',
                    label='Unknown',
                )
            )
    if legend_handles:
        fig.legend(
            handles=legend_handles,
            loc='upper center',
            bbox_to_anchor=(0.5, 0.92),
            ncol=min(5, len(legend_handles)),
            frameon=False,
            fontsize=12,
            title='Responder status',
            title_fontsize=13,
        )
        fig.tight_layout(rect=[0, 0.06, 1, 0.88])
    else:
        fig.tight_layout(rect=[0, 0.06, 1, 0.90])

    responder_suffix = '_responder_status' if responder_colored else ''
    out_name = (
        f'Endogenous_Bargraph_RemMinusForg_{measure_label}_{phase_label.lower()}_'
        f'{anchor}_connections_{band.lower().replace(" ", "_")}{responder_suffix}.png'
    )
    plt.savefig(os.path.join(out_dir, out_name), dpi=300, bbox_inches='tight')
    finalize_figure(fig)


# ---------------------------------------------------------------------------
# Extract endogenous memory data from loaded data dict
# ---------------------------------------------------------------------------

def extract_endogenous(data, measure='power'):
    """Pull nostim-only remembered/forgotten from a loaded data dictionary.

    Works for both power (keys: nostim_rem, nostim_forg, bc_nostim_rem, etc.)
    and PAC (keys: post_nostim_rem, post_nostim_forg, diff_nostim_rem, etc.).
    """
    is_pac = measure == 'pac'
    if is_pac:
        return {
            'remembered': data.get('post_nostim_rem', {}),
            'forgotten': data.get('post_nostim_forg', {}),
            'bc_remembered': data.get('diff_nostim_rem', {}),
            'bc_forgotten': data.get('diff_nostim_forg', {}),
            'overall': data.get('post_nostim', {}),
            'bc_overall': data.get('diff_nostim', {}),
            'freqs_post': data.get('freqs_post'),
            'freqs_diff': data.get('freqs_diff'),
        }
    else:
        return {
            'remembered': data.get('nostim_rem', {}),
            'forgotten': data.get('nostim_forg', {}),
            'bc_remembered': data.get('bc_nostim_rem', {}),
            'bc_forgotten': data.get('bc_nostim_forg', {}),
            'overall': data.get('nostim_power', data.get('nostim_coherence', {})),
            'bc_overall': data.get('bc_nostim', {}),
            'freqs_post': data.get('freqs_post'),
            'freqs_diff': data.get('freqs_diff'),
        }


# ---------------------------------------------------------------------------
# Plotting functions — Endogenous Memory (nostim only, rem vs forg)
# ---------------------------------------------------------------------------

def plot_overall_by_roi(endo, out_dir, label, measure_label, phase_label):
    """Mean spectra per ROI, nostim trials only (collapsed across memory)."""
    overall = endo['overall']
    freqs = endo['freqs_post']
    if not overall or freqs is None:
        return
    fig, ax = plt.subplots(figsize=(12, 8))
    for roi in sorted(overall.keys()):
        mat = np.array(list(overall[roi].values()), dtype=np.float64)
        mean, std = mat.mean(0), mat.std(0)
        c = ROI_COLORS_SPEC.get(roi, 'gray')
        ax.plot(freqs, mean, color=c, label=f'{roi} ({mat.shape[0]})')
        ax.fill_between(freqs, mean - std, mean + std, alpha=0.2, color=c)
    ax.set_xlabel('Frequency (Hz)', fontsize=18, fontweight='bold')
    ax.set_ylabel(f'{measure_label}', fontsize=18, fontweight='bold')
    ax.set_title(f'{label} {phase_label} Endogenous {measure_label} by ROI (NoStim Only)',
                 fontsize=18, fontweight='bold')
    ax.tick_params(axis='both', labelsize=14)
    set_freq_xlim(ax, freqs)
    ax.legend(bbox_to_anchor=(1.02, 0.5), loc='center left', prop={'weight': 'bold', 'size': 12})
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(os.path.join(out_dir, f'Endogenous_{measure_label}_byROI_{phase_label.lower()}.png'),
                dpi=300, bbox_inches='tight')
    finalize_figure()


def plot_overall_by_patient(endo, out_dir, label, measure_label, phase_label):
    """Mean spectra per patient, nostim trials only (collapsed across memory)."""
    overall = endo['overall']
    freqs = endo['freqs_post']
    if not overall or freqs is None:
        return
    patient_spectra = {}
    for roi, sd in overall.items():
        for subj, pv in sd.items():
            patient_spectra.setdefault(subj, []).append(pv)
    if not patient_spectra:
        return
    fig, ax = plt.subplots(figsize=(12, 8))
    for subj in sorted(patient_spectra):
        mean_p = np.array(patient_spectra[subj]).mean(0)
        ax.plot(freqs, mean_p, label=subj)
    ax.set_xlabel('Frequency (Hz)', fontsize=18, fontweight='bold')
    ax.set_ylabel(f'{measure_label}', fontsize=18, fontweight='bold')
    ax.set_title(f'{label} {phase_label} Endogenous {measure_label} by Patient (NoStim Only)',
                 fontsize=18, fontweight='bold')
    ax.tick_params(axis='both', labelsize=14)
    set_freq_xlim(ax, freqs)
    ax.legend(bbox_to_anchor=(1.02, 0.5), loc='center left', prop={'weight': 'bold', 'size': 10})
    plt.tight_layout(rect=[0, 0, 0.82, 1])
    plt.savefig(os.path.join(out_dir, f'Endogenous_{measure_label}_byPatient_{phase_label.lower()}.png'),
                dpi=300, bbox_inches='tight')
    finalize_figure()


def plot_remembered_vs_forgotten_spectra(endo, out_dir, label, measure_label, phase_label):
    """Side-by-side remembered vs forgotten spectra by ROI (raw post)."""
    rem_d, forg_d = endo['remembered'], endo['forgotten']
    freqs = endo['freqs_post']
    if not rem_d and not forg_d or freqs is None:
        return
    all_rois = sorted(set(rem_d.keys()) | set(forg_d.keys()))
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharex=True, sharey=True)
    for ax, (title, rd) in zip(axes, [('Remembered', rem_d), ('Forgotten', forg_d)]):
        for roi in all_rois:
            if roi not in rd:
                continue
            mat = np.array(list(rd[roi].values()), dtype=np.float64)
            mean, std = mat.mean(0), mat.std(0)
            c = ROI_COLORS_SPEC.get(roi, 'gray')
            ax.plot(freqs, mean, color=c, label=f'{roi} ({mat.shape[0]})')
            ax.fill_between(freqs, mean - std, mean + std, alpha=0.2, color=c)
        ax.set_title(title, fontsize=18, fontweight='bold')
        ax.set_xlabel('Frequency (Hz)', fontsize=16, fontweight='bold')
        ax.tick_params(axis='both', labelsize=12)
    axes[0].set_ylabel(f'{measure_label}', fontsize=16, fontweight='bold')
    for ax in axes:
        set_freq_xlim(ax, freqs)
    axes[1].legend(bbox_to_anchor=(1.02, 0.5), loc='center left', prop={'weight': 'bold', 'size': 11})
    fig.suptitle(f'{label} {phase_label} Endogenous {measure_label}: Remembered vs Forgotten (NoStim)',
                 fontsize=18, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 0.88, 0.95])
    plt.savefig(os.path.join(out_dir,
                f'Endogenous_{measure_label}_RemVsForg_{phase_label.lower()}.png'),
                dpi=300, bbox_inches='tight')
    finalize_figure()


def plot_per_roi_rem_vs_forg_patients(endo, out_dir, label, measure_label, phase_label):
    """Per-ROI remembered vs forgotten with individual patient lines (raw post)."""
    rem_d, forg_d = endo['remembered'], endo['forgotten']
    freqs = endo['freqs_post']
    if not rem_d and not forg_d or freqs is None:
        return
    all_rois = sorted(set(rem_d.keys()) | set(forg_d.keys()))
    for roi in all_rois:
        roi_subjects = set(rem_d.get(roi, {})) | set(forg_d.get(roi, {}))
        if not roi_subjects:
            continue
        subject_colors = make_subject_color_map(roi_subjects)
        fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharex=True, sharey=True)
        for ax, (title, rd) in zip(axes, [(f'{roi} - Remembered', rem_d), (f'{roi} - Forgotten', forg_d)]):
            if roi not in rd:
                ax.set_title(title, fontsize=16, fontweight='bold')
                continue
            for subj in sorted(rd[roi]):
                ax.plot(freqs, rd[roi][subj], label=subj, color=subject_colors[subj])
            ax.set_title(title, fontsize=16, fontweight='bold')
            ax.set_xlabel('Frequency (Hz)', fontsize=16, fontweight='bold')
            ax.tick_params(axis='both', labelsize=12)
        axes[0].set_ylabel(f'{measure_label}', fontsize=16, fontweight='bold')
        for ax in axes:
            set_freq_xlim(ax, freqs)
        handles, labels_ = collect_unique_legend_items(axes)
        if handles:
            fig.legend(handles, labels_, bbox_to_anchor=(0.88, 0.5), loc='center left',
                       prop={'weight': 'bold', 'size': 10})
        fig.suptitle(f'{label} {phase_label} Endogenous {roi} by Patient (NoStim)',
                     fontsize=18, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 0.82, 0.95])
        plt.savefig(os.path.join(out_dir,
                    f'Endogenous_{measure_label}_{roi}_byPatient_RemForg_{phase_label.lower()}.png'),
                    dpi=300, bbox_inches='tight')
        finalize_figure()


def plot_bc_rem_vs_forg_per_roi(endo, out_dir, label, measure_label, phase_label):
    """Baseline-corrected remembered vs forgotten spectra per ROI (mean ± SEM)."""
    bc_rem, bc_forg = endo['bc_remembered'], endo['bc_forgotten']
    freqs = endo['freqs_diff']
    if not bc_rem and not bc_forg or freqs is None:
        return
    all_rois = sorted(set(bc_rem.keys()) | set(bc_forg.keys()))
    for roi in all_rois:
        has_data = roi in bc_rem or roi in bc_forg
        if not has_data:
            continue
        fig, ax = plt.subplots(figsize=(10, 6))
        for mem_label_inner, rd, color in [('Remembered', bc_rem, MEMORY_COLORS['Remembered']),
                                           ('Forgotten', bc_forg, MEMORY_COLORS['Forgotten'])]:
            if roi not in rd:
                continue
            mat = np.array(list(rd[roi].values()), dtype=np.float64)
            n = mat.shape[0]
            mean, se = mat.mean(0), mat.std(0) / np.sqrt(n)
            ax.plot(freqs, mean, color=color, linewidth=2, label=f'{mem_label_inner} ({n})')
            ax.fill_between(freqs, mean - se, mean + se, alpha=0.2, color=color)
        ax.axhline(0, color='lightgray', linestyle='--', linewidth=1)
        ax.set_xlabel('Frequency (Hz)', fontsize=16, fontweight='bold')
        ax.set_ylabel(f'Baseline-Corrected {measure_label}', fontsize=16, fontweight='bold')
        ax.set_title(f'{label} {phase_label} Endogenous {roi}\nRemembered vs Forgotten (NoStim, BC)',
                     fontsize=16, fontweight='bold')
        set_freq_xlim(ax, freqs)
        ax.tick_params(axis='both', labelsize=14)
        ax.legend(fontsize=13, loc='best')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir,
                    f'Endogenous_BC_{measure_label}_{roi}_RemVsForg_{phase_label.lower()}.png'),
                    dpi=300, bbox_inches='tight')
        finalize_figure()


def plot_pac_memory_mi_difference_per_roi(endo, out_dir, label, phase_label):
    """Plot remembered-minus-forgotten MI difference per ROI for no-stim trials only."""
    bc_rem, bc_forg = endo['bc_remembered'], endo['bc_forgotten']
    freqs = endo['freqs_diff']
    if freqs is None or (not bc_rem and not bc_forg):
        return

    all_rois = sorted(set(bc_rem.keys()) | set(bc_forg.keys()))
    for roi in all_rois:
        shared_subjects = sorted(set(bc_rem.get(roi, {})) & set(bc_forg.get(roi, {})))
        if not shared_subjects:
            continue

        diff_matrix = np.array(
            [
                np.asarray(bc_rem[roi][patient], dtype=np.float64)
                - np.asarray(bc_forg[roi][patient], dtype=np.float64)
                for patient in shared_subjects
            ],
            dtype=np.float64,
        )
        mean = diff_matrix.mean(axis=0)
        sem = diff_matrix.std(axis=0) / np.sqrt(diff_matrix.shape[0])

        fig, ax = plt.subplots(figsize=(11, 6.8))
        ax.axvspan(PAC_BANDS['Slow gamma'][0], PAC_BANDS['Slow gamma'][1], color='#E8E8E8', alpha=1.0, zorder=0)
        ax.axhline(0, color='red', linewidth=1.8, linestyle=(0, (4, 4)), zorder=1)
        ax.plot(freqs, mean, color='#8E8E8E', linewidth=3, zorder=3)
        ax.fill_between(freqs, mean - sem, mean + sem, color='#BDBDBD', alpha=0.85, zorder=2)
        ax.set_title(f'{label} {phase_label} {roi} MI Difference (Remembered - Forgotten, NoStim)',
                     fontsize=17, fontweight='bold')
        ax.set_xlabel('Amplitude Frequency (Hz)', fontsize=15, fontweight='bold')
        ax.set_ylabel('MI Difference', fontsize=15, fontweight='bold')
        ax.tick_params(axis='both', labelsize=12, width=1.6, length=5)
        set_freq_xlim(ax, freqs)
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        ax.spines['left'].set_linewidth(1.6)
        ax.spines['bottom'].set_linewidth(1.6)
        plt.tight_layout()
        plt.savefig(
            os.path.join(out_dir, f'Endogenous_MI_difference_{roi}_{phase_label.lower()}.png'),
            dpi=300,
            bbox_inches='tight',
        )
        finalize_figure(fig)


def plot_bc_band_bargraph(endo, out_dir, label, measure_label, phase_label, band_ranges=None):
    """Bar graph of remembered minus forgotten band power per ROI with individual dots."""
    bc_rem, bc_forg = endo['bc_remembered'], endo['bc_forgotten']
    freqs = endo['freqs_diff']
    if not bc_rem or not bc_forg or freqs is None:
        return
    if band_ranges is None:
        band_ranges = POWER_RANGES

    rows = []
    all_rois = sorted(set(bc_rem.keys()) & set(bc_forg.keys()))
    for roi in all_rois:
        for subj in sorted(set(bc_rem[roi]) & set(bc_forg[roi])):
            rem_vec = np.asarray(bc_rem[roi][subj], dtype=np.float64)
            forg_vec = np.asarray(bc_forg[roi][subj], dtype=np.float64)
            for bn, (lo, hi) in band_ranges.items():
                mask = (freqs >= lo) & (freqs <= hi)
                rows.append({
                    'Patient': subj,
                    'Region': roi,
                    'power_range': bn,
                    'rem_minus_forg': rem_vec[mask].mean() - forg_vec[mask].mean(),
                })
    if not rows:
        return
    df = pd.DataFrame(rows)

    # PAC and Coherence: split by anchor region (too many pairs for single figure)
    if measure_label in ('PAC', 'Coherence'):
        for band in band_ranges:
            band_df = df[df['power_range'] == band].copy()
            if band_df.empty:
                continue
            band_df['anchor'] = band_df['Region'].map(get_anchor_region)
            for anchor in sorted(band_df['anchor'].unique()):
                subset_df = band_df[band_df['anchor'] == anchor].copy()
                if subset_df.empty:
                    continue
                plot_anchor_subset_bargraph(
                    subset_df=subset_df,
                    out_dir=out_dir,
                    label=label,
                    phase_label=phase_label,
                    band=band,
                    anchor=anchor,
                    measure_label=measure_label,
                    responder_colored=False,
                )
                plot_anchor_subset_bargraph(
                    subset_df=subset_df,
                    out_dir=out_dir,
                    label=label,
                    phase_label=phase_label,
                    band=band,
                    anchor=anchor,
                    measure_label=measure_label,
                    responder_colored=True,
                )
        return

    # Power (and other non-pair measures): single bargraph per band
    for band in band_ranges:
        band_df = df[df['power_range'] == band].copy()
        if band_df.empty:
            continue
        roi_order = [r for r in ROI_COLORS_BAR if r in band_df['Region'].unique()]
        if not roi_order:
            roi_order = sorted(band_df['Region'].unique())

        for responder_colored in [False, True]:
            fig, ax = plt.subplots(figsize=(11.5, 8.2))
            x = np.arange(len(roi_order))
            summary = band_df.groupby('Region')['rem_minus_forg'].agg(['mean', 'sem']).reindex(roi_order)
            gray_values = np.linspace(0.85, 0.45, len(roi_order))
            bar_colors_map = {roi: matplotlib.colors.to_hex((g, g, g)) for roi, g in zip(roi_order, gray_values)}
            ax.bar(
                x, summary['mean'].to_numpy(),
                yerr=summary['sem'].fillna(0).to_numpy(),
                color=[bar_colors_map[r] for r in roi_order],
                edgecolor='#4D4D4D', linewidth=1.2, width=0.72,
                capsize=3, ecolor='#4D4D4D', zorder=1,
            )
            rng = np.random.default_rng(7)
            plot_df = band_df.copy()
            if responder_colored:
                plot_df['Responder status'] = plot_df['Patient'].map(get_responder_status_map()).fillna('Unknown')

            for idx, roi in enumerate(roi_order):
                roi_df = plot_df[plot_df['Region'] == roi]
                jitter = rng.uniform(-0.16, 0.16, len(roi_df))
                if responder_colored:
                    colors = [RESPONDER_PALETTE.get(s, RESPONDER_PALETTE['Unknown']) for s in roi_df['Responder status']]
                    ax.scatter(
                        np.full(len(roi_df), idx, dtype=float) + jitter,
                        roi_df['rem_minus_forg'], c=colors, s=52, alpha=0.85,
                        edgecolors='none', zorder=3,
                    )
                else:
                    ax.scatter(
                        np.full(len(roi_df), idx, dtype=float) + jitter,
                        roi_df['rem_minus_forg'], c='black', s=42, alpha=0.6, zorder=3,
                    )
            ax.axhline(0, color='#4D4D4D', linewidth=1.2, zorder=0)
            ax.set_xticks(x)
            ax.set_xticklabels(roi_order, fontsize=16, fontweight='bold')
            ax.tick_params(axis='y', labelsize=15, width=2, length=6)
            ax.set_ylabel(f'Remembered − Forgotten\n(Baseline-Corrected {measure_label})',
                          fontsize=16, fontweight='bold')
            ax.set_title(band, fontsize=20, fontweight='bold')
            fig.suptitle(f'{label} {phase_label} Endogenous Memory Effect (NoStim)',
                         fontsize=22, fontweight='bold', y=0.98)
            caption = (
                'Method: For each patient and ROI, baseline-corrected spectra are averaged across '
                'remembered and forgotten NoStim trials separately, band means are computed, then '
                'Remembered − Forgotten is taken. Bars = mean ± SEM, dots = individual patients.'
            )
            if responder_colored:
                caption += ' Dots are colored by responder-status CSV.'
            fig.text(0.015, 0.015, caption, ha='left', va='bottom', fontsize=9, wrap=True)

            legend_handles = []
            if responder_colored:
                for status in RESPONDER_ORDER:
                    if status in plot_df['Responder status'].values:
                        legend_handles.append(
                            Line2D([0], [0], marker='o', linestyle='', markersize=9,
                                   markerfacecolor=RESPONDER_PALETTE[status],
                                   markeredgecolor='none', label=status))
                if 'Unknown' in plot_df['Responder status'].values:
                    legend_handles.append(
                        Line2D([0], [0], marker='o', linestyle='', markersize=9,
                               markerfacecolor=RESPONDER_PALETTE['Unknown'],
                               markeredgecolor='none', label='Unknown'))
            if legend_handles:
                fig.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, 0.92),
                           ncol=min(5, len(legend_handles)), frameon=False, fontsize=12,
                           title='Responder status', title_fontsize=13)
                fig.tight_layout(rect=[0, 0.06, 1, 0.88])
            else:
                fig.tight_layout(rect=[0, 0.06, 1, 0.90])

            responder_suffix = '_responder_status' if responder_colored else ''
            plt.savefig(os.path.join(out_dir,
                        band_filename(f'Endogenous_Bargraph_RemMinusForg_{measure_label}_{phase_label.lower()}{responder_suffix}.png', band)),
                        dpi=300, bbox_inches='tight')
            finalize_figure(fig)


def plot_bc_paired_rem_forg_bars(endo, out_dir, label, measure_label, phase_label, band_ranges=None):
    """Paired bar graph showing remembered and forgotten separately per ROI."""
    bc_rem, bc_forg = endo['bc_remembered'], endo['bc_forgotten']
    freqs = endo['freqs_diff']
    if not bc_rem or not bc_forg or freqs is None:
        return
    if band_ranges is None:
        band_ranges = POWER_RANGES

    rows = []
    all_rois = sorted(set(bc_rem.keys()) | set(bc_forg.keys()))
    for roi in all_rois:
        for mem_key, mem_label_inner, rd in [('rem', 'Remembered', bc_rem), ('forg', 'Forgotten', bc_forg)]:
            if roi not in rd:
                continue
            for subj, vec in rd[roi].items():
                arr = np.asarray(vec, dtype=np.float64)
                for bn, (lo, hi) in band_ranges.items():
                    mask = (freqs >= lo) & (freqs <= hi)
                    rows.append({
                        'Patient': subj, 'Region': roi, 'power_range': bn,
                        'condition': mem_label_inner, 'value': arr[mask].mean(),
                    })
    if not rows:
        return
    df = pd.DataFrame(rows)

    for band in band_ranges:
        band_df = df[df['power_range'] == band]
        if band_df.empty:
            continue
        roi_order = [r for r in ROI_COLORS_BAR if r in band_df['Region'].unique()]
        if not roi_order:
            roi_order = sorted(band_df['Region'].unique())

        fig, ax = plt.subplots(figsize=(14, 8))
        x_base = np.arange(len(roi_order))
        bar_width = 0.34
        offsets = {'Remembered': -bar_width / 2, 'Forgotten': bar_width / 2}

        for i, roi in enumerate(roi_order):
            roi_df = band_df[band_df['Region'] == roi]
            for cond in ['Remembered', 'Forgotten']:
                vals = roi_df[roi_df['condition'] == cond]['value'].dropna()
                if len(vals) == 0:
                    continue
                mean = vals.mean()
                sem = vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0
                ax.bar(
                    x_base[i] + offsets[cond], mean,
                    width=bar_width, color=MEMORY_COLORS[cond],
                    edgecolor='black', linewidth=1.2, yerr=sem, capsize=4, zorder=1,
                )

        # Draw patient-level connected dots
        patients = sorted(band_df['Patient'].unique())
        for roi_idx, roi in enumerate(roi_order):
            roi_df = band_df[band_df['Region'] == roi]
            for pat in patients:
                rem_vals = roi_df[(roi_df['Patient'] == pat) & (roi_df['condition'] == 'Remembered')]['value']
                forg_vals = roi_df[(roi_df['Patient'] == pat) & (roi_df['condition'] == 'Forgotten')]['value']
                if len(rem_vals) == 1 and len(forg_vals) == 1:
                    x_r = x_base[roi_idx] + offsets['Remembered']
                    x_f = x_base[roi_idx] + offsets['Forgotten']
                    ax.plot([x_r, x_f], [rem_vals.iloc[0], forg_vals.iloc[0]],
                            color='black', alpha=0.25, linewidth=0.8)
                    ax.scatter(x_r, rem_vals.iloc[0], color='black', alpha=0.5, s=20, zorder=3)
                    ax.scatter(x_f, forg_vals.iloc[0], color='black', alpha=0.5, s=20, zorder=3)

        ax.set_xticks(x_base)
        ax.set_xticklabels(roi_order, fontsize=14, fontweight='bold')
        ax.tick_params(axis='y', labelsize=14, width=2, length=6)
        ax.axhline(0, color='grey')
        ax.set_ylabel(f'Mean Baseline-Corrected {measure_label}', fontsize=16, fontweight='bold')
        fig.suptitle(f'{label} {phase_label} Endogenous {measure_label} — Remembered vs Forgotten (NoStim)',
                     fontsize=20, fontweight='bold', y=0.97)
        ax.set_title(band, fontsize=18, fontweight='bold')
        legend_handles = [
            Line2D([0], [0], color=MEMORY_COLORS['Remembered'], lw=8, label='Remembered'),
            Line2D([0], [0], color=MEMORY_COLORS['Forgotten'], lw=8, label='Forgotten'),
        ]
        fig.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, 0.92),
                   ncol=2, frameon=False, fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.86])
        plt.savefig(os.path.join(out_dir,
                    band_filename(f'Endogenous_PairedBars_RemForg_{measure_label}_{phase_label.lower()}.png', band)),
                    dpi=300, bbox_inches='tight')
        finalize_figure(fig)


def generate_endogenous_plots(endo, out_dir, label, measure_label, phase_label, band_ranges=None):
    """Generate all endogenous memory plots for one measure and phase."""
    ensure_dir(out_dir)
    print(f"  Generating endogenous memory plots: {label} {phase_label} {measure_label}...")
    plot_overall_by_roi(endo, out_dir, label, measure_label, phase_label)
    plot_overall_by_patient(endo, out_dir, label, measure_label, phase_label)
    plot_remembered_vs_forgotten_spectra(endo, out_dir, label, measure_label, phase_label)
    plot_per_roi_rem_vs_forg_patients(endo, out_dir, label, measure_label, phase_label)
    plot_bc_rem_vs_forg_per_roi(endo, out_dir, label, measure_label, phase_label)
    if measure_label == 'PAC':
        plot_pac_memory_mi_difference_per_roi(endo, out_dir, label, phase_label)
    plot_bc_band_bargraph(endo, out_dir, label, measure_label, phase_label, band_ranges=band_ranges)
    plot_bc_paired_rem_forg_bars(endo, out_dir, label, measure_label, phase_label, band_ranges=band_ranges)


# ---------------------------------------------------------------------------
# MLMR CSV export — nostim only
# ---------------------------------------------------------------------------

def export_endogenous_mlmr_csv(data, csv_dir, file_stem, measure_name='Power'):
    """Export trial-level MLMR CSV filtered to nostim trials only."""
    frames = data.get('mlmr_export_frames', [])
    if not frames:
        print(f"  Skipping CSV export for {file_stem}: no export frames.")
        return None
    df = pd.concat(frames, ignore_index=True)
    df = df[df['trial_type'] == 'nostim'].copy()
    if df.empty:
        print(f"  Skipping CSV export for {file_stem}: no nostim trials.")
        return None
    df['Measure'] = measure_name
    base_cols = ['Measure', 'Patient', 'Region', 'trial_type', 'yes_or_no']
    freq_cols = sorted_freq_cols(df, 'diff_Freq_')
    df = df[base_cols + freq_cols].sort_values(base_cols).reset_index(drop=True)
    ensure_dir(csv_dir)
    csv_path = os.path.join(csv_dir, f'{file_stem}.csv')
    df.to_csv(csv_path, index=False)
    print(f"  Exported endogenous MLMR CSV: {csv_path}")
    return csv_path


# ---------------------------------------------------------------------------
# POWER analysis
# ---------------------------------------------------------------------------

def run_power_analysis():
    print("\n" + "=" * 60)
    print("Endogenous Memory — POWER")
    print("=" * 60)

    from combined_encoding_power import (
        load_blaes_encoding, load_amme_encoding,
        build_all_encoding_data, augment_with_power_composites,
        merge_dicts as pw_merge_dicts,
    )
    from combined_retrieval_power import (
        load_blaes_retrieval, load_amme_retrieval,
        build_all_retrieval_data,
    )

    # --- Encoding ---
    print("\n  Loading encoding power data...")
    blaes_enc = load_blaes_encoding()
    amme_enc = load_amme_encoding()
    all_enc = build_all_encoding_data(blaes_enc, amme_enc)

    for group_label, data in [('BLAES', blaes_enc), ('AMME', amme_enc), ('All', all_enc)]:
        endo = extract_endogenous(data, measure='power')
        out = ensure_dir(os.path.join(OUTPUT_ROOT, 'encoding_power', group_label.lower()))
        generate_endogenous_plots(endo, out, group_label, 'Power', 'Encoding')

    export_endogenous_mlmr_csv(blaes_enc, CSV_OUTPUT_DIR, 'endogenous_encoding_power_blaes_mlmr_input')
    export_endogenous_mlmr_csv(amme_enc, CSV_OUTPUT_DIR, 'endogenous_encoding_power_amme_mlmr_input')
    export_endogenous_mlmr_csv(all_enc, CSV_OUTPUT_DIR, 'endogenous_encoding_power_all_mlmr_input')

    # --- Retrieval ---
    print("\n  Loading retrieval power data...")
    blaes_ret = load_blaes_retrieval()
    amme_ret = load_amme_retrieval()
    all_ret = build_all_retrieval_data(blaes_ret, amme_ret)

    for group_label, data in [('BLAES', blaes_ret), ('AMME', amme_ret), ('All', all_ret)]:
        endo = extract_endogenous(data, measure='power')
        out = ensure_dir(os.path.join(OUTPUT_ROOT, 'retrieval_power', group_label.lower()))
        generate_endogenous_plots(endo, out, group_label, 'Power', 'Retrieval')

    export_endogenous_mlmr_csv(blaes_ret, CSV_OUTPUT_DIR, 'endogenous_retrieval_power_blaes_mlmr_input')
    export_endogenous_mlmr_csv(amme_ret, CSV_OUTPUT_DIR, 'endogenous_retrieval_power_amme_mlmr_input')
    export_endogenous_mlmr_csv(all_ret, CSV_OUTPUT_DIR, 'endogenous_retrieval_power_all_mlmr_input')

    print("  Power analysis complete.")


# ---------------------------------------------------------------------------
# COHERENCE analysis
# ---------------------------------------------------------------------------

def _merge_coherence_data(blaes, amme):
    """Merge BLAES and AMME coherence data dicts (mirrors inline logic from coherence scripts)."""
    all_freqs_post = None
    if blaes['freqs_post'] is not None and amme['freqs_post'] is not None:
        all_freqs_post = blaes['freqs_post'] if np.array_equal(blaes['freqs_post'], amme['freqs_post']) else blaes['freqs_post']
    else:
        all_freqs_post = blaes['freqs_post'] if blaes['freqs_post'] is not None else amme['freqs_post']

    all_freqs_diff = None
    if blaes['freqs_diff'] is not None and amme['freqs_diff'] is not None:
        all_freqs_diff = blaes['freqs_diff'] if np.array_equal(blaes['freqs_diff'], amme['freqs_diff']) else blaes['freqs_diff']
    else:
        all_freqs_diff = blaes['freqs_diff'] if blaes['freqs_diff'] is not None else amme['freqs_diff']

    return {
        'group_all_power': merge_dicts(blaes.get('group_all_power', {}), amme.get('group_all_power', {})),
        'stim_power': merge_dicts(blaes.get('stim_power', {}), amme.get('stim_power', {})),
        'nostim_power': merge_dicts(blaes.get('nostim_power', {}), amme.get('nostim_power', {})),
        'bc_stim': merge_dicts(blaes.get('bc_stim', {}), amme.get('bc_stim', {})),
        'bc_nostim': merge_dicts(blaes.get('bc_nostim', {}), amme.get('bc_nostim', {})),
        'freqs_post': all_freqs_post,
        'freqs_diff': all_freqs_diff,
        'has_memory': True,
        'stim_rem': merge_dicts(blaes.get('stim_rem', {}), amme.get('stim_rem', {})),
        'stim_forg': merge_dicts(blaes.get('stim_forg', {}), amme.get('stim_forg', {})),
        'nostim_rem': merge_dicts(blaes.get('nostim_rem', {}), amme.get('nostim_rem', {})),
        'nostim_forg': merge_dicts(blaes.get('nostim_forg', {}), amme.get('nostim_forg', {})),
        'bc_stim_rem': merge_dicts(blaes.get('bc_stim_rem', {}), amme.get('bc_stim_rem', {})),
        'bc_stim_forg': merge_dicts(blaes.get('bc_stim_forg', {}), amme.get('bc_stim_forg', {})),
        'bc_nostim_rem': merge_dicts(blaes.get('bc_nostim_rem', {}), amme.get('bc_nostim_rem', {})),
        'bc_nostim_forg': merge_dicts(blaes.get('bc_nostim_forg', {}), amme.get('bc_nostim_forg', {})),
        'mlmr_export_frames': [
            df.copy() for df in blaes.get('mlmr_export_frames', []) + amme.get('mlmr_export_frames', [])
        ],
    }


def run_coherence_analysis():
    print("\n" + "=" * 60)
    print("Endogenous Memory — COHERENCE")
    print("=" * 60)

    from combined_encoding_coherence import (
        load_blaes_encoding as load_blaes_encoding_coh,
        load_amme_encoding as load_amme_encoding_coh,
    )
    from combined_retrieval_coherence import (
        load_blaes_retrieval as load_blaes_retrieval_coh,
        load_amme_retrieval as load_amme_retrieval_coh,
    )

    # --- Encoding ---
    print("\n  Loading encoding coherence data...")
    blaes_enc = load_blaes_encoding_coh()
    amme_enc = load_amme_encoding_coh()
    all_enc = _merge_coherence_data(blaes_enc, amme_enc)

    for group_label, data in [('BLAES', blaes_enc), ('AMME', amme_enc), ('All', all_enc)]:
        endo = extract_endogenous(data, measure='coherence')
        out = ensure_dir(os.path.join(OUTPUT_ROOT, 'encoding_coherence', group_label.lower()))
        generate_endogenous_plots(endo, out, group_label, 'Coherence', 'Encoding')

    export_endogenous_mlmr_csv(blaes_enc, CSV_OUTPUT_DIR, 'endogenous_encoding_coherence_blaes_mlmr_input',
                               measure_name='Coherence')
    export_endogenous_mlmr_csv(amme_enc, CSV_OUTPUT_DIR, 'endogenous_encoding_coherence_amme_mlmr_input',
                               measure_name='Coherence')
    export_endogenous_mlmr_csv(all_enc, CSV_OUTPUT_DIR, 'endogenous_encoding_coherence_all_mlmr_input',
                               measure_name='Coherence')

    # --- Retrieval ---
    print("\n  Loading retrieval coherence data...")
    blaes_ret = load_blaes_retrieval_coh()
    amme_ret = load_amme_retrieval_coh()
    all_ret = _merge_coherence_data(blaes_ret, amme_ret)

    for group_label, data in [('BLAES', blaes_ret), ('AMME', amme_ret), ('All', all_ret)]:
        endo = extract_endogenous(data, measure='coherence')
        out = ensure_dir(os.path.join(OUTPUT_ROOT, 'retrieval_coherence', group_label.lower()))
        generate_endogenous_plots(endo, out, group_label, 'Coherence', 'Retrieval')

    export_endogenous_mlmr_csv(blaes_ret, CSV_OUTPUT_DIR, 'endogenous_retrieval_coherence_blaes_mlmr_input',
                               measure_name='Coherence')
    export_endogenous_mlmr_csv(amme_ret, CSV_OUTPUT_DIR, 'endogenous_retrieval_coherence_amme_mlmr_input',
                               measure_name='Coherence')
    export_endogenous_mlmr_csv(all_ret, CSV_OUTPUT_DIR, 'endogenous_retrieval_coherence_all_mlmr_input',
                               measure_name='Coherence')

    print("  Coherence analysis complete.")


# ---------------------------------------------------------------------------
# PAC analysis
# ---------------------------------------------------------------------------

def run_pac_analysis():
    print("\n" + "=" * 60)
    print("Endogenous Memory — PAC")
    print("=" * 60)

    from combined_encoding_pac import load_grouped_encoding_pac_data
    from combined_retrieval_pac import load_grouped_retrieval_pac_data

    # --- Encoding ---
    print("\n  Loading encoding PAC data...")
    grouped_enc = load_grouped_encoding_pac_data()
    all_enc = grouped_enc['all']

    endo = extract_endogenous(all_enc, measure='pac')
    out = ensure_dir(os.path.join(OUTPUT_ROOT, 'encoding_pac', 'all'))
    generate_endogenous_plots(endo, out, 'All', 'PAC', 'Encoding', band_ranges=PAC_BANDS)

    # --- Retrieval ---
    print("\n  Loading retrieval PAC data...")
    grouped_ret = load_grouped_retrieval_pac_data()
    all_ret = grouped_ret['all']

    endo = extract_endogenous(all_ret, measure='pac')
    out = ensure_dir(os.path.join(OUTPUT_ROOT, 'retrieval_pac', 'all'))
    generate_endogenous_plots(endo, out, 'All', 'PAC', 'Retrieval', band_ranges=PAC_BANDS)

    print("  PAC analysis complete.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Endogenous Memory Analysis")
    print("Remembered vs Forgotten — NoStim Trials Only")
    print("=" * 60)

    ensure_dir(OUTPUT_ROOT)
    ensure_dir(CSV_OUTPUT_DIR)

    run_power_analysis()
    run_coherence_analysis()
    run_pac_analysis()

    print("\n" + "=" * 60)
    print(f"Done! All endogenous memory outputs saved to: {OUTPUT_ROOT}")
    print("=" * 60)


if __name__ == '__main__':
    main()
