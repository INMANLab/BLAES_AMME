#!/usr/bin/env python
"""
Balanced Memory Trials Analysis
================================
Reruns all endogenous memory analyses (remembered vs forgotten, NoStim only)
but EXCLUDES any subject who has fewer than 10 trials in either the remembered
or forgotten condition. This prevents skewed effects from imbalanced trial counts.

Covers power, coherence, and PAC for both encoding and retrieval.
"""

# --- AMME_BLAES sys.path bootstrap ---
import sys as _sys
from pathlib import Path as _Path
_root = _Path(__file__).resolve().parent.parent
for _p in (_root, _root / 'encoding', _root / 'retrieval', _root / 'endogenous_memory',
          _root / 'regressions', _root / 'behavioral', _root / 'balanced_memory'):
    if str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))
# --- end bootstrap ---

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

OUTPUT_ROOT = os.path.join(PROJECT_ROOT, 'outputs', 'balanced_memory_trials')
CSV_OUTPUT_DIR = os.path.join(OUTPUT_ROOT, 'csvs')
SUMMARY_OUTPUT_PATH = os.path.join(OUTPUT_ROOT, 'balanced_memory_conditions_summary.txt')

MIN_TRIALS_PER_CONDITION = 10

from endogenous_memory import (
    ensure_dir,
    extract_endogenous,
    generate_endogenous_plots,
    export_endogenous_mlmr_csv,
    filter_pnas_regions,
    PAC_BANDS,
    POWER_RANGES,
    ROI_COLORS_SPEC,
    ROI_COLORS_BAR,
    MEMORY_COLORS,
    finalize_figure,
    set_freq_xlim,
    band_filename,
    get_anchor_region,
    make_subject_color_map,
    collect_unique_legend_items,
)


def balanced_plot_label(group_label):
    return f'{group_label} Balanced Memory Conditions'


# ---------------------------------------------------------------------------
# Subject filtering logic
# ---------------------------------------------------------------------------

def get_nostim_trial_counts(data, measure='power'):
    """Count nostim remembered/forgotten trials per subject from mlmr export frames.

    For PAC data, trial counts come from the loaded data dict directly.
    """
    if measure == 'pac':
        return _count_from_data_dict(data)

    frames = data.get('mlmr_export_frames', [])
    if not frames:
        return {}

    df = pd.concat(frames, ignore_index=True)
    df = df[df['trial_type'] == 'nostim']
    if df.empty:
        return {}

    # Deduplicate across regions — pick first region per patient
    first_region = df.groupby('Patient')['Region'].first().to_dict()
    df = df[df.apply(lambda r: r['Region'] == first_region[r['Patient']], axis=1)]

    counts = {}
    for pat, grp in df.groupby('Patient'):
        n_rem = (grp['yes_or_no'] == 'yes').sum()
        n_forg = (grp['yes_or_no'] == 'no').sum()
        counts[pat] = {'remembered': int(n_rem), 'forgotten': int(n_forg)}
    return counts


def _count_from_data_dict(data):
    """Count subjects from PAC data dicts (no mlmr_export_frames for PAC)."""
    # Use nostim_rem and nostim_forg dicts — count subjects that appear
    # PAC keys: post_nostim_rem, post_nostim_forg
    rem_d = data.get('post_nostim_rem', {})
    forg_d = data.get('post_nostim_forg', {})

    all_subjects = set()
    for rd in [rem_d, forg_d]:
        for roi, sd in rd.items():
            all_subjects.update(sd.keys())

    # For PAC we don't have raw trial counts in the dict (already averaged).
    # We'll rely on the encoding/retrieval power trial counts to determine exclusions.
    # Return None to signal that PAC should use an external exclusion list.
    return None


def summarize_balanced_subjects(trial_counts, min_trials=MIN_TRIALS_PER_CONDITION):
    """Return inclusion/exclusion details for the balanced-memory filter."""
    if not trial_counts:
        return {
            'included': set(),
            'excluded': [],
            'total': 0,
            'min_trials': min_trials,
        }

    included = set()
    excluded = []
    for pat, cnts in sorted(trial_counts.items()):
        remembered = int(cnts['remembered'])
        forgotten = int(cnts['forgotten'])
        if remembered >= min_trials and forgotten >= min_trials:
            included.add(pat)
        else:
            excluded.append({
                'patient': pat,
                'remembered': remembered,
                'forgotten': forgotten,
            })

    return {
        'included': included,
        'excluded': excluded,
        'total': len(trial_counts),
        'min_trials': min_trials,
    }


def get_balanced_subjects(trial_counts, min_trials=MIN_TRIALS_PER_CONDITION):
    """Return set of subjects with >= min_trials in BOTH conditions."""
    summary = summarize_balanced_subjects(trial_counts, min_trials=min_trials)
    if summary['total'] == 0:
        return None
    balanced = summary['included']
    excluded = summary['excluded']
    print(f"    Balanced subjects: {len(balanced)} / {summary['total']} "
          f"(excluded {len(excluded)} with <{min_trials} trials in a condition)")
    for row in excluded:
        print(f"      Excluded: {row['patient']} (rem={row['remembered']}, forg={row['forgotten']})")
    return balanced


def get_region_subject_counts(data):
    """Count remaining unique subjects per region from a filtered data dict."""
    region_counts = {}
    for value in data.values():
        if not isinstance(value, dict) or not value:
            continue
        first_val = next(iter(value.values()), None)
        if not isinstance(first_val, dict):
            continue
        for region, subject_dict in value.items():
            region_counts.setdefault(region, set()).update(map(str, subject_dict.keys()))
    return {region: len(subjects) for region, subjects in sorted(region_counts.items())}


def format_region_counts(region_counts):
    if not region_counts:
        return ['None']
    return [f'{region}: {count}' for region, count in region_counts.items()]


def write_balanced_summary_report(summary_rows, out_path=SUMMARY_OUTPUT_PATH):
    lines = [
        'Balanced Memory Conditions Summary',
        '=' * 80,
        f'Criterion: keep subjects with at least {MIN_TRIALS_PER_CONDITION} NoStim remembered trials',
        f'and at least {MIN_TRIALS_PER_CONDITION} NoStim forgotten trials.',
        '',
    ]

    for row in summary_rows:
        lines.extend([
            f"{row['measure']} | {row['phase']} | {row['group']}",
            '-' * 80,
        ])
        if row.get('note'):
            lines.append(f"Note: {row['note']}")
        lines.extend([
            f"Included: {row['included_n']} / {row['total_n']}",
            f"Excluded: {row['excluded_n']}",
            'Excluded subjects:',
        ])
        if row['excluded']:
            for excluded_row in row['excluded']:
                lines.append(
                    f"  {excluded_row['patient']}: remembered={excluded_row['remembered']}, "
                    f"forgotten={excluded_row['forgotten']}"
                )
        else:
            lines.append('  None (see Power section for exclusion details)')
        lines.append('Remaining N across regions:')
        lines.extend([f'  {line}' for line in format_region_counts(row['region_counts'])])
        lines.append('')

    ensure_dir(os.path.dirname(out_path))
    with open(out_path, 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(lines).rstrip() + '\n')
    print(f'  Wrote balanced-memory summary: {out_path}')
    return out_path


def filter_subject_dicts(data, keep_subjects):
    """Filter any {region: {subject: values}} mappings to only keep_subjects."""
    if keep_subjects is None:
        return data
    filtered = {}
    for key, value in data.items():
        if isinstance(value, dict) and value:
            # Check if it's a {region: {subject: vector}} dict
            first_val = next(iter(value.values()), None)
            if isinstance(first_val, dict):
                filtered[key] = {
                    roi: {subj: vec for subj, vec in sd.items() if subj in keep_subjects}
                    for roi, sd in value.items()
                }
                # Remove empty regions
                filtered[key] = {k: v for k, v in filtered[key].items() if v}
            else:
                filtered[key] = value
        else:
            filtered[key] = value
    return filtered


def filter_endogenous_data(endo, keep_subjects):
    """Backward-compatible alias for filtering endogenous data dicts."""
    return filter_subject_dicts(endo, keep_subjects)


def filter_mlmr_frames(data, keep_subjects):
    """Return a copy of data with mlmr_export_frames filtered to keep_subjects."""
    if keep_subjects is None or 'mlmr_export_frames' not in data:
        return data
    filtered = dict(data)
    filtered['mlmr_export_frames'] = [
        df[df['Patient'].isin(keep_subjects)].copy()
        for df in data.get('mlmr_export_frames', [])
    ]
    return filtered


def add_balanced_stim_memory_plots(filtered_data, out_dir, label, common_plotter, memory_plotter):
    """Add the same stim/no-stim and stim x memory outputs used in the main scripts."""
    common_plotter(filtered_data, out_dir, label)
    memory_plotter(filtered_data, out_dir, label)


# ---------------------------------------------------------------------------
# Endogenous-only combined-style plots
# ---------------------------------------------------------------------------

ENCODING_BAND_RANGES = {'Theta': (4, 8), 'Slow gamma': (30, 55), 'HFA': (70, 150)}
CORE_REGIONS = {'BLA', 'CA', 'DG', 'EC', 'HPC', 'PRC'}
CORE_ROI_COLORS = {
    'BLA': 'teal', 'CA': 'black', 'DG': 'skyblue',
    'EC': '#8A2BE2', 'HPC': 'orange', 'PRC': '#E61C59',
}


def _extract_nostim_data(data, measure):
    """Extract nostim-only data from a full data dict."""
    if measure == 'pac':
        return {
            'overall': data.get('post_nostim', {}),
            'bc_overall': data.get('diff_nostim', {}),
            'rem': data.get('post_nostim_rem', {}),
            'forg': data.get('post_nostim_forg', {}),
            'bc_rem': data.get('diff_nostim_rem', {}),
            'bc_forg': data.get('diff_nostim_forg', {}),
            'freqs_post': data.get('freqs_post'),
            'freqs_diff': data.get('freqs_diff'),
        }
    return {
        'overall': data.get('nostim_power', data.get('nostim_coherence', {})),
        'bc_overall': data.get('bc_nostim', {}),
        'rem': data.get('nostim_rem', {}),
        'forg': data.get('nostim_forg', {}),
        'bc_rem': data.get('bc_nostim_rem', {}),
        'bc_forg': data.get('bc_nostim_forg', {}),
        'freqs_post': data.get('freqs_post'),
        'freqs_diff': data.get('freqs_diff'),
    }


def _endo_plot_bc_overall_spectra(nd, freqs, out_dir, label, measure_label, phase_label):
    """Baseline-corrected spectra per ROI (nostim only, collapsed across memory)."""
    if not nd or freqs is None:
        return
    for roi in sorted(nd.keys()):
        sd = nd[roi]
        if not sd:
            continue
        mat = np.array(list(sd.values()), dtype=np.float64)
        n = mat.shape[0]
        mean, se = mat.mean(0), mat.std(0) / np.sqrt(n)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(freqs, mean, color='#1f77b4', linewidth=2, label=f'NoStim ({n})')
        ax.fill_between(freqs, mean - se, mean + se, alpha=0.25, color='#1f77b4')
        ax.axhline(0, color='lightgray', linestyle='--', linewidth=1)
        ax.set_xlabel('Frequency (Hz)', fontsize=16, fontweight='bold')
        ax.set_ylabel(f'Baseline-Corrected {measure_label}', fontsize=16, fontweight='bold')
        ax.set_title(f'{label} {phase_label} {roi} BC {measure_label} (Endogenous Only)',
                     fontsize=16, fontweight='bold')
        set_freq_xlim(ax, freqs)
        ax.tick_params(axis='both', labelsize=14)
        ax.legend(fontsize=13, loc='best')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir,
                    f'EndoOnly_BC_{measure_label}_{roi}_{phase_label.lower()}.png'),
                    dpi=300, bbox_inches='tight')
        finalize_figure()


def _endo_plot_bc_overall_bargraph(nd, freqs, out_dir, label, measure_label,
                                   phase_label, band_ranges):
    """BC band-average bar graph per ROI (nostim only, collapsed across memory)."""
    if not nd or freqs is None:
        return
    rows = []
    for roi in sorted(nd.keys()):
        for subj, vec in nd[roi].items():
            arr = np.asarray(vec, dtype=np.float64)
            for bn, (lo, hi) in band_ranges.items():
                mask = (freqs >= lo) & (freqs <= hi)
                rows.append({
                    'Patient': subj, 'Region': roi,
                    'power_range': bn, 'value': arr[mask].mean(),
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

        fig, ax = plt.subplots(figsize=(11.5, 8.2))
        x = np.arange(len(roi_order))
        summary = band_df.groupby('Region')['value'].agg(['mean', 'sem']).reindex(roi_order)
        gray_vals = np.linspace(0.85, 0.45, len(roi_order))
        bar_colors = {roi: matplotlib.colors.to_hex((g, g, g))
                      for roi, g in zip(roi_order, gray_vals)}
        ax.bar(x, summary['mean'].to_numpy(),
               yerr=summary['sem'].fillna(0).to_numpy(),
               color=[bar_colors[r] for r in roi_order],
               edgecolor='#4D4D4D', linewidth=1.2, width=0.72, capsize=3,
               ecolor='#4D4D4D', zorder=1)
        rng = np.random.default_rng(7)
        for idx, roi in enumerate(roi_order):
            roi_df = band_df[band_df['Region'] == roi]
            jitter = rng.uniform(-0.16, 0.16, len(roi_df))
            ax.scatter(np.full(len(roi_df), idx, dtype=float) + jitter,
                       roi_df['value'], c='black', s=40, alpha=0.6,
                       edgecolors='none', zorder=2)
        ax.set_xticks(x)
        ax.set_xticklabels(roi_order, fontsize=14, fontweight='bold')
        ax.tick_params(axis='y', labelsize=14)
        ax.axhline(0, color='grey', linewidth=0.8)
        ax.set_ylabel(f'Baseline-Corrected {measure_label}', fontsize=16, fontweight='bold')
        ax.set_title(band, fontsize=18, fontweight='bold')
        fig.suptitle(f'{label} {phase_label} BC {measure_label} by ROI (Endogenous Only)',
                     fontsize=20, fontweight='bold', y=0.98)
        fig.tight_layout(rect=[0, 0, 1, 0.9])
        plt.savefig(os.path.join(out_dir,
                    band_filename(f'EndoOnly_BC_Bargraph_{measure_label}_{phase_label.lower()}.png', band)),
                    dpi=300, bbox_inches='tight')
        finalize_figure(fig)


def _endo_plot_core_regions(overall, freqs, out_dir, label, measure_label, phase_label):
    """Overall spectra limited to core regions (nostim only)."""
    if not overall or freqs is None:
        return
    has_core = any(roi in CORE_REGIONS for roi in overall)
    if not has_core:
        return
    fig, ax = plt.subplots(figsize=(12, 8))
    for roi in sorted(overall.keys()):
        if roi not in CORE_REGIONS:
            continue
        mat = np.array(list(overall[roi].values()), dtype=np.float64)
        mean, std = mat.mean(0), mat.std(0)
        c = CORE_ROI_COLORS.get(roi, 'gray')
        ax.plot(freqs, mean, color=c, label=f'{roi} ({mat.shape[0]})')
        ax.fill_between(freqs, mean - std, mean + std, alpha=0.2, color=c)
    ax.set_xlabel('Frequency (Hz)', fontsize=18, fontweight='bold')
    ax.set_ylabel(f'{measure_label}', fontsize=18, fontweight='bold')
    ax.set_title(f'{label} {phase_label} {measure_label} by ROI — Core Regions (Endogenous Only)',
                 fontsize=18, fontweight='bold')
    ax.tick_params(axis='both', labelsize=14)
    set_freq_xlim(ax, freqs)
    ax.legend(bbox_to_anchor=(1.02, 0.5), loc='center left',
              prop={'weight': 'bold', 'size': 12})
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(os.path.join(out_dir,
                f'EndoOnly_{measure_label}_CoreRegions_{phase_label.lower()}.png'),
                dpi=300, bbox_inches='tight')
    finalize_figure()


def _endo_plot_connected_dots_memory(bc_rem, bc_forg, freqs, out_dir, label,
                                     measure_label, phase_label, band_ranges,
                                     show_patients=True):
    """Connected-dots plot: remembered vs forgotten per patient (nostim only)."""
    if not bc_rem or not bc_forg or freqs is None:
        return
    all_rows = []
    all_rois = sorted(set(bc_rem.keys()) | set(bc_forg.keys()))
    for roi in all_rois:
        for subj in sorted(set(bc_rem.get(roi, {})) & set(bc_forg.get(roi, {}))):
            rem_vec = np.asarray(bc_rem[roi][subj], dtype=np.float64)
            forg_vec = np.asarray(bc_forg[roi][subj], dtype=np.float64)
            for bn, (lo, hi) in band_ranges.items():
                mask = (freqs >= lo) & (freqs <= hi)
                all_rows.append({
                    'Patient': subj, 'Region': roi, 'power_range': bn,
                    'remembered': rem_vec[mask].mean(),
                    'forgotten': forg_vec[mask].mean(),
                })
    if not all_rows:
        return
    df = pd.DataFrame(all_rows)
    clean_suffix = '_clean' if not show_patients else ''

    if show_patients:
        patients = sorted(df['Patient'].unique())
        filled_markers = ['o', 's', '^', 'D', 'v', 'P', 'X', '*', '<', '>', 'h', '8', 'p', 'H', 'd']
        marker_styles = (
            [(m, True) for m in filled_markers]
            + [(m, False) for m in filled_markers]
            + [('1', True), ('2', True), ('3', True), ('4', True),
               ('+', True), ('x', True), ('|', True), ('_', True)]
            + [(f'${i}$', True) for i in range(1, 11)]
        )
        patient_markers = {p: marker_styles[i] for i, p in enumerate(patients)}

    for band in band_ranges:
        df_band = df[df['power_range'] == band]
        if df_band.empty:
            continue
        # Use anchor-based grouping for coherence/PAC
        if measure_label in ('Coherence', 'PAC'):
            df_band = df_band.copy()
            df_band['anchor'] = df_band['Region'].map(get_anchor_region)
            for anchor in sorted(df_band['anchor'].unique()):
                sub = df_band[df_band['anchor'] == anchor]
                roi_order = sorted(sub['Region'].unique())
                _draw_endo_connected_dots(
                    sub, roi_order, band, out_dir, label, measure_label,
                    phase_label, show_patients,
                    patient_markers if show_patients else None,
                    clean_suffix, anchor_tag=anchor,
                )
        else:
            roi_order = [r for r in ROI_COLORS_BAR if r in df_band['Region'].unique()]
            if not roi_order:
                roi_order = sorted(df_band['Region'].unique())
            _draw_endo_connected_dots(
                df_band, roi_order, band, out_dir, label, measure_label,
                phase_label, show_patients,
                patient_markers if show_patients else None,
                clean_suffix,
            )


def _draw_endo_connected_dots(df_band, roi_order, band, out_dir, label,
                              measure_label, phase_label, show_patients,
                              patient_markers, clean_suffix, anchor_tag=None):
    """Draw a single connected-dots figure for endogenous rem vs forg."""
    if df_band.empty or not roi_order:
        return
    fig, ax = plt.subplots(figsize=(16, 11))
    x_base = np.arange(len(roi_order))
    bar_width = 0.34
    offsets = {'remembered': -bar_width / 2, 'forgotten': bar_width / 2}

    for i, roi in enumerate(roi_order):
        roi_df = df_band[df_band['Region'] == roi]
        for cond in ['remembered', 'forgotten']:
            vals = roi_df[cond].dropna()
            if len(vals) == 0:
                continue
            mean = vals.mean()
            sem = vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0
            ax.bar(x_base[i] + offsets[cond], mean, width=bar_width,
                   color=MEMORY_COLORS[cond.capitalize()],
                   edgecolor='black', linewidth=1.2, yerr=sem, capsize=4, zorder=1)

    if show_patients and patient_markers:
        for i, roi in enumerate(roi_order):
            roi_df = df_band[df_band['Region'] == roi]
            for _, row in roi_df.iterrows():
                x_r = x_base[i] + offsets['remembered']
                x_f = x_base[i] + offsets['forgotten']
                ax.plot([x_r, x_f], [row['remembered'], row['forgotten']],
                        color='black', alpha=0.35, linewidth=1)
                m, filled = patient_markers[row['Patient']]
                for x_pt, val in [(x_r, row['remembered']), (x_f, row['forgotten'])]:
                    if filled:
                        ax.scatter(x_pt, val, marker=m, color='black',
                                   alpha=0.65, s=55, zorder=3)
                    else:
                        ax.scatter(x_pt, val, marker=m, facecolors='white',
                                   edgecolors='black', linewidths=1.1,
                                   alpha=0.85, s=60, zorder=3)

    ax.set_xticks(x_base)
    ax.set_xticklabels(roi_order, fontsize=14, fontweight='bold', rotation=45 if len(roi_order) > 8 else 0, ha='right' if len(roi_order) > 8 else 'center')
    ax.tick_params(axis='y', labelsize=14, width=2, length=6)
    ax.axhline(0, color='grey')
    ax.set_ylabel(f'Mean BC {measure_label}', fontsize=17, fontweight='bold')
    fig.suptitle(f'{band} — Remembered vs Forgotten (Endogenous Only)',
                 fontsize=22, fontweight='bold', y=0.965)

    legend_handles = [
        Line2D([0], [0], color=MEMORY_COLORS['Remembered'], lw=8, label='Remembered'),
        Line2D([0], [0], color=MEMORY_COLORS['Forgotten'], lw=8, label='Forgotten'),
    ]
    fig.legend(handles=legend_handles, loc='upper center',
               bbox_to_anchor=(0.50, 0.925), ncol=2, frameon=False,
               fontsize=12, title='Memory', title_fontsize=13)

    if show_patients and patient_markers:
        patient_handles = [
            Line2D([0], [0], marker=patient_markers[p][0], color='black',
                   linestyle='None',
                   markerfacecolor='black' if patient_markers[p][1] else 'white',
                   markeredgecolor='black', markeredgewidth=1.1, markersize=8,
                   label=str(p))
            for p in sorted(df_band['Patient'].unique())
        ]
        fig.legend(handles=patient_handles, loc='center left',
                   bbox_to_anchor=(0.81, 0.50), ncol=2, columnspacing=0.9,
                   handletextpad=0.4, labelspacing=0.45, frameon=False,
                   fontsize=10, title='Patient', title_fontsize=12)

    fig.subplots_adjust(left=0.10, right=0.78 if show_patients else 0.95,
                        bottom=0.15, top=0.86)
    tag = f'_{anchor_tag}' if anchor_tag else ''
    plt.savefig(os.path.join(out_dir,
                band_filename(f'EndoOnly_RemVsForg{tag}_{phase_label.lower()}{clean_suffix}.png', band)),
                bbox_inches='tight', dpi=300)
    finalize_figure()


def _endo_plot_freq_x_memory_bargraph(bc_rem, bc_forg, freqs, out_dir, label,
                                      measure_label, phase_label, band_ranges):
    """Frequency x Memory bargraph: multiple bands side-by-side, rem vs forg dots (nostim only)."""
    if not bc_rem or not bc_forg or freqs is None:
        return
    rows = []
    all_rois = sorted(set(bc_rem.keys()) & set(bc_forg.keys()))
    for roi in all_rois:
        shared = sorted(set(bc_rem[roi]) & set(bc_forg[roi]))
        for subj in shared:
            rem_vec = np.asarray(bc_rem[roi][subj], dtype=np.float64)
            forg_vec = np.asarray(bc_forg[roi][subj], dtype=np.float64)
            for bn, (lo, hi) in band_ranges.items():
                mask = (freqs >= lo) & (freqs <= hi)
                rows.append({
                    'Patient': subj, 'Region': roi, 'power_range': bn,
                    'memory_cond': 'Remembered',
                    'value': rem_vec[mask].mean(),
                })
                rows.append({
                    'Patient': subj, 'Region': roi, 'power_range': bn,
                    'memory_cond': 'Forgotten',
                    'value': forg_vec[mask].mean(),
                })
    if not rows:
        return
    df = pd.DataFrame(rows)

    # For coherence/PAC: split by anchor region
    if measure_label in ('Coherence', 'PAC'):
        df['anchor'] = df['Region'].map(get_anchor_region)
        for anchor in sorted(df['anchor'].unique()):
            sub = df[df['anchor'] == anchor]
            _draw_endo_freq_x_memory(sub, out_dir, label, measure_label,
                                     phase_label, band_ranges, anchor_tag=anchor)
    else:
        core_df = df[df['Region'].isin(CORE_REGIONS)]
        if not core_df.empty:
            _draw_endo_freq_x_memory(core_df, out_dir, label, measure_label,
                                     phase_label, band_ranges)
        # Also full version
        _draw_endo_freq_x_memory(df, out_dir, label, measure_label,
                                 phase_label, band_ranges, anchor_tag='all')


def _draw_endo_freq_x_memory(df, out_dir, label, measure_label, phase_label,
                             band_ranges, anchor_tag=None):
    """Draw a single frequency x memory bargraph for endogenous data."""
    if df.empty:
        return
    roi_order = sorted(df['Region'].unique())
    band_order = [b for b in band_ranges if b in df['power_range'].unique()]
    if not band_order or not roi_order:
        return

    n_rois = len(roi_order)
    n_bands = len(band_order)
    bar_width = 0.25
    x = np.arange(n_rois)
    band_colors = {'Theta': '#4C72B0', 'Slow gamma': '#DD8452', 'HFA': '#55A868'}

    fig, ax = plt.subplots(figsize=(max(11.5, n_rois * 1.8), 8.2))
    for i, band in enumerate(band_order):
        band_df = df[df['power_range'] == band]
        means, sems = [], []
        for roi in roi_order:
            roi_vals = band_df[band_df['Region'] == roi]['value']
            means.append(roi_vals.mean() if len(roi_vals) else 0)
            sems.append(roi_vals.sem() if len(roi_vals) > 1 else 0)
        offset = (i - (n_bands - 1) / 2) * bar_width
        ax.bar(x + offset, means, bar_width, yerr=sems, capsize=4,
               color=band_colors.get(band, 'gray'), label=band, alpha=0.85,
               edgecolor='black', linewidth=0.5)

        for mem_label, fc, ec, alpha in [
            ('Remembered', 'black', 'black', 0.7),
            ('Forgotten', 'white', 'black', 0.9),
        ]:
            sub = band_df[band_df['memory_cond'] == mem_label]
            for j, roi in enumerate(roi_order):
                pts = sub[sub['Region'] == roi]['value']
                if pts.empty:
                    continue
                jitter = np.random.default_rng(42).uniform(
                    -bar_width * 0.3, bar_width * 0.3, len(pts))
                ax.scatter(x[j] + offset + jitter, pts, marker='o', s=30,
                           facecolors=fc, edgecolors=ec, linewidths=1.2,
                           zorder=5, alpha=alpha)

    ax.set_xticks(x)
    ax.set_xticklabels(roi_order, fontsize=14, fontweight='bold',
                       rotation=45 if n_rois > 8 else 0,
                       ha='right' if n_rois > 8 else 'center')
    ax.set_ylabel(f'BC {measure_label} (Endogenous)', fontsize=16, fontweight='bold')
    ax.tick_params(axis='y', labelsize=14)
    ax.axhline(0, color='grey', linewidth=0.8)

    band_handles = [plt.Rectangle((0, 0), 1, 1, fc=band_colors.get(b, 'gray'),
                                  ec='black', lw=0.5) for b in band_order]
    mem_handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='black',
               markeredgecolor='black', markersize=7, label='Remembered'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='white',
               markeredgecolor='black', markeredgewidth=1.2, markersize=7,
               label='Forgotten'),
    ]
    ax.legend(handles=band_handles + mem_handles,
              labels=band_order + ['Remembered', 'Forgotten'],
              loc='lower right', fontsize=11, framealpha=0.9)

    tag = f' ({anchor_tag})' if anchor_tag else ''
    fig.suptitle(f'{label} {phase_label} Freq × Memory{tag} (Endogenous Only)',
                 fontsize=20, fontweight='bold', y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    file_tag = f'_{anchor_tag}' if anchor_tag else ''
    plt.savefig(os.path.join(out_dir,
                f'EndoOnly_FreqXMemory_{measure_label}{file_tag}_{phase_label.lower()}.png'),
                dpi=300, bbox_inches='tight')
    finalize_figure(fig)


def _endo_plot_quadrant_memory(rem, forg, freqs, out_dir, label, measure_label,
                               phase_label):
    """2-panel (rem vs forg) spectra by ROI (nostim only, raw post)."""
    if not rem and not forg or freqs is None:
        return
    all_rois = sorted(set(rem.keys()) | set(forg.keys()))
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharex=True, sharey=True)
    for ax, (title, rd) in zip(axes, [('Remembered', rem), ('Forgotten', forg)]):
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
    axes[1].legend(bbox_to_anchor=(1.02, 0.5), loc='center left',
                   prop={'weight': 'bold', 'size': 11})
    fig.suptitle(f'{label} {phase_label} {measure_label}: Rem vs Forg (Endogenous Only)',
                 fontsize=18, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 0.88, 0.95])
    plt.savefig(os.path.join(out_dir,
                f'EndoOnly_RemVsForg_{measure_label}_{phase_label.lower()}.png'),
                dpi=300, bbox_inches='tight')
    finalize_figure()


def _endo_plot_per_roi_patients(rem, forg, freqs, out_dir, label,
                                measure_label, phase_label):
    """Per-ROI patient-level rem vs forg panels (nostim only)."""
    if not rem and not forg or freqs is None:
        return
    all_rois = sorted(set(rem.keys()) | set(forg.keys()))
    for roi in all_rois:
        roi_subjects = set(rem.get(roi, {})) | set(forg.get(roi, {}))
        if not roi_subjects:
            continue
        subject_colors = make_subject_color_map(roi_subjects)
        fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharex=True, sharey=True)
        for ax, (title, rd) in zip(axes, [
            (f'{roi} — Remembered', rem),
            (f'{roi} — Forgotten', forg),
        ]):
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
            fig.legend(handles, labels_, bbox_to_anchor=(0.88, 0.5),
                       loc='center left', prop={'weight': 'bold', 'size': 10})
        fig.suptitle(f'{label} {phase_label} {roi} by Patient (Endogenous Only)',
                     fontsize=18, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 0.82, 0.95])
        plt.savefig(os.path.join(out_dir,
                    f'EndoOnly_{measure_label}_{roi}_byPatient_{phase_label.lower()}.png'),
                    dpi=300, bbox_inches='tight')
        finalize_figure()


def _endo_plot_bc_quadrant_memory(bc_rem, bc_forg, freqs, out_dir, label,
                                  measure_label, phase_label):
    """2-panel baseline-corrected rem vs forg spectra (nostim only)."""
    if not bc_rem and not bc_forg or freqs is None:
        return
    all_rois = sorted(set(bc_rem.keys()) | set(bc_forg.keys()))
    gspr = {}
    for cd in [bc_rem, bc_forg]:
        for roi, sd in cd.items():
            gspr.setdefault(roi, set()).update(sd.keys())

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharex=True, sharey=True)
    plotted = set()
    for ax, (title, rd) in zip(axes, [('Remembered', bc_rem), ('Forgotten', bc_forg)]):
        for roi in all_rois:
            if roi not in rd:
                continue
            mat = np.array(list(rd[roi].values()), dtype=np.float64)
            if mat.ndim != 2 or mat.shape[0] == 0:
                continue
            mean = np.nanmean(mat, 0)
            std = np.nanstd(mat, 0)
            c = ROI_COLORS_SPEC.get(roi, 'gray')
            lbl = None
            if roi not in plotted:
                lbl = f'{roi} ({len(gspr.get(roi, set()))})'
                plotted.add(roi)
            ax.plot(freqs, mean, color=c, linewidth=2, label=lbl)
            ax.fill_between(freqs, mean - std, mean + std, color=c, alpha=0.2)
        ax.set_title(title, fontsize=18, fontweight='bold')
        ax.tick_params(axis='both', labelsize=14)
        ax.axhline(0, color='lightgray', linestyle='--', linewidth=1)
    axes[0].set_ylabel(f'BC {measure_label}', fontsize=16, fontweight='bold')
    fig.text(0.5, 0.04, 'Frequency (Hz)', ha='center', fontsize=18, fontweight='bold')
    fig.suptitle(f'{label} {phase_label} BC {measure_label} Rem vs Forg (Endogenous Only)',
                 fontsize=18, fontweight='bold')
    handles, labels_ = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        for hh, ll in zip(h, l):
            if ll not in labels_:
                handles.append(hh)
                labels_.append(ll)
    fig.legend(handles, labels_, bbox_to_anchor=(0.88, 0.5), loc='center left',
               prop={'weight': 'bold', 'size': 11})
    plt.tight_layout(rect=[0, 0, 0.85, 0.95])
    plt.savefig(os.path.join(out_dir,
                f'EndoOnly_BC_RemVsForg_{measure_label}_{phase_label.lower()}.png'),
                dpi=300, bbox_inches='tight')
    finalize_figure()


def generate_endogenous_only_combined_plots(data, out_dir, label, measure,
                                            phase, band_ranges=None):
    """Generate combined-style figures using only endogenous (nostim) data.

    Produces the same types of figures as the combined-script
    generate_common_plots / generate_memory_plots but restricted to
    nostim trials.  Outputs go to an ``endogenous_only`` subdirectory.
    """
    endo_dir = ensure_dir(os.path.join(out_dir, 'endogenous_only'))
    nd = _extract_nostim_data(data, measure)
    measure_label = {'power': 'Power', 'coherence': 'Coherence', 'pac': 'PAC'}[measure]
    phase_label = phase.capitalize()
    if band_ranges is None:
        band_ranges = PAC_BANDS if measure == 'pac' else POWER_RANGES

    print(f"  Generating endogenous-only combined plots: {label} {phase_label} {measure_label}...")

    # --- Common-style plots ---
    # Core-region spectra (nostim only)
    _endo_plot_core_regions(nd['overall'], nd['freqs_post'], endo_dir,
                           label, measure_label, phase_label)
    # BC overall spectra per ROI
    _endo_plot_bc_overall_spectra(nd['bc_overall'], nd['freqs_diff'], endo_dir,
                                 label, measure_label, phase_label)
    # BC overall bargraph per ROI per band
    _endo_plot_bc_overall_bargraph(nd['bc_overall'], nd['freqs_diff'], endo_dir,
                                  label, measure_label, phase_label, band_ranges)

    # --- Memory-style plots ---
    # 2-panel rem vs forg spectra (raw)
    _endo_plot_quadrant_memory(nd['rem'], nd['forg'], nd['freqs_post'],
                               endo_dir, label, measure_label, phase_label)
    # Per-ROI patient-level rem vs forg
    _endo_plot_per_roi_patients(nd['rem'], nd['forg'], nd['freqs_post'],
                                endo_dir, label, measure_label, phase_label)
    # BC 2-panel rem vs forg
    _endo_plot_bc_quadrant_memory(nd['bc_rem'], nd['bc_forg'], nd['freqs_diff'],
                                  endo_dir, label, measure_label, phase_label)
    # Connected dots — rem vs forg
    _endo_plot_connected_dots_memory(nd['bc_rem'], nd['bc_forg'], nd['freqs_diff'],
                                    endo_dir, label, measure_label, phase_label,
                                    band_ranges)
    _endo_plot_connected_dots_memory(nd['bc_rem'], nd['bc_forg'], nd['freqs_diff'],
                                    endo_dir, label, measure_label, phase_label,
                                    band_ranges, show_patients=False)
    # Frequency x memory bargraph
    _endo_plot_freq_x_memory_bargraph(nd['bc_rem'], nd['bc_forg'], nd['freqs_diff'],
                                     endo_dir, label, measure_label, phase_label,
                                     band_ranges)


# ---------------------------------------------------------------------------
# POWER
# ---------------------------------------------------------------------------

def run_power_analysis():
    print("\n" + "=" * 60)
    print("Balanced Memory Trials — POWER")
    print("=" * 60)

    from combined_encoding_power import (
        load_blaes_encoding, load_amme_encoding,
        build_all_encoding_data,
        generate_common_plots as generate_encoding_common_plots,
        generate_memory_plots as generate_encoding_memory_plots,
    )
    from combined_retrieval_power import (
        load_blaes_retrieval, load_amme_retrieval,
        build_all_retrieval_data,
        generate_common_plots as generate_retrieval_common_plots,
        generate_memory_plots as generate_retrieval_memory_plots,
    )

    summary_rows = []

    # --- Encoding ---
    print("\n  Loading encoding power data...")
    blaes_enc = filter_pnas_regions(load_blaes_encoding())
    amme_enc = filter_pnas_regions(load_amme_encoding())
    all_enc = filter_pnas_regions(build_all_encoding_data(blaes_enc, amme_enc))

    # Get balanced subjects for each group
    print("\n  Determining balanced subjects (encoding)...")
    balanced_subjects = {}
    for group_label, data in [('blaes', blaes_enc), ('amme', amme_enc), ('all', all_enc)]:
        print(f"    {group_label.upper()}:")
        counts = get_nostim_trial_counts(data, measure='power')
        balanced_subjects[group_label] = get_balanced_subjects(counts)

    for group_label, data in [('BLAES', blaes_enc), ('AMME', amme_enc), ('All', all_enc)]:
        keep = balanced_subjects[group_label.lower()]
        filtered_data = filter_subject_dicts(data, keep)
        endo = extract_endogenous(filtered_data, measure='power')
        out = ensure_dir(os.path.join(OUTPUT_ROOT, 'encoding_power', group_label.lower()))
        plot_label = balanced_plot_label(group_label)
        generate_endogenous_plots(endo, out, plot_label, 'Power', 'Encoding')
        add_balanced_stim_memory_plots(
            filtered_data,
            out,
            plot_label,
            generate_encoding_common_plots,
            generate_encoding_memory_plots,
        )
        generate_endogenous_only_combined_plots(
            filtered_data, out, plot_label, 'power', 'encoding')
        counts = get_nostim_trial_counts(data, measure='power')
        summary = summarize_balanced_subjects(counts)
        summary_rows.append({
            'measure': 'Power',
            'phase': 'Encoding',
            'group': group_label,
            'included_n': len(summary['included']),
            'total_n': summary['total'],
            'excluded_n': len(summary['excluded']),
            'excluded': summary['excluded'],
            'region_counts': get_region_subject_counts(filtered_data),
        })

    for group_label, data in [('blaes', blaes_enc), ('amme', amme_enc), ('all', all_enc)]:
        keep = balanced_subjects[group_label]
        filtered_data = filter_mlmr_frames(data, keep)
        export_endogenous_mlmr_csv(filtered_data, CSV_OUTPUT_DIR,
                                   f'balanced_encoding_power_{group_label}_mlmr_input')

    # --- Retrieval ---
    print("\n  Loading retrieval power data...")
    blaes_ret = filter_pnas_regions(load_blaes_retrieval())
    amme_ret = filter_pnas_regions(load_amme_retrieval())
    all_ret = filter_pnas_regions(build_all_retrieval_data(blaes_ret, amme_ret))

    print("\n  Determining balanced subjects (retrieval)...")
    balanced_subjects_ret = {}
    for group_label, data in [('blaes', blaes_ret), ('amme', amme_ret), ('all', all_ret)]:
        print(f"    {group_label.upper()}:")
        counts = get_nostim_trial_counts(data, measure='power')
        balanced_subjects_ret[group_label] = get_balanced_subjects(counts)

    for group_label, data in [('BLAES', blaes_ret), ('AMME', amme_ret), ('All', all_ret)]:
        keep = balanced_subjects_ret[group_label.lower()]
        filtered_data = filter_subject_dicts(data, keep)
        endo = extract_endogenous(filtered_data, measure='power')
        out = ensure_dir(os.path.join(OUTPUT_ROOT, 'retrieval_power', group_label.lower()))
        plot_label = balanced_plot_label(group_label)
        generate_endogenous_plots(endo, out, plot_label, 'Power', 'Retrieval')
        add_balanced_stim_memory_plots(
            filtered_data,
            out,
            plot_label,
            generate_retrieval_common_plots,
            generate_retrieval_memory_plots,
        )
        generate_endogenous_only_combined_plots(
            filtered_data, out, plot_label, 'power', 'retrieval')
        counts = get_nostim_trial_counts(data, measure='power')
        summary = summarize_balanced_subjects(counts)
        summary_rows.append({
            'measure': 'Power',
            'phase': 'Retrieval',
            'group': group_label,
            'included_n': len(summary['included']),
            'total_n': summary['total'],
            'excluded_n': len(summary['excluded']),
            'excluded': summary['excluded'],
            'region_counts': get_region_subject_counts(filtered_data),
        })

    for group_label, data in [('blaes', blaes_ret), ('amme', amme_ret), ('all', all_ret)]:
        keep = balanced_subjects_ret[group_label]
        filtered_data = filter_mlmr_frames(data, keep)
        export_endogenous_mlmr_csv(filtered_data, CSV_OUTPUT_DIR,
                                   f'balanced_retrieval_power_{group_label}_mlmr_input')

    print("  Power analysis complete.")
    return balanced_subjects, balanced_subjects_ret, summary_rows


# ---------------------------------------------------------------------------
# COHERENCE
# ---------------------------------------------------------------------------

def run_coherence_analysis(enc_balanced_subjects=None, ret_balanced_subjects=None):
    """Run coherence analysis using the SAME balanced subject lists from power.

    Trial counts are identical across measures (same behavioral task), so we use
    power-derived exclusion lists to ensure consistency.
    """
    print("\n" + "=" * 60)
    print("Balanced Memory Trials — COHERENCE")
    print("=" * 60)

    from combined_encoding_coherence import (
        load_blaes_encoding as load_blaes_encoding_coh,
        load_amme_encoding as load_amme_encoding_coh,
        generate_common_plots as generate_encoding_common_plots,
        generate_memory_plots as generate_encoding_memory_plots,
    )
    from combined_retrieval_coherence import (
        load_blaes_retrieval as load_blaes_retrieval_coh,
        load_amme_retrieval as load_amme_retrieval_coh,
        generate_common_plots as generate_retrieval_common_plots,
        generate_memory_plots as generate_retrieval_memory_plots,
    )
    from endogenous_memory import _merge_coherence_data

    summary_rows = []

    # --- Encoding ---
    print("\n  Loading encoding coherence data...")
    blaes_enc = filter_pnas_regions(load_blaes_encoding_coh())
    amme_enc = filter_pnas_regions(load_amme_encoding_coh())
    all_enc = filter_pnas_regions(_merge_coherence_data(blaes_enc, amme_enc))

    for group_label, data in [('BLAES', blaes_enc), ('AMME', amme_enc), ('All', all_enc)]:
        keep = enc_balanced_subjects.get(group_label.lower()) if enc_balanced_subjects else None
        if keep is not None:
            print(f"    {group_label}: using {len(keep)} balanced subjects from power encoding filter")
        filtered_data = filter_subject_dicts(data, keep)
        endo = extract_endogenous(filtered_data, measure='coherence')
        out = ensure_dir(os.path.join(OUTPUT_ROOT, 'encoding_coherence', group_label.lower()))
        plot_label = balanced_plot_label(group_label)
        generate_endogenous_plots(endo, out, plot_label, 'Coherence', 'Encoding')
        add_balanced_stim_memory_plots(
            filtered_data,
            out,
            plot_label,
            generate_encoding_common_plots,
            generate_encoding_memory_plots,
        )
        generate_endogenous_only_combined_plots(
            filtered_data, out, plot_label, 'coherence', 'encoding')
        summary_rows.append({
            'measure': 'Coherence',
            'phase': 'Encoding',
            'group': group_label,
            'included_n': len(keep) if keep else '?',
            'total_n': len(keep) if keep else '?',
            'excluded_n': 0,
            'excluded': [],
            'region_counts': get_region_subject_counts(filtered_data),
            'note': 'Uses power-derived balanced subject list',
        })

    for group_label, data in [('blaes', blaes_enc), ('amme', amme_enc), ('all', all_enc)]:
        keep = enc_balanced_subjects.get(group_label) if enc_balanced_subjects else None
        filtered_data = filter_mlmr_frames(data, keep)
        export_endogenous_mlmr_csv(filtered_data, CSV_OUTPUT_DIR,
                                   f'balanced_encoding_coherence_{group_label}_mlmr_input',
                                   measure_name='Coherence')

    # --- Retrieval ---
    print("\n  Loading retrieval coherence data...")
    blaes_ret = filter_pnas_regions(load_blaes_retrieval_coh())
    amme_ret = filter_pnas_regions(load_amme_retrieval_coh())
    all_ret = filter_pnas_regions(_merge_coherence_data(blaes_ret, amme_ret))

    for group_label, data in [('BLAES', blaes_ret), ('AMME', amme_ret), ('All', all_ret)]:
        keep = ret_balanced_subjects.get(group_label.lower()) if ret_balanced_subjects else None
        if keep is not None:
            print(f"    {group_label}: using {len(keep)} balanced subjects from power retrieval filter")
        filtered_data = filter_subject_dicts(data, keep)
        endo = extract_endogenous(filtered_data, measure='coherence')
        out = ensure_dir(os.path.join(OUTPUT_ROOT, 'retrieval_coherence', group_label.lower()))
        plot_label = balanced_plot_label(group_label)
        generate_endogenous_plots(endo, out, plot_label, 'Coherence', 'Retrieval')
        add_balanced_stim_memory_plots(
            filtered_data,
            out,
            plot_label,
            generate_retrieval_common_plots,
            generate_retrieval_memory_plots,
        )
        generate_endogenous_only_combined_plots(
            filtered_data, out, plot_label, 'coherence', 'retrieval')
        summary_rows.append({
            'measure': 'Coherence',
            'phase': 'Retrieval',
            'group': group_label,
            'included_n': len(keep) if keep else '?',
            'total_n': len(keep) if keep else '?',
            'excluded_n': 0,
            'excluded': [],
            'region_counts': get_region_subject_counts(filtered_data),
            'note': 'Uses power-derived balanced subject list',
        })

    for group_label, data in [('blaes', blaes_ret), ('amme', amme_ret), ('all', all_ret)]:
        keep = ret_balanced_subjects.get(group_label) if ret_balanced_subjects else None
        filtered_data = filter_mlmr_frames(data, keep)
        export_endogenous_mlmr_csv(filtered_data, CSV_OUTPUT_DIR,
                                   f'balanced_retrieval_coherence_{group_label}_mlmr_input',
                                   measure_name='Coherence')

    print("  Coherence analysis complete.")
    return summary_rows


# ---------------------------------------------------------------------------
# PAC
# ---------------------------------------------------------------------------

def run_pac_analysis(enc_balanced_subjects=None, ret_balanced_subjects=None):
    print("\n" + "=" * 60)
    print("Balanced Memory Trials — PAC")
    print("=" * 60)

    from combined_encoding_pac import (
        load_grouped_encoding_pac_data,
        generate_common_plots as generate_encoding_common_plots,
        generate_memory_plots as generate_encoding_memory_plots,
    )
    from combined_retrieval_pac import (
        load_grouped_retrieval_pac_data,
        generate_common_plots as generate_retrieval_common_plots,
        generate_memory_plots as generate_retrieval_memory_plots,
    )

    summary_rows = []

    # --- Encoding ---
    print("\n  Loading encoding PAC data...")
    grouped_enc = load_grouped_encoding_pac_data()
    for group_key, label in [('blaes', 'BLAES'), ('amme', 'AMME'), ('all', 'All')]:
        plot_label = balanced_plot_label(label)
        keep = enc_balanced_subjects.get(group_key) if enc_balanced_subjects else None
        if keep is not None:
            print(f"    {label}: using {len(keep)} balanced subjects from power encoding filter")
        filtered_data = filter_pnas_regions(filter_subject_dicts(grouped_enc[group_key], keep))
        out = ensure_dir(os.path.join(OUTPUT_ROOT, 'encoding_pac', group_key))
        add_balanced_stim_memory_plots(
            filtered_data,
            out,
            plot_label,
            generate_encoding_common_plots,
            generate_encoding_memory_plots,
        )
        endo = extract_endogenous(filtered_data, measure='pac')
        generate_endogenous_plots(endo, out, plot_label, 'PAC', 'Encoding', band_ranges=PAC_BANDS)
        generate_endogenous_only_combined_plots(
            filtered_data, out, plot_label, 'pac', 'encoding',
            band_ranges=PAC_BANDS)
        summary_rows.append({
            'measure': 'PAC',
            'phase': 'Encoding',
            'group': label,
            'included_n': len(keep) if keep is not None else 0,
            'total_n': len(keep) if keep is not None else 0,
            'excluded_n': 0,
            'excluded': [],
            'region_counts': get_region_subject_counts(filtered_data),
            'note': 'Uses power-derived balanced subject list',
        })

    # --- Retrieval ---
    print("\n  Loading retrieval PAC data...")
    grouped_ret = load_grouped_retrieval_pac_data()
    for group_key, label in [('blaes', 'BLAES'), ('amme', 'AMME'), ('all', 'All')]:
        plot_label = balanced_plot_label(label)
        keep = ret_balanced_subjects.get(group_key) if ret_balanced_subjects else None
        if keep is not None:
            print(f"    {label}: using {len(keep)} balanced subjects from power retrieval filter")
        filtered_data = filter_pnas_regions(filter_subject_dicts(grouped_ret[group_key], keep))
        out = ensure_dir(os.path.join(OUTPUT_ROOT, 'retrieval_pac', group_key))
        add_balanced_stim_memory_plots(
            filtered_data,
            out,
            plot_label,
            generate_retrieval_common_plots,
            generate_retrieval_memory_plots,
        )
        endo = extract_endogenous(filtered_data, measure='pac')
        generate_endogenous_plots(endo, out, plot_label, 'PAC', 'Retrieval', band_ranges=PAC_BANDS)
        generate_endogenous_only_combined_plots(
            filtered_data, out, plot_label, 'pac', 'retrieval',
            band_ranges=PAC_BANDS)
        summary_rows.append({
            'measure': 'PAC',
            'phase': 'Retrieval',
            'group': label,
            'included_n': len(keep) if keep is not None else 0,
            'total_n': len(keep) if keep is not None else 0,
            'excluded_n': 0,
            'excluded': [],
            'region_counts': get_region_subject_counts(filtered_data),
            'note': 'Uses power-derived balanced subject list',
        })

    print("  PAC analysis complete.")
    return summary_rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Balanced Memory Trials Analysis")
    print(f"Excluding subjects with <{MIN_TRIALS_PER_CONDITION} NoStim trials")
    print("in either remembered or forgotten condition")
    print("=" * 60)

    ensure_dir(OUTPUT_ROOT)
    ensure_dir(CSV_OUTPUT_DIR)

    enc_balanced, ret_balanced, power_summary = run_power_analysis()
    coherence_summary = run_coherence_analysis(
        enc_balanced_subjects=enc_balanced, ret_balanced_subjects=ret_balanced)
    pac_summary = run_pac_analysis(
        enc_balanced_subjects=enc_balanced, ret_balanced_subjects=ret_balanced)
    write_balanced_summary_report(power_summary + coherence_summary + pac_summary)

    print("\n" + "=" * 60)
    print(f"Done! Balanced memory trial outputs saved to: {OUTPUT_ROOT}")
    print("=" * 60)


if __name__ == '__main__':
    main()
