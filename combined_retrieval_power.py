#!/usr/bin/env python
"""
Combined Group Power Analysis - Retrieval
Combines BLAES and AMME retrieval power analyses.
Generates outputs for each group (outputs/BLAES, outputs/AMME) and all combined (outputs/all).
"""

import os
import glob
import shutil
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
from combined_power_common import ALLHPC, MTL, augment_with_power_composites

def running_in_notebook():
    try:
        from IPython import get_ipython
        shell = get_ipython()
        return shell is not None and shell.__class__.__name__ == 'ZMQInteractiveShell'
    except Exception:
        return False


IN_NOTEBOOK = running_in_notebook()
if not IN_NOTEBOOK:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from textwrap import fill

warnings.filterwarnings('ignore')

def get_script_dir():
    if '__file__' in globals():
        return os.path.dirname(os.path.abspath(__file__))
    return os.getcwd()


SCRIPT_DIR = get_script_dir()
OUTPUT_ROOT = os.path.join(SCRIPT_DIR, 'outputs')
OUTPUT_BASE = os.path.join(OUTPUT_ROOT, 'retrieval_power')
CSV_OUTPUT_DIR = os.path.join(OUTPUT_ROOT, 'csvs')
RESPONDER_STATUS_CSV = os.path.join(CSV_OUTPUT_DIR, 'AMMEBLAES_responder_status.csv')

POWER_RANGES = {'Theta': (4, 8), 'Slow gamma': (30, 55)}
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

ROI_COLORS_SPEC = {
    'BLA': 'teal', ALLHPC: '#5B4B8A', MTL: '#2E8B57',
    'PRC': '#E61C59', 'EC': '#8A2BE2', 'PHG': 'blue',
    'CA': 'black', 'DG': 'skyblue', 'HPC': 'orange',
}

ROI_COLORS_BAR = {
    'BLA': '#008080', ALLHPC: '#5B4B8A', MTL: '#2E8B57',
    'CA': '#00CED1', 'DG': '#87CEEB', 'HPC': '#FFA500',
    'EC': '#800080', 'PHG': '#0000FF', 'PRC': '#FFC0CB',
}

AMME_RETRIEVAL_REGION_EXCLUSIONS = {
    'BLA': {'amyg057'},
    'HPC': {'amyg030'},
    'CA': {'amyg034'},
}

_RESPONDER_STATUS_MAP = None


# =========================================================================
#  HELPERS
# =========================================================================

def fix_region(df):
    df = df.copy()
    df['Region'] = df['Region'].replace('ER', 'EC')
    return df

def apply_subject_region_exclusions(df, exclude_map):
    if not exclude_map or 'Patient' not in df.columns or 'Region' not in df.columns:
        return df
    return df[~df.apply(
        lambda row: row['Region'] in exclude_map and row['Patient'] in exclude_map[row['Region']],
        axis=1,
    )]

def make_subject_color_map(subjects):
    ordered_subjects = sorted(subjects)
    palette = sns.color_palette('husl', len(ordered_subjects))
    return {subject: color for subject, color in zip(ordered_subjects, palette)}

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path

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

def band_filename(filename, band):
    root, ext = os.path.splitext(filename)
    band_slug = band.lower().replace(' ', '_')
    return f'{root}_{band_slug}{ext}'

def plot_responder_status_bargraph(df_collapsed, out_dir, title, ylabel, filename, rotate_xticks=False, caption_text=None):
    if df_collapsed.empty:
        return
    plot_df = df_collapsed.copy()
    plot_df['Responder status'] = plot_df['Patient'].map(get_responder_status_map()).fillna('Unknown')
    power_order = [band for band in POWER_RANGES if band in plot_df['power_range'].unique()]
    if not power_order:
        power_order = list(dict.fromkeys(plot_df['power_range']))
    roi_order = list(dict.fromkeys(plot_df['Region']))
    if not roi_order:
        return

    gray_values = np.linspace(0.85, 0.45, len(roi_order))
    bar_colors = {
        roi: matplotlib.colors.to_hex((gray, gray, gray))
        for roi, gray in zip(roi_order, gray_values)
    }

    responder_handles = []
    for status in RESPONDER_ORDER:
        if status in plot_df['Responder status'].values:
            responder_handles.append(
                Line2D([0], [0], marker='o', linestyle='None', markersize=9,
                       markerfacecolor=RESPONDER_PALETTE[status], markeredgecolor='none',
                       label=status)
            )
    if 'Unknown' in plot_df['Responder status'].values:
        responder_handles.append(
            Line2D([0], [0], marker='o', linestyle='None', markersize=9,
                   markerfacecolor=RESPONDER_PALETTE['Unknown'], markeredgecolor='none',
                   label='Unknown')
        )

    for band in power_order:
        band_df = plot_df[plot_df['power_range'] == band].copy()
        if band_df.empty:
            continue
        rng = np.random.default_rng(7)
        fig, ax = plt.subplots(figsize=(11.5, 8.2))
        summary = band_df.groupby('Region')['mean_power_diff'].agg(['mean', 'sem']).reindex(roi_order)
        x = np.arange(len(roi_order))
        ax.bar(
            x,
            summary['mean'].to_numpy(),
            yerr=summary['sem'].fillna(0).to_numpy(),
            color=[bar_colors[roi] for roi in roi_order],
            edgecolor='#4D4D4D',
            linewidth=1.2,
            width=0.72,
            capsize=3,
            ecolor='#4D4D4D',
            zorder=1,
        )
        for idx, roi in enumerate(roi_order):
            roi_df = band_df[band_df['Region'] == roi]
            if roi_df.empty:
                continue
            jitter = rng.uniform(-0.16, 0.16, len(roi_df))
            colors = [RESPONDER_PALETTE.get(status, RESPONDER_PALETTE['Unknown']) for status in roi_df['Responder status']]
            ax.scatter(
                np.full(len(roi_df), idx, dtype=float) + jitter,
                roi_df['mean_power_diff'],
                c=colors,
                s=42,
                alpha=0.75,
                edgecolors='none',
                zorder=3,
            )
        ax.axhline(0, color='#4D4D4D', linewidth=1.2, zorder=0)
        ax.set_title(band, fontsize=20, fontweight='bold')
        ax.set_xlabel('')
        ax.set_xticks(x)
        ax.set_xticklabels(
            roi_order,
            rotation=45 if rotate_xticks else 0,
            ha='right' if rotate_xticks else 'center',
            fontsize=16,
            fontweight='bold',
        )
        ax.tick_params(axis='y', labelsize=15, width=2, length=6)
        ax.tick_params(axis='x', width=2, length=6)
        ax.set_ylabel(ylabel, fontsize=18, fontweight='bold')
        fig.suptitle(f'{title} - {band}', fontsize=24, fontweight='bold', y=0.98)
        if responder_handles:
            fig.legend(
                handles=responder_handles,
                title='Responder status',
                loc='upper center',
                bbox_to_anchor=(0.5, 0.91),
                ncol=min(4, len(responder_handles)),
                frameon=False,
                fontsize=14,
                title_fontsize=16,
            )
        add_method_caption(
            fig,
            caption_text or 'Method: Bars = mean across patients, error bars = SEM, dots = patient values colored by responder-status CSV.',
        )
        fig.tight_layout(rect=[0, 0.05, 1, 0.84])
        fig.savefig(os.path.join(out_dir, band_filename(filename, band)), dpi=300, bbox_inches='tight')
        finalize_figure(fig)

def reset_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)
    return path

def finalize_figure(fig=None):
    if IN_NOTEBOOK:
        plt.show()
    plt.close(fig)

def add_method_caption(fig, text, y=0.012, fontsize=9, width=150):
    fig.text(
        0.5,
        y,
        fill(text, width=width),
        ha='center',
        va='bottom',
        fontsize=fontsize,
        color='dimgray',
    )

def merge_dicts(*dicts):
    merged = {}
    for d in dicts:
        for roi, sd in d.items():
            merged.setdefault(roi, {}).update(sd)
    return merged

def merge_sets(*sets_):
    merged = set()
    for s in sets_:
        merged.update(s or set())
    return merged

def collect_unique_legend_items(axes):
    handles = []
    labels = []
    for ax in axes:
        ax_handles, ax_labels = ax.get_legend_handles_labels()
        for handle, label in zip(ax_handles, ax_labels):
            if label and label not in labels:
                handles.append(handle)
                labels.append(label)
    return handles, labels

def sorted_freq_cols(df, prefix):
    cols = [c for c in df.columns if c.startswith(prefix)]
    return sorted(cols, key=lambda x: float(x.split(prefix)[1]))

def freq_vals(cols, prefix):
    return np.array([float(c.split(prefix)[1]) for c in cols], dtype=float)

def compute_diff_df(stim_d, nostim_d, freqs, ranges):
    rows = []
    all_rois = sorted(set(stim_d.keys()) | set(nostim_d.keys()))
    for roi in all_rois:
        if roi not in stim_d or roi not in nostim_d:
            continue
        for subj in sorted(set(stim_d[roi]) & set(nostim_d[roi])):
            spv, npv = stim_d[roi][subj], nostim_d[roi][subj]
            for bn, (lo, hi) in ranges.items():
                mask = (freqs >= lo) & (freqs <= hi)
                rows.append({'Patient': subj, 'Region': roi, 'power_range': bn,
                             'mean_power_diff': spv[mask].mean() - npv[mask].mean()})
    return pd.DataFrame(rows)

def compute_condition_band_df(stim_d, nostim_d, freqs, ranges, exclude_rois=None):
    rows = []
    exclude_rois = set(exclude_rois or set())
    all_rois = sorted(set(stim_d.keys()) | set(nostim_d.keys()))
    for roi in all_rois:
        if roi.startswith('PNAS') or roi in exclude_rois:
            continue
        for subj in sorted(set(stim_d.get(roi, {})) & set(nostim_d.get(roi, {}))):
            spv = np.asarray(stim_d[roi][subj], dtype=np.float64)
            npv = np.asarray(nostim_d[roi][subj], dtype=np.float64)
            for bn, (lo, hi) in ranges.items():
                mask = (freqs >= lo) & (freqs <= hi)
                rows.append({
                    'Patient': subj,
                    'Region': roi,
                    'power_range': bn,
                    'stim': spv[mask].mean(),
                    'nostim': npv[mask].mean(),
                })
    return pd.DataFrame(rows)

def plot_grouped_condition_bars(df, out_dir, label, phase_label, filename_template, title_suffix, show_patients=True):
    if df.empty:
        return
    band_order = [band for band in POWER_RANGES if band in df['power_range'].unique()]
    if not band_order:
        band_order = sorted(df['power_range'].unique())
    if show_patients:
        patients = sorted(df['Patient'].unique())
        filled_markers = ['o', 's', '^', 'D', 'v', 'P', 'X', '*', '<', '>', 'h', '8', 'p', 'H', 'd']
        marker_styles = ([(m, True) for m in filled_markers] +
                         [(m, False) for m in filled_markers] +
                         [('1', True), ('2', True), ('3', True), ('4', True),
                          ('+', True), ('x', True), ('|', True), ('_', True)] +
                         [(f'${i}$', True) for i in range(1, 11)])
        patient_markers = {p: marker_styles[i] for i, p in enumerate(patients)}
    bar_colors = {'nostim': '#1f77b4', 'stim': '#d62728'}

    for band in band_order:
        df_band = df[df['power_range'] == band]
        if df_band.empty:
            continue
        roi_order = [r for r in ROI_COLORS_BAR if r in df_band['Region'].unique()]
        if not roi_order:
            roi_order = sorted(df_band['Region'].unique())
        if not roi_order:
            continue

        fig, ax = plt.subplots(figsize=(16, 11))
        x_base = np.arange(len(roi_order))
        bar_width = 0.34
        offsets = {'nostim': -bar_width / 2, 'stim': bar_width / 2}

        for i, roi in enumerate(roi_order):
            roi_df = df_band[df_band['Region'] == roi]
            for trial in ['nostim', 'stim']:
                vals = roi_df[trial].dropna()
                if len(vals) == 0:
                    continue
                mean = vals.mean()
                sem = vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0
                ax.bar(
                    x_base[i] + offsets[trial],
                    mean,
                    width=bar_width,
                    color=bar_colors[trial],
                    edgecolor='black',
                    linewidth=1.2,
                    yerr=sem,
                    capsize=4,
                    zorder=1,
                )

        if show_patients:
            for i, roi in enumerate(roi_order):
                roi_df = df_band[df_band['Region'] == roi]
                for _, row in roi_df.iterrows():
                    x_ns = x_base[i] + offsets['nostim']
                    x_s = x_base[i] + offsets['stim']
                    ax.plot([x_ns, x_s], [row['nostim'], row['stim']], color='black', alpha=0.35, linewidth=1)
                    marker, filled = patient_markers[row['Patient']]
                    if filled:
                        ax.scatter(x_ns, row['nostim'], marker=marker, color='black', alpha=0.65, s=55, zorder=3)
                        ax.scatter(x_s, row['stim'], marker=marker, color='black', alpha=0.95, s=55, zorder=3)
                    else:
                        ax.scatter(
                            x_ns,
                            row['nostim'],
                            marker=marker,
                            facecolors='white',
                            edgecolors='black',
                            linewidths=1.1,
                            alpha=0.85,
                            s=60,
                            zorder=3,
                        )
                        ax.scatter(
                            x_s,
                            row['stim'],
                            marker=marker,
                            facecolors='white',
                            edgecolors='black',
                            linewidths=1.3,
                            alpha=1.0,
                            s=60,
                            zorder=3,
                        )

        ax.set_xticks(x_base)
        ax.set_xticklabels(roi_order, fontsize=14, fontweight='bold')
        ax.tick_params(axis='x', labelsize=14, width=2, length=6)
        ax.tick_params(axis='y', labelsize=14, width=2, length=6)
        ax.axhline(0, color='grey')
        ax.set_ylabel('Mean Baseline-Corrected Power (dB)', fontsize=17, fontweight='bold')
        fig.suptitle(f'{label} {phase_label} Baseline-Corrected Power - {title_suffix}', fontsize=22, fontweight='bold', y=0.965)
        ax.set_title(band, fontsize=18, fontweight='bold')

        y_values = pd.concat([df_band['nostim'], df_band['stim']], ignore_index=True).dropna()
        if not y_values.empty:
            y_min = y_values.min()
            y_max = y_values.max()
            if np.isclose(y_min, y_max):
                ax.set_ylim(y_min - 0.1, y_max + 0.1)
            else:
                y_pad = 0.25 * (y_max - y_min)
                ax.set_ylim(y_min - y_pad, y_max + y_pad)

        stim_legend = [
            Line2D([0], [0], color=bar_colors['stim'], lw=8, label='Stim'),
            Line2D([0], [0], color=bar_colors['nostim'], lw=8, label='No Stim'),
        ]
        fig.legend(handles=stim_legend, loc='upper center', bbox_to_anchor=(0.50, 0.925),
                   ncol=2, frameon=False, fontsize=12, title='Condition', title_fontsize=13,
                   handlelength=1.8, handletextpad=0.6, columnspacing=1.2, borderaxespad=0.2)

        if show_patients:
            patient_handles = [
                Line2D([0], [0], marker=patient_markers[p][0], color='black', linestyle='None',
                       markerfacecolor='black' if patient_markers[p][1] else 'white',
                       markeredgecolor='black', markeredgewidth=1.1, markersize=8, label=str(p))
                for p in sorted(df_band['Patient'].unique())
            ]
            fig.legend(handles=patient_handles, loc='center left', bbox_to_anchor=(0.81, 0.50),
                       ncol=2, columnspacing=0.9, handletextpad=0.4, labelspacing=0.45,
                       frameon=False, fontsize=10, title='Patient', title_fontsize=12)
        add_method_caption(
            fig,
            'Method: For each patient, ROI, and band, separate no-stim and stim means are computed from baseline-corrected spectra. Bars show across-patient means with SEM and dots show patient values.',
        )
        fig.subplots_adjust(left=0.10, right=0.78 if show_patients else 0.95, bottom=0.15, top=0.86)
        plt.savefig(os.path.join(out_dir, band_filename(filename_template, band)), bbox_inches='tight', dpi=300)
        finalize_figure(fig)

def collapse_power_across_memory_conditions(data):
    """Match notebook logic: average each subject's available stim x memory spectra."""
    collapsed = {}
    for key in ['nostim_rem', 'nostim_forg', 'stim_rem', 'stim_forg']:
        cond_data = data.get(key, {})
        for roi, subj_dict in cond_data.items():
            if roi.startswith('PNAS'):
                continue
            collapsed.setdefault(roi, {})
            for subject, vec in subj_dict.items():
                collapsed[roi].setdefault(subject, []).append(np.asarray(vec, dtype=np.float64))

    for roi, subj_dict in collapsed.items():
        for subject, vecs in subj_dict.items():
            collapsed[roi][subject] = np.nanmean(np.vstack(vecs), axis=0)
    return collapsed

def get_overall_power_for_plot(data):
    if data.get('use_memory_collapsed_overall'):
        collapsed = collapse_power_across_memory_conditions(data)
        if collapsed:
            return collapsed
    return data['group_all_power']


def build_trial_level_mlmr_export(df, diff_cols, measure_name='Power'):
    if not diff_cols:
        return pd.DataFrame()

    base_cols = ['Measure', 'Patient', 'Region', 'trial_type', 'yes_or_no']
    export_df = df.copy()
    export_df['Measure'] = measure_name
    export_df = export_df[base_cols + diff_cols]
    export_df = export_df[~export_df['Region'].astype(str).str.startswith('PNAS')]
    return export_df.sort_values(base_cols).reset_index(drop=True)


def build_retrieval_mlmr_export(data, measure_name='Power'):
    export_frames = data.get('mlmr_export_frames', [])
    if not export_frames:
        return pd.DataFrame()

    df_export = pd.concat(export_frames, ignore_index=True)
    if df_export.empty:
        return df_export

    df_export['Measure'] = measure_name
    base_cols = ['Measure', 'Patient', 'Region', 'trial_type', 'yes_or_no']
    freq_cols = sorted_freq_cols(df_export, 'diff_Freq_')
    return df_export[base_cols + freq_cols].sort_values(base_cols).reset_index(drop=True)


def export_retrieval_mlmr_csv(data, csv_dir, file_stem, measure_name='Power'):
    df_export = build_retrieval_mlmr_export(data, measure_name=measure_name)
    if df_export.empty:
        print(f"  Skipping CSV export for {file_stem}: no baseline-corrected stim x memory data.")
        return None

    ensure_dir(csv_dir)
    csv_path = os.path.join(csv_dir, f'{file_stem}.csv')
    df_export.to_csv(csv_path, index=False)
    print(f"  Exported MLMR CSV: {csv_path}")
    return csv_path


# =========================================================================
#  BLAES RETRIEVAL - MEMORY HELPERS
# =========================================================================

def normalize_stimulation_blaes(x):
    if pd.isna(x):
        return None
    if isinstance(x, str):
        x = x.strip().lower()
    if x in {0, '0', 0.0, False, 'false', 'nostim', 'no_stim', 'no stim'}:
        return 'nostim'
    if x in {1, '1', 1.0, True, 'true', 'stim'}:
        return 'stim'
    return None

def normalize_memory_condition_blaes(x):
    if pd.isna(x):
        return None
    s = str(x).strip().lower()
    if s in {'targ', 'old'}:
        return 'old'
    if s in {'new', 'foil', 'lure'}:
        return 'new'
    return None  # skip unexpected values instead of raising

def normalize_response_blaes(x):
    if pd.isna(x):
        return None
    s = str(x).strip().lower()
    if s in {'old', 'yes'}:
        return 'old'
    if s in {'new', 'no'}:
        return 'new'
    old_codes = {'78', '66', '37'}
    new_codes = {'67', '86', '39'}
    if s in old_codes:
        return 'old'
    if s in new_codes:
        return 'new'
    try:
        n = int(float(s))
        if n in {78, 66, 37}:
            return 'old'
        if n in {67, 86, 39}:
            return 'new'
    except Exception:
        pass
    return None

def add_memory_condition_blaes(df):
    df = df.copy()
    if 'trial_type_raw' not in df.columns:
        if 'trial_type' not in df.columns:
            return df
        df['trial_type_raw'] = df['trial_type']

    response_col = next((c for c in ['response', 'yes_or_no'] if c in df.columns), None)
    if response_col is None:
        return df

    df['memory_condition'] = df['trial_type_raw'].apply(normalize_memory_condition_blaes)
    df['normalized_response'] = df[response_col].apply(normalize_response_blaes)

    conditions = [
        (df['memory_condition'] == 'old') & (df['normalized_response'] == 'old'),
        (df['memory_condition'] == 'old') & (df['normalized_response'] == 'new'),
    ]
    values = ['remembered', 'forgotten']
    df['memory_cond'] = np.select(conditions, values, default=np.nan)
    # Only keep hit (remembered) and miss (forgotten), drop FA/CR
    df = df[df['memory_cond'].isin(['remembered', 'forgotten'])]
    return df


def normalize_trial_type_amme(ttype):
    if isinstance(ttype, str) and 'stim' in ttype.lower() and ttype.lower() != 'nostim':
        return 'stim'
    elif isinstance(ttype, str) and ttype.lower() == 'nostim':
        return 'nostim'
    return None


# =========================================================================
#  BLAES RETRIEVAL LOADING
# =========================================================================

def load_blaes_retrieval():
    project_path = '/Users/martinahollearn/Library/CloudStorage/Box-Box/InmanLab/BLAES_data/dissertation/LFP_analyses'
    data_path = os.path.join(project_path, 'Results_CSVOutput')

    power_files = glob.glob(os.path.join(data_path, "*phase3*Power*.csv"))

    data = {
        'group_all_power': {}, 'stim_power': {}, 'nostim_power': {},
        'bc_stim': {}, 'bc_nostim': {},
        'freqs_post': None, 'freqs_diff': None,
        'has_memory': True,
        'use_memory_collapsed_overall': True,
        'bc_bar_exclude_rois': set(),
        'bc_memory_collapsed_exclude_rois': set(),
        'stim_rem': {}, 'stim_forg': {},
        'nostim_rem': {}, 'nostim_forg': {},
        'bc_stim_rem': {}, 'bc_stim_forg': {},
        'bc_nostim_rem': {}, 'bc_nostim_forg': {},
        'mlmr_export_frames': [],
    }

    for file in power_files:
        df = fix_region(pd.read_csv(file))
        if 'Patient' not in df.columns or 'Region' not in df.columns:
            continue

        subject = df['Patient'].iloc[0]

        # Preserve original trial_type for memory logic
        if 'trial_type' in df.columns:
            df['trial_type_raw'] = df['trial_type']

        # Normalize stimulation
        if 'stimulation' not in df.columns:
            continue
        df['trial_type'] = df['stimulation'].apply(normalize_stimulation_blaes)
        df = df[df['trial_type'].notnull()]
        if df.empty or df['trial_type'].nunique() < 2:
            continue

        # Add memory condition
        df = add_memory_condition_blaes(df)
        if df.empty:
            continue
        df['yes_or_no'] = np.where(
            df['normalized_response'] == 'old',
            'yes',
            np.where(df['normalized_response'] == 'new', 'no', np.nan),
        )

        # post_Freq columns
        freq_cols = sorted_freq_cols(df, 'post_Freq_')
        if not freq_cols:
            continue
        if data['freqs_post'] is None:
            data['freqs_post'] = freq_vals(freq_cols, 'post_Freq_')

        print(f"  BLAES retrieval: {os.path.basename(file)} [{subject}]")

        # Overall power
        for _, row in df.groupby(['Patient', 'Region'])[freq_cols].mean().reset_index().iterrows():
            data['group_all_power'].setdefault(row['Region'], {})[row['Patient']] = row[freq_cols].values.astype(np.float64)

        # By stim
        grouped = df.groupby(['Patient', 'Region', 'trial_type'])[freq_cols].mean().reset_index()
        for _, row in grouped[grouped['trial_type'] == 'stim'].iterrows():
            data['stim_power'].setdefault(row['Region'], {})[row['Patient']] = row[freq_cols].values.astype(np.float64)
        for _, row in grouped[grouped['trial_type'] == 'nostim'].iterrows():
            data['nostim_power'].setdefault(row['Region'], {})[row['Patient']] = row[freq_cols].values.astype(np.float64)

        # By stim x memory
        grouped_mem = df.groupby(['Patient', 'Region', 'trial_type', 'memory_cond'])[freq_cols].mean().reset_index()
        mem_key_map = {
            ('stim', 'remembered'): 'stim_rem', ('stim', 'forgotten'): 'stim_forg',
            ('nostim', 'remembered'): 'nostim_rem', ('nostim', 'forgotten'): 'nostim_forg',
        }
        for _, row in grouped_mem.iterrows():
            k = mem_key_map.get((row['trial_type'], row['memory_cond']))
            if k:
                data[k].setdefault(row['Region'], {})[row['Patient']] = row[freq_cols].values.astype(np.float64)

        # Baseline-corrected
        diff_cols = sorted_freq_cols(df, 'diff_Freq_')
        if diff_cols:
            if data['freqs_diff'] is None:
                data['freqs_diff'] = freq_vals(diff_cols, 'diff_Freq_')

            df_export = build_trial_level_mlmr_export(df, diff_cols, measure_name='Power')
            if not df_export.empty:
                data['mlmr_export_frames'].append(df_export)

            grouped_bc = df.groupby(['Patient', 'Region', 'trial_type'])[diff_cols].mean().reset_index()
            for _, row in grouped_bc[grouped_bc['trial_type'] == 'stim'].iterrows():
                data['bc_stim'].setdefault(row['Region'], {})[row['Patient']] = row[diff_cols].values.astype(np.float64)
            for _, row in grouped_bc[grouped_bc['trial_type'] == 'nostim'].iterrows():
                data['bc_nostim'].setdefault(row['Region'], {})[row['Patient']] = row[diff_cols].values.astype(np.float64)

            grouped_bc_mem = df.groupby(['Patient', 'Region', 'trial_type', 'memory_cond'])[diff_cols].mean().reset_index()
            bc_key_map = {
                ('stim', 'remembered'): 'bc_stim_rem', ('stim', 'forgotten'): 'bc_stim_forg',
                ('nostim', 'remembered'): 'bc_nostim_rem', ('nostim', 'forgotten'): 'bc_nostim_forg',
            }
            for _, row in grouped_bc_mem.iterrows():
                k = bc_key_map.get((row['trial_type'], row['memory_cond']))
                if k:
                    data[k].setdefault(row['Region'], {})[row['Patient']] = row[diff_cols].values.astype(np.float64)

    return augment_with_power_composites(data)


# =========================================================================
#  AMME RETRIEVAL LOADING
# =========================================================================

def load_amme_retrieval():
    project_path = '/Users/martinahollearn/Library/CloudStorage/Box-Box/InmanLab/AMME_Data_Emory/AMME_Data/LFP_analyses_Martina'
    data_path = os.path.join(project_path, 'Results_CSVOutput', 'Phase2')

    power_files = glob.glob(os.path.join(data_path, "*Power*.csv"))

    data = {
        'group_all_power': {}, 'stim_power': {}, 'nostim_power': {},
        'bc_stim': {}, 'bc_nostim': {},
        'freqs_post': None, 'freqs_diff': None,
        'has_memory': True,
        'use_memory_collapsed_overall': True,
        'bc_bar_exclude_rois': set(),
        'bc_memory_collapsed_exclude_rois': set(),
        'stim_rem': {}, 'stim_forg': {},
        'nostim_rem': {}, 'nostim_forg': {},
        'bc_stim_rem': {}, 'bc_stim_forg': {},
        'bc_nostim_rem': {}, 'bc_nostim_forg': {},
        'mlmr_export_frames': [],
    }

    for file in power_files:
        df = pd.read_csv(file)
        if 'Patient' not in df.columns or 'Region' not in df.columns:
            continue

        # Apply per-ROI exclusions
        df = apply_subject_region_exclusions(df, AMME_RETRIEVAL_REGION_EXCLUSIONS)

        # Normalize trial type
        df['trial_type'] = df['trial_type'].apply(normalize_trial_type_amme)
        df = df[df['trial_type'].notnull()]
        if df.empty:
            continue

        # Memory condition
        df = df[df['yes_or_no'].isin(['yes', 'no'])]
        df['memory_cond'] = np.where(df['yes_or_no'] == 'yes', 'remembered', 'forgotten')

        freq_cols = sorted_freq_cols(df, 'post_Freq_')
        if not freq_cols:
            continue
        if data['freqs_post'] is None:
            data['freqs_post'] = freq_vals(freq_cols, 'post_Freq_')

        subject = df['Patient'].iloc[0]
        print(f"  AMME retrieval: {os.path.basename(file)} [{subject}]")

        # Overall power
        for _, row in df.groupby(['Patient', 'Region'])[freq_cols].mean().reset_index().iterrows():
            data['group_all_power'].setdefault(row['Region'], {})[row['Patient']] = row[freq_cols].values.astype(np.float64)

        # By stim
        grouped = df.groupby(['Patient', 'Region', 'trial_type'])[freq_cols].mean().reset_index()
        for _, row in grouped[grouped['trial_type'] == 'stim'].iterrows():
            data['stim_power'].setdefault(row['Region'], {})[row['Patient']] = row[freq_cols].values.astype(np.float64)
        for _, row in grouped[grouped['trial_type'] == 'nostim'].iterrows():
            data['nostim_power'].setdefault(row['Region'], {})[row['Patient']] = row[freq_cols].values.astype(np.float64)

        # By stim x memory
        grouped_mem = df.groupby(['Patient', 'Region', 'trial_type', 'memory_cond'])[freq_cols].mean().reset_index()
        mem_key_map = {
            ('stim', 'remembered'): 'stim_rem', ('stim', 'forgotten'): 'stim_forg',
            ('nostim', 'remembered'): 'nostim_rem', ('nostim', 'forgotten'): 'nostim_forg',
        }
        for _, row in grouped_mem.iterrows():
            k = mem_key_map.get((row['trial_type'], row['memory_cond']))
            if k:
                data[k].setdefault(row['Region'], {})[row['Patient']] = row[freq_cols].values.astype(np.float64)

        # Baseline-corrected
        diff_cols = sorted_freq_cols(df, 'diff_Freq_')
        if diff_cols:
            if data['freqs_diff'] is None:
                data['freqs_diff'] = freq_vals(diff_cols, 'diff_Freq_')

            df_export = build_trial_level_mlmr_export(df, diff_cols, measure_name='Power')
            if not df_export.empty:
                data['mlmr_export_frames'].append(df_export)

            grouped_bc = df.groupby(['Patient', 'Region', 'trial_type'])[diff_cols].mean().reset_index()
            for _, row in grouped_bc[grouped_bc['trial_type'] == 'stim'].iterrows():
                data['bc_stim'].setdefault(row['Region'], {})[row['Patient']] = row[diff_cols].values.astype(np.float64)
            for _, row in grouped_bc[grouped_bc['trial_type'] == 'nostim'].iterrows():
                data['bc_nostim'].setdefault(row['Region'], {})[row['Patient']] = row[diff_cols].values.astype(np.float64)

            grouped_bc_mem = df.groupby(['Patient', 'Region', 'trial_type', 'memory_cond'])[diff_cols].mean().reset_index()
            bc_key_map = {
                ('stim', 'remembered'): 'bc_stim_rem', ('stim', 'forgotten'): 'bc_stim_forg',
                ('nostim', 'remembered'): 'bc_nostim_rem', ('nostim', 'forgotten'): 'bc_nostim_forg',
            }
            for _, row in grouped_bc_mem.iterrows():
                k = bc_key_map.get((row['trial_type'], row['memory_cond']))
                if k:
                    data[k].setdefault(row['Region'], {})[row['Patient']] = row[diff_cols].values.astype(np.float64)

    return augment_with_power_composites(data)


# =========================================================================
#  PLOTTING FUNCTIONS - COMMON (non-memory)
# =========================================================================

def plot_power_by_roi(data, out_dir, label):
    gap = get_overall_power_for_plot(data)
    freqs = data['freqs_post']
    if not gap or freqs is None:
        return
    fig, ax = plt.subplots(figsize=(12, 8))
    for roi in sorted(gap.keys()):
        if roi.startswith('PNAS'):
            continue
        mat = np.array(list(gap[roi].values()), dtype=np.float64)
        mean, std = mat.mean(0), mat.std(0)
        c = ROI_COLORS_SPEC.get(roi, 'gray')
        ax.plot(freqs, mean, color=c, label=f'{roi} ({mat.shape[0]})')
        ax.fill_between(freqs, mean - std, mean + std, alpha=0.2, color=c)
    ax.set_xlabel('Frequency (Hz)', fontsize=18, fontweight='bold')
    ax.set_ylabel('Power (dB)', fontsize=18, fontweight='bold')
    ax.set_title(f'{label} Retrieval Group Power by ROI', fontsize=20, fontweight='bold')
    ax.tick_params(axis='both', labelsize=14)
    ax.legend(bbox_to_anchor=(1.02, 0.5), loc='center left', prop={'weight': 'bold', 'size': 12})
    add_method_caption(
        fig,
        "Method: Each ROI line is the mean retrieval spectrum across patients after averaging each patient's available stim/no-stim and remembered/forgotten spectra. Shading shows ±1 SD across patients.",
    )
    plt.tight_layout(rect=[0, 0.05, 0.85, 1])
    plt.savefig(os.path.join(out_dir, 'GroupPower_byROI_retrieval.png'), dpi=300, bbox_inches='tight')
    finalize_figure()


def plot_power_by_patient(data, out_dir, label):
    gap = get_overall_power_for_plot(data)
    freqs = data['freqs_post']
    if not gap or freqs is None:
        return
    patient_power = {}
    for roi, sd in gap.items():
        for subj, pv in sd.items():
            patient_power.setdefault(subj, []).append(pv)
    fig, ax = plt.subplots(figsize=(12, 8))
    for subj in sorted(patient_power):
        mean_p = np.array(patient_power[subj]).mean(0)
        ax.plot(freqs, mean_p, label=subj)
    ax.set_xlabel('Frequency (Hz)', fontsize=18, fontweight='bold')
    ax.set_ylabel('Power (dB)', fontsize=18, fontweight='bold')
    ax.set_title(f'{label} Retrieval Group Power by Patient', fontsize=20, fontweight='bold')
    ax.tick_params(axis='both', labelsize=14)
    ax.legend(bbox_to_anchor=(1.02, 0.5), loc='center left', prop={'weight': 'bold', 'size': 12})
    add_method_caption(
        fig,
        "Method: Each line is one patient's mean retrieval spectrum after averaging that patient's ROI spectra across available stim/no-stim and remembered/forgotten conditions.",
    )
    plt.tight_layout(rect=[0, 0.05, 0.82, 1])
    plt.savefig(os.path.join(out_dir, 'GroupPower_byPatient_retrieval.png'), dpi=300, bbox_inches='tight')
    finalize_figure()


def plot_stim_vs_nostim(data, out_dir, label):
    freqs = data['freqs_post']
    sp, nsp = data['stim_power'], data['nostim_power']
    if not sp or not nsp or freqs is None:
        return
    all_rois = sorted(set(sp.keys()) | set(nsp.keys()))
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharex=True, sharey=True)
    for ax, (title, rd) in zip(axes, [('No Stim', nsp), ('Stim', sp)]):
        for roi in all_rois:
            if roi not in rd or roi.startswith('PNAS'):
                continue
            mat = np.array(list(rd[roi].values()), dtype=np.float64)
            mean, std = mat.mean(0), mat.std(0)
            c = ROI_COLORS_SPEC.get(roi, 'gray')
            ax.plot(freqs, mean, color=c, label=f'{roi} ({mat.shape[0]})')
            ax.fill_between(freqs, mean - std, mean + std, alpha=0.2, color=c)
        ax.set_title(title, fontsize=18, fontweight='bold')
        ax.set_xlabel('Frequency (Hz)', fontsize=16, fontweight='bold')
        ax.tick_params(axis='both', labelsize=12)
    axes[0].set_ylabel('Power (dB)', fontsize=16, fontweight='bold')
    axes[1].legend(bbox_to_anchor=(1.02, 0.5), loc='center left', prop={'weight': 'bold', 'size': 11})
    fig.suptitle(f'{label} Retrieval Group Power: No Stim vs Stim', fontsize=20, fontweight='bold')
    add_method_caption(
        fig,
        'Method: Within each ROI and stimulation condition, each patient is first averaged across all matching retrieval trials. Lines show ROI means across patients and shading shows ±1 SD.',
    )
    plt.tight_layout(rect=[0, 0.06, 0.88, 0.95])
    plt.savefig(os.path.join(out_dir, 'GroupPower_byROI_retrieval_stim_vs_nostim.png'), dpi=300, bbox_inches='tight')
    finalize_figure()


def plot_per_roi_patient_stim_nostim(data, out_dir, label):
    freqs = data['freqs_post']
    sp, nsp = data['stim_power'], data['nostim_power']
    if not sp or not nsp or freqs is None:
        return
    all_rois = sorted(set(sp.keys()) | set(nsp.keys()))
    for roi in all_rois:
        if roi.startswith('PNAS'):
            continue
        roi_subjects = set(nsp.get(roi, {})) | set(sp.get(roi, {}))
        subject_colors = make_subject_color_map(roi_subjects)
        fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharex=True, sharey=True)
        plotted_any = False
        for ax, (title, rd) in zip(axes, [(f'{roi} - No Stim', nsp), (f'{roi} - Stim', sp)]):
            if roi not in rd or len(rd[roi]) == 0:
                ax.set_title(title, fontsize=18, fontweight='bold')
                ax.set_xlabel('Frequency (Hz)', fontsize=16, fontweight='bold')
                ax.tick_params(axis='both', labelsize=12)
                continue
            for subj in sorted(rd[roi]):
                ax.plot(freqs, rd[roi][subj], label=subj, color=subject_colors[subj])
            plotted_any = True
            ax.set_title(title, fontsize=18, fontweight='bold')
            ax.set_xlabel('Frequency (Hz)', fontsize=16, fontweight='bold')
            ax.tick_params(axis='both', labelsize=12)
        axes[0].set_ylabel('Power (dB)', fontsize=16, fontweight='bold')
        fig.suptitle(f'{label} Retrieval Power: {roi} by Patient', fontsize=20, fontweight='bold')
        if plotted_any:
            handles, labels_ = collect_unique_legend_items(axes)
            if handles:
                fig.legend(handles, labels_, bbox_to_anchor=(0.88, 0.5), loc='center left',
                           prop={'weight': 'bold', 'size': 10})
            add_method_caption(
                fig,
                'Method: For this ROI, each line is one patient averaged across all matching retrieval trials within the no-stim or stim condition.',
            )
            plt.tight_layout(rect=[0, 0.06, 0.82, 0.95])
            plt.savefig(os.path.join(out_dir, f'GroupPower_{roi}_byPatient_retrieval_stim_vs_nostim.png'), dpi=300, bbox_inches='tight')
        finalize_figure()


def plot_bc_bar_graph(data, out_dir, label):
    freqs = data['freqs_diff']
    bcs, bcn = data['bc_stim'], data['bc_nostim']
    if not bcs or not bcn or freqs is None:
        return
    df_diff = compute_diff_df(bcs, bcn, freqs, POWER_RANGES)
    if df_diff.empty:
        return
    df_diff = df_diff[~df_diff['Region'].str.startswith('PNAS')]
    exclude_rois = data.get('bc_bar_exclude_rois', set())
    if exclude_rois:
        df_diff = df_diff[~df_diff['Region'].isin(exclude_rois)]
    if df_diff.empty:
        return
    unique_rois = [r for r in df_diff['Region'].unique() if r in ROI_COLORS_BAR]
    if not unique_rois:
        return
    palette = {r: ROI_COLORS_BAR[r] for r in unique_rois}
    power_order = [band for band in POWER_RANGES if band in df_diff['power_range'].unique()]
    if not power_order:
        power_order = sorted(df_diff['power_range'].unique())
    for band in power_order:
        band_df = df_diff[df_diff['power_range'] == band].copy()
        if band_df.empty:
            continue
        fig, ax = plt.subplots(figsize=(11.5, 8.2))
        sns.barplot(
            data=band_df,
            x='Region',
            y='mean_power_diff',
            hue='Region',
            errorbar='se',
            palette=palette,
            legend=False,
            order=unique_rois,
            dodge=False,
            ax=ax,
        )
        sns.stripplot(
            data=band_df,
            x='Region',
            y='mean_power_diff',
            ax=ax,
            color='black',
            alpha=0.5,
            jitter=0.2,
            order=unique_rois,
        )
        ax.set_xlabel('')
        ax.set_ylabel('Baseline-Corrected Power Diff', fontsize=16, fontweight='bold')
        ax.tick_params(axis='x', labelsize=14, width=2, length=6)
        ax.tick_params(axis='y', labelsize=14, width=2, length=6)
        for l in ax.get_xticklabels():
            l.set_fontsize(14)
            l.set_fontweight('bold')
        ax.axhline(0, linestyle='-', color='grey')
        ax.set_title(band, fontsize=18, fontweight='bold')
        fig.suptitle(f'{label} Retrieval Baseline-Corrected Power Diff (Stim - No Stim)',
                     fontsize=20, fontweight='bold', y=0.98)
        add_method_caption(
            fig,
            'Method: For each patient and ROI, baseline-corrected spectra are averaged across all stim trials and all no-stim trials separately, band means are computed, then Stim-NoStim is taken. Bars = mean across patients, error bars = SEM, dots = patient values.',
        )
        fig.tight_layout(rect=[0, 0.05, 1, 0.9])
        plt.savefig(
            os.path.join(out_dir, band_filename('Bargraph_baseline_corrected_powerDiff_byROI_retrieval.png', band)),
            bbox_inches='tight',
            dpi=300,
        )
        finalize_figure(fig)
    df_condition = compute_condition_band_df(bcs, bcn, freqs, POWER_RANGES, data.get('bc_bar_exclude_rois', set()))
    plot_grouped_condition_bars(
        df_condition,
        out_dir,
        label,
        'Retrieval',
        'StimVsNoStim_alltrials_retrieval.png',
        'All Trials',
    )
    plot_grouped_condition_bars(
        df_condition,
        out_dir,
        label,
        'Retrieval',
        'StimVsNoStim_alltrials_retrieval_clean.png',
        'All Trials',
        show_patients=False,
    )
    plot_responder_status_bargraph(
        df_diff[df_diff['Region'].isin(unique_rois)],
        out_dir,
        f'{label} Retrieval Baseline-Corrected Power Diff (Stim - No Stim)',
        'Baseline-Corrected Power Diff',
        'Bargraph_baseline_corrected_powerDiff_byROI_retrieval_responder_status.png',
        caption_text='Method: For each patient and ROI, baseline-corrected spectra are averaged across all stim trials and all no-stim trials separately, band means are computed, then Stim-NoStim is taken. Bars = mean across patients, error bars = SEM, dots = patient values colored by responder-status CSV.',
    )


def plot_bc_per_roi(data, out_dir, label):
    freqs = data['freqs_diff']
    bcs, bcn = data['bc_stim'], data['bc_nostim']
    if freqs is None:
        return
    all_rois = sorted(set(bcs.keys()) | set(bcn.keys()))
    for roi in all_rois:
        if roi.startswith('PNAS'):
            continue
        nd, sd = bcn.get(roi, {}), bcs.get(roi, {})
        if not nd and not sd:
            continue
        fig, ax = plt.subplots(figsize=(8, 6))
        for cond_name, cond_d, color in [('No Stim', nd, 'blue'), ('Stim', sd, 'red')]:
            if not cond_d:
                continue
            mat = np.array(list(cond_d.values()), dtype=np.float64)
            n = mat.shape[0]
            mean, se = mat.mean(0), mat.std(0) / np.sqrt(n)
            ax.plot(freqs, mean, color=color, label=f'{cond_name} (n={n})')
            ax.fill_between(freqs, mean - se, mean + se, alpha=0.2, color=color)
        ax.set_title(f'{label} Retrieval {roi} Baseline-Corrected Power', fontsize=18, fontweight='bold')
        ax.set_xlabel('Frequency (Hz)', fontsize=16, fontweight='bold')
        ax.set_ylabel('Power (dB)', fontsize=16, fontweight='bold')
        ax.tick_params(axis='both', labelsize=14)
        ax.set_xlim(1, 100)
        ax.legend(fontsize=12)
        add_method_caption(
            fig,
            'Method: For this ROI, each patient is averaged on baseline-corrected retrieval spectra separately for no-stim and stim trials. Lines show means across patients and shading shows SEM.',
        )
        plt.tight_layout(rect=[0, 0.06, 1, 1])
        plt.savefig(os.path.join(out_dir, f'{roi}_BaselineCorrectedPower_retrieval_stim_vs_nostim.png'), dpi=300)
        finalize_figure()


# =========================================================================
#  PLOTTING FUNCTIONS - MEMORY SPECIFIC
# =========================================================================

def plot_quadrant_stim_memory(data, out_dir, label):
    freqs = data['freqs_post']
    if freqs is None:
        return
    mem_dicts = {
        'NoStim Remembered': data.get('nostim_rem', {}),
        'NoStim Forgotten': data.get('nostim_forg', {}),
        'AvgStim Remembered': data.get('stim_rem', {}),
        'AvgStim Forgotten': data.get('stim_forg', {}),
    }
    if not any(mem_dicts.values()):
        return
    all_rois = sorted(set().union(*[d.keys() for d in mem_dicts.values()]))
    gspr = {}
    for cd in mem_dicts.values():
        for roi, sd in cd.items():
            gspr.setdefault(roi, set()).update(sd.keys())

    cond_order = ['NoStim Remembered', 'NoStim Forgotten', 'AvgStim Remembered', 'AvgStim Forgotten']
    fig, axes = plt.subplots(2, 2, figsize=(18, 10), sharex=True, sharey=True)
    axes_flat = axes.flatten()
    plotted = set()
    for ax, cn in zip(axes_flat, cond_order):
        cd = mem_dicts[cn]
        for roi in all_rois:
            if roi.startswith('PNAS') or roi not in cd:
                continue
            mat = np.array(list(cd[roi].values()), dtype=np.float64)
            mean, std = mat.mean(0), mat.std(0)
            c = ROI_COLORS_SPEC.get(roi, 'gray')
            lbl = None
            if roi not in plotted:
                lbl = f'{roi} ({len(gspr.get(roi, set()))})'
                plotted.add(roi)
            ax.plot(freqs, mean, color=c, label=lbl)
            ax.fill_between(freqs, mean - std, mean + std, alpha=0.2, color=c)
        ax.set_title(cn, fontsize=18, fontweight='bold')
        ax.tick_params(axis='both', labelsize=14)
    fig.text(0.5, 0.04, 'Frequency (Hz)', ha='center', fontsize=18, fontweight='bold')
    fig.text(0.04, 0.5, 'Power (dB)', va='center', rotation='vertical', fontsize=18, fontweight='bold')
    fig.suptitle(f'{label} Retrieval Group Power by Stim x Memory', fontsize=20, fontweight='bold')
    handles, labels_ = [], []
    for ax in axes_flat:
        h, l = ax.get_legend_handles_labels()
        for hh, ll in zip(h, l):
            if ll not in labels_:
                handles.append(hh)
                labels_.append(ll)
    fig.legend(handles, labels_, bbox_to_anchor=(0.82, 0.5), loc='center left',
               prop={'weight': 'bold', 'size': 12})
    add_method_caption(
        fig,
        'Method: Each panel shows ROI mean retrieval spectra for one stim x memory condition. Within each condition, each patient is averaged across matching trials before ROI means and ±1 SD are computed.',
    )
    plt.tight_layout(rect=[0.06, 0.10, 0.82, 0.94])
    plt.savefig(os.path.join(out_dir, 'Quadrant_StimMemory_Power_retrieval.png'), dpi=300)
    finalize_figure()


def plot_per_roi_quadrant(data, out_dir, label):
    freqs = data['freqs_post']
    if freqs is None:
        return
    qc = {
        'Stim Remembered': data.get('stim_rem', {}),
        'Stim Forgotten': data.get('stim_forg', {}),
        'NoStim Remembered': data.get('nostim_rem', {}),
        'NoStim Forgotten': data.get('nostim_forg', {}),
    }
    qo = {'Stim Remembered': (0, 0), 'Stim Forgotten': (0, 1),
          'NoStim Remembered': (1, 0), 'NoStim Forgotten': (1, 1)}
    all_rois = sorted(set().union(*[d.keys() for d in qc.values()]))
    for roi in all_rois:
        if roi.startswith('PNAS'):
            continue
        roi_subjects = set()
        for cond_dict in qc.values():
            roi_subjects.update(cond_dict.get(roi, {}))
        subject_colors = make_subject_color_map(roi_subjects)
        fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
        has_data = False
        for cn, cd in qc.items():
            r, c = qo[cn]
            ax = axes[r, c]
            if roi not in cd or not cd[roi]:
                ax.set_title(f'{cn} (no data)', fontsize=14, fontweight='bold')
                ax.set_axis_off()
                continue
            has_data = True
            for subj, pv in sorted(cd[roi].items()):
                ax.plot(freqs, pv, label=subj, color=subject_colors[subj], alpha=0.7)
            ax.set_title(cn, fontsize=14, fontweight='bold')
            ax.tick_params(axis='both', labelsize=10)
        if not has_data:
            finalize_figure(fig)
            continue
        fig.suptitle(f'{label} Retrieval {roi} — Individual by Stim x Memory', fontsize=18, fontweight='bold')
        fig.text(0.5, 0.04, 'Frequency (Hz)', ha='center', fontsize=14, fontweight='bold')
        fig.text(0.04, 0.5, 'Power (dB)', va='center', rotation='vertical', fontsize=14, fontweight='bold')
        handles, labels_ = collect_unique_legend_items(axes.flatten())
        if handles:
            fig.legend(handles, labels_, loc='center left', bbox_to_anchor=(0.88, 0.5), fontsize=8)
        add_method_caption(
            fig,
            'Method: For this ROI, each line is one patient averaged across retrieval trials within the specified stim x memory condition.',
        )
        plt.tight_layout(rect=[0.06, 0.10, 0.84, 0.92])
        plt.savefig(os.path.join(out_dir, f'IndivQuadrant_{roi}_StimMemory_retrieval.png'), dpi=300)
        finalize_figure()


def plot_bc_quadrant_stim_memory(data, out_dir, label):
    """Baseline-corrected quadrant plot by stim x memory."""
    freqs = data['freqs_diff']
    if freqs is None:
        return
    mem_dicts = {
        'NoStim Remembered': data.get('bc_nostim_rem', {}),
        'NoStim Forgotten': data.get('bc_nostim_forg', {}),
        'Stim Remembered': data.get('bc_stim_rem', {}),
        'Stim Forgotten': data.get('bc_stim_forg', {}),
    }
    if not any(mem_dicts.values()):
        return
    all_rois = sorted(set().union(*[d.keys() for d in mem_dicts.values()]))
    gspr = {}
    for cd in mem_dicts.values():
        for roi, sd in cd.items():
            gspr.setdefault(roi, set()).update(sd.keys())

    cond_order = ['NoStim Remembered', 'NoStim Forgotten', 'Stim Remembered', 'Stim Forgotten']
    fig, axes = plt.subplots(2, 2, figsize=(18, 10), sharex=True, sharey=True)
    axes_flat = axes.flatten()
    plotted = set()
    for ax, cn in zip(axes_flat, cond_order):
        cd = mem_dicts[cn]
        for roi in all_rois:
            if roi not in cd:
                continue
            mat = np.array(list(cd[roi].values()), dtype=np.float64)
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
        ax.set_title(cn, fontsize=18, fontweight='bold')
        ax.tick_params(axis='both', labelsize=14)
        ax.axhline(0, color='lightgray', linestyle='--', linewidth=1)
    fig.text(0.5, 0.04, 'Frequency (Hz)', ha='center', fontsize=18, fontweight='bold')
    fig.text(0.04, 0.5, 'Baseline-Corrected Power (dB)', va='center', rotation='vertical', fontsize=18, fontweight='bold')
    fig.suptitle(f'{label} Baseline-Corrected Power by Stim x Memory', fontsize=20, fontweight='bold')
    handles, labels_ = [], []
    for ax in axes_flat:
        h, l = ax.get_legend_handles_labels()
        for hh, ll in zip(h, l):
            if ll not in labels_:
                handles.append(hh)
                labels_.append(ll)
    fig.legend(handles, labels_, bbox_to_anchor=(0.82, 0.5), loc='center left',
               prop={'weight': 'bold', 'size': 12})
    add_method_caption(
        fig,
        'Method: Each panel shows baseline-corrected ROI spectra for one stim x memory condition. Patients are averaged within condition first, then ROI means and ±1 SD are plotted.',
    )
    plt.tight_layout(rect=[0.06, 0.10, 0.82, 0.94])
    plt.savefig(os.path.join(out_dir, 'Quadrant_BaselineCorrected_StimMemory_Power_retrieval.png'),
                dpi=300, bbox_inches='tight')
    finalize_figure()


def plot_bc_bar_by_memory(data, out_dir, label):
    freqs = data['freqs_diff']
    if freqs is None:
        return
    mem_pairs = [
        ('remembered', data.get('bc_stim_rem', {}), data.get('bc_nostim_rem', {})),
        ('forgotten', data.get('bc_stim_forg', {}), data.get('bc_nostim_forg', {})),
    ]
    all_rows = []
    for mem, sd, nd in mem_pairs:
        df_diff = compute_diff_df(sd, nd, freqs, POWER_RANGES)
        if not df_diff.empty:
            df_diff['memory_cond'] = mem
            all_rows.append(df_diff)
    if not all_rows:
        return
    df_all = pd.concat(all_rows, ignore_index=True)
    df_all = df_all[~df_all['Region'].str.startswith('PNAS')]
    if df_all.empty:
        return
    mem_source_lookup = {mem: (sd, nd) for mem, sd, nd in mem_pairs}

    # Split by memory
    for mem, mem_label in [('remembered', 'Remembered'), ('forgotten', 'Forgotten')]:
        df_mem = df_all[df_all['memory_cond'] == mem]
        if df_mem.empty:
            continue
        unique_rois = [r for r in df_mem['Region'].unique() if r in ROI_COLORS_BAR]
        if not unique_rois:
            continue
        sd, nd = mem_source_lookup[mem]
        df_condition = compute_condition_band_df(sd, nd, freqs, POWER_RANGES)
        nostim_responder_df = df_condition.rename(columns={'nostim': 'mean_power_diff'})[
            ['Patient', 'Region', 'power_range', 'mean_power_diff']
        ].copy()
        nostim_responder_df = nostim_responder_df[nostim_responder_df['Region'].isin(unique_rois)]
        if not nostim_responder_df.empty:
            plot_responder_status_bargraph(
                nostim_responder_df,
                out_dir,
                f'{label} Retrieval Baseline-Corrected Power, No Stim {mem_label} Trials',
                'Baseline-Corrected Power',
                f'Bargraph_baseline_corrected_power_byROI_retrieval_nostim_{mem}_responder_status.png',
                caption_text=f'Method: For each patient and ROI, baseline-corrected spectra are averaged across no-stim {mem_label.lower()} retrieval trials, then band means are computed. Bars = mean across patients, error bars = SEM, dots = patient values colored by responder-status CSV.',
            )
        plot_responder_status_bargraph(
            df_mem[df_mem['Region'].isin(unique_rois)],
            out_dir,
            f'{label} Retrieval Baseline-Corrected Power Diff (Stim - No Stim), {mem_label} Trials',
            'Baseline-Corrected Power Diff',
            f'Bargraph_baseline_corrected_powerDiff_byROI_retrieval_{mem}_responder_status.png',
            caption_text=f'Method: For each patient and ROI, baseline-corrected spectra are averaged across stim and no-stim {mem_label.lower()} retrieval trials separately, band means are computed, then Stim-NoStim is taken. Bars = mean across patients, error bars = SEM, dots = patient values colored by responder-status CSV.',
        )
        palette = {r: ROI_COLORS_BAR.get(r, '#808080') for r in unique_rois}
        power_order = [band for band in POWER_RANGES if band in df_mem['power_range'].unique()]
        if not power_order:
            power_order = sorted(df_mem['power_range'].unique())
        for band in power_order:
            band_df = df_mem[df_mem['power_range'] == band].copy()
            if band_df.empty:
                continue
            fig, ax = plt.subplots(figsize=(11.5, 8.2))
            sns.barplot(
                data=band_df,
                x='Region',
                y='mean_power_diff',
                hue='Region',
                errorbar='se',
                palette=palette,
                legend=False,
                order=unique_rois,
                dodge=False,
                ax=ax,
            )
            sns.stripplot(
                data=band_df,
                x='Region',
                y='mean_power_diff',
                ax=ax,
                color='black',
                alpha=0.5,
                jitter=0.2,
                order=unique_rois,
            )
            ax.set_xlabel('')
            ax.set_ylabel('Power Diff', fontsize=16, fontweight='bold')
            ax.tick_params(axis='x', labelsize=14)
            ax.tick_params(axis='y', labelsize=14)
            for l in ax.get_xticklabels():
                l.set_fontweight('bold')
            ax.axhline(0, color='grey')
            ax.set_title(band, fontsize=18, fontweight='bold')
            fig.suptitle(f'{label} Retrieval Power Diff, {mem_label} Trials',
                         fontsize=20, fontweight='bold', y=0.98)
            add_method_caption(
                fig,
                f'Method: For each patient and ROI, band diff = Stim-NoStim using only {mem_label.lower()} retrieval trials. Bars = mean across patients, error bars = SEM, dots = patient values.',
            )
            fig.tight_layout(rect=[0, 0.05, 1, 0.9])
            plt.savefig(
                os.path.join(out_dir, band_filename(f'Bargraph_Power_diff_{mem}_retrieval.png', band)),
                bbox_inches='tight',
                dpi=300,
            )
            finalize_figure(fig)


def plot_bc_remembered_forgotten(data, out_dir, label):
    freqs = data['freqs_diff']
    if freqs is None:
        return
    mem_to_dicts = {
        'Remembered': {'stim': data.get('bc_stim_rem', {}), 'nostim': data.get('bc_nostim_rem', {})},
        'Forgotten': {'stim': data.get('bc_stim_forg', {}), 'nostim': data.get('bc_nostim_forg', {})},
    }
    all_rois = set()
    for md in mem_to_dicts.values():
        all_rois.update(md['stim'].keys())
        all_rois.update(md['nostim'].keys())
    for roi in sorted(all_rois):
        if roi.startswith('PNAS'):
            continue
        has_roi = any(roi in mem_to_dicts[ml]['stim'] or roi in mem_to_dicts[ml]['nostim']
                      for ml in mem_to_dicts)
        if not has_roi:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
        for ax, ml in zip(axes, ['Remembered', 'Forgotten']):
            for cn, cd, color in [('nostim', mem_to_dicts[ml]['nostim'], 'blue'),
                                  ('stim', mem_to_dicts[ml]['stim'], 'red')]:
                if roi not in cd:
                    continue
                mat = np.array(list(cd[roi].values()), dtype=np.float64)
                n = mat.shape[0]
                mean, se = mat.mean(0), mat.std(0) / np.sqrt(n)
                ax.plot(freqs, mean, color=color, label=f'{cn} ({n})')
                ax.fill_between(freqs, mean - se, mean + se, alpha=0.2, color=color)
            ax.set_title(f'{ml} Trials', fontsize=16, fontweight='bold')
            ax.set_xlabel('Frequency (Hz)', fontsize=16, fontweight='bold')
            ax.tick_params(axis='both', labelsize=14)
            ax.set_xlim(1, 100)
        axes[0].set_ylabel('Power (dB)', fontsize=16, fontweight='bold')
        handles, labels_ = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels_, loc='center left', bbox_to_anchor=(0.90, 0.5), fontsize=12)
        fig.suptitle(f'{label} Retrieval {roi} Baseline-Corrected Power', fontsize=18, fontweight='bold')
        add_method_caption(
            fig,
            'Method: For this ROI and memory split, each patient is averaged within no-stim and stim retrieval trials on baseline-corrected spectra. Lines show means across patients and shading shows SEM.',
        )
        plt.tight_layout(rect=[0, 0.08, 0.92, 0.94])
        plt.savefig(os.path.join(out_dir, f'{roi}_Baseline_Adjusted_Power_RememberedForgotten_retrieval.png'), dpi=300)
        finalize_figure()


def plot_connected_dots(data, out_dir, label, show_patients=True):
    """Connected dots plot showing stim vs nostim differences per patient."""
    freqs = data['freqs_diff']
    if freqs is None:
        return
    mem_pairs = [
        ('remembered', data.get('bc_stim_rem', {}), data.get('bc_nostim_rem', {})),
        ('forgotten', data.get('bc_stim_forg', {}), data.get('bc_nostim_forg', {})),
    ]

    # Build long-form data
    all_rows = []
    for mem, sd, nd in mem_pairs:
        all_rois_here = sorted(set(sd.keys()) | set(nd.keys()))
        for roi in all_rois_here:
            if roi.startswith('PNAS'):
                continue
            for subj in sorted(set(sd.get(roi, {}).keys()) & set(nd.get(roi, {}).keys())):
                for bn, (lo, hi) in POWER_RANGES.items():
                    mask = (freqs >= lo) & (freqs <= hi)
                    stim_val = sd[roi][subj][mask].mean()
                    nostim_val = nd[roi][subj][mask].mean()
                    all_rows.append({'Patient': subj, 'Region': roi, 'power_range': bn,
                                     'memory_cond': mem, 'stim': stim_val, 'nostim': nostim_val})

    if not all_rows:
        return

    df = pd.DataFrame(all_rows)
    band_order = list(POWER_RANGES.keys())
    clean_suffix = '_clean' if not show_patients else ''
    if show_patients:
        patients = sorted(df['Patient'].unique())
        filled_markers = ['o', 's', '^', 'D', 'v', 'P', 'X', '*', '<', '>', 'h', '8', 'p', 'H', 'd']
        marker_styles = ([(m, True) for m in filled_markers] +
                         [(m, False) for m in filled_markers] +
                         [('1', True), ('2', True), ('3', True), ('4', True),
                          ('+', True), ('x', True), ('|', True), ('_', True)] +
                         [(f'${i}$', True) for i in range(1, 11)])
        patient_markers = {p: marker_styles[i] for i, p in enumerate(patients)}

    for mem, mem_label in [('remembered', 'Remembered'), ('forgotten', 'Forgotten')]:
        df_mem = df[df['memory_cond'] == mem]
        if df_mem.empty:
            continue
        for band in band_order:
            df_band = df_mem[df_mem['power_range'] == band]
            if df_band.empty:
                continue
            roi_order = [r for r in ROI_COLORS_BAR if r in df_band['Region'].unique()]
            if not roi_order:
                continue

            fig, ax = plt.subplots(figsize=(16, 11))
            x_base = np.arange(len(roi_order))
            bar_width = 0.34
            offsets = {'nostim': -bar_width / 2, 'stim': bar_width / 2}
            bar_colors = {'nostim': '#1f77b4', 'stim': '#d62728'}

            for i, roi in enumerate(roi_order):
                roi_df = df_band[df_band['Region'] == roi]
                for trial in ['nostim', 'stim']:
                    vals = roi_df[trial].dropna()
                    if len(vals) == 0:
                        continue
                    mean = vals.mean()
                    sem = vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0
                    ax.bar(x_base[i] + offsets[trial], mean, width=bar_width,
                           color=bar_colors[trial],
                           edgecolor='black', linewidth=1.2, yerr=sem, capsize=4, zorder=1)

            if show_patients:
                for i, roi in enumerate(roi_order):
                    roi_df = df_band[df_band['Region'] == roi]
                    for _, row in roi_df.iterrows():
                        x_ns = x_base[i] + offsets['nostim']
                        x_s = x_base[i] + offsets['stim']
                        ax.plot([x_ns, x_s], [row['nostim'], row['stim']],
                                color='black', alpha=0.35, linewidth=1)
                        m, filled = patient_markers[row['Patient']]
                        if filled:
                            ax.scatter(x_ns, row['nostim'], marker=m, color='black', alpha=0.65, s=55, zorder=3)
                            ax.scatter(x_s, row['stim'], marker=m, color='black', alpha=0.95, s=55, zorder=3)
                        else:
                            ax.scatter(x_ns, row['nostim'], marker=m, facecolors='white', edgecolors='black',
                                       linewidths=1.1, alpha=0.85, s=60, zorder=3)
                            ax.scatter(x_s, row['stim'], marker=m, facecolors='white', edgecolors='black',
                                       linewidths=1.3, alpha=1.0, s=60, zorder=3)

            ax.set_xticks(x_base)
            ax.set_xticklabels(roi_order, fontsize=14, fontweight='bold')
            ax.tick_params(axis='x', labelsize=14, width=2, length=6)
            ax.tick_params(axis='y', labelsize=14, width=2, length=6)
            ax.axhline(0, color='grey')
            ax.set_ylabel('Mean Baseline-Corrected Power (dB)', fontsize=17, fontweight='bold')
            fig.suptitle(f'{band} \u2014 {mem_label}', fontsize=22, fontweight='bold', y=0.965)

            y_values = pd.concat([df_band['nostim'], df_band['stim']], ignore_index=True).dropna()
            if not y_values.empty:
                y_min = y_values.min()
                y_max = y_values.max()
                if np.isclose(y_min, y_max):
                    ax.set_ylim(y_min - 0.1, y_max + 0.1)
                else:
                    y_pad = 0.25 * (y_max - y_min)
                    ax.set_ylim(y_min - y_pad, y_max + y_pad)

            stim_legend = [
                Line2D([0], [0], color=bar_colors['stim'], lw=8, label='Stim'),
                Line2D([0], [0], color=bar_colors['nostim'], lw=8, label='No Stim'),
            ]
            fig.legend(handles=stim_legend, loc='upper center', bbox_to_anchor=(0.50, 0.925),
                       ncol=2, frameon=False, fontsize=12, title='Condition', title_fontsize=13,
                       handlelength=1.8, handletextpad=0.6, columnspacing=1.2, borderaxespad=0.2)

            if show_patients:
                patient_handles = [
                    Line2D([0], [0], marker=patient_markers[p][0], color='black', linestyle='None',
                           markerfacecolor='black' if patient_markers[p][1] else 'white',
                           markeredgecolor='black', markeredgewidth=1.1, markersize=8, label=str(p))
                    for p in sorted(df_band['Patient'].unique())
                ]
                fig.legend(handles=patient_handles, loc='center left', bbox_to_anchor=(0.81, 0.50),
                           ncol=2, columnspacing=0.9, handletextpad=0.4, labelspacing=0.45,
                           frameon=False, fontsize=10, title='Patient', title_fontsize=12)

            add_method_caption(
                fig,
                'Method: For each patient, ROI, band, and memory condition, separate no-stim and stim band means are computed from baseline-corrected spectra and connected by a line. Bars show across-patient means with SEM.',
            )
            fig.subplots_adjust(left=0.10, right=0.78 if show_patients else 0.95, bottom=0.15, top=0.86)
            plt.savefig(os.path.join(out_dir, band_filename(f'StimVsNoStim_{mem}_retrieval{clean_suffix}.png', band)),
                        bbox_inches='tight', dpi=300)
            print(f"\n{label}: N subjects per ROI for {band} {mem_label}")
            print(df_band.groupby(['Region', 'Patient']).size().reset_index().groupby('Region')['Patient'].nunique())
            finalize_figure()


CORE_REGIONS = {'BLA', 'CA', 'DG', 'EC', 'HPC', 'PRC'}

CORE_ROI_COLORS_SPEC = {
    'BLA': 'teal', 'CA': 'black', 'DG': 'skyblue',
    'EC': '#8A2BE2', 'HPC': 'orange', 'PRC': '#E61C59',
}


def plot_power_by_roi_core(data, out_dir, label):
    """1/f spectral plot limited to BLA, CA, DG, EC, HPC, PRC."""
    gap = get_overall_power_for_plot(data)
    freqs = data['freqs_post']
    if not gap or freqs is None:
        return
    fig, ax = plt.subplots(figsize=(12, 8))
    for roi in sorted(gap.keys()):
        if roi not in CORE_REGIONS:
            continue
        mat = np.array(list(gap[roi].values()), dtype=np.float64)
        mean, std = mat.mean(0), mat.std(0)
        c = CORE_ROI_COLORS_SPEC.get(roi, 'gray')
        ax.plot(freqs, mean, color=c, label=f'{roi} ({mat.shape[0]})')
        ax.fill_between(freqs, mean - std, mean + std, alpha=0.2, color=c)
    ax.set_xlabel('Frequency (Hz)', fontsize=18, fontweight='bold')
    ax.set_ylabel('Power (dB)', fontsize=18, fontweight='bold')
    ax.set_title(f'{label} Retrieval Group Power by ROI (Core Regions)', fontsize=20, fontweight='bold')
    ax.tick_params(axis='both', labelsize=14)
    ax.legend(bbox_to_anchor=(1.02, 0.5), loc='center left', prop={'weight': 'bold', 'size': 12})
    add_method_caption(
        fig,
        "Method: Each ROI line is the mean retrieval spectrum across patients after averaging each patient's available stim/no-stim and remembered/forgotten spectra. Shading shows ±1 SD across patients. Limited to BLA, CA, DG, EC, HPC, PRC.",
    )
    plt.tight_layout(rect=[0, 0.05, 0.85, 1])
    plt.savefig(os.path.join(out_dir, 'GroupPower_byROI_retrieval_core_regions.png'), dpi=300, bbox_inches='tight')
    finalize_figure()


def plot_freq_x_memory_power_bargraph(data, out_dir, label):
    """Frequency x Memory (Power) boxplots separated by band with remembered/forgotten strips."""
    freqs = data['freqs_diff']
    if freqs is None:
        return
    mem_pairs = [
        ('Remembered', data.get('bc_stim_rem', {}), data.get('bc_nostim_rem', {})),
        ('Forgotten', data.get('bc_stim_forg', {}), data.get('bc_nostim_forg', {})),
    ]
    all_rows = []
    for mem_label, sd, nd in mem_pairs:
        df_diff = compute_diff_df(sd, nd, freqs, POWER_RANGES)
        if not df_diff.empty:
            df_diff['memory_cond'] = mem_label
            all_rows.append(df_diff)
    if not all_rows:
        return
    df_all = pd.concat(all_rows, ignore_index=True)
    df_all = df_all[df_all['Region'].isin(CORE_REGIONS)]
    if df_all.empty:
        return

    roi_order = sorted(df_all['Region'].unique())
    band_order = [b for b in POWER_RANGES if b in df_all['power_range'].unique()]
    if not band_order:
        return

    band_colors = {'Theta': '#4C72B0', 'Slow gamma': '#DD8452'}
    memory_styles = {
        'Remembered': {'facecolor': 'black', 'edgecolor': 'black', 'alpha': 0.7},
        'Forgotten': {'facecolor': 'white', 'edgecolor': 'black', 'alpha': 0.9},
    }

    fig, axes = plt.subplots(
        1,
        len(band_order),
        figsize=(max(12.5, len(roi_order) * 2.2 * len(band_order)), 8.2),
        sharey=True,
    )
    if len(band_order) == 1:
        axes = [axes]
    rng = np.random.default_rng(42)

    for ax, band in zip(axes, band_order):
        band_df = df_all[df_all['power_range'] == band]
        if band_df.empty:
            continue
        x = np.arange(len(roi_order))

        for mem_label, style in memory_styles.items():
            sub = band_df[band_df['memory_cond'] == mem_label]
            for j, roi in enumerate(roi_order):
                pts = sub[sub['Region'] == roi]['mean_power_diff']
                if pts.empty:
                    continue
                jitter = rng.uniform(-0.14, 0.14, len(pts))
                ax.scatter(
                    np.full(len(pts), x[j]) + jitter,
                    pts,
                    marker='o',
                    s=34,
                    facecolors=style['facecolor'],
                    edgecolors=style['edgecolor'],
                    linewidths=1.2,
                    zorder=2,
                    alpha=style['alpha'],
                )

        grouped = [
            band_df[band_df['Region'] == roi]['mean_power_diff'].to_numpy()
            for roi in roi_order
            if not band_df[band_df['Region'] == roi].empty
        ]
        present_rois = [roi for roi in roi_order if not band_df[band_df['Region'] == roi].empty]
        present_positions = [roi_order.index(roi) for roi in present_rois]
        if grouped:
            bp = ax.boxplot(
                grouped,
                positions=present_positions,
                widths=0.42,
                patch_artist=True,
                showfliers=False,
                medianprops={'color': 'black', 'linewidth': 2.0, 'zorder': 5},
                whiskerprops={'color': 'black', 'linewidth': 1.4, 'zorder': 4},
                capprops={'color': 'black', 'linewidth': 1.4, 'zorder': 4},
                boxprops={'edgecolor': 'black', 'linewidth': 1.5, 'zorder': 4},
            )
            for box in bp['boxes']:
                box.set_facecolor(band_colors.get(band, 'gray'))
                box.set_alpha(0.78)
                box.set_zorder(4)

        ax.set_xticks(x)
        ax.set_xticklabels(roi_order, fontsize=14, fontweight='bold')
        ax.tick_params(axis='y', labelsize=14)
        ax.axhline(0, color='grey', linewidth=0.8, zorder=1)
        ax.set_title(band, fontsize=17, fontweight='bold')

    axes[0].set_ylabel('Power Diff (Stim − No Stim)', fontsize=16, fontweight='bold')

    mem_handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markeredgecolor='black', markersize=7, label='Remembered'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='white', markeredgecolor='black', markeredgewidth=1.2, markersize=7, label='Forgotten'),
    ]
    fig.legend(handles=mem_handles, labels=['Remembered', 'Forgotten'],
               loc='upper right', bbox_to_anchor=(0.98, 0.86), fontsize=11, framealpha=0.9)

    fig.suptitle(f'{label} Retrieval Frequency × Memory (Power) Boxplots',
                 fontsize=20, fontweight='bold', y=0.98)
    add_method_caption(
        fig,
        'Method: For each patient and ROI, band diff = Stim−NoStim. Panels separate theta and slow gamma. '
        'Boxes summarize the across-patient distribution within each ROI. '
        'Black circles = remembered trials, light gray-outline circles = forgotten trials. '
        'Core regions only (BLA, CA, DG, EC, HPC, PRC).',
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.93])
    plt.savefig(os.path.join(out_dir, 'Frequency_x_Memory_Power_bargraph.png'), dpi=300, bbox_inches='tight')
    finalize_figure(fig)


# =========================================================================
#  MAIN
# =========================================================================

def generate_common_plots(data, out_dir, label):
    ensure_dir(out_dir)
    print(f"\n  Generating common plots for {label}...")
    plot_power_by_roi(data, out_dir, label)
    plot_power_by_roi_core(data, out_dir, label)
    plot_power_by_patient(data, out_dir, label)
    plot_stim_vs_nostim(data, out_dir, label)
    plot_per_roi_patient_stim_nostim(data, out_dir, label)
    plot_bc_bar_graph(data, out_dir, label)
    plot_bc_per_roi(data, out_dir, label)


def generate_memory_plots(data, out_dir, label):
    if not data.get('has_memory'):
        return
    ensure_dir(out_dir)
    print(f"  Generating memory plots for {label}...")
    plot_quadrant_stim_memory(data, out_dir, label)
    plot_per_roi_quadrant(data, out_dir, label)
    plot_bc_quadrant_stim_memory(data, out_dir, label)
    plot_bc_bar_by_memory(data, out_dir, label)
    plot_bc_remembered_forgotten(data, out_dir, label)
    plot_connected_dots(data, out_dir, label)
    plot_connected_dots(data, out_dir, label, show_patients=False)
    plot_freq_x_memory_power_bargraph(data, out_dir, label)

def build_all_retrieval_data(blaes, amme):
    all_freqs_post = None
    if blaes['freqs_post'] is not None and amme['freqs_post'] is not None:
        if np.array_equal(blaes['freqs_post'], amme['freqs_post']):
            all_freqs_post = blaes['freqs_post']
        else:
            print("  WARNING: Different post frequencies. Using BLAES frequencies for 'all'.")
            all_freqs_post = blaes['freqs_post']
    else:
        all_freqs_post = blaes['freqs_post'] if blaes['freqs_post'] is not None else amme['freqs_post']

    all_freqs_diff = None
    if blaes['freqs_diff'] is not None and amme['freqs_diff'] is not None:
        if np.array_equal(blaes['freqs_diff'], amme['freqs_diff']):
            all_freqs_diff = blaes['freqs_diff']
        else:
            print("  WARNING: Different diff frequencies. Using BLAES frequencies for 'all'.")
            all_freqs_diff = blaes['freqs_diff']
    else:
        all_freqs_diff = blaes['freqs_diff'] if blaes['freqs_diff'] is not None else amme['freqs_diff']

    return augment_with_power_composites({
        'group_all_power': merge_dicts(blaes['group_all_power'], amme['group_all_power']),
        'stim_power': merge_dicts(blaes['stim_power'], amme['stim_power']),
        'nostim_power': merge_dicts(blaes['nostim_power'], amme['nostim_power']),
        'bc_stim': merge_dicts(blaes['bc_stim'], amme['bc_stim']),
        'bc_nostim': merge_dicts(blaes['bc_nostim'], amme['bc_nostim']),
        'freqs_post': all_freqs_post,
        'freqs_diff': all_freqs_diff,
        'has_memory': True,
        'use_memory_collapsed_overall': True,
        'bc_bar_exclude_rois': merge_sets(blaes.get('bc_bar_exclude_rois'), amme.get('bc_bar_exclude_rois')),
        'bc_memory_collapsed_exclude_rois': merge_sets(
            blaes.get('bc_memory_collapsed_exclude_rois'),
            amme.get('bc_memory_collapsed_exclude_rois'),
        ),
        'stim_rem': merge_dicts(blaes.get('stim_rem', {}), amme.get('stim_rem', {})),
        'stim_forg': merge_dicts(blaes.get('stim_forg', {}), amme.get('stim_forg', {})),
        'nostim_rem': merge_dicts(blaes.get('nostim_rem', {}), amme.get('nostim_rem', {})),
        'nostim_forg': merge_dicts(blaes.get('nostim_forg', {}), amme.get('nostim_forg', {})),
        'bc_stim_rem': merge_dicts(blaes.get('bc_stim_rem', {}), amme.get('bc_stim_rem', {})),
        'bc_stim_forg': merge_dicts(blaes.get('bc_stim_forg', {}), amme.get('bc_stim_forg', {})),
        'bc_nostim_rem': merge_dicts(blaes.get('bc_nostim_rem', {}), amme.get('bc_nostim_rem', {})),
        'bc_nostim_forg': merge_dicts(blaes.get('bc_nostim_forg', {}), amme.get('bc_nostim_forg', {})),
        'mlmr_export_frames': [
            df.copy()
            for df in blaes.get('mlmr_export_frames', []) + amme.get('mlmr_export_frames', [])
        ],
    })


# %% [markdown]
# Workflow Setup

# %%
if __name__ == '__main__':
    print("=" * 60)
    print("Combined Group Power Analysis - Retrieval")
    print("=" * 60)

    print("\nLoading BLAES retrieval data...")
    blaes = load_blaes_retrieval()

    print("\nLoading AMME retrieval data...")
    amme = load_amme_retrieval()

    # Per-group plots
    reset_dir(OUTPUT_BASE)
    blaes_dir = ensure_dir(os.path.join(OUTPUT_BASE, 'blaes'))
    amme_dir = ensure_dir(os.path.join(OUTPUT_BASE, 'amme'))
    all_dir = ensure_dir(os.path.join(OUTPUT_BASE, 'all'))

    print("\nMerging data for all-combined analysis...")
    all_data = build_all_retrieval_data(blaes, amme)

    print("\nExporting retrieval MLMR CSVs...")
    ensure_dir(CSV_OUTPUT_DIR)
    export_retrieval_mlmr_csv(blaes, CSV_OUTPUT_DIR, 'combined_retrieval_power_blaes_mlmr_input', measure_name='Power')
    export_retrieval_mlmr_csv(amme, CSV_OUTPUT_DIR, 'combined_retrieval_power_amme_mlmr_input', measure_name='Power')
    export_retrieval_mlmr_csv(all_data, CSV_OUTPUT_DIR, 'combined_retrieval_power_all_mlmr_input', measure_name='Power')


# %% [markdown]
# BLAES Common Plots

# %%
if __name__ == '__main__':
    plot_power_by_roi(blaes, blaes_dir, 'BLAES')

# %%
if __name__ == '__main__':
    plot_power_by_roi_core(blaes, blaes_dir, 'BLAES')

# %%
if __name__ == '__main__':
    plot_power_by_patient(blaes, blaes_dir, 'BLAES')

# %%
if __name__ == '__main__':
    plot_stim_vs_nostim(blaes, blaes_dir, 'BLAES')

# %%
if __name__ == '__main__':
    plot_per_roi_patient_stim_nostim(blaes, blaes_dir, 'BLAES')

# %%
if __name__ == '__main__':
    plot_bc_bar_graph(blaes, blaes_dir, 'BLAES')

# %%
if __name__ == '__main__':
    plot_bc_per_roi(blaes, blaes_dir, 'BLAES')


# %% [markdown]
# BLAES Memory Plots

# %%
if __name__ == '__main__':
    plot_quadrant_stim_memory(blaes, blaes_dir, 'BLAES')

# %%
if __name__ == '__main__':
    plot_per_roi_quadrant(blaes, blaes_dir, 'BLAES')

# %%
if __name__ == '__main__':
    plot_bc_quadrant_stim_memory(blaes, blaes_dir, 'BLAES')

# %%
if __name__ == '__main__':
    plot_bc_bar_by_memory(blaes, blaes_dir, 'BLAES')

# %%
if __name__ == '__main__':
    plot_bc_remembered_forgotten(blaes, blaes_dir, 'BLAES')

# %%
if __name__ == '__main__':
    plot_connected_dots(blaes, blaes_dir, 'BLAES')

# %%
if __name__ == '__main__':
    plot_freq_x_memory_power_bargraph(blaes, blaes_dir, 'BLAES')


# %% [markdown]
# AMME Common Plots

# %%
if __name__ == '__main__':
    plot_power_by_roi(amme, amme_dir, 'AMME')

# %%
if __name__ == '__main__':
    plot_power_by_roi_core(amme, amme_dir, 'AMME')

# %%
if __name__ == '__main__':
    plot_power_by_patient(amme, amme_dir, 'AMME')

# %%
if __name__ == '__main__':
    plot_stim_vs_nostim(amme, amme_dir, 'AMME')

# %%
if __name__ == '__main__':
    plot_per_roi_patient_stim_nostim(amme, amme_dir, 'AMME')

# %%
if __name__ == '__main__':
    plot_bc_bar_graph(amme, amme_dir, 'AMME')

# %%
if __name__ == '__main__':
    plot_bc_per_roi(amme, amme_dir, 'AMME')


# %% [markdown]
# AMME Memory Plots

# %%
if __name__ == '__main__':
    plot_quadrant_stim_memory(amme, amme_dir, 'AMME')

# %%
if __name__ == '__main__':
    plot_per_roi_quadrant(amme, amme_dir, 'AMME')

# %%
if __name__ == '__main__':
    plot_bc_quadrant_stim_memory(amme, amme_dir, 'AMME')

# %%
if __name__ == '__main__':
    plot_bc_bar_by_memory(amme, amme_dir, 'AMME')

# %%
if __name__ == '__main__':
    plot_bc_remembered_forgotten(amme, amme_dir, 'AMME')

# %%
if __name__ == '__main__':
    plot_connected_dots(amme, amme_dir, 'AMME')

# %%
if __name__ == '__main__':
    plot_freq_x_memory_power_bargraph(amme, amme_dir, 'AMME')


# %% [markdown]
# All Common Plots

# %%
if __name__ == '__main__':
    plot_power_by_roi(all_data, all_dir, 'All')

# %%
if __name__ == '__main__':
    plot_power_by_roi_core(all_data, all_dir, 'All')

# %%
if __name__ == '__main__':
    plot_power_by_patient(all_data, all_dir, 'All')

# %%
if __name__ == '__main__':
    plot_stim_vs_nostim(all_data, all_dir, 'All')

# %%
if __name__ == '__main__':
    plot_per_roi_patient_stim_nostim(all_data, all_dir, 'All')

# %%
if __name__ == '__main__':
    plot_bc_bar_graph(all_data, all_dir, 'All')

# %%
if __name__ == '__main__':
    plot_bc_per_roi(all_data, all_dir, 'All')


# %% [markdown]
# All Memory Plots

# %%
if __name__ == '__main__':
    plot_quadrant_stim_memory(all_data, all_dir, 'All')

# %%
if __name__ == '__main__':
    plot_per_roi_quadrant(all_data, all_dir, 'All')

# %%
if __name__ == '__main__':
    plot_bc_quadrant_stim_memory(all_data, all_dir, 'All')

# %%
if __name__ == '__main__':
    plot_bc_bar_by_memory(all_data, all_dir, 'All')

# %%
if __name__ == '__main__':
    plot_bc_remembered_forgotten(all_data, all_dir, 'All')

# %%
if __name__ == '__main__':
    plot_connected_dots(all_data, all_dir, 'All')

# %%
if __name__ == '__main__':
    plot_freq_x_memory_power_bargraph(all_data, all_dir, 'All')


# %% [markdown]
# Finish

# %%
if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("Done! Retrieval outputs saved to:", OUTPUT_BASE)
    print("=" * 60)
