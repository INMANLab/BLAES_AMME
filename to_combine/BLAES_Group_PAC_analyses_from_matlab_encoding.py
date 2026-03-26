#!/usr/bin/env python
# %% [markdown]
# ## Phase-Amplitude Coupling visualizations
#
# Author: Martina Hollearn
#
# Updated to generate BLAES encoding PAC visualizations and summary tables in
# `outputs/PAC_encoding`.

# %%
import os
import glob
import shutil
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib


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
else:
    try:
        from IPython import get_ipython
        shell = get_ipython()
        if shell is not None:
            shell.run_line_magic('matplotlib', 'inline')
            shell.run_line_magic('config', "InlineBackend.figure_format = 'retina'")
    except Exception:
        pass

import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
sns.set_theme(style='white')
matplotlib.rcParams['axes.grid'] = False
matplotlib.rcParams['grid.alpha'] = 0.0


# %% [markdown]
# ## Configuration

# %%
def get_repo_dir():
    if '__file__' in globals():
        script_path = Path(__file__).resolve()
        return script_path.parent.parent if script_path.parent.name == 'to_combine' else script_path.parent
    cwd = Path.cwd().resolve()
    return cwd.parent if cwd.name == 'to_combine' else cwd


REPO_DIR = get_repo_dir()
OUTPUT_DIR = REPO_DIR / 'outputs' / 'PAC_encoding'
CSV_DIR = OUTPUT_DIR / 'csvs'

PROJECT_PATH = Path('/Users/martinahollearn/Library/CloudStorage/Box-Box/InmanLab/BLAES_data/dissertation/LFP_analyses')
DATA_PATH = PROJECT_PATH / 'Results_CSVOutput'
PAC_FILES = sorted(glob.glob(str(DATA_PATH / '*phase1*PAC_TG*.csv')))

PAC_BANDS = {
    'Slow gamma': (35, 50),
}

MEMORY_LABELS = {
    'Old': 'remembered',
    'New': 'forgotten',
}


# %% [markdown]
# ## Helpers

# %%
def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)
    return Path(path)


def finalize_figure(fig=None):
    if IN_NOTEBOOK:
        plt.show()
    plt.close(fig)


def save_figure_output(fig, out_path, footer_text=None, rect=None, bbox_inches='tight'):
    if footer_text:
        fig.text(0.015, 0.015, footer_text, ha='left', va='bottom', fontsize=9, wrap=True)
        if rect is None:
            rect = [0, 0.08, 1, 1]
    if rect is None:
        fig.tight_layout()
    else:
        fig.tight_layout(rect=rect)
    fig.savefig(out_path, dpi=300, bbox_inches=bbox_inches)
    finalize_figure(fig)


def sorted_freq_cols(df, prefix):
    cols = [col for col in df.columns if col.startswith(prefix)]
    return sorted(cols, key=lambda col: float(col.split(prefix, 1)[1]))


def freq_values(freq_cols, prefix):
    return np.array([float(col.split(prefix, 1)[1]) for col in freq_cols], dtype=float)


def clean_region_label(region):
    parts = [part.strip() for part in str(region).split('_') if part.strip()]
    cleaned = ['EC' if part == 'ER' else part for part in parts]
    if len(cleaned) == 2:
        cleaned = sorted(cleaned)
    return '_'.join(cleaned)


def is_same_region_comparison(region):
    parts = str(region).split('_')
    return len(parts) == 2 and parts[0] == parts[1]


def has_excluded_region_token(region):
    return 'PNAS' in str(region).split('_')


def make_subject_color_map(subjects):
    ordered = sorted(subjects)
    palette = sns.color_palette('husl', len(ordered)) if ordered else []
    return {subject: color for subject, color in zip(ordered, palette)}


def make_roi_color_map(rois):
    ordered = sorted(rois)
    palette = sns.color_palette('husl', len(ordered)) if ordered else []
    return {roi: color for roi, color in zip(ordered, palette)}


def collect_unique_legend_items(axes):
    seen = set()
    handles = []
    labels = []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        for handle, label in zip(h, l):
            if not label or label in seen:
                continue
            seen.add(label)
            handles.append(handle)
            labels.append(label)
    return handles, labels


def style_axis(ax, hide_top_right=False):
    ax.grid(False)
    ax.set_facecolor('white')
    if hide_top_right:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)


def add_grouped_vectors(target, df, group_cols, value_cols):
    if df.empty:
        return
    grouped = df.groupby(group_cols)[value_cols].mean().reset_index()
    for _, row in grouped.iterrows():
        subject = row['Patient']
        region = row['Region']
        target.setdefault(region, {})[subject] = row[value_cols].to_numpy(dtype=np.float64)


def build_band_summary_rows(source_dict, freqs, bands, condition, measure_name):
    rows = []
    if not source_dict or freqs is None:
        return rows
    for region, subject_dict in source_dict.items():
        for patient, values in subject_dict.items():
            for band_name, (low, high) in bands.items():
                mask = (freqs >= low) & (freqs <= high)
                if not mask.any():
                    continue
                rows.append({
                    'Measure': measure_name,
                    'Patient': patient,
                    'Region': region,
                    'condition': condition,
                    'band': band_name,
                    'mean_value': float(np.nanmean(values[mask])),
                })
    return rows


def compute_condition_diff_df(stim_dict, nostim_dict, freqs, bands, value_name='mean_pac_diff'):
    rows = []
    if not stim_dict or not nostim_dict or freqs is None:
        return pd.DataFrame(columns=['Patient', 'Region', 'band', value_name])
    all_rois = sorted(set(stim_dict.keys()) | set(nostim_dict.keys()))
    for roi in all_rois:
        shared_subjects = sorted(set(stim_dict.get(roi, {})) & set(nostim_dict.get(roi, {})))
        for patient in shared_subjects:
            diff_vec = np.asarray(stim_dict[roi][patient], dtype=np.float64) - np.asarray(nostim_dict[roi][patient], dtype=np.float64)
            for band_name, (low, high) in bands.items():
                mask = (freqs >= low) & (freqs <= high)
                if not mask.any():
                    continue
                rows.append({
                    'Patient': patient,
                    'Region': roi,
                    'band': band_name,
                    value_name: float(np.nanmean(diff_vec[mask])),
                })
    return pd.DataFrame(rows)


def compute_condition_band_df(stim_dict, nostim_dict, freqs, bands):
    rows = []
    if not stim_dict or not nostim_dict or freqs is None:
        return pd.DataFrame(columns=['Patient', 'Region', 'band', 'stim', 'nostim'])
    all_rois = sorted(set(stim_dict.keys()) | set(nostim_dict.keys()))
    for roi in all_rois:
        shared_subjects = sorted(set(stim_dict.get(roi, {})) & set(nostim_dict.get(roi, {})))
        for patient in shared_subjects:
            stim_vec = np.asarray(stim_dict[roi][patient], dtype=np.float64)
            nostim_vec = np.asarray(nostim_dict[roi][patient], dtype=np.float64)
            for band_name, (low, high) in bands.items():
                mask = (freqs >= low) & (freqs <= high)
                if not mask.any():
                    continue
                rows.append({
                    'Patient': patient,
                    'Region': roi,
                    'band': band_name,
                    'stim': float(np.nanmean(stim_vec[mask])),
                    'nostim': float(np.nanmean(nostim_vec[mask])),
                })
    return pd.DataFrame(rows)


def plot_grouped_condition_bars(
    condition_df,
    out_dir,
    label,
    phase_label,
    filename_template,
    title_suffix,
    footer_text=None,
):
    if condition_df.empty:
        return
    roi_order = sorted(condition_df['Region'].unique())
    if not roi_order:
        return
    bar_colors = {'nostim': '#1f77b4', 'stim': '#d62728'}

    for band in [name for name in PAC_BANDS if name in condition_df['band'].unique()]:
        band_df = condition_df[condition_df['band'] == band].copy()
        if band_df.empty:
            continue
        fig, ax = plt.subplots(figsize=(13, 8.5))
        x_base = np.arange(len(roi_order))
        bar_width = 0.34
        offsets = {'nostim': -bar_width / 2, 'stim': bar_width / 2}

        for i, roi in enumerate(roi_order):
            roi_df = band_df[band_df['Region'] == roi]
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
                    label='No Stim' if trial == 'nostim' and i == 0 else ('Stim' if trial == 'stim' and i == 0 else None),
                )

        for i, roi in enumerate(roi_order):
            roi_df = band_df[band_df['Region'] == roi]
            for _, row in roi_df.iterrows():
                x_ns = x_base[i] + offsets['nostim']
                x_s = x_base[i] + offsets['stim']
                ax.plot([x_ns, x_s], [row['nostim'], row['stim']], color='black', alpha=0.28, linewidth=1)
                ax.scatter(x_ns, row['nostim'], color='black', alpha=0.65, s=38, zorder=3)
                ax.scatter(x_s, row['stim'], color='black', alpha=0.9, s=38, zorder=3)

        ax.axhline(0, color='gray', linewidth=1.2)
        ax.set_xlabel('')
        ax.set_ylabel('Mean Baseline-corrected PAC', fontsize=16, fontweight='bold')
        ax.set_title(f'{band} - {title_suffix}', fontsize=18, fontweight='bold')
        ax.set_xticks(x_base)
        ax.set_xticklabels(roi_order, rotation=45, ha='right')
        ax.tick_params(axis='x', labelsize=10)
        ax.tick_params(axis='y', labelsize=12)
        style_axis(ax, hide_top_right=True)
        ax.legend(frameon=False, fontsize=11, loc='upper right')
        fig.suptitle(f'{label} {phase_label} Baseline-corrected PAC - Stim vs No Stim', fontsize=20, fontweight='bold', y=0.98)
        save_figure_output(
            fig,
            out_dir / filename_template.format(band=band.lower().replace(' ', '_')),
            footer_text=footer_text,
            rect=[0, 0, 1, 0.92],
        )


def normalize_trial_type_amme(ttype):
    if isinstance(ttype, str) and 'stim' in ttype.lower() and ttype.lower() != 'nostim':
        return 'stim'
    if isinstance(ttype, str) and ttype.lower() == 'nostim':
        return 'nostim'
    return None


# %% [markdown]
# ## Load and organize PAC data

# %%
def load_blaes_encoding_pac():
    data = {
        'post_all': {},
        'post_stim': {},
        'post_nostim': {},
        'post_stim_rem': {},
        'post_stim_forg': {},
        'post_nostim_rem': {},
        'post_nostim_forg': {},
        'diff_stim': {},
        'diff_nostim': {},
        'diff_stim_rem': {},
        'diff_stim_forg': {},
        'diff_nostim_rem': {},
        'diff_nostim_forg': {},
        'freqs_post': None,
        'freqs_diff': None,
        'trial_count_rows': [],
    }

    for pac_file in PAC_FILES:
        df = pd.read_csv(pac_file)

        base_required_cols = {'Patient', 'Region', 'stimulation'}
        if not base_required_cols.issubset(df.columns):
            print(f"Skipping {os.path.basename(pac_file)}; missing required columns.")
            continue

        df = df.copy()
        df['Region'] = df['Region'].map(clean_region_label)
        df['stimulation'] = pd.to_numeric(df['stimulation'], errors='coerce')
        df = df[df['stimulation'].isin([0, 1])].copy()
        if 'ret_response' in df.columns:
            df['memory_cond'] = df['ret_response'].map(MEMORY_LABELS)
            df = df[df['memory_cond'].isin(['remembered', 'forgotten'])].copy()
        elif 'test_yes_or_no' in df.columns:
            if 'test_trial_type' in df.columns:
                df['test_trial_type'] = df['test_trial_type'].apply(normalize_trial_type_amme)
            df = df[df['test_yes_or_no'].isin(['yes', 'no'])].copy()
            df['memory_cond'] = np.where(df['test_yes_or_no'] == 'yes', 'remembered', 'forgotten')
        else:
            print(f"Skipping {os.path.basename(pac_file)}; missing encoding PAC memory columns.")
            continue
        df = df[~df['Region'].map(is_same_region_comparison)].copy()
        df = df[~df['Region'].map(has_excluded_region_token)].copy()
        if df.empty:
            print(f"Skipping {os.path.basename(pac_file)}; no eligible cross-region PAC rows after filtering.")
            continue

        pre_cols = sorted_freq_cols(df, 'pre_Freq_')
        post_cols = sorted_freq_cols(df, 'post_Freq_')
        if not pre_cols or not post_cols:
            print(f"Skipping {os.path.basename(pac_file)}; missing pre/post frequency columns.")
            continue

        shared_freqs = sorted(set(freq_values(pre_cols, 'pre_Freq_')) & set(freq_values(post_cols, 'post_Freq_')))
        if not shared_freqs:
            print(f"Skipping {os.path.basename(pac_file)}; no shared PAC frequencies.")
            continue

        pre_cols = [f'pre_Freq_{int(freq) if float(freq).is_integer() else freq}' for freq in shared_freqs]
        post_cols = [f'post_Freq_{int(freq) if float(freq).is_integer() else freq}' for freq in shared_freqs]
        diff_cols = []
        for freq, pre_col, post_col in zip(shared_freqs, pre_cols, post_cols):
            diff_col = f'diff_Freq_{int(freq) if float(freq).is_integer() else freq}'
            df[diff_col] = df[post_col] - df[pre_col]
            diff_cols.append(diff_col)

        if data['freqs_post'] is None:
            data['freqs_post'] = np.array(shared_freqs, dtype=float)
        if data['freqs_diff'] is None:
            data['freqs_diff'] = np.array(shared_freqs, dtype=float)

        patient = str(df['Patient'].iloc[0])
        print(f"Processing {os.path.basename(pac_file)} [{patient}]")

        add_grouped_vectors(data['post_all'], df, ['Patient', 'Region'], post_cols)
        add_grouped_vectors(data['post_stim'], df[df['stimulation'] == 1], ['Patient', 'Region'], post_cols)
        add_grouped_vectors(data['post_nostim'], df[df['stimulation'] == 0], ['Patient', 'Region'], post_cols)

        add_grouped_vectors(
            data['post_stim_rem'],
            df[(df['stimulation'] == 1) & (df['memory_cond'] == 'remembered')],
            ['Patient', 'Region'],
            post_cols,
        )
        add_grouped_vectors(
            data['post_stim_forg'],
            df[(df['stimulation'] == 1) & (df['memory_cond'] == 'forgotten')],
            ['Patient', 'Region'],
            post_cols,
        )
        add_grouped_vectors(
            data['post_nostim_rem'],
            df[(df['stimulation'] == 0) & (df['memory_cond'] == 'remembered')],
            ['Patient', 'Region'],
            post_cols,
        )
        add_grouped_vectors(
            data['post_nostim_forg'],
            df[(df['stimulation'] == 0) & (df['memory_cond'] == 'forgotten')],
            ['Patient', 'Region'],
            post_cols,
        )

        add_grouped_vectors(data['diff_stim'], df[df['stimulation'] == 1], ['Patient', 'Region'], diff_cols)
        add_grouped_vectors(data['diff_nostim'], df[df['stimulation'] == 0], ['Patient', 'Region'], diff_cols)

        add_grouped_vectors(
            data['diff_stim_rem'],
            df[(df['stimulation'] == 1) & (df['memory_cond'] == 'remembered')],
            ['Patient', 'Region'],
            diff_cols,
        )
        add_grouped_vectors(
            data['diff_stim_forg'],
            df[(df['stimulation'] == 1) & (df['memory_cond'] == 'forgotten')],
            ['Patient', 'Region'],
            diff_cols,
        )
        add_grouped_vectors(
            data['diff_nostim_rem'],
            df[(df['stimulation'] == 0) & (df['memory_cond'] == 'remembered')],
            ['Patient', 'Region'],
            diff_cols,
        )
        add_grouped_vectors(
            data['diff_nostim_forg'],
            df[(df['stimulation'] == 0) & (df['memory_cond'] == 'forgotten')],
            ['Patient', 'Region'],
            diff_cols,
        )

        counts = (
            df.groupby(['Patient', 'Region', 'stimulation', 'memory_cond'])
            .size()
            .reset_index(name='n_trials')
        )
        data['trial_count_rows'].append(counts)

    return data


# %% [markdown]
# ## Plotting functions

# %%
def plot_pac_by_roi(data, out_dir, label='BLAES', filename='GroupPAC_byROI_encoding.png', footer_text=None):
    roi_dict = data['post_all']
    freqs = data['freqs_post']
    if not roi_dict or freqs is None:
        return
    roi_colors = make_roi_color_map(roi_dict.keys())
    fig, ax = plt.subplots(figsize=(14, 9))
    for roi in sorted(roi_dict):
        matrix = np.array(list(roi_dict[roi].values()), dtype=np.float64)
        mean = matrix.mean(axis=0)
        std = matrix.std(axis=0)
        color = roi_colors.get(roi, 'gray')
        ax.plot(freqs, mean, color=color, linewidth=2, label=f'{roi} ({matrix.shape[0]})')
        ax.fill_between(freqs, mean - std, mean + std, color=color, alpha=0.18)
    ax.set_xlabel('Amplitude Frequency (Hz)', fontsize=18, fontweight='bold')
    ax.set_ylabel('PAC', fontsize=18, fontweight='bold')
    ax.set_title(f'{label} Encoding PAC by ROI', fontsize=22, fontweight='bold')
    ax.tick_params(axis='both', labelsize=13)
    style_axis(ax)
    ax.legend(bbox_to_anchor=(1.02, 0.5), loc='center left', fontsize=9, frameon=False)
    save_figure_output(fig, out_dir / filename, footer_text=footer_text, rect=[0, 0, 0.82, 1])


def plot_pac_by_patient(data, out_dir, label='BLAES', filename='GroupPAC_byPatient_encoding.png', footer_text=None):
    roi_dict = data['post_all']
    freqs = data['freqs_post']
    if not roi_dict or freqs is None:
        return
    patient_vectors = {}
    for roi, subject_dict in roi_dict.items():
        for patient, values in subject_dict.items():
            patient_vectors.setdefault(patient, []).append(values)
    if not patient_vectors:
        return
    fig, ax = plt.subplots(figsize=(14, 9))
    for patient in sorted(patient_vectors):
        mean_vec = np.array(patient_vectors[patient], dtype=np.float64).mean(axis=0)
        ax.plot(freqs, mean_vec, linewidth=1.8, label=patient)
    ax.set_xlabel('Amplitude Frequency (Hz)', fontsize=18, fontweight='bold')
    ax.set_ylabel('PAC', fontsize=18, fontweight='bold')
    ax.set_title(f'{label} Encoding PAC by Patient', fontsize=22, fontweight='bold')
    ax.tick_params(axis='both', labelsize=13)
    style_axis(ax)
    ax.legend(bbox_to_anchor=(1.02, 0.5), loc='center left', fontsize=9, frameon=False)
    save_figure_output(fig, out_dir / filename, footer_text=footer_text, rect=[0, 0, 0.82, 1])


def plot_stim_vs_nostim(data, out_dir, label='BLAES', filename='GroupPAC_byROI_encoding_stim_vs_nostim.png', footer_text=None):
    freqs = data['freqs_post']
    stim_dict = data['post_stim']
    nostim_dict = data['post_nostim']
    if freqs is None or not stim_dict or not nostim_dict:
        return
    all_rois = sorted(set(stim_dict.keys()) | set(nostim_dict.keys()))
    roi_colors = make_roi_color_map(all_rois)
    fig, axes = plt.subplots(1, 2, figsize=(18, 8), sharex=True, sharey=True)
    for ax, title, roi_dict in zip(axes, ['No Stim', 'Stim'], [nostim_dict, stim_dict]):
        for roi in all_rois:
            if roi not in roi_dict:
                continue
            matrix = np.array(list(roi_dict[roi].values()), dtype=np.float64)
            mean = matrix.mean(axis=0)
            std = matrix.std(axis=0)
            color = roi_colors.get(roi, 'gray')
            ax.plot(freqs, mean, color=color, linewidth=2, label=f'{roi} ({matrix.shape[0]})')
            ax.fill_between(freqs, mean - std, mean + std, color=color, alpha=0.18)
        ax.set_title(title, fontsize=18, fontweight='bold')
        ax.set_xlabel('Amplitude Frequency (Hz)', fontsize=16, fontweight='bold')
        ax.tick_params(axis='both', labelsize=12)
        style_axis(ax)
    axes[0].set_ylabel('PAC', fontsize=16, fontweight='bold')
    axes[1].legend(bbox_to_anchor=(1.02, 0.5), loc='center left', fontsize=8.5, frameon=False)
    fig.suptitle(f'{label} Encoding PAC: No Stim vs Stim', fontsize=22, fontweight='bold')
    save_figure_output(fig, out_dir / filename, footer_text=footer_text, rect=[0, 0, 0.84, 0.95])


def plot_per_roi_patient_stim_nostim(
    data,
    out_dir,
    label='BLAES',
    filename_template='GroupPAC_{roi}_byPatient_encoding_stim_vs_nostim.png',
    footer_text=None,
):
    freqs = data['freqs_post']
    stim_dict = data['post_stim']
    nostim_dict = data['post_nostim']
    if freqs is None or (not stim_dict and not nostim_dict):
        return
    all_rois = sorted(set(stim_dict.keys()) | set(nostim_dict.keys()))
    for roi in all_rois:
        roi_subjects = set(stim_dict.get(roi, {})) | set(nostim_dict.get(roi, {}))
        if not roi_subjects:
            continue
        subject_colors = make_subject_color_map(roi_subjects)
        fig, axes = plt.subplots(1, 2, figsize=(18, 8), sharex=True, sharey=True)
        plotted_any = False
        for ax, title, roi_dict in zip(axes, [f'{roi} - No Stim', f'{roi} - Stim'], [nostim_dict, stim_dict]):
            for patient, values in sorted(roi_dict.get(roi, {}).items()):
                ax.plot(freqs, values, color=subject_colors[patient], linewidth=1.6, label=patient)
                plotted_any = True
            ax.set_title(title, fontsize=16, fontweight='bold')
            ax.set_xlabel('Amplitude Frequency (Hz)', fontsize=15, fontweight='bold')
            ax.tick_params(axis='both', labelsize=11)
            style_axis(ax)
        axes[0].set_ylabel('PAC', fontsize=15, fontweight='bold')
        fig.suptitle(f'{label} Encoding PAC: {roi} by Patient', fontsize=20, fontweight='bold')
        if plotted_any:
            handles, labels = collect_unique_legend_items(axes)
            if handles:
                fig.legend(handles, labels, bbox_to_anchor=(0.86, 0.5), loc='center left', fontsize=8, frameon=False)
            save_figure_output(
                fig,
                out_dir / filename_template.format(roi=roi),
                footer_text=footer_text,
                rect=[0, 0, 0.8, 0.95],
            )
        else:
            finalize_figure(fig)


def plot_bc_bar_graph(
    data,
    out_dir,
    label='BLAES',
    filename_template='Bargraph_baseline_corrected_PACDiff_byROI_encoding_{band}.png',
    footer_text=None,
):
    freqs = data['freqs_diff']
    diff_df = compute_condition_diff_df(data['diff_stim'], data['diff_nostim'], freqs, PAC_BANDS)
    if diff_df.empty:
        return
    roi_order = sorted(diff_df['Region'].unique())
    gray_values = np.linspace(0.85, 0.45, len(roi_order))
    palette = {roi: (gray, gray, gray) for roi, gray in zip(roi_order, gray_values)}
    for band in [name for name in PAC_BANDS if name in diff_df['band'].unique()]:
        band_df = diff_df[diff_df['band'] == band].copy()
        if band_df.empty:
            continue
        fig, ax = plt.subplots(figsize=(13, 8.5))
        sns.barplot(
            data=band_df,
            x='Region',
            y='mean_pac_diff',
            hue='Region',
            palette=palette,
            errorbar='se',
            dodge=False,
            legend=False,
            order=roi_order,
            ax=ax,
        )
        sns.stripplot(
            data=band_df,
            x='Region',
            y='mean_pac_diff',
            color='black',
            alpha=0.55,
            jitter=0.18,
            order=roi_order,
            ax=ax,
        )
        ax.axhline(0, color='gray', linewidth=1.2)
        ax.set_xlabel('')
        ax.set_ylabel('Baseline-corrected PAC Diff', fontsize=16, fontweight='bold')
        ax.set_title(band, fontsize=18, fontweight='bold')
        ax.tick_params(axis='x', labelsize=10, rotation=45)
        ax.tick_params(axis='y', labelsize=12)
        style_axis(ax, hide_top_right=True)
        fig.suptitle(f'{label} Encoding Baseline-corrected PAC Diff (Stim - No Stim)', fontsize=20, fontweight='bold', y=0.98)
        save_figure_output(
            fig,
            out_dir / filename_template.format(band=band.lower().replace(' ', '_')),
            footer_text=footer_text,
            rect=[0, 0, 1, 0.92],
        )
    plot_grouped_condition_bars(
        compute_condition_band_df(data['diff_stim'], data['diff_nostim'], freqs, PAC_BANDS),
        out_dir,
        label,
        'Encoding',
        'StimVsNoStim_alltrials_encoding_{band}.png',
        'All Trials',
        footer_text=footer_text,
    )


def plot_mi_difference_per_roi(
    data,
    out_dir,
    label='BLAES',
    filename_template='{roi}_MI_difference_encoding.png',
    footer_text=None,
):
    freqs = data['freqs_diff']
    memory_maps = {
        'Remembered': {'stim': data['diff_stim_rem'], 'nostim': data['diff_nostim_rem']},
        'Forgotten': {'stim': data['diff_stim_forg'], 'nostim': data['diff_nostim_forg']},
    }
    if freqs is None or not any(mapping['stim'] or mapping['nostim'] for mapping in memory_maps.values()):
        return

    all_rois = set()
    for mapping in memory_maps.values():
        all_rois.update(mapping['stim'].keys())
        all_rois.update(mapping['nostim'].keys())

    for roi in sorted(all_rois):
        has_data = False
        fig, axes = plt.subplots(1, 2, figsize=(16, 6.5), sharey=True)
        for ax, memory_label in zip(axes, ['Remembered', 'Forgotten']):
            stim_dict = memory_maps[memory_label]['stim']
            nostim_dict = memory_maps[memory_label]['nostim']
            shared_subjects = sorted(set(stim_dict.get(roi, {})) & set(nostim_dict.get(roi, {})))
            if not shared_subjects:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=12)
                ax.set_axis_off()
                continue
            has_data = True
            diff_matrix = np.array(
                [
                    np.asarray(stim_dict[roi][patient], dtype=np.float64) - np.asarray(nostim_dict[roi][patient], dtype=np.float64)
                    for patient in shared_subjects
                ],
                dtype=np.float64,
            )
            mean = diff_matrix.mean(axis=0)
            sem = diff_matrix.std(axis=0) / np.sqrt(diff_matrix.shape[0])

            ax.axvspan(PAC_BANDS['Slow gamma'][0], PAC_BANDS['Slow gamma'][1], color='#E8E8E8', alpha=1.0, zorder=0)
            ax.axhline(0, color='red', linewidth=1.8, linestyle=(0, (4, 4)), zorder=1)
            ax.plot(freqs, mean, color='#8E8E8E', linewidth=3, zorder=3)
            ax.fill_between(freqs, mean - sem, mean + sem, color='#BDBDBD', alpha=0.85, zorder=2)
            ax.set_title(f'{memory_label} Trials', fontsize=15, fontweight='bold')
            ax.set_xlabel('Amplitude Frequency (Hz)', fontsize=15, fontweight='bold')
            ax.tick_params(axis='both', labelsize=11, width=1.6, length=5)
            style_axis(ax, hide_top_right=True)

        if not has_data:
            finalize_figure(fig)
            continue

        axes[0].set_ylabel('MI Difference', fontsize=15, fontweight='bold')
        fig.suptitle(f'{label} Encoding {roi} MI Difference (Stim - No Stim)', fontsize=18, fontweight='bold')
        save_figure_output(
            fig,
            out_dir / filename_template.format(roi=roi),
            footer_text=footer_text,
            rect=[0, 0, 1, 0.93],
        )


def plot_bc_per_roi(
    data,
    out_dir,
    label='BLAES',
    filename_template='{roi}_BaselineCorrectedPAC_encoding_stim_vs_nostim.png',
    footer_text=None,
):
    freqs = data['freqs_diff']
    stim_dict = data['diff_stim']
    nostim_dict = data['diff_nostim']
    if freqs is None or (not stim_dict and not nostim_dict):
        return
    for roi in sorted(set(stim_dict.keys()) | set(nostim_dict.keys())):
        stim_subjects = stim_dict.get(roi, {})
        nostim_subjects = nostim_dict.get(roi, {})
        if not stim_subjects and not nostim_subjects:
            continue
        fig, ax = plt.subplots(figsize=(10, 7))
        for title, subject_dict, color in [('No Stim', nostim_subjects, '#1f77b4'), ('Stim', stim_subjects, '#d62728')]:
            if not subject_dict:
                continue
            matrix = np.array(list(subject_dict.values()), dtype=np.float64)
            mean = matrix.mean(axis=0)
            sem = matrix.std(axis=0) / np.sqrt(matrix.shape[0])
            ax.plot(freqs, mean, color=color, linewidth=2, label=f'{title} (n={matrix.shape[0]})')
            ax.fill_between(freqs, mean - sem, mean + sem, color=color, alpha=0.2)
        ax.set_title(f'{label} Encoding {roi} Baseline-corrected PAC', fontsize=18, fontweight='bold')
        ax.set_xlabel('Amplitude Frequency (Hz)', fontsize=15, fontweight='bold')
        ax.set_ylabel('PAC', fontsize=15, fontweight='bold')
        ax.tick_params(axis='both', labelsize=12)
        style_axis(ax)
        ax.legend(frameon=False, fontsize=11)
        save_figure_output(fig, out_dir / filename_template.format(roi=roi), footer_text=footer_text)


def plot_quadrant_stim_memory(
    data,
    out_dir,
    label='BLAES',
    filename='Quadrant_StimMemory_PAC_encoding.png',
    footer_text=None,
):
    freqs = data['freqs_post']
    memory_dicts = {
        'NoStim Remembered': data['post_nostim_rem'],
        'NoStim Forgotten': data['post_nostim_forg'],
        'Stim Remembered': data['post_stim_rem'],
        'Stim Forgotten': data['post_stim_forg'],
    }
    if freqs is None or not any(memory_dicts.values()):
        return
    all_rois = sorted(set().union(*[d.keys() for d in memory_dicts.values() if d]))
    roi_colors = make_roi_color_map(all_rois)
    fig, axes = plt.subplots(2, 2, figsize=(18, 11), sharex=True, sharey=True)
    plotted = set()
    for ax, condition in zip(axes.flatten(), ['NoStim Remembered', 'NoStim Forgotten', 'Stim Remembered', 'Stim Forgotten']):
        roi_dict = memory_dicts[condition]
        for roi in all_rois:
            if roi not in roi_dict:
                continue
            matrix = np.array(list(roi_dict[roi].values()), dtype=np.float64)
            mean = matrix.mean(axis=0)
            std = matrix.std(axis=0)
            label_text = f'{roi} ({matrix.shape[0]})' if roi not in plotted else None
            plotted.add(roi)
            color = roi_colors.get(roi, 'gray')
            ax.plot(freqs, mean, color=color, linewidth=2, label=label_text)
            ax.fill_between(freqs, mean - std, mean + std, color=color, alpha=0.18)
        ax.set_title(condition, fontsize=16, fontweight='bold')
        ax.tick_params(axis='both', labelsize=11)
        style_axis(ax)
    fig.text(0.5, 0.04, 'Amplitude Frequency (Hz)', ha='center', fontsize=16, fontweight='bold')
    fig.text(0.04, 0.5, 'PAC', va='center', rotation='vertical', fontsize=16, fontweight='bold')
    fig.suptitle(f'{label} Encoding PAC by Stim x Memory', fontsize=20, fontweight='bold')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, bbox_to_anchor=(0.84, 0.5), loc='center left', fontsize=8.5, frameon=False)
    save_figure_output(fig, out_dir / filename, footer_text=footer_text, rect=[0.06, 0.06, 0.82, 0.94])


def plot_per_roi_quadrant(
    data,
    out_dir,
    label='BLAES',
    filename_template='IndivQuadrant_{roi}_StimMemory_PAC_encoding.png',
    footer_text=None,
):
    freqs = data['freqs_post']
    quadrant_dicts = {
        'Stim Remembered': data['post_stim_rem'],
        'Stim Forgotten': data['post_stim_forg'],
        'NoStim Remembered': data['post_nostim_rem'],
        'NoStim Forgotten': data['post_nostim_forg'],
    }
    if freqs is None or not any(quadrant_dicts.values()):
        return
    roi_positions = {
        'Stim Remembered': (0, 0),
        'Stim Forgotten': (0, 1),
        'NoStim Remembered': (1, 0),
        'NoStim Forgotten': (1, 1),
    }
    all_rois = sorted(set().union(*[d.keys() for d in quadrant_dicts.values() if d]))
    for roi in all_rois:
        roi_subjects = set()
        for condition_dict in quadrant_dicts.values():
            roi_subjects.update(condition_dict.get(roi, {}))
        if not roi_subjects:
            continue
        subject_colors = make_subject_color_map(roi_subjects)
        fig, axes = plt.subplots(2, 2, figsize=(15, 11), sharex=True, sharey=True)
        has_data = False
        for condition, roi_dict in quadrant_dicts.items():
            r, c = roi_positions[condition]
            ax = axes[r, c]
            subject_dict = roi_dict.get(roi, {})
            if not subject_dict:
                ax.set_title(f'{condition} (no data)', fontsize=13, fontweight='bold')
                ax.set_axis_off()
                continue
            has_data = True
            for patient, values in sorted(subject_dict.items()):
                ax.plot(freqs, values, color=subject_colors[patient], linewidth=1.5, label=patient, alpha=0.8)
            ax.set_title(condition, fontsize=13, fontweight='bold')
            ax.tick_params(axis='both', labelsize=10)
            style_axis(ax)
        if has_data:
            fig.suptitle(f'{label} Encoding {roi} PAC by Patient and Memory', fontsize=18, fontweight='bold')
            fig.text(0.5, 0.04, 'Amplitude Frequency (Hz)', ha='center', fontsize=14, fontweight='bold')
            fig.text(0.04, 0.5, 'PAC', va='center', rotation='vertical', fontsize=14, fontweight='bold')
            handles, labels = collect_unique_legend_items(axes.flatten())
            if handles:
                fig.legend(handles, labels, bbox_to_anchor=(0.88, 0.5), loc='center left', fontsize=7.5, frameon=False)
            save_figure_output(
                fig,
                out_dir / filename_template.format(roi=roi),
                footer_text=footer_text,
                rect=[0.06, 0.06, 0.84, 0.92],
            )
        else:
            finalize_figure(fig)


def plot_bc_bar_by_memory(
    data,
    out_dir,
    label='BLAES',
    filename_template='Bargraph_PAC_diff_{mem}_encoding_{band}.png',
    footer_text=None,
):
    freqs = data['freqs_diff']
    if freqs is None:
        return
    remembered_df = compute_condition_diff_df(data['diff_stim_rem'], data['diff_nostim_rem'], freqs, PAC_BANDS)
    forgotten_df = compute_condition_diff_df(data['diff_stim_forg'], data['diff_nostim_forg'], freqs, PAC_BANDS)
    if remembered_df.empty and forgotten_df.empty:
        return
    if not remembered_df.empty:
        remembered_df['memory_cond'] = 'remembered'
    if not forgotten_df.empty:
        forgotten_df['memory_cond'] = 'forgotten'
    df_all = pd.concat([df for df in [remembered_df, forgotten_df] if not df.empty], ignore_index=True)
    roi_order = sorted(df_all['Region'].unique())
    gray_values = np.linspace(0.85, 0.45, len(roi_order))
    palette = {roi: (gray, gray, gray) for roi, gray in zip(roi_order, gray_values)}
    mem_source_lookup = {
        'remembered': (data['diff_stim_rem'], data['diff_nostim_rem']),
        'forgotten': (data['diff_stim_forg'], data['diff_nostim_forg']),
    }
    for memory_cond, memory_label in [('remembered', 'Remembered'), ('forgotten', 'Forgotten')]:
        subset = df_all[df_all['memory_cond'] == memory_cond]
        if subset.empty:
            continue
        for band in [name for name in PAC_BANDS if name in subset['band'].unique()]:
            band_df = subset[subset['band'] == band].copy()
            if band_df.empty:
                continue
            fig, ax = plt.subplots(figsize=(13, 8.5))
            sns.barplot(
                data=band_df,
                x='Region',
                y='mean_pac_diff',
                hue='Region',
                palette=palette,
                errorbar='se',
                dodge=False,
                legend=False,
                order=roi_order,
                ax=ax,
            )
            sns.stripplot(
                data=band_df,
                x='Region',
                y='mean_pac_diff',
                color='black',
                alpha=0.55,
                jitter=0.18,
                order=roi_order,
                ax=ax,
            )
            ax.axhline(0, color='gray', linewidth=1.2)
            ax.set_xlabel('')
            ax.set_ylabel('Baseline-corrected PAC Diff', fontsize=16, fontweight='bold')
            ax.set_title(band, fontsize=18, fontweight='bold')
            ax.tick_params(axis='x', labelsize=10, rotation=45)
            ax.tick_params(axis='y', labelsize=12)
            style_axis(ax, hide_top_right=True)
            fig.suptitle(f'{label} Encoding PAC Diff, {memory_label} Trials', fontsize=20, fontweight='bold', y=0.98)
            save_figure_output(
                fig,
                out_dir / filename_template.format(mem=memory_cond, band=band.lower().replace(' ', '_')),
                footer_text=footer_text,
                rect=[0, 0, 1, 0.92],
            )
        stim_dict, nostim_dict = mem_source_lookup[memory_cond]
        plot_grouped_condition_bars(
            compute_condition_band_df(stim_dict, nostim_dict, freqs, PAC_BANDS),
            out_dir,
            label,
            'Encoding',
            f'StimVsNoStim_{memory_cond}_encoding_{{band}}.png',
            f'{memory_label} Trials',
            footer_text=footer_text,
        )


def plot_bc_remembered_forgotten(
    data,
    out_dir,
    label='BLAES',
    filename_template='{roi}_Baseline_Adjusted_PAC_RememberedForgotten_encoding.png',
    footer_text=None,
):
    freqs = data['freqs_diff']
    if freqs is None:
        return
    memory_maps = {
        'Remembered': {'stim': data['diff_stim_rem'], 'nostim': data['diff_nostim_rem']},
        'Forgotten': {'stim': data['diff_stim_forg'], 'nostim': data['diff_nostim_forg']},
    }
    all_rois = set()
    for mapping in memory_maps.values():
        all_rois.update(mapping['stim'].keys())
        all_rois.update(mapping['nostim'].keys())
    for roi in sorted(all_rois):
        has_data = False
        fig, axes = plt.subplots(1, 2, figsize=(17, 6.5), sharey=True)
        for ax, memory_label in zip(axes, ['Remembered', 'Forgotten']):
            for condition_name, condition_dict, color in [
                ('No Stim', memory_maps[memory_label]['nostim'], '#1f77b4'),
                ('Stim', memory_maps[memory_label]['stim'], '#d62728'),
            ]:
                subject_dict = condition_dict.get(roi, {})
                if not subject_dict:
                    continue
                has_data = True
                matrix = np.array(list(subject_dict.values()), dtype=np.float64)
                mean = matrix.mean(axis=0)
                sem = matrix.std(axis=0) / np.sqrt(matrix.shape[0])
                ax.plot(freqs, mean, color=color, linewidth=2, label=f'{condition_name} (n={matrix.shape[0]})')
                ax.fill_between(freqs, mean - sem, mean + sem, color=color, alpha=0.2)
            ax.set_title(f'{memory_label} Trials', fontsize=15, fontweight='bold')
            ax.set_xlabel('Amplitude Frequency (Hz)', fontsize=14, fontweight='bold')
            ax.tick_params(axis='both', labelsize=11)
            style_axis(ax)
        axes[0].set_ylabel('PAC', fontsize=14, fontweight='bold')
        if has_data:
            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(handles, labels, bbox_to_anchor=(0.92, 0.5), loc='center left', fontsize=10, frameon=False)
            fig.suptitle(f'{label} Encoding {roi} Baseline-corrected PAC', fontsize=18, fontweight='bold')
            save_figure_output(
                fig,
                out_dir / filename_template.format(roi=roi),
                footer_text=footer_text,
                rect=[0, 0, 0.9, 0.93],
            )
        else:
            finalize_figure(fig)


# %% [markdown]
# ## Summary table exports

# %%
def export_summary_tables(data, out_dir):
    ensure_dir(out_dir)

    post_rows = []
    post_rows.extend(build_band_summary_rows(data['post_all'], data['freqs_post'], PAC_BANDS, 'overall', 'PAC'))
    post_rows.extend(build_band_summary_rows(data['post_stim'], data['freqs_post'], PAC_BANDS, 'stim', 'PAC'))
    post_rows.extend(build_band_summary_rows(data['post_nostim'], data['freqs_post'], PAC_BANDS, 'nostim', 'PAC'))
    post_rows.extend(build_band_summary_rows(data['post_stim_rem'], data['freqs_post'], PAC_BANDS, 'stim_remembered', 'PAC'))
    post_rows.extend(build_band_summary_rows(data['post_stim_forg'], data['freqs_post'], PAC_BANDS, 'stim_forgotten', 'PAC'))
    post_rows.extend(build_band_summary_rows(data['post_nostim_rem'], data['freqs_post'], PAC_BANDS, 'nostim_remembered', 'PAC'))
    post_rows.extend(build_band_summary_rows(data['post_nostim_forg'], data['freqs_post'], PAC_BANDS, 'nostim_forgotten', 'PAC'))
    if post_rows:
        pd.DataFrame(post_rows).sort_values(['condition', 'Region', 'Patient', 'band']).to_csv(
            out_dir / 'post_band_summary_encoding.csv', index=False
        )

    diff_rows = []
    diff_rows.extend(build_band_summary_rows(data['diff_stim'], data['freqs_diff'], PAC_BANDS, 'stim_bc', 'PAC'))
    diff_rows.extend(build_band_summary_rows(data['diff_nostim'], data['freqs_diff'], PAC_BANDS, 'nostim_bc', 'PAC'))
    diff_rows.extend(build_band_summary_rows(data['diff_stim_rem'], data['freqs_diff'], PAC_BANDS, 'stim_bc_remembered', 'PAC'))
    diff_rows.extend(build_band_summary_rows(data['diff_stim_forg'], data['freqs_diff'], PAC_BANDS, 'stim_bc_forgotten', 'PAC'))
    diff_rows.extend(build_band_summary_rows(data['diff_nostim_rem'], data['freqs_diff'], PAC_BANDS, 'nostim_bc_remembered', 'PAC'))
    diff_rows.extend(build_band_summary_rows(data['diff_nostim_forg'], data['freqs_diff'], PAC_BANDS, 'nostim_bc_forgotten', 'PAC'))
    if diff_rows:
        pd.DataFrame(diff_rows).sort_values(['condition', 'Region', 'Patient', 'band']).to_csv(
            out_dir / 'baseline_corrected_band_summary_encoding.csv', index=False
        )

    stim_minus_nostim_df = compute_condition_diff_df(data['diff_stim'], data['diff_nostim'], data['freqs_diff'], PAC_BANDS)
    if not stim_minus_nostim_df.empty:
        stim_minus_nostim_df.sort_values(['Region', 'Patient', 'band']).to_csv(
            out_dir / 'stim_minus_nostim_baseline_corrected_band_diff_encoding.csv', index=False
        )

    if data['trial_count_rows']:
        pd.concat(data['trial_count_rows'], ignore_index=True).sort_values(
            ['Patient', 'Region', 'stimulation', 'memory_cond']
        ).to_csv(out_dir / 'trial_counts_by_region_condition_encoding.csv', index=False)


# %% [markdown]
# ## Main

# %%
def main():
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    ensure_dir(OUTPUT_DIR)
    ensure_dir(CSV_DIR)

    print(f'Data path: {DATA_PATH}')
    print(f'Output path: {OUTPUT_DIR}')
    print(f'Found {len(PAC_FILES)} PAC Phase 1 files.')

    data = load_blaes_encoding_pac()

    plot_pac_by_roi(data, OUTPUT_DIR)
    plot_pac_by_patient(data, OUTPUT_DIR)
    plot_stim_vs_nostim(data, OUTPUT_DIR)
    plot_per_roi_patient_stim_nostim(data, OUTPUT_DIR)
    plot_bc_bar_graph(data, OUTPUT_DIR)
    plot_mi_difference_per_roi(data, OUTPUT_DIR)
    plot_bc_per_roi(data, OUTPUT_DIR)
    plot_quadrant_stim_memory(data, OUTPUT_DIR)
    plot_per_roi_quadrant(data, OUTPUT_DIR)
    plot_bc_bar_by_memory(data, OUTPUT_DIR)
    plot_bc_remembered_forgotten(data, OUTPUT_DIR)
    export_summary_tables(data, CSV_DIR)

    print('PAC encoding analysis complete.')


if __name__ == '__main__':
    main()
