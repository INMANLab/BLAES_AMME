#!/usr/bin/env python
"""
One-Second-Stim Trials Analysis
================================
Reruns all encoding/retrieval power, coherence, and PAC pipelines but
restricts AMME stim trials to the one-second-stim subset:

  AMME Timing patients (amyg045, 046, 048, 054, 057, 059, 061, 066, 072):
    encoding stim trials: keep only stimulation == 1 (already enforced upstream)
    retrieval stim trials: keep only trial_type == "After stim"
    drop "Before stim" and "During stim" everywhere.

  AMME Duration patients (amyg030, 033, 034, 037):
    encoding: keep stimulation in {0, 1}; drop stimulation == 3 (3-sec stim)
    retrieval: keep trial_type in {"nostim", "1s stim"}; drop "3s stim"

  BLAES patients: untouched (already only 1-sec stim).

Excludes regions MTL, PHG, and any PNAS variant from every output. All other
regions are kept.

This is a sibling of balanced_memory_code/balanced_memory_trials.py and
follows the same dispatch pattern: power -> coherence -> PAC, encoding then
retrieval, with outputs grouped by phase x analysis x cohort.
"""

# --- AMME_BLAES sys.path bootstrap ---
import sys as _sys
from pathlib import Path as _Path
_root = _Path(__file__).resolve().parent.parent
_search_paths = [
    _root,
    _root / 'encoding_code',
    _root / 'retrieval_code',
    _root / 'endogenous_memory_code',
    _root / 'balanced_memory_code',
    _root / 'regressions_code',
    _root / 'permutation',
]
for _p in _search_paths:
    if _p.exists() and str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))
# --- end bootstrap ---

import os
import sys
import warnings
from contextlib import contextmanager

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')

warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

OUTPUT_ROOT = os.path.join(PROJECT_ROOT, 'outputs', 'onesecstim_trials')
CSV_OUTPUT_DIR = os.path.join(OUTPUT_ROOT, 'csvs')
SUMMARY_OUTPUT_PATH = os.path.join(OUTPUT_ROOT, 'onesecstim_conditions_summary.txt')

# AMME cohort definitions (mirrors permutation/build_onesec_filtered_csvs.py).
AMME_TIMING = [f'amyg{n:03d}' for n in (45, 46, 48, 54, 57, 59, 61, 66, 72)]
AMME_DURATION = ['amyg030', 'amyg033', 'amyg034', 'amyg037']
AMME_ALL = set(AMME_TIMING + AMME_DURATION)

# Region exclusion list — applied to every output. MTL, PHG and any PNAS
# variant (e.g. "BLA_PNAS") are dropped; all other ROIs are kept.
EXCLUDED_REGIONS = {'MTL', 'PHG'}
EXCLUDED_REGION_SUBSTRINGS = ('PNAS',)

from endogenous_memory import (
    ensure_dir,
    extract_endogenous,
    generate_endogenous_plots,
    export_endogenous_mlmr_csv,
    PAC_BANDS,
    POWER_RANGES,
)
from balanced_memory_trials import (
    filter_subject_dicts,
    filter_mlmr_frames,
    add_balanced_stim_memory_plots,
    generate_endogenous_only_combined_plots,
    get_region_subject_counts,
    format_region_counts,
)


def onesec_plot_label(group_label):
    return f'{group_label} One-Second-Stim Trials'


# ---------------------------------------------------------------------------
# Region-exclusion helper (MTL / PHG / PNAS)
# ---------------------------------------------------------------------------

def _is_excluded_region(roi):
    s = str(roi)
    if s in EXCLUDED_REGIONS:
        return True
    for sub in EXCLUDED_REGION_SUBSTRINGS:
        if sub in s.upper():
            return True
    # Coherence/PAC pair regions like "BLA_MTL" — drop if any half is excluded.
    if '_' in s:
        parts = s.split('_')
        for part in parts:
            if part in EXCLUDED_REGIONS:
                return True
            for sub in EXCLUDED_REGION_SUBSTRINGS:
                if sub in part.upper():
                    return True
    return False


def filter_excluded_regions(data):
    """Drop any region key matching MTL, PHG, or PNAS from a data dict."""
    filtered = {}
    for key, value in data.items():
        if isinstance(value, dict) and value:
            first_val = next(iter(value.values()), None)
            if isinstance(first_val, dict):
                filtered[key] = {
                    roi: sd for roi, sd in value.items()
                    if not _is_excluded_region(roi)
                }
            else:
                filtered[key] = value
        elif isinstance(value, list) and key == 'mlmr_export_frames':
            filtered[key] = [
                df[~df['Region'].apply(_is_excluded_region)].copy()
                if 'Region' in df.columns else df
                for df in value
            ]
        else:
            filtered[key] = value
    return filtered


# ---------------------------------------------------------------------------
# Onesec trial filter — patches pandas.read_csv during AMME loader calls.
#
# The encoding/retrieval AMME loaders read raw CSVs and aggregate per
# (Patient, Region, trial_type). We can't easily change the aggregation, but
# we CAN intercept the raw read and drop unwanted trial rows BEFORE the
# loader sees them. The patched read is a no-op for any CSV that doesn't
# carry the expected columns (Patient + stimulation, or Patient + trial_type
# for retrieval), so phase3 lookup reads, BLAES reads and all other CSV I/O
# in the program are unaffected.
# ---------------------------------------------------------------------------

@contextmanager
def _onesec_amme_filter(phase):
    """Patch pd.read_csv to drop non-1-sec AMME trial rows during loader call.

    phase: 1 (encoding) or 3 (retrieval).
    """
    orig_read_csv = pd.read_csv

    def _filter_phase1(df):
        # AMME encoding files. The loader already drops "Before stim" and
        # "During stim" via test_trial_type, so AMME Timing is correct.
        # We additionally drop AMME Duration rows where stimulation == 3.
        if 'Patient' not in df.columns or 'stimulation' not in df.columns:
            return df
        mask_dur_3s = (
            df['Patient'].astype(str).isin(AMME_DURATION)
            & (pd.to_numeric(df['stimulation'], errors='coerce') == 3)
        )
        if not mask_dur_3s.any():
            return df
        before = len(df)
        df = df[~mask_dur_3s].copy()
        print(f"    [onesec patch] phase1 dropped {before - len(df)} "
              f"AMME-Duration 3s-stim rows")
        return df

    def _filter_phase3(df):
        # AMME retrieval files. trial_type column carries:
        #   AMME Timing:    {nostim, "Before stim", "During stim",
        #                    "After stim", "new"}
        #   AMME Duration:  {nostim, "1s stim", "3s stim", "new"}
        # Drop everything except {nostim, After stim} for Timing and
        # {nostim, 1s stim} for Duration. "new" foils are dropped because
        # they aren't analyzed in any of the downstream pipelines.
        if 'Patient' not in df.columns or 'trial_type' not in df.columns:
            return df
        patient_col = df['Patient'].astype(str)
        tt_col = df['trial_type'].astype(str)

        keep_timing = patient_col.isin(AMME_TIMING) & tt_col.isin(
            {'nostim', 'After stim'}
        )
        keep_duration = patient_col.isin(AMME_DURATION) & tt_col.isin(
            {'nostim', '1s stim'}
        )
        keep_other = ~patient_col.isin(AMME_ALL)
        mask_keep = keep_timing | keep_duration | keep_other
        if mask_keep.all():
            return df
        before = len(df)
        df = df[mask_keep].copy()
        print(f"    [onesec patch] phase3 dropped {before - len(df)} "
              f"non-onesec AMME trial rows")
        return df

    apply_filter = _filter_phase1 if phase == 1 else _filter_phase3

    def patched(*args, **kwargs):
        df = orig_read_csv(*args, **kwargs)
        if isinstance(df, pd.DataFrame):
            return apply_filter(df)
        return df

    pd.read_csv = patched
    try:
        yield
    finally:
        pd.read_csv = orig_read_csv


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------

def write_onesec_summary_report(summary_rows, out_path=SUMMARY_OUTPUT_PATH):
    lines = [
        'One-Second-Stim Trials Summary',
        '=' * 80,
        'AMME Timing: stim = After-stim only (Before/During dropped).',
        'AMME Duration: stim = 1s-stim only (3s-stim dropped).',
        'BLAES: untouched (already 1-sec stim).',
        f'Excluded regions: MTL, PHG, any PNAS variant.',
        '',
    ]
    for row in summary_rows:
        lines.extend([
            f"{row['measure']} | {row['phase']} | {row['group']}",
            '-' * 80,
        ])
        if row.get('note'):
            lines.append(f"Note: {row['note']}")
        lines.append('Remaining N across regions:')
        lines.extend([f'  {ln}' for ln in format_region_counts(row['region_counts'])])
        lines.append('')

    ensure_dir(os.path.dirname(out_path))
    with open(out_path, 'w', encoding='utf-8') as h:
        h.write('\n'.join(lines).rstrip() + '\n')
    print(f'  Wrote one-sec-stim summary: {out_path}')
    return out_path


# ---------------------------------------------------------------------------
# Common pipeline runner — encoding + retrieval, parametrized by measure.
# ---------------------------------------------------------------------------

def _summary_row(measure, phase, group_label, filtered_data, note=''):
    return {
        'measure': measure,
        'phase': phase,
        'group': group_label,
        'region_counts': get_region_subject_counts(filtered_data),
        'note': note,
    }


def _run_phase(measure, phase, loaders, common_plotter, memory_plotter,
               group_keys=('blaes', 'amme', 'all'),
               extra_export_kwargs=None):
    """Run one (measure, phase) combo across blaes/amme/all groups.

    loaders is a dict with keys 'blaes', 'amme', 'all_builder' where the
    last is a callable taking (blaes_data, amme_data) and returning the
    merged 'all' data dict.
    """
    extra_export_kwargs = extra_export_kwargs or {}
    raw_phase = 1 if phase == 'Encoding' else 3
    summary_rows = []

    print(f"\n  Loading {phase.lower()} {measure.lower()} data (onesec filter)...")
    with _onesec_amme_filter(raw_phase):
        blaes_data = loaders['blaes']()
        amme_data = loaders['amme']()
    all_data = loaders['all_builder'](blaes_data, amme_data)

    # Apply region exclusions to every cohort.
    blaes_data = filter_excluded_regions(blaes_data)
    amme_data = filter_excluded_regions(amme_data)
    all_data = filter_excluded_regions(all_data)

    cohort_data = {'blaes': blaes_data, 'amme': amme_data, 'all': all_data}
    out_subdir = f'{phase.lower()}_{measure.lower()}'

    for key in group_keys:
        data = cohort_data[key]
        # No subject filter for onesecstim — restriction is at trial level
        # via the pd.read_csv patch above.
        endo = extract_endogenous(data, measure=measure.lower())
        out_dir = ensure_dir(os.path.join(OUTPUT_ROOT, out_subdir, key))
        plot_label = onesec_plot_label(key.upper() if key != 'all' else 'All')
        if measure == 'PAC':
            generate_endogenous_plots(endo, out_dir, plot_label, 'PAC',
                                      phase, band_ranges=PAC_BANDS)
        else:
            generate_endogenous_plots(endo, out_dir, plot_label, measure, phase)
        add_balanced_stim_memory_plots(
            data, out_dir, plot_label, common_plotter, memory_plotter)
        if measure == 'PAC':
            generate_endogenous_only_combined_plots(
                data, out_dir, plot_label, 'pac', phase.lower(),
                band_ranges=PAC_BANDS)
        else:
            generate_endogenous_only_combined_plots(
                data, out_dir, plot_label, measure.lower(), phase.lower())

        summary_rows.append(_summary_row(measure, phase, key.upper(), data))

        # MLMR CSV export (BC trial-level frame) — power/coherence have it.
        if 'mlmr_export_frames' in data and data.get('mlmr_export_frames'):
            stem = f'onesec_{phase.lower()}_{measure.lower()}_{key}_mlmr_input'
            export_endogenous_mlmr_csv(data, CSV_OUTPUT_DIR, stem,
                                       **extra_export_kwargs)

    return summary_rows


# ---------------------------------------------------------------------------
# POWER
# ---------------------------------------------------------------------------

def run_power_analysis():
    print('\n' + '=' * 60)
    print('One-Second-Stim Trials — POWER')
    print('=' * 60)

    from combined_encoding_power import (
        load_blaes_encoding, load_amme_encoding, build_all_encoding_data,
        generate_common_plots as enc_common,
        generate_memory_plots as enc_memory,
    )
    from combined_retrieval_power import (
        load_blaes_retrieval, load_amme_retrieval, build_all_retrieval_data,
        generate_common_plots as ret_common,
        generate_memory_plots as ret_memory,
    )

    enc_loaders = {
        'blaes': load_blaes_encoding,
        'amme': load_amme_encoding,
        'all_builder': build_all_encoding_data,
    }
    ret_loaders = {
        'blaes': load_blaes_retrieval,
        'amme': load_amme_retrieval,
        'all_builder': build_all_retrieval_data,
    }

    rows = []
    rows += _run_phase('Power', 'Encoding', enc_loaders, enc_common, enc_memory)
    rows += _run_phase('Power', 'Retrieval', ret_loaders, ret_common, ret_memory)
    print('  Power analysis complete.')
    return rows


# ---------------------------------------------------------------------------
# COHERENCE
# ---------------------------------------------------------------------------

def run_coherence_analysis():
    print('\n' + '=' * 60)
    print('One-Second-Stim Trials — COHERENCE')
    print('=' * 60)

    from combined_encoding_coherence import (
        load_blaes_encoding as load_blaes_enc_coh,
        load_amme_encoding as load_amme_enc_coh,
        generate_common_plots as enc_common,
        generate_memory_plots as enc_memory,
    )
    from combined_retrieval_coherence import (
        load_blaes_retrieval as load_blaes_ret_coh,
        load_amme_retrieval as load_amme_ret_coh,
        generate_common_plots as ret_common,
        generate_memory_plots as ret_memory,
    )
    from endogenous_memory import _merge_coherence_data

    enc_loaders = {
        'blaes': load_blaes_enc_coh,
        'amme': load_amme_enc_coh,
        'all_builder': _merge_coherence_data,
    }
    ret_loaders = {
        'blaes': load_blaes_ret_coh,
        'amme': load_amme_ret_coh,
        'all_builder': _merge_coherence_data,
    }

    rows = []
    rows += _run_phase('Coherence', 'Encoding', enc_loaders, enc_common,
                       enc_memory,
                       extra_export_kwargs={'measure_name': 'Coherence'})
    rows += _run_phase('Coherence', 'Retrieval', ret_loaders, ret_common,
                       ret_memory,
                       extra_export_kwargs={'measure_name': 'Coherence'})
    print('  Coherence analysis complete.')
    return rows


# ---------------------------------------------------------------------------
# PAC
# ---------------------------------------------------------------------------

def run_pac_analysis():
    print('\n' + '=' * 60)
    print('One-Second-Stim Trials — PAC')
    print('=' * 60)

    from combined_encoding_pac import (
        load_grouped_encoding_pac_data,
        generate_common_plots as enc_common,
        generate_memory_plots as enc_memory,
    )
    from combined_retrieval_pac import (
        load_grouped_retrieval_pac_data,
        generate_common_plots as ret_common,
        generate_memory_plots as ret_memory,
    )

    summary_rows = []

    print('\n  Loading encoding PAC data (onesec filter)...')
    with _onesec_amme_filter(phase=1):
        grouped_enc = load_grouped_encoding_pac_data()
    for key in ('blaes', 'amme', 'all'):
        data = filter_excluded_regions(grouped_enc[key])
        out_dir = ensure_dir(os.path.join(OUTPUT_ROOT, 'encoding_pac', key))
        plot_label = onesec_plot_label(key.upper() if key != 'all' else 'All')
        add_balanced_stim_memory_plots(data, out_dir, plot_label,
                                       enc_common, enc_memory)
        endo = extract_endogenous(data, measure='pac')
        generate_endogenous_plots(endo, out_dir, plot_label, 'PAC',
                                  'Encoding', band_ranges=PAC_BANDS)
        generate_endogenous_only_combined_plots(
            data, out_dir, plot_label, 'pac', 'encoding',
            band_ranges=PAC_BANDS)
        summary_rows.append(_summary_row('PAC', 'Encoding', key.upper(), data))

    print('\n  Loading retrieval PAC data (onesec filter)...')
    with _onesec_amme_filter(phase=3):
        grouped_ret = load_grouped_retrieval_pac_data()
    for key in ('blaes', 'amme', 'all'):
        data = filter_excluded_regions(grouped_ret[key])
        out_dir = ensure_dir(os.path.join(OUTPUT_ROOT, 'retrieval_pac', key))
        plot_label = onesec_plot_label(key.upper() if key != 'all' else 'All')
        add_balanced_stim_memory_plots(data, out_dir, plot_label,
                                       ret_common, ret_memory)
        endo = extract_endogenous(data, measure='pac')
        generate_endogenous_plots(endo, out_dir, plot_label, 'PAC',
                                  'Retrieval', band_ranges=PAC_BANDS)
        generate_endogenous_only_combined_plots(
            data, out_dir, plot_label, 'pac', 'retrieval',
            band_ranges=PAC_BANDS)
        summary_rows.append(_summary_row('PAC', 'Retrieval', key.upper(), data))

    print('  PAC analysis complete.')
    return summary_rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(measures=None):
    """Run the onesec analyses. measures = subset of {'power','coherence','pac'}."""
    print('=' * 60)
    print('One-Second-Stim Trials Analysis')
    print(f'AMME Timing: stim = After-stim only (Before/During dropped)')
    print(f'AMME Duration: stim = 1s only (3s dropped)')
    print(f'Excluded regions: MTL, PHG, PNAS')
    print('=' * 60)

    ensure_dir(OUTPUT_ROOT)
    ensure_dir(CSV_OUTPUT_DIR)

    measures = measures or ('power', 'coherence', 'pac')
    rows = []
    if 'power' in measures:
        rows += run_power_analysis()
    if 'coherence' in measures:
        rows += run_coherence_analysis()
    if 'pac' in measures:
        rows += run_pac_analysis()
    write_onesec_summary_report(rows)

    print('\n' + '=' * 60)
    print(f'Done! One-sec-stim outputs saved to: {OUTPUT_ROOT}')
    print('=' * 60)


if __name__ == '__main__':
    main()
