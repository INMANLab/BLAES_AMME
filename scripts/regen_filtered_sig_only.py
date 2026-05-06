#!/usr/bin/env python
"""Regenerate ONLY the filtered_sig bargraphs (BLA-pairs / non-BLA-pairs panels)
for PAC retrieval, PAC encoding, coherence retrieval, and coherence encoding.

Loads the same data as the full scripts, but skips the hundreds of legacy
per-ROI/per-patient/per-pair plots. Use this when you only need to refresh
the new filtered_sig figures (≈30-60 s instead of 30+ min).
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for p in [ROOT, ROOT / 'retrieval_code', ROOT / 'encoding_code',
          ROOT / 'to_combine']:
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

import os
import pandas as pd

from combined_pac_common import (
    augment_with_bla_composites,
    build_bla_composite_data as build_bla_composite_pac_data,
    split_full_pac_data,
)


def _ensure_dir(p):
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def regenerate_pac_filtered_sig(side):
    """side: 'retrieval' or 'encoding'."""
    if side == 'retrieval':
        from BLAES_Group_PAC_analyses_from_matlab_retrieval import (
            load_blaes_retrieval_pac as load_full,
            compute_condition_diff_df,
            PAC_BANDS,
            plot_pac_filtered_significance_bargraph,
            PAC_BLA_PANEL_PAIRS,
            PAC_NONBLA_PANEL_PAIRS,
        )
    else:
        from BLAES_Group_PAC_analyses_from_matlab_encoding import (
            load_blaes_encoding_pac as load_full,
            compute_condition_diff_df,
            PAC_BANDS,
            plot_pac_filtered_significance_bargraph,
            PAC_BLA_PANEL_PAIRS,
            PAC_NONBLA_PANEL_PAIRS,
        )

    out_root = ROOT / 'OUTPUTS' / f'{side}_pac'
    full = load_full()
    grouped = {k: augment_with_bla_composites(v) for k, v in split_full_pac_data(full).items()}

    side_label = side.capitalize()
    for key, label in [('blaes', 'BLAES'), ('amme', 'AMME'), ('all', 'All')]:
        data = grouped.get(key)
        if not data:
            continue
        if side == 'encoding':
            out_dir = _ensure_dir(out_root / 'all') if key == 'all' else _ensure_dir(out_root / key)
        else:
            out_dir = _ensure_dir(out_root / key)

        freqs = data.get('freqs_diff')
        if freqs is None:
            continue

        # All-trials
        df_all = compute_condition_diff_df(data.get('diff_stim', {}), data.get('diff_nostim', {}), freqs, PAC_BANDS)
        if not df_all.empty:
            plot_pac_filtered_significance_bargraph(
                df_all, out_dir,
                title=f'{label} {side_label} MI Difference (Stim - No Stim) - BLA Pairs',
                filename=f'Bargraph_PAC_MIDiff_byROI_{side}_filtered_sig_BLApairs.png',
                allowed_pairs=PAC_BLA_PANEL_PAIRS,
            )
            plot_pac_filtered_significance_bargraph(
                df_all, out_dir,
                title=f'{label} {side_label} MI Difference (Stim - No Stim) - ALLHPC/EC/PRC Pairs',
                filename=f'Bargraph_PAC_MIDiff_byROI_{side}_filtered_sig_nonBLApairs.png',
                allowed_pairs=PAC_NONBLA_PANEL_PAIRS,
            )

        # Memory-conditioned
        for mem_key, mem_label in [('rem', 'Remembered'), ('forg', 'Forgotten')]:
            sd = data.get(f'diff_stim_{mem_key}', {})
            nd = data.get(f'diff_nostim_{mem_key}', {})
            if not sd or not nd:
                continue
            df_mem = compute_condition_diff_df(sd, nd, freqs, PAC_BANDS)
            if df_mem.empty:
                continue
            mem_cond = 'remembered' if mem_key == 'rem' else 'forgotten'
            plot_pac_filtered_significance_bargraph(
                df_mem, out_dir,
                title=f'{label} {side_label} MI Difference (Stim - No Stim), {mem_label} Trials - BLA Pairs',
                filename=f'Bargraph_PAC_MIDiff_byROI_{side}_{mem_cond}_filtered_sig_BLApairs.png',
                allowed_pairs=PAC_BLA_PANEL_PAIRS,
            )
            plot_pac_filtered_significance_bargraph(
                df_mem, out_dir,
                title=f'{label} {side_label} MI Difference (Stim - No Stim), {mem_label} Trials - ALLHPC/EC/PRC Pairs',
                filename=f'Bargraph_PAC_MIDiff_byROI_{side}_{mem_cond}_filtered_sig_nonBLApairs.png',
                allowed_pairs=PAC_NONBLA_PANEL_PAIRS,
            )
        print(f'  PAC {side} {label}: filtered_sig figures written to {out_dir}')


def regenerate_coherence_filtered_sig(side):
    """side: 'retrieval' or 'encoding'.
    Loads data via the existing combined_*_coherence loader entry points, then
    emits only filtered_sig figures.
    """
    import importlib.util

    if side == 'retrieval':
        path = ROOT / 'retrieval_code' / 'combined_retrieval_coherence.py'
    else:
        path = ROOT / 'encoding_code' / 'combined_encoding_coherence.py'
    spec = importlib.util.spec_from_file_location(f'_coh_{side}', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    side_label = side.capitalize()

    if side == 'retrieval':
        blaes = mod.augment_with_allhpc_pair_composites(mod.load_blaes_retrieval())
        amme = mod.augment_with_allhpc_pair_composites(mod.load_amme_retrieval())
    else:
        blaes = mod.augment_with_allhpc_pair_composites(mod.load_blaes_encoding())
        amme = mod.augment_with_allhpc_pair_composites(mod.load_amme_encoding())

    # Build the merged "all" set the way the main script does
    def _merge(*dicts):
        merged = {}
        for d in dicts:
            for k, v in (d or {}).items():
                if isinstance(v, dict):
                    merged.setdefault(k, {}).update(v)
        return merged

    all_data = {}
    for key in [
        'group_all_power', 'stim_power', 'nostim_power',
        'bc_stim', 'bc_nostim',
        'stim_rem', 'stim_forg', 'nostim_rem', 'nostim_forg',
        'bc_stim_rem', 'bc_stim_forg', 'bc_nostim_rem', 'bc_nostim_forg',
    ]:
        all_data[key] = _merge(blaes.get(key, {}), amme.get(key, {}))
    fp_b = blaes.get('freqs_post'); fp_a = amme.get('freqs_post')
    all_data['freqs_post'] = fp_b if fp_b is not None else fp_a
    fd_b = blaes.get('freqs_diff'); fd_a = amme.get('freqs_diff')
    all_data['freqs_diff'] = fd_b if fd_b is not None else fd_a
    all_data['has_memory'] = True
    all_data['use_memory_collapsed_overall'] = blaes.get('use_memory_collapsed_overall', False) or amme.get('use_memory_collapsed_overall', False)
    all_data['has_bc_quadrant_plot'] = blaes.get('has_bc_quadrant_plot', False) or amme.get('has_bc_quadrant_plot', False)
    all_data['has_connected_dots_plot'] = blaes.get('has_connected_dots_plot', False) or amme.get('has_connected_dots_plot', False)
    all_data['overall_plot_exclude_substrings'] = set(blaes.get('overall_plot_exclude_substrings', set()) or set()) | set(amme.get('overall_plot_exclude_substrings', set()) or set())
    all_data['stim_plot_exclude_substrings'] = set(blaes.get('stim_plot_exclude_substrings', set()) or set()) | set(amme.get('stim_plot_exclude_substrings', set()) or set())
    all_data['memory_plot_exclude_substrings'] = set(blaes.get('memory_plot_exclude_substrings', set()) or set()) | set(amme.get('memory_plot_exclude_substrings', set()) or set())
    all_data['bc_plot_exclude_substrings'] = set(blaes.get('bc_plot_exclude_substrings', set()) or set()) | set(amme.get('bc_plot_exclude_substrings', set()) or set())
    all_data = mod.augment_with_allhpc_pair_composites(mod.canonicalize_coherence_data(all_data))

    out_root = ROOT / 'OUTPUTS' / f'{side}_coherence'

    POWER_RANGES = mod.POWER_RANGES
    COH_BLA = mod.COH_BLA_PANEL_PAIRS
    COH_NONBLA = mod.COH_NONBLA_PANEL_PAIRS

    for key, label, data in [('blaes', 'BLAES', blaes), ('amme', 'AMME', amme), ('all', 'All', all_data)]:
        out_dir = _ensure_dir(out_root / key)
        freqs = data.get('freqs_diff')
        if freqs is None:
            continue
        bcs = data.get('bc_stim', {})
        bcn = data.get('bc_nostim', {})
        if not bcs or not bcn:
            continue
        df_diff = mod.compute_diff_df(bcs, bcn, freqs, POWER_RANGES)
        if df_diff.empty:
            continue
        df_diff = mod.filter_region_df_by_substrings(df_diff, data.get('bc_plot_exclude_substrings', set()))
        if df_diff.empty:
            continue
        unique_rois = sorted(df_diff['Region'].unique())
        df_filtered = df_diff[df_diff['Region'].isin(unique_rois)]

        # All-trials
        df_bla_aug = mod._augment_with_bla_allhpc_diff(df_filtered, data, freqs, POWER_RANGES)
        mod.plot_filtered_significance_bargraph(
            df_bla_aug, out_dir,
            f'{label} {side_label} Baseline-Corrected Coherency Diff (Stim - No Stim) - BLA Pairs',
            'Baseline-Corrected Coherency Diff',
            f'Bargraph_baseline_corrected_coherencyDiff_byROI_{side}_filtered_sig_BLApairs.png',
            rotate_xticks=True,
            allowed_pairs=COH_BLA,
        )
        mod.plot_filtered_significance_bargraph(
            df_filtered, out_dir,
            f'{label} {side_label} Baseline-Corrected Coherency Diff (Stim - No Stim) - ALLHPC/EC/PRC Pairs',
            'Baseline-Corrected Coherency Diff',
            f'Bargraph_baseline_corrected_coherencyDiff_byROI_{side}_filtered_sig_nonBLApairs.png',
            rotate_xticks=True,
            allowed_pairs=COH_NONBLA,
        )

        # Memory-conditioned
        mem_pairs = [
            ('remembered', data.get('bc_stim_rem', {}), data.get('bc_nostim_rem', {})),
            ('forgotten', data.get('bc_stim_forg', {}), data.get('bc_nostim_forg', {})),
        ]
        for mem, sd, nd in mem_pairs:
            if not sd or not nd:
                continue
            df_mem = mod.compute_diff_df(sd, nd, freqs, POWER_RANGES)
            if df_mem.empty:
                continue
            df_mem = mod.filter_region_df_by_substrings(df_mem, data.get('bc_plot_exclude_substrings', set()))
            if df_mem.empty:
                continue
            mem_label = mem.capitalize()
            df_mem_bla_aug = mod._augment_with_bla_allhpc_diff(df_mem, data, freqs, POWER_RANGES, mem_key=mem)
            mod.plot_filtered_significance_bargraph(
                df_mem_bla_aug, out_dir,
                f'{label} {side_label} Baseline-Corrected Coherency Diff (Stim - No Stim), {mem_label} Trials - BLA Pairs',
                'Baseline-Corrected Coherency Diff',
                f'Bargraph_baseline_corrected_coherencyDiff_byROI_{side}_{mem}_filtered_sig_BLApairs.png',
                rotate_xticks=True,
                allowed_pairs=COH_BLA,
            )
            mod.plot_filtered_significance_bargraph(
                df_mem, out_dir,
                f'{label} {side_label} Baseline-Corrected Coherency Diff (Stim - No Stim), {mem_label} Trials - ALLHPC/EC/PRC Pairs',
                'Baseline-Corrected Coherency Diff',
                f'Bargraph_baseline_corrected_coherencyDiff_byROI_{side}_{mem}_filtered_sig_nonBLApairs.png',
                rotate_xticks=True,
                allowed_pairs=COH_NONBLA,
            )
        print(f'  Coherence {side} {label}: filtered_sig figures written to {out_dir}')


def regenerate_power_filtered_sig(side):
    """side: 'retrieval' or 'encoding'."""
    import importlib.util
    if side == 'retrieval':
        path = ROOT / 'retrieval_code' / 'combined_retrieval_power.py'
    else:
        path = ROOT / 'encoding_code' / 'combined_encoding_power.py'
    spec = importlib.util.spec_from_file_location(f'_pwr_{side}', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    side_label = side.capitalize()
    if side == 'retrieval':
        blaes = mod.load_blaes_retrieval()
        amme = mod.load_amme_retrieval()
    else:
        blaes = mod.load_blaes_encoding()
        amme = mod.load_amme_encoding()

    def _merge(*dicts):
        merged = {}
        for d in dicts:
            for k, v in (d or {}).items():
                if isinstance(v, dict):
                    merged.setdefault(k, {}).update(v)
        return merged

    all_data = {}
    for key in [
        'group_all_power', 'stim_power', 'nostim_power',
        'bc_stim', 'bc_nostim',
        'stim_rem', 'stim_forg', 'nostim_rem', 'nostim_forg',
        'bc_stim_rem', 'bc_stim_forg', 'bc_nostim_rem', 'bc_nostim_forg',
    ]:
        all_data[key] = _merge(blaes.get(key, {}), amme.get(key, {}))
    fp_b = blaes.get('freqs_post'); fp_a = amme.get('freqs_post')
    all_data['freqs_post'] = fp_b if fp_b is not None else fp_a
    fd_b = blaes.get('freqs_diff'); fd_a = amme.get('freqs_diff')
    all_data['freqs_diff'] = fd_b if fd_b is not None else fd_a
    if hasattr(mod, 'augment_with_power_composites'):
        all_data = mod.augment_with_power_composites(all_data)

    out_root = ROOT / 'OUTPUTS' / f'{side}_power'
    POWER_RANGES = mod.POWER_RANGES

    cohorts = [('blaes', 'BLAES', mod.augment_with_power_composites(blaes) if hasattr(mod, 'augment_with_power_composites') else blaes),
               ('amme', 'AMME', mod.augment_with_power_composites(amme) if hasattr(mod, 'augment_with_power_composites') else amme),
               ('all', 'All', all_data)]
    for key, label, data in cohorts:
        out_dir = _ensure_dir(out_root / key)
        freqs = data.get('freqs_diff')
        if freqs is None:
            continue
        bcs = data.get('bc_stim', {})
        bcn = data.get('bc_nostim', {})
        if not bcs or not bcn:
            continue
        df_diff = mod.compute_diff_df(bcs, bcn, freqs, POWER_RANGES)
        if df_diff.empty:
            continue
        unique_rois = sorted(df_diff['Region'].unique())
        df_filtered = df_diff[df_diff['Region'].isin(unique_rois)]
        mod.plot_filtered_significance_bargraph(
            df_filtered, out_dir,
            f'{label} {side_label} Baseline-Corrected Power Diff (Stim - No Stim)',
            'Baseline-Corrected Power Diff',
            f'Bargraph_baseline_corrected_powerDiff_byROI_{side}_filtered_sig.png',
            caption_text=f'Method: For each patient and ROI, baseline-corrected spectra are averaged across all stim trials and all no-stim trials separately, band means are computed, then Stim-NoStim is taken. Bars = mean across patients, error bars = SEM, dots = patient values.',
        )

        # Memory-conditioned
        for mem_key, mem_label in [('rem', 'Remembered'), ('forg', 'Forgotten')]:
            sd = data.get(f'bc_stim_{mem_key}', {})
            nd = data.get(f'bc_nostim_{mem_key}', {})
            if not sd or not nd:
                continue
            df_mem = mod.compute_diff_df(sd, nd, freqs, POWER_RANGES)
            if df_mem.empty:
                continue
            mem_cond = 'remembered' if mem_key == 'rem' else 'forgotten'
            df_mem_filtered = df_mem[df_mem['Region'].isin(unique_rois)]
            mod.plot_filtered_significance_bargraph(
                df_mem_filtered, out_dir,
                f'{label} {side_label} Baseline-Corrected Power Diff (Stim - No Stim), {mem_label} Trials',
                'Baseline-Corrected Power Diff',
                f'Bargraph_baseline_corrected_powerDiff_byROI_{side}_{mem_cond}_filtered_sig.png',
                caption_text=f'Method: For each patient and ROI, baseline-corrected spectra are averaged across stim and no-stim {mem_label.lower()} trials separately, band means are computed, then Stim-NoStim is taken. Bars = mean across patients, error bars = SEM, dots = patient values.',
            )
        print(f'  Power {side} {label}: filtered_sig figures written to {out_dir}')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--targets', nargs='+', default=['pwr_retrieval', 'pwr_encoding', 'coh_retrieval', 'coh_encoding', 'pac_retrieval', 'pac_encoding'])
    args = parser.parse_args()

    if 'pwr_retrieval' in args.targets:
        print('=== Power retrieval filtered_sig ===')
        regenerate_power_filtered_sig('retrieval')
    if 'pwr_encoding' in args.targets:
        print('=== Power encoding filtered_sig ===')
        regenerate_power_filtered_sig('encoding')
    if 'coh_retrieval' in args.targets:
        print('=== Coherence retrieval filtered_sig ===')
        regenerate_coherence_filtered_sig('retrieval')
    if 'coh_encoding' in args.targets:
        print('=== Coherence encoding filtered_sig ===')
        regenerate_coherence_filtered_sig('encoding')
    if 'pac_retrieval' in args.targets:
        print('=== PAC retrieval filtered_sig ===')
        regenerate_pac_filtered_sig('retrieval')
    if 'pac_encoding' in args.targets:
        print('=== PAC encoding filtered_sig ===')
        regenerate_pac_filtered_sig('encoding')
    print('Done.')
