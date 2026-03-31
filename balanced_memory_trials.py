#!/usr/bin/env python
"""
Balanced Memory Trials Analysis
================================
Reruns all endogenous memory analyses (remembered vs forgotten, NoStim only)
but EXCLUDES any subject who has fewer than 10 trials in either the remembered
or forgotten condition. This prevents skewed effects from imbalanced trial counts.

Covers power, coherence, and PAC for both encoding and retrieval.
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')

warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

OUTPUT_ROOT = os.path.join(SCRIPT_DIR, 'outputs', 'balanced_memory_trials')
CSV_OUTPUT_DIR = os.path.join(OUTPUT_ROOT, 'csvs')

MIN_TRIALS_PER_CONDITION = 10

from endogenous_memory import (
    ensure_dir,
    extract_endogenous,
    generate_endogenous_plots,
    export_endogenous_mlmr_csv,
    PAC_BANDS,
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


def get_balanced_subjects(trial_counts, min_trials=MIN_TRIALS_PER_CONDITION):
    """Return set of subjects with >= min_trials in BOTH conditions."""
    if not trial_counts:
        return None
    balanced = set()
    excluded = []
    for pat, cnts in sorted(trial_counts.items()):
        if cnts['remembered'] >= min_trials and cnts['forgotten'] >= min_trials:
            balanced.add(pat)
        else:
            excluded.append((pat, cnts['remembered'], cnts['forgotten']))
    print(f"    Balanced subjects: {len(balanced)} / {len(trial_counts)} "
          f"(excluded {len(excluded)} with <{min_trials} trials in a condition)")
    for pat, r, f in excluded:
        print(f"      Excluded: {pat} (rem={r}, forg={f})")
    return balanced


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

    # --- Encoding ---
    print("\n  Loading encoding power data...")
    blaes_enc = load_blaes_encoding()
    amme_enc = load_amme_encoding()
    all_enc = build_all_encoding_data(blaes_enc, amme_enc)

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

    for group_label, data in [('blaes', blaes_enc), ('amme', amme_enc), ('all', all_enc)]:
        keep = balanced_subjects[group_label]
        filtered_data = filter_mlmr_frames(data, keep)
        export_endogenous_mlmr_csv(filtered_data, CSV_OUTPUT_DIR,
                                   f'balanced_encoding_power_{group_label}_mlmr_input')

    # --- Retrieval ---
    print("\n  Loading retrieval power data...")
    blaes_ret = load_blaes_retrieval()
    amme_ret = load_amme_retrieval()
    all_ret = build_all_retrieval_data(blaes_ret, amme_ret)

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

    for group_label, data in [('blaes', blaes_ret), ('amme', amme_ret), ('all', all_ret)]:
        keep = balanced_subjects_ret[group_label]
        filtered_data = filter_mlmr_frames(data, keep)
        export_endogenous_mlmr_csv(filtered_data, CSV_OUTPUT_DIR,
                                   f'balanced_retrieval_power_{group_label}_mlmr_input')

    print("  Power analysis complete.")
    return balanced_subjects, balanced_subjects_ret


# ---------------------------------------------------------------------------
# COHERENCE
# ---------------------------------------------------------------------------

def run_coherence_analysis():
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

    # --- Encoding ---
    print("\n  Loading encoding coherence data...")
    blaes_enc = load_blaes_encoding_coh()
    amme_enc = load_amme_encoding_coh()
    all_enc = _merge_coherence_data(blaes_enc, amme_enc)

    print("\n  Determining balanced subjects (encoding coherence)...")
    balanced_subjects = {}
    for group_label, data in [('blaes', blaes_enc), ('amme', amme_enc), ('all', all_enc)]:
        print(f"    {group_label.upper()}:")
        counts = get_nostim_trial_counts(data, measure='coherence')
        balanced_subjects[group_label] = get_balanced_subjects(counts)

    for group_label, data in [('BLAES', blaes_enc), ('AMME', amme_enc), ('All', all_enc)]:
        keep = balanced_subjects[group_label.lower()]
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

    for group_label, data in [('blaes', blaes_enc), ('amme', amme_enc), ('all', all_enc)]:
        keep = balanced_subjects[group_label]
        filtered_data = filter_mlmr_frames(data, keep)
        export_endogenous_mlmr_csv(filtered_data, CSV_OUTPUT_DIR,
                                   f'balanced_encoding_coherence_{group_label}_mlmr_input',
                                   measure_name='Coherence')

    # --- Retrieval ---
    print("\n  Loading retrieval coherence data...")
    blaes_ret = load_blaes_retrieval_coh()
    amme_ret = load_amme_retrieval_coh()
    all_ret = _merge_coherence_data(blaes_ret, amme_ret)

    print("\n  Determining balanced subjects (retrieval coherence)...")
    balanced_subjects_ret = {}
    for group_label, data in [('blaes', blaes_ret), ('amme', amme_ret), ('all', all_ret)]:
        print(f"    {group_label.upper()}:")
        counts = get_nostim_trial_counts(data, measure='coherence')
        balanced_subjects_ret[group_label] = get_balanced_subjects(counts)

    for group_label, data in [('BLAES', blaes_ret), ('AMME', amme_ret), ('All', all_ret)]:
        keep = balanced_subjects_ret[group_label.lower()]
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

    for group_label, data in [('blaes', blaes_ret), ('amme', amme_ret), ('all', all_ret)]:
        keep = balanced_subjects_ret[group_label]
        filtered_data = filter_mlmr_frames(data, keep)
        export_endogenous_mlmr_csv(filtered_data, CSV_OUTPUT_DIR,
                                   f'balanced_retrieval_coherence_{group_label}_mlmr_input',
                                   measure_name='Coherence')

    print("  Coherence analysis complete.")


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

    # --- Encoding ---
    print("\n  Loading encoding PAC data...")
    grouped_enc = load_grouped_encoding_pac_data()
    for group_key, label in [('blaes', 'BLAES'), ('amme', 'AMME'), ('all', 'All')]:
        plot_label = balanced_plot_label(label)
        keep = enc_balanced_subjects.get(group_key) if enc_balanced_subjects else None
        if keep is not None:
            print(f"    {label}: using {len(keep)} balanced subjects from power encoding filter")
        filtered_data = filter_subject_dicts(grouped_enc[group_key], keep)
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

    # --- Retrieval ---
    print("\n  Loading retrieval PAC data...")
    grouped_ret = load_grouped_retrieval_pac_data()
    for group_key, label in [('blaes', 'BLAES'), ('amme', 'AMME'), ('all', 'All')]:
        plot_label = balanced_plot_label(label)
        keep = ret_balanced_subjects.get(group_key) if ret_balanced_subjects else None
        if keep is not None:
            print(f"    {label}: using {len(keep)} balanced subjects from power retrieval filter")
        filtered_data = filter_subject_dicts(grouped_ret[group_key], keep)
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

    print("  PAC analysis complete.")


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

    enc_balanced, ret_balanced = run_power_analysis()
    run_coherence_analysis()
    run_pac_analysis(enc_balanced_subjects=enc_balanced, ret_balanced_subjects=ret_balanced)

    print("\n" + "=" * 60)
    print(f"Done! Balanced memory trial outputs saved to: {OUTPUT_ROOT}")
    print("=" * 60)


if __name__ == '__main__':
    main()
