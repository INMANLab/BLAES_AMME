#!/usr/bin/env python
"""
Endogenous Composite Scores
============================
Computes per-subject, per-region composite scores from baseline-corrected
endogenous (NoStim-only) data across power, coherence, and PAC for both
encoding and retrieval.

For each subject/region/band the script produces:
  - Remembered score  (band-mean of BC activity on remembered trials)
  - Forgotten score   (band-mean of BC activity on forgotten trials)
  - Difference score  (Remembered - Forgotten)

Visualisations:
  - Ranked horizontal bar charts per region/band
  - Subject x region heatmaps per band
  - Violin + dot overlays per region (split by anchor for coherence/PAC)

Frequency bands:
  Power/Coherence -- Theta (4-8 Hz), Slow gamma (35-50 Hz), HFA (70-100 Hz)
  PAC amplitude   -- Slow gamma (35-50 Hz), HFA (70-100 Hz)

Runs in two modes:
  - Unbalanced (all subjects)
  - Balanced   (exclude subjects with <10 NoStim trials in either condition)
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns

warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

OUTPUT_ROOT_UNBALANCED = os.path.join(SCRIPT_DIR, 'outputs', 'endogenous_composite_scores')
OUTPUT_ROOT_BALANCED = os.path.join(SCRIPT_DIR, 'outputs', 'endogenous_composite_scores_balanced')

from endogenous_memory import (
    ensure_dir,
    extract_endogenous,
    filter_pnas_regions,
    _merge_coherence_data,
    get_responder_status_map,
    get_anchor_region,
    RESPONDER_ORDER,
    RESPONDER_PALETTE,
)

MIN_TRIALS_PER_CONDITION = 10

# ---------------------------------------------------------------------------
# Band definitions
# ---------------------------------------------------------------------------
POWER_BANDS = {
    'Theta': (4, 8),
    'Slow gamma': (35, 50),
    'HFA': (70, 100),
}
PAC_BANDS = {
    'Slow gamma': (35, 50),
    'HFA': (70, 100),
}

# Full measure-label expansions for axis labels
MEASURE_AXIS_LABELS = {
    'Power': 'Baseline-Corrected Power (dB)',
    'Coherence': 'Baseline-Corrected Coherence',
    'PAC': 'Baseline-Corrected Modulation Index (MI)',
}


# ---------------------------------------------------------------------------
# Balanced subject filtering (reuse logic from balanced_memory_trials)
# ---------------------------------------------------------------------------

def get_nostim_trial_counts_from_power(data):
    """Count nostim rem/forg per subject from power mlmr_export_frames."""
    frames = data.get('mlmr_export_frames', [])
    if not frames:
        return {}
    df = pd.concat(frames, ignore_index=True)
    df = df[df['trial_type'] == 'nostim']
    if df.empty:
        return {}
    first_region = df.groupby('Patient')['Region'].first().to_dict()
    df = df[df.apply(lambda r: r['Region'] == first_region[r['Patient']], axis=1)]
    counts = {}
    for pat, grp in df.groupby('Patient'):
        n_rem = (grp['yes_or_no'] == 'yes').sum()
        n_forg = (grp['yes_or_no'] == 'no').sum()
        counts[pat] = {'remembered': int(n_rem), 'forgotten': int(n_forg)}
    return counts


def get_balanced_subjects(trial_counts, min_trials=MIN_TRIALS_PER_CONDITION):
    """Return set of subjects with >= min_trials in BOTH conditions."""
    if not trial_counts:
        return set()
    balanced = set()
    for pat, cnts in trial_counts.items():
        if cnts['remembered'] >= min_trials and cnts['forgotten'] >= min_trials:
            balanced.add(pat)
    return balanced


def filter_subject_dicts(data, keep_subjects):
    """Filter {region: {subject: vector}} dicts to only keep_subjects."""
    if keep_subjects is None:
        return data
    filtered = {}
    for key, value in data.items():
        if isinstance(value, dict) and value:
            first_val = next(iter(value.values()), None)
            if isinstance(first_val, dict):
                filtered[key] = {
                    roi: {s: v for s, v in sd.items() if s in keep_subjects}
                    for roi, sd in value.items()
                }
                filtered[key] = {k: v for k, v in filtered[key].items() if v}
            else:
                filtered[key] = value
        else:
            filtered[key] = value
    return filtered


# ---------------------------------------------------------------------------
# Composite score computation
# ---------------------------------------------------------------------------

def compute_composite_scores(endo, band_ranges):
    """Compute per-subject, per-region, per-band composite scores."""
    bc_rem = endo['bc_remembered']
    bc_forg = endo['bc_forgotten']
    freqs = endo['freqs_diff']
    if freqs is None or (not bc_rem and not bc_forg):
        return pd.DataFrame()

    rows = []
    all_rois = sorted(set(bc_rem.keys()) | set(bc_forg.keys()))
    for roi in all_rois:
        subjects = sorted(set(bc_rem.get(roi, {})) | set(bc_forg.get(roi, {})))
        for subj in subjects:
            for band_name, (lo, hi) in band_ranges.items():
                mask = (freqs >= lo) & (freqs <= hi)
                if not mask.any():
                    continue
                rem_val = np.nan
                forg_val = np.nan
                if roi in bc_rem and subj in bc_rem[roi]:
                    rem_val = float(np.asarray(bc_rem[roi][subj], dtype=np.float64)[mask].mean())
                if roi in bc_forg and subj in bc_forg[roi]:
                    forg_val = float(np.asarray(bc_forg[roi][subj], dtype=np.float64)[mask].mean())
                diff_val = np.nan
                if not np.isnan(rem_val) and not np.isnan(forg_val):
                    diff_val = rem_val - forg_val
                rows.append({
                    'Patient': subj,
                    'Region': roi,
                    'Band': band_name,
                    'remembered': rem_val,
                    'forgotten': forg_val,
                    'difference': diff_val,
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _add_responder_colors(df):
    rmap = get_responder_status_map()
    df = df.copy()
    df['Responder_status'] = df['Patient'].map(rmap).fillna('Unknown')
    return df


def _responder_legend(df_with_status, fontsize=13):
    """Build responder legend handles from a df that has Responder_status."""
    present = set(df_with_status['Responder_status'])
    handles = []
    for status in RESPONDER_ORDER:
        if status in present:
            handles.append(
                Line2D([0], [0], color=RESPONDER_PALETTE[status], lw=8, label=status))
    if 'Unknown' in present:
        handles.append(
            Line2D([0], [0], color=RESPONDER_PALETTE['Unknown'], lw=8, label='Unknown'))
    return handles


def _build_title(label, phase_label, measure_label, region_or_band, n_subjects, balanced):
    """Build a standardised figure title."""
    bal_tag = 'Balanced Memory' if balanced else 'Unbalanced (All Subjects)'
    return (
        f'{label} {phase_label} {measure_label} -- {region_or_band}\n'
        f'{bal_tag}  |  N = {n_subjects}'
    )


def _axis_label(measure_label):
    return MEASURE_AXIS_LABELS.get(measure_label, f'BC {measure_label}')


# ---------------------------------------------------------------------------
# Ranked bars
# ---------------------------------------------------------------------------

def plot_ranked_bars(df, out_dir, region, band, measure_label, phase_label, label, balanced):
    sub = df[(df['Region'] == region) & (df['Band'] == band)].copy()
    if sub.empty:
        return
    sub = _add_responder_colors(sub)
    n_subj = sub['Patient'].nunique()

    fig, axes = plt.subplots(1, 3, figsize=(22, max(5, 0.38 * len(sub))))
    value_cols = ['remembered', 'forgotten', 'difference']
    panel_titles = ['Remembered (NoStim)', 'Forgotten (NoStim)', 'Remembered - Forgotten']

    for ax, col, ptitle in zip(axes, value_cols, panel_titles):
        plot_sub = sub.dropna(subset=[col]).sort_values(col)
        if plot_sub.empty:
            ax.set_visible(False)
            continue
        colors = [RESPONDER_PALETTE.get(s, RESPONDER_PALETTE['Unknown'])
                  for s in plot_sub['Responder_status']]
        ax.barh(range(len(plot_sub)), plot_sub[col], color=colors,
                edgecolor='#4D4D4D', linewidth=0.6, height=0.72)
        ax.set_yticks(range(len(plot_sub)))
        ax.set_yticklabels(plot_sub['Patient'], fontsize=10, fontweight='bold')
        ax.axvline(0, color='#4D4D4D', linewidth=1)
        ax.set_xlabel(_axis_label(measure_label), fontsize=12, fontweight='bold')
        ax.set_title(ptitle, fontsize=14, fontweight='bold')
        ax.tick_params(axis='x', labelsize=10)

    title = _build_title(label, phase_label, measure_label, f'{region} -- {band}', n_subj, balanced)
    fig.suptitle(title, fontsize=16, fontweight='bold', y=1.04)

    legend_handles = _responder_legend(sub, fontsize=13)
    if legend_handles:
        fig.legend(handles=legend_handles, loc='upper center',
                   bbox_to_anchor=(0.5, 0.99), ncol=min(5, len(legend_handles)),
                   frameon=False, fontsize=13)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    safe = f'{region}_{band}'.replace('/', '_').replace(' ', '_').lower()
    fname = f'Ranked_{measure_label}_{phase_label.lower()}_{safe}.png'
    plt.savefig(os.path.join(out_dir, fname), dpi=300, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Heatmaps
# ---------------------------------------------------------------------------

def plot_heatmap(df, out_dir, band, measure_label, phase_label, label, value_col, title_suffix, balanced):
    sub = df[df['Band'] == band].dropna(subset=[value_col])
    if sub.empty:
        return
    n_subj = sub['Patient'].nunique()
    pivot = sub.pivot_table(index='Patient', columns='Region', values=value_col)
    if pivot.empty:
        return
    pivot = pivot.loc[pivot.mean(axis=1).sort_values().index]

    fig_height = max(5, 0.35 * len(pivot))
    fig_width = max(8, 0.8 * len(pivot.columns) + 3)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    vmax = max(abs(pivot.min().min()), abs(pivot.max().max()))
    if np.isnan(vmax) or vmax == 0:
        vmax = 1
    sns.heatmap(pivot, ax=ax, cmap='RdBu_r', center=0, vmin=-vmax, vmax=vmax,
                linewidths=0.5, linecolor='white',
                cbar_kws={'label': _axis_label(measure_label), 'shrink': 0.7},
                xticklabels=True, yticklabels=True)
    ax.set_xlabel('Region', fontsize=13, fontweight='bold')
    ax.set_ylabel('Patient', fontsize=13, fontweight='bold')
    ax.tick_params(axis='x', labelsize=9, rotation=45)
    ax.tick_params(axis='y', labelsize=9)

    title = _build_title(label, phase_label, measure_label,
                         f'{band} -- {title_suffix}', n_subj, balanced)
    ax.set_title(title, fontsize=15, fontweight='bold')
    fig.tight_layout()
    safe_band = band.lower().replace(' ', '_')
    safe_suffix = title_suffix.lower().replace(' ', '_').replace('-', 'minus')
    fname = f'Heatmap_{measure_label}_{phase_label.lower()}_{safe_band}_{safe_suffix}.png'
    plt.savefig(os.path.join(out_dir, fname), dpi=300, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Violin overlays — with anchor-region splitting for coherence/PAC
# ---------------------------------------------------------------------------

def _plot_single_violin(sub, regions, out_dir, measure_label, phase_label,
                        label, band, value_col, title_suffix, balanced,
                        fname_extra=''):
    """Render one violin figure for the given subset of regions."""
    if not regions:
        return
    n_subj = sub['Patient'].nunique()

    fig_width = max(10, 1.1 * len(regions) + 3)
    fig, ax = plt.subplots(figsize=(fig_width, 8))

    violin_data = [sub[sub['Region'] == r][value_col].dropna().values for r in regions]
    # Skip regions with <2 data points for violin (need at least 2)
    valid_mask = [len(d) >= 2 for d in violin_data]
    valid_regions = [r for r, v in zip(regions, valid_mask) if v]
    valid_data = [d for d, v in zip(violin_data, valid_mask) if v]

    if valid_data:
        parts = ax.violinplot(valid_data,
                              positions=[regions.index(r) for r in valid_regions],
                              showmeans=True, showmedians=False, showextrema=False)
        for pc in parts['bodies']:
            pc.set_facecolor('#D3D3D3')
            pc.set_alpha(0.5)
        parts['cmeans'].set_color('#4D4D4D')
        parts['cmeans'].set_linewidth(2)

    rng = np.random.default_rng(42)
    for idx, region in enumerate(regions):
        rsub = sub[sub['Region'] == region]
        if rsub.empty:
            continue
        jitter = rng.uniform(-0.18, 0.18, len(rsub))
        colors = [RESPONDER_PALETTE.get(s, RESPONDER_PALETTE['Unknown'])
                  for s in rsub['Responder_status']]
        ax.scatter(np.full(len(rsub), idx, dtype=float) + jitter,
                   rsub[value_col], c=colors, s=55, alpha=0.85,
                   edgecolors='none', zorder=3)

    ax.axhline(0, color='#4D4D4D', linewidth=1, linestyle='--')
    ax.set_xticks(range(len(regions)))
    ax.set_xticklabels(regions, fontsize=11, fontweight='bold', rotation=45, ha='right')
    ax.set_ylabel(_axis_label(measure_label), fontsize=14, fontweight='bold')
    ax.tick_params(axis='y', labelsize=12)

    region_tag = f'{fname_extra} -- ' if fname_extra else ''
    title = _build_title(label, phase_label, measure_label,
                         f'{region_tag}{band} -- {title_suffix}', n_subj, balanced)
    fig.suptitle(title, fontsize=16, fontweight='bold', y=1.04)

    legend_handles = _responder_legend(sub, fontsize=13)
    if legend_handles:
        fig.legend(handles=legend_handles, loc='upper center',
                   bbox_to_anchor=(0.5, 0.99), ncol=min(5, len(legend_handles)),
                   frameon=False, fontsize=13)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    safe_band = band.lower().replace(' ', '_')
    safe_suffix = title_suffix.lower().replace(' ', '_').replace('-', 'minus')
    safe_extra = fname_extra.lower().replace(' ', '_') if fname_extra else ''
    extra_part = f'_{safe_extra}' if safe_extra else ''
    fname = f'Violin_{measure_label}_{phase_label.lower()}_{safe_band}_{safe_suffix}{extra_part}.png'
    plt.savefig(os.path.join(out_dir, fname), dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_violin_overlay(df, out_dir, band, measure_label, phase_label, label,
                        value_col, title_suffix, balanced, split_by_anchor=False):
    """Violin plot — optionally split by anchor region for readability."""
    sub = df[df['Band'] == band].dropna(subset=[value_col])
    if sub.empty:
        return
    sub = _add_responder_colors(sub)
    regions = sorted(sub['Region'].unique())

    if not split_by_anchor or '_' not in regions[0]:
        _plot_single_violin(sub, regions, out_dir, measure_label, phase_label,
                            label, band, value_col, title_suffix, balanced)
        return

    # Split by anchor (first part before _)
    anchor_groups = {}
    for r in regions:
        anchor = get_anchor_region(r)
        anchor_groups.setdefault(anchor, []).append(r)

    for anchor in sorted(anchor_groups):
        group_regions = anchor_groups[anchor]
        group_sub = sub[sub['Region'].isin(group_regions)]
        if group_sub.empty:
            continue
        _plot_single_violin(group_sub, group_regions, out_dir, measure_label,
                            phase_label, label, band, value_col, title_suffix,
                            balanced, fname_extra=f'{anchor}_connections')


# ---------------------------------------------------------------------------
# Generate all plots for one measure/phase
# ---------------------------------------------------------------------------

def generate_composite_outputs(endo, out_dir, label, measure_label, phase_label,
                               band_ranges, balanced=False):
    """Compute scores and generate all composite visualisations."""
    ensure_dir(out_dir)
    scores = compute_composite_scores(endo, band_ranges)
    if scores.empty:
        print(f"  No composite scores for {label} {phase_label} {measure_label} -- skipping.")
        return scores

    n_subjects = scores['Patient'].nunique()
    n_regions = scores['Region'].nunique()
    print(f"  {label} {phase_label} {measure_label}: "
          f"{n_subjects} subjects, {n_regions} regions, {len(scores)} rows")

    csv_path = os.path.join(out_dir, f'composite_scores_{measure_label.lower()}_{phase_label.lower()}.csv')
    scores.to_csv(csv_path, index=False)

    # Decide whether to split violins by anchor
    split_anchor = measure_label in ('Coherence', 'PAC')

    # Ranked bars per region
    ranked_dir = ensure_dir(os.path.join(out_dir, 'ranked_bars'))
    for region in sorted(scores['Region'].unique()):
        for band in band_ranges:
            plot_ranked_bars(scores, ranked_dir, region, band, measure_label,
                            phase_label, label, balanced)

    # Heatmaps
    heatmap_dir = ensure_dir(os.path.join(out_dir, 'heatmaps'))
    for band in band_ranges:
        for col, suffix in [('remembered', 'Remembered'),
                            ('forgotten', 'Forgotten'),
                            ('difference', 'Remembered - Forgotten')]:
            plot_heatmap(scores, heatmap_dir, band, measure_label, phase_label,
                         label, col, suffix, balanced)

    # Violin overlays
    violin_dir = ensure_dir(os.path.join(out_dir, 'violins'))
    for band in band_ranges:
        for col, suffix in [('remembered', 'Remembered'),
                            ('forgotten', 'Forgotten'),
                            ('difference', 'Remembered - Forgotten')]:
            plot_violin_overlay(scores, violin_dir, band, measure_label, phase_label,
                                label, col, suffix, balanced, split_by_anchor=split_anchor)

    return scores


# ---------------------------------------------------------------------------
# Data loading helpers (shared by balanced and unbalanced)
# ---------------------------------------------------------------------------

def _load_power_data():
    from combined_encoding_power import (
        load_blaes_encoding, load_amme_encoding, build_all_encoding_data,
    )
    from combined_retrieval_power import (
        load_blaes_retrieval, load_amme_retrieval, build_all_retrieval_data,
    )
    print("\n  Loading encoding power...")
    blaes_enc = filter_pnas_regions(load_blaes_encoding())
    amme_enc = filter_pnas_regions(load_amme_encoding())
    all_enc = filter_pnas_regions(build_all_encoding_data(blaes_enc, amme_enc))

    print("\n  Loading retrieval power...")
    blaes_ret = filter_pnas_regions(load_blaes_retrieval())
    amme_ret = filter_pnas_regions(load_amme_retrieval())
    all_ret = filter_pnas_regions(build_all_retrieval_data(blaes_ret, amme_ret))

    return {
        'encoding': [('BLAES', blaes_enc), ('AMME', amme_enc), ('All', all_enc)],
        'retrieval': [('BLAES', blaes_ret), ('AMME', amme_ret), ('All', all_ret)],
    }


def _load_coherence_data():
    from combined_encoding_coherence import (
        load_blaes_encoding as lb_enc, load_amme_encoding as la_enc,
    )
    from combined_retrieval_coherence import (
        load_blaes_retrieval as lb_ret, load_amme_retrieval as la_ret,
    )
    print("\n  Loading encoding coherence...")
    blaes_enc = filter_pnas_regions(lb_enc())
    amme_enc = filter_pnas_regions(la_enc())
    all_enc = filter_pnas_regions(_merge_coherence_data(blaes_enc, amme_enc))

    print("\n  Loading retrieval coherence...")
    blaes_ret = filter_pnas_regions(lb_ret())
    amme_ret = filter_pnas_regions(la_ret())
    all_ret = filter_pnas_regions(_merge_coherence_data(blaes_ret, amme_ret))

    return {
        'encoding': [('BLAES', blaes_enc), ('AMME', amme_enc), ('All', all_enc)],
        'retrieval': [('BLAES', blaes_ret), ('AMME', amme_ret), ('All', all_ret)],
    }


def _load_pac_data():
    from combined_encoding_pac import load_grouped_encoding_pac_data
    from combined_retrieval_pac import load_grouped_retrieval_pac_data

    print("\n  Loading encoding PAC...")
    grouped_enc = load_grouped_encoding_pac_data()
    print("\n  Loading retrieval PAC...")
    grouped_ret = load_grouped_retrieval_pac_data()

    enc_list = [(gl, filter_pnas_regions(grouped_enc[gk]))
                for gk, gl in [('blaes', 'BLAES'), ('amme', 'AMME'), ('all', 'All')]]
    ret_list = [(gl, filter_pnas_regions(grouped_ret[gk]))
                for gk, gl in [('blaes', 'BLAES'), ('amme', 'AMME'), ('all', 'All')]]
    return {'encoding': enc_list, 'retrieval': ret_list}


# ---------------------------------------------------------------------------
# Run one pass (balanced or unbalanced)
# ---------------------------------------------------------------------------

def run_all(output_root, balanced=False, balanced_subjects_enc=None, balanced_subjects_ret=None):
    """Run composite score analysis for all measures."""

    # --- POWER ---
    print("\n" + "=" * 60)
    tag = "BALANCED" if balanced else "UNBALANCED"
    print(f"Endogenous Composite Scores ({tag}) -- POWER")
    print("=" * 60)
    power = _load_power_data()

    for phase in ['encoding', 'retrieval']:
        bal_map = balanced_subjects_enc if phase == 'encoding' else balanced_subjects_ret
        for group_label, data in power[phase]:
            if balanced and bal_map:
                keep = bal_map.get(group_label.lower())
                data = filter_subject_dicts(data, keep)
            endo = extract_endogenous(data, measure='power')
            out = ensure_dir(os.path.join(output_root, f'{phase}_power', group_label.lower()))
            generate_composite_outputs(endo, out, group_label, 'Power',
                                       phase.capitalize(), POWER_BANDS, balanced=balanced)

    # --- COHERENCE ---
    print("\n" + "=" * 60)
    print(f"Endogenous Composite Scores ({tag}) -- COHERENCE")
    print("=" * 60)
    coherence = _load_coherence_data()

    for phase in ['encoding', 'retrieval']:
        bal_map = balanced_subjects_enc if phase == 'encoding' else balanced_subjects_ret
        for group_label, data in coherence[phase]:
            if balanced and bal_map:
                keep = bal_map.get(group_label.lower())
                data = filter_subject_dicts(data, keep)
            endo = extract_endogenous(data, measure='coherence')
            out = ensure_dir(os.path.join(output_root, f'{phase}_coherence', group_label.lower()))
            generate_composite_outputs(endo, out, group_label, 'Coherence',
                                       phase.capitalize(), POWER_BANDS, balanced=balanced)

    # --- PAC ---
    print("\n" + "=" * 60)
    print(f"Endogenous Composite Scores ({tag}) -- PAC")
    print("=" * 60)
    pac = _load_pac_data()

    for phase in ['encoding', 'retrieval']:
        bal_map = balanced_subjects_enc if phase == 'encoding' else balanced_subjects_ret
        for group_label, data in pac[phase]:
            if balanced and bal_map:
                keep = bal_map.get(group_label.lower())
                data = filter_subject_dicts(data, keep)
            endo = extract_endogenous(data, measure='pac')
            out = ensure_dir(os.path.join(output_root, f'{phase}_pac', group_label.lower()))
            generate_composite_outputs(endo, out, group_label, 'PAC',
                                       phase.capitalize(), PAC_BANDS, balanced=balanced)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Endogenous Composite Scores")
    print("Per-subject BC band-mean scores for NoStim remembered & forgotten")
    print("=" * 60)

    # ---- 1) UNBALANCED (all subjects) ----
    ensure_dir(OUTPUT_ROOT_UNBALANCED)
    run_all(OUTPUT_ROOT_UNBALANCED, balanced=False)

    # ---- 2) BALANCED (exclude subjects with <10 trials in either condition) ----
    # Compute balanced subject lists from power data
    print("\n" + "=" * 60)
    print("Computing balanced subject lists from power data...")
    print("=" * 60)

    from combined_encoding_power import (
        load_blaes_encoding, load_amme_encoding, build_all_encoding_data,
    )
    from combined_retrieval_power import (
        load_blaes_retrieval, load_amme_retrieval, build_all_retrieval_data,
    )

    blaes_enc = filter_pnas_regions(load_blaes_encoding())
    amme_enc = filter_pnas_regions(load_amme_encoding())
    all_enc = filter_pnas_regions(build_all_encoding_data(blaes_enc, amme_enc))

    blaes_ret = filter_pnas_regions(load_blaes_retrieval())
    amme_ret = filter_pnas_regions(load_amme_retrieval())
    all_ret = filter_pnas_regions(build_all_retrieval_data(blaes_ret, amme_ret))

    balanced_enc = {}
    for gk, data in [('blaes', blaes_enc), ('amme', amme_enc), ('all', all_enc)]:
        counts = get_nostim_trial_counts_from_power(data)
        balanced_enc[gk] = get_balanced_subjects(counts)
        print(f"  Encoding {gk.upper()}: {len(balanced_enc[gk])} balanced subjects")

    balanced_ret = {}
    for gk, data in [('blaes', blaes_ret), ('amme', amme_ret), ('all', all_ret)]:
        counts = get_nostim_trial_counts_from_power(data)
        balanced_ret[gk] = get_balanced_subjects(counts)
        print(f"  Retrieval {gk.upper()}: {len(balanced_ret[gk])} balanced subjects")

    ensure_dir(OUTPUT_ROOT_BALANCED)
    run_all(OUTPUT_ROOT_BALANCED, balanced=True,
            balanced_subjects_enc=balanced_enc,
            balanced_subjects_ret=balanced_ret)

    print("\n" + "=" * 60)
    print(f"Done!")
    print(f"  Unbalanced: {OUTPUT_ROOT_UNBALANCED}")
    print(f"  Balanced:   {OUTPUT_ROOT_BALANCED}")
    print("=" * 60)


if __name__ == '__main__':
    main()
