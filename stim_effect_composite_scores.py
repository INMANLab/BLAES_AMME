#!/usr/bin/env python
"""
Stimulation Effect on Composite Scores
========================================
Shows how stimulation changes endogenous memory composite scores.
For each subject/region/band, computes:
  - NoStim Remembered   vs  Stim Remembered   (side by side)
  - NoStim Forgotten    vs  Stim Forgotten     (side by side)

Produces ranked bar charts and violin plots (no heatmaps).
Coherence and PAC violins are split by anchor region.
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
from matplotlib.colors import to_rgba
import seaborn as sns

warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

OUTPUT_ROOT = os.path.join(SCRIPT_DIR, 'outputs', 'stim_effect_composite_scores')

from endogenous_memory import (
    ensure_dir,
    filter_pnas_regions,
    _merge_coherence_data,
    get_responder_status_map,
    get_anchor_region,
    RESPONDER_ORDER,
    RESPONDER_PALETTE,
)

POWER_BANDS = {
    'Theta': (4, 8),
    'Slow gamma': (35, 50),
    'HFA': (70, 100),
}
PAC_BANDS = {
    'Slow gamma': (35, 50),
    'HFA': (70, 100),
}

MEASURE_AXIS_LABELS = {
    'Power': 'Baseline-Corrected Power (dB)',
    'Coherence': 'Baseline-Corrected Coherence',
    'PAC': 'Baseline-Corrected Modulation Index (MI)',
}

STIM_COLORS = {
    'NoStim': '#1f77b4',
    'Stim': '#d62728',
}


# ---------------------------------------------------------------------------
# Extract both nostim and stim BC data
# ---------------------------------------------------------------------------

def extract_stim_and_nostim(data, measure='power'):
    """Pull nostim AND stim remembered/forgotten from loaded data dict."""
    is_pac = measure == 'pac'
    if is_pac:
        return {
            'nostim_rem': data.get('diff_nostim_rem', {}),
            'nostim_forg': data.get('diff_nostim_forg', {}),
            'stim_rem': data.get('diff_stim_rem', {}),
            'stim_forg': data.get('diff_stim_forg', {}),
            'freqs': data.get('freqs_diff'),
        }
    else:
        return {
            'nostim_rem': data.get('bc_nostim_rem', {}),
            'nostim_forg': data.get('bc_nostim_forg', {}),
            'stim_rem': data.get('bc_stim_rem', {}),
            'stim_forg': data.get('bc_stim_forg', {}),
            'freqs': data.get('freqs_diff'),
        }


# ---------------------------------------------------------------------------
# Composite scores: both conditions
# ---------------------------------------------------------------------------

def compute_stim_composite(extracted, band_ranges):
    """Compute composite scores for nostim and stim, remembered and forgotten.

    Returns DataFrame with columns:
        Patient, Region, Band, nostim_rem, stim_rem, nostim_forg, stim_forg
    """
    freqs = extracted['freqs']
    if freqs is None:
        return pd.DataFrame()

    condition_keys = ['nostim_rem', 'stim_rem', 'nostim_forg', 'stim_forg']
    all_rois = set()
    for ck in condition_keys:
        all_rois.update(extracted[ck].keys())
    all_rois = sorted(all_rois)

    all_subjects = set()
    for ck in condition_keys:
        for roi, sd in extracted[ck].items():
            all_subjects.update(sd.keys())

    rows = []
    for roi in all_rois:
        for subj in sorted(all_subjects):
            for band_name, (lo, hi) in band_ranges.items():
                mask = (freqs >= lo) & (freqs <= hi)
                if not mask.any():
                    continue
                vals = {}
                for ck in condition_keys:
                    if roi in extracted[ck] and subj in extracted[ck][roi]:
                        arr = np.asarray(extracted[ck][roi][subj], dtype=np.float64)
                        vals[ck] = float(arr[mask].mean())
                    else:
                        vals[ck] = np.nan
                # Only include if subject has at least one non-nan value
                if all(np.isnan(v) for v in vals.values()):
                    continue
                rows.append({
                    'Patient': subj,
                    'Region': roi,
                    'Band': band_name,
                    **vals,
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _add_responder(df):
    rmap = get_responder_status_map()
    df = df.copy()
    df['Responder_status'] = df['Patient'].map(rmap).fillna('Unknown')
    return df


def _responder_legend(df, fontsize=13):
    present = set(df['Responder_status'])
    handles = []
    for s in RESPONDER_ORDER:
        if s in present:
            handles.append(Line2D([0], [0], marker='o', linestyle='', markersize=9,
                                  markerfacecolor=RESPONDER_PALETTE[s],
                                  markeredgecolor='none', label=s))
    if 'Unknown' in present:
        handles.append(Line2D([0], [0], marker='o', linestyle='', markersize=9,
                              markerfacecolor=RESPONDER_PALETTE['Unknown'],
                              markeredgecolor='none', label='Unknown'))
    return handles


def _axis_label(measure_label):
    return MEASURE_AXIS_LABELS.get(measure_label, f'BC {measure_label}')


def _draw_violin(ax, values, position, color, width=0.32):
    """Draw a filled violin when enough samples exist; return whether it rendered."""
    if len(values) < 2:
        return False
    parts = ax.violinplot(
        values,
        positions=[position],
        widths=width,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for body in parts['bodies']:
        body.set_facecolor(to_rgba(color, alpha=0.22))
        body.set_edgecolor(color)
        body.set_linewidth(1.5)
    return True


def plot_ranked_stim_effect(df, out_dir, region, band, measure_label, phase_label, label):
    """4-panel ranked bar: NoStim Rem, Stim Rem, NoStim Forg, Stim Forg."""
    sub = df[(df['Region'] == region) & (df['Band'] == band)].copy()
    if sub.empty:
        return
    sub = _add_responder(sub)
    n_subj = sub['Patient'].nunique()

    fig, axes = plt.subplots(1, 4, figsize=(28, max(5, 0.38 * len(sub))))
    panels = [
        ('nostim_rem', 'NoStim Remembered', STIM_COLORS['NoStim']),
        ('stim_rem', 'Stim Remembered', STIM_COLORS['Stim']),
        ('nostim_forg', 'NoStim Forgotten', STIM_COLORS['NoStim']),
        ('stim_forg', 'Stim Forgotten', STIM_COLORS['Stim']),
    ]

    for ax, (col, ptitle, base_color) in zip(axes, panels):
        plot_sub = sub.dropna(subset=[col]).sort_values(col)
        if plot_sub.empty:
            ax.set_visible(False)
            continue
        if col.startswith('nostim'):
            colors = [to_rgba(base_color, alpha=0.7)] * len(plot_sub)
        else:
            colors = [RESPONDER_PALETTE.get(s, RESPONDER_PALETTE['Unknown'])
                      for s in plot_sub['Responder_status']]
        bars = ax.barh(range(len(plot_sub)), plot_sub[col], color=colors,
                       edgecolor=base_color, linewidth=1.5, height=0.72)
        ax.set_yticks(range(len(plot_sub)))
        ax.set_yticklabels(plot_sub['Patient'], fontsize=9, fontweight='bold')
        ax.axvline(0, color='#4D4D4D', linewidth=1)
        ax.set_xlabel(_axis_label(measure_label), fontsize=11, fontweight='bold')
        ax.set_title(ptitle, fontsize=13, fontweight='bold', color=base_color)
        ax.tick_params(axis='x', labelsize=9)

    fig.suptitle(
        f'{label} {phase_label} {measure_label} -- {region} -- {band}\n'
        f'Stimulation Effect on Memory Composite Scores  |  N = {n_subj}',
        fontsize=16, fontweight='bold', y=1.04,
    )
    legend_handles = _responder_legend(sub)
    if legend_handles:
        fig.legend(handles=legend_handles, loc='upper center',
                   bbox_to_anchor=(0.5, 0.99), ncol=min(5, len(legend_handles)),
                   frameon=False, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    safe = f'{region}_{band}'.replace('/', '_').replace(' ', '_').lower()
    fname = f'StimEffect_Ranked_{measure_label}_{phase_label.lower()}_{safe}.png'
    plt.savefig(os.path.join(out_dir, fname), dpi=300, bbox_inches='tight')
    plt.close(fig)


def _plot_single_stim_violin(sub, regions, out_dir, measure_label, phase_label,
                              label, band, fname_extra=''):
    """2-row violin: top = Remembered (NoStim vs Stim), bottom = Forgotten."""
    if not regions:
        return
    n_subj = sub['Patient'].nunique()

    fig_width = max(10, 1.1 * len(regions) + 3)
    fig, (ax_rem, ax_forg) = plt.subplots(2, 1, figsize=(fig_width, 14), sharex=True)

    rng = np.random.default_rng(42)
    bar_width = 0.35

    for ax, (ns_col, st_col, row_title) in zip(
        [ax_rem, ax_forg],
        [('nostim_rem', 'stim_rem', 'Remembered'),
         ('nostim_forg', 'stim_forg', 'Forgotten')],
    ):
        for idx, region in enumerate(regions):
            rsub = sub[sub['Region'] == region]
            if rsub.empty:
                continue

            # NoStim dots (left)
            ns_vals = rsub[ns_col].dropna()
            if len(ns_vals) > 0:
                ns_pos = idx - bar_width / 2
                _draw_violin(ax, ns_vals.values, ns_pos, STIM_COLORS['NoStim'])
                jitter = rng.uniform(-0.12, 0.12, len(ns_vals))
                ax.scatter(np.full(len(ns_vals), ns_pos) + jitter,
                           ns_vals, c=STIM_COLORS['NoStim'], s=50, alpha=0.7,
                           edgecolors='none', zorder=3)
                ax.plot([ns_pos - 0.2, ns_pos + 0.2],
                        [ns_vals.mean(), ns_vals.mean()],
                        color=STIM_COLORS['NoStim'], linewidth=3, zorder=4)

            # Stim dots (right)
            st_vals = rsub[st_col].dropna()
            if len(st_vals) > 0:
                st_pos = idx + bar_width / 2
                _draw_violin(ax, st_vals.values, st_pos, STIM_COLORS['Stim'])
                jitter = rng.uniform(-0.12, 0.12, len(st_vals))
                ax.scatter(np.full(len(st_vals), st_pos) + jitter,
                           st_vals, c=STIM_COLORS['Stim'], s=50, alpha=0.7,
                           edgecolors='none', zorder=3)
                ax.plot([st_pos - 0.2, st_pos + 0.2],
                        [st_vals.mean(), st_vals.mean()],
                        color=STIM_COLORS['Stim'], linewidth=3, zorder=4)

            # Connect paired subjects
            shared = rsub.dropna(subset=[ns_col, st_col])
            for _, row in shared.iterrows():
                ax.plot([idx - bar_width / 2, idx + bar_width / 2],
                        [row[ns_col], row[st_col]],
                        color='gray', alpha=0.25, linewidth=0.7, zorder=2)

        ax.axhline(0, color='#4D4D4D', linewidth=1, linestyle='--')
        ax.set_ylabel(_axis_label(measure_label), fontsize=13, fontweight='bold')
        ax.set_title(f'{row_title} Trials', fontsize=15, fontweight='bold')
        ax.tick_params(axis='y', labelsize=11)

    ax_forg.set_xticks(range(len(regions)))
    ax_forg.set_xticklabels(regions, fontsize=11, fontweight='bold', rotation=45, ha='right')

    region_tag = f'{fname_extra} -- ' if fname_extra else ''
    fig.suptitle(
        f'{label} {phase_label} {measure_label} -- {region_tag}{band}\n'
        f'Stimulation Effect on Memory  |  N = {n_subj}',
        fontsize=16, fontweight='bold', y=1.02,
    )

    legend_handles = [
        Line2D([0], [0], marker='o', linestyle='', markersize=10,
               markerfacecolor=STIM_COLORS['NoStim'], markeredgecolor='none', label='NoStim'),
        Line2D([0], [0], marker='o', linestyle='', markersize=10,
               markerfacecolor=STIM_COLORS['Stim'], markeredgecolor='none', label='Stim'),
    ]
    fig.legend(handles=legend_handles, loc='upper center',
               bbox_to_anchor=(0.5, 0.98), ncol=len(legend_handles),
               frameon=False, fontsize=12)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    safe_band = band.lower().replace(' ', '_')
    safe_extra = fname_extra.lower().replace(' ', '_') if fname_extra else ''
    extra_part = f'_{safe_extra}' if safe_extra else ''
    fname = f'StimEffect_Violin_{measure_label}_{phase_label.lower()}_{safe_band}{extra_part}.png'
    plt.savefig(os.path.join(out_dir, fname), dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_stim_violin(df, out_dir, band, measure_label, phase_label, label,
                     split_by_anchor=False):
    """Violin overlay for stim effect — optionally split by anchor."""
    sub = df[df['Band'] == band].copy()
    if sub.empty:
        return
    sub = _add_responder(sub)
    regions = sorted(sub['Region'].unique())

    if not split_by_anchor or '_' not in regions[0]:
        _plot_single_stim_violin(sub, regions, out_dir, measure_label,
                                  phase_label, label, band)
        return

    anchor_groups = {}
    for r in regions:
        anchor = get_anchor_region(r)
        anchor_groups.setdefault(anchor, []).append(r)

    for anchor in sorted(anchor_groups):
        group_regions = anchor_groups[anchor]
        group_sub = sub[sub['Region'].isin(group_regions)]
        if group_sub.empty:
            continue
        _plot_single_stim_violin(group_sub, group_regions, out_dir, measure_label,
                                  phase_label, label, band,
                                  fname_extra=f'{anchor}_connections')


# ---------------------------------------------------------------------------
# Generate all outputs for one measure/phase
# ---------------------------------------------------------------------------

def generate_stim_outputs(data, out_dir, label, measure_label, phase_label, band_ranges):
    ensure_dir(out_dir)
    extracted = extract_stim_and_nostim(data, measure=measure_label.lower())
    scores = compute_stim_composite(extracted, band_ranges)
    if scores.empty:
        print(f"  No stim composite scores for {label} {phase_label} {measure_label} -- skipping.")
        return

    n_subj = scores['Patient'].nunique()
    n_regions = scores['Region'].nunique()
    print(f"  {label} {phase_label} {measure_label}: "
          f"{n_subj} subjects, {n_regions} regions, {len(scores)} rows")

    csv_path = os.path.join(out_dir, f'stim_composite_{measure_label.lower()}_{phase_label.lower()}.csv')
    scores.to_csv(csv_path, index=False)

    split_anchor = measure_label in ('Coherence', 'PAC')

    ranked_dir = ensure_dir(os.path.join(out_dir, 'ranked_bars'))
    for region in sorted(scores['Region'].unique()):
        for band in band_ranges:
            plot_ranked_stim_effect(scores, ranked_dir, region, band, measure_label,
                                   phase_label, label)

    violin_dir = ensure_dir(os.path.join(out_dir, 'violins'))
    for band in band_ranges:
        plot_stim_violin(scores, violin_dir, band, measure_label, phase_label,
                         label, split_by_anchor=split_anchor)


# ---------------------------------------------------------------------------
# Analysis runners
# ---------------------------------------------------------------------------

def run_power():
    print("\n" + "=" * 60)
    print("Stim Effect Composite Scores -- POWER")
    print("=" * 60)
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
    for gl, data in [('BLAES', blaes_enc), ('AMME', amme_enc), ('All', all_enc)]:
        out = ensure_dir(os.path.join(OUTPUT_ROOT, 'encoding_power', gl.lower()))
        generate_stim_outputs(data, out, gl, 'Power', 'Encoding', POWER_BANDS)

    print("\n  Loading retrieval power...")
    blaes_ret = filter_pnas_regions(load_blaes_retrieval())
    amme_ret = filter_pnas_regions(load_amme_retrieval())
    all_ret = filter_pnas_regions(build_all_retrieval_data(blaes_ret, amme_ret))
    for gl, data in [('BLAES', blaes_ret), ('AMME', amme_ret), ('All', all_ret)]:
        out = ensure_dir(os.path.join(OUTPUT_ROOT, 'retrieval_power', gl.lower()))
        generate_stim_outputs(data, out, gl, 'Power', 'Retrieval', POWER_BANDS)


def run_coherence():
    print("\n" + "=" * 60)
    print("Stim Effect Composite Scores -- COHERENCE")
    print("=" * 60)
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
    for gl, data in [('BLAES', blaes_enc), ('AMME', amme_enc), ('All', all_enc)]:
        out = ensure_dir(os.path.join(OUTPUT_ROOT, 'encoding_coherence', gl.lower()))
        generate_stim_outputs(data, out, gl, 'Coherence', 'Encoding', POWER_BANDS)

    print("\n  Loading retrieval coherence...")
    blaes_ret = filter_pnas_regions(lb_ret())
    amme_ret = filter_pnas_regions(la_ret())
    all_ret = filter_pnas_regions(_merge_coherence_data(blaes_ret, amme_ret))
    for gl, data in [('BLAES', blaes_ret), ('AMME', amme_ret), ('All', all_ret)]:
        out = ensure_dir(os.path.join(OUTPUT_ROOT, 'retrieval_coherence', gl.lower()))
        generate_stim_outputs(data, out, gl, 'Coherence', 'Retrieval', POWER_BANDS)


def run_pac():
    print("\n" + "=" * 60)
    print("Stim Effect Composite Scores -- PAC")
    print("=" * 60)
    from combined_encoding_pac import load_grouped_encoding_pac_data
    from combined_retrieval_pac import load_grouped_retrieval_pac_data

    print("\n  Loading encoding PAC...")
    grouped_enc = load_grouped_encoding_pac_data()
    for gk, gl in [('blaes', 'BLAES'), ('amme', 'AMME'), ('all', 'All')]:
        data = filter_pnas_regions(grouped_enc[gk])
        out = ensure_dir(os.path.join(OUTPUT_ROOT, 'encoding_pac', gk))
        generate_stim_outputs(data, out, gl, 'PAC', 'Encoding', PAC_BANDS)

    print("\n  Loading retrieval PAC...")
    grouped_ret = load_grouped_retrieval_pac_data()
    for gk, gl in [('blaes', 'BLAES'), ('amme', 'AMME'), ('all', 'All')]:
        data = filter_pnas_regions(grouped_ret[gk])
        out = ensure_dir(os.path.join(OUTPUT_ROOT, 'retrieval_pac', gk))
        generate_stim_outputs(data, out, gl, 'PAC', 'Retrieval', PAC_BANDS)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Stimulation Effect on Composite Scores")
    print("NoStim vs Stim within Remembered & Forgotten")
    print("=" * 60)
    ensure_dir(OUTPUT_ROOT)
    run_power()
    run_coherence()
    run_pac()
    print("\n" + "=" * 60)
    print(f"Done! Outputs: {OUTPUT_ROOT}")
    print("=" * 60)


if __name__ == '__main__':
    main()
