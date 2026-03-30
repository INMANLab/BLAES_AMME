#!/usr/bin/env python
"""Generate per-hub-region PAC box plots for encoding and retrieval.

For each source region (e.g. BLA), creates a plot showing only that
region's connections (BLA_CA, BLA_DG, BLA_HPC, etc.) with stim vs
no-stim baseline-corrected PAC side by side. Produces cleaner, more
readable plots than the all-regions-at-once version.
"""

import csv
import os
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_BASE = SCRIPT_DIR / 'outputs'

# Base regions to use as hubs (exclude composites like ALLHPC, MTL)
BASE_REGIONS = ('BLA', 'CA', 'DG', 'EC', 'HPC', 'PRC', 'PHG')
COMPOSITE_TAGS = ('ALLHPC', 'MTL')

PAC_BAND = 'Slow gamma'


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_bc_summary(csv_path):
    """Load baseline-corrected band summary CSV into a list of dicts."""
    rows = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def is_base_pair(region):
    """Return True if the region pair uses only base regions (no composites)."""
    parts = region.split('_')
    if len(parts) != 2:
        return False
    return all(p not in COMPOSITE_TAGS for p in parts)


def build_hub_data(rows, band=PAC_BAND):
    """Organize data by hub (source) region.

    Returns dict: hub -> target -> condition -> [values]
    where condition is 'stim_bc' or 'nostim_bc'.
    """
    hub_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for row in rows:
        region = row['Region']
        condition = row['condition']
        row_band = row['band']

        if row_band != band:
            continue
        if condition not in ('stim_bc', 'nostim_bc'):
            continue
        if not is_base_pair(region):
            continue

        source, target = region.split('_')
        value = float(row['mean_value'])
        hub_data[source][target][condition].append(value)

    return hub_data


def plot_hub_region(hub, target_data, phase_label, out_dir):
    """Create a stim vs no-stim box plot for one hub region's connections."""
    # Sort targets alphabetically
    targets = sorted(target_data.keys())
    if not targets:
        return

    roi_labels = [f'{hub}_{t}' for t in targets]
    n_rois = len(targets)

    fig, ax = plt.subplots(figsize=(max(6, n_rois * 1.5 + 2), 7))

    x_base = np.arange(n_rois)
    bar_width = 0.34
    offsets = {'nostim_bc': -bar_width / 2, 'stim_bc': bar_width / 2}
    colors = {'nostim_bc': '#1f77b4', 'stim_bc': '#d62728'}
    cond_labels = {'nostim_bc': 'No Stim', 'stim_bc': 'Stim'}

    for i, target in enumerate(targets):
        for cond in ['nostim_bc', 'stim_bc']:
            vals = np.array(target_data[target].get(cond, []))
            if len(vals) == 0:
                continue
            mean = np.nanmean(vals)
            sem = np.nanstd(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0

            ax.bar(
                x_base[i] + offsets[cond],
                mean,
                width=bar_width,
                color=colors[cond],
                edgecolor='black',
                linewidth=1.0,
                yerr=sem,
                capsize=4,
                zorder=1,
            )

        # Draw individual patient connected dots
        stim_vals = target_data[target].get('stim_bc', [])
        nostim_vals = target_data[target].get('nostim_bc', [])
        n_paired = min(len(nostim_vals), len(stim_vals))
        for j in range(n_paired):
            x_ns = x_base[i] + offsets['nostim_bc']
            x_s = x_base[i] + offsets['stim_bc']
            ax.plot([x_ns, x_s], [nostim_vals[j], stim_vals[j]],
                    color='black', alpha=0.2, linewidth=0.8, zorder=2)
            ax.scatter(x_ns, nostim_vals[j], color='black', alpha=0.5, s=25, zorder=3)
            ax.scatter(x_s, stim_vals[j], color='black', alpha=0.5, s=25, zorder=3)
        # Plot any unpaired points
        for j in range(n_paired, len(nostim_vals)):
            ax.scatter(x_base[i] + offsets['nostim_bc'], nostim_vals[j],
                       color='black', alpha=0.5, s=25, zorder=3)
        for j in range(n_paired, len(stim_vals)):
            ax.scatter(x_base[i] + offsets['stim_bc'], stim_vals[j],
                       color='black', alpha=0.5, s=25, zorder=3)

    ax.axhline(0, color='gray', linewidth=1.0)
    ax.set_xticks(x_base)
    ax.set_xticklabels(roi_labels, rotation=30, ha='right', fontsize=12)
    ax.set_ylabel('Mean Baseline-corrected PAC', fontsize=14, fontweight='bold')
    ax.set_title(f'{PAC_BAND}', fontsize=16, fontweight='bold')
    ax.tick_params(axis='y', labelsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Legend
    handles = [
        mpatches.Patch(facecolor=colors['nostim_bc'], edgecolor='black', label='No Stim'),
        mpatches.Patch(facecolor=colors['stim_bc'], edgecolor='black', label='Stim'),
    ]
    fig.legend(handles=handles, loc='upper center', ncol=2, fontsize=11,
               frameon=True, bbox_to_anchor=(0.5, 0.97))

    fig.suptitle(
        f'{hub} Connections — {phase_label} Baseline-corrected PAC (Stim vs No Stim)',
        fontsize=16, fontweight='bold', y=1.02,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out_path = out_dir / f'{hub}_connections_{phase_label.lower()}_pac_{PAC_BAND.lower().replace(" ", "_")}.png'
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out_path.name}')


def process_phase(phase_label, csv_path, out_dir):
    """Generate per-hub plots for one phase (encoding or retrieval)."""
    if not csv_path.exists():
        print(f'  Skipping {phase_label}: CSV not found at {csv_path}')
        return

    rows = load_bc_summary(csv_path)
    hub_data = build_hub_data(rows)

    print(f'\n{phase_label} PAC — generating per-hub-region plots')
    for hub in BASE_REGIONS:
        if hub not in hub_data:
            print(f'  {hub}: no data, skipping')
            continue
        target_data = hub_data[hub]
        n_targets = len(target_data)
        print(f'  {hub}: {n_targets} connections')
        plot_hub_region(hub, target_data, phase_label, out_dir)


def main():
    # Retrieval — save into retrieval_pac/all/
    retrieval_dir = ensure_dir(OUTPUT_BASE / 'retrieval_pac' / 'all' / 'hub_region_plots')
    retrieval_csv = OUTPUT_BASE / 'retrieval_pac' / 'all' / 'csvs' / 'baseline_corrected_band_summary_retrieval.csv'
    process_phase('Retrieval', retrieval_csv, retrieval_dir)

    # Encoding — save into encoding_pac/all/
    encoding_dir = ensure_dir(OUTPUT_BASE / 'encoding_pac' / 'all' / 'hub_region_plots')
    encoding_csv = OUTPUT_BASE / 'encoding_pac' / 'all' / 'csvs' / 'baseline_corrected_band_summary_encoding.csv'
    process_phase('Encoding', encoding_csv, encoding_dir)

    print(f'\nDone! Hub region PAC plots saved to:')
    print(f'  Retrieval: {retrieval_dir}')
    print(f'  Encoding:  {encoding_dir}')


if __name__ == '__main__':
    main()
