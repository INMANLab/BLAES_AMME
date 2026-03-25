#!/usr/bin/env python
"""Shared remembered/forgotten regression panel helpers."""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


MEMORY_ORDER = ['remembered', 'forgotten']
MEMORY_TITLES = {
    'remembered': 'Remembered',
    'forgotten': 'Forgotten',
}


def average_condition_region_dicts(*region_dicts):
    merged = {}
    for region_dict in region_dicts:
        for region, subject_dict in (region_dict or {}).items():
            for patient, values in subject_dict.items():
                merged.setdefault(region, {}).setdefault(str(patient), []).append(
                    np.asarray(values, dtype=np.float64)
                )
    averaged = {}
    for region, subject_dict in merged.items():
        for patient, vectors in subject_dict.items():
            if vectors:
                averaged.setdefault(region, {})[patient] = np.nanmean(np.vstack(vectors), axis=0)
    return averaged


def build_memory_band_table(memory_to_region_dict, freqs, band_specs, value_col):
    rows = []
    freqs = np.asarray(freqs, dtype=np.float64)
    for memory_name in MEMORY_ORDER:
        for region, subject_dict in (memory_to_region_dict.get(memory_name, {}) or {}).items():
            for patient, spectrum in subject_dict.items():
                arr = np.asarray(spectrum, dtype=np.float64)
                if arr.size == 0:
                    continue
                for band_name, (lo, hi) in band_specs.items():
                    mask = (freqs >= lo) & (freqs <= hi)
                    if not np.any(mask):
                        continue
                    rows.append(
                        {
                            'Patient': str(patient),
                            'Region': region,
                            'Band': band_name,
                            'Memory': memory_name,
                            value_col: float(np.nanmean(arr[mask])),
                        }
                    )
    return pd.DataFrame(rows)


def fit_regression(df, value_col, target_col, undefined_label):
    n = int(len(df))
    if n < 3:
        return {
            'status': 'too_few_points',
            'n': n,
            'slope': np.nan,
            'intercept': np.nan,
            'r': np.nan,
            'r_squared': np.nan,
            'p_value': np.nan,
            'stderr': np.nan,
            'intercept_stderr': np.nan,
        }
    if df[value_col].nunique() < 2:
        return {
            'status': f'undefined_{undefined_label}',
            'n': n,
            'slope': np.nan,
            'intercept': np.nan,
            'r': np.nan,
            'r_squared': np.nan,
            'p_value': np.nan,
            'stderr': np.nan,
            'intercept_stderr': np.nan,
        }
    if df[target_col].nunique() < 2:
        return {
            'status': 'undefined_memory_modulation',
            'n': n,
            'slope': np.nan,
            'intercept': np.nan,
            'r': np.nan,
            'r_squared': np.nan,
            'p_value': np.nan,
            'stderr': np.nan,
            'intercept_stderr': np.nan,
        }
    result = stats.linregress(df[value_col], df[target_col])
    out = {
        'status': 'ok',
        'n': n,
        'slope': float(result.slope),
        'intercept': float(result.intercept),
        'r': float(result.rvalue),
        'r_squared': float(result.rvalue ** 2),
        'p_value': float(result.pvalue),
        'stderr': float(result.stderr),
        'intercept_stderr': float(result.intercept_stderr),
    }
    if not np.isfinite(out['slope']) or not np.isfinite(out['r']) or not np.isfinite(out['p_value']):
        out['status'] = 'undefined_fit'
    return out


def make_annotation(stats_row, undefined_label):
    if stats_row['status'] != 'ok':
        reason = {
            'too_few_points': 'fewer than 3 patients',
            f'undefined_{undefined_label}': f'{undefined_label.replace("_", " ")} has no variability',
            'undefined_memory_modulation': 'memory modulation has no variability',
            'undefined_fit': 'fit returned non-finite values',
        }.get(stats_row['status'], stats_row['status'])
        return '\n'.join([f"n = {stats_row['n']}", 'Regression undefined', reason])
    return '\n'.join(
        [
            f"n = {stats_row['n']}",
            f"slope = {stats_row['slope']:.4f}",
            f"intercept = {stats_row['intercept']:.4f}",
            f"r = {stats_row['r']:.3f}",
            f"R^2 = {stats_row['r_squared']:.3f}",
            f"p = {stats_row['p_value']:.4g}",
        ]
    )


def plot_memory_panel_regression(
    df,
    stats_by_memory,
    value_col,
    target_col,
    target_label,
    band_label,
    x_label,
    title_prefix,
    out_path: Path,
    logic_note,
    point_color='#1f4e79',
    line_color='#b22222',
    undefined_label='baseline_measure',
):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), sharey=True)
    region = df['Region'].iloc[0]
    for ax, memory_name in zip(axes, MEMORY_ORDER):
        memory_df = df[df['Memory'] == memory_name].copy()
        stats_row = stats_by_memory.get(memory_name)
        if memory_df.empty or stats_row is None:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=12)
            ax.set_axis_off()
            continue
        x = memory_df[value_col].to_numpy(dtype=float)
        y = memory_df[target_col].to_numpy(dtype=float)
        ax.scatter(x, y, s=55, alpha=0.8, color=point_color, edgecolors='white', linewidths=0.6)
        if stats_row['status'] == 'ok':
            order = np.argsort(x)
            x_sorted = x[order]
            y_fit = stats_row['intercept'] + stats_row['slope'] * x_sorted
            ax.plot(x_sorted, y_fit, color=line_color, linewidth=2)
        ax.set_title(MEMORY_TITLES[memory_name], fontsize=14, fontweight='bold')
        ax.set_xlabel(f'{x_label} ({band_label})', fontsize=12, fontweight='bold')
        ax.tick_params(axis='both', labelsize=10)
        ax.text(
            0.98,
            0.98,
            make_annotation(stats_row, undefined_label),
            transform=ax.transAxes,
            ha='right',
            va='top',
            fontsize=9,
            bbox=dict(boxstyle='round,pad=0.35', facecolor='white', alpha=0.9, edgecolor='#808080'),
        )
    axes[0].set_ylabel(f'Memory Modulation ({target_label})', fontsize=12, fontweight='bold')
    fig.suptitle(f'{title_prefix}: {region} - {band_label}', fontsize=15, fontweight='bold')
    fig.text(0.015, 0.015, logic_note, ha='left', va='bottom', fontsize=9, wrap=True)
    fig.tight_layout(rect=[0, 0.08, 1, 0.95])
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_memory_panel_histogram(df, value_col, band_label, x_label, title_prefix, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    region = df['Region'].iloc[0]
    for ax, memory_name in zip(axes, MEMORY_ORDER):
        memory_df = df[df['Memory'] == memory_name].copy()
        values = memory_df[value_col].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=12)
            ax.set_axis_off()
            continue
        bins = min(12, max(5, int(np.sqrt(len(values)))))
        ax.hist(values, bins=bins, color='#72b7b2', edgecolor='white', alpha=0.9)
        ax.set_title(MEMORY_TITLES[memory_name], fontsize=14, fontweight='bold')
        ax.set_xlabel(f'{x_label} ({band_label})', fontsize=12, fontweight='bold')
        ax.tick_params(axis='both', labelsize=10)
    axes[0].set_ylabel('Patient Count', fontsize=12, fontweight='bold')
    fig.suptitle(f'{title_prefix}: {region} - {band_label}', fontsize=15, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
