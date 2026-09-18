#!/usr/bin/env python
"""
IED Multi-Window Timing vs Memory — Encoding & Retrieval Phases
================================================================
Test whether IEDs spanning multiple timing windows on the same trial
have a greater impact on memory than single-window IEDs.

Encoding windows: Before Image, During Image, After Image, During Stim (0-4)
Retrieval windows: Before Image, During Image (0-2)

Outputs in outputs/ied_timing_memory/:
  Encoding:
    - encoding_multiwindow_dose_response.png
    - encoding_multiwindow_combinations.png
    - encoding_multiwindow_by_subject.png
  Retrieval:
    - retrieval_multiwindow_dose_response.png
    - retrieval_multiwindow_combinations.png
    - retrieval_multiwindow_by_subject.png
  Stats:
    - encoding_multiwindow_stats.csv
    - retrieval_multiwindow_stats.csv
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
from scipy.stats import chi2_contingency, spearmanr, fisher_exact

warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
IED_DIR = os.path.join(REPO_ROOT, 'IED')
ENCODING_CSV = os.path.join(IED_DIR,
                            'AMMEBLAES_IEDs_trial_level_dissertation_study_usethis_cleaned_with_memory.csv')
RETRIEVAL_CSV = os.path.join(IED_DIR,
                             'AMMEBLAES_IEDs_trial_level_dissertation_test_usethis_cleaned_with_memory.csv')
OUTPUT_DIR = os.path.join(IED_DIR, 'ied_timing_memory')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Colorblind-friendly palette from seaborn
# ---------------------------------------------------------------------------
CB_PALETTE = sns.color_palette('colorblind')
# Colors for dose-response bars: 1, 2, 3, 4 windows
DOSE_COLORS = [CB_PALETTE[0], CB_PALETTE[1], CB_PALETTE[3], CB_PALETTE[4]]
SIG_COLOR = '#C44E52'  # red used elsewhere for significant p-values

sns.set_style('ticks')
sns.set_context('talk', font_scale=0.9)

# Encoding timing
ENC_TIMING_COLS = ['BeforeImgITI', 'DuringImg', 'AfterImgITI', 'DuringStim']
ENC_TIMING_SHORT = {
    'BeforeImgITI': 'Before',
    'DuringImg': 'During',
    'AfterImgITI': 'After',
    'DuringStim': 'Stim',
}

# Retrieval timing
RET_TIMING_COLS = ['BeforeImgITI', 'DuringImgITI']
RET_TIMING_SHORT = {
    'BeforeImgITI': 'Before',
    'DuringImgITI': 'During',
}


# ========================== DATA LOADING ==========================

def _load_and_collapse(csv_path, timing_cols, timing_short, group_cols):
    df = pd.read_csv(csv_path)
    if 'MemoryOutcome' not in df.columns:
        return pd.DataFrame()
    dm = df[df['MemoryOutcome'].isin(['remembered', 'forgotten'])].copy()
    if len(dm) == 0:
        return pd.DataFrame()

    agg_dict = {c: (lambda x: 'Y' if (x == 'Y').any() else 'N') for c in timing_cols}
    trial_level = dm.groupby(group_cols).agg(agg_dict).reset_index()

    trial_level['n_windows'] = (trial_level[timing_cols] == 'Y').sum(axis=1)

    def combo_label(row):
        parts = [timing_short[c] for c in timing_cols if row[c] == 'Y']
        return ' + '.join(parts) if parts else 'None'
    trial_level['combo'] = trial_level.apply(combo_label, axis=1)

    return trial_level


def load_encoding_trials():
    return _load_and_collapse(ENCODING_CSV, ENC_TIMING_COLS, ENC_TIMING_SHORT,
                              ['Patient', 'Trial', 'MemoryOutcome', 'StimCond'])


def load_retrieval_trials():
    return _load_and_collapse(RETRIEVAL_CSV, RET_TIMING_COLS, RET_TIMING_SHORT,
                              ['Patient', 'Trial', 'MemoryOutcome'])


# ========================== PLOTS ==========================

def _sig_star(p):
    """Return APA significance asterisks; '' when n.s."""
    if p is None or np.isnan(p):
        return ''
    if p < 0.001:
        return '***'
    if p < 0.01:
        return '**'
    if p < 0.05:
        return '*'
    return ''


def bh_fdr(pvals):
    """Benjamini-Hochberg adjusted q-values for a vector of p-values.

    Returns q-values aligned to the input; NaNs pass through (e.g. the
    reference bar, which is not tested). Matches the implementation in
    run_all_analyses.py (_bh_fdr).
    """
    p = np.asarray(pvals, dtype=float)
    n = int(np.sum(~np.isnan(p)))
    if n == 0:
        return np.full_like(p, np.nan)
    order = np.argsort(np.where(np.isnan(p), 1.0, p))
    ranks = np.empty(len(p), dtype=int)
    ranks[order] = np.arange(1, len(p) + 1)
    q_raw = p * n / ranks
    q = np.full_like(p, np.nan)
    sorted_q = q_raw[order]
    for i in range(len(sorted_q) - 2, -1, -1):
        if not np.isnan(sorted_q[i]) and not np.isnan(sorted_q[i + 1]):
            sorted_q[i] = min(sorted_q[i], sorted_q[i + 1])
    sorted_q = np.minimum(sorted_q, 1.0)
    q[order] = sorted_q
    return q


def _fisher_vs_ref(rem_ref, forg_ref, rem_x, forg_x):
    """Two-sided Fisher's exact test on the 2x2 table
        [[rem_ref, forg_ref],
         [rem_x,   forg_x]].
    Returns p (NaN if any group is empty)."""
    if (rem_ref + forg_ref) == 0 or (rem_x + forg_x) == 0:
        return np.nan
    _, p = fisher_exact([[rem_ref, forg_ref], [rem_x, forg_x]],
                        alternative='two-sided')
    return p


def glmm_dose_q(coefs_csv, phase):
    """Return {n_windows: q} for the k-vs-1-window contrasts from the GLMM
    (mixed model with patient random intercept) coefficient CSV, BH-FDR
    corrected within phase. Encoding uses the factor model
    (factor(n_windows)2/3/4); retrieval uses the continuous n_windows slope as
    the single 1-vs-2-window contrast. Returns {} if the CSV is unavailable."""
    if not os.path.exists(coefs_csv):
        print(f'  WARNING: GLMM coefs not found ({coefs_csv}); no stars drawn')
        return {}
    df = pd.read_csv(coefs_csv)
    term = df['term'].astype(str)
    if phase == 'encoding':
        sub = df[term.str.startswith('factor(n_windows)')].copy()
        if sub.empty:
            return {}
        sub['nw'] = sub['term'].astype(str).str.extract(r'(\d+)').astype(int)
        sub = sub.sort_values('nw')
        qs = bh_fdr(sub['p.value'].values)
        return {int(nw): float(q) for nw, q in zip(sub['nw'], qs)}
    row = df[term == 'n_windows']
    return {2: float(row.iloc[0]['p.value'])} if not row.empty else {}


def plot_dose_response(trial_level, qmap, phase_label, filename):
    """% remembered as a function of number of timing windows with IED.
    Significance stars come from the GLMM k-vs-1-window contrasts (`qmap`)."""
    fig, ax = plt.subplots(figsize=(6, 7))

    # Restrict to trials with >= 1 timing window so the reference bar is
    # "1 window" (drop the degenerate 0-window group, e.g. retrieval n=1).
    window_counts = [nw for nw in sorted(trial_level['n_windows'].unique())
                     if nw >= 1]
    pcts, ns, rems, forgs = [], [], [], []

    for nw in window_counts:
        sub = trial_level[trial_level['n_windows'] == nw]
        n = len(sub)
        n_rem = int((sub['MemoryOutcome'] == 'remembered').sum())
        n_forg = n - n_rem
        pct = n_rem / n * 100 if n > 0 else 0
        pcts.append(pct)
        ns.append(n)
        rems.append(n_rem)
        forgs.append(n_forg)

    colors = DOSE_COLORS[:len(window_counts)]
    bars = ax.bar(window_counts, pcts, color=colors,
                  edgecolor='black', linewidth=1, width=0.6)

    # Significance vs. the 1-window reference comes from the GLMM contrasts
    # (patient random intercept), BH-FDR corrected within phase.
    for i, (bar, pct, n) in enumerate(zip(bars, pcts, ns)):
        nw = window_counts[i]
        if i == 0:
            sig_label = '(ref)'
        else:
            star = _sig_star(qmap.get(nw, np.nan))
            sig_label = star if star else 'n.s.'
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f'{pct:.1f}%\n(n={n})\n{sig_label}',
                ha='center', va='bottom', fontsize=11, fontweight='bold',
                color=(SIG_COLOR if (sig_label not in ('(ref)', 'n.s.', ''))
                       else 'black'))

    ax.set_xlabel('Number of Timing Windows with IED', fontsize=17,
                  fontweight='bold')
    ax.set_ylabel('% Trials Remembered', fontsize=17, fontweight='bold')
    ax.set_title(phase_label.replace(' Phase', ''),
                 fontsize=17, fontweight='bold')
    ax.set_xticks(window_counts)
    ax.set_ylim(0, 100)
    sns.despine(ax=ax)
    ax.tick_params(axis='both', labelsize=12)
    ax.text(0.0, -0.13,
            'Stars vs. 1-window reference: GLMM (patient random effect), '
            'Benjamini-Hochberg FDR q-values (corrected within this phase).',
            transform=ax.transAxes, fontsize=8, color='#555555')
    fig.tight_layout()

    out = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out}')


def plot_combinations(trial_level, phase_label, filename):
    """% remembered for each specific multi-window combination (min 5 trials)."""
    combo_stats = []
    for combo_name, sub in trial_level.groupby('combo'):
        n = len(sub)
        if n < 5:
            continue
        n_rem = (sub['MemoryOutcome'] == 'remembered').sum()
        n_windows = sub['n_windows'].iloc[0]
        combo_stats.append({
            'combo': combo_name,
            'n_windows': n_windows,
            'n': n,
            'n_rem': n_rem,
            'pct_rem': n_rem / n * 100,
        })

    if not combo_stats:
        print(f'  Skipping {filename}: no combos with >= 5 trials')
        return

    combo_df = pd.DataFrame(combo_stats).sort_values(['n_windows', 'pct_rem'],
                                                      ascending=[True, False])

    fig, ax = plt.subplots(figsize=(max(10, len(combo_df) * 1.5), 7))
    x = np.arange(len(combo_df))

    cmap = {i: DOSE_COLORS[min(i, len(DOSE_COLORS) - 1)]
            for i in combo_df['n_windows'].unique()}
    colors = [cmap[row['n_windows']] for _, row in combo_df.iterrows()]

    bars = ax.bar(x, combo_df['pct_rem'], color=colors,
                  edgecolor='black', linewidth=0.8)

    # Per-bar Fisher's exact vs. the leftmost (reference) combination.
    combo_df = combo_df.reset_index(drop=True)
    ref_rem  = int(combo_df.loc[0, 'n_rem'])
    ref_forg = int(combo_df.loc[0, 'n'] - combo_df.loc[0, 'n_rem'])
    p_per_bar = [np.nan]
    for i in range(1, len(combo_df)):
        rem_i  = int(combo_df.loc[i, 'n_rem'])
        forg_i = int(combo_df.loc[i, 'n'] - combo_df.loc[i, 'n_rem'])
        p_per_bar.append(_fisher_vs_ref(ref_rem, ref_forg, rem_i, forg_i))

    # Benjamini-Hochberg FDR across the non-reference combination comparisons
    # (this figure / phase is its own family).
    q_per_bar = bh_fdr(p_per_bar)

    for i, (bar, row) in enumerate(zip(bars, combo_df.itertuples())):
        if i == 0:
            sig_label = '(ref)'
        else:
            star = _sig_star(q_per_bar[i])
            sig_label = star if star else 'n.s.'
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f'{row.pct_rem:.0f}%\n(n={row.n})\n{sig_label}',
                ha='center', va='bottom', fontsize=9, fontweight='bold',
                color=(SIG_COLOR if (sig_label not in ('(ref)', 'n.s.', ''))
                       else 'black'))

    ax.set_xticks(x)
    ax.set_xticklabels(combo_df['combo'], rotation=35, ha='right', fontsize=11)
    ax.set_ylabel('% Trials Remembered', fontsize=17, fontweight='bold')
    ax.set_title(phase_label.replace(' Phase', ''),
                 fontsize=17, fontweight='bold')
    ax.set_ylim(0, 105)
    sns.despine(ax=ax)
    ax.tick_params(axis='y', labelsize=12)
    ax.text(0.0, -0.22,
            'Stars vs. reference bar: Fisher exact, Benjamini-Hochberg FDR '
            'q-values (corrected within this phase).',
            transform=ax.transAxes, fontsize=8, color='#555555')

    legend_elements = [Patch(facecolor=cmap[i], edgecolor='black',
                             label=f'{i} window{"s" if i > 1 else ""}')
                       for i in sorted(cmap.keys())]
    ax.legend(handles=legend_elements, prop={'weight': 'bold', 'size': 13},
              frameon=True, loc='upper right')

    fig.tight_layout()

    out = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out}')


def plot_window_count_combo(trial_level, qmap, phase_label, filename):
    """Combinations-style figure showing % remembered for trials with IEDs in
    1, 2, 3, ... and all timing windows. Reference = the 1-window group;
    significance stars come from the GLMM k-vs-1-window contrasts (`qmap`),
    BH-FDR corrected within this phase."""
    counts = [nw for nw in sorted(trial_level['n_windows'].unique()) if nw >= 1]
    if len(counts) < 2:
        print(f'  Skipping {filename}: <2 window-count groups')
        return

    def _grp(sub, nw, label):
        n = len(sub)
        n_rem = int((sub['MemoryOutcome'] == 'remembered').sum())
        return {'nw': nw, 'label': label, 'n': n, 'n_rem': n_rem,
                'n_forg': n - n_rem, 'pct': n_rem / n * 100 if n else 0}

    rows = [_grp(trial_level[trial_level['n_windows'] == nw], nw,
                 f"{nw} window{'s' if nw > 1 else ''}") for nw in counts]

    fig, ax = plt.subplots(figsize=(max(6, len(rows) * 1.5), 7))
    x = np.arange(len(rows))
    colors = [DOSE_COLORS[min(r['nw'] - 1, len(DOSE_COLORS) - 1)] for r in rows]
    bars = ax.bar(x, [r['pct'] for r in rows], color=colors,
                  edgecolor='black', linewidth=0.8, width=0.6)

    # Significance vs. the 1-window reference from the GLMM contrasts.
    for i, (bar, r) in enumerate(zip(bars, rows)):
        if i == 0:
            sig_label = '(ref)'
        else:
            star = _sig_star(qmap.get(r['nw'], np.nan))
            sig_label = star if star else 'n.s.'
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{r['pct']:.0f}%\n(n={r['n']})\n{sig_label}",
                ha='center', va='bottom', fontsize=10, fontweight='bold',
                color=(SIG_COLOR if sig_label not in ('(ref)', 'n.s.', '')
                       else 'black'))

    ax.set_xticks(x)
    ax.set_xticklabels([r['label'] for r in rows], fontsize=12)
    ax.set_ylabel('% Trials Remembered', fontsize=17, fontweight='bold')
    ax.set_title(phase_label.replace(' Phase', ''),
                 fontsize=17, fontweight='bold')
    ax.set_ylim(0, 105)
    sns.despine(ax=ax)
    ax.tick_params(axis='y', labelsize=12)
    ax.text(0.0, -0.13,
            'Stars vs. 1-window reference: GLMM (patient random effect), '
            'Benjamini-Hochberg FDR q-values (corrected within this phase).',
            transform=ax.transAxes, fontsize=8, color='#555555')

    fig.tight_layout()

    out = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out}')


def plot_by_subject(trial_level, phase_label, filename):
    """Per-subject % remembered by number of windows, with group mean."""
    max_nw = trial_level['n_windows'].max()
    window_counts = [nw for nw in range(1, max_nw + 1)
                     if (trial_level['n_windows'] == nw).sum() >= 5]
    if not window_counts:
        print(f'  Skipping {filename}: no window counts with enough data')
        return

    fig, ax = plt.subplots(figsize=(max(8, len(window_counts) * 3), 7))
    rng = np.random.default_rng(42)
    bar_width = 0.6

    for xi, nw in enumerate(window_counts):
        color = DOSE_COLORS[min(nw, len(DOSE_COLORS)) - 1]
        patient_pcts = []
        for pat in trial_level['Patient'].unique():
            sub = trial_level[(trial_level['Patient'] == pat) &
                              (trial_level['n_windows'] == nw)]
            if len(sub) >= 2:
                pct = (sub['MemoryOutcome'] == 'remembered').mean() * 100
                patient_pcts.append(pct)

        if not patient_pcts:
            continue

        mean_val = np.mean(patient_pcts)
        sem_val = (np.std(patient_pcts, ddof=1) / np.sqrt(len(patient_pcts))
                   if len(patient_pcts) > 1 else 0)

        ax.bar(xi, mean_val, yerr=sem_val, color=color, edgecolor='black',
               linewidth=1, width=bar_width, capsize=5, alpha=0.7, zorder=1)
        jitter = rng.uniform(-0.12, 0.12, len(patient_pcts))
        ax.scatter(np.full(len(patient_pcts), xi) + jitter, patient_pcts,
                   color='black', alpha=0.4, s=30, zorder=2)
        ax.text(xi, mean_val + sem_val + 3,
                f'{mean_val:.1f}%\n(n={len(patient_pcts)} pts)',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_xticks(range(len(window_counts)))
    ax.set_xticklabels([f'{nw} window{"s" if nw > 1 else ""}' for nw in window_counts],
                       fontsize=13)
    ax.set_ylabel('% Trials Remembered', fontsize=15)
    ax.set_title(f'{phase_label}: IED Multi-Window Effect by Subject\n'
                 f'(mean \u00b1 SEM, dots = patients)',
                 fontsize=16, fontweight='bold')
    ax.set_ylim(0, 115)
    sns.despine(ax=ax)
    ax.tick_params(axis='y', labelsize=12)
    fig.tight_layout()

    out = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out}')


# ========================== STATS ==========================

def save_stats(trial_level, filename):
    rows = []

    for nw in sorted(trial_level['n_windows'].unique()):
        sub = trial_level[trial_level['n_windows'] == nw]
        n = len(sub)
        n_rem = (sub['MemoryOutcome'] == 'remembered').sum()
        rows.append({
            'Category': f'{nw} window(s)',
            'N_Remembered': n_rem,
            'N_Forgotten': n - n_rem,
            'N_Total': n,
            'Pct_Remembered': round(n_rem / n * 100, 1) if n > 0 else np.nan,
        })

    for combo_name, sub in trial_level.groupby('combo'):
        n = len(sub)
        if n < 5:
            continue
        n_rem = (sub['MemoryOutcome'] == 'remembered').sum()
        rows.append({
            'Category': f'Combo: {combo_name}',
            'N_Remembered': n_rem,
            'N_Forgotten': n - n_rem,
            'N_Total': n,
            'Pct_Remembered': round(n_rem / n * 100, 1) if n > 0 else np.nan,
        })

    trial_level['rem_bin'] = (trial_level['MemoryOutcome'] == 'remembered').astype(int)
    rho, p_sp = spearmanr(trial_level['n_windows'], trial_level['rem_bin'])
    rows.append({'Category': f'Spearman rho={rho:.3f}, p={p_sp:.4f}',
                 'N_Total': len(trial_level)})

    ct = pd.crosstab(trial_level['n_windows'], trial_level['MemoryOutcome'])
    chi2, p_chi, _, _ = chi2_contingency(ct)
    rows.append({'Category': f'Chi2={chi2:.2f}, p={p_chi:.4f}',
                 'N_Total': len(trial_level)})

    stats_df = pd.DataFrame(rows)
    out = os.path.join(OUTPUT_DIR, filename)
    stats_df.to_csv(out, index=False)
    print(f'Saved {out}')
    print(stats_df.to_string(index=False))


# ========================== MAIN ==========================

def main():
    # --- Encoding Phase ---
    print('=' * 60)
    print('ENCODING PHASE')
    print('=' * 60)
    enc = load_encoding_trials()
    print(f"Encoding: {len(enc)} trials, {enc['Patient'].nunique()} patients")
    print(f"  n_windows: {enc['n_windows'].value_counts().sort_index().to_dict()}")
    print()

    save_stats(enc, 'encoding_multiwindow_stats.csv')
    print()
    enc_q = glmm_dose_q(os.path.join(OUTPUT_DIR, 'encoding_multiwindow_mlm_coefs.csv'),
                        'encoding')
    print(f'  GLMM dose q-values (vs 1 window): {enc_q}')
    plot_dose_response(enc, enc_q, 'Encoding Phase', 'encoding_multiwindow_dose_response.png')
    plot_combinations(enc, 'Encoding Phase', 'encoding_multiwindow_combinations.png')
    plot_window_count_combo(enc, enc_q, 'Encoding Phase', 'encoding_multiwindow_window_count.png')
    plot_by_subject(enc, 'Encoding Phase', 'encoding_multiwindow_by_subject.png')

    # --- Retrieval Phase ---
    print()
    print('=' * 60)
    print('RETRIEVAL PHASE')
    print('=' * 60)
    ret = load_retrieval_trials()
    if len(ret) > 0:
        print(f"Retrieval: {len(ret)} trials, {ret['Patient'].nunique()} patients")
        print(f"  n_windows: {ret['n_windows'].value_counts().sort_index().to_dict()}")
        print()

        save_stats(ret, 'retrieval_multiwindow_stats.csv')
        print()
        ret_q = glmm_dose_q(os.path.join(OUTPUT_DIR, 'retrieval_multiwindow_mlm_coefs.csv'),
                            'retrieval')
        print(f'  GLMM dose q-values (vs 1 window): {ret_q}')
        plot_dose_response(ret, ret_q, 'Retrieval Phase', 'retrieval_multiwindow_dose_response.png')
        plot_combinations(ret, 'Retrieval Phase', 'retrieval_multiwindow_combinations.png')
        plot_window_count_combo(ret, ret_q, 'Retrieval Phase', 'retrieval_multiwindow_window_count.png')
        plot_by_subject(ret, 'Retrieval Phase', 'retrieval_multiwindow_by_subject.png')
    else:
        print('No retrieval data available for multiwindow analysis.')


if __name__ == '__main__':
    main()
