#!/usr/bin/env python
"""
Regenerate the original Quon Figure 2 Panel A (Yes/No feature definition)
for encoding and retrieval, adding +/- 1 SE error bars.

Yes/No timing-window features: e.g. Before Image = IED is in that window (Y)
vs. not (N). This is the pre-Option-C definition that produced the panel
with significant After-Image and During-Stim effects. The current
classical_v3 figures use the Option-C "Only vs. Other" definition which
removes window-overlap contamination, so the two panels disagree on
After-Image significance by design.

Outputs:
  outputs/ied_Quon_paper/figure2_ied_features_memory_no_white_nonmtl_bold_axes.png
  outputs/ied_Quon_paper/retrieval/figure2_retrieval_ied_features_memory_bold_axes.png
  outputs/ied_Quon_paper/figure2_yesno_stats_encoding.csv
  outputs/ied_Quon_paper/retrieval/figure2_yesno_stats_retrieval.csv
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, NullFormatter
import seaborn as sns
from scipy.stats import mannwhitneyu, pointbiserialr

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
IED_DIR = os.path.join(REPO_ROOT, 'IED')
IED_ENC_CSV = os.path.join(IED_DIR,
    'AMMEBLAES_IEDs_trial_level_dissertation_study_usethis_cleaned_with_memory.csv')
IED_RET_CSV = os.path.join(IED_DIR,
    'AMMEBLAES_IEDs_trial_level_dissertation_test_usethis_cleaned_with_memory.csv')

OUT_DIR = os.path.join(IED_DIR, 'ied_Quon_paper')
OUT_DIR_RET = os.path.join(OUT_DIR, 'retrieval')
os.makedirs(OUT_DIR_RET, exist_ok=True)

ENC_FIG = os.path.join(OUT_DIR,
    'figure2_ied_features_memory_no_white_nonmtl_bold_axes.png')
RET_FIG = os.path.join(OUT_DIR_RET,
    'figure2_retrieval_ied_features_memory_bold_axes.png')
ENC_CSV_OUT = os.path.join(OUT_DIR, 'figure2_yesno_stats_encoding.csv')
RET_CSV_OUT = os.path.join(OUT_DIR_RET, 'figure2_yesno_stats_retrieval.csv')

# Match the retrieval-phase outlier exclusion used by ied_quon_retrieval.py
RETRIEVAL_EXCLUDE = {'BJH042', 'UIC202306'}

SIG_COLOR = '#C44E52'
NS_COLOR = '#555555'


def bh_fdr(pvals):
    """Benjamini-Hochberg adjusted q-values for a vector of p-values.

    Returns q-values aligned to the input; NaNs pass through. Matches the
    implementation in run_all_analyses.py (_bh_fdr).
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


def pb_effect(feature, memory):
    mask = feature.notna() & memory.notna()
    n = int(mask.sum())
    if n < 10:
        return np.nan, np.nan, n, np.nan
    r, p = pointbiserialr(memory[mask], feature[mask])
    se = float(np.sqrt(max(0.0, 1.0 - r * r) / max(1, n - 2)))
    return r, p, n, se


def se_propdiff(y_grp, n_grp):
    n1, n2 = len(y_grp), len(n_grp)
    if n1 < 2 or n2 < 2:
        return np.nan
    p1, p2 = y_grp.mean(), n_grp.mean()
    return float(np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2))


def prop_diff_test(yes, no):
    diff = yes.mean() - no.mean()
    if len(yes) < 5 or len(no) < 5:
        return diff, np.nan, np.nan
    _, p = mannwhitneyu(yes, no, alternative='two-sided')
    se = se_propdiff(yes, no)
    return diff, p, se


def odds_ratio(feature, memory):
    """Odds ratio for remembering (memory==1) vs forgetting from a univariable
    logistic regression memory ~ feature, fit by IRLS. Returns (OR, ci_lo,
    ci_hi) using the Wald 95% CI. For binary 0/1 features this equals the 2x2
    sample OR; for continuous features pass a standardized (z-scored) feature
    so the OR is per +1 SD. OR > 1 => feature associated with remembering.
    """
    mask = feature.notna() & memory.notna()
    x = np.asarray(feature[mask], dtype=float)
    y = np.asarray(memory[mask], dtype=float)
    if len(np.unique(y)) < 2 or len(x) < 10:
        return np.nan, np.nan, np.nan
    X = np.column_stack([np.ones_like(x), x])
    b = np.zeros(2)
    for _ in range(100):
        eta = X @ b
        mu = 1.0 / (1.0 + np.exp(-eta))
        W = np.clip(mu * (1 - mu), 1e-9, None)
        z = eta + (y - mu) / W
        try:
            b_new = np.linalg.solve((X * W[:, None]).T @ X, (X * W[:, None]).T @ z)
        except np.linalg.LinAlgError:
            return np.nan, np.nan, np.nan
        if np.max(np.abs(b_new - b)) < 1e-8:
            b = b_new
            break
        b = b_new
    cov = np.linalg.inv((X * W[:, None]).T @ X)
    se = float(np.sqrt(np.diag(cov))[1])
    return (float(np.exp(b[1])),
            float(np.exp(b[1] - 1.96 * se)),
            float(np.exp(b[1] + 1.96 * se)))


def _zscore(s):
    sd = s.std()
    return (s - s.mean()) / sd if sd and np.isfinite(sd) else s * np.nan


def load_encoding():
    df = pd.read_csv(IED_ENC_CSV)
    dm = df[df['MemoryOutcome'].isin(['remembered', 'forgotten'])].copy()
    dm['memory_binary'] = (dm['MemoryOutcome'] == 'remembered').astype(int)
    return dm


def load_retrieval():
    df = pd.read_csv(IED_RET_CSV)
    dm = df[df['MemoryOutcome'].isin(['remembered', 'forgotten'])].copy()
    dm = dm[~dm['Patient'].isin(RETRIEVAL_EXCLUDE)].copy()
    dm['memory_binary'] = (dm['MemoryOutcome'] == 'remembered').astype(int)
    if 'DuringImgITI' in dm.columns:
        dm['DuringImg'] = dm['DuringImgITI']
    elif 'DuringImg' not in dm.columns:
        dm['DuringImg'] = 'N'
    return dm


def build_rows(dm, phase):
    """Return feature-level stats. Effect size = odds ratio for remembering vs
    forgetting (continuous features: OR per +1 SD; binary features: 2x2 OR),
    with Wald 95% CI. p-values are the same association tests as before
    (point-biserial for continuous, Mann-Whitney for the Yes/No features)."""
    rows = []

    # Continuous: IED rate (OR per +1 SD)
    _, p, n, _ = pb_effect(dm['Rate(within trials)'], dm['memory_binary'])
    orr, lo, hi = odds_ratio(_zscore(dm['Rate(within trials)']), dm['memory_binary'])
    rows.append({'Feature': 'IED Rate\n(IEDs/trial)',
                 'effect': orr, 'lo': lo, 'hi': hi, 'p': p, 'n': n})

    # Hemisphere R - L (R coded 1, L coded 0)
    hem = dm[dm['Hemisphere'].isin(['L', 'R'])]
    left = hem[hem['Hemisphere'] == 'L']['memory_binary']
    right = hem[hem['Hemisphere'] == 'R']['memory_binary']
    _, p, _ = prop_diff_test(right, left)
    orr, lo, hi = odds_ratio((hem['Hemisphere'] == 'R').astype(int),
                             hem['memory_binary'])
    rows.append({'Feature': 'IED Hemisphere\nRight -\nLeft (ref)',
                 'effect': orr, 'lo': lo, 'hi': hi, 'p': p,
                 'n': int(len(right) + len(left))})

    # Channel spread (continuous, OR per +1 SD)
    _, p, n, _ = pb_effect(dm['ChannelSpread'], dm['memory_binary'])
    orr, lo, hi = odds_ratio(_zscore(dm['ChannelSpread']), dm['memory_binary'])
    rows.append({'Feature': 'Channel\nSpread',
                 'effect': orr, 'lo': lo, 'hi': hi, 'p': p, 'n': n})

    # Region spread (continuous, OR per +1 SD)
    _, p, n, _ = pb_effect(dm['RegionSpread'], dm['memory_binary'])
    orr, lo, hi = odds_ratio(_zscore(dm['RegionSpread']), dm['memory_binary'])
    rows.append({'Feature': 'Region\nSpread',
                 'effect': orr, 'lo': lo, 'hi': hi, 'p': p, 'n': n})

    # Timing-window Yes vs No features (Y coded 1, N coded 0)
    if phase == 'encoding':
        window_cols = [
            ('BeforeImgITI', 'Before Image\nYes -\nNo (ref)'),
            ('DuringImg',    'During Image\nYes -\nNo (ref)'),
            ('AfterImgITI',  'After Image\nYes -\nNo (ref)'),
            ('DuringStim',   'During Stim\nYes -\nNo (ref)'),
        ]
    else:  # retrieval: only Before + During (Image/ITI)
        window_cols = [
            ('BeforeImgITI', 'Before Image\nYes -\nNo (ref)'),
            ('DuringImg',    'During Image/ITI\nYes -\nNo (ref)'),
        ]

    for col, label in window_cols:
        if col not in dm.columns:
            rows.append({'Feature': label, 'effect': np.nan,
                         'lo': np.nan, 'hi': np.nan, 'p': np.nan, 'n': 0})
            continue
        sub = dm[dm[col].isin(['Y', 'N'])]
        yes = sub[sub[col] == 'Y']['memory_binary']
        no = sub[sub[col] == 'N']['memory_binary']
        _, p, _ = prop_diff_test(yes, no)
        orr, lo, hi = odds_ratio((sub[col] == 'Y').astype(int),
                                 sub['memory_binary'])
        rows.append({'Feature': label, 'effect': orr, 'lo': lo, 'hi': hi,
                     'p': p, 'n': int(len(yes) + len(no))})

    return pd.DataFrame(rows)


def star(p):
    if pd.isna(p): return ''
    if p < 0.001:  return '***'
    if p < 0.01:   return '**'
    if p < 0.05:   return '*'
    return ''


def draw_panel(df, title, out_path):
    sns.set_style('ticks')
    sns.set_context('talk', font_scale=0.85)

    fig, ax = plt.subplots(figsize=(max(10, 1.4 * len(df)), 6))
    x = np.arange(len(df))
    colors = [SIG_COLOR if (not pd.isna(qv) and qv < 0.05) else NS_COLOR
              for qv in df['q']]

    eff = df['effect'].astype(float).values
    lo_err = eff - df['lo'].astype(float).values
    hi_err = df['hi'].astype(float).values - eff
    ax.errorbar(x, eff, yerr=[lo_err, hi_err], fmt='none',
                ecolor='black', capsize=5, linewidth=1.5, zorder=2)
    ax.scatter(x, eff, c=colors, s=110, edgecolor='black', linewidth=1.0,
               zorder=3)
    ax.axhline(1.0, color='black', linestyle='--', linewidth=0.9)
    ax.set_yscale('log')

    ax.set_xticks(x)
    ax.set_xticklabels(df['Feature'], fontsize=10)
    ax.set_xlim(-0.6, len(df) - 0.4)
    ax.set_ylabel('Odds Ratio (remembered vs. forgotten)\n[95% CI, log scale]',
                  fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Readable, non-scientific log-scale ticks bounded to the data range
    lo_min = float(df['lo'].astype(float).min())
    hi_max = float(df['hi'].astype(float).max())
    ymin, ymax = min(0.5, lo_min * 0.85), max(2.0, hi_max * 1.25)
    ax.set_ylim(ymin, ymax)
    ticks = [t for t in (0.25, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0)
             if ymin <= t <= ymax]
    ax.set_yticks(ticks)
    sf = ScalarFormatter()
    sf.set_scientific(False)
    ax.yaxis.set_major_formatter(sf)
    ax.yaxis.set_minor_formatter(NullFormatter())

    for i, r in df.iterrows():
        s = star(r['q'])
        if not s or pd.isna(r['hi']):
            continue
        ax.text(i, r['hi'] * 1.06, s, ha='center', va='bottom',
                fontsize=14, fontweight='bold', color=SIG_COLOR)

    ax.text(0.0, -0.18,
            'Points colored / starred when Benjamini-Hochberg FDR q < 0.05 '
            '(corrected across features in this panel). OR > 1 = more '
            'remembering; dashed line = no effect.',
            transform=ax.transAxes, fontsize=8, color='#555555')

    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Saved {out_path}')


def main():
    enc = load_encoding()
    ret = load_retrieval()

    enc_df = build_rows(enc, 'encoding')
    ret_df = build_rows(ret, 'retrieval')

    # Benjamini-Hochberg FDR correction across the features within each panel
    enc_df['q'] = bh_fdr(enc_df['p'].values)
    ret_df['q'] = bh_fdr(ret_df['p'].values)

    print('\nEncoding stats:')
    print(enc_df.to_string(index=False))
    print('\nRetrieval stats:')
    print(ret_df.to_string(index=False))

    enc_df.to_csv(ENC_CSV_OUT, index=False)
    ret_df.to_csv(RET_CSV_OUT, index=False)
    print(f'\nSaved {ENC_CSV_OUT}')
    print(f'Saved {RET_CSV_OUT}')

    draw_panel(enc_df,
               'Encoding',
               ENC_FIG)
    draw_panel(ret_df,
               'Retrieval',
               RET_FIG)


if __name__ == '__main__':
    main()
