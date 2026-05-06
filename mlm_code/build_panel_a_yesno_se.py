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
import seaborn as sns
from scipy.stats import mannwhitneyu, pointbiserialr

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IED_ENC_CSV = os.path.join(SCRIPT_DIR, 'IED',
    'AMMEBLAES_IEDs_trial_level_dissertation_study_usethis_cleaned_with_memory.csv')
IED_RET_CSV = os.path.join(SCRIPT_DIR, 'IED',
    'AMMEBLAES_IEDs_trial_level_dissertation_test_usethis_cleaned_with_memory.csv')

OUT_DIR = os.path.join(SCRIPT_DIR, 'outputs', 'ied_Quon_paper')
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
    """Return feature-level stats using Yes/No (presence) coding."""
    rows = []

    # Continuous: IED rate
    r, p, n, se = pb_effect(dm['Rate(within trials)'], dm['memory_binary'])
    rows.append({'Feature': 'IED Rate\n(IEDs/trial)',
                 'effect': r, 'se': se, 'p': p, 'n': n})

    # Hemisphere R - L
    hem = dm[dm['Hemisphere'].isin(['L', 'R'])]
    left = hem[hem['Hemisphere'] == 'L']['memory_binary']
    right = hem[hem['Hemisphere'] == 'R']['memory_binary']
    diff, p, se = prop_diff_test(right, left)
    rows.append({'Feature': 'IED Hemisphere\nRight -\nLeft (ref)',
                 'effect': diff, 'se': se, 'p': p,
                 'n': int(len(right) + len(left))})

    # Channel spread (continuous)
    r, p, n, se = pb_effect(dm['ChannelSpread'], dm['memory_binary'])
    rows.append({'Feature': 'Channel\nSpread',
                 'effect': r, 'se': se, 'p': p, 'n': n})

    # Region spread (continuous)
    r, p, n, se = pb_effect(dm['RegionSpread'], dm['memory_binary'])
    rows.append({'Feature': 'Region\nSpread',
                 'effect': r, 'se': se, 'p': p, 'n': n})

    # Timing-window Yes vs No features
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
                         'se': np.nan, 'p': np.nan, 'n': 0})
            continue
        sub = dm[dm[col].isin(['Y', 'N'])]
        yes = sub[sub[col] == 'Y']['memory_binary']
        no = sub[sub[col] == 'N']['memory_binary']
        diff, p, se = prop_diff_test(yes, no)
        rows.append({'Feature': label, 'effect': diff, 'se': se,
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
    colors = [SIG_COLOR if (not pd.isna(pv) and pv < 0.05) else NS_COLOR
              for pv in df['p']]
    ax.bar(range(len(df)), df['effect'],
           color=colors, edgecolor='black', linewidth=0.6)
    ax.errorbar(range(len(df)), df['effect'],
                yerr=df['se'].astype(float),
                fmt='none', ecolor='black', capsize=4, linewidth=1.2)
    ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df['Feature'], fontsize=10)
    ax.set_ylabel('Effect Size (r or prop. difference) +/- 1 SE', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    max_abs = float((df['effect'].abs().max() or 1))
    for i, r in df.iterrows():
        s = star(r['p'])
        if not s:
            continue
        eff = r['effect']
        se_i = r['se'] if pd.notna(r['se']) else 0
        offset = max_abs * 0.04
        y_pos = eff + se_i + offset if eff >= 0 else eff - se_i - offset
        va = 'bottom' if eff >= 0 else 'top'
        ax.text(i, y_pos, s, ha='center', va=va,
                fontsize=14, fontweight='bold', color=SIG_COLOR)

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

    print('\nEncoding stats:')
    print(enc_df.to_string(index=False))
    print('\nRetrieval stats:')
    print(ret_df.to_string(index=False))

    enc_df.to_csv(ENC_CSV_OUT, index=False)
    ret_df.to_csv(RET_CSV_OUT, index=False)
    print(f'\nSaved {ENC_CSV_OUT}')
    print(f'Saved {RET_CSV_OUT}')

    draw_panel(enc_df,
               '(A) IED Features Associated with Memory Encoding',
               ENC_FIG)
    draw_panel(ret_df,
               '(A) Retrieval: IED Features Associated with Memory',
               RET_FIG)


if __name__ == '__main__':
    main()
