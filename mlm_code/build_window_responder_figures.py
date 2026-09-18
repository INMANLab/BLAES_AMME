#!/Users/martinahollearn/anaconda3/bin/python
"""
Per-timing-window IED forgetting / burden figures by responder group.
======================================================================
Generalizes the During-Stimulation responder figures (Figure 4.6, built by
build_during_stim_responder_report.py + recolor_during_stim_figures.py) to the
other timing windows, reusing the same computation and the Fig 4.6 responder
palette/styling.

For each focal timing window the script produces the same two panels as
Fig 4.6:

  * forgetting_boost_by_patient (Fig 4.6 Panel D)
      Per-patient percentage-point difference in forgetting rate between
      IED-positive trials whose IED fell in the focal window and the same
      patient's IED-positive trials whose IED fell only in other windows.
      Ordered by avg_stim_dprime_diff; bar color = responder group.

  * burden_by_responder_group (Fig 4.6 Panels B + C)
      Left  = pooled forgetting rate on focal-window-IED trials (solid) vs.
              all-other-window IED trials (hatched), by responder group.
      Right = mean per-patient count of focal-window-IED trials, by group.

Windows
  Encoding (stim trials only, mirrors Fig 4.6):
      Before Image (BeforeImgITI), During Image (DuringImg), After Image
      (AfterImgITI).  [During Stim already covered by the original scripts.]
  Retrieval (all old-item IED-positive trials; not stimulated):
      Before Image (BeforeImgITI), During Image (DuringImgITI).

Outputs -> IED/ied_timing_memory/
  encoding_<win>_forgetting_boost_by_patient.png
  encoding_<win>_burden_by_responder_group.png
  retrieval_<win>_forgetting_boost_by_patient.png
  retrieval_<win>_burden_by_responder_group.png
  plus per-patient / per-group summary CSVs alongside each window.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns

ROOT = ('/Users/martinahollearn/Library/CloudStorage/Box-Box/InmanLab/'
        'BLAES_data/dissertation/AMME_BLAES')
ENC_CSV = os.path.join(ROOT, 'IED',
    'AMMEBLAES_IEDs_trial_level_dissertation_study_usethis_cleaned_with_memory.csv')
RET_CSV = os.path.join(ROOT, 'IED',
    'AMMEBLAES_IEDs_trial_level_dissertation_test_usethis_cleaned_with_memory.csv')
RESP_CSV = os.path.join(ROOT, 'OUTPUTS', 'csvs', 'AMMEBLAES_responder_status.csv')
OUT_DIR = os.path.join(ROOT, 'IED', 'ied_timing_memory')

# Responder palette / order matching Figure 4.6 (from recolor_during_stim_figures.py).
SWARM_PALETTE = {
    'Strong responders':   '#5B1A78',
    'Moderate responders': '#B21D6B',
    'Non-responders':      '#F04646',
    'Anti-responders':     '#F79B62',
}
RESP_ORDER = ['Anti-responders', 'Non-responders',
              'Moderate responders', 'Strong responders']

# All timing-window flag columns present in each phase's CSV.
ENC_TIMING_COLS = ['BeforeImgITI', 'DuringImg', 'AfterImgITI', 'DuringStim']
RET_TIMING_COLS = ['BeforeImgITI', 'DuringImgITI']

# (focal_col, human label, file slug) for each window to build.
ENC_WINDOWS = [
    ('BeforeImgITI', 'Before Image', 'before_img'),
    ('DuringImg',    'During Image', 'during_img'),
    ('AfterImgITI',  'After Image',  'after_img'),
]
RET_WINDOWS = [
    ('BeforeImgITI', 'Before Image', 'before_img'),
    ('DuringImgITI', 'During Image', 'during_img'),
]


# -------------------------------------------------------------------------
#  Computation (generalized from compute_summaries() in
#  build_during_stim_responder_report.py)
# -------------------------------------------------------------------------
def compute_window_summaries(csv_path, timing_cols, focal_col, restrict_stim):
    """Per-patient and per-responder-group focal-window forgetting/burden.

    Returns (summary, grp). The focal window is contrasted against "IED fell
    only in some other timing window" on the same trial set. When
    restrict_stim is True (encoding), the trial set is the patient's
    stimulated (S) trials; otherwise (retrieval) it is all the patient's
    old-item IED-positive trials.
    """
    df = pd.read_csv(csv_path)
    df = df[df['MemoryOutcome'].isin(['remembered', 'forgotten'])].copy()

    has_stim = 'StimCond' in df.columns
    group_cols = ['Patient', 'Trial', 'MemoryOutcome']
    if has_stim:
        group_cols.append('StimCond')

    def any_yes(s):
        return 'Y' if (s == 'Y').any() else 'N'

    trial = df.groupby(group_cols)[timing_cols].agg(any_yes).reset_index()
    trial['forg'] = (trial['MemoryOutcome'] == 'forgotten').astype(int)

    resp = pd.read_csv(RESP_CSV).rename(
        columns={'Responder status': 'responder_status'})

    rows = []
    for p in sorted(trial['Patient'].unique()):
        sub = trial[trial['Patient'] == p]
        if restrict_stim and has_stim:
            sub = sub[sub['StimCond'] == 'S']
        if sub.empty:
            continue
        foc = sub[sub[focal_col] == 'Y']
        oth = sub[sub[focal_col] == 'N']
        n_f, n_o = len(foc), len(oth)
        if n_f == 0:
            continue  # patient contributes no focal-window IED trial
        forg_f = int((foc['forg'] == 1).sum())
        forg_o = int((oth['forg'] == 1).sum())
        pf_f = forg_f / n_f if n_f else np.nan
        pf_o = forg_o / n_o if n_o else np.nan
        rows.append({
            'Patient':          p,
            'n_focal_trials':   n_f,
            'n_other_trials':   n_o,
            'forg_focal':       forg_f,
            'forg_other':       forg_o,
            'focal_share':      (n_f / (n_f + n_o)) if (n_f + n_o) else np.nan,
            'pct_forg_focal':   100 * pf_f if n_f else np.nan,
            'pct_forg_other':   100 * pf_o if n_o else np.nan,
            'focal_forg_boost': (100 * (pf_f - pf_o)) if (n_f and n_o) else np.nan,
        })

    summary = (pd.DataFrame(rows)
               .merge(resp, on='Patient', how='left')
               .sort_values('avg_stim_dprime_diff')
               .reset_index(drop=True))

    grp_rows = []
    for g in RESP_ORDER:
        gdf = summary[summary['responder_status'] == g]
        if gdf.empty:
            continue
        n_f = int(gdf['n_focal_trials'].sum())
        n_o = int(gdf['n_other_trials'].sum())
        forg_f = int(gdf['forg_focal'].sum())
        forg_o = int(gdf['forg_other'].sum())
        grp_rows.append({
            'responder_status':    g,
            'n_pts':               len(gdf),
            'mean_dprime_diff':    gdf['avg_stim_dprime_diff'].mean(),
            'n_focal_trials':      n_f,
            'n_other_trials':      n_o,
            'pct_forg_focal':      (100 * forg_f / n_f) if n_f else np.nan,
            'pct_forg_other':      (100 * forg_o / n_o) if n_o else np.nan,
            'mean_n_focal_trials': gdf['n_focal_trials'].mean(),
        })
    grp = pd.DataFrame(grp_rows)
    grp['focal_forg_boost'] = grp['pct_forg_focal'] - grp['pct_forg_other']

    return summary, grp


# -------------------------------------------------------------------------
#  Figures (mirror recolor_during_stim_figures.py styling)
# -------------------------------------------------------------------------
def plot_boost_by_patient(summary, out_path, window_label, phase_label):
    """Fig 4.6 Panel D, generalized to an arbitrary focal window."""
    df = summary.dropna(subset=['focal_forg_boost']).copy()
    df = df.sort_values('avg_stim_dprime_diff').reset_index(drop=True)
    colors = [SWARM_PALETTE.get(g, '#888888') for g in df['responder_status']]

    sns.set_style('ticks')
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(range(len(df)), df['focal_forg_boost'],
           color=colors, edgecolor='black', linewidth=0.5)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xticks(range(len(df)))
    xlabels = [f"{p}\n({d:+.2f})" for p, d in
               zip(df['Patient'], df['avg_stim_dprime_diff'])]
    ax.set_xticklabels(xlabels, rotation=60, ha='right', fontsize=9)
    ax.set_ylabel(f'Forgetting boost of IED presence in the\n'
                  f'{window_label} time window (pp)',
                  fontsize=11, fontweight='bold')
    ax.set_title(f'{phase_label}: {window_label} time window',
                 fontsize=13, fontweight='bold')

    handles = [Patch(facecolor=SWARM_PALETTE[g], edgecolor='black',
                     linewidth=0.5, label=g)
               for g in RESP_ORDER if g in df['responder_status'].values]
    ax.legend(handles=handles, loc='upper right', fontsize=9, frameon=False)

    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Saved {out_path}')


def plot_burden_by_group(grp, out_path, window_label, restrict_stim):
    """Fig 4.6 Panels B (left) + C (right), generalized to a focal window."""
    sns.set_style('ticks')
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    g_order = [g for g in RESP_ORDER if g in grp['responder_status'].values]
    grp2 = grp.set_index('responder_status').reindex(g_order).reset_index()
    colors = [SWARM_PALETTE[g] for g in grp2['responder_status']]

    # Phase-specific framing of the forgetting axis / legend.
    if restrict_stim:
        forg_ylabel = ('subsequent forgetting rate (%) of trials\n'
                       'with post-encoding stimulation')
        legend_title = 'Stimulated only trials'
    else:
        forg_ylabel = ('subsequent forgetting rate (%)\n'
                       'of retrieval trials')
        legend_title = 'Retrieval trials'

    # Left: pooled %forg focal window vs. all other windows
    x = np.arange(len(grp2))
    w = 0.35
    axes[0].bar(x - w / 2, grp2['pct_forg_focal'], width=w,
                color=colors, edgecolor='black', linewidth=0.5)
    axes[0].bar(x + w / 2, grp2['pct_forg_other'], width=w,
                color=colors, edgecolor='black', linewidth=0.5,
                alpha=0.5, hatch='//')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([g.replace(' responders', '')
                             for g in grp2['responder_status']], fontsize=10)
    axes[0].set_ylabel(forg_ylabel, fontsize=11, fontweight='bold')
    legend_handles = [
        Patch(facecolor='#888888', edgecolor='black', linewidth=0.5,
              label=f'{window_label} time window'),
        Patch(facecolor='#888888', edgecolor='black', linewidth=0.5,
              alpha=0.5, hatch='//', label='all other time windows'),
    ]
    axes[0].legend(handles=legend_handles, title=legend_title,
                   fontsize=9, frameon=False, loc='upper right')

    # Right: focal-window IED burden (mean per-patient trial count)
    axes[1].bar(x, grp2['mean_n_focal_trials'],
                color=colors, edgecolor='black', linewidth=0.5)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([g.replace(' responders', '')
                             for g in grp2['responder_status']], fontsize=10)
    axes[1].set_ylabel(f'Mean # of IEDs in the {window_label}\n'
                       f'time window per patient',
                       fontsize=11, fontweight='bold')
    for i, n in enumerate(grp2['mean_n_focal_trials']):
        axes[1].text(i, n + 0.15, f'{n:.1f}', ha='center', va='bottom',
                     fontsize=9)

    sns.despine()
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Saved {out_path}')


# -------------------------------------------------------------------------
#  Driver
# -------------------------------------------------------------------------
def build_phase(phase, csv_path, timing_cols, windows, restrict_stim):
    print('=' * 64)
    print(f'{phase.upper()} PHASE  (restrict_stim={restrict_stim})')
    print('=' * 64)
    for focal_col, window_label, slug in windows:
        summary, grp = compute_window_summaries(
            csv_path, timing_cols, focal_col, restrict_stim)

        prefix = f'{phase}_{slug}'
        summary.to_csv(os.path.join(OUT_DIR, f'{prefix}_responder_per_patient.csv'),
                       index=False)
        grp.to_csv(os.path.join(OUT_DIR, f'{prefix}_responder_group_summary.csv'),
                   index=False)

        boost_path = os.path.join(OUT_DIR,
            f'{prefix}_forgetting_boost_by_patient.png')
        burden_path = os.path.join(OUT_DIR,
            f'{prefix}_burden_by_responder_group.png')
        print(f'\n{window_label} ({focal_col}): '
              f'{len(summary)} patients with a focal-window IED trial')
        plot_boost_by_patient(summary, boost_path, window_label, phase.capitalize())
        plot_burden_by_group(grp, burden_path, window_label, restrict_stim)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    build_phase('encoding', ENC_CSV, ENC_TIMING_COLS, ENC_WINDOWS,
                restrict_stim=True)
    build_phase('retrieval', RET_CSV, RET_TIMING_COLS, RET_WINDOWS,
                restrict_stim=False)


if __name__ == '__main__':
    main()
