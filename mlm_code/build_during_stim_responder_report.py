#!/usr/bin/env python
"""
Build PDF report: During-Stim IEDs and Responder Status.

Two driver questions for the encoding cohort (22 patients with at least
one During-Stim-window IED on a stim trial):

  A) Among IED-positive stim trials, are the trials whose IED falls in the
     During-Stim window forgotten more often than the patient's other
     IED-positive stim trials?  (Per-patient DS_forg_boost)

  B) Does cumulative During-Stim-IED burden push patients toward
     anti/non-responder status?  (Per-responder-group DS-IED counts and
     share, plus per-patient correlation with avg_stim_dprime_diff.)

Output:
  outputs/ied_timing_memory/During_Stim_IEDs_and_Responder_Status.pdf
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
try:
    from fpdf import FPDF
except ImportError:
    FPDF = None

ROOT = '/Users/martinahollearn/Library/CloudStorage/Box-Box/InmanLab/BLAES_data/dissertation/AMME_BLAES'
ENC_CSV = os.path.join(ROOT, 'IED',
    'AMMEBLAES_IEDs_trial_level_dissertation_study_usethis_cleaned_with_memory.csv')
RESP_CSV = os.path.join(ROOT, 'OUTPUTS', 'csvs', 'AMMEBLAES_responder_status.csv')

OUT_DIR = os.path.join(ROOT, 'IED', 'ied_timing_memory')
FIG_BOOST = os.path.join(OUT_DIR, 'during_stim_forgetting_boost_by_patient.png')
FIG_BURDEN = os.path.join(OUT_DIR, 'during_stim_burden_by_responder_group.png')
PDF_PATH = os.path.join(OUT_DIR, 'During_Stim_IEDs_and_Responder_Status.pdf')
# Inferential within-group contrasts (precomputed by
# compute_during_stim_within_group_contrasts.py under anaconda, because
# statsmodels and fpdf are not installed in the same environment).
CONTRASTS_CSV = os.path.join(OUT_DIR, 'during_stim_within_group_contrasts_fdr.csv')

RESP_ORDER = ['Anti-responders', 'Non-responders',
              'Moderate responders', 'Strong responders']
REGION_BUCKETS = ['Hippocampus only', 'Amygdala only', 'HPC+Amyg', 'Other only']
RESP_PALETTE = {
    'Anti-responders':     '#C44E52',
    'Non-responders':      '#DD8452',
    'Moderate responders': '#8172B3',
    'Strong responders':   '#55A868',
}

TIMING_COLS = ['BeforeImgITI', 'DuringImg', 'AfterImgITI', 'DuringStim']


# -------------------------------------------------------------------------
#  Compute the two summary tables
# -------------------------------------------------------------------------
def compute_summaries():
    ied = pd.read_csv(ENC_CSV)
    ied = ied[ied['MemoryOutcome'].isin(['remembered', 'forgotten'])].copy()

    def any_yes(s): return 'Y' if (s == 'Y').any() else 'N'
    trial = (ied.groupby(['Patient', 'Trial', 'MemoryOutcome', 'StimCond'])
                [TIMING_COLS].agg(any_yes).reset_index())
    trial['forg'] = (trial['MemoryOutcome'] == 'forgotten').astype(int)

    resp = pd.read_csv(RESP_CSV).rename(
        columns={'Responder status': 'responder_status'})

    all_pats = sorted(trial['Patient'].unique())
    stim_pats = [p for p in all_pats
                 if ((trial['Patient'] == p) &
                     (trial['DuringStim'] == 'Y')).sum() > 0]

    rows = []
    for p in stim_pats:
        sub = trial[(trial['Patient'] == p) & (trial['StimCond'] == 'S')]
        if sub.empty:
            continue
        ds = sub[sub['DuringStim'] == 'Y']
        no = sub[sub['DuringStim'] == 'N']
        n_ds, n_no = len(ds), len(no)
        forg_ds = int((ds['forg'] == 1).sum())
        forg_no = int((no['forg'] == 1).sum())

        pf_ds = forg_ds / n_ds if n_ds else np.nan
        pf_no = forg_no / n_no if n_no else np.nan
        rows.append({
            'Patient':         p,
            'n_S_IED_trials':  n_ds + n_no,
            'n_S_DS_trials':   n_ds,
            'n_S_noDS_trials': n_no,
            'forg_S_DS':       forg_ds,
            'forg_S_noDS':     forg_no,
            'DS_share':        (n_ds / (n_ds + n_no)) if (n_ds + n_no) else np.nan,
            'pct_forg_DS':     100 * pf_ds if n_ds else np.nan,
            'pct_forg_noDS':   100 * pf_no if n_no else np.nan,
            'DS_forg_boost':   (100 * (pf_ds - pf_no)) if (n_ds and n_no) else np.nan,
        })

    summary = (pd.DataFrame(rows)
               .merge(resp, on='Patient', how='left')
               .sort_values('avg_stim_dprime_diff')
               .reset_index(drop=True))

    # Aggregate by responder status (pool trial counts)
    grp_rows = []
    for g in RESP_ORDER:
        gdf = summary[summary['responder_status'] == g]
        if gdf.empty:
            continue
        n_ds = int(gdf['n_S_DS_trials'].sum())
        n_no = int(gdf['n_S_noDS_trials'].sum())
        forg_ds = int(gdf['forg_S_DS'].sum())
        forg_no = int(gdf['forg_S_noDS'].sum())
        grp_rows.append({
            'responder_status':   g,
            'n_pts':              len(gdf),
            'mean_dprime_diff':   gdf['avg_stim_dprime_diff'].mean(),
            'n_S_DS_trials':      n_ds,
            'n_S_noDS_trials':    n_no,
            'pct_forg_DS':        (100 * forg_ds / n_ds) if n_ds else np.nan,
            'pct_forg_noDS':      (100 * forg_no / n_no) if n_no else np.nan,
            'mean_n_S_DS_trials': gdf['n_S_DS_trials'].mean(),
            'mean_DS_share':      100 * gdf['DS_share'].mean(),
        })
    grp = pd.DataFrame(grp_rows)
    grp['DS_forg_boost'] = grp['pct_forg_DS'] - grp['pct_forg_noDS']
    # Pooled DS share = n_S_DS_trials / (n_S_DS_trials + n_S_noDS_trials)
    grp['pooled_DS_share'] = (100 * grp['n_S_DS_trials'] /
        (grp['n_S_DS_trials'] + grp['n_S_noDS_trials']))
    grp['pooled_noDS_share'] = 100 - grp['pooled_DS_share']

    # Cohort tracker: full 33 -> 22 with DS-IED on stim trial
    cohort = {
        'n_total_encoding_pts':  int(trial['Patient'].nunique()),
        'n_DS_pts':              len(stim_pats),
    }

    # Region anatomy at the channel-row level for DS-IED stim trials
    chan_ds = ied[(ied['DuringStim'] == 'Y') & (ied['StimCond'] == 'S') &
                  (ied['MemoryOutcome'].isin(['remembered', 'forgotten']))].copy()
    chan_ds = chan_ds.merge(resp[['Patient', 'responder_status']],
                            on='Patient', how='left')

    # Bucket on the channel's anatomical label. The amygdala IS part of MTL,
    # so we do NOT collapse hippocampus + entorhinal + parahippocampal into a
    # single "MTL" bucket. Buckets reflect what the channel label literally
    # says: "Hippocampus" or "Amygdala" (or both, for bipolar pairs that span
    # the border).
    def _bucket(r):
        if pd.isna(r):
            return 'Other'
        rl = str(r).lower()
        has_hpc = 'hippocamp' in rl
        has_amyg = 'amygdal' in rl
        if has_hpc and has_amyg: return 'HPC+Amyg'
        if has_amyg:             return 'Amygdala'
        if has_hpc:              return 'Hippocampus'
        return 'Other'
    chan_ds['region_bucket'] = chan_ds['Region'].apply(_bucket)

    # Trial-level region of DS-IED (one row per (patient, trial)).
    # A trial is HPC+Amyg if any of its DS-IED channels involves amygdala AND
    # any involves hippocampus (whether on the same channel or different ones).
    trial_ds = (chan_ds.groupby(['Patient', 'Trial', 'MemoryOutcome',
                                  'responder_status'])['region_bucket']
                .apply(lambda s: sorted(set(s))).reset_index())
    def _trial_region(parts):
        ps = set(parts)
        has_hpc = ('Hippocampus' in ps) or ('HPC+Amyg' in ps)
        has_amyg = ('Amygdala' in ps) or ('HPC+Amyg' in ps)
        if has_hpc and has_amyg: return 'HPC+Amyg'
        if has_hpc:              return 'Hippocampus only'
        if has_amyg:             return 'Amygdala only'
        return 'Other only'
    trial_ds['trial_region'] = trial_ds['region_bucket'].apply(_trial_region)

    # Build per-group region tables (overall + forgotten + remembered)
    def _region_table(df):
        out = []
        for g in RESP_ORDER:
            sub = df[df['responder_status'] == g]
            row = {'responder_status': g, 'n_trials': len(sub)}
            for b in REGION_BUCKETS:
                row[b] = int((sub['trial_region'] == b).sum())
            out.append(row)
        return pd.DataFrame(out)

    region_all = _region_table(trial_ds)
    region_forg = _region_table(trial_ds[trial_ds['MemoryOutcome'] == 'forgotten'])
    region_rem = _region_table(trial_ds[trial_ds['MemoryOutcome'] == 'remembered'])

    return summary, grp, cohort, region_all, region_forg, region_rem


# -------------------------------------------------------------------------
#  Figures
# -------------------------------------------------------------------------
def plot_boost_by_patient(summary: pd.DataFrame, out_path: str):
    df = summary.dropna(subset=['DS_forg_boost']).copy()
    df = df.sort_values('avg_stim_dprime_diff').reset_index(drop=True)
    colors = [RESP_PALETTE.get(g, '#888') for g in df['responder_status']]

    sns.set_style('ticks')
    fig, ax = plt.subplots(figsize=(11, 5.5))
    bars = ax.bar(range(len(df)), df['DS_forg_boost'],
                  color=colors, edgecolor='black', linewidth=0.5)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xticks(range(len(df)))
    xlabels = [f"{p}\n({d:+.2f})" for p, d in
               zip(df['Patient'], df['avg_stim_dprime_diff'])]
    ax.set_xticklabels(xlabels, rotation=60, ha='right', fontsize=9)
    ax.set_ylabel('Forgetting boost of IED presence in the During Stimulation '
                  'time window (pp)',
                  fontsize=11)
    ax.set_title('Patient-based distribution of IEDs in the During Stimulation '
                 'time window compared to all other time windows in stimulation '
                 'trials',
                 fontsize=12, fontweight='bold')

    # Legend
    handles = [plt.Rectangle((0, 0), 1, 1, color=RESP_PALETTE[g],
                             edgecolor='black', linewidth=0.5)
               for g in RESP_ORDER if g in df['responder_status'].values]
    labels = [g for g in RESP_ORDER if g in df['responder_status'].values]
    ax.legend(handles, labels, loc='upper right', fontsize=9, frameon=False)

    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Saved {out_path}')


def plot_burden_by_group(grp: pd.DataFrame, out_path: str):
    sns.set_style('ticks')
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    g_order = [g for g in RESP_ORDER if g in grp['responder_status'].values]
    grp2 = grp.set_index('responder_status').reindex(g_order).reset_index()
    colors = [RESP_PALETTE[g] for g in grp2['responder_status']]

    # Left: pooled %forg DS vs noDS
    x = np.arange(len(grp2))
    w = 0.35
    axes[0].bar(x - w/2, grp2['pct_forg_DS'], width=w,
                color=colors, edgecolor='black', linewidth=0.5,
                label='DS-IED stim trials')
    axes[0].bar(x + w/2, grp2['pct_forg_noDS'], width=w,
                color=colors, edgecolor='black', linewidth=0.5,
                alpha=0.5, hatch='//', label='Non-DS IED stim trials')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([g.replace(' responders', '') for g in grp2['responder_status']],
                            fontsize=10)
    axes[0].set_ylabel('% forgotten', fontsize=11)
    axes[0].set_title('Forgetting on stim trials\n(pooled across patients in group)',
                      fontsize=11, fontweight='bold')
    axes[0].legend(fontsize=9, frameon=False, loc='upper right')

    # Right: DS-IED burden
    axes[1].bar(x, grp2['mean_n_S_DS_trials'],
                color=colors, edgecolor='black', linewidth=0.5)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([g.replace(' responders', '') for g in grp2['responder_status']],
                            fontsize=10)
    axes[1].set_ylabel('Mean # DS-IED stim trials per patient', fontsize=11)
    axes[1].set_title('DS-IED burden by responder group',
                      fontsize=11, fontweight='bold')
    for i, n in enumerate(grp2['mean_n_S_DS_trials']):
        axes[1].text(i, n + 0.15, f'{n:.1f}', ha='center', va='bottom', fontsize=9)

    sns.despine()
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Saved {out_path}')


# -------------------------------------------------------------------------
#  PDF helpers (APA-ish, mirrors the other build_*_report.py files)
# -------------------------------------------------------------------------
if FPDF is not None:
    class Report(FPDF):
        def __init__(self):
            super().__init__()
            self.set_auto_page_break(auto=True, margin=20)

        def header(self):
            self.set_font('Helvetica', 'I', 8)
            self.cell(0, 5, 'During-Stim IEDs and Responder Status', align='R')
            self.ln(8)

        def footer(self):
            self.set_y(-15)
            self.set_font('Helvetica', 'I', 8)
            self.cell(0, 10, f'Page {self.page_no()}', align='C')

        def h1(self, text):
            self.set_font('Helvetica', 'B', 15)
            self.ln(2)
            self.cell(0, 9, text, new_x="LMARGIN", new_y="NEXT")
            self.ln(1)

        def h2(self, text):
            self.set_font('Helvetica', 'B', 12)
            self.ln(2)
            self.cell(0, 7, text, new_x="LMARGIN", new_y="NEXT")

        def body(self, text):
            self.set_font('Times', '', 11)
            self.multi_cell(0, 5.5, text)
            self.ln(1)
else:
    class Report:
        pass


def _hline(pdf: Report, width: float, thickness: float = 0.3):
    x = pdf.get_x()
    y = pdf.get_y()
    pdf.set_line_width(thickness)
    pdf.line(x, y, x + width, y)
    pdf.ln(0.8)


_TABLE_COUNTER = {'n': 0}


def apa_table(pdf: Report, title_text: str, note_text: str,
              header_labels, header_italic_mask, widths, row_dicts, row_aligns,
              row_colors=None):
    _TABLE_COUNTER['n'] += 1
    tnum = _TABLE_COUNTER['n']

    pdf.ln(2)
    pdf.set_font('Times', 'B', 11)
    pdf.cell(0, 5.5, f'Table {tnum}', new_x="LMARGIN", new_y="NEXT")
    pdf.set_font('Times', '', 11)
    pdf.multi_cell(0, 5.5, title_text)
    pdf.ln(1)

    total_w = sum(widths)

    _hline(pdf, total_w, 0.5)
    for label, w, align in zip(header_labels, widths, row_aligns):
        # Headers: bold, no italics, aligned the same as their column's data.
        pdf.set_font('Times', 'B', 10.5)
        pdf.cell(w, 5.8, label, align=align)
    pdf.ln(5.8)
    _hline(pdf, total_w, 0.3)
    pdf.ln(0.5)

    pdf.set_font('Times', '', 10.5)
    for i, row in enumerate(row_dicts):
        if row_colors is not None and i < len(row_colors) and row_colors[i]:
            pdf.set_text_color(*row_colors[i])
        else:
            pdf.set_text_color(0, 0, 0)
        for label, w, align in zip(header_labels, widths, row_aligns):
            pdf.cell(w, 5.2, str(row.get(label, '')), align=align)
        pdf.ln(5.2)
    pdf.set_text_color(0, 0, 0)

    _hline(pdf, total_w, 0.5)
    pdf.ln(1)

    if note_text:
        pdf.set_font('Times', '', 9.5)
        pdf.cell(pdf.get_string_width('Note. '), 4.8, 'Note.')
        pdf.set_font('Times', '', 9.5)
        pdf.multi_cell(0, 4.8, ' ' + note_text)
        pdf.ln(1)


def fmt_num(v, digits=1):
    if pd.isna(v):
        return '-'
    return f'{v:.{digits}f}'


def fmt_signed(v, digits=2):
    if pd.isna(v):
        return '-'
    return f'{v:+.{digits}f}'


# -------------------------------------------------------------------------
#  Build the PDF
# -------------------------------------------------------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    (summary, grp, cohort,
     region_all, region_forg, region_rem) = compute_summaries()

    # Persist tables next to the PDF for verifiability
    summary.to_csv(os.path.join(OUT_DIR,
        'during_stim_responder_per_patient.csv'), index=False)
    grp.to_csv(os.path.join(OUT_DIR,
        'during_stim_responder_group_summary.csv'), index=False)
    region_all.to_csv(os.path.join(OUT_DIR,
        'during_stim_region_all_trials.csv'), index=False)
    region_forg.to_csv(os.path.join(OUT_DIR,
        'during_stim_region_forgotten_trials.csv'), index=False)
    region_rem.to_csv(os.path.join(OUT_DIR,
        'during_stim_region_remembered_trials.csv'), index=False)

    plot_boost_by_patient(summary, FIG_BOOST)
    plot_burden_by_group(grp, FIG_BURDEN)

    if FPDF is None:
        print('Skipping PDF build because fpdf is not installed in this Python environment.')
        return

    pdf = Report()
    pdf.add_page()

    # ---- Title ----
    pdf.set_font('Helvetica', 'B', 17)
    pdf.cell(0, 10, 'During-Stim IEDs and Responder Status',
             new_x="LMARGIN", new_y="NEXT")
    pdf.set_font('Helvetica', '', 11)
    pdf.cell(0, 6, 'Do During-Stim-window IEDs drive forgetting and anti-responder status?',
             new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    pdf.h1('Overview')
    pdf.body(
        f"Of the {cohort['n_total_encoding_pts']} encoding patients with at "
        f"least one memory-scored IED row, {cohort['n_DS_pts']} had at least "
        "one IED that fell inside the 1.8-s During-Stim window on a "
        "stimulated (S) trial. Most tables below describe that "
        f"{cohort['n_DS_pts']}-patient subset using counts and percentages; "
        "the one inferential table (Table 3, Section A2) reports the "
        "patient-clustered GEE test of the During-Stim forgetting effect "
        "within each responder group, with Benjamini-Hochberg FDR correction."
    )
    pdf.body(
        'Trial-level rows are formed by collapsing the encoding IED CSV to '
        'one row per (patient, trial); each timing window is flagged Y if '
        'any channel of that patient had an IED in the window on that trial. '
        'Memory and forgetting are scored at the trial level. Anatomy is '
        'tabulated from the original per-channel IED rows so that the region '
        'of each DS-IED is preserved.'
    )

    # ============== Analysis A ==============
    pdf.add_page()
    pdf.h1('A. Per-patient During-Stim forgetting boost')

    pdf.h2('Definition of Boost')
    pdf.body(
        'Boost is the percentage-point difference between the forgetting '
        'rate on DS-IED stim trials and the forgetting rate on non-DS IED '
        'stim trials, computed within the same patient (or pooled within '
        'the same responder group):'
    )
    pdf.set_font('Courier', '', 10)
    pdf.multi_cell(0, 5,
        '    Boost = %forg DS  -  %forg noDS\n\n'
        '    %forg DS   = 100 * (n forgotten S trials with a DS-window IED)\n'
        '                       / (n S trials with a DS-window IED)\n'
        '    %forg noDS = 100 * (n forgotten S trials whose IED was only in\n'
        '                       another timing window)\n'
        '                       / (n such S trials)')
    pdf.ln(2)
    pdf.body(
        'Both denominators are restricted to stimulated (S) trials that had '
        'at least one IED somewhere in the trial, so the comparison holds '
        'patient, stim condition, and IED presence constant - the only thing '
        'that varies is whether the IED fell in the During-Stim window or in '
        'some other timing window. Boost is in percentage points, not a '
        'ratio. Positive Boost = DS-IED stim trials are forgotten more often '
        'than the patient\'s other IED-positive stim trials; negative = '
        'forgotten less often; zero = no difference. A patient with zero '
        'trials in either bucket has Boost = "-".'
    )
    pdf.body(
        'Example (anti-responders, pooled in Table 2): 39 DS-IED S trials, '
        '18 forgotten -> %forg DS = 46.2%; 94 non-DS IED S trials, 27 '
        'forgotten -> %forg noDS = 29.1%; Boost = 46.2 - 29.1 = +17.1 pp.'
    )
    pdf.body(
        'Per-patient Boost (Table 1) is computed separately within each '
        'patient. Pooled-group Boost (Table 2) is computed by summing the '
        'four trial counts across patients in the group first, then taking '
        'the percentages and the difference, so larger-trial patients '
        'weight more.'
    )

    if os.path.exists(FIG_BOOST):
        pdf.image(FIG_BOOST, w=180)
        pdf.set_font('Times', 'I', 9.5)
        pdf.multi_cell(0, 4.8,
            'Figure 1. Per-patient DS forgetting boost (percentage points). '
            'Patients are ordered left to right by avg_stim_dprime_diff '
            '(printed under the patient ID); bar color shows responder group.')
        pdf.ln(2)

    # Per-patient table
    headers = ['Patient', 'Group', 'd\' diff',
               'n DS', 'n noDS', '%forg DS', '%forg noDS', 'Boost']
    italics = [False, False, True, True, True, False, False, True]
    widths  = [22, 26, 18, 14, 18, 22, 24, 22]
    aligns  = ['L', 'L', 'R', 'R', 'R', 'R', 'R', 'R']

    rows = []
    colors = []
    for _, r in summary.iterrows():
        rows.append({
            'Patient':    r['Patient'],
            'Group':      r['responder_status'].replace(' responders', '')
                          if pd.notna(r['responder_status']) else '-',
            'd\' diff':    fmt_signed(r['avg_stim_dprime_diff']),
            'n DS':       int(r['n_S_DS_trials']),
            'n noDS':     int(r['n_S_noDS_trials']),
            '%forg DS':   fmt_num(r['pct_forg_DS']),
            '%forg noDS': fmt_num(r['pct_forg_noDS']),
            'Boost':      fmt_signed(r['DS_forg_boost'], 1),
        })
        # Tint anti-responder rows red, strong-responder rows green
        rs = r['responder_status']
        if rs == 'Anti-responders':
            colors.append((155, 30, 30))
        elif rs == 'Strong responders':
            colors.append((30, 110, 50))
        else:
            colors.append(None)

    apa_table(pdf,
        title_text='Per-patient DS-IED forgetting on stimulated trials, sorted by avg_stim_dprime_diff.',
        note_text=('n DS / n noDS = number of IED-positive S trials with vs. '
                   'without a During-Stim-window IED. Boost = %forg DS - %forg '
                   'noDS (percentage points). Anti-responder rows in red, '
                   'strong-responder rows in green for visual scanning.'),
        header_labels=headers, header_italic_mask=italics,
        widths=widths, row_dicts=rows, row_aligns=aligns, row_colors=colors)

    # Aggregate by responder group
    pdf.add_page()
    pdf.h2('A1. Aggregate by responder group (pooled trial counts)')

    headers2 = ['Group', 'n pts', 'n DS trials', 'n noDS trials',
                '%forg DS', '%forg noDS', 'Boost']
    italics2 = [False, True, True, True, False, False, True]
    widths2  = [40, 18, 26, 30, 22, 24, 20]
    aligns2  = ['L', 'R', 'R', 'R', 'R', 'R', 'R']

    rows2 = []
    for _, r in grp.iterrows():
        rows2.append({
            'Group':         r['responder_status'],
            'n pts':         int(r['n_pts']),
            'n DS trials':   int(r['n_S_DS_trials']),
            'n noDS trials': int(r['n_S_noDS_trials']),
            '%forg DS':      fmt_num(r['pct_forg_DS']),
            '%forg noDS':    fmt_num(r['pct_forg_noDS']),
            'Boost':         fmt_signed(r['DS_forg_boost'], 1),
        })

    apa_table(pdf,
        title_text='Forgetting on DS-IED vs. non-DS IED stimulated trials, pooled within responder group.',
        note_text=('Trial counts are summed across patients in the group; '
                   'percentages are pooled (sum forgotten / sum trials) rather '
                   'than averaged across patients, so larger-trial patients '
                   'weight more.'),
        header_labels=headers2, header_italic_mask=italics2,
        widths=widths2, row_dicts=rows2, row_aligns=aligns2)

    # ---- A2: inferential within-group contrasts (precomputed GEE + FDR) ----
    pdf.ln(2)
    pdf.h2('A2. Inferential test: within-group During-Stim forgetting (patient-clustered GEE)')
    if os.path.exists(CONTRASTS_CSV):
        contrasts = pd.read_csv(CONTRASTS_CSV)
        order = ['Strong responders', 'Anti-responders',
                 'Non-responders', 'Moderate responders']
        contrasts['Predictor'] = pd.Categorical(contrasts['Group'],
                                                categories=order, ordered=True)
        contrasts = contrasts.sort_values('Predictor')
        n_trials = int(contrasts['n_trials_total'].iloc[0])
        n_pats = int(contrasts['n_patients'].iloc[0])

        pdf.body(
            'Each row is the simple effect of a During-Stimulation-window IED '
            'within that responder group, from a patient-clustered binomial GEE '
            '(forgotten ~ During-Stim x responder group, exchangeable working '
            'correlation). The odds ratio compares forgetting on stimulated '
            'IED-positive trials whose IED fell in the During-Stim window vs. '
            "the same group's stimulated IED trials whose IED fell only in other "
            'time windows (the reference). Odds ratios above 1 mean During-Stim '
            'IEDs are forgotten more often.'
        )

        def _fmt_p(v):
            # APA: no leading zero; clamp the degenerate 1.000 to .999.
            if v < 0.001:
                return '<.001'
            s = f'{v:.3f}'
            if s == '1.000':
                s = '.999'
            return s[1:] if s.startswith('0.') else s

        def _fmt_z(v):
            return '0.00' if abs(v) < 0.005 else f'{v:.2f}'

        headers_i = ['Predictor', 'OR', '95% CI', 'z', 'p', 'FDR q']
        italics_i = [False, False, False, False, False, False]
        widths_i = [46, 18, 42, 18, 22, 22]
        aligns_i = ['L', 'C', 'C', 'C', 'C', 'C']
        rows_i = []
        for _, r in contrasts.iterrows():
            rows_i.append({
                'Predictor': r['Group'],
                'OR': f"{r['OR']:.2f}",
                '95% CI': f"[{r['CI_low']:.2f}, {r['CI_high']:.2f}]",
                'z': _fmt_z(r['z']),
                'p': _fmt_p(r['p']),
                'FDR q': _fmt_p(r['q_fdr']),
            })

        apa_table(pdf,
            title_text=('Within-group During-Stimulation vs. other-window '
                        'forgetting contrasts (patient-clustered binomial GEE).'),
            note_text=(f'Model fit on {n_trials} stimulated IED-positive trials '
                       f'from {n_pats} patients. Reference within each group = '
                       'stimulated IED trials whose IED fell only in other time '
                       'windows (during_stim = 0); OR > 1 = more forgetting on '
                       'During-Stim-window IED trials. Strong responders is the '
                       'modeling reference group for the underlying interaction '
                       'terms. p = two-sided Wald; FDR q = Benjamini-Hochberg '
                       'across the four groups.'),
            header_labels=headers_i, header_italic_mask=italics_i,
            widths=widths_i, row_dicts=rows_i, row_aligns=aligns_i)
    else:
        pdf.body(
            'Inferential contrasts CSV not found. Run '
            'compute_during_stim_within_group_contrasts.py (anaconda env) to '
            'generate during_stim_within_group_contrasts_fdr.csv, then rebuild '
            'this report.'
        )

    # ============== Analysis B ==============
    pdf.add_page()
    pdf.h1('B. During-Stim-IED burden vs. responder status')

    pdf.body(
        'If During-Stim-IED occurrence is what pushes patients into '
        'anti/non-responder territory, anti-responders should have more '
        'DS-IED trials (and a larger DS share of their IED-positive S trials) '
        'than strong responders. The cohort summary below tests that '
        'descriptively.'
    )

    pdf.h2('How Section B was calculated')
    pdf.body(
        'Section B is built in two steps: per-patient quantities, then '
        'group-level summaries. Cohort = the same 22-patient subset used in '
        'Section A.'
    )

    pdf.set_font('Times', 'B', 11)
    pdf.cell(0, 5.5, 'Step 1. Per-patient quantities (within each patient).',
             new_x="LMARGIN", new_y="NEXT")
    pdf.set_font('Courier', '', 10)
    pdf.multi_cell(0, 5,
        '    For each patient, restrict to that patient\'s S trials with at\n'
        '    least one IED somewhere in the trial, then compute:\n\n'
        '    n_S_IED_trials = (n_S_DS_trials) + (n_S_noDS_trials)\n'
        '    n_S_DS_trials  = # S trials with a DS-window IED\n'
        '    n_S_noDS_trials= # S trials whose IED was only in another\n'
        '                     timing window\n'
        '    DS_share       = n_S_DS_trials / n_S_IED_trials')
    pdf.ln(2)
    pdf.body(
        'avg_stim_dprime_diff (the d\' modulation score) and responder_status '
        'are read from AMMEBLAES_responder_status.csv and joined to the '
        'patient row.'
    )

    pdf.set_font('Times', 'B', 11)
    pdf.cell(0, 5.5, 'Step 2. Group-level aggregates (within each responder group).',
             new_x="LMARGIN", new_y="NEXT")
    pdf.set_font('Courier', '', 10)
    pdf.multi_cell(0, 5,
        '    For each responder group g (Anti, Non, Moderate, Strong):\n\n'
        '    n_pts            = # patients in g\n'
        '    mean_dprime_diff = mean(avg_stim_dprime_diff) over patients in g\n'
        '    mean_n_S_DS      = mean(n_S_DS_trials) over patients in g\n'
        '    mean_DS_share    = 100 * mean(DS_share) over patients in g\n'
        '    sum_DS_trials    = sum(n_S_DS_trials) over patients in g\n'
        '    sum_noDS_trials  = sum(n_S_noDS_trials) over patients in g')
    pdf.ln(2)

    pdf.h2('Important distinction: mean-of-shares vs. pooled share')
    pdf.body(
        'Table 3 reports mean DS share, which is the average of per-patient '
        'DS_share values - so every patient counts equally regardless of how '
        'many trials they contributed. One patient with a single S trial '
        'that happened to be a DS trial contributes 100% to the mean.'
    )
    pdf.body(
        'Table 4 (Section C) reports % DS, which is the pooled share: '
        'sum(n_S_DS_trials) / sum(n_S_IED_trials) within group. This weights '
        'each trial equally, so patients with more trials count more. The '
        'two numbers can differ substantially - for example, the Strong '
        'group\'s mean DS share is ~43% but its pooled % DS is ~32% because '
        'one strong-responder patient contributed only one S(IED+) trial '
        'and it happened to be DS.'
    )
    pdf.body(
        'Both views are kept because they answer different questions: mean '
        'DS share asks "in a typical patient, what fraction of IED-positive '
        'stim trials had a DS-IED?", whereas pooled % DS asks "across all '
        'IED-positive stim trials in this group, what fraction had a DS-IED?". '
        'For comparing groups on a per-trial basis, the pooled value is the '
        'fairer metric; for comparing groups on a per-patient basis, the '
        'mean-of-shares is.'
    )

    pdf.h2('Figure 2 and Table 3')
    pdf.body(
        'Figure 2 (left) plots, within each responder group, the pooled '
        '%forg on DS-IED stim trials (solid bars) vs. on non-DS IED stim '
        'trials (hatched). These are the same numbers shown in Table 2 '
        '(Section A1) - re-plotted here so the burden-vs.-impact contrast '
        'is visible side-by-side. Figure 2 (right) plots mean_n_S_DS_trials '
        'for each group: the average per-patient count of DS-IED stim trials.'
    )

    if os.path.exists(FIG_BURDEN):
        pdf.image(FIG_BURDEN, w=180)
        pdf.set_font('Times', 'I', 9.5)
        pdf.multi_cell(0, 4.8,
            'Figure 2. Left: pooled forgetting rate on DS-IED stim trials '
            '(solid) vs. non-DS IED stim trials (hatched), by responder group. '
            'Right: mean per-patient count of DS-IED stim trials in each '
            'responder group.')
        pdf.ln(2)

    headers3 = ['Group', 'n pts', 'mean d\' diff',
                'mean n DS', 'mean DS share',
                'sum DS', 'sum noDS']
    italics3 = [False, True, True, False, False, False, False]
    widths3  = [40, 14, 26, 24, 28, 20, 22]
    aligns3  = ['L', 'R', 'R', 'R', 'R', 'R', 'R']

    rows3 = []
    for _, r in grp.iterrows():
        rows3.append({
            'Group':           r['responder_status'],
            'n pts':           int(r['n_pts']),
            'mean d\' diff':    fmt_signed(r['mean_dprime_diff']),
            'mean n DS':       fmt_num(r['mean_n_S_DS_trials']),
            'mean DS share':   f"{r['mean_DS_share']:.1f}%",
            'sum DS':          int(r['n_S_DS_trials']),
            'sum noDS':        int(r['n_S_noDS_trials']),
        })

    apa_table(pdf,
        title_text='During-Stim-IED burden by responder group.',
        note_text=('mean n DS = average per-patient count of IED-positive S '
                   'trials whose IED falls in the During-Stim window. mean DS '
                   'share = average per-patient % of IED-positive S trials '
                   'with a DS-window IED.'),
        header_labels=headers3, header_italic_mask=italics3,
        widths=widths3, row_dicts=rows3, row_aligns=aligns3)

    # ============== Section C: Proportion of S(IED+) trials with DS-IED ==============
    pdf.add_page()
    pdf.h1('C. Proportion of IED-positive stim trials with vs. without a DS-IED')

    pdf.body(
        'Pooled across patients within each responder group: of all stim (S) '
        'trials that had at least one IED somewhere in the trial, what '
        'fraction had a During-Stim-window IED vs. only IEDs in other windows. '
        'This is the per-trial denominator that pairs with the forgetting '
        'percentages in Tables 1-3.'
    )

    headers_p = ['Group', 'n S(IED+)', 'n DS', 'n noDS', '% DS', '% noDS']
    italics_p = [False, True, True, True, False, False]
    widths_p  = [40, 26, 22, 26, 22, 26]
    aligns_p  = ['L', 'R', 'R', 'R', 'R', 'R']

    rows_p = []
    for _, r in grp.iterrows():
        n = int(r['n_S_DS_trials']) + int(r['n_S_noDS_trials'])
        rows_p.append({
            'Group':     r['responder_status'],
            'n S(IED+)': n,
            'n DS':      int(r['n_S_DS_trials']),
            'n noDS':    int(r['n_S_noDS_trials']),
            '% DS':      f"{r['pooled_DS_share']:.1f}",
            '% noDS':    f"{r['pooled_noDS_share']:.1f}",
        })
    apa_table(pdf,
        title_text='Proportion of IED-positive stim trials with vs. without a During-Stim-window IED, by responder group.',
        note_text=('n S(IED+) = number of S trials with any IED. n DS / n '
                   'noDS = number of those trials with vs. without an IED in '
                   'the During-Stim window. Percentages are pooled across '
                   'patients within group (sum of trial counts), not averaged '
                   'across patients.'),
        header_labels=headers_p, header_italic_mask=italics_p,
        widths=widths_p, row_dicts=rows_p, row_aligns=aligns_p)

    # ============== Section D: Anatomy of DS-IED stim trials ==============
    pdf.add_page()
    pdf.h1('D. Anatomy of DS-IED stim trials (where the DS-IED occurred)')

    pdf.body(
        'Each S trial with a DS-IED is classified by the region(s) of the '
        'channel(s) that fired in the During-Stim window. Buckets reflect '
        'the literal anatomical label on the channel: Hippocampus only, '
        'Amygdala only, HPC+Amyg (the trial has DS-IED channels in both '
        'regions, or a single bipolar pair labeled "Amygdala, Hippocampus" '
        'that spans the border), or Other only (parahippocampal, entorhinal, '
        'and any non-MTL channel - the amygdala is itself part of MTL, so we '
        'deliberately do not lump it with hippocampus or with parahippocampal/'
        'entorhinal). One row per trial; totals match Table 4. Tables 5 and '
        '6 split the same trials by memory outcome.'
    )

    # Shorten the region headers so they fit/read (data keys stay as REGION_BUCKETS).
    REGION_LABELS = {
        'Hippocampus only': 'HPC only',
        'Amygdala only': 'AMYG only',
        'HPC+Amyg': 'HPC+AMYG',
        'Other only': 'Other only',
    }

    def _region_rows(rt):
        out = []
        for _, r in rt.iterrows():
            row = {'Group': r['responder_status'], 'n trials': int(r['n_trials'])}
            for b in REGION_BUCKETS:
                row[REGION_LABELS[b]] = int(r[b])
            out.append(row)
        return out

    headers_r = ['Group', 'n trials'] + [REGION_LABELS[b] for b in REGION_BUCKETS]
    italics_r = [False, True, False, False, False, False]
    widths_r  = [40, 18, 28, 24, 22, 28]
    aligns_r  = ['L', 'R', 'R', 'R', 'R', 'R']

    apa_table(pdf,
        title_text='Region of DS-IED on stim trials (all memory outcomes), by responder group.',
        note_text=('Trial-level counts. A trial enters HPC+Amyg if it has at '
                   'least one DS-IED channel labeled with hippocampus and at '
                   'least one labeled with amygdala (or a single bipolar pair '
                   'labeled "Amygdala, Hippocampus"). Otherwise classified by '
                   'the single category present.'),
        header_labels=headers_r, header_italic_mask=italics_r,
        widths=widths_r, row_dicts=_region_rows(region_all), row_aligns=aligns_r)

    apa_table(pdf,
        title_text='Region of DS-IED on FORGOTTEN stim trials, by responder group.',
        note_text='Same conventions as Table 4. Subset to forgotten trials only.',
        header_labels=headers_r, header_italic_mask=italics_r,
        widths=widths_r, row_dicts=_region_rows(region_forg), row_aligns=aligns_r)

    apa_table(pdf,
        title_text='Region of DS-IED on REMEMBERED stim trials, by responder group.',
        note_text='Same conventions as Table 4. Subset to remembered trials only.',
        header_labels=headers_r, header_italic_mask=italics_r,
        widths=widths_r, row_dicts=_region_rows(region_rem), row_aligns=aligns_r)

    pdf.h2('Caveats')
    pdf.body(
        '- The non-DS comparison set is S trials with an IED in some other '
        'timing window. The encoding IED CSV does not contain S trials with '
        'no IED at all, so a no-IED baseline is not available.'
    )
    pdf.body(
        '- DS-IED means at least one channel had an IED inside the During-Stim '
        'window for that trial; it is not weighted by the number of channels '
        'involved or by IED amplitude.'
    )
    pdf.body(
        '- Region buckets reflect the channel\'s anatomical label in the '
        'encoding IED CSV. Buckets are explicitly hippocampus-labeled vs. '
        'amygdala-labeled (the amygdala IS part of MTL, so we do not collapse '
        'it with hippocampus into a generic "MTL" bucket); bipolar pairs '
        'spanning the border ("Amygdala, Hippocampus") are placed in HPC+Amyg, '
        'and parahippocampal/entorhinal channels go into "Other only".'
    )
    pdf.body(
        '- A small count of pure-amygdala DS-IED trials in this table is not '
        'in tension with elsewhere-in-the-paper findings that amygdala '
        'measures (e.g., amygdala power) interact with DS-IED occurrence. '
        'Those analyses index the amygdala as the *recording / measurement* '
        'site (which receives stim) regardless of where on the brain the IED '
        'fired; this table indexes the *channel that fired the IED itself*. '
        'Most DS-IEDs in this dataset arise on hippocampal channels even on '
        'trials whose amygdala power is also being modulated.'
    )

    pdf.output(PDF_PATH)
    print(f'Saved {PDF_PATH}')


if __name__ == '__main__':
    main()
