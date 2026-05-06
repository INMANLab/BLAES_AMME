#!/usr/bin/env python
"""
Build APA-style PDF for the *original* Quon Figure 2A analyses (encoding +
retrieval) that used Mann-Whitney U tests and point-biserial correlations
on channel-level IED rows.

This is a companion to IED_Features_GLMM_Report.pdf; the GLMM version is
preferred because the Mann-Whitney / point-biserial tests treat channel
rows as independent (pseudoreplication) and do not adjust each feature
for the others. This report exists so the original numbers are available
in a clean, citable format.

Inputs (already on disk):
  outputs/ied_Quon_paper/figure2_stats.csv
  outputs/ied_Quon_paper/figure2_ied_features_memory_no_white_nonmtl_bold_axes.png
  outputs/ied_Quon_paper/retrieval/figure2_retrieval_stats.csv
  outputs/ied_Quon_paper/retrieval/figure2_retrieval_ied_features_memory_bold_axes.png

Output:
  outputs/ied_Quon_paper/IED_Features_Classical_Report.pdf
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
from fpdf import FPDF

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, 'outputs', 'ied_Quon_paper')

ENC_STATS_CSV = os.path.join(OUT_DIR, 'figure2_stats.csv')
RET_STATS_CSV = os.path.join(OUT_DIR, 'retrieval', 'figure2_retrieval_stats.csv')
# Raw retrieval IED CSV - used to compute the Before-Image Mann-Whitney
# that the upstream ied_quon_retrieval.py script does not currently emit.
RET_IED_CSV = os.path.join(SCRIPT_DIR, 'IED',
    'AMMEBLAES_IEDs_trial_level_dissertation_test_usethis_cleaned_with_memory.csv')

# Both Panel-A figures are (re)built below from the upstream stats CSVs so
# the feature lineup, Option-C window-only contrasts, SE error bars, and
# retrieval patient-exclusion stay in sync.
ENC_FIG = os.path.join(OUT_DIR,
    'figure2_ied_features_memory_classical_v3.png')
RET_FIG = os.path.join(OUT_DIR, 'retrieval',
    'figure2_retrieval_ied_features_memory_classical_v3.png')

PDF_PATH = os.path.join(OUT_DIR, 'IED_Features_Classical_Report.pdf')

SIG_COLOR = '#C44E52'
NS_COLOR = '#555555'

# Order the features the same way they appear on the figures.  Anything not
# listed here is filtered out (e.g. retrieval region-pair contrasts which
# belong to a separate supplementary panel).
ENC_FEATURE_ORDER = [
    'IED Rate',
    'Hemisphere (R-L)',
    'Channel Spread',
    'Region Spread',
    'Before Image (Only)',
    'During Image (Only)',
    'After Image (Only)',
    'During Stim (Only)',
]
RET_FEATURE_ORDER = [
    'IED Rate',
    'Hemisphere (R-L)',
    'Channel Spread',
    'Region Spread',
    'Before Image (Only)',
    'During Image/ITI (Only)',
]

# Pretty terms and the predictor-type label that goes in the table.
PRETTY = {
    'IED Rate':                 ('IED Rate',              'continuous (IEDs / trial)'),
    'Hemisphere (R-L)':         ('Hemisphere (R - L)',    'binary (R vs. L)'),
    'Region (nonMTL-MTL)':      ('Region (nonMTL - MTL)', 'binary (nonMTL vs. MTL)'),
    'Channel Spread':           ('Channel Spread',        'continuous'),
    'Region Spread':            ('Region Spread',         'continuous'),
    'Before Image (Only)':      ('Before Image (Only vs. Others)',  'binary (window-only vs. others)'),
    'During Image (Only)':      ('During Image (Only vs. Others)',  'binary (window-only vs. others)'),
    'After Image (Only)':       ('After Image (Only vs. Others)',   'binary (window-only vs. others)'),
    'During Stim (Only)':       ('During Stim (Only vs. Others)',   'binary (window-only vs. others)'),
    'During Image/ITI (Only)':  ('During Image/ITI (Only vs. Others)', 'binary (window-only vs. others)'),
}


# =========================================================================
#  Build a Panel-A bar figure with SE error bars from a stats CSV
# =========================================================================
PRETTY_X = {
    'IED Rate':                'IED Rate\n(IEDs/trial)',
    'Hemisphere (R-L)':        'IED Hemisphere\nRight -\nLeft (ref)',
    'Channel Spread':          'Channel\nSpread',
    'Region Spread':           'Region\nSpread',
    'Before Image (Only)':     'Before Image\nOnly -\nOther (ref)',
    'During Image (Only)':     'During Image\nOnly -\nOther (ref)',
    'After Image (Only)':      'After Image\nOnly -\nOther (ref)',
    'During Stim (Only)':      'During Stim\nOnly -\nOther (ref)',
    'During Image/ITI (Only)': 'During Image/ITI\nOnly -\nOther (ref)',
}


def build_panel_a(stats_df: pd.DataFrame, order, out_path: str, title: str):
    sub = stats_df[stats_df['Feature'].isin(order)].copy()
    sub['order'] = sub['Feature'].map({f: i for i, f in enumerate(order)})
    sub = sub.sort_values('order').reset_index(drop=True)

    sns.set_style('ticks')
    sns.set_context('talk', font_scale=0.85)

    fig, ax = plt.subplots(figsize=(max(10, 1.4 * len(sub)), 6))
    colors = [SIG_COLOR if pv < 0.05 else NS_COLOR for pv in sub['p_value']]
    ax.bar(range(len(sub)), sub['r_or_beta'],
           color=colors, edgecolor='black', linewidth=0.6)
    if 'se' in sub.columns:
        ax.errorbar(range(len(sub)), sub['r_or_beta'],
                    yerr=sub['se'].astype(float),
                    fmt='none', ecolor='black', capsize=4, linewidth=1.2)
    ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
    ax.set_xticks(range(len(sub)))
    ax.set_xticklabels([PRETTY_X.get(f, f) for f in sub['Feature']], fontsize=10)
    ax.set_ylabel('Effect Size (r or prop. difference) +/- 1 SE', fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')

    max_abs = float((sub['r_or_beta'].abs().max() or 1))
    for i, r in sub.iterrows():
        pv = r['p_value']
        if pd.isna(pv):
            continue
        if pv < 0.001: star = '***'
        elif pv < 0.01: star = '**'
        elif pv < 0.05: star = '*'
        else: star = ''
        if not star:
            continue
        eff = r['r_or_beta']
        se_i = r['se'] if 'se' in r and pd.notna(r['se']) else 0
        offset = max_abs * 0.04
        y_pos = eff + se_i + offset if eff >= 0 else eff - se_i - offset
        va = 'bottom' if eff >= 0 else 'top'
        ax.text(i, y_pos, star, ha='center', va=va,
                fontsize=14, fontweight='bold', color=SIG_COLOR)

    sns.despine(ax=ax)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Saved {out_path}')


def fmt_p(p):
    if pd.isna(p):
        return ''
    if p < 0.001:
        return '< .001'
    return f'= {p:.3f}'


def fmt_num(v, digits=3):
    if pd.isna(v):
        return '-'
    return f'{v:.{digits}f}'


# =========================================================================
#  PDF helpers (APA style, matching IED_Features_GLMM_Report)
# =========================================================================
class Report(FPDF):
    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=20)

    def header(self):
        self.set_font('Helvetica', 'I', 8)
        self.cell(0, 5, 'IED Features vs. Memory (classical) - Quon Replication',
                  align='R')
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


def _hline(pdf: Report, width: float, thickness: float = 0.3):
    x = pdf.get_x()
    y = pdf.get_y()
    pdf.set_line_width(thickness)
    pdf.line(x, y, x + width, y)
    pdf.ln(0.8)


_TABLE_COUNTER = {'n': 0}


def apa_table(pdf: Report, title_text: str, note_text: str,
              header_labels, header_italic_mask, widths, row_dicts,
              row_aligns):
    _TABLE_COUNTER['n'] += 1
    tnum = _TABLE_COUNTER['n']

    pdf.ln(2)
    pdf.set_font('Times', 'B', 11)
    pdf.cell(0, 5.5, f'Table {tnum}', new_x="LMARGIN", new_y="NEXT")
    pdf.set_font('Times', 'I', 11)
    pdf.multi_cell(0, 5.5, title_text)
    pdf.ln(1)

    total_w = sum(widths)

    _hline(pdf, total_w, 0.5)
    for label, italic, w, align in zip(header_labels, header_italic_mask,
                                       widths, row_aligns):
        pdf.set_font('Times', 'I' if italic else '', 10.5)
        pdf.cell(w, 5.8, label, align=align)
    pdf.ln(5.8)
    _hline(pdf, total_w, 0.3)
    pdf.ln(0.5)

    pdf.set_font('Times', '', 10.5)
    for row in row_dicts:
        for label, w, align in zip(header_labels, widths, row_aligns):
            pdf.cell(w, 5.2, str(row.get(label, '')), align=align)
        pdf.ln(5.2)

    _hline(pdf, total_w, 0.5)
    pdf.ln(1)

    if note_text:
        pdf.set_font('Times', 'I', 9.5)
        pdf.cell(pdf.get_string_width('Note. '), 4.8, 'Note.')
        pdf.set_font('Times', '', 9.5)
        pdf.multi_cell(0, 4.8, ' ' + note_text)
        pdf.ln(1)


def stats_table(pdf: Report, df: pd.DataFrame, order, title, note):
    df2 = df[df['Feature'].isin(order)].copy()
    df2['order'] = df2['Feature'].map({f: i for i, f in enumerate(order)})
    df2 = df2.sort_values('order').reset_index(drop=True)

    headers = ['Predictor', 'Type', 'Test', 'r / diff', 'p']
    italics = [False, False, False, True, True]
    widths  = [50, 44, 30, 25, 22]
    aligns  = ['L', 'L', 'L', 'R', 'R']

    rows = []
    for _, r in df2.iterrows():
        pretty, ptype = PRETTY.get(r['Feature'], (r['Feature'], ''))
        test_label = ('Mann-Whitney U' if r['test'] == 'Mann-Whitney'
                      else 'point-biserial r')
        rows.append({
            'Predictor': pretty,
            'Type':      ptype,
            'Test':      test_label,
            'r / diff':  fmt_num(r['r_or_beta'], 3),
            'p':         fmt_p(r['p_value']).lstrip('= ').strip(),
        })

    apa_table(pdf, title_text=title, note_text=note,
              header_labels=headers, header_italic_mask=italics,
              widths=widths, row_dicts=rows, row_aligns=aligns)


def main():
    if not os.path.exists(ENC_STATS_CSV) or not os.path.exists(RET_STATS_CSV):
        raise SystemExit(
            'Missing classical stats CSVs.  Run ied_quon_replication.py and '
            'ied_quon_retrieval.py first.')

    enc = pd.read_csv(ENC_STATS_CSV)
    ret = pd.read_csv(RET_STATS_CSV)

    # Drop Region (nonMTL-MTL) from retrieval (per request).
    ret = ret[ret['Feature'] != 'Region (nonMTL-MTL)'].copy()
    ret_out_csv = os.path.join(OUT_DIR, 'figure2_retrieval_stats_classical_v2.csv')
    ret[ret['Feature'].isin(RET_FEATURE_ORDER)].to_csv(ret_out_csv, index=False)
    print(f'Saved {ret_out_csv}')

    # (Re)build both Panel A figures with SE error bars and Option-C windows.
    build_panel_a(
        enc, ENC_FEATURE_ORDER, ENC_FIG,
        title='(A) IED Features Associated with Memory Encoding'
    )
    build_panel_a(
        ret, RET_FEATURE_ORDER, RET_FIG,
        title='(A) Retrieval: IED Features Associated with Memory (excl. BJH042, UIC202306)'
    )

    pdf = Report()
    pdf.add_page()

    # -------- Title page --------
    pdf.set_font('Helvetica', 'B', 17)
    pdf.cell(0, 10, 'IED Features vs. Memory - Classical Tests',
             new_x="LMARGIN", new_y="NEXT")
    pdf.set_font('Helvetica', '', 11)
    pdf.cell(0, 6, 'Quon (2021) Figure 2A analog: Mann-Whitney U + point-biserial r',
             new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    pdf.h1('Overview')
    pdf.body(
        'For each candidate IED feature, encoding and retrieval channel-level '
        'IED rows were compared between subsequently remembered and forgotten '
        'trials. Continuous features (IED rate, channel spread, region spread) '
        'were tested with point-biserial correlation against the binary '
        'memory outcome. Binary features (hemisphere, timing windows) were '
        'tested with the two-sided Mann-Whitney U test comparing the '
        'proportion of remembered trials between the two groups; the '
        'reported effect is the simple proportion difference (group_1 - '
        'group_0). No multiple-comparison correction was applied. White '
        'matter and the broad MTL/non-MTL contrast are dropped from both '
        'phases per request; the supplementary region-pair contrasts at '
        'retrieval are also omitted.'
    )
    pdf.body(
        'Caveat: each row in the underlying tables is a single channel x trial '
        'IED entry, so the same trial contributes multiple rows when more '
        'than one channel had an IED. These tests therefore treat channel '
        'rows as independent, which is the pseudoreplication concern that '
        'the companion IED_Features_GLMM_Report.pdf addresses by collapsing '
        'IEDs to one row per (patient, trial) and adding a patient random '
        'intercept. The numbers in this report are kept for direct '
        'comparison with the published Quon (2021) replication panel.'
    )

    # -------- Encoding --------
    pdf.add_page()
    pdf.h1('Encoding phase')
    pdf.h2('(A) IED features associated with encoding memory')
    if os.path.exists(ENC_FIG):
        pdf.image(ENC_FIG, w=180)
        pdf.set_font('Times', 'I', 9.5)
        pdf.multi_cell(0, 4.8,
                       'Figure 1. Effect size (point-biserial r for continuous '
                       'features; proportion remembered difference for binary '
                       'features) for each IED feature on encoding trials. '
                       'Red bars indicate p < .05.')
        pdf.ln(2)

    stats_table(
        pdf, enc, ENC_FEATURE_ORDER,
        title='Encoding-phase classical effect sizes for each IED feature.',
        note=('Continuous features tested with point-biserial r vs. memory '
              '(0/1). Binary features tested with two-sided Mann-Whitney U; '
              'the reported effect is the proportion remembered difference '
              '(level_1 - level_0). Channel-level IED rows; uncorrected p '
              'values. Compare with the joint-GLMM table in '
              'IED_Features_GLMM_Report.pdf.')
    )

    # -------- Retrieval --------
    pdf.add_page()
    pdf.h1('Retrieval phase')
    pdf.h2('(A) Retrieval IED features associated with memory')
    if os.path.exists(RET_FIG):
        pdf.image(RET_FIG, w=180)
        pdf.set_font('Times', 'I', 9.5)
        pdf.multi_cell(0, 4.8,
                       'Figure 2. Effect size for each IED feature on '
                       'retrieval trials (old items only). Same conventions '
                       'as Figure 1. Retrieval timing windows are Before '
                       'Image and During Image / ITI; the broad MTL vs. '
                       'non-MTL contrast is omitted to match the encoding '
                       'panel.')
        pdf.ln(2)

    stats_table(
        pdf, ret, RET_FEATURE_ORDER,
        title='Retrieval-phase classical effect sizes for each IED feature.',
        note=('Same conventions as the encoding table. The retrieval '
              'sample is much smaller than encoding and the channel-level '
              'pseudoreplication caveat applies equally; cross-reference '
              'with the GLMM retrieval table in '
              'IED_Features_GLMM_Report.pdf.')
    )

    # -------- Interpretation --------
    pdf.add_page()
    pdf.h1('Interpretation')

    pdf.h2('Encoding')
    enc2 = enc[enc['Feature'].isin(ENC_FEATURE_ORDER)].copy()
    sig_enc = enc2[enc2['p_value'] < 0.05].sort_values('p_value')
    if sig_enc.empty:
        pdf.body('No encoding feature reached p < .05.')
    else:
        pdf.body(
            'Features reaching p < .05 in the encoding classical tests, in '
            'ascending p order:'
        )
        for _, r in sig_enc.iterrows():
            pretty = PRETTY.get(r['Feature'], (r['Feature'], ''))[0]
            sign = 'increases' if r['r_or_beta'] > 0 else 'reduces'
            pdf.set_font('Times', '', 11)
            pdf.multi_cell(0, 5.5,
                f"- {pretty}: r/diff = {fmt_num(r['r_or_beta'], 3)}, "
                f"p {fmt_p(r['p_value'])}; the feature {sign} the proportion "
                "of trials remembered.")
            pdf.ln(0.3)

    pdf.h2('Retrieval')
    ret2 = ret[ret['Feature'].isin(RET_FEATURE_ORDER)].copy()
    sig_ret = ret2[ret2['p_value'] < 0.05].sort_values('p_value')
    if sig_ret.empty:
        pdf.body('No retrieval feature reached p < .05.')
    else:
        pdf.body(
            'Features reaching p < .05 in the retrieval classical tests, in '
            'ascending p order:'
        )
        for _, r in sig_ret.iterrows():
            pretty = PRETTY.get(r['Feature'], (r['Feature'], ''))[0]
            sign = 'increases' if r['r_or_beta'] > 0 else 'reduces'
            pdf.set_font('Times', '', 11)
            pdf.multi_cell(0, 5.5,
                f"- {pretty}: r/diff = {fmt_num(r['r_or_beta'], 3)}, "
                f"p {fmt_p(r['p_value'])}; the feature {sign} the proportion "
                "of trials remembered.")
            pdf.ln(0.3)

    pdf.ln(2)
    pdf.body(
        'These classical tests are useful as a direct visual analog of '
        'Quon (2021) Figure 2A but should be interpreted alongside the '
        'trial-level joint GLMM in IED_Features_GLMM_Report.pdf, which '
        'removes channel-row pseudoreplication, adds a patient random '
        'intercept, and adjusts each feature for the others simultaneously.'
    )

    pdf.output(PDF_PATH)
    print(f'Saved {PDF_PATH}')


if __name__ == '__main__':
    main()
