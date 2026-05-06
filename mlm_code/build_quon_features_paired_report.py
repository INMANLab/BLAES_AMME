#!/usr/bin/env python
"""
Build APA-style PDF for the *within-patient paired* analyses of the Quon
Figure 2A IED features (encoding + retrieval).

Each patient is used as their own control: per-patient summaries are
computed for each IED feature, then a one-sample paired test is run
across patients.

Per feature:
  - Binary feature (hemisphere, timing windows):
      * Per-patient delta = P(remembered | feat=1) - P(remembered | feat=0)
      * Patients without both levels are dropped.
      * Wilcoxon signed-rank test on the deltas (primary).
      * Cochran-Mantel-Haenszel chi-square stratified by patient (secondary,
        binary-outcome twin of the paired t-test).
  - Continuous feature (IED rate, channel spread, region spread):
      * Per-patient point-biserial r between feature value and memory.
      * Patients with constant memory or constant feature are dropped.
      * Wilcoxon signed-rank test on r vs. 0.

Inputs (already on disk, written by ied_quon_features_glmm.R):
  outputs/ied_Quon_paper/figure2_glmm_trial_features.csv
  outputs/ied_Quon_paper/figure2_glmm_trial_features_retrieval.csv

Output:
  outputs/ied_Quon_paper/IED_Features_Paired_Report.pdf
  outputs/ied_Quon_paper/figure2_paired_stats.csv
  outputs/ied_Quon_paper/figure2_paired_stats_retrieval.csv
  outputs/ied_Quon_paper/figure2_ied_features_memory_paired.png
  outputs/ied_Quon_paper/figure2_ied_features_memory_paired_retrieval.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from fpdf import FPDF

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, 'outputs', 'ied_Quon_paper')

ENC_TRIAL_CSV = os.path.join(OUT_DIR, 'figure2_glmm_trial_features.csv')
RET_TRIAL_CSV = os.path.join(OUT_DIR, 'figure2_glmm_trial_features_retrieval.csv')

ENC_FIG = os.path.join(OUT_DIR, 'figure2_ied_features_memory_paired.png')
RET_FIG = os.path.join(OUT_DIR, 'figure2_ied_features_memory_paired_retrieval.png')
ENC_STATS_OUT = os.path.join(OUT_DIR, 'figure2_paired_stats.csv')
RET_STATS_OUT = os.path.join(OUT_DIR, 'figure2_paired_stats_retrieval.csv')
PDF_PATH = os.path.join(OUT_DIR, 'IED_Features_Paired_Report.pdf')

SIG_COLOR = '#C44E52'
NS_COLOR = '#555555'

# Per-phase feature lists: (display name, column, kind 'binary'/'continuous')
ENC_FEATURES = [
    ('IED Rate',                 'ied_rate',    'continuous'),
    ('Hemisphere (R vs. none)',  'any_R_ied',   'binary'),
    ('Channel Spread',           'ch_spread',   'continuous'),
    ('Region Spread',            'reg_spread',  'continuous'),
    ('Before Image',             'before_img',  'binary'),
    ('During Image',             'during_img',  'binary'),
    ('After Image',              'after_img',   'binary'),
    ('During Stim',              'during_stim', 'binary'),
]
RET_FEATURES = [
    ('IED Rate',                 'ied_rate',    'continuous'),
    ('Hemisphere (R vs. none)',  'any_R_ied',   'binary'),
    ('Channel Spread',           'ch_spread',   'continuous'),
    ('Region Spread',            'reg_spread',  'continuous'),
    ('Before Image',             'before_img',  'binary'),
    ('During Image',             'during_img',  'binary'),
]


# =========================================================================
#  STATS
# =========================================================================
def per_patient_binary(df: pd.DataFrame, col: str):
    """Return DataFrame with one row per patient that has both levels of
    `col`: columns p_yes, p_no, delta = p_yes - p_no, n_yes, n_no."""
    rows = []
    for pid, g in df.groupby('patient_id'):
        yes = g[g[col] == 1]['memory']
        no  = g[g[col] == 0]['memory']
        if len(yes) == 0 or len(no) == 0:
            continue
        rows.append({
            'patient_id': pid,
            'p_yes': yes.mean(),
            'p_no':  no.mean(),
            'delta': yes.mean() - no.mean(),
            'n_yes': int(len(yes)),
            'n_no':  int(len(no)),
        })
    return pd.DataFrame(rows)


def cmh_test(df: pd.DataFrame, col: str):
    """Cochran-Mantel-Haenszel chi-square stratified by patient on the 2x2
    table of memory (1/0) x feature (1/0).  Implemented directly so we
    don't need statsmodels.

    For each patient k, build a 2x2 table:
        a_k = #(memory=1, feat=1)   b_k = #(memory=1, feat=0)
        c_k = #(memory=0, feat=1)   d_k = #(memory=0, feat=0)
        n_k = a_k + b_k + c_k + d_k

    CMH chi-square (continuity-corrected) is:
        ( |sum a_k - sum E(a_k)| - 0.5 )^2 / sum Var(a_k)
    with E(a_k) = (a_k+b_k)(a_k+c_k)/n_k and
         Var(a_k) = (a_k+b_k)(c_k+d_k)(a_k+c_k)(b_k+d_k) / (n_k^2 (n_k-1))
    Strata with n_k <= 1 or zero margins contribute nothing.
    """
    a_sum = e_sum = v_sum = 0.0
    n_strata = 0
    for _, g in df.groupby('patient_id'):
        a = int(((g['memory'] == 1) & (g[col] == 1)).sum())
        b = int(((g['memory'] == 1) & (g[col] == 0)).sum())
        c = int(((g['memory'] == 0) & (g[col] == 1)).sum())
        d = int(((g['memory'] == 0) & (g[col] == 0)).sum())
        n = a + b + c + d
        if n < 2:
            continue
        # Need both feature levels and both outcome levels for variance to
        # be > 0; otherwise the patient stratum contributes 0 variance.
        if (a + b) == 0 or (c + d) == 0 or (a + c) == 0 or (b + d) == 0:
            continue
        ea = (a + b) * (a + c) / n
        va = (a + b) * (c + d) * (a + c) * (b + d) / (n * n * (n - 1))
        a_sum += a
        e_sum += ea
        v_sum += va
        n_strata += 1
    if v_sum <= 0 or n_strata == 0:
        return np.nan, np.nan, n_strata
    chi2 = (abs(a_sum - e_sum) - 0.5) ** 2 / v_sum
    p = 1 - stats.chi2.cdf(chi2, df=1)
    return chi2, p, n_strata


def per_patient_continuous(df: pd.DataFrame, col: str):
    """Within-patient point-biserial r between continuous `col` and memory.
    Patients with no variation in either variable are dropped."""
    rows = []
    for pid, g in df.groupby('patient_id'):
        x = g[col].astype(float).to_numpy()
        y = g['memory'].astype(float).to_numpy()
        if len(x) < 3 or np.all(x == x[0]) or np.all(y == y[0]):
            continue
        # point-biserial = Pearson r when y is binary
        r = np.corrcoef(x, y)[0, 1]
        if np.isnan(r):
            continue
        rows.append({'patient_id': pid, 'r': r, 'n_trials': int(len(x))})
    return pd.DataFrame(rows)


def run_phase(df: pd.DataFrame, features):
    """Return a tidy DataFrame of per-feature paired test results."""
    out = []
    for label, col, kind in features:
        if kind == 'binary':
            pp = per_patient_binary(df, col)
            if pp.empty or pp['delta'].nunique() < 2:
                out.append(_empty_row(label, kind, n_pat=len(pp)))
                continue
            mean_delta = pp['delta'].mean()
            median_delta = pp['delta'].median()
            try:
                w_stat, w_p = stats.wilcoxon(pp['delta'].values,
                                             zero_method='wilcox',
                                             alternative='two-sided')
            except ValueError:
                w_stat, w_p = np.nan, np.nan
            chi2, cmh_p, n_strata = cmh_test(df, col)
            out.append({
                'Feature': label, 'Kind': kind,
                'n_patients': len(pp),
                'effect': mean_delta,
                'median_effect': median_delta,
                'wilcoxon_W': w_stat,
                'wilcoxon_p': w_p,
                'cmh_chi2': chi2,
                'cmh_df': 1 if not np.isnan(chi2) else np.nan,
                'cmh_p': cmh_p,
                'cmh_n_strata': n_strata,
            })
        else:
            pp = per_patient_continuous(df, col)
            if pp.empty or pp['r'].nunique() < 2:
                out.append(_empty_row(label, kind, n_pat=len(pp)))
                continue
            mean_r = pp['r'].mean()
            median_r = pp['r'].median()
            try:
                w_stat, w_p = stats.wilcoxon(pp['r'].values,
                                             zero_method='wilcox',
                                             alternative='two-sided')
            except ValueError:
                w_stat, w_p = np.nan, np.nan
            out.append({
                'Feature': label, 'Kind': kind,
                'n_patients': len(pp),
                'effect': mean_r,
                'median_effect': median_r,
                'wilcoxon_W': w_stat,
                'wilcoxon_p': w_p,
                'cmh_chi2': np.nan,
                'cmh_df': np.nan,
                'cmh_p': np.nan,
                'cmh_n_strata': np.nan,
            })
    return pd.DataFrame(out)


def _empty_row(label, kind, n_pat=0):
    return {
        'Feature': label, 'Kind': kind,
        'n_patients': n_pat,
        'effect': np.nan, 'median_effect': np.nan,
        'wilcoxon_W': np.nan, 'wilcoxon_p': np.nan,
        'cmh_chi2': np.nan, 'cmh_df': np.nan,
        'cmh_p': np.nan, 'cmh_n_strata': np.nan,
    }


# =========================================================================
#  FIGURE
# =========================================================================
def build_figure(stats_df: pd.DataFrame, out_path: str, title: str):
    sns.set_style('ticks')
    sns.set_context('talk', font_scale=0.85)

    fig, ax = plt.subplots(figsize=(max(10, 1.4 * len(stats_df)), 6))
    p = stats_df['wilcoxon_p'].fillna(1.0).values
    colors = [SIG_COLOR if pv < 0.05 else NS_COLOR for pv in p]
    ax.bar(range(len(stats_df)), stats_df['effect'],
           color=colors, edgecolor='black', linewidth=0.6)
    ax.axhline(0, color='black', linestyle='--', linewidth=0.8)

    ax.set_xticks(range(len(stats_df)))
    labels = []
    for _, r in stats_df.iterrows():
        suffix = '\n(mean delta P)' if r['Kind'] == 'binary' else '\n(mean r_pb)'
        labels.append(r['Feature'] + suffix)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel('Within-patient effect (mean across patients)', fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')

    for i, r in stats_df.iterrows():
        pv = r['wilcoxon_p']
        if pd.isna(pv):
            continue
        if pv < 0.001: star = '***'
        elif pv < 0.01: star = '**'
        elif pv < 0.05: star = '*'
        else: star = ''
        if not star:
            continue
        eff = r['effect']
        offset = (abs(stats_df['effect']).max() or 1) * 0.04
        y_pos = eff + offset if eff >= 0 else eff - offset
        va = 'bottom' if eff >= 0 else 'top'
        ax.text(i, y_pos, star, ha='center', va=va,
                fontsize=14, fontweight='bold', color=SIG_COLOR)

    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Saved {out_path}')


# =========================================================================
#  PDF (APA style — same conventions as the other two reports)
# =========================================================================
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


class Report(FPDF):
    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=20)

    def header(self):
        self.set_font('Helvetica', 'I', 8)
        self.cell(0, 5, 'IED Features vs. Memory (paired/within-patient) - Quon Replication',
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


def wilcoxon_table(pdf: Report, sdf: pd.DataFrame, title, note):
    headers = ['Predictor', 'Kind', 'n pts', 'effect', 'W', 'p']
    italics = [False, False, False, True, True, True]
    widths  = [55, 32, 18, 22, 22, 22]
    aligns  = ['L', 'L', 'R', 'R', 'R', 'R']
    rows = []
    for _, r in sdf.iterrows():
        kind_label = ('binary (delta P)' if r['Kind'] == 'binary'
                      else 'cont (r_pb)')
        rows.append({
            'Predictor': r['Feature'],
            'Kind':      kind_label,
            'n pts':     f"{int(r['n_patients'])}",
            'effect':    fmt_num(r['effect'], 3),
            'W':         fmt_num(r['wilcoxon_W'], 1),
            'p':         fmt_p(r['wilcoxon_p']).lstrip('= ').strip(),
        })
    apa_table(pdf, title_text=title, note_text=note,
              header_labels=headers, header_italic_mask=italics,
              widths=widths, row_dicts=rows, row_aligns=aligns)


def cmh_table(pdf: Report, sdf: pd.DataFrame, title, note):
    sub = sdf[sdf['Kind'] == 'binary'].copy()
    if sub.empty:
        return
    headers = ['Predictor', 'k strata', 'chi2', 'df', 'p']
    italics = [False, False, True, True, True]
    widths  = [70, 22, 25, 15, 25]
    aligns  = ['L', 'R', 'R', 'R', 'R']
    rows = []
    for _, r in sub.iterrows():
        rows.append({
            'Predictor': r['Feature'],
            'k strata':  f"{int(r['cmh_n_strata'])}" if not pd.isna(r['cmh_n_strata']) else '',
            'chi2':      fmt_num(r['cmh_chi2'], 2),
            'df':        '1' if not pd.isna(r['cmh_df']) else '',
            'p':         fmt_p(r['cmh_p']).lstrip('= ').strip(),
        })
    apa_table(pdf, title_text=title, note_text=note,
              header_labels=headers, header_italic_mask=italics,
              widths=widths, row_dicts=rows, row_aligns=aligns)


# =========================================================================
#  MAIN
# =========================================================================
def main():
    if not (os.path.exists(ENC_TRIAL_CSV) and os.path.exists(RET_TRIAL_CSV)):
        raise SystemExit(
            'Missing trial-level CSVs.  Run Rscript ied_quon_features_glmm.R first.')

    enc_df = pd.read_csv(ENC_TRIAL_CSV)
    ret_df = pd.read_csv(RET_TRIAL_CSV)

    enc_stats = run_phase(enc_df, ENC_FEATURES)
    ret_stats = run_phase(ret_df, RET_FEATURES)

    enc_stats.to_csv(ENC_STATS_OUT, index=False)
    ret_stats.to_csv(RET_STATS_OUT, index=False)
    print(f'Saved {ENC_STATS_OUT}')
    print(f'Saved {RET_STATS_OUT}')

    build_figure(enc_stats, ENC_FIG,
                 title='(A) IED features and encoding memory - within-patient paired tests')
    build_figure(ret_stats, RET_FIG,
                 title='(A) IED features and retrieval memory - within-patient paired tests')

    # ---- PDF ----
    pdf = Report()
    pdf.add_page()

    pdf.set_font('Helvetica', 'B', 17)
    pdf.cell(0, 10, 'IED Features vs. Memory - Within-Patient Paired Tests',
             new_x="LMARGIN", new_y="NEXT")
    pdf.set_font('Helvetica', '', 11)
    pdf.cell(0, 6, 'Quon (2021) Figure 2A analog: patient as own control',
             new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    pdf.h1('Overview')
    pdf.body(
        'Each IED feature is tested separately, using the patient as their '
        'own control to remove the within-patient correlation that the '
        'channel-level Mann-Whitney / point-biserial tests in '
        'IED_Features_Classical_Report.pdf ignore. Concretely:'
    )
    pdf.body(
        '- Binary features (Hemisphere, Before/During/After Image, During '
        'Stim): for each patient, compute the within-patient difference in '
        'proportion remembered between feature-present and feature-absent '
        'trials (delta P). Patients without trials in both levels are '
        'dropped. A two-sided Wilcoxon signed-rank test on the deltas '
        'asks whether the within-patient difference is reliably non-zero. '
        'A Cochran-Mantel-Haenszel chi-square stratified by patient is '
        'reported as the binary-outcome analog (continuity corrected).'
    )
    pdf.body(
        '- Continuous features (IED Rate, Channel Spread, Region Spread): '
        'for each patient, compute the within-patient point-biserial r '
        'between the feature value and trial memory. A two-sided Wilcoxon '
        'signed-rank test then asks whether the per-patient r values '
        'differ from zero.'
    )
    pdf.body(
        'Compared with the joint multivariable GLMM in '
        'IED_Features_GLMM_Report.pdf, this approach also tests one '
        'feature at a time (so it is not subject to the multicollinearity '
        'shrinkage that drove the joint-GLMM null result at encoding), but '
        'unlike the classical channel-level tests it respects the '
        'within-patient structure of the data.'
    )

    # ---- ENCODING ----
    pdf.add_page()
    pdf.h1('Encoding phase')
    pdf.h2('(A) IED features associated with encoding memory')
    if os.path.exists(ENC_FIG):
        pdf.image(ENC_FIG, w=180)
        pdf.set_font('Times', 'I', 9.5)
        pdf.multi_cell(0, 4.8,
                       'Figure 1. Within-patient effect for each IED feature '
                       '(encoding). Bars show the mean across patients of the '
                       'per-patient effect: delta P(rem) for binary features, '
                       'point-biserial r for continuous features. Red bars '
                       'indicate Wilcoxon signed-rank p < .05.')
        pdf.ln(2)

    wilcoxon_table(
        pdf, enc_stats,
        title='Within-patient Wilcoxon signed-rank tests, encoding phase.',
        note=('For binary features, the per-patient effect is delta P = '
              'P(remembered | feature=1) - P(remembered | feature=0); for '
              'continuous features it is the within-patient point-biserial '
              'r. n pts is the number of patients contributing to the test '
              '(patients lacking both feature levels or with no within-'
              'patient variation are dropped). W is the Wilcoxon test '
              'statistic; p is two-sided.')
    )

    cmh_table(
        pdf, enc_stats,
        title='Cochran-Mantel-Haenszel chi-square tests stratified by '
              'patient (binary IED features only), encoding phase.',
        note=('Continuity-corrected CMH chi-square on the 2x2 table '
              'memory x feature, stratified by patient. k strata is the '
              'number of patients with non-degenerate 2x2 tables.')
    )

    # ---- RETRIEVAL ----
    pdf.add_page()
    pdf.h1('Retrieval phase')
    pdf.h2('(A) IED features associated with retrieval memory')
    if os.path.exists(RET_FIG):
        pdf.image(RET_FIG, w=180)
        pdf.set_font('Times', 'I', 9.5)
        pdf.multi_cell(0, 4.8,
                       'Figure 2. Within-patient effect for each IED feature '
                       '(retrieval, old items only). Same conventions as '
                       'Figure 1. Retrieval has only Before-Image and '
                       'During-Image timing windows.')
        pdf.ln(2)

    wilcoxon_table(
        pdf, ret_stats,
        title='Within-patient Wilcoxon signed-rank tests, retrieval phase.',
        note=('Same conventions as the encoding table. The retrieval '
              'sample is much smaller than encoding (~18 patients), so '
              'tests are underpowered; effects should be interpreted '
              'cautiously.')
    )

    cmh_table(
        pdf, ret_stats,
        title='Cochran-Mantel-Haenszel chi-square tests stratified by '
              'patient (binary IED features only), retrieval phase.',
        note=('Continuity-corrected CMH chi-square on the 2x2 table '
              'memory x feature, stratified by patient.')
    )

    # ---- INTERPRETATION ----
    pdf.add_page()
    pdf.h1('Interpretation')

    for phase_label, sdf in [('Encoding', enc_stats), ('Retrieval', ret_stats)]:
        pdf.h2(phase_label)
        sig = sdf[sdf['wilcoxon_p'] < 0.05].sort_values('wilcoxon_p')
        if sig.empty:
            pdf.body(f'No {phase_label.lower()} feature reached p < .05 '
                     'in the Wilcoxon signed-rank test.')
        else:
            pdf.body(
                f'Features reaching Wilcoxon p < .05 in {phase_label.lower()}, '
                'in ascending p order:'
            )
            for _, r in sig.iterrows():
                kind_label = ('delta P' if r['Kind'] == 'binary'
                              else 'mean r_pb')
                sign = ('increases' if r['effect'] > 0 else 'reduces')
                cmh_part = ''
                if r['Kind'] == 'binary' and not pd.isna(r['cmh_p']):
                    cmh_part = (f"  CMH chi2(1) = {fmt_num(r['cmh_chi2'], 2)}, "
                                f"p {fmt_p(r['cmh_p'])}.")
                pdf.set_font('Times', '', 11)
                pdf.multi_cell(0, 5.5,
                    f"- {r['Feature']}: {kind_label} = "
                    f"{fmt_num(r['effect'], 3)} across "
                    f"{int(r['n_patients'])} patients, Wilcoxon p "
                    f"{fmt_p(r['wilcoxon_p'])}; the feature {sign} the odds "
                    f"of remembering within patient.{cmh_part}")
                pdf.ln(0.3)
        pdf.ln(1)

    pdf.body(
        'The within-patient framing answers the question "does this feature '
        'shift memory inside a patient?" rather than "does it shift memory '
        'when channel-level rows are pooled across patients" (classical '
        'report) or "does it predict memory adjusting for every other '
        'feature simultaneously" (joint GLMM). Because the predictors are '
        'highly correlated (see the association matrices in '
        'IED_Features_GLMM_Report.pdf), this per-feature paired view is '
        'the preferred primary analysis.'
    )

    pdf.output(PDF_PATH)
    print(f'Saved {PDF_PATH}')


if __name__ == '__main__':
    main()
