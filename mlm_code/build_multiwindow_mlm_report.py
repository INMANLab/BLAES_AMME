#!/usr/bin/env python
"""
Build PDF report for the IED Multi-Window Dose-Response MLM.

Tests whether the number of IED timing windows on a trial (dose) predicts
subsequent memory, controlling for patient random intercept and stimulation.

Inputs (from ied_multiwindow_mlm.R):
  outputs/ied_timing_memory/encoding_multiwindow_mlm_coefs.csv
  outputs/ied_timing_memory/retrieval_multiwindow_mlm_coefs.csv

Figures:
  outputs/ied_timing_memory/encoding_multiwindow_dose_response.png
  outputs/ied_timing_memory/encoding_multiwindow_combinations.png
  outputs/ied_timing_memory/retrieval_multiwindow_dose_response.png
  outputs/ied_timing_memory/retrieval_multiwindow_combinations.png

Output: outputs/ied_timing_memory/IED_Multiwindow_MLM_Report.pdf
"""

import os
import numpy as np
import pandas as pd
from fpdf import FPDF

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, 'outputs', 'ied_timing_memory')
PDF_PATH = os.path.join(OUT_DIR, 'IED_Multiwindow_MLM_Report.pdf')

ENC_COEFS = os.path.join(OUT_DIR, 'encoding_multiwindow_mlm_coefs.csv')
RET_COEFS = os.path.join(OUT_DIR, 'retrieval_multiwindow_mlm_coefs.csv')
ENC_STATS = os.path.join(OUT_DIR, 'encoding_multiwindow_stats.csv')
RET_STATS = os.path.join(OUT_DIR, 'retrieval_multiwindow_stats.csv')

FIGS = {
    'enc_dose':  os.path.join(OUT_DIR, 'encoding_multiwindow_dose_response.png'),
    'enc_combo': os.path.join(OUT_DIR, 'encoding_multiwindow_combinations.png'),
    'ret_dose':  os.path.join(OUT_DIR, 'retrieval_multiwindow_dose_response.png'),
    'ret_combo': os.path.join(OUT_DIR, 'retrieval_multiwindow_combinations.png'),
}


def fmt_p(p):
    if pd.isna(p):
        return ''
    if p < 0.001:
        return '< .001'
    return f'= {p:.3f}'


def p_str(p):
    if isinstance(p, str):
        return p
    if pd.isna(p):
        return ''
    if p < .001:
        return '< .001'
    return f'{p:.3f}'.lstrip('0')


def sig_str(p):
    if isinstance(p, str):
        return ''
    if pd.isna(p):
        return ''
    if p < .001:
        return '***'
    elif p < .01:
        return '**'
    elif p < .05:
        return '*'
    return ''


def fmt_or(v):
    if pd.isna(v):
        return '—'
    return f'{v:.2f}'


def load_diagnostics(fn):
    path = os.path.join(OUT_DIR, fn)
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    return dict(zip(df['metric'], df['value']))


def load_lrt(fn):
    path = os.path.join(OUT_DIR, fn)
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def _fmt(v, fmt='.3f'):
    if pd.isna(v):
        return '--'
    return f'{v:{fmt}}'


class Report(FPDF):
    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=20)

    def header(self):
        self.set_font('Helvetica', 'I', 8)
        self.cell(0, 5, 'IED Multi-Window Dose-Response MLM', align='R')
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

    def italic(self, text):
        self.set_font('Times', 'I', 10.5)
        self.multi_cell(0, 5.2, text)
        self.ln(1)


_TABLE_COUNTER = {'n': 0}


def _hline(pdf: Report, width: float, thickness: float = 0.3):
    x = pdf.get_x()
    y = pdf.get_y()
    pdf.set_line_width(thickness)
    pdf.line(x, y, x + width, y)
    pdf.ln(0.8)


def model_table(pdf: Report, coefs: pd.DataFrame, model_label: str,
                title: str, note: str = ''):
    """Render an APA-style regression table for one model.

    APA conventions applied:
      - Table number (bold) and title (italic) above the table
      - Horizontal rules only: top, under header, bottom
      - No vertical rules, no internal shading
      - Column headers in plain (non-bold) font; statistical symbols italicized
      - Numbers right-aligned, labels left-aligned
      - Italic note beneath the table
    """
    sub = coefs[coefs['model'] == model_label]
    if sub.empty:
        pdf.body(f'  [No results for {model_label}]')
        return

    _TABLE_COUNTER['n'] += 1
    tnum = _TABLE_COUNTER['n']

    # --- Table number + title (APA: "Table N" bold, then italic title) ---
    pdf.ln(2)
    pdf.set_font('Times', 'B', 11)
    pdf.cell(0, 5.5, f'Table {tnum}', new_x="LMARGIN", new_y="NEXT")
    pdf.set_font('Times', 'I', 11)
    pdf.multi_cell(0, 5.5, title)
    pdf.ln(1)

    # --- Column layout ---
    # APA: labels left-aligned, numerics right-aligned.
    columns = [
        ('term',   'Predictor', 42, 'L', 'L'),
        ('or',     'OR',        14, 'R', 'R'),
        ('se',     'SE(B)',     14, 'R', 'R'),
        ('ci',     '95% CI',    34, 'C', 'C'),
        ('z',      'z',         14, 'R', 'R'),
        ('p',      'p',         22, 'R', 'R'),
    ]
    total_w = sum(c[2] for c in columns)

    # Top rule
    _hline(pdf, total_w, thickness=0.5)

    # Header row: plain font, statistical symbols italicized
    header_y = pdf.get_y()
    x_start = pdf.get_x()
    for _, label, w, _, halign in columns:
        pdf.set_font('Times', 'B', 10.5)
        pdf.cell(w, 5.8, label, align=halign)
    pdf.ln(5.8)

    # Rule under header
    _hline(pdf, total_w, thickness=0.3)
    pdf.ln(0.5)

    # Data rows
    pdf.set_font('Times', '', 10.5)
    for _, row in sub.iterrows():
        cells = {
            'term': str(row['term']),
            'or':   fmt_or(row['estimate']),
            'se':   f"{row['std.error']:.3f}" if pd.notna(row['std.error']) else '',
            'ci':   f"[{fmt_or(row['conf.low'])}, {fmt_or(row['conf.high'])}]",
            'z':    f"{row['statistic']:.2f}" if pd.notna(row['statistic']) else '',
            'p':    fmt_p(row['p.value']).lstrip('= ').strip(),
        }
        for key, _, w, _, halign in columns:
            pdf.cell(w, 5.2, cells[key], align=halign)
        pdf.ln(5.2)

    # Bottom rule
    _hline(pdf, total_w, thickness=0.5)
    pdf.ln(1)

    # Note (APA: italic "Note." prefix, rest in regular)
    if note:
        pdf.set_font('Times', 'I', 9.5)
        pdf.cell(pdf.get_string_width('Note. '), 4.8, 'Note.')
        pdf.set_font('Times', '', 9.5)
        pdf.multi_cell(0, 4.8, ' ' + note)
        pdf.ln(1)


def add_figure(pdf: Report, path: str, caption: str, width: float = 175):
    if not os.path.exists(path):
        pdf.italic(f'  [Figure not found: {os.path.basename(path)}]')
        return
    # Leave room for caption; if not enough, add page
    if pdf.get_y() > 200:
        pdf.add_page()
    pdf.image(path, w=width)
    pdf.set_font('Times', 'I', 9)
    pdf.multi_cell(0, 4.5, caption)
    pdf.ln(2)


def main():
    if not os.path.exists(ENC_COEFS) or not os.path.exists(RET_COEFS):
        raise SystemExit(
            'Missing MLM coef files. Run Rscript ied_multiwindow_mlm.R first.'
        )

    enc = pd.read_csv(ENC_COEFS)
    ret = pd.read_csv(RET_COEFS)

    pdf = Report()
    pdf.add_page()

    # ---- Title page / overview ----
    pdf.set_font('Helvetica', 'B', 17)
    pdf.cell(0, 10, 'IED Multi-Window Dose-Response MLM', new_x="LMARGIN", new_y="NEXT")
    pdf.set_font('Helvetica', '', 11)
    pdf.cell(0, 6, 'Does the number of IED timing windows on a trial predict memory?',
             new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    pdf.h1('Overview')
    pdf.body(
        'Each encoding trial has up to four timing windows in which an IED may occur '
        '(Before Image, During Image, After Image, During Stim); retrieval trials have '
        'two (Before Image, During Image). We sum the number of windows with at least '
        'one IED to obtain a per-trial dose variable (n_windows). Generalized linear '
        'mixed-effects models (GLMM, binomial) test whether n_windows predicts trial-level '
        'memory, with a random intercept for patient. All models were fit in R using '
        'lme4 / lmerTest with bobyqa optimizer. Coefficients are reported as odds ratios '
        '(OR) with 95% Wald CIs.'
    )

    pdf.h1('Sample')
    pdf.body(
        'Encoding: 802 trials from 33 patients (571 remembered, 231 forgotten). '
        'n_windows distribution: 1=426, 2=283, 3=80, 4=13.\n'
        'Retrieval: 145 trials from 18 patients (110 remembered, 35 forgotten). '
        'n_windows distribution: 0=1, 1=96, 2=48.'
    )

    # ---- Statistical Approach ----
    pdf.h1('Statistical Approach')
    pdf.body(
        'All models were fit as generalized linear mixed models (GLMMs) '
        'with a binomial family and logit link using lme4::glmer in R, '
        'with a bobyqa optimizer (max 200,000 iterations). Each model '
        'included a random intercept for patient (1 | patient_id) to '
        'account for repeated measures within patients.'
    )
    pdf.body(
        'Fixed effects are reported as odds ratios (OR) with 95% Wald '
        'confidence intervals, standard errors on the log-odds scale '
        'SE(B), and Wald z-statistics. Significance of individual '
        'coefficients was assessed via the Wald z-test. Model comparisons '
        'between nested models were performed via likelihood ratio tests '
        '(LRTs), reported as chi-squared with degrees of freedom and '
        'p-value.'
    )
    pdf.body(
        'The intraclass correlation coefficient (ICC) was computed using '
        'the latent-variable approach for binary GLMMs: '
        'ICC = sigma^2_u0 / (sigma^2_u0 + pi^2/3), where sigma^2_u0 is '
        'the patient random intercept variance and pi^2/3 (~3.29) is the '
        'level-1 residual variance implied by the logistic distribution '
        '(Goldstein et al., 2002; Snijders & Bosker, 2012).'
    )
    pdf.body(
        'Effect sizes are reported as Nakagawa pseudo-R-squared '
        '(Nakagawa & Schielzeth, 2013; Nakagawa, Johnson, & Schielzeth, '
        '2017): marginal R-squared (R^2_m) represents variance explained '
        'by fixed effects only, and conditional R-squared (R^2_c) '
        'represents variance explained by fixed plus random effects, '
        'computed via the performance R package (Ludecke et al., 2021).'
    )

    # ---- Model Diagnostics Table ----
    pdf.h2('Model Diagnostics')
    diag_models = [
        ('Enc Null',         'mw_enc_null_diagnostics.csv'),
        ('E1: n_windows',    'mw_E1_diagnostics.csv'),
        ('E2: factor(n_win)','mw_E2_diagnostics.csv'),
        ('E3: + stim',       'mw_E3_diagnostics.csv'),
        ('E4: n_win x stim', 'mw_E4_diagnostics.csv'),
        ('Ret Null',         'mw_ret_null_diagnostics.csv'),
        ('R1: n_windows',    'mw_R1_diagnostics.csv'),
        ('R2: factor(n_win)','mw_R2_diagnostics.csv'),
        ('R3: timing windows','mw_R3_diagnostics.csv'),
    ]
    diag_headers = ['Model', 'ICC', 'sigma^2_u0', 'SD_u0', 'AIC', 'BIC',
                    'R^2_m', 'R^2_c']
    diag_widths = [32, 13, 16, 13, 16, 16, 13, 13]

    _TABLE_COUNTER['n'] += 1
    tnum_diag = _TABLE_COUNTER['n']
    pdf.ln(2)
    pdf.set_font('Times', 'B', 11)
    pdf.cell(0, 5.5, f'Table {tnum_diag}', new_x="LMARGIN", new_y="NEXT")
    pdf.set_font('Times', 'I', 11)
    pdf.multi_cell(0, 5.5,
                   'Model Diagnostics: Variance Components, ICC, and Effect Sizes')
    pdf.ln(1)

    total_diag_w = sum(diag_widths)
    _hline(pdf, total_diag_w, 0.5)

    pdf.set_font('Times', 'B', 9)
    for h, w in zip(diag_headers, diag_widths):
        pdf.cell(w, 5.8, h, align='C')
    pdf.ln(5.8)
    _hline(pdf, total_diag_w, 0.3)
    pdf.ln(0.5)

    pdf.set_font('Times', '', 9)
    for label, fn in diag_models:
        d = load_diagnostics(fn)
        if d is None:
            continue
        vals = [
            label,
            _fmt(d.get('ICC_latent')),
            _fmt(d.get('rand_intercept_var')),
            _fmt(d.get('rand_intercept_sd')),
            _fmt(d.get('AIC'), '.1f'),
            _fmt(d.get('BIC'), '.1f'),
            _fmt(d.get('R2_marginal')),
            _fmt(d.get('R2_conditional')),
        ]
        for v, w in zip(vals, diag_widths):
            align = 'L' if v == label else 'C'
            pdf.cell(w, 5.2, v, align=align)
        pdf.ln(5.2)

    _hline(pdf, total_diag_w, 0.5)
    pdf.ln(1)
    pdf.set_font('Times', 'I', 9)
    pdf.cell(pdf.get_string_width('Note. '), 4.8, 'Note.')
    pdf.set_font('Times', '', 9)
    pdf.multi_cell(0, 4.8,
        ' ICC = latent-variable ICC for binary GLMM. '
        'sigma^2_u0 = patient random intercept variance. '
        'R^2_m = Nakagawa marginal (fixed effects only). '
        'R^2_c = Nakagawa conditional (fixed + random).')
    pdf.ln(2)

    # ---- LRT Table ----
    pdf.h2('Model Comparisons (Likelihood Ratio Tests)')
    lrt_comparisons = [
        ('E1 vs. Null',  'lrt_mw_E1_vs_null.csv'),
        ('E2 vs. Null',  'lrt_mw_E2_vs_null.csv'),
        ('E4 vs. E3',    'lrt_mw_E4_vs_E3.csv'),
        ('R1 vs. Null',  'lrt_mw_R1_vs_null.csv'),
        ('R2 vs. Null',  'lrt_mw_R2_vs_null.csv'),
        ('R3 vs. Null',  'lrt_mw_R3_vs_null.csv'),
    ]
    lrt_headers = ['Comparison', 'chi^2', 'df', 'p', 'dAIC', 'dBIC']
    lrt_widths = [30, 18, 14, 22, 22, 22]

    _TABLE_COUNTER['n'] += 1
    tnum_lrt = _TABLE_COUNTER['n']
    pdf.ln(2)
    pdf.set_font('Times', 'B', 11)
    pdf.cell(0, 5.5, f'Table {tnum_lrt}', new_x="LMARGIN", new_y="NEXT")
    pdf.set_font('Times', 'I', 11)
    pdf.multi_cell(0, 5.5, 'Likelihood Ratio Tests')
    pdf.ln(1)

    total_lrt_w = sum(lrt_widths)
    _hline(pdf, total_lrt_w, 0.5)

    pdf.set_font('Times', 'B', 9)
    for h, w in zip(lrt_headers, lrt_widths):
        pdf.cell(w, 5.8, h, align='C')
    pdf.ln(5.8)
    _hline(pdf, total_lrt_w, 0.3)
    pdf.ln(0.5)

    pdf.set_font('Times', '', 9)
    for label, fn in lrt_comparisons:
        lrt = load_lrt(fn)
        if lrt is None:
            continue
        full_row = lrt[lrt['model'] == 'Full'].iloc[0]
        red_row = lrt[lrt['model'] == 'Reduced'].iloc[0]
        chisq = full_row['Chisq']
        df_val = full_row['Df']
        p_val = np.nan
        for col in lrt.columns:
            if 'Chisq' in col and 'Pr' in col:
                p_val = full_row[col]
                break
        d_aic = full_row['AIC'] - red_row['AIC']
        d_bic = full_row['BIC'] - red_row['BIC']
        vals = [
            label,
            f'{chisq:.2f}',
            f'{int(df_val)}',
            f'{p_str(p_val)}{sig_str(p_val)}',
            f'{d_aic:+.1f}',
            f'{d_bic:+.1f}',
        ]
        for v, w in zip(vals, lrt_widths):
            align = 'L' if v == label else 'C'
            pdf.cell(w, 5.2, v, align=align)
        pdf.ln(5.2)

    _hline(pdf, total_lrt_w, 0.5)
    pdf.ln(1)
    pdf.set_font('Times', 'I', 9)
    pdf.cell(pdf.get_string_width('Note. '), 4.8, 'Note.')
    pdf.set_font('Times', '', 9)
    pdf.multi_cell(0, 4.8,
        ' chi^2 = likelihood ratio chi-squared statistic. '
        'dAIC and dBIC = change from reduced to full model '
        '(negative favors the more complex model). '
        'E1 vs. Null tests continuous dose effect. '
        'E2 vs. Null tests omnibus factor dose effect. '
        'E4 vs. E3 tests dose x stim interaction. '
        'R1/R2 vs. Null test retrieval dose effects. '
        'R3 vs. Null tests retrieval timing window effects.')
    pdf.ln(2)

    # ---- Encoding ----
    pdf.add_page()
    pdf.h1('Encoding Phase')
    pdf.body(
        'Four models were fit on encoding trials. E1 treats n_windows as a continuous '
        'linear predictor (interpreting the OR as the multiplicative change in odds of '
        'remembering per additional window with an IED). E2 treats n_windows as a factor '
        'with 1 window as the reference level, and is compared to an intercept-only model '
        'via likelihood-ratio test to produce an omnibus test across levels. E3 adds '
        'stimulation condition (S vs NS) as a covariate. E4 adds the n_windows x stim '
        'interaction.'
    )

    model_table(
        pdf, enc, 'E1: n_windows (continuous)',
        title='Model E1: Memory as a function of IED-window dose (encoding).',
        note=('Binomial GLMM: memory ~ n_windows + (1 | patient_id). '
              'N = 802 trials, 33 patients. OR = odds ratio; SE(B) = standard error '
              'of log-odds coefficient; z = Wald z-statistic; CI = confidence interval. '
              'Each additional window with an IED multiplies the odds of remembering '
              'by 0.73 (27% reduction per window). *p < .05. **p < .01. ***p < .001.')
    )

    model_table(
        pdf, enc, 'E2: factor(n_windows)',
        title='Model E2: Memory by number of IED windows as a factor (encoding).',
        note=('Binomial GLMM: memory ~ factor(n_windows) + (1 | patient_id), '
              '1-window trials as the reference. SE(B) = standard error of log-odds '
              'coefficient; z = Wald z-statistic. Omnibus likelihood-ratio test vs. '
              'intercept-only model: see LRT table above. '
              'N = 802 trials, 33 patients. *p < .05. **p < .01. ***p < .001.')
    )

    model_table(
        pdf, enc, 'E3: n_windows + stim',
        title='Model E3: Dose effect controlling for stimulation (encoding).',
        note=('Binomial GLMM: memory ~ n_windows + stim + (1 | patient_id). '
              'Stim coded S vs. NS (reference). SE(B) = standard error of log-odds '
              'coefficient; z = Wald z-statistic. N = 802 trials, 33 patients. '
              '*p < .05. **p < .01. ***p < .001.')
    )

    model_table(
        pdf, enc, 'E4: n_windows * stim',
        title='Model E4: Dose by stimulation interaction (encoding).',
        note=('Binomial GLMM: memory ~ n_windows * stim + (1 | patient_id). '
              'SE(B) = standard error of log-odds coefficient; z = Wald z-statistic. '
              'N = 802 trials, 33 patients. The interaction is not significant, '
              'indicating the dose effect does not differ between stim and no-stim '
              'trials. *p < .05. **p < .01. ***p < .001.')
    )

    # Figures
    pdf.add_page()
    pdf.h2('Figure E1. Encoding dose-response')
    add_figure(pdf, FIGS['enc_dose'],
               'Percent of trials remembered as a function of the number of timing '
               'windows containing an IED. Descriptive: Spearman rho = -0.096, p = .007; '
               'chi-squared = 11.05, p = .011.')

    pdf.h2('Figure E2. Encoding IED timing combinations')
    add_figure(pdf, FIGS['enc_combo'],
               'Percent remembered for each specific combination of timing windows with '
               'IEDs (combinations with at least 5 trials). Bars are colored by number of '
               'windows spanned.')

    # ---- Retrieval ----
    pdf.add_page()
    pdf.h1('Retrieval Phase')
    pdf.body(
        'Three models were fit on retrieval trials. R1 treats n_windows as a continuous '
        'predictor; R2 treats it as a factor and is compared against an intercept-only '
        'model via LRT. R3 enters the two retrieval timing windows (Before Image and '
        'During Image) as separate binary predictors, paralleling the encoding window '
        'analysis. Note the restricted dose range (0-2 windows) and smaller sample '
        '(n = 145 trials from 18 patients), which limits power relative to encoding.'
    )

    model_table(
        pdf, ret, 'R1: n_windows (continuous)',
        title='Model R1: Memory as a function of IED-window dose (retrieval).',
        note=('Binomial GLMM: memory ~ n_windows + (1 | patient_id). '
              'SE(B) = standard error of log-odds coefficient; z = Wald z-statistic. '
              'N = 145 trials, 18 patients. Point estimate is in the same direction as '
              'encoding (OR = 0.67 per additional window) but does not reach '
              'significance. *p < .05. **p < .01. ***p < .001.')
    )

    # Omnibus for retrieval - reported from LRT in the R script output
    pdf.h2('R2: factor(n_windows) - omnibus LRT')
    pdf.set_font('Times', '', 11)
    pdf.multi_cell(0, 5.5,
                   'Likelihood-ratio test (factor(n_windows) vs intercept-only): '
                   'Chi-sq(2) = 4.15, p = .125. (Coefficient estimates for R2 are '
                   'unstable due to the single 0-window trial and are omitted here; '
                   'the LRT is the interpretable test.)')
    pdf.ln(2)

    model_table(
        pdf, ret, 'R3: timing windows',
        title='Model R3: Retrieval IED timing windows predicting memory.',
        note=('Binomial GLMM: memory ~ BeforeImgITI + DuringImgITI + (1 | patient_id). '
              'SE(B) = standard error of log-odds coefficient; z = Wald z-statistic. '
              'N = 145 trials, 18 patients. Neither timing window significantly '
              'predicted memory. *p < .05. **p < .01. ***p < .001.')
    )

    pdf.add_page()
    pdf.h2('Figure R1. Retrieval dose-response')
    add_figure(pdf, FIGS['ret_dose'],
               'Retrieval-phase dose-response. Trend is in the same direction as '
               'encoding but does not reach significance in the MLM.')
    pdf.h2('Figure R2. Retrieval IED timing combinations')
    add_figure(pdf, FIGS['ret_combo'],
               'Percent remembered for each retrieval IED timing combination '
               '(combinations with at least 5 trials).')

    # ---- Summary ----
    pdf.add_page()
    pdf.h1('Summary')
    pdf.body(
        'Encoding: the number of timing windows containing an IED on a trial '
        'significantly reduces the odds of that trial being subsequently remembered. '
        'Each additional window reduces odds by about 27% (OR = 0.73, p = .007 continuous; '
        'LRT chi-squared(3) = 9.77, p = .021 omnibus). The effect is not driven by '
        'stimulation (p = .78) and does not interact with stim (p = .56).'
    )
    pdf.body(
        'Retrieval: trend is in the same direction (OR = 0.67 per window) but is not '
        'statistically significant (p = .37 continuous; LRT chi-squared(2) = 4.15, '
        'p = .125). The restricted dose range (0-2 windows) and smaller sample size limit '
        'sensitivity.'
    )
    pdf.body(
        'Together, these results support the interpretation that broader temporal coverage '
        'of IEDs within a trial - not just their presence - is associated with worse '
        'encoding memory, consistent with a dose-response interpretation.'
    )

    pdf.output(PDF_PATH)
    print(f'Saved {PDF_PATH}')


if __name__ == '__main__':
    main()
