#!/usr/bin/env python
"""
Build the PDF report for the retrieval IED timing x memory GLMMs.
=================================================================
Two binomial GLMMs (patient random intercept) at retrieval, on retrieved old
items (remembered / forgotten):

  Model A - timing main effects:
      memory ~ BeforeImg + DuringImg + (1 | patient)
  Model B - prior-stimulation interaction:
      memory ~ BeforeImg*prior_stim + DuringImg*prior_stim + (1 | patient)

Benjamini-Hochberg FDR is applied within two separate families:
  - Family 1 = the two timing main effects (Model A).
  - Family 2 = the two prior-stim interactions (Model B).

Inputs (from retrieval_timing_memory_glmm.R, in ied_timing_memory/):
  retrieval_timing_memory_trials.csv
  retrieval_timing_maineffects_coefs.csv
  retrieval_timing_interaction_coefs.csv
  retrieval_timing_maineffects_diagnostics.csv
  retrieval_timing_interaction_diagnostics.csv
  retrieval_timing_lrt.csv

Output:
  ied_timing_memory/Retrieval_Timing_Memory_GLMM_Report.pdf
"""

import os
import numpy as np
import pandas as pd
from fpdf import FPDF

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, 'ied_timing_memory')
PDF_PATH = os.path.join(OUT_DIR, 'Retrieval_Timing_Memory_GLMM_Report.pdf')

TRIALS = os.path.join(OUT_DIR, 'retrieval_timing_memory_trials.csv')
MAIN_COEFS = os.path.join(OUT_DIR, 'retrieval_timing_maineffects_coefs.csv')
INT_COEFS = os.path.join(OUT_DIR, 'retrieval_timing_interaction_coefs.csv')
MAIN_DIAG = os.path.join(OUT_DIR, 'retrieval_timing_maineffects_diagnostics.csv')
INT_DIAG = os.path.join(OUT_DIR, 'retrieval_timing_interaction_diagnostics.csv')
LRT = os.path.join(OUT_DIR, 'retrieval_timing_lrt.csv')

TERM_LABELS = {
    '(Intercept)': 'Intercept',
    'BeforeImgY': 'Before-image IED',
    'DuringImgY': 'During-image IED',
    'prior_stimS': 'Prior stim',
    'BeforeImgY:prior_stimS': 'Before-image IED x Prior stim',
    'prior_stimS:DuringImgY': 'During-image IED x Prior stim',
    'DuringImgY:prior_stimS': 'During-image IED x Prior stim',
}


def p_str(p):
    if pd.isna(p):
        return '--'
    if p < .001:
        return '< .001'
    return f'{p:.3f}'.lstrip('0')


def sig_str(p):
    if pd.isna(p):
        return ''
    if p < .001:
        return '***'
    if p < .01:
        return '**'
    if p < .05:
        return '*'
    return ''


def fmt(v, nd=2):
    if pd.isna(v):
        return '--'
    return f'{v:.{nd}f}'


def diag(path):
    if not os.path.exists(path):
        return {}
    d = pd.read_csv(path)
    return dict(zip(d['metric'], d['value']))


class Report(FPDF):
    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=18)

    def header(self):
        self.set_font('Helvetica', 'I', 8)
        self.cell(0, 5, 'Retrieval IED Timing x Memory GLMM', align='R')
        self.ln(7)

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
        self.set_font('Times', 'I', 10), self.set_text_color(90, 90, 90)
        self.multi_cell(0, 5.0, text)
        self.set_text_color(0, 0, 0)
        self.ln(1)


_TC = {'n': 0}


def _hline(pdf, width, thickness=0.3):
    x, y = pdf.get_x(), pdf.get_y()
    pdf.set_line_width(thickness)
    pdf.line(x, y, x + width, y)
    pdf.ln(0.8)


def model_table(pdf, coefs, title, note):
    """APA-style GLMM table: Predictor, OR, 95% CI, z, p, q(FDR)."""
    _TC['n'] += 1
    pdf.ln(2)
    pdf.set_font('Times', 'B', 11)
    pdf.cell(0, 5.5, f'Table {_TC["n"]}', new_x="LMARGIN", new_y="NEXT")
    pdf.set_font('Times', 'I', 11)
    pdf.multi_cell(0, 5.5, title)
    pdf.ln(1)

    cols = [('Predictor', 48, 'C'), ('OR', 16, 'C'), ('95% CI', 32, 'C'),
            ('z', 14, 'C'), ('p', 16, 'C'), ('FDR q', 18, 'C')]
    total_w = sum(c[1] for c in cols)

    pdf.set_font('Times', 'B', 10)
    _hline(pdf, total_w, 0.4)
    for name, w, align in cols:
        pdf.cell(w, 6, name, align=align)
    pdf.ln(6)
    _hline(pdf, total_w, 0.3)

    for _, r in coefs.iterrows():
        label = TERM_LABELS.get(r['term'], r['term'])
        ci = f"[{fmt(r['conf.low'])}, {fmt(r['conf.high'])}]"
        z = r.get('statistic', np.nan)
        q = r.get('q_fdr', np.nan)
        vals = [(label, 48, 'L'), (fmt(r['estimate']), 16, 'C'),
                (ci, 32, 'C'), (fmt(z, 2), 14, 'C'),
                (p_str(r['p.value']), 16, 'C'),
                ('--' if pd.isna(q) else f'{p_str(q)}{sig_str(q)}', 18, 'C')]
        pdf.set_font('Times', '', 10)
        for txt, w, align in vals:
            pdf.cell(w, 5.6, txt, align=align)
        pdf.ln(5.6)

    _hline(pdf, total_w, 0.4)
    pdf.ln(1)
    pdf.set_font('Times', 'I', 9)
    pdf.set_text_color(90, 90, 90)
    pdf.multi_cell(0, 4.6, note)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(2)


def lrt_table(pdf, lrt, title):
    _TC['n'] += 1
    pdf.ln(2)
    pdf.set_font('Times', 'B', 11)
    pdf.cell(0, 5.5, f'Table {_TC["n"]}', new_x="LMARGIN", new_y="NEXT")
    pdf.set_font('Times', 'I', 11)
    pdf.multi_cell(0, 5.5, title)
    pdf.ln(1)

    cols = [('Likelihood-ratio test', 92, 'C'), ('chi2', 22, 'C'),
            ('df', 16, 'C'), ('p', 22, 'C')]
    total_w = sum(c[1] for c in cols)
    pdf.set_font('Times', 'B', 10)
    _hline(pdf, total_w, 0.4)
    for name, w, align in cols:
        pdf.cell(w, 6, name, align=align)
    pdf.ln(6)
    _hline(pdf, total_w, 0.3)
    pdf.set_font('Times', '', 10)
    for _, r in lrt.iterrows():
        pdf.cell(92, 5.6, r['test'], align='L')
        pdf.cell(22, 5.6, fmt(r['Chisq'], 2), align='C')
        pdf.cell(16, 5.6, f"{int(r['Df'])}", align='C')
        pdf.cell(22, 5.6, p_str(r['p.value']), align='C')
        pdf.ln(5.6)
    _hline(pdf, total_w, 0.4)
    pdf.ln(3)


def descriptives_table(pdf, dat):
    _TC['n'] += 1
    pdf.ln(2)
    pdf.set_font('Times', 'B', 11)
    pdf.cell(0, 5.5, f'Table {_TC["n"]}', new_x="LMARGIN", new_y="NEXT")
    pdf.set_font('Times', 'I', 11)
    pdf.multi_cell(0, 5.5, 'Retrieved old-item trials: cell counts and % remembered.')
    pdf.ln(1)

    def pct(sub):
        n = len(sub)
        rem = int((sub['memory'] == 1).sum())
        return rem, n - rem, n, (100 * rem / n if n else 0)

    rows = [('All old items', dat)]
    rows += [(f'Prior stim = {s}', dat[dat['prior_stim'] == s]) for s in ['NS', 'S']]
    rows += [('Before-image IED present', dat[dat['BeforeImg'] == 'Y']),
             ('Before-image IED absent',  dat[dat['BeforeImg'] == 'N']),
             ('During-image IED present', dat[dat['DuringImg'] == 'Y']),
             ('During-image IED absent',  dat[dat['DuringImg'] == 'N'])]

    cols = [('Group', 78, 'C'), ('Remembered', 26, 'C'), ('Forgotten', 24, 'C'),
            ('N', 16, 'C'), ('% Rem.', 22, 'C')]
    total_w = sum(c[1] for c in cols)
    pdf.set_font('Times', 'B', 10)
    _hline(pdf, total_w, 0.4)
    for name, w, align in cols:
        pdf.cell(w, 6, name, align=align)
    pdf.ln(6)
    _hline(pdf, total_w, 0.3)
    for label, sub in rows:
        rem, forg, n, pc = pct(sub)
        pdf.set_font('Times', 'B' if label == 'All old items' else '', 10)
        pdf.cell(78, 5.6, label, align='L')
        pdf.cell(26, 5.6, str(rem), align='C')
        pdf.cell(24, 5.6, str(forg), align='C')
        pdf.cell(16, 5.6, str(n), align='C')
        pdf.cell(22, 5.6, f'{pc:.1f}', align='C')
        pdf.ln(5.6)
    _hline(pdf, total_w, 0.4)
    pdf.ln(3)


def main():
    dat = pd.read_csv(TRIALS)
    main_c = pd.read_csv(MAIN_COEFS)
    int_c = pd.read_csv(INT_COEFS)
    lrt = pd.read_csv(LRT)
    md = diag(MAIN_DIAG)
    idg = diag(INT_DIAG)

    n_trials = len(dat)
    n_pat = dat['Patient'].nunique()
    n_rem = int((dat['memory'] == 1).sum())
    n_forg = n_trials - n_rem

    pdf = Report()
    pdf.add_page()

    pdf.h1('Retrieval IED Timing and Subsequent Memory')
    pdf.set_font('Times', '', 11)
    pdf.body(
        'Two binomial generalized linear mixed models (GLMMs) tested whether the '
        'timing of an interictal epileptiform discharge (IED) during a retrieval '
        'trial was associated with whether the item was remembered, and whether '
        'any such association depended on whether that item had been stimulated '
        'when it was studied (prior stimulation). Each model included a random '
        'intercept for patient. The outcome was memory (remembered = 1, '
        'forgotten = 0) for retrieved old items only (foils/new items excluded). '
        'Two retrieval timing windows were modeled: an IED before image onset '
        '(Before-image, the pre-stimulus ITI) and an IED while the image was on '
        'screen (During-image). Each window was coded present (Y) vs. absent (N), '
        'collapsed across channels within a trial. Prior stimulation was coded S '
        'vs. NS (reference = NS), recovered per item from the raw phase-3 test '
        'files (BLAES: stimulation flag; AMME: stim/nostim trial type).')

    pdf.body(
        f'Sample: {n_trials} retrieved old-item trials from {n_pat} patients '
        f'({n_rem} remembered, {n_forg} forgotten). Before-image IED present on '
        f'{int((dat["BeforeImg"]=="Y").sum())} trials; During-image IED present on '
        f'{int((dat["DuringImg"]=="Y").sum())} trials; prior-stim items: '
        f'{int((dat["prior_stim"]=="S").sum())} S vs '
        f'{int((dat["prior_stim"]=="NS").sum())} NS.')

    pdf.h2('FDR correction')
    pdf.body(
        'Benjamini-Hochberg FDR was applied within two separate families, as '
        'pre-specified: (Family 1) the two timing main effects from Model A '
        '[Before-image IED, During-image IED]; and (Family 2) the two prior-stim '
        'interaction terms from Model B [Before-image x stim, During-image x '
        'stim]. q-values are reported only for the members of each family; '
        'intercepts and lower-order terms in Model B are not part of either '
        'family and are shown uncorrected for reference.')

    descriptives_table(pdf, dat)

    pdf.h2('Model A - Timing main effects')
    pdf.body(
        'memory ~ Before-image IED + During-image IED + (1 | patient). '
        f'Random-intercept SD = {fmt(md.get("rand_intercept_sd"), 2)}, '
        f'ICC = {fmt(md.get("ICC_latent"), 3)}; marginal R2 = '
        f'{fmt(md.get("R2_marginal"), 3)}, conditional R2 = '
        f'{fmt(md.get("R2_conditional"), 3)}.')
    model_table(
        pdf, main_c,
        'GLMM of subsequent memory on retrieval IED timing windows (Model A). '
        'Odds ratios > 1 indicate higher odds of remembering.',
        'OR = odds ratio; CI = profile/Wald confidence interval; FDR q = '
        'Benjamini-Hochberg FDR-corrected p-value within the timing-main-effects '
        'family (2 tests). Reference levels: IED absent (N). '
        'Stars denote FDR significance: *q < .05. **q < .01. ***q < .001.')

    pdf.h2('Model B - Prior-stimulation interaction')
    pdf.body(
        'memory ~ Before-image IED * prior stim + During-image IED * prior stim '
        f'+ (1 | patient). Random-intercept SD = '
        f'{fmt(idg.get("rand_intercept_sd"), 2)}, ICC = '
        f'{fmt(idg.get("ICC_latent"), 3)}; marginal R2 = '
        f'{fmt(idg.get("R2_marginal"), 3)}, conditional R2 = '
        f'{fmt(idg.get("R2_conditional"), 3)}.')
    model_table(
        pdf, int_c,
        'GLMM of subsequent memory on retrieval IED timing windows, prior '
        'stimulation, and their interactions (Model B).',
        'FDR q shown only for the two interaction terms (Family 2, 2 tests); '
        'main effects are not part of that family (--). '
        'Reference levels: IED absent (N), prior stim = NS. The During-image x '
        'prior-stim term indexes how the during-image IED effect differs for '
        'previously stimulated vs. non-stimulated items. '
        'Stars denote FDR significance: *q < .05. **q < .01. ***q < .001.')

    pdf.h2('Omnibus likelihood-ratio tests')
    lrt_table(pdf, lrt,
              'Joint LRTs: timing main effects vs. null model, and the two '
              'prior-stim interactions vs. the additive model.')

    pdf.set_font('Times', 'I', 9)
    pdf.set_text_color(90, 90, 90)
    pdf.multi_cell(0, 4.6,
                   'Models fit with lme4::glmer (bobyqa, binomial). Terms are '
                   'starred when FDR-significant (q < .05) within their family. With only 16 '
                   'During-image-absent trials, the During-image estimates and '
                   'their interaction rest on a sparse cell and should be read with '
                   'caution.')
    pdf.set_text_color(0, 0, 0)

    pdf.output(PDF_PATH)
    print(f'Saved {PDF_PATH}')


if __name__ == '__main__':
    main()
