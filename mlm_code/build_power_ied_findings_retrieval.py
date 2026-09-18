#!/usr/bin/env python
"""
Build narrative PDF report: Power x IED Findings at RETRIEVAL.

Retrieval mirror of build_power_ied_findings.py. Differences:
  - IED timing windows are BeforeImg + DuringImg only (no during-stim /
    after-image at retrieval).
  - The stim variable is prior_stim (stimulated at STUDY, S vs NS).
  - No stim-only sensitivity model and no window-specification robustness
    (those addressed the encoding during-stim confound, absent at retrieval).

Organized into four self-contained region x band sections (Amygdala-Theta,
Amygdala-Slow Gamma, HPC-Theta, HPC-Slow Gamma); no table mixes regions or
frequencies. FDR families (Benjamini-Hochberg) per region x band x type:
  - Main effects (T):  power, ied_before_image, ied_during_image
  - Interaction (TI):  power x before-image, power x during-image
  - Stimulation (AI):  power x prior_stim
TF (focused) and A (power-alone) models are supplementary (q = '--').
"""

import os
import numpy as np
import pandas as pd
from fpdf import FPDF

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
OUTPUT_DIR = os.path.join(REPO_ROOT, 'IED', 'ied_timing_memory')

REPORT_TITLE = 'Neural Power and IED Effects on Retrieval Memory'

BANDS = ['theta', 'slow_gamma']
BAND_LABELS = {'theta': 'Theta', 'slow_gamma': 'Slow Gamma'}
REGIONS = [('BLA', 'Amygdala'), ('HPC', 'HPC')]
SECTIONS = [(reg, rl, b) for reg, rl in REGIONS for b in BANDS]


class APAReport(FPDF):
    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=25)

    def header(self):
        self.set_font('Helvetica', 'I', 8)
        self.cell(0, 5, REPORT_TITLE, align='R')
        self.ln(8)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', align='C')

    def section_title(self, title):
        self.set_font('Helvetica', 'B', 13)
        self.ln(4)
        self.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def subsection_title(self, title):
        self.set_font('Helvetica', 'B', 11)
        self.ln(2)
        self.cell(0, 7, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def body_text(self, text):
        self.set_font('Times', '', 11)
        self.multi_cell(0, 5.5, text)
        self.ln(1)

    def apa_table(self, title, headers, rows, col_widths=None, note=None):
        self.ln(3)
        self.set_font('Times', 'I', 10)
        self.multi_cell(0, 5, title)
        self.ln(1)
        if col_widths is None:
            avail = self.w - self.l_margin - self.r_margin
            col_widths = [avail / len(headers)] * len(headers)
        x_start = self.get_x()
        total_w = sum(col_widths)
        self.set_line_width(0.5)
        self.line(x_start, self.get_y(), x_start + total_w, self.get_y())
        self.ln(1)
        self.set_font('Times', 'B', 9)
        for i, h in enumerate(headers):
            self.cell(col_widths[i], 5, str(h), align='C')
        self.ln()
        self.set_line_width(0.3)
        self.line(x_start, self.get_y(), x_start + total_w, self.get_y())
        self.ln(1)
        self.set_font('Times', '', 9)
        for row in rows:
            if self.get_y() > self.h - 35:
                self.set_line_width(0.5)
                self.line(x_start, self.get_y(), x_start + total_w, self.get_y())
                self.add_page()
                self.set_font('Times', 'I', 10)
                self.multi_cell(0, 5, title + ' (continued)')
                self.ln(1)
                self.set_line_width(0.5)
                x_start = self.get_x()
                self.line(x_start, self.get_y(), x_start + total_w, self.get_y())
                self.ln(1)
                self.set_font('Times', 'B', 9)
                for i, h in enumerate(headers):
                    self.cell(col_widths[i], 5, str(h), align='C')
                self.ln()
                self.set_line_width(0.3)
                self.line(x_start, self.get_y(), x_start + total_w, self.get_y())
                self.ln(1)
                self.set_font('Times', '', 9)
            for i, val in enumerate(row):
                align = 'L' if i == 0 else 'C'
                self.cell(col_widths[i], 5, str(val), align=align)
            self.ln()
        self.set_line_width(0.5)
        self.line(x_start, self.get_y(), x_start + total_w, self.get_y())
        self.ln(1)
        if note:
            self.set_font('Times', 'I', 8)
            self.multi_cell(0, 4, f'Note. {note}')
            self.ln(2)


# ── Formatting helpers ─────────────────────────────────────────────────────
def p_str(p):
    if isinstance(p, str): return p
    if p < .001: return '< .001'
    return f'{p:.3f}'.lstrip('0')

def sig_str(v):
    if isinstance(v, str): return ''
    if v < .001: return '***'
    elif v < .01: return '**'
    elif v < .05: return '*'
    elif v < .1: return '+'
    return ''

def or_str(v): return f'{v:.2f}'
def ci_str(lo, hi): return f'[{lo:.2f}, {hi:.2f}]'
def z_str(v): return f'{v:.2f}'

def cell_q(q):
    if q is None or (isinstance(q, str) and q == '--'):
        return '--'
    return p_str(q) + sig_str(q)

def p_inline(s):
    return f'p {s}' if s.startswith('<') else f'p = {s}'

def q_inline(q):
    s = p_str(q)
    return f'q {s}' if s.startswith('<') else f'q = {s}'

def fmt_r2(v): return '--' if pd.isna(v) else f'{v:.3f}'
def fmt_icc(v): return '--' if pd.isna(v) else f'{v:.3f}'
def fmt_aic(v): return '--' if pd.isna(v) else f'{v:.1f}'
def fmt_var(v): return '--' if pd.isna(v) else f'{v:.3f}'


TERM_LABELS = {
    '(Intercept)': 'Intercept',
    'pow_theta_z': 'Theta Power (z)',
    'pow_slow_gamma_z': 'Slow Gamma Power (z)',
    'stim': 'Prior Stim (S)',
    'ied_before_image': 'IED Before-Image Window',
    'ied_during_image': 'IED During-Image Window',
}
for _band in BANDS:
    _bn = {'theta': 'Theta', 'slow_gamma': 'SG'}[_band]
    _z = f'pow_{_band}_z'
    TERM_LABELS[f'{_z}:ied_before_image'] = f'{_bn} Pow x IED Before-Img'
    TERM_LABELS[f'ied_before_image:{_z}'] = f'{_bn} Pow x IED Before-Img'
    TERM_LABELS[f'{_z}:ied_during_image'] = f'{_bn} Pow x IED During-Img'
    TERM_LABELS[f'ied_during_image:{_z}'] = f'{_bn} Pow x IED During-Img'
    TERM_LABELS[f'{_z}:stim'] = f'{_bn} Pow x Prior-Stim'
    TERM_LABELS[f'stim:{_z}'] = f'{_bn} Pow x Prior-Stim'

FULL_HEADERS = ['Predictor', 'OR', '95% CI', 'z', 'p', 'FDR q']
FULL_WIDTHS = [56, 16, 34, 15, 22, 22]
DIAG_HEADERS = ['Model', 'ICC', 'sigma^2_u0', 'SD_u0', 'AIC', 'BIC', 'R^2_m', 'R^2_c']
DIAG_WIDTHS = [40, 14, 16, 14, 16, 16, 14, 14]


# ── Data loading ───────────────────────────────────────────────────────────
def load_or(filename):
    path = os.path.join(OUTPUT_DIR, filename)
    return pd.read_csv(path) if os.path.exists(path) else None

def load_diagnostics(filename):
    path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(path): return None
    df = pd.read_csv(path)
    return dict(zip(df['metric'], df['value']))

def load_lrt(filename):
    path = os.path.join(OUTPUT_DIR, filename)
    return pd.read_csv(path) if os.path.exists(path) else None


def term_variants(term):
    if ':' in term:
        a, b = term.split(':', 1)
        return [term, f'{b}:{a}']
    return [term]


def get_term_stats(df, term):
    if df is None:
        return None
    for t in term_variants(term):
        row = df[df['term'] == t]
        if len(row):
            r = row.iloc[0]
            return {'or_s': or_str(r['estimate']), 'z_s': z_str(r['statistic']),
                    'ci_s': ci_str(r['conf.low'], r['conf.high']),
                    'p_s': p_str(r['p.value'])}
    return None


def get_n(prefix, reg, b):
    d = load_diagnostics(f'powret_{prefix}_{reg}_{b}_diagnostics.csv')
    if not d:
        return None, None
    nobs, ngr = d.get('nobs'), d.get('ngrps')
    nobs = int(nobs) if nobs is not None and nobs == nobs else None
    ngr = int(ngr) if ngr is not None and ngr == ngr else None
    return nobs, ngr

def n_str(prefix, reg, b):
    nobs, ngr = get_n(prefix, reg, b)
    return '' if nobs is None else f'N = {nobs}, {ngr} patients'


# ── FDR families ───────────────────────────────────────────────────────────
def bh_qvalues(pvals):
    p = list(pvals)
    n = len(p)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: p[i])
    q = [0.0] * n
    prev = 1.0
    for rank in range(n - 1, -1, -1):
        i = order[rank]
        prev = min(prev, p[i] * n / (rank + 1))
        q[i] = min(prev, 1.0)
    return q


def build_q_map():
    Q = {}
    for reg, _ in REGIONS:
        for b in BANDS:
            z = f'pow_{b}_z'
            families = {
                'T': [z, 'ied_before_image', 'ied_during_image'],
                'TI': [f'{z}:ied_before_image', f'ied_before_image:{z}',
                       f'{z}:ied_during_image', f'ied_during_image:{z}'],
                'AI': [f'{z}:stim', f'stim:{z}'],
            }
            for prefix, cand in families.items():
                df = load_or(f'powret_{prefix}_{reg}_{b}_odds_ratios.csv')
                if df is None:
                    Q[(prefix, reg, b)] = {}
                    continue
                present = [t for t in cand if t in set(df['term'])]
                ps = [df.loc[df['term'] == t, 'p.value'].iloc[0] for t in present]
                Q[(prefix, reg, b)] = dict(zip(present, bh_qvalues(ps)))
    return Q


def make_q_lookup(Q):
    def get_q(prefix, reg, b, term):
        fam = Q.get((prefix, reg, b), {})
        for t in term_variants(term):
            if t in fam:
                return fam[t]
        return '--'
    return get_q


# ── Row / table builders ───────────────────────────────────────────────────
def make_or_rows(df, prefix, reg, b, get_q, skip_intercept=True):
    rows = []
    for _, r in df.iterrows():
        term = r['term']
        if skip_intercept and term == '(Intercept)':
            continue
        q = get_q(prefix, reg, b, term)
        rows.append([TERM_LABELS.get(term, term), or_str(r['estimate']),
                     ci_str(r['conf.low'], r['conf.high']), z_str(r['statistic']),
                     p_str(r['p.value']), cell_q(q)])
    return rows


def full_table(pdf, tn, prefix, reg, b, get_q, title, note):
    df = load_or(f'powret_{prefix}_{reg}_{b}_odds_ratios.csv')
    if df is None:
        return tn
    rows = make_or_rows(df, prefix, reg, b, get_q)
    pdf.apa_table(f'Table {tn}. {title}', FULL_HEADERS, rows, FULL_WIDTHS, note=note)
    return tn + 1


def diag_table(pdf, tn, reg, b):
    rows = []
    for prefix, label in [('T', 'Main effects (T)'), ('TI', 'Interaction (TI)')]:
        d = load_diagnostics(f'powret_{prefix}_{reg}_{b}_diagnostics.csv')
        if not d:
            continue
        rows.append([label, fmt_icc(d.get('ICC_latent')),
                     fmt_var(d.get('rand_intercept_var')),
                     fmt_var(d.get('rand_intercept_sd')), fmt_aic(d.get('AIC')),
                     fmt_aic(d.get('BIC')), fmt_r2(d.get('R2_marginal')),
                     fmt_r2(d.get('R2_conditional'))])
    if not rows:
        return tn
    pdf.apa_table(f'Table {tn}. Model Diagnostics (this region and band)',
                  DIAG_HEADERS, rows, DIAG_WIDTHS,
                  note='ICC = latent-variable ICC for binary GLMM. R^2_m / R^2_c '
                       '= Nakagawa marginal / conditional pseudo-R-squared.')
    return tn + 1


def lrt_note(reg, b):
    lrt = load_lrt(f'lrt_ret_TI_vs_T_{reg}_{b}.csv')
    if lrt is None or 'model' not in lrt.columns:
        return ''
    full_row = lrt[lrt['model'] == 'Full']
    if not len(full_row):
        return ''
    full_row = full_row.iloc[0]
    chisq, df_val = full_row['Chisq'], int(full_row['Df'])
    p_val = np.nan
    for col in lrt.columns:
        if 'Chisq' in col and 'Pr' in col:
            p_val = full_row[col]
            break
    return (f' LRT vs the main-effects (T) model: chi^2({df_val}) = '
            f'{chisq:.2f}, {p_inline(p_str(p_val))} (unadjusted).')


def phrase(prefix, reg, b, term, get_q):
    df = load_or(f'powret_{prefix}_{reg}_{b}_odds_ratios.csv')
    s = get_term_stats(df, term)
    if s is None:
        return '(n/a)'
    out = f"OR = {s['or_s']}, 95% CI {s['ci_s']}, {p_inline(s['p_s'])}"
    q = get_q(prefix, reg, b, term)
    if q != '--':
        out += f", {q_inline(q)}"
    return out


# ── Per-region x band section ──────────────────────────────────────────────
def render_section(pdf, tn, reg, reg_label, b, get_q):
    bl = BAND_LABELS[b]
    rl = reg_label.lower()
    z = f'pow_{b}_z'
    pdf.add_page()
    pdf.section_title(f'{reg_label} -- {bl}')
    pdf.body_text(
        f'This section reports {rl} {bl.lower()} power at retrieval only. The '
        f'amygdala and HPC, and theta and slow gamma, are analyzed in fully '
        f'separate models and are not compared to one another. Old-item IED '
        f'trials: {n_str("T", reg, b)}.'
    )

    pdf.subsection_title('Main Effects: Power + IED Timing Windows')
    tn = full_table(
        pdf, tn, 'T', reg, b, get_q,
        'Main Effects (Power + IED Before-Image + IED During-Image)',
        f'memory ~ {z} + ied_before_image + ied_during_image + (1|patient_id). '
        f'FDR q (Benjamini-Hochberg) within this three-term main-effects family.')
    pdf.body_text(
        f'IED before the probe image: {phrase("T", reg, b, "ied_before_image", get_q)}. '
        f'IED during the probe image: {phrase("T", reg, b, "ied_during_image", get_q)}. '
        f'Power main effect: {phrase("T", reg, b, z, get_q)}. '
        f'(OR < 1 = lower odds of remembering.)'
    )

    pdf.subsection_title('Power x IED Timing Interaction')
    tn = full_table(
        pdf, tn, 'TI', reg, b, get_q,
        'Power x IED Timing Interactions',
        f'memory ~ {z} * (ied_before_image + ied_during_image) + (1|patient_id). '
        f'FDR q applies to the two interaction terms; lower-order main effects '
        f'shown for context (q = "--").' + lrt_note(reg, b))
    pdf.body_text(
        f'Power x IED-before-image interaction: '
        f'{phrase("TI", reg, b, z + ":ied_before_image", get_q)}. '
        f'Power x IED-during-image interaction: '
        f'{phrase("TI", reg, b, z + ":ied_during_image", get_q)}.'
    )

    pdf.subsection_title('Model Diagnostics')
    tn = diag_table(pdf, tn, reg, b)

    pdf.subsection_title('Focused Model (Covariate-Adjusted)')
    tn = full_table(
        pdf, tn, 'TF', reg, b, get_q,
        'Focused Model: Both Windows + Prior-Stim Covariate',
        f'memory ~ {z} + ied_before_image + ied_during_image + stim + '
        f'(1|patient_id). Supplementary; not part of an FDR family (q = "--"). '
        f'Prior stim (S vs NS) is a covariate.')

    pdf.subsection_title('Stimulation: Power x Prior-Stim')
    tn = full_table(
        pdf, tn, 'AI', reg, b, get_q,
        f'Power x Prior-Stim Interaction ({n_str("AI", reg, b)})',
        f'memory ~ {z} * stim + (1|patient_id). All old-item IED trials. '
        f'prior_stim = stimulated at study (S vs NS). FDR q applies to the '
        f'power x prior-stim interaction (stimulation family).')
    pdf.body_text(
        f'Power x prior-stim interaction: {phrase("AI", reg, b, z + ":stim", get_q)}.'
    )

    pdf.subsection_title('Power Alone (All Retrieval Trials)')
    tn = full_table(
        pdf, tn, 'A', reg, b, get_q,
        f'Power Main Effect, All Trials ({n_str("A", reg, b)})',
        f'memory ~ {z} + stim + (1|patient_id). All retrieval power trials (not '
        f'restricted to IED trials). Supplementary; not FDR-corrected (q = "--").')
    return tn


# ── Report ─────────────────────────────────────────────────────────────────
def build_report():
    Q = build_q_map()
    get_q = make_q_lookup(Q)

    pdf = APAReport()
    pdf.add_page()
    pdf.set_font('Helvetica', 'B', 16)
    pdf.cell(0, 10, REPORT_TITLE, new_x="LMARGIN", new_y="NEXT", align='C')
    pdf.set_font('Helvetica', '', 11)
    pdf.cell(0, 6, 'Retrieval Phase - Trial-Level Analysis',
             new_x="LMARGIN", new_y="NEXT", align='C')
    pdf.ln(8)

    merged = pd.read_csv(os.path.join(OUTPUT_DIR, 'retrieval_power_ied_merged.csv'))

    tn = 1
    sample_rows = []
    during_pct = {}
    for reg, reg_label in REGIONS:
        sub = merged[merged['region'] == reg]
        pc = sub.groupby('patient_id').size()
        filt = sub[sub['patient_id'].isin(pc[pc >= 5].index)]
        if len(filt):
            sample_rows.append([reg_label, str(len(filt)),
                                str(filt['patient_id'].nunique()),
                                f'{filt["memory"].mean()*100:.1f}%',
                                str(int(filt['ied_before_image'].sum())),
                                str(int(filt['ied_during_image'].sum()))])
            during_pct[reg] = filt['ied_during_image'].mean() * 100
    pdf.apa_table(
        f'Table {tn}. Sample Overview (Retrieval Old-Item IED Trials, Descriptive)',
        ['Region', 'N trials', 'N pat.', 'Mem %', 'Before Img', 'During Img'],
        sample_rows, col_widths=[28, 24, 22, 24, 30, 30],
        note='Descriptive counts only (no statistical comparison between '
             'regions). Old retrieved items with power and IED data; patients '
             'with < 5 trials excluded.')
    tn += 1

    pdf.section_title('Statistical Approach')
    pdf.body_text(
        'All models are binomial GLMMs (lme4::glmer, bobyqa, max 200,000 '
        'iterations) with a patient random intercept, predicting memory '
        '(remembered vs forgotten) for retrieved OLD items.'
    )
    pdf.body_text(
        'At retrieval there is no stimulation, so the IED timing windows are the '
        'before-image and during-image windows only, and the stim variable is '
        'prior_stim = whether the item was stimulated at study (S vs NS). The '
        'amygdala (BLA) and HPC, and theta and slow gamma, are treated as '
        'independent questions: every model is fit separately for one region '
        'and one band, and no table or test compares regions or frequencies. '
        'The report is organized into four self-contained sections.'
    )
    if during_pct:
        dpct = ', '.join(f'{rl} {during_pct.get(reg, float("nan")):.0f}%'
                         for reg, rl in REGIONS if reg in during_pct)
        pdf.body_text(
            'Caveat: an IED during the probe image is present on the large '
            f'majority of old-item trials ({dpct}), so the during-image '
            'predictor has limited variance and its estimates (and any '
            'interaction with it) are imprecise -- interpret during-image terms '
            'with caution.'
        )
    pdf.body_text(
        'Multiple comparisons use the Benjamini-Hochberg FDR within families '
        'defined by region x band x analysis type: a main-effects family '
        '(power, IED before-image, IED during-image), an interaction family '
        '(power x before-image, power x during-image), and a stimulation family '
        '(power x prior_stim). FDR q is in the rightmost column with stars on q. '
        'The focused (TF) and all-trials power-alone (A) models are '
        'supplementary and not FDR-corrected (q = "--").'
    )

    for reg, reg_label, b in SECTIONS:
        tn = render_section(pdf, tn, reg, reg_label, b, get_q)

    out_path = os.path.join(OUTPUT_DIR, 'Power_IED_Findings_Report_Retrieval.pdf')
    pdf.output(out_path)
    print(f'Report saved -> {out_path}')


if __name__ == '__main__':
    build_report()
