#!/usr/bin/env python
"""
Build narrative PDF report: Power x IED Findings Summary.

Streamlined design:
  - Bands: theta + slow gamma only (HFA dropped).
  - IED timing windows: after-image + during-stim only (before/during-image dropped).
  - Spread and 3-way models dropped.

The amygdala and HPC, and theta and slow gamma, are treated as fully independent
analyses -- they are never compared to each other. The report is therefore
organized into four self-contained sections (Amygdala-Theta, Amygdala-Slow Gamma,
HPC-Theta, HPC-Slow Gamma); no table places two regions or two frequencies side
by side.

FDR correction (Benjamini-Hochberg) is applied within families defined by
region x band x analysis type:
  - Main effects family (T model):  power, ied_after_image, ied_during_stim
  - Interaction family   (TI model): power x after-image, power x during-stim
  - Stimulation family   (AI model): power x stim
The focused model (TF), stim-only sensitivity model (SO), and all-trials
power-alone model (A) are supplementary and not part of any FDR family
(FDR q shown as '--').
"""

import os
import numpy as np
import pandas as pd
from fpdf import FPDF
from scipy.stats import binomtest

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
# Data/outputs live in the sibling IED/ied_timing_memory dir (scripts were moved
# into mlm_code/ during the topic-dir reorg, but the data did not move).
OUTPUT_DIR = os.path.join(REPO_ROOT, 'IED', 'ied_timing_memory')

REPORT_TITLE = 'Neural Power and IED Effects on Encoding Memory'

BANDS = ['theta', 'slow_gamma']
BAND_LABELS = {'theta': 'Theta', 'slow_gamma': 'Slow Gamma'}
REGIONS = [('BLA', 'Amygdala'), ('HPC', 'HPC')]
# One self-contained section per (region, band). Order: region outer, band inner.
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
    """Significance stars, applied to the FDR q value per report convention."""
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
    'stim': 'Stim Condition',
    'ied_after_image': 'IED After-Image Window',
    'ied_during_stim': 'IED During-Stim Window',
}
for _band in BANDS:
    _bn = {'theta': 'Theta', 'slow_gamma': 'SG'}[_band]
    _z = f'pow_{_band}_z'
    TERM_LABELS[f'{_z}:ied_after_image'] = f'{_bn} Pow x IED After-Img'
    TERM_LABELS[f'ied_after_image:{_z}'] = f'{_bn} Pow x IED After-Img'
    TERM_LABELS[f'{_z}:ied_during_stim'] = f'{_bn} Pow x IED During-Stim'
    TERM_LABELS[f'ied_during_stim:{_z}'] = f'{_bn} Pow x IED During-Stim'
    TERM_LABELS[f'{_z}:stim'] = f'{_bn} Pow x Stim'
    TERM_LABELS[f'stim:{_z}'] = f'{_bn} Pow x Stim'

# Predictor | OR | 95% CI | z | p | FDR q  (stars on q, '--' for non-family)
FULL_HEADERS = ['Predictor', 'OR', '95% CI', 'z', 'p', 'FDR q']
FULL_WIDTHS = [56, 16, 34, 15, 22, 22]
# Robustness comparison (raw p only; sensitivity, no FDR)
ROB_HEADERS = ['Specification', 'OR', '95% CI', 'z', 'p']
ROB_WIDTHS = [62, 18, 36, 18, 24]
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
    """Both orderings of an interaction term (R may emit either)."""
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
            return {
                'or_s': or_str(r['estimate']), 'z_s': z_str(r['statistic']),
                'ci_s': ci_str(r['conf.low'], r['conf.high']),
                'p_s': p_str(r['p.value']),
            }
    return None


def get_n(prefix, reg, b):
    """(nobs, ngrps) for a model, read from its diagnostics CSV."""
    d = load_diagnostics(f'pow_{prefix}_{reg}_{b}_diagnostics.csv')
    if not d:
        return None, None
    nobs, ngr = d.get('nobs'), d.get('ngrps')
    nobs = int(nobs) if nobs is not None and nobs == nobs else None
    ngr = int(ngr) if ngr is not None and ngr == ngr else None
    return nobs, ngr

def n_str(prefix, reg, b):
    nobs, ngr = get_n(prefix, reg, b)
    if nobs is None:
        return ''
    return f'N = {nobs}, {ngr} patients'


# ── FDR families ───────────────────────────────────────────────────────────
def bh_qvalues(pvals):
    """Benjamini-Hochberg step-up q-values, preserving input order."""
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
                'T': [z, 'ied_after_image', 'ied_during_stim'],
                'TI': [f'{z}:ied_after_image', f'ied_after_image:{z}',
                       f'{z}:ied_during_stim', f'ied_during_stim:{z}'],
                'AI': [f'{z}:stim', f'stim:{z}'],
            }
            for prefix, cand in families.items():
                df = load_or(f'pow_{prefix}_{reg}_{b}_odds_ratios.csv')
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
    """Render one model's full coefficient table; return next table number."""
    df = load_or(f'pow_{prefix}_{reg}_{b}_odds_ratios.csv')
    if df is None:
        return tn
    rows = make_or_rows(df, prefix, reg, b, get_q)
    pdf.apa_table(f'Table {tn}. {title}', FULL_HEADERS, rows, FULL_WIDTHS, note=note)
    return tn + 1


def diag_table(pdf, tn, reg, b):
    rows = []
    for prefix, label in [('T', 'Main effects (T)'), ('TI', 'Interaction (TI)')]:
        d = load_diagnostics(f'pow_{prefix}_{reg}_{b}_diagnostics.csv')
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
                  note='ICC = latent-variable ICC for binary GLMM '
                       '(sigma^2_u0 / [sigma^2_u0 + pi^2/3]). R^2_m / R^2_c = '
                       'Nakagawa marginal / conditional pseudo-R-squared.')
    return tn + 1


def lrt_note(reg, b):
    lrt = load_lrt(f'lrt_TI_vs_T_{reg}_{b}.csv')
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
    df = load_or(f'pow_{prefix}_{reg}_{b}_odds_ratios.csv')
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
        f'This section reports {rl} {bl.lower()} power only. The amygdala and '
        f'HPC, and theta and slow gamma, are analyzed in fully separate models '
        f'and are not compared to one another. IED-trial sample: '
        f'{n_str("T", reg, b)}.'
    )

    # Main effects (T)
    pdf.subsection_title('Main Effects: Power + IED Timing Windows')
    tn = full_table(
        pdf, tn, 'T', reg, b, get_q,
        'Main Effects (Power + IED After-Image + IED During-Stim)',
        f'memory ~ {z} + ied_after_image + ied_during_stim + (1|patient_id). '
        f'FDR q (Benjamini-Hochberg) is computed within this three-term '
        f'main-effects family.')
    pdf.body_text(
        f'IED during the stimulation window: {phrase("T", reg, b, "ied_during_stim", get_q)}. '
        f'IED after image presentation: {phrase("T", reg, b, "ied_after_image", get_q)}. '
        f'Power main effect: {phrase("T", reg, b, z, get_q)}. '
        f'(OR < 1 = lower odds of remembering.)'
    )

    # Interaction (TI)
    pdf.subsection_title('Power x IED Timing Interaction')
    tn = full_table(
        pdf, tn, 'TI', reg, b, get_q,
        'Power x IED Timing Interactions',
        f'memory ~ {z} * (ied_after_image + ied_during_stim) + (1|patient_id). '
        f'FDR q applies to the two interaction terms (interaction family); '
        f'lower-order main effects are shown for context (q = "--").'
        + lrt_note(reg, b))
    pdf.body_text(
        f'Power x IED-during-stim interaction: '
        f'{phrase("TI", reg, b, z + ":ied_during_stim", get_q)}. '
        f'Power x IED-after-image interaction: '
        f'{phrase("TI", reg, b, z + ":ied_after_image", get_q)}.'
    )

    # Robustness of the during-stim interaction across window specifications
    pdf.subsection_title('Power x During-Stim Interaction: Window-Specification Robustness')
    rob_rows = []
    for prefix, label in [('TID', 'During-stim only'),
                          ('TI', '+ After-image (2-window)'),
                          ('TI4', 'All four windows')]:
        df = load_or(f'pow_{prefix}_{reg}_{b}_odds_ratios.csv')
        s = get_term_stats(df, f'{z}:ied_during_stim')
        if s:
            rob_rows.append([label, s['or_s'], s['ci_s'], s['z_s'], s['p_s']])
    pdf.apa_table(
        f'Table {tn}. Power x IED-During-Stim Interaction Across Window '
        f'Specifications',
        ROB_HEADERS, rob_rows, ROB_WIDTHS,
        note='The power x during-stim interaction term only, estimated under '
             'three specifications: during-stim alone; with the after-image '
             'interaction added (the 2-window model used above); and with all '
             'four window interactions (before-image, during-image, '
             'after-image, during-stim). Because IEDs co-occur across windows, '
             'these interaction terms are correlated, so the during-stim '
             'estimate is conditional on which others are included. Raw p '
             '(sensitivity comparison; not FDR-corrected).')
    tn += 1

    # Diagnostics (T + TI)
    pdf.subsection_title('Model Diagnostics')
    tn = diag_table(pdf, tn, reg, b)

    # Focused model (TF)
    pdf.subsection_title('Focused Model (Covariate-Adjusted)')
    tn = full_table(
        pdf, tn, 'TF', reg, b, get_q,
        'Focused Model: Both Windows + Stim Covariate',
        f'memory ~ {z} + ied_during_stim + ied_after_image + stim + '
        f'(1|patient_id). Supplementary parsimonious model; not part of an FDR '
        f'family (q = "--"). Stim is a covariate, not an interaction.')

    # Power x Stim (AI)
    pdf.subsection_title('Stimulation: Power x Stim (After-Image IED Trials)')
    tn = full_table(
        pdf, tn, 'AI', reg, b, get_q,
        f'Power x Stim Interaction ({n_str("AI", reg, b)})',
        f'memory ~ {z} * stim + (1|patient_id). After-image IED trials only '
        f'(patients with >= 5 such trials); this is the timing window with '
        f'adequate stim/sham representation. FDR q applies to the power x stim '
        f'interaction (stimulation family).')
    pdf.body_text(
        f'Power x stim interaction: {phrase("AI", reg, b, z + ":stim", get_q)}.'
    )

    # Sensitivity: stim-only trials (SO)
    pdf.subsection_title('Sensitivity: Power x IED Within Stim Trials Only')
    tn = full_table(
        pdf, tn, 'SO', reg, b, get_q,
        f'Power x IED Timing, Stim Trials Only ({n_str("SO", reg, b)})',
        f'memory ~ {z} * (ied_after_image + ied_during_stim) + (1|patient_id). '
        f'Stim trials only. Supplementary sensitivity check on the interaction '
        f'above; not FDR-corrected (q = "--").')
    pdf.body_text(
        f'Within stim-only trials, power x IED-during-stim: '
        f'{phrase("SO", reg, b, z + ":ied_during_stim", get_q)}.'
    )

    # Power alone (A, all trials)
    pdf.subsection_title('Power Alone (All Encoding Trials)')
    tn = full_table(
        pdf, tn, 'A', reg, b, get_q,
        f'Power Main Effect, All Trials ({n_str("A", reg, b)})',
        f'memory ~ {z} + stim + (1|patient_id). All encoding power trials (not '
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
    pdf.cell(0, 6, 'Encoding Phase - Trial-Level Analysis',
             new_x="LMARGIN", new_y="NEXT", align='C')
    pdf.ln(8)

    ied_merged = pd.read_csv(os.path.join(OUTPUT_DIR, 'power_ied_merged.csv'))
    tn = 1

    # Descriptive sample overview (counts only; not an inferential comparison)
    sample_rows = []
    for reg, reg_label in REGIONS:
        sub = ied_merged[ied_merged['region'] == reg]
        pc = sub.groupby('patient_id').size()
        filt = sub[sub['patient_id'].isin(pc[pc >= 5].index)]
        if len(filt):
            sample_rows.append([reg_label, str(len(filt)),
                                str(filt['patient_id'].nunique()),
                                f'{filt["memory"].mean()*100:.1f}%',
                                str(int(filt['ied_after_image'].sum())),
                                str(int(filt['ied_during_stim'].sum()))])
    pdf.apa_table(
        f'Table {tn}. Sample Overview (IED Trials, Descriptive)',
        ['Region', 'N trials', 'N pat.', 'Mem %', 'After Img', 'During Stim'],
        sample_rows, col_widths=[28, 24, 22, 24, 30, 30],
        note='Descriptive counts only (no statistical comparison between '
             'regions). Patients with < 5 IED+power trials excluded. The two '
             'retained windows are not mutually exclusive.')
    tn += 1

    # General statistical approach
    pdf.section_title('Statistical Approach')
    pdf.body_text(
        'All models are generalized linear mixed models (GLMMs) with a binomial '
        'family and logit link (lme4::glmer, bobyqa optimizer, max 200,000 '
        'iterations), each with a random intercept for patient (1 | patient_id).'
    )
    pdf.body_text(
        'The analysis is restricted to theta (4-8 Hz) and slow gamma (30-55 Hz) '
        'power and to the after-image and during-stim IED timing windows. The '
        'amygdala (BLA) and HPC, and the two frequency bands, are treated as '
        'independent questions: every model is fit separately for one region '
        'and one band, and no table or test compares regions or frequencies to '
        'each other. The report is organized into four self-contained sections '
        'accordingly.'
    )
    pdf.body_text(
        'Fixed effects are odds ratios (OR) with 95% Wald confidence intervals '
        'and Wald z-statistics. Multiple comparisons are controlled with the '
        'Benjamini-Hochberg false discovery rate (FDR) within families defined '
        'by region x band x analysis type: a main-effects family (power, IED '
        'after-image, IED during-stim from the T model), an interaction family '
        '(power x after-image, power x during-stim from the TI model), and a '
        'stimulation family (power x stim from the AI model). FDR q appears in '
        'the rightmost column of each table, with significance stars on q. The '
        'focused (TF), stim-only sensitivity (SO), and all-trials power-alone '
        '(A) models are supplementary and not FDR-corrected (q = "--"). ICC is '
        'the latent-variable ICC for binary GLMMs; R^2 values are Nakagawa '
        'marginal and conditional pseudo-R-squared.'
    )

    # Four self-contained region x band sections
    for reg, reg_label, b in SECTIONS:
        tn = render_section(pdf, tn, reg, reg_label, b, get_q)

    # ── Interpretation (narrative discussion, not a statistical comparison) ──
    pdf.add_page()
    pdf.section_title('Interpretation: Stimulation Protocol Context')
    pdf.body_text(
        'The following is narrative interpretation of the independently '
        'estimated models above; it is not a statistical comparison between '
        'regions or frequencies.'
    )

    pdf.subsection_title('Stimulation Protocol')
    pdf.body_text(
        'The stimulation is a theta-modulated gamma burst protocol (8 x 50 Hz) '
        'delivered to the amygdala (BLA). It occurs only during the during-stim '
        'window on stimulated trials, after the image goes off screen -- i.e., '
        'during post-encoding consolidation -- and is designed to drive '
        'amygdala-to-hippocampus communication at theta-gamma frequencies.'
    )

    pdf.subsection_title('Power Measurement Timing')
    pdf.body_text(
        'Power values reflect endogenous baseline activity measured about 2.5 s '
        'before stimulation begins (post minus pre baseline correction); they '
        'capture neural state prior to stimulation, not the stimulation-evoked '
        'response.'
    )

    pdf.subsection_title('Where Do the During-Stim IEDs Occur?')
    pdf.body_text(
        'The IED timing-window flags are trial-level, not region-specific. The '
        'raw IED detection sheet was re-tabulated by anatomical label to '
        'characterize the during-stim window:'
    )

    ied_csv = os.path.join(REPO_ROOT, 'IED',
        'AMMEBLAES_IEDs_trial_level_dissertation_study_usethis_cleaned_with_memory.csv')
    try:
        ied_raw = pd.read_csv(ied_csv)
        ds_ieds = ied_raw[ied_raw['DuringStim'] == 'Y'].copy()
        n_ds = len(ds_ieds)

        def classify_region(r):
            tokens = [tok.strip() for tok in
                      str(r).lower().replace('/', ',').split(',') if tok.strip()]
            has_amyg = any('amygdala' in tok or tok == 'bla' for tok in tokens)
            has_hipp = any(('hippocampus' in tok) or ('hippocampal' in tok)
                           or tok in {'ca', 'dg'} for tok in tokens)
            if has_amyg and has_hipp:
                return 'Both Amygdala + Hippocampal'
            if has_hipp:
                return 'Hippocampal'
            if has_amyg:
                return 'Amygdala only'
            return 'Other (non-hippocampal)'

        ds_ieds['location'] = ds_ieds['Region'].apply(classify_region)
        loc_counts = ds_ieds['location'].value_counts()
        hipp_n = int(ds_ieds['location'].isin(
            ['Hippocampal', 'Both Amygdala + Hippocampal']).sum())
        nonhipp_n = int(n_ds - hipp_n)
        hipp_pct = (hipp_n / n_ds * 100) if n_ds else np.nan
        nonhipp_pct = (nonhipp_n / n_ds * 100) if n_ds else np.nan
        hipp_p = binomtest(hipp_n, n_ds, 0.5).pvalue if n_ds else np.nan

        ds_trials = (ds_ieds.groupby(['Patient', 'Trial'])['location']
                     .agg(lambda x: sorted(set(x)))
                     .reset_index(name='locations'))
        n_ds_trials = len(ds_trials)
        ds_trials['has_hipp'] = ds_trials['locations'].apply(
            lambda locs: any(loc in {'Hippocampal', 'Both Amygdala + Hippocampal'}
                             for loc in locs))
        ds_trials['has_nonhipp'] = ds_trials['locations'].apply(
            lambda locs: any(loc in {'Amygdala only', 'Other (non-hippocampal)'}
                             for loc in locs))
        only_hipp_n = int((ds_trials['has_hipp'] & ~ds_trials['has_nonhipp']).sum())
        only_nonhipp_n = int((~ds_trials['has_hipp'] & ds_trials['has_nonhipp']).sum())
        mixed_n = int((ds_trials['has_hipp'] & ds_trials['has_nonhipp']).sum())
        any_hipp_n = int(ds_trials['has_hipp'].sum())
        exclusive_n = only_hipp_n + only_nonhipp_n
        exclusive_p = (binomtest(only_hipp_n, exclusive_n, 0.5).pvalue
                       if exclusive_n else np.nan)

        loc_rows = []
        for loc in ['Hippocampal', 'Other (non-hippocampal)',
                    'Both Amygdala + Hippocampal', 'Amygdala only']:
            n = loc_counts.get(loc, 0)
            loc_rows.append([loc, str(n), f'{n/n_ds*100:.1f}%'])
        pdf.apa_table(
            f'Table {tn}. Anatomical Origin of During-Stim Window IEDs',
            ['Location', 'N', '%'], loc_rows, col_widths=[60, 30, 30],
            note=f'N = {n_ds} during-stim IED detections across all patients '
                 'and trials. A single trial may have multiple detections.')
        tn += 1

        trial_rows = [
            ['Hippocampal only', str(only_hipp_n),
             f'{only_hipp_n / n_ds_trials * 100:.1f}%'],
            ['Mixed hippocampal + non-hippocampal', str(mixed_n),
             f'{mixed_n / n_ds_trials * 100:.1f}%'],
            ['Non-hippocampal only', str(only_nonhipp_n),
             f'{only_nonhipp_n / n_ds_trials * 100:.1f}%'],
        ]
        pdf.apa_table(
            f'Table {tn}. During-Stim IED Trial Composition by Region Class',
            ['Trial Class', 'N', '%'], trial_rows, col_widths=[80, 20, 20],
            note=f'N = {n_ds_trials} unique patient-trials with at least one '
                 'during-stim IED detection.')
        tn += 1

        pdf.body_text(
            f'During-stim IEDs were hippocampal-dominant but not hippocampal-'
            f'only. At the detection level, {hipp_n} of {n_ds} '
            f'({hipp_pct:.1f}%) involved a hippocampal region vs. {nonhipp_n} '
            f'({nonhipp_pct:.1f}%) non-hippocampal (exact binomial '
            f'{p_inline(p_str(hipp_p))}). At the trial level, {any_hipp_n} of '
            f'{n_ds_trials} during-stim trials '
            f'({any_hipp_n / n_ds_trials * 100:.1f}%) included a hippocampal '
            f'detection, but only {only_hipp_n} '
            f'({only_hipp_n / n_ds_trials * 100:.1f}%) were hippocampal-only '
            f'(hippocampal-only vs. non-hippocampal-only exact binomial '
            f'{p_inline(p_str(exclusive_p))}).'
        )
    except Exception:
        pdf.body_text('(IED location data unavailable for table generation.)')

    pdf.subsection_title('Endogenous Power x IED Interaction')
    pdf.body_text(
        'The frequency bands examined -- theta (4-8 Hz) and slow gamma '
        '(30-55 Hz) -- match the stimulation protocol (theta-modulated 50 Hz '
        'gamma bursts). The amygdala-section models indicate that when '
        "endogenous amygdala theta or slow gamma power is elevated at baseline "
        'and a during-stim IED (often hippocampal) coincides with the '
        'stimulation window, encoding is impaired, whereas power alone is not '
        'detrimental. Two processes may converge: sender saturation (an '
        'already-oscillating amygdala may not be effectively augmented by '
        'same-frequency stimulation) and receiver disruption (the hippocampus, '
        'the target of the consolidation signal, is simultaneously disrupted by '
        'an IED). Each region and band was estimated independently; the '
        'amygdala vs. HPC pattern described here is interpretive, not a '
        'statistical contrast.'
    )

    out_path = os.path.join(OUTPUT_DIR, 'Power_IED_Findings_Report.pdf')
    pdf.output(out_path)
    print(f'Report saved -> {out_path}')


if __name__ == '__main__':
    build_report()
