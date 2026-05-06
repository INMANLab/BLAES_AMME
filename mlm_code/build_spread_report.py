#!/usr/bin/env python
"""
Build narrative PDF report: IED Spatial Spread and Encoding Memory.
Separate report from the main Power x IED findings.
"""

import os
import numpy as np
import pandas as pd
from fpdf import FPDF

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'outputs', 'ied_timing_memory')

REPORT_TITLE = 'IED Spatial Spread and Encoding Memory'


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

    def italic_text(self, text):
        self.set_font('Times', 'I', 11)
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


def p_str(p):
    if isinstance(p, str): return p
    if p < .001: return '< .001'
    return f'{p:.3f}'.lstrip('0')

def sig_str(p):
    if isinstance(p, str): return ''
    if p < .001: return '***'
    elif p < .01: return '**'
    elif p < .05: return '*'
    elif p < .1: return '+'
    return ''

def or_str(v): return f'{v:.2f}'
def ci_str(lo, hi): return f'[{lo:.2f}, {hi:.2f}]'
def se_str(v): return f'{v:.3f}'
def z_str(v): return f'{v:.2f}'
def fmt_r2(v):
    if pd.isna(v): return '--'
    return f'{v:.3f}'
def fmt_icc(v):
    if pd.isna(v): return '--'
    return f'{v:.3f}'
def fmt_aic(v):
    if pd.isna(v): return '--'
    return f'{v:.1f}'
def fmt_var(v):
    if pd.isna(v): return '--'
    return f'{v:.3f}'

TERM_LABELS = {
    '(Intercept)': 'Intercept',
    'pow_theta_z': 'Theta Power (z)',
    'pow_slow_gamma_z': 'Slow Gamma Power (z)',
    'pow_hfa_z': 'HFA Power (z)',
    'n_regions_c': 'Region Spread (centered)',
    'n_channels_c': 'Channel Spread (centered)',
    'ied_before_image': 'IED Before-Image',
    'ied_during_image': 'IED During-Image',
    'ied_after_image': 'IED After-Image',
    'ied_during_stim': 'IED During-Stim Window',
}

OR_HEADERS = ['Predictor', 'OR', 'SE(B)', 'z', '95% CI', 'p', '']
OR_WIDTHS = [42, 13, 14, 13, 30, 22, 8]


def load_or(filename):
    path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(path): return None
    return pd.read_csv(path)

def load_diagnostics(filename):
    path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(path): return None
    df = pd.read_csv(path)
    return dict(zip(df['metric'], df['value']))

def load_lrt(filename):
    path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(path): return None
    return pd.read_csv(path)

def make_or_rows(df, skip_intercept=True):
    rows = []
    for _, r in df.iterrows():
        term = r['term']
        if skip_intercept and term == '(Intercept)': continue
        label = TERM_LABELS.get(term, term)
        rows.append([label, or_str(r['estimate']),
                     se_str(r['std.error']), z_str(r['statistic']),
                     ci_str(r['conf.low'], r['conf.high']),
                     p_str(r['p.value']), sig_str(r['p.value'])])
    return rows


def get_term_stats(df, term):
    row = df[df['term'] == term]
    if len(row) == 0:
        return None
    r = row.iloc[0]
    return {
        'or': r['estimate'], 'lo': r['conf.low'], 'hi': r['conf.high'],
        'p': r['p.value'], 'se': r['std.error'], 'z': r['statistic'],
        'or_s': or_str(r['estimate']),
        'se_s': se_str(r['std.error']),
        'z_s': z_str(r['statistic']),
        'ci_s': ci_str(r['conf.low'], r['conf.high']),
        'p_s': p_str(r['p.value']),
        'sig': sig_str(r['p.value']),
    }


def build_report():
    pdf = APAReport()
    pdf.add_page()

    # Title
    pdf.set_font('Helvetica', 'B', 16)
    pdf.cell(0, 10, REPORT_TITLE, new_x="LMARGIN", new_y="NEXT", align='C')
    pdf.set_font('Helvetica', '', 11)
    pdf.cell(0, 6, 'Encoding Phase - Trial-Level Analysis',
             new_x="LMARGIN", new_y="NEXT", align='C')
    pdf.ln(8)

    # Load trial-level IED data (from original IED CSV, aggregated to patient x trial)
    ied_trials = pd.read_csv(os.path.join(OUTPUT_DIR, 'ied_spread_trials.csv'))

    tbl_num = 1

    # ══════════════════════════════════════════════════════════════
    # SECTION 1: DESCRIPTIVES
    # ══════════════════════════════════════════════════════════════
    pdf.section_title('1. Spread Descriptives')
    pdf.body_text(
        'IED spatial spread was quantified at the trial level using two '
        'measures: (1) region spread -- the number of distinct brain '
        'regions (e.g., amygdala, hippocampus, entorhinal cortex) where '
        'IED activity was detected on a given trial, and (2) channel '
        'spread -- the number of individual recording channels within '
        'the region of interest showing IED activity on that trial. '
        'These measures capture different aspects of IED propagation: '
        'region spread reflects cross-regional network involvement, '
        'while channel spread reflects local within-region synchrony.'
    )

    n_trials = len(ied_trials)
    n_patients = ied_trials['Patient'].nunique()

    # Descriptive stats
    nr_mean = ied_trials['n_regions'].mean()
    nr_sd = ied_trials['n_regions'].std()
    nc_mean = ied_trials['n_channels'].mean()
    nc_sd = ied_trials['n_channels'].std()
    r_corr = ied_trials['n_regions'].corr(ied_trials['n_channels'])

    desc_rows = [[
        str(n_trials), str(n_patients),
        f'{nr_mean:.2f} ({nr_sd:.2f})',
        f'[{ied_trials["n_regions"].min()}, {ied_trials["n_regions"].max()}]',
        f'{nc_mean:.2f} ({nc_sd:.2f})',
        f'[{ied_trials["n_channels"].min()}, {ied_trials["n_channels"].max()}]',
        f'{r_corr:.2f}'
    ]]

    pdf.apa_table(
        f'Table {tbl_num}. IED Spread Descriptives',
        ['N Trials', 'N Pts', 'Reg. Spread M (SD)', 'Range',
         'Ch. Spread M (SD)', 'Range', 'r'],
        desc_rows,
        col_widths=[18, 14, 32, 18, 32, 18, 12],
        note='Region spread = number of distinct brain regions with IED '
             'activity on each trial. Channel spread = number of recording '
             'channels showing IED activity on each trial. '
             'r = Pearson correlation between region spread and channel '
             'spread. Data from the original IED CSV aggregated to the '
             'trial level (not restricted to power subsample).'
    )
    tbl_num += 1

    pdf.body_text(
        f'Region spread and channel spread were moderately correlated '
        f'(r = {r_corr:.2f}), meaning that trials with IEDs spreading '
        f'across more brain regions also tended to have more channels '
        f'involved. This collinearity motivates testing each spread '
        f'measure in separate models rather than together, where mutual '
        f'suppression or inflation could distort estimates.'
    )

    # Memory rate by spread
    pdf.subsection_title('Memory Rate by Spread')

    mem_rows = []
    for nr in sorted(ied_trials['n_regions'].unique()):
        s = ied_trials[ied_trials['n_regions'] == nr]
        if len(s) < 3:
            continue
        mem_rows.append([
            str(int(nr)), str(len(s)),
            f'{s["memory"].mean()*100:.1f}%'
        ])

    pdf.apa_table(
        f'Table {tbl_num}. Memory Rate by Number of Regions with IED Activity',
        ['N Regions', 'N Trials', 'Remembered'],
        mem_rows,
        col_widths=[30, 30, 30],
        note='All encoding IED trials. Rows with < 3 trials omitted.'
    )
    tbl_num += 1

    mem_ch_rows = []
    for nc in sorted(ied_trials['n_channels'].unique()):
        s = ied_trials[ied_trials['n_channels'] == nc]
        if len(s) < 3:
            continue
        mem_ch_rows.append([
            str(int(nc)), str(len(s)),
            f'{s["memory"].mean()*100:.1f}%'
        ])

    pdf.apa_table(
        f'Table {tbl_num}. Memory Rate by Number of Channels with IED Activity',
        ['N Channels', 'N Trials', 'Remembered'],
        mem_ch_rows,
        col_widths=[30, 30, 30],
        note='All encoding IED trials. Rows with < 3 trials omitted.'
    )
    tbl_num += 1

    # ══════════════════════════════════════════════════════════════
    # STATISTICAL APPROACH & MODEL DIAGNOSTICS
    # ══════════════════════════════════════════════════════════════
    pdf.add_page()
    pdf.section_title('Statistical Approach')
    pdf.body_text(
        'All models were fit as generalized linear mixed models (GLMMs) '
        'with a binomial family and logit link using lme4::glmer in R, '
        'with a bobyqa optimizer (max 200,000 iterations). Each model '
        'included a random intercept for patient (1 | patient_id) to '
        'account for repeated measures within patients. Each spread '
        'measure was tested in its own model to avoid collinearity '
        'between region spread and channel spread.'
    )
    pdf.body_text(
        'Fixed effects are reported as odds ratios (OR) with 95% Wald '
        'confidence intervals, standard errors on the log-odds scale '
        'SE(B), and Wald z-statistics. Significance of individual '
        'coefficients was assessed via the Wald z-test. Model '
        'comparisons against the intercept-only (null) model were '
        'performed via likelihood ratio tests (LRTs), reported as '
        'chi-squared with degrees of freedom and p-value.'
    )
    pdf.body_text(
        'The intraclass correlation coefficient (ICC) was computed '
        'using the latent-variable approach for binary GLMMs: '
        'ICC = sigma^2_u0 / (sigma^2_u0 + pi^2/3), where sigma^2_u0 is '
        'the patient random intercept variance and pi^2/3 (~3.29) is '
        'the level-1 residual variance implied by the logistic '
        'distribution. This method is recommended for binary outcomes '
        'because the logistic residual variance is fixed (Goldstein '
        'et al., 2002; Snijders & Bosker, 2012).'
    )
    pdf.body_text(
        'Effect sizes are reported as Nakagawa pseudo-R-squared '
        '(Nakagawa & Schielzeth, 2013; Nakagawa, Johnson, & Schielzeth, '
        '2017): marginal R-squared (R^2_m) represents variance explained '
        'by fixed effects only, and conditional R-squared (R^2_c) '
        'represents variance explained by fixed plus random effects. '
        'These were computed using the performance R package (Ludecke '
        'et al., 2021), which implements the trigamma-based '
        'approximation for binomial GLMMs.'
    )

    # ── Model diagnostics table ──
    pdf.subsection_title('Model Diagnostics')
    diag_rows = []
    for label, prefix in [('Null (intercept only)', 'spread_null'),
                           ('Region Spread', 'spread_region'),
                           ('Channel Spread', 'spread_channel')]:
        d = load_diagnostics(f'{prefix}_diagnostics.csv')
        if d is None: continue
        diag_rows.append([
            label,
            fmt_icc(d.get('ICC_latent')),
            fmt_var(d.get('rand_intercept_var')),
            fmt_var(d.get('rand_intercept_sd')),
            fmt_aic(d.get('AIC')),
            fmt_aic(d.get('BIC')),
            fmt_r2(d.get('R2_marginal')),
            fmt_r2(d.get('R2_conditional')),
        ])
    pdf.apa_table(
        f'Table {tbl_num}. Model Diagnostics',
        ['Model', 'ICC', 'sigma^2_u0', 'SD_u0', 'AIC', 'BIC',
         'R^2_m', 'R^2_c'],
        diag_rows,
        col_widths=[32, 14, 16, 14, 16, 16, 14, 14],
        note='ICC = latent-variable ICC for binary GLMM '
             '(sigma^2_u0 / [sigma^2_u0 + pi^2/3]). sigma^2_u0 = patient '
             'random intercept variance. R^2_m = Nakagawa marginal '
             '(fixed effects only). R^2_c = Nakagawa conditional '
             '(fixed + random). Null model: memory ~ (1|patient_id). '
             'Region spread model: memory ~ n_regions_c + (1|patient_id). '
             'Channel spread model: memory ~ n_channels_c + (1|patient_id). '
             f'{n_trials} trials, {n_patients} patients.'
    )
    tbl_num += 1

    # ── LRT table: each spread model vs null ──
    pdf.subsection_title('Model Comparisons: Spread vs. Null (LRT)')
    lrt_rows = []
    for label, lrt_file in [('Region Spread vs. Null', 'lrt_spread_region_vs_null.csv'),
                             ('Channel Spread vs. Null', 'lrt_spread_channel_vs_null.csv')]:
        lrt = load_lrt(lrt_file)
        if lrt is None: continue
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
        lrt_rows.append([
            label,
            f'{chisq:.2f}',
            f'{int(df_val)}',
            p_str(p_val),
            f'{d_aic:+.1f}',
            f'{d_bic:+.1f}',
            sig_str(p_val),
        ])
    pdf.apa_table(
        f'Table {tbl_num}. Likelihood Ratio Tests: Spread Models vs. Null',
        ['Comparison', 'chi^2', 'df', 'p', 'dAIC', 'dBIC', ''],
        lrt_rows,
        col_widths=[42, 16, 12, 22, 16, 16, 8],
        note='Likelihood ratio test comparing each spread model to the '
             'intercept-only null model: memory ~ (1|patient_id). '
             'chi^2 = likelihood ratio chi-squared statistic. '
             'dAIC and dBIC = change in AIC/BIC (negative favors '
             'the spread model). df = difference in number of parameters.'
    )
    tbl_num += 1

    # ══════════════════════════════════════════════════════════════
    # SECTION 2: REGION SPREAD MODEL
    # ══════════════════════════════════════════════════════════════
    pdf.add_page()
    pdf.section_title('2. Region Spread and Memory')
    pdf.body_text(
        'Region spread (number of distinct brain regions with IED '
        'activity on a given trial) was tested as a predictor of '
        'subsequent memory. Each spread measure is tested in its own '
        'model to avoid collinearity.'
    )
    pdf.body_text(
        'Model: memory ~ n_regions_c + (1 | patient_id)'
    )

    spr_note = (
        'memory ~ n_regions_c + (1|patient_id). '
        'Region spread is mean-centered. '
        f'{n_trials} trials, {n_patients} patients. '
        'Channel spread is NOT included.'
    )

    df = load_or('spread_region_odds_ratios.csv')
    if df is not None:
        rows = make_or_rows(df)
        pdf.apa_table(
            f'Table {tbl_num}. Region Spread Effect on Memory '
            f'(N = {n_trials}, {n_patients} patients)',
            OR_HEADERS, rows, OR_WIDTHS,
            note=spr_note + ' OR < 1 indicates lower odds of remembering '
                 'with each additional brain region showing IED activity. '
                 'SE(B) = standard error of the log-odds coefficient; '
                 'z = Wald z-statistic.'
        )
        tbl_num += 1

        s = get_term_stats(df, 'n_regions_c')
        if s:
            pdf.body_text(
                f'Region spread did not significantly predict memory '
                f'(OR = {s["or_s"]}, SE(B) = {s["se_s"]}, z = {s["z_s"]}, '
                f'95% CI {s["ci_s"]}, p = {s["p_s"]}). '
                f'The direction was negative, suggesting a trend toward '
                f'worse memory with greater cross-regional IED involvement, '
                f'but the effect did not reach significance.'
            )

    # ══════════════════════════════════════════════════════════════
    # SECTION 3: CHANNEL SPREAD MODEL
    # ══════════════════════════════════════════════════════════════
    pdf.section_title('3. Channel Spread and Memory')
    pdf.body_text(
        'Channel spread (number of recording channels showing IED '
        'activity on a given trial) was tested as a predictor of '
        'subsequent memory.'
    )
    pdf.body_text(
        'Model: memory ~ n_channels_c + (1 | patient_id)'
    )

    spc_note = (
        'memory ~ n_channels_c + (1|patient_id). '
        'Channel spread is mean-centered. '
        f'{n_trials} trials, {n_patients} patients. '
        'Region spread is NOT included.'
    )

    df = load_or('spread_channel_odds_ratios.csv')
    if df is not None:
        rows = make_or_rows(df)
        pdf.apa_table(
            f'Table {tbl_num}. Channel Spread Effect on Memory '
            f'(N = {n_trials}, {n_patients} patients)',
            OR_HEADERS, rows, OR_WIDTHS,
            note=spc_note + ' OR < 1 indicates lower odds of remembering '
                 'with each additional channel showing IED activity. '
                 'SE(B) = standard error of the log-odds coefficient; '
                 'z = Wald z-statistic.'
        )
        tbl_num += 1

        s = get_term_stats(df, 'n_channels_c')
        if s:
            pdf.body_text(
                f'Channel spread did not significantly predict memory '
                f'(OR = {s["or_s"]}, SE(B) = {s["se_s"]}, z = {s["z_s"]}, '
                f'95% CI {s["ci_s"]}, p = {s["p_s"]}). '
            )

    # ══════════════════════════════════════════════════════════════
    # SECTION 4: SUMMARY
    # ══════════════════════════════════════════════════════════════
    pdf.add_page()
    pdf.section_title('4. Summary')
    pdf.body_text(
        'This report tested whether IED spatial spread -- both across '
        'brain regions and across channels -- predicts subsequent memory '
        f'impairment on encoding trials ({n_trials} trials, '
        f'{n_patients} patients). Key findings:'
    )
    pdf.body_text(
        f'1. Neither region spread (OR = 0.93, p = .36) nor channel '
        f'spread (OR = 0.91, p = .27) significantly predicted memory '
        f'when tested in separate models.'
    )
    pdf.body_text(
        '2. These results suggest that how far IEDs spread (number of '
        'regions or channels) does not meaningfully predict encoding '
        'memory impairment in this sample.'
    )

    # Save
    out_path = os.path.join(OUTPUT_DIR, 'IED_Spread_Memory_Report.pdf')
    pdf.output(out_path)
    print(f'Report saved -> {out_path}')


if __name__ == '__main__':
    build_report()
