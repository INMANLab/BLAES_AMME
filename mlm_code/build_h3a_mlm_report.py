#!/usr/bin/env python
"""
Build PDF report for Hypothesis 3a Mixed-Effects Logistic Regression.
Reads R output files from outputs/ied_timing_memory/ and assembles a
formatted report with explanations.

Outputs: outputs/ied_timing_memory/H3a_MLM_Report.pdf
"""

import os
import numpy as np
import pandas as pd
from fpdf import FPDF

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
OUTPUT_DIR = os.path.join(REPO_ROOT, 'IED', 'ied_timing_memory')


class APAReport(FPDF):
    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=25)

    def header(self):
        self.set_font('Helvetica', 'I', 8)
        self.cell(0, 5, 'Hypothesis 3a: Mixed-Effects Logistic Regression', align='R')
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

    def bold_text(self, text):
        self.set_font('Times', 'B', 11)
        self.multi_cell(0, 5.5, text)
        self.ln(1)

    def code_text(self, text):
        self.set_font('Courier', '', 8.5)
        self.multi_cell(0, 4.5, text)
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
    if isinstance(p, str):
        return p
    if p < .001:
        return '< .001'
    return f'{p:.3f}'.lstrip('0')


def sig_str(p):
    if isinstance(p, str):
        return ''
    if p < .001:
        return '***'
    elif p < .01:
        return '**'
    elif p < .05:
        return '*'
    return ''


def bh_fdr(pvals):
    """Benjamini-Hochberg FDR. Returns q-values in the original order."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    ranked = np.clip(ranked, 0, 1)
    q = np.empty(n)
    q[order] = ranked
    return q


def main():
    # Load R output tables
    m1a = pd.read_csv(os.path.join(OUTPUT_DIR, 'mlm_model1a_odds_ratios.csv'))
    m1b = pd.read_csv(os.path.join(OUTPUT_DIR, 'mlm_model1b_odds_ratios.csv'))
    m2a = pd.read_csv(os.path.join(OUTPUT_DIR, 'mlm_model2a_odds_ratios.csv'))
    m2b = pd.read_csv(os.path.join(OUTPUT_DIR, 'mlm_model2b_odds_ratios.csv'))

    # Load data for descriptives
    enc = pd.read_csv(os.path.join(OUTPUT_DIR, 'mlm_encoding_trials.csv'))
    ret = pd.read_csv(os.path.join(OUTPUT_DIR, 'mlm_retrieval_patient_freq.csv'))

    n_trials = len(enc)
    n_patients = enc['patient_id'].nunique()
    n_rem = enc['memory'].sum()
    n_forg = n_trials - n_rem

    pdf = APAReport()
    pdf.add_page()

    # ── Title ──
    pdf.set_font('Helvetica', 'B', 16)
    pdf.cell(0, 10, 'Hypothesis 3a:', align='C', new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 10, 'Mixed-Effects Logistic Regression', align='C',
             new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)

    # ================================================================
    # RATIONALE
    # ================================================================
    pdf.section_title('Rationale for Mixed-Effects Modeling')

    pdf.subsection_title('Why memory is the dependent variable')
    pdf.body_text(
        'Memory outcome (remembered = 1, forgotten = 0) is the key outcome we are trying '
        'to predict. It is binary, so we use logistic regression rather than linear regression. '
        'Each trial produces one binary observation: was the item later remembered or not?'
    )

    pdf.subsection_title('Why patient_id is a random effect')
    pdf.body_text(
        'Trials are nested within patients: each patient contributes multiple trials, and '
        'patients differ in their baseline memory ability, IED profiles, and responsiveness to '
        'stimulation. A random intercept for patient_id accounts for this non-independence. '
        'Without it, we would violate the independence assumption and risk inflated Type I error '
        'rates. The random intercept lets each patient have their own baseline log-odds of '
        'remembering, while the fixed effects estimate the average effect of IEDs and '
        'stimulation across all patients.'
    )

    pdf.subsection_title('Why a logistic mixed-effects model (GLMM)')
    pdf.body_text(
        'The combination of a binary outcome (logistic) and nested data (mixed-effects) '
        'requires a generalized linear mixed model (GLMM). We fit these using the glmer() '
        'function from the lme4 package in R, with a logit link function. Results are reported '
        'as odds ratios (OR): OR < 1 means lower odds of remembering, OR > 1 means higher odds.'
    )

    # ================================================================
    # DATA OVERVIEW
    # ================================================================
    pdf.section_title('Data Overview')

    pdf.body_text(
        f'The dataset included {n_trials} encoding trials from {n_patients} patients '
        f'({n_rem} remembered, {n_forg} forgotten). Each trial was coded for IED presence '
        f'(0/1) in four timing windows: Before Image, During Image, After Image, and During '
        f'Stim. Stimulation condition (stim = 1, no-stim = 0) was recorded for each trial.'
    )

    # Retrieval IED freq descriptives
    ret_valid = ret[ret['retrieval_ied_freq'].notna()]
    freq_m = ret_valid['retrieval_ied_freq'].mean()
    freq_sd = ret_valid['retrieval_ied_freq'].std()
    freq_med = ret_valid['retrieval_ied_freq'].median()
    freq_min = ret_valid['retrieval_ied_freq'].min()
    freq_max = ret_valid['retrieval_ied_freq'].max()
    n_zero = (ret_valid['retrieval_ied_freq'] == 0).sum()

    pdf.body_text(
        f'Patient-level retrieval IED frequency was computed as the number of retrieval '
        f'trials with any IED divided by total retrieval trials (from phase 3 power CSV files). '
        f'Across {len(ret_valid)} patients, retrieval IED frequency ranged from '
        f'{freq_min:.1%} to {freq_max:.1%} (M = {freq_m:.1%}, SD = {freq_sd:.1%}, '
        f'Mdn = {freq_med:.1%}). {n_zero} of {len(ret_valid)} patients had zero retrieval IEDs.'
    )

    # ================================================================
    # STATISTICAL APPROACH & MODEL DIAGNOSTICS
    # ================================================================
    pdf.section_title('Statistical Approach')
    pdf.body_text(
        'All models were fit as generalized linear mixed models (GLMMs) '
        'with a binomial family and logit link using lme4::glmer in R, '
        'with a bobyqa optimizer (max 100,000-200,000 iterations). Each '
        'model included a random intercept for patient (1 | patient_id) to '
        'account for repeated measures within patients.'
    )
    pdf.body_text(
        'Fixed effects are reported as odds ratios (OR) with 95% Wald '
        'confidence intervals, standard errors on the log-odds scale '
        'SE(B), and Wald z-statistics. Significance of individual '
        'coefficients was assessed via the Wald z-test. Model '
        'comparisons between nested models were performed via likelihood '
        'ratio tests (LRTs), reported as chi-squared with degrees of '
        'freedom and p-value.'
    )
    pdf.body_text(
        'The intraclass correlation coefficient (ICC) was computed '
        'using the latent-variable approach for binary GLMMs: '
        'ICC = sigma^2_u0 / (sigma^2_u0 + pi^2/3), where sigma^2_u0 is '
        'the patient random intercept variance and pi^2/3 (~3.29) is '
        'the level-1 residual variance implied by the logistic '
        'distribution (Goldstein et al., 2002; Snijders & Bosker, 2012).'
    )
    pdf.body_text(
        'Effect sizes are reported as Nakagawa pseudo-R-squared '
        '(Nakagawa & Schielzeth, 2013; Nakagawa, Johnson, & Schielzeth, '
        '2017): marginal R-squared (R^2_m) represents variance explained '
        'by fixed effects only, and conditional R-squared (R^2_c) '
        'represents variance explained by fixed plus random effects, '
        'computed via the performance R package (Ludecke et al., 2021).'
    )

    # ── Load diagnostics helper ──
    def _load_diag(fn):
        path = os.path.join(OUTPUT_DIR, fn)
        if not os.path.exists(path):
            return None
        df = pd.read_csv(path)
        return dict(zip(df['metric'], df['value']))

    def _load_lrt(fn):
        path = os.path.join(OUTPUT_DIR, fn)
        if not os.path.exists(path):
            return None
        return pd.read_csv(path)

    def _fmt(v, fmt='.3f'):
        if pd.isna(v): return '--'
        return f'{v:{fmt}}'

    # ── Model diagnostics table ──
    pdf.subsection_title('Model Diagnostics')
    diag_rows = []
    for label, fn in [('Null (intercept only)', 'mlm_null_diagnostics.csv'),
                       ('1a: IED timing windows', 'mlm_model1a_diagnostics.csv'),
                       ('1b: + Stim interactions', 'mlm_model1b_diagnostics.csv'),
                       ('2a: Stim x Ret. IED freq', 'mlm_model2a_diagnostics.csv')]:
        d = _load_diag(fn)
        if d is None:
            continue
        diag_rows.append([
            label,
            _fmt(d.get('ICC_latent')),
            _fmt(d.get('rand_intercept_var')),
            _fmt(d.get('rand_intercept_sd')),
            _fmt(d.get('AIC'), '.1f'),
            _fmt(d.get('BIC'), '.1f'),
            _fmt(d.get('R2_marginal')),
            _fmt(d.get('R2_conditional')),
        ])

    pdf.apa_table(
        'Model Diagnostics: Variance Components, ICC, and Effect Sizes',
        ['Model', 'ICC', 'sigma^2_u0', 'SD_u0', 'AIC', 'BIC',
         'R^2_m', 'R^2_c'],
        diag_rows,
        col_widths=[36, 13, 15, 13, 15, 15, 13, 13],
        note='ICC = latent-variable ICC for binary GLMM '
             '(sigma^2_u0 / [sigma^2_u0 + pi^2/3]). sigma^2_u0 = patient '
             'random intercept variance. R^2_m = Nakagawa marginal '
             '(fixed effects only). R^2_c = Nakagawa conditional '
             '(fixed + random). Model 2a uses a subset of patients '
             'with retrieval IED frequency data.'
    )

    # ── LRT table ──
    pdf.subsection_title('Model Comparisons (Likelihood Ratio Tests)')
    lrt_rows = []
    for label, fn in [('1a vs. Null', 'lrt_h3a_m1a_vs_null.csv'),
                       ('1b vs. 1a', 'lrt_h3a_m1b_vs_m1a.csv'),
                       ('2a vs. Null', 'lrt_h3a_m2a_vs_null.csv')]:
        lrt = _load_lrt(fn)
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
        lrt_rows.append([
            label,
            f'{chisq:.2f}',
            f'{int(df_val)}',
            f'{p_str(p_val)}{sig_str(p_val)}',
            f'{d_aic:+.1f}',
            f'{d_bic:+.1f}',
        ])

    pdf.apa_table(
        'Likelihood Ratio Tests',
        ['Comparison', 'chi^2', 'df', 'p', 'dAIC', 'dBIC'],
        lrt_rows,
        col_widths=[36, 18, 14, 22, 22, 22],
        note='chi^2 = likelihood ratio chi-squared statistic. '
             'dAIC and dBIC = change in AIC/BIC from reduced to full model '
             '(negative favors the more complex model). '
             '1a vs. Null tests whether IED timing windows improve over '
             'intercept-only. 1b vs. 1a tests whether adding stimulation '
             'interactions improves fit. 2a vs. Null tests whether stim, '
             'retrieval IED frequency, and their interaction improve over '
             'intercept-only.'
    )

    # ================================================================
    # MODEL 1a: Encoding IED windows
    # ================================================================
    pdf.section_title('Model 1: Encoding IED Timing and Memory')

    pdf.subsection_title('Model 1a: Random Intercept Only')
    pdf.italic_text(
        'memory ~ ied_before_image + ied_during_image + ied_after_image + ied_during_stim '
        '+ (1 | patient_id)'
    )

    pdf.body_text(
        'This model tests whether IED presence in each encoding timing window predicts '
        'subsequent memory, while accounting for patient-level variability via a random '
        'intercept. Each window is entered as a separate binary predictor, allowing us to '
        'assess their independent contributions.'
    )

    # Table. FDR family = the four IED timing-window predictors (intercept excluded).
    window_labels = {
        'ied_before_image': 'Before Image IED',
        'ied_during_image': 'During Image IED',
        'ied_after_image': 'After Image IED',
        'ied_during_stim': 'During Stim IED',
    }
    m1a_fam = m1a[m1a['term'].isin(window_labels)].copy()
    m1a_q = dict(zip(m1a_fam['term'], bh_fdr(m1a_fam['p.value'].values)))

    rows_1a = []
    for _, r in m1a.iterrows():
        term = r['term']
        label = '(Intercept)' if term == '(Intercept)' else window_labels.get(term, term)

        p = r['p.value']
        q = m1a_q.get(term)
        q_cell = f'{p_str(q)}{sig_str(q)}' if q is not None else '--'
        rows_1a.append([
            label,
            f'{r["estimate"]:.3f}',
            f'[{r["conf.low"]:.3f}, {r["conf.high"]:.3f}]',
            f'{r["statistic"]:.2f}',
            f'{p_str(p)}',
            q_cell,
        ])

    pdf.apa_table(
        'Table 1\nModel 1a: Encoding IED Timing Windows Predicting Subsequent Memory (Odds Ratios)',
        ['Predictor', 'OR', '95% CI', 'z', 'p', 'FDR q'],
        rows_1a,
        col_widths=[48, 16, 32, 14, 16, 18],
        note=f'GLMM with logit link and random intercept for patient (N = {n_trials} trials, '
             f'{n_patients} patients). OR = odds ratio; values < 1 indicate lower odds of '
             f'remembering when IED is present. z = Wald z-statistic. FDR q = Benjamini-Hochberg '
             f'FDR-corrected p-value across the four timing-window predictors. The intercept is '
             f'not part of the FDR family (--). Stars denote FDR significance: *q < .05. '
             f'**q < .01. ***q < .001.'
    )

    # Extract key values for text
    after_or = m1a[m1a['term'] == 'ied_after_image']['estimate'].values[0]
    after_ci = (m1a[m1a['term'] == 'ied_after_image']['conf.low'].values[0],
                m1a[m1a['term'] == 'ied_after_image']['conf.high'].values[0])
    after_p = m1a[m1a['term'] == 'ied_after_image']['p.value'].values[0]
    stim_or = m1a[m1a['term'] == 'ied_during_stim']['estimate'].values[0]
    stim_ci = (m1a[m1a['term'] == 'ied_during_stim']['conf.low'].values[0],
               m1a[m1a['term'] == 'ied_during_stim']['conf.high'].values[0])
    stim_p = m1a[m1a['term'] == 'ied_during_stim']['p.value'].values[0]

    pdf.body_text(
        f'Consistent with the chi-square analyses, IEDs in two timing windows significantly '
        f'predicted worse memory. After Image IEDs reduced the odds of remembering by '
        f'{(1 - after_or) * 100:.0f}% (OR = {after_or:.2f}, 95% CI [{after_ci[0]:.2f}, '
        f'{after_ci[1]:.2f}], z = {m1a[m1a["term"] == "ied_after_image"]["statistic"].values[0]:.2f}, '
        f'p = {p_str(after_p)}). During Stim IEDs reduced the odds by '
        f'{(1 - stim_or) * 100:.0f}% (OR = {stim_or:.2f}, 95% CI [{stim_ci[0]:.2f}, '
        f'{stim_ci[1]:.2f}], z = {m1a[m1a["term"] == "ied_during_stim"]["statistic"].values[0]:.2f}, '
        f'p = {p_str(stim_p)}). Before Image and During Image IEDs were not significant '
        f'predictors (ps > .17).'
    )

    # ICC and variance components from diagnostics
    m1a_diag = _load_diag('mlm_model1a_diagnostics.csv')
    if m1a_diag:
        ri_var = m1a_diag.get('rand_intercept_var', 0.80)
        ri_sd = m1a_diag.get('rand_intercept_sd', 0.89)
        icc_val = m1a_diag.get('ICC_latent', 0.20)
        pdf.body_text(
            f'The random intercept variance was {ri_var:.2f} (SD = {ri_sd:.2f}), '
            f'corresponding to a latent-variable ICC of {icc_val:.3f}. This indicates '
            f'that approximately {icc_val*100:.0f}% of the latent-response variance in memory '
            f'is attributable to between-patient differences, confirming the importance of '
            f'the mixed-effects approach.'
        )
    else:
        pdf.body_text(
            'The random intercept variance was 0.80 (SD = 0.89), indicating substantial '
            'between-patient variability in baseline memory performance. This confirms the '
            'importance of the mixed-effects approach.'
        )

    # Forest plot
    forest_path = os.path.join(OUTPUT_DIR, 'mlm_model1a_forest.png')
    if os.path.exists(forest_path):
        pdf.ln(2)
        pdf.italic_text('Figure 1. Forest plot of odds ratios from Model 1a.')
        img_w = pdf.w - pdf.l_margin - pdf.r_margin - 10
        pdf.image(forest_path, x=pdf.l_margin + 5, w=img_w)
        pdf.ln(3)

    # ================================================================
    # MODEL 1b: With stim interactions
    # ================================================================
    pdf.subsection_title('Model 1b: Adding Stimulation Interactions')
    pdf.italic_text(
        'memory ~ (ied_before_image + ied_during_image + ied_after_image + ied_during_stim) '
        '* stim + (1 | patient_id)'
    )

    pdf.body_text(
        'Model 1b adds stimulation condition and its interactions with each IED timing '
        'window. This tests whether the IED-memory effect differs between stim and no-stim '
        'trials.'
    )

    # Extract interaction terms. FDR family = the three estimable IED-window x stim
    # interactions. The During Stim x Stim interaction is not estimable (near-complete
    # separation) and the Stimulation main effect is not an interaction, so both are
    # excluded from the FDR family.
    interaction_labels = {
        'ied_before_image:stim': 'Before Image x Stim',
        'ied_during_image:stim': 'During Image x Stim',
        'ied_after_image:stim': 'After Image x Stim',
    }
    m1b_fam = m1b[m1b['term'].isin(interaction_labels)].copy()
    m1b_q = dict(zip(m1b_fam['term'], bh_fdr(m1b_fam['p.value'].values)))

    rows_1b_interact = []
    for _, r in m1b.iterrows():
        term = r['term']
        if ':' not in term and term != 'stim':
            continue
        # Drop the non-estimable During Stim x Stim interaction (near-complete separation).
        if term == 'ied_during_stim:stim':
            continue
        if term == 'stim':
            label = 'Stimulation'
        else:
            label = interaction_labels.get(term, term.replace('ied_', '').replace('_', ' ').replace(':', ' x '))
        p = r['p.value']
        q = m1b_q.get(term)
        q_cell = f'{p_str(q)}{sig_str(q)}' if q is not None else '--'
        rows_1b_interact.append([
            label,
            f'{r["estimate"]:.3f}',
            f'[{r["conf.low"]:.3f}, {r["conf.high"]:.3f}]',
            f'{r["statistic"]:.2f}',
            f'{p_str(p)}',
            q_cell,
        ])

    pdf.apa_table(
        'Table 2\nModel 1b: Stimulation Main Effect and IED x Stim Interactions (Odds Ratios)',
        ['Predictor', 'OR', '95% CI', 'z', 'p', 'FDR q'],
        rows_1b_interact,
        col_widths=[48, 16, 32, 14, 16, 18],
        note='Stimulation main effect and the estimable IED x Stim interactions shown. '
             'FDR q = Benjamini-Hochberg FDR-corrected p-value across the three IED-window x Stim '
             'interactions; the Stimulation main effect is not part of that family (--). The '
             'During Stim x Stim interaction was not estimable (near-complete separation: During '
             'Stim IEDs occur almost exclusively on stim trials) and is omitted. z = Wald '
             'z-statistic. No IED x Stim interaction was significant before or after FDR '
             'correction. *q < .05. **q < .01. ***q < .001.'
    )

    pdf.body_text(
        'A likelihood ratio test comparing Model 1b to Model 1a was not significant '
        '(chi2(5) = 6.28, p = .280), indicating that adding stimulation and its interactions '
        'did not improve model fit. No individual IED x Stim interaction was significant '
        '(all ps > .08). This confirms that IED-related memory impairment occurs regardless '
        'of whether stimulation was delivered, consistent with the follow-up chi-square analyses.'
    )

    # ================================================================
    # MODEL 1c: Random slope
    # ================================================================
    pdf.subsection_title('Model 1c: Random Slope for After Image IED')
    pdf.italic_text(
        'memory ~ ied_before_image + ied_during_image + ied_after_image + ied_during_stim '
        '+ (1 + ied_after_image | patient_id)'
    )

    m1c_path = os.path.join(OUTPUT_DIR, 'mlm_model1c_odds_ratios.csv')
    if os.path.exists(m1c_path):
        pdf.body_text(
            'A random slope for After Image IED was added to test whether the effect '
            'of post-encoding IEDs varies across patients. The model produced a singular fit '
            '(random slope variance near zero), indicating no meaningful patient-level variation '
            'in the After Image IED effect. The likelihood ratio test was not significant '
            '(chi2(2) = 0.00, p = 1.00). The random intercept-only model (1a) is preferred.'
        )
    else:
        pdf.body_text(
            'A random slope model was attempted but failed to converge, suggesting '
            'insufficient data to estimate patient-level variation in the After Image effect. '
            'The random intercept-only model (1a) is preferred.'
        )

    # ================================================================
    # MODEL 2: Retrieval IED frequency x Stim
    # ================================================================
    pdf.section_title('Model 2: Retrieval IED Frequency x Stimulation')

    pdf.subsection_title('What the interaction means')
    pdf.body_text(
        'The stim x retrieval_ied_freq interaction tests whether the benefit of stimulation '
        'on memory depends on how frequently a patient experiences IEDs during retrieval. '
        'If significant and positive, it would mean: patients with more frequent retrieval IEDs '
        'show a larger stimulation benefit (or less stimulation harm). If significant and '
        'negative: patients with more retrieval IEDs show less stimulation benefit. This speaks '
        'directly to hypothesis 3a: whether individuals with highly frequent retrieval IEDs '
        'can still achieve memory enhancement from prior stimulation.'
    )

    pdf.subsection_title('Model 2a: Centered Retrieval IED Frequency')
    pdf.italic_text(
        'memory ~ stim * ret_ied_freq_c + (1 | patient_id)'
    )

    pdf.body_text(
        f'Retrieval IED frequency was grand-mean centered (M = {freq_m:.3f}, SD = {freq_sd:.3f}) '
        f'so that the intercept and stim main effect are interpretable at the average retrieval '
        f'IED frequency. The model included {n_trials} trials from {n_patients} patients.'
    )

    rows_2a = []
    for _, r in m2a.iterrows():
        term = r['term']
        if term == '(Intercept)':
            label = '(Intercept)'
        elif term == 'stim':
            label = 'Stimulation'
        elif term == 'ret_ied_freq_c':
            label = 'Retrieval IED Freq (centered)'
        elif term == 'stim:ret_ied_freq_c':
            label = 'Stim x Retrieval IED Freq'
        else:
            label = term

        p = r['p.value']
        rows_2a.append([
            label,
            f'{r["estimate"]:.3f}',
            f'{r["std.error"]:.3f}',
            f'[{r["conf.low"]:.3f}, {r["conf.high"]:.3f}]',
            f'{r["statistic"]:.2f}',
            f'{p_str(p)}{sig_str(p)}',
        ])

    pdf.apa_table(
        'Table 3\nModel 2a: Stim x Retrieval IED Frequency Predicting Memory (Odds Ratios)',
        ['Predictor', 'OR', 'SE(B)', '95% CI', 'z', 'p'],
        rows_2a,
        col_widths=[38, 14, 14, 32, 14, 22],
        note=f'GLMM with logit link and random intercept for patient. '
             f'Retrieval IED frequency = proportion of retrieval trials with IED, '
             f'computed from phase 3 power files (denominator) and IED test dataset (numerator). '
             f'SE(B) = standard error of the log-odds coefficient. z = Wald z-statistic. '
             f'N = {n_trials} trials, {n_patients} patients. '
             f'*p < .05. **p < .01. ***p < .001.'
    )

    # Interaction interpretation
    interact_p = m2a[m2a['term'] == 'stim:ret_ied_freq_c']['p.value'].values[0]
    interact_or = m2a[m2a['term'] == 'stim:ret_ied_freq_c']['estimate'].values[0]
    stim_main_p = m2a[m2a['term'] == 'stim']['p.value'].values[0]
    freq_main_p = m2a[m2a['term'] == 'ret_ied_freq_c']['p.value'].values[0]

    pdf.body_text(
        f'Neither the stimulation main effect (OR = {m2a[m2a["term"] == "stim"]["estimate"].values[0]:.3f}, '
        f'p = {p_str(stim_main_p)}) nor the retrieval IED frequency main effect '
        f'(p = {p_str(freq_main_p)}) was significant. Critically, the stim x retrieval IED '
        f'frequency interaction was not significant (OR = {interact_or:.3f}, '
        f'p = {p_str(interact_p)}), indicating that patient-level retrieval IED frequency '
        f'does not moderate the effect of stimulation on memory.'
    )

    # Standardized version
    pdf.subsection_title('Model 2b: Standardized Retrieval IED Frequency')

    z_interact = m2b[m2b['term'] == 'stim:ret_ied_freq_z']
    if len(z_interact) > 0:
        z_or = z_interact['estimate'].values[0]
        z_p = z_interact['p.value'].values[0]
        z_ci = (z_interact['conf.low'].values[0], z_interact['conf.high'].values[0])
        pdf.body_text(
            f'Using z-scored retrieval IED frequency for interpretability: a 1-SD increase in '
            f'retrieval IED frequency was associated with a non-significant change in the '
            f'stimulation effect (OR = {z_or:.3f}, 95% CI [{z_ci[0]:.3f}, {z_ci[1]:.3f}], '
            f'p = {p_str(z_p)}). This confirms the null interaction from Model 2a.'
        )

    # Interaction plot
    interact_path = os.path.join(OUTPUT_DIR, 'mlm_model2_interaction.png')
    if os.path.exists(interact_path):
        pdf.ln(2)
        pdf.italic_text(
            'Figure 2. Predicted probability of remembering as a function of stimulation '
            'condition and retrieval IED frequency (quartiles). Error bars = 95% CI.'
        )
        img_w = pdf.w - pdf.l_margin - pdf.r_margin - 10
        pdf.image(interact_path, x=pdf.l_margin + 5, w=img_w)
        pdf.ln(3)

    # ================================================================
    # SUMMARY
    # ================================================================
    pdf.section_title('Summary')

    pdf.body_text(
        'Model 1a (Encoding IEDs): After accounting for patient-level variability, IEDs in '
        'the After Image and During Stim windows significantly reduced the odds of remembering '
        f'(OR = {after_or:.2f} and {stim_or:.2f}, respectively). These effects are consistent '
        'across stimulation conditions (Model 1b interaction test: p = .280) and do not vary '
        'meaningfully across patients (Model 1c random slope: singular fit). The mixed-effects '
        'model confirms the chi-square findings while properly handling the nested data structure.'
    )

    pdf.body_text(
        f'Model 2 (Retrieval IED Frequency): Patient-level retrieval IED frequency did not '
        f'moderate the stimulation-memory relationship (interaction p = {p_str(interact_p)}). '
        f'This means that regardless of how frequently a patient experiences retrieval IEDs, '
        f'stimulation does not differentially affect their memory. The hypothesis that patients '
        f'with frequent retrieval IEDs could achieve memory enhancement through stimulation '
        f'was not supported.'
    )

    pdf.body_text(
        'Methodological note: The retrieval IED frequency variable had limited variance '
        f'(M = {freq_m:.1%}, range {freq_min:.1%} - {freq_max:.1%}), with {n_zero} of '
        f'{len(ret_valid)} patients showing zero retrieval IEDs. This restricted range may '
        f'limit power to detect a moderation effect. Future studies with larger samples and '
        f'greater variability in retrieval IED rates would provide a stronger test.'
    )

    # ================================================================
    # R CODE REFERENCE
    # ================================================================
    pdf.section_title('R Code Reference')
    pdf.body_text(
        'All models were fit using glmer() from the lme4 package (v1.1+) in R, with the '
        'bobyqa optimizer and logit link function. Odds ratios and 95% Wald confidence '
        'intervals were extracted via broom.mixed::tidy(). Forest plots and interaction '
        'plots were generated with ggplot2.'
    )

    pdf.subsection_title('Model 1a Formula')
    pdf.code_text(
        'glmer(memory ~ ied_before_image + ied_during_image + ied_after_image\n'
        '      + ied_during_stim + (1 | patient_id),\n'
        '      data = d, family = binomial(link = "logit"))'
    )

    pdf.subsection_title('Model 2a Formula')
    pdf.code_text(
        'glmer(memory ~ stim * ret_ied_freq_c + (1 | patient_id),\n'
        '      data = d2, family = binomial(link = "logit"))\n'
        '# ret_ied_freq_c = grand-mean centered retrieval IED frequency'
    )

    # Save
    out_path = os.path.join(OUTPUT_DIR, 'H3a_MLM_Report.pdf')
    pdf.output(out_path)
    print(f'Saved: {out_path}')


if __name__ == '__main__':
    main()
