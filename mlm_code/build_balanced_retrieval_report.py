#!/usr/bin/env python
"""
Build PDF report: Does Baseline Neural Activity x Stimulation Impact Memory?
Encoding-phase balanced-trials MLM results for power, coherence, and PAC.
"""

import os
import numpy as np
import pandas as pd
from fpdf import FPDF

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, 'outputs', 'balanced_retrieval_mlm')
REPORT_TITLE = 'Does Baseline Neural Activity x Stim Impact Memory - Retrieval'
OUTPUT_PDF = os.path.join(SCRIPT_DIR, 'outputs',
    'Does_Baseline_Neural_Activity_x_Stim_Impact_Memory_Retrieval_Balanced_Trials.pdf')


# ── Region labels ──────────────────────────────────────────────────────────

REGION_LABELS = {
    'BLA': 'Amygdala', 'HPC': 'HPC', 'CA': 'CA', 'DG': 'DG',
    'EC': 'EC', 'PRC': 'PRC', 'PHG': 'PHG',
}

def display_region(r):
    """BLA_HPC -> Amygdala-HPC, BLA -> Amygdala."""
    parts = r.split('_')
    return '-'.join(REGION_LABELS.get(p, p) for p in parts)


BAND_LABELS = {
    'theta': 'Theta (4-8 Hz)',
    'slow_gamma': 'Slow Gamma (30-55 Hz)',
    'fast_gamma': 'HFA (70-100 Hz)',
}

SET_LABELS = {
    'MTL': 'MTL Cortical (Amygdala, HPC, EC, PRC)',
    'HPC_subfields': 'HPC Subfields (Amygdala, CA, DG)',
}

TERM_LABELS = {
    '(Intercept)': 'Intercept',
    'StimCondstim': 'StimCond [stim]',
    'band_c': 'Band power (centered)',
    'StimCondstim:band_c': 'Band x StimCond [stim]',
    'slow_gamma_pac_c': 'Slow Gamma PAC (centered)',
    'hfa_pac_c': 'HFA PAC (centered)',
    'slow_gamma_pac_c:StimCondstim': 'SG PAC x StimCond [stim]',
    'hfa_pac_c:StimCondstim': 'HFA PAC x StimCond [stim]',
    'slow_gamma_pac_z': 'SG PAC (z)',
    'hfa_pac_z': 'HFA PAC (z)',
    'slow_gamma_pac_z:StimCondstim': 'SG PAC (z) x StimCond [stim]',
    'hfa_pac_z:StimCondstim': 'HFA PAC (z) x StimCond [stim]',
}

# ── Sample sizes from the R output ────────────────────────────────────────

POWER_SAMPLES = {
    'MTL': {'n_trials': 9920, 'n_patients': 36,
            'regions': 'Amygdala, EC, HPC, PRC',
            'mem_rate': 68.9, 'stim_rate': 61.0},
    'HPC_subfields': {'n_trials': 8360, 'n_patients': 37,
                      'regions': 'Amygdala, CA, DG',
                      'mem_rate': 66.6, 'stim_rate': 62.4},
}

COH_SAMPLES = {
    'MTL': {'n_trials': 8960, 'n_patients': 34,
            'regions': 'Amygdala-EC, Amygdala-HPC, Amygdala-PRC, EC-HPC, EC-PRC, HPC-PRC',
            'mem_rate': None, 'stim_rate': None},
    'HPC_subfields': {'n_trials': 5640, 'n_patients': 29,
                      'regions': 'Amygdala-CA, Amygdala-DG, CA-DG',
                      'mem_rate': None, 'stim_rate': None},
}

# PAC sample sizes from R output
PAC_SAMPLES = {
    'BLA_CA': {'n': 2718, 'k': 17}, 'BLA_DG': {'n': 640, 'k': 4},
    'BLA_EC': {'n': 1440, 'k': 9}, 'BLA_HPC': {'n': 2880, 'k': 18},
    'BLA_PRC': {'n': 1440, 'k': 9},
    'CA_DG': {'n': 800, 'k': 5}, 'CA_EC': {'n': 1440, 'k': 9},
    'CA_HPC': {'n': 2880, 'k': 18}, 'CA_PRC': {'n': 1280, 'k': 8},
    'DG_HPC': {'n': 800, 'k': 5},
    'EC_HPC': {'n': 1600, 'k': 10}, 'EC_PRC': {'n': 1280, 'k': 8},
    'HPC_PRC': {'n': 1440, 'k': 9},
}

# Intra-hippocampal pairs to exclude from PAC (doesn't make sense)
EXCLUDE_PAC_PAIRS = {'CA_DG', 'CA_HPC', 'DG_HPC'}


# ── PDF class ─────────────────────────────────────────────────────────────

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

    def subsubsection_title(self, title):
        self.set_font('Helvetica', 'BI', 10)
        self.ln(1)
        self.cell(0, 6, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def body_text(self, text):
        self.set_font('Times', '', 11)
        self.multi_cell(0, 5.5, text)
        self.ln(1)

    def italic_text(self, text):
        self.set_font('Times', 'I', 10)
        self.multi_cell(0, 5, text)
        self.ln(1)

    def bold_text(self, text):
        self.set_font('Times', 'B', 11)
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
        # Top rule
        self.set_line_width(0.5)
        self.line(x_start, self.get_y(), x_start + total_w, self.get_y())
        self.ln(1)
        # Header row
        self.set_font('Times', 'B', 9)
        for i, h in enumerate(headers):
            self.cell(col_widths[i], 5, str(h), align='C')
        self.ln()
        self.set_line_width(0.3)
        self.line(x_start, self.get_y(), x_start + total_w, self.get_y())
        self.ln(1)
        # Body
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
        # Bottom rule
        self.set_line_width(0.5)
        self.line(x_start, self.get_y(), x_start + total_w, self.get_y())
        self.ln(1)
        if note:
            self.set_font('Times', 'I', 8)
            self.multi_cell(0, 4, f'Note. {note}')
            self.ln(2)


# ── Helpers ───────────────────────────────────────────────────────────────

def p_str(p):
    if isinstance(p, str): return p
    if p < .001: return '< .001'
    return f'{p:.3f}'.lstrip('0')

def sig_str(p):
    if isinstance(p, str): return ''
    if p < .001: return '***'
    elif p < .01: return '**'
    elif p < .05: return '*'
    elif p < .10: return '+'  # marginal
    return ''

def or_str(v):
    if abs(v) > 1e4 or abs(v) < 1e-4:
        return f'{v:.2e}'
    return f'{v:.4f}'

def ci_str(lo, hi):
    if abs(lo) > 1e4 or abs(hi) > 1e4 or abs(lo) < 1e-4:
        return f'[{lo:.2e}, {hi:.2e}]'
    return f'[{lo:.4f}, {hi:.4f}]'

def load_coefs(filename):
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)

def label_term(term):
    return TERM_LABELS.get(term, term)

def get_interaction_row(df, pattern):
    """Get the interaction term row from coefficient dataframe."""
    mask = df['term'].str.contains(pattern, na=False)
    if mask.any():
        return df[mask].iloc[0]
    return None


# ── Build report ──────────────────────────────────────────────────────────

def build_report():
    pdf = APAReport()
    pdf.add_page()

    # ── Title ──
    pdf.set_font('Helvetica', 'B', 16)
    pdf.multi_cell(0, 9, 'Does Baseline Neural Activity x Stimulation\nImpact Subsequent Memory?')
    pdf.ln(2)
    pdf.set_font('Helvetica', '', 11)
    pdf.cell(0, 6, 'Retrieval Phase - Balanced Trials Analysis', new_x="LMARGIN", new_y="NEXT")
    pdf.ln(6)

    # ── Section 1: Overview ──
    pdf.section_title('1. Overview')
    pdf.body_text(
        'This report tests whether baseline (pre-stimulation) neural activity '
        'differentially predicts subsequent memory as a function of stimulation '
        'condition. The central question is: when endogenous neural activity '
        '(power, coherence, or phase-amplitude coupling) is already high in '
        'theta and/or slow gamma frequencies, does adding BLA stimulation '
        'disrupt subsequent memory?'
    )
    pdf.body_text(
        'All models use trial-level data from the retrieval phase with a '
        'balanced-trials filter: subjects with fewer than 10 trials in either '
        'the remembered or forgotten condition were excluded to prevent '
        'extreme estimates from imbalanced memory outcomes.'
    )

    pdf.subsection_title('Stimulation Protocol')
    pdf.body_text(
        'Stimulation was theta-modulated gamma (8 x 50 Hz) delivered to the '
        'amygdala (BLA) after encoding. Baseline neural activity was measured '
        'during the retrieval window. The key question is whether this '
        'endogenous retrieval-phase state interacts with the exogenous '
        'stimulation to predict subsequent memory.'
    )

    pdf.subsection_title('Statistical Approach')
    pdf.body_text(
        'All models are trial-level generalized linear mixed models (GLMMs) '
        'with binomial family (logit link), random intercept for patient, and '
        'bobyqa optimizer (maxfun = 200,000). Results are reported as odds '
        'ratios (OR) with 95% Wald confidence intervals. The critical test in '
        'each model is the Band x StimCond interaction term, which indicates '
        'whether the relationship between neural activity and memory differs '
        'between stim and no-stim trials.'
    )
    pdf.body_text(
        'Models for power and coherence pool across regions within each region '
        'set, including Region as a covariate and a Band x Region interaction. '
        'PAC models are fit separately per region pair because PAC is computed '
        'between specific region pairs.'
    )

    # ── Section 2: Subject Exclusions ──
    pdf.section_title('2. Subject Exclusions')

    excl_power = [
        ['BJH025', '74', '6'], ['BJH027', '72', '8'], ['BJH029', '71', '9'],
        ['BJH040', '77', '3'], ['BJH042', '73', '6'], ['SLCH018', '73', '7'],
        ['amyg003', '9', '71'], ['amyg008', '74', '6'], ['amyg009', '73', '7'],
        ['amyg033', '115', '5'],
    ]
    pdf.apa_table(
        'Table 1. Power: Excluded Subjects (< 10 trials in one condition)',
        ['Patient', 'Remembered', 'Forgotten'],
        excl_power,
        col_widths=[50, 40, 40],
        note='37 of 47 patients included. Trial counts are per-region unique trials.'
    )

    excl_coh = [
        ['BJH025', '74', '6'], ['BJH027', '72', '8'], ['BJH029', '71', '9'],
        ['BJH040', '77', '3'], ['BJH042', '73', '6'],
        ['amyg003', '8', '60'], ['amyg008', '62', '5'],
        ['amyg009', '49', '5'], ['amyg033', '115', '5'],
    ]
    pdf.apa_table(
        'Table 2. Coherence: Excluded Subjects',
        ['Patient', 'Remembered', 'Forgotten'],
        excl_coh,
        col_widths=[50, 40, 40],
        note='36 of 45 patients included.'
    )

    pdf.body_text(
        'PAC: 22 of 23 patients included. Only BJH040 was excluded '
        '(remembered = 154, forgotten = 6).'
    )

    # ── Section 3: Power Results ──
    pdf.section_title('3. Power x Stimulation Condition')
    pdf.body_text(
        'Model: Accuracy ~ StimCond + Region + band_c + band_c:StimCond '
        '+ band_c:Region + (1 | Patient)'
    )
    pdf.body_text(
        'Band power was computed as the mean of baseline-corrected spectral '
        'power across frequencies in each band (theta: 4-8 Hz, slow gamma: '
        '30-55 Hz, HFA: 70-100 Hz), then grand-mean centered. The Band x '
        'StimCond interaction tests whether the power-memory relationship '
        'differs between stim and no-stim trials.'
    )

    for region_set in ['MTL', 'HPC_subfields']:
        info = POWER_SAMPLES[region_set]
        pdf.subsection_title(f'3.{1 if region_set == "MTL" else 2}. {SET_LABELS[region_set]}')
        pdf.body_text(
            f'N = {info["n_patients"]} patients, {info["n_trials"]:,} trials. '
            f'Regions: {info["regions"]}. '
            f'Memory rate: {info["mem_rate"]}%. Stim rate: {info["stim_rate"]}%.'
        )

        summary_rows = []
        for band_key, band_label in BAND_LABELS.items():
            fname = f'power_{region_set}_{band_key}_coefs.csv'
            df = load_coefs(fname)
            if df is None:
                continue
            row = get_interaction_row(df, 'StimCondstim:band_c')
            if row is not None:
                summary_rows.append([
                    band_label,
                    or_str(row['estimate']),
                    ci_str(row['conf.low'], row['conf.high']),
                    f'{row["statistic"]:.3f}',
                    p_str(row['p.value']),
                    sig_str(row['p.value']),
                ])

        pdf.apa_table(
            f'Table. Power x StimCond Interaction - {SET_LABELS[region_set]}',
            ['Band', 'OR', '95% CI', 'z', 'p', ''],
            summary_rows,
            col_widths=[40, 18, 45, 15, 18, 8],
            note='OR = odds ratio for the Band x StimCond [stim] interaction. '
                 'OR > 1 means higher power is more beneficial for memory on '
                 'stim trials than nostim trials.'
        )

        # Full coefficient tables per band
        for band_key, band_label in BAND_LABELS.items():
            fname = f'power_{region_set}_{band_key}_coefs.csv'
            df = load_coefs(fname)
            if df is None:
                continue
            rows = []
            for _, r in df.iterrows():
                rows.append([
                    label_term(r['term']),
                    or_str(r['estimate']),
                    ci_str(r['conf.low'], r['conf.high']),
                    f'{r["statistic"]:.3f}',
                    p_str(r['p.value']),
                    sig_str(r['p.value']),
                ])
            # Add region terms labels
            for i, row in enumerate(rows):
                term = df.iloc[i]['term']
                if term.startswith('Region') and ':' not in term:
                    region_name = term.replace('Region', '')
                    rows[i][0] = f'Region [{display_region(region_name)}]'
                elif term.startswith('Region') and ':band_c' in term:
                    region_name = term.replace('Region', '').replace(':band_c', '')
                    rows[i][0] = f'Band x Region [{display_region(region_name)}]'

            pdf.apa_table(
                f'Table. Full Model: Power {band_label} - {SET_LABELS[region_set]}',
                ['Predictor', 'OR', '95% CI', 'z', 'p', ''],
                rows,
                col_widths=[48, 18, 40, 15, 18, 8],
            )

    pdf.subsection_title('3.3. Power Summary')
    pdf.body_text(
        'The only trend was MTL HFA power x StimCond (OR = 1.03, p = .090), '
        'a marginal effect suggesting slightly better memory on stim trials '
        'when HFA power was higher. All other power x stimulation interactions '
        'were non-significant.'
    )
    pdf.body_text(
        'Interpretation: Retrieval-phase spectral power does not meaningfully '
        'interact with stimulation condition to predict memory. The marginal '
        'MTL HFA trend is small in magnitude (3% change in odds per unit).'
    )

    # ── Section 4: Coherence Results ──
    pdf.section_title('4. Coherence x Stimulation Condition')
    pdf.body_text(
        'Model: Accuracy ~ StimCond + Region + band_c + band_c:StimCond '
        '+ band_c:Region + (1 | Patient)'
    )
    pdf.body_text(
        'Coherence was computed between each region pair. Band-averaged '
        'coherence was grand-mean centered. Region here refers to the region '
        'pair (e.g., Amygdala-HPC).'
    )

    for region_set in ['MTL', 'HPC_subfields']:
        info = COH_SAMPLES[region_set]
        pdf.subsection_title(f'4.{1 if region_set == "MTL" else 2}. Coherence - {SET_LABELS[region_set]}')
        pdf.body_text(
            f'N = {info["n_patients"]} patients, {info["n_trials"]:,} trials. '
            f'Region pairs: {info["regions"]}.'
        )

        summary_rows = []
        for band_key, band_label in BAND_LABELS.items():
            fname = f'coherence_{region_set}_{band_key}_coefs.csv'
            df = load_coefs(fname)
            if df is None:
                continue
            row = get_interaction_row(df, 'StimCondstim:band_c')
            if row is not None:
                summary_rows.append([
                    band_label,
                    or_str(row['estimate']),
                    ci_str(row['conf.low'], row['conf.high']),
                    f'{row["statistic"]:.3f}',
                    p_str(row['p.value']),
                    sig_str(row['p.value']),
                ])

        pdf.apa_table(
            f'Table. Coherence x StimCond Interaction - {SET_LABELS[region_set]}',
            ['Band', 'OR', '95% CI', 'z', 'p', ''],
            summary_rows,
            col_widths=[40, 18, 45, 15, 18, 8],
            note='OR = odds ratio for the Band x StimCond [stim] interaction.'
        )

        # Full coefficient tables
        for band_key, band_label in BAND_LABELS.items():
            fname = f'coherence_{region_set}_{band_key}_coefs.csv'
            df = load_coefs(fname)
            if df is None:
                continue
            rows = []
            for _, r in df.iterrows():
                term = r['term']
                display = label_term(term)
                if term.startswith('Region') and ':' not in term:
                    region_name = term.replace('Region', '')
                    display = f'Region [{display_region(region_name)}]'
                elif term.startswith('Region') and ':band_c' in term:
                    region_name = term.replace('Region', '').replace(':band_c', '')
                    display = f'Band x Region [{display_region(region_name)}]'
                rows.append([
                    display,
                    or_str(r['estimate']),
                    ci_str(r['conf.low'], r['conf.high']),
                    f'{r["statistic"]:.3f}',
                    p_str(r['p.value']),
                    sig_str(r['p.value']),
                ])

            pdf.apa_table(
                f'Table. Full Model: Coherence {band_label} - {SET_LABELS[region_set]}',
                ['Predictor', 'OR', '95% CI', 'z', 'p', ''],
                rows,
                col_widths=[52, 18, 40, 15, 15, 8],
            )

    pdf.subsection_title('4.3. Coherence Summary')
    pdf.body_text(
        'HPC subfield theta coherence x StimCond was significant '
        '(OR = 1.30, p = .039), indicating that higher theta coherence '
        'among amygdala-hippocampal subfield pairs was associated with '
        'better memory specifically on stim trials during retrieval. MTL '
        'slow gamma coherence showed a marginal trend (OR = 1.55, p = .094).'
    )
    pdf.body_text(
        'Unlike encoding (where coherence interactions were null), retrieval '
        'shows evidence that theta-band functional connectivity between '
        'amygdala and hippocampal subfields interacts with stimulation '
        'condition to predict memory.'
    )

    # ── Section 5: PAC Results (Z-scored) ──
    pdf.section_title('5. Phase-Amplitude Coupling (PAC) x Stimulation Condition')
    pdf.body_text(
        'Model (per region pair): Accuracy ~ (slow_gamma_pac_z + hfa_pac_z) '
        '* StimCond + (1 | Patient)'
    )
    pdf.body_text(
        'PAC measures theta-phase x amplitude coupling at two frequency '
        'ranges: slow gamma (30-50 Hz) and HFA (70-100 Hz). Models were fit '
        'separately for each region pair. PAC predictors were z-scored '
        '(standardized) within each region pair so that coefficients represent '
        'the change in log-odds of remembering per 1 SD increase in PAC.'
    )

    pdf.body_text(
        'Note: Intra-hippocampal pairs (CA-DG, CA-HPC, DG-HPC) were excluded '
        'because PAC within the same structure (hippocampal subfields) is not '
        'a meaningful between-region coupling measure.'
    )

    pdf.subsection_title('5.1. Note on Predictor Scaling')
    pdf.body_text(
        'Initial models using grand-mean centered (but unscaled) PAC '
        'predictors produced astronomically extreme odds ratios (10^7 to '
        '10^15) due to quasi-complete separation. PAC values are very small '
        'in magnitude (SD ~ 0.01-0.02), so a 1-unit change in raw PAC is '
        'far beyond the actual data range. Z-scoring resolved this entirely, '
        'producing stable estimates (OR range: 0.7-1.6). The model fit '
        'statistics (AIC, log-likelihood) are identical because z-scoring '
        'is a linear transformation; only the coefficient scale changes.'
    )

    # Collect z-scored PAC results
    pac_z_files = sorted([f for f in os.listdir(DATA_DIR)
                          if f.startswith('pac_zscored_') and f.endswith('_coefs.csv')
                          and f.replace('pac_zscored_', '').replace('_coefs.csv', '')
                              not in EXCLUDE_PAC_PAIRS])

    # Summary table: all interactions
    pac_all_rows = []
    pac_sig_rows = []
    for fname in pac_z_files:
        region = fname.replace('pac_zscored_', '').replace('_coefs.csv', '')
        df = load_coefs(fname)
        if df is None:
            continue
        samples = PAC_SAMPLES.get(region, {})
        n = samples.get('n', '?')
        k = samples.get('k', '?')

        for pac_type, pattern in [('SG PAC', 'slow_gamma_pac_z:StimCond'),
                                   ('HFA PAC', 'hfa_pac_z:StimCond')]:
            row = get_interaction_row(df, pattern)
            if row is None:
                continue
            entry = [
                display_region(region),
                pac_type,
                or_str(row['estimate']),
                ci_str(row['conf.low'], row['conf.high']),
                f'{row["statistic"]:.2f}',
                p_str(row['p.value']),
                sig_str(row['p.value']),
            ]
            pac_all_rows.append(entry)
            if row['p.value'] < 0.10:
                pac_sig_rows.append(entry + [str(n), str(k)])

    # Significant / marginal PAC interactions
    pdf.subsection_title('5.2. Significant and Marginal PAC x StimCond Interactions')
    if pac_sig_rows:
        pdf.apa_table(
            'Table. PAC x StimCond Interactions (p < .10)',
            ['Region Pair', 'PAC', 'OR', '95% CI', 'z', 'p', '', 'Trials', 'N'],
            pac_sig_rows,
            col_widths=[26, 17, 14, 32, 10, 13, 7, 14, 10],
            note='OR per 1 SD increase in PAC. OR > 1 means higher PAC is more '
                 'beneficial for memory on stim trials. * p < .05, + p < .10.'
        )
    else:
        pdf.body_text('No PAC x StimCond interactions reached p < .10.')

    # Full table: all region pairs
    pdf.subsection_title('5.3. All PAC x StimCond Interactions')
    pdf.apa_table(
        'Table. All PAC x StimCond Interactions (Z-scored)',
        ['Region Pair', 'PAC', 'OR', '95% CI', 'z', 'p', ''],
        pac_all_rows,
        col_widths=[26, 17, 14, 34, 12, 14, 7],
        note='OR per 1 SD increase in z-scored PAC within region pair. '
             '* p < .05, + p < .10.'
    )

    # Full coefficient tables for significant pairs
    sig_regions = set()
    for fname in pac_z_files:
        region = fname.replace('pac_zscored_', '').replace('_coefs.csv', '')
        df = load_coefs(fname)
        if df is None:
            continue
        for pattern in ['slow_gamma_pac_z:StimCond', 'hfa_pac_z:StimCond']:
            row = get_interaction_row(df, pattern)
            if row is not None and row['p.value'] < 0.05:
                sig_regions.add(region)

    for region in sorted(sig_regions):
        fname = f'pac_zscored_{region}_coefs.csv'
        df = load_coefs(fname)
        if df is None:
            continue
        samples = PAC_SAMPLES.get(region, {})
        rows = []
        for _, r in df.iterrows():
            term = r['term']
            display = label_term(term)
            # Handle z-scored terms
            display = display.replace('slow_gamma_pac_z', 'SG PAC (z)')
            display = display.replace('hfa_pac_z', 'HFA PAC (z)')
            if 'pac_z' in term and 'StimCond' in term:
                if 'slow_gamma' in term:
                    display = 'SG PAC (z) x StimCond [stim]'
                else:
                    display = 'HFA PAC (z) x StimCond [stim]'
            elif 'pac_z' in term:
                if 'slow_gamma' in term:
                    display = 'SG PAC (z)'
                else:
                    display = 'HFA PAC (z)'
            rows.append([
                display,
                or_str(r['estimate']),
                ci_str(r['conf.low'], r['conf.high']),
                f'{r["statistic"]:.3f}',
                p_str(r['p.value']),
                sig_str(r['p.value']),
            ])

        pdf.apa_table(
            f'Table. Full Model: PAC {display_region(region)} '
            f'(N = {samples.get("k", "?")} patients, {samples.get("n", "?")} trials)',
            ['Predictor', 'OR', '95% CI', 'z', 'p', ''],
            rows,
            col_widths=[48, 16, 40, 14, 16, 8],
        )

    pdf.subsection_title('5.4. PAC Summary')
    pdf.body_text(
        'No PAC x StimCond interactions reached significance (p < .05) during '
        'retrieval. The only marginal trend was Amygdala-HPC slow gamma PAC x '
        'StimCond (OR = 1.21, p = .056), suggesting a weak tendency for higher '
        'theta-slow gamma coupling between amygdala and hippocampus to be '
        'associated with better memory on stim trials.'
    )
    pdf.body_text(
        'This contrasts sharply with encoding, where HFA PAC x StimCond was '
        'significant in three amygdala pairs (Amygdala-CA, Amygdala-HPC, '
        'Amygdala-PRC, all p < .02). The retrieval phase shows no evidence '
        'that baseline theta-HFA coupling interacts with stimulation condition '
        'to predict memory.'
    )
    pdf.body_text(
        'Intra-hippocampal pairs (CA-DG, CA-HPC, DG-HPC) were excluded '
        'because within-structure PAC is not meaningful for this analysis.'
    )

    # ── Section 6: Overall Summary ──
    pdf.section_title('6. Overall Summary')

    # Summary table
    summary_headers = ['Measure', 'Region Set', 'Band', 'OR', 'p', '']
    summary_data = []
    # Power
    for rs in ['MTL', 'HPC_subfields']:
        for bk, bl in BAND_LABELS.items():
            df = load_coefs(f'power_{rs}_{bk}_coefs.csv')
            if df is None: continue
            row = get_interaction_row(df, 'StimCondstim:band_c')
            if row is not None:
                summary_data.append([
                    'Power', SET_LABELS[rs].split(' (')[0],
                    bl.split(' (')[0],
                    or_str(row['estimate']),
                    p_str(row['p.value']),
                    sig_str(row['p.value']),
                ])
    # Coherence
    for rs in ['MTL', 'HPC_subfields']:
        for bk, bl in BAND_LABELS.items():
            df = load_coefs(f'coherence_{rs}_{bk}_coefs.csv')
            if df is None: continue
            row = get_interaction_row(df, 'StimCondstim:band_c')
            if row is not None:
                summary_data.append([
                    'Coherence', SET_LABELS[rs].split(' (')[0],
                    bl.split(' (')[0],
                    or_str(row['estimate']),
                    p_str(row['p.value']),
                    sig_str(row['p.value']),
                ])

    # Add key PAC interactions to summary (marginal or notable)
    for region, pac_type_key, pac_label in [
        ('BLA_HPC', 'slow_gamma_pac_z:StimCond', 'SG PAC'),
        ('BLA_HPC', 'hfa_pac_z:StimCond', 'HFA PAC'),
    ]:
        df = load_coefs(f'pac_zscored_{region}_coefs.csv')
        if df is None:
            continue
        row = get_interaction_row(df, pac_type_key)
        if row is not None:
            summary_data.append([
                'PAC', display_region(region),
                pac_label,
                or_str(row['estimate']),
                p_str(row['p.value']),
                sig_str(row['p.value']),
            ])

    pdf.apa_table(
        'Table. Summary of Key Band/PAC x StimCond Interactions',
        summary_headers,
        summary_data,
        col_widths=[24, 30, 28, 20, 20, 8],
        note='* p < .05, + p < .10 (marginal). PAC results use z-scored '
             'predictors (OR per 1 SD). Only significant/marginal PAC pairs shown.'
    )

    pdf.subsection_title('6.1. Key Findings')
    pdf.body_text(
        '1. Power x StimCond: Mostly null. The only trend was MTL HFA power '
        '(OR = 1.03, p = .090), a marginal effect. Retrieval-phase power '
        'does not meaningfully interact with stimulation condition.'
    )
    pdf.body_text(
        '2. Coherence x StimCond: HPC subfield theta coherence was '
        'significant (OR = 1.30, p = .039), indicating higher theta '
        'coherence among amygdala-hippocampal subfield pairs predicts better '
        'memory specifically on stim trials. MTL slow gamma coherence showed '
        'a marginal trend (OR = 1.55, p = .094).'
    )
    pdf.body_text(
        '3. PAC x StimCond: No significant interactions during retrieval. '
        'The only marginal trend was Amygdala-HPC slow gamma PAC (OR = 1.21, '
        'p = .056). This contrasts sharply with encoding, where HFA PAC x '
        'StimCond was significant in three amygdala pairs (all p < .02).'
    )
    pdf.body_text(
        '4. The retrieval pattern differs from encoding: during encoding, '
        'the dominant effect was PAC (theta-HFA coupling); during retrieval, '
        'the dominant effect is coherence (theta connectivity between '
        'amygdala and hippocampal subfields).'
    )

    # ── Section 7: Mechanistic Interpretation ──
    pdf.section_title('7. Mechanistic Interpretation')

    pdf.subsection_title('7.1. Sender Saturation Hypothesis')
    pdf.body_text(
        'The sender saturation hypothesis predicted that when amygdala '
        'endogenous theta/slow gamma activity was already high, exogenous '
        'theta-modulated gamma stimulation would be unable to effectively '
        'entrain or add to the signal, leading to worse memory on stim trials '
        'for those with high baseline power. This would manifest as a '
        'negative Band x StimCond interaction (OR < 1) specifically in '
        'amygdala theta and slow gamma.'
    )
    pdf.body_text(
        'Result: Not supported during retrieval. Power x StimCond interactions '
        'were non-significant (only MTL HFA marginal at OR = 1.03, p = .090). '
        'No evidence that high retrieval-phase power makes stimulation less '
        'effective. This is consistent with the encoding-phase null result.'
    )

    pdf.subsection_title('7.2. Encoding vs. Retrieval Differences')
    pdf.body_text(
        'The most striking finding is the phase-specificity of neural activity '
        'x stimulation interactions. During encoding, HFA PAC (theta-HFA '
        'coupling) between amygdala and hippocampal targets was the dominant '
        'predictor (3 significant pairs, all p < .02). During retrieval, '
        'PAC interactions were null, and instead theta coherence between '
        'amygdala and hippocampal subfields was significant (OR = 1.30, '
        'p = .039).'
    )
    pdf.body_text(
        'This suggests different neural mechanisms mediate the stim-memory '
        'relationship at each phase: local cross-frequency coupling (PAC) '
        'matters for encoding, while long-range theta connectivity '
        '(coherence) matters for retrieval.'
    )

    pdf.subsection_title('7.3. Theta Coherence and Retrieval')
    pdf.body_text(
        'The significant HPC subfield theta coherence x StimCond interaction '
        '(OR = 1.30, p = .039) indicates that when theta-band connectivity '
        'between amygdala and hippocampal subfields (CA, DG) is higher during '
        'retrieval, memory is better specifically on stim trials. This is '
        'consistent with theta oscillations supporting memory retrieval '
        'through coordinated reactivation between amygdala and hippocampus.'
    )
    pdf.body_text(
        'The stimulation protocol (theta-modulated gamma) may enhance '
        'retrieval when the amygdala-hippocampal theta network is already '
        'engaged, providing a complementary boost to endogenous theta '
        'synchrony.'
    )

    pdf.subsection_title('7.4. Integrated Model (Encoding + Retrieval)')
    pdf.body_text(
        'Combining encoding and retrieval results suggests a refined model:'
    )
    pdf.body_text(
        '(a) Sender saturation is not supported in either phase. High '
        'baseline power does not impair stimulation efficacy.'
    )
    pdf.body_text(
        '(b) During encoding, theta-HFA PAC between amygdala and '
        'hippocampal targets facilitates stimulation effects (OR = 1.38-1.57). '
        'Local cross-frequency coupling creates a receptive state for the '
        'theta-modulated gamma stimulation.'
    )
    pdf.body_text(
        '(c) During retrieval, theta coherence between amygdala and '
        'hippocampal subfields facilitates stimulation effects (OR = 1.30). '
        'Long-range theta synchrony supports coordinated reactivation.'
    )
    pdf.body_text(
        '(d) The key predictor of stimulation failure remains the receiver '
        'state (hippocampal IED burden), consistent with prior IED analyses.'
    )

    pdf.subsection_title('7.5. Clinical Implications')
    pdf.body_text(
        'These findings have practical implications for optimizing '
        'amygdala stimulation for memory enhancement:'
    )
    pdf.body_text(
        '1. During encoding, stimulation may be most effective when '
        'amygdala-hippocampal theta-HFA coupling is strong (PAC finding).'
    )
    pdf.body_text(
        '2. During retrieval, stimulation may be most effective when '
        'amygdala-hippocampal theta coherence is high (coherence finding).'
    )
    pdf.body_text(
        '3. Avoiding stimulation during hippocampal IED events could '
        'prevent receiver disruption that causes stimulation to fail.'
    )

    # ── Output ──
    os.makedirs(os.path.dirname(OUTPUT_PDF), exist_ok=True)
    pdf.output(OUTPUT_PDF)
    print(f'Report saved: {OUTPUT_PDF}')


if __name__ == '__main__':
    build_report()
