#!/usr/bin/env python
"""
Build PDF report: Does Baseline Neural Activity x Stimulation Impact Memory?
Encoding-phase ALL-SUBJECTS (imbalanced trials) MLM results.
"""

import os
import numpy as np
import pandas as pd
from fpdf import FPDF

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, 'outputs', 'imbalanced_encoding_mlm')
REPORT_TITLE = 'Does Baseline Neural Activity x Stim Impact Memory - Encoding - All Subjects'
OUTPUT_PDF = os.path.join(SCRIPT_DIR, 'outputs',
    'Does_Baseline_Neural_Activity_x_Stim_Impact_Memory_Encoding_ImBalanced_Trials.pdf')


# ── Region labels ──────────────────────────────────────────────────────────

REGION_LABELS = {
    'BLA': 'Amygdala', 'HPC': 'HPC', 'CA': 'CA', 'DG': 'DG',
    'EC': 'EC', 'PRC': 'PRC', 'PHG': 'PHG',
}

def display_region(r):
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
    'band_c': 'Band power',
    'StimCondstim:band_c': 'Band x StimCond [stim]',
    'slow_gamma_pac_z': 'SG PAC (z)',
    'hfa_pac_z': 'HFA PAC (z)',
    'slow_gamma_pac_z:StimCondstim': 'SG PAC (z) x StimCond [stim]',
    'hfa_pac_z:StimCondstim': 'HFA PAC (z) x StimCond [stim]',
}

# Intra-hippocampal pairs to exclude from PAC
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
    elif p < .10: return '+'
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
    pdf.cell(0, 6, 'Encoding Phase - All Subjects (Imbalanced Trials)', new_x="LMARGIN", new_y="NEXT")
    pdf.ln(6)

    # ── Section 1: Overview ──
    pdf.section_title('1. Overview')
    pdf.body_text(
        'This report tests whether baseline (pre-stimulation) neural activity '
        'differentially predicts subsequent memory as a function of stimulation '
        'condition. Unlike the balanced-trials analysis, this version includes '
        'ALL subjects regardless of the distribution of remembered vs. forgotten '
        'trials. This provides maximum statistical power but may include subjects '
        'whose extreme memory rates could bias estimates.'
    )

    pdf.subsection_title('Statistical Approach')
    pdf.body_text(
        'All models are trial-level generalized linear mixed models (GLMMs) '
        'with binomial family (logit link), random intercept for patient, and '
        'bobyqa optimizer (maxfun = 200,000). Results are reported as odds '
        'ratios (OR) with 95% Wald confidence intervals. The critical test in '
        'each model is the Band x StimCond interaction term.'
    )

    # ── Section 2: Power Results ──
    pdf.section_title('2. Power x Stimulation Condition')
    pdf.body_text(
        'Model: Accuracy ~ StimCond + Region + band_c + band_c:StimCond '
        '+ band_c:Region + (1 | Patient)'
    )
    pdf.body_text(
        'Band power was computed as the mean of baseline-corrected spectral '
        'power across frequencies in each band (theta: 4-8 Hz, slow gamma: '
        '30-55 Hz, HFA: 70-100 Hz). Models use the raw band value; on '
        'convergence failure the band is grand-mean centered as a '
        'numerical-stability fallback.'
    )

    for region_set in ['MTL', 'HPC_subfields']:
        pdf.subsection_title(f'2.{1 if region_set == "MTL" else 2}. {SET_LABELS[region_set]}')

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

    # ── Section 3: Coherence Results ──
    pdf.section_title('3. Coherence x Stimulation Condition')
    pdf.body_text(
        'Model: Accuracy ~ StimCond + Region + band_c + band_c:StimCond '
        '+ band_c:Region + (1 | Patient)'
    )

    for region_set in ['MTL', 'HPC_subfields']:
        pdf.subsection_title(f'3.{1 if region_set == "MTL" else 2}. Coherence - {SET_LABELS[region_set]}')

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

    # ── Section 4: PAC Results (Z-scored) ──
    pdf.section_title('4. Phase-Amplitude Coupling (PAC) x Stimulation Condition')
    pdf.body_text(
        'Model (per region pair): Accuracy ~ (slow_gamma_pac_z + hfa_pac_z) '
        '* StimCond + (1 | Patient)'
    )
    pdf.body_text(
        'PAC measures theta-phase x amplitude coupling at two frequency '
        'ranges: slow gamma (30-50 Hz) and HFA (70-100 Hz). PAC predictors '
        'were z-scored within each region pair.'
    )
    pdf.body_text(
        'Note: Intra-hippocampal pairs (CA-DG, CA-HPC, DG-HPC) were excluded '
        'because PAC within the same structure is not a meaningful '
        'between-region coupling measure.'
    )

    # Collect z-scored PAC results
    pac_z_files = sorted([f for f in os.listdir(DATA_DIR)
                          if f.startswith('pac_zscored_') and f.endswith('_coefs.csv')
                          and f.replace('pac_zscored_', '').replace('_coefs.csv', '')
                              not in EXCLUDE_PAC_PAIRS])

    pac_all_rows = []
    pac_sig_rows = []
    for fname in pac_z_files:
        region = fname.replace('pac_zscored_', '').replace('_coefs.csv', '')
        df = load_coefs(fname)
        if df is None:
            continue

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
                pac_sig_rows.append(entry)

    # Significant / marginal PAC interactions
    pdf.subsection_title('4.1. Significant and Marginal PAC x StimCond Interactions')
    if pac_sig_rows:
        pdf.apa_table(
            'Table. PAC x StimCond Interactions (p < .10)',
            ['Region Pair', 'PAC', 'OR', '95% CI', 'z', 'p', ''],
            pac_sig_rows,
            col_widths=[26, 17, 14, 34, 12, 14, 7],
            note='OR per 1 SD increase in PAC. * p < .05, + p < .10.'
        )
    else:
        pdf.body_text('No PAC x StimCond interactions reached p < .10.')

    # Full table
    pdf.subsection_title('4.2. All PAC x StimCond Interactions')
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
        rows = []
        for _, r in df.iterrows():
            term = r['term']
            display = label_term(term)
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
            f'Table. Full Model: PAC {display_region(region)}',
            ['Predictor', 'OR', '95% CI', 'z', 'p', ''],
            rows,
            col_widths=[48, 16, 40, 14, 16, 8],
        )

    # ── Section 5: Overall Summary ──
    pdf.section_title('5. Overall Summary')

    summary_headers = ['Measure', 'Region Set', 'Band', 'OR', 'p', '']
    summary_data = []
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

    # Add notable PAC interactions
    for fname in pac_z_files:
        region = fname.replace('pac_zscored_', '').replace('_coefs.csv', '')
        df = load_coefs(fname)
        if df is None:
            continue
        for pac_type_key, pac_label in [('slow_gamma_pac_z:StimCond', 'SG PAC'),
                                         ('hfa_pac_z:StimCond', 'HFA PAC')]:
            row = get_interaction_row(df, pac_type_key)
            if row is not None and row['p.value'] < 0.10:
                summary_data.append([
                    'PAC', display_region(region),
                    pac_label,
                    or_str(row['estimate']),
                    p_str(row['p.value']),
                    sig_str(row['p.value']),
                ])

    pdf.apa_table(
        'Table. Summary of All Band/PAC x StimCond Interactions',
        summary_headers,
        summary_data,
        col_widths=[24, 30, 28, 20, 20, 8],
        note='* p < .05, ** p < .01, + p < .10 (marginal). '
             'PAC results use z-scored predictors. Only sig/marginal PAC shown.'
    )

    # ── Output ──
    os.makedirs(os.path.dirname(OUTPUT_PDF), exist_ok=True)
    pdf.output(OUTPUT_PDF)
    print(f'Report saved: {OUTPUT_PDF}')


if __name__ == '__main__':
    build_report()
