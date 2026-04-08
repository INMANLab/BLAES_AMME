#!/usr/bin/env python
"""
Build PDF report: Does Baseline Neural Activity x Stimulation Impact Memory?
v2: Per-region + across-region models, restricted PAC pairs.

Usage:
  python build_v2_report.py encoding balanced
  python build_v2_report.py encoding imbalanced
  python build_v2_report.py retrieval balanced
  python build_v2_report.py retrieval imbalanced
"""

import os
import sys
import numpy as np
import pandas as pd
from fpdf import FPDF

# ── Parse args ──────────────────────────────────────────────────────────────
if len(sys.argv) < 3:
    print("Usage: python build_v2_report.py <encoding|retrieval> <balanced|imbalanced>")
    sys.exit(1)

PHASE = sys.argv[1]       # encoding or retrieval
FILTER = sys.argv[2]      # balanced or imbalanced

assert PHASE in ('encoding', 'retrieval'), f"Invalid phase: {PHASE}"
assert FILTER in ('balanced', 'imbalanced'), f"Invalid filter: {FILTER}"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, 'outputs', f'{FILTER}_{PHASE}_mlm')

FILTER_LABEL = 'Balanced Trials' if FILTER == 'balanced' else 'ImBalanced Trials'
PHASE_LABEL = PHASE.capitalize()

REPORT_TITLE = f'Baseline Neural Activity x Stim - {PHASE_LABEL} - {FILTER_LABEL}'
OUTPUT_PDF = os.path.join(SCRIPT_DIR, 'outputs',
    f'Does_Baseline_Neural_Activity_x_Stim_Impact_Memory_{PHASE_LABEL}_{FILTER_LABEL.replace(" ", "_")}.pdf')


# ── Region labels ──────────────────────────────────────────────────────────

REGION_LABELS = {
    'BLA': 'Amygdala', 'HPC': 'HPC', 'CA': 'CA', 'DG': 'DG',
    'EC': 'EC', 'PRC': 'PRC', 'PHG': 'PHG',
}

def display_region(r):
    parts = r.split('_')
    return '-'.join(REGION_LABELS.get(p, p) for p in parts)

# Encoding: theta, slow gamma, HFA; Retrieval: theta, slow gamma
if PHASE == 'encoding':
    BAND_LABELS = {
        'theta': 'Theta (4-8 Hz)',
        'slow_gamma': 'Slow Gamma (30-55 Hz)',
        'fast_gamma': 'HFA (70-100 Hz)',
    }
else:
    BAND_LABELS = {
        'theta': 'Theta (4-8 Hz)',
        'slow_gamma': 'Slow Gamma (30-55 Hz)',
    }

PAC_TYPES = {
    'slow_gamma': 'SG PAC (30-50 Hz)',
    'hfa': 'HFA PAC (70-100 Hz)',
}

SET_LABELS = {
    'MTL': 'MTL (Amygdala, HPC, EC, PRC)',
    'HPC_subfields': 'HPC Subfields (Amygdala, CA, DG)',
}

COH_SET_LABELS = {
    'MTL': 'MTL Pairs (Amygdala-HPC, Amygdala-EC, Amygdala-PRC, EC-HPC, EC-PRC, HPC-PRC)',
    'HPC_subfields': 'HPC Subfield Pairs (Amygdala-CA, Amygdala-DG, CA-DG)',
}

PAC_SET_LABELS = {
    'MTL': 'MTL (Amygdala-HPC, Amygdala-EC, Amygdala-PRC)',
    'HPC_subfields': 'HPC Subfields (Amygdala-HPC, Amygdala-CA, Amygdala-DG)',
}

# Power regions
MTL_REGIONS = ['BLA', 'HPC', 'EC', 'PRC']
HPC_REGIONS = ['BLA', 'CA', 'DG']
ALL_REGIONS = ['BLA', 'CA', 'DG', 'EC', 'HPC', 'PRC']

# Coherence pairs
MTL_COH_PAIRS = ['BLA_EC', 'BLA_HPC', 'BLA_PRC', 'EC_HPC', 'EC_PRC', 'HPC_PRC']
HPC_COH_PAIRS = ['BLA_CA', 'BLA_DG', 'CA_DG']


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

def get_interaction_row(df, pattern):
    mask = df['term'].str.contains(pattern, na=False)
    if mask.any():
        return df[mask].iloc[0]
    return None

def format_term(term):
    labels = {
        '(Intercept)': 'Intercept',
        'StimCondstim': 'StimCond [stim]',
        'band_c': 'Band (centered)',
        'band_c:StimCondstim': 'Band x StimCond',
        'StimCondstim:band_c': 'Band x StimCond',
        'pac_z': 'PAC (z)',
        'StimCondstim:pac_z': 'PAC (z) x StimCond',
        'pac_z:StimCondstim': 'PAC (z) x StimCond',
    }
    if term in labels:
        return labels[term]
    if term.startswith('Region') and ':' not in term:
        return f'Region [{display_region(term.replace("Region", ""))}]'
    if term.startswith('Region') and ':band_c' in term:
        return f'Band x Region [{display_region(term.replace("Region","").replace(":band_c",""))}]'
    if 'band_c:Region' in term:
        return f'Band x Region [{display_region(term.replace("band_c:Region",""))}]'
    return term


def coef_table_rows(df):
    """Build formatted rows from a coefficient dataframe."""
    rows = []
    for _, r in df.iterrows():
        rows.append([
            format_term(r['term']),
            or_str(r['estimate']),
            ci_str(r['conf.low'], r['conf.high']),
            f'{r["statistic"]:.3f}',
            p_str(r['p.value']),
            sig_str(r['p.value']),
        ])
    return rows


# ── Build report ──────────────────────────────────────────────────────────

def build_report():
    pdf = APAReport()
    pdf.add_page()

    # ── Title ──
    pdf.set_font('Helvetica', 'B', 16)
    pdf.multi_cell(0, 9, 'Does Baseline Neural Activity x Stimulation\nImpact Subsequent Memory?')
    pdf.ln(2)
    pdf.set_font('Helvetica', '', 11)
    pdf.cell(0, 6, f'{PHASE_LABEL} Phase - {FILTER_LABEL}',
             new_x="LMARGIN", new_y="NEXT")
    pdf.ln(6)

    # ── Section 1: Overview ──
    sec = 1
    pdf.section_title(f'{sec}. Overview')
    pdf.body_text(
        f'This report tests whether baseline neural activity during the '
        f'{PHASE} phase differentially predicts subsequent memory as a '
        f'function of stimulation condition (BLA theta-modulated gamma '
        f'stimulation vs. no stimulation).'
    )
    if FILTER == 'balanced':
        pdf.body_text(
            'Balanced-trials filter applied: subjects with fewer than 10 '
            'trials in either the remembered or forgotten condition were '
            'excluded to prevent extreme estimates.'
        )
    else:
        pdf.body_text(
            'No trial-balance filter applied: ALL subjects are included '
            'regardless of the distribution of remembered vs. forgotten '
            'trials. This maximizes power but may include subjects whose '
            'extreme memory rates could bias estimates.'
        )

    pdf.subsection_title('Statistical Approach')
    pdf.body_text(
        'All models are trial-level GLMMs (binomial, logit link, bobyqa '
        'optimizer, maxfun = 200,000, random intercept for patient). '
        'Results reported as odds ratios (OR) with 95% Wald CIs.'
    )
    pdf.body_text(
        'Per-region models: Accuracy ~ band_c * StimCond + (1|Patient). '
        'Tests the Band x StimCond interaction within each region separately.'
    )
    pdf.body_text(
        'Across-region models: Accuracy ~ StimCond + Region + band_c + '
        'band_c:StimCond + band_c:Region + (1|Patient). '
        'Pools across regions within a frequency band.'
    )
    pdf.body_text(
        'PAC models: Accuracy ~ pac_z * StimCond + Region + (1|Patient). '
        'PAC z-scored within the pooled region set. Separate models for '
        'slow gamma PAC (30-50 Hz) and HFA PAC (70-100 Hz).'
    )

    bands_desc = 'theta (4-8 Hz), slow gamma (30-55 Hz), and HFA (70-100 Hz)' if PHASE == 'encoding' \
        else 'theta (4-8 Hz) and slow gamma (30-55 Hz)'
    pdf.body_text(f'Frequency bands analyzed: {bands_desc}.')

    # ═══════════════════════════════════════════════════════════════════════
    # POWER
    # ═══════════════════════════════════════════════════════════════════════
    sec += 1
    pdf.section_title(f'{sec}. Power x Stimulation Condition')

    # ── Per-region power ──
    pdf.subsection_title(f'{sec}.1. Per-Region Power Models')
    pw_summary = []
    for reg in ALL_REGIONS:
        for bk, bl in BAND_LABELS.items():
            df = load_coefs(f'power_region_{reg}_{bk}_coefs.csv')
            if df is None:
                continue
            row = get_interaction_row(df, 'band_c:StimCond')
            if row is None:
                row = get_interaction_row(df, 'StimCondstim:band_c')
            if row is not None:
                pw_summary.append([
                    display_region(reg), bl.split(' (')[0],
                    or_str(row['estimate']),
                    ci_str(row['conf.low'], row['conf.high']),
                    f'{row["statistic"]:.3f}',
                    p_str(row['p.value']),
                    sig_str(row['p.value']),
                ])

    if pw_summary:
        pdf.apa_table(
            'Table. Per-Region Power x StimCond Interactions',
            ['Region', 'Band', 'OR', '95% CI', 'z', 'p', ''],
            pw_summary,
            col_widths=[22, 22, 16, 38, 14, 16, 7],
            note='OR for Band x StimCond [stim]. OR > 1: higher power more '
                 'beneficial on stim trials. * p < .05, + p < .10.'
        )

    # Full tables for significant per-region models
    for reg in ALL_REGIONS:
        for bk, bl in BAND_LABELS.items():
            df = load_coefs(f'power_region_{reg}_{bk}_coefs.csv')
            if df is None:
                continue
            row = get_interaction_row(df, 'band_c:StimCond')
            if row is None:
                row = get_interaction_row(df, 'StimCondstim:band_c')
            if row is not None and row['p.value'] < 0.05:
                pdf.apa_table(
                    f'Table. Full Model: Power {bl} - {display_region(reg)}',
                    ['Predictor', 'OR', '95% CI', 'z', 'p', ''],
                    coef_table_rows(df),
                    col_widths=[42, 18, 40, 15, 18, 8],
                )

    # ── Across-region power ──
    pdf.subsection_title(f'{sec}.2. Across-Region Power Models')
    for region_set in ['MTL', 'HPC_subfields']:
        pw_across = []
        for bk, bl in BAND_LABELS.items():
            df = load_coefs(f'power_across_{region_set}_{bk}_coefs.csv')
            if df is None:
                continue
            row = get_interaction_row(df, 'StimCondstim:band_c')
            if row is None:
                row = get_interaction_row(df, 'band_c:StimCond')
            if row is not None:
                pw_across.append([
                    bl.split(' (')[0],
                    or_str(row['estimate']),
                    ci_str(row['conf.low'], row['conf.high']),
                    f'{row["statistic"]:.3f}',
                    p_str(row['p.value']),
                    sig_str(row['p.value']),
                ])

        if pw_across:
            pdf.apa_table(
                f'Table. Across-Region Power x StimCond - {SET_LABELS[region_set]}',
                ['Band', 'OR', '95% CI', 'z', 'p', ''],
                pw_across,
                col_widths=[35, 18, 42, 15, 18, 8],
                note='Pooled across regions within each frequency band.'
            )

        # Full tables for significant across-region models
        for bk, bl in BAND_LABELS.items():
            df = load_coefs(f'power_across_{region_set}_{bk}_coefs.csv')
            if df is None:
                continue
            row = get_interaction_row(df, 'StimCondstim:band_c')
            if row is None:
                row = get_interaction_row(df, 'band_c:StimCond')
            if row is not None and row['p.value'] < 0.05:
                pdf.apa_table(
                    f'Table. Full Model: Power {bl} Across {SET_LABELS[region_set]}',
                    ['Predictor', 'OR', '95% CI', 'z', 'p', ''],
                    coef_table_rows(df),
                    col_widths=[48, 16, 38, 14, 16, 8],
                )

    # ═══════════════════════════════════════════════════════════════════════
    # COHERENCE
    # ═══════════════════════════════════════════════════════════════════════
    sec += 1
    pdf.section_title(f'{sec}. Coherence x Stimulation Condition')

    # ── Per-pair coherence ──
    pdf.subsection_title(f'{sec}.1. Per-Region-Pair Coherence Models')

    for set_name, pairs in [('MTL', MTL_COH_PAIRS), ('HPC Subfields', HPC_COH_PAIRS)]:
        coh_summary = []
        for pair in pairs:
            for bk, bl in BAND_LABELS.items():
                df = load_coefs(f'coherence_region_{pair}_{bk}_coefs.csv')
                if df is None:
                    continue
                row = get_interaction_row(df, 'band_c:StimCond')
                if row is None:
                    row = get_interaction_row(df, 'StimCondstim:band_c')
                if row is not None:
                    coh_summary.append([
                        display_region(pair), bl.split(' (')[0],
                        or_str(row['estimate']),
                        ci_str(row['conf.low'], row['conf.high']),
                        f'{row["statistic"]:.3f}',
                        p_str(row['p.value']),
                        sig_str(row['p.value']),
                    ])

        if coh_summary:
            pdf.apa_table(
                f'Table. Per-Pair Coherence x StimCond - {set_name}',
                ['Pair', 'Band', 'OR', '95% CI', 'z', 'p', ''],
                coh_summary,
                col_widths=[28, 22, 16, 34, 14, 14, 7],
                note='* p < .05, + p < .10.'
            )

    # Full tables for significant per-pair models
    for pair in MTL_COH_PAIRS + HPC_COH_PAIRS:
        for bk, bl in BAND_LABELS.items():
            df = load_coefs(f'coherence_region_{pair}_{bk}_coefs.csv')
            if df is None:
                continue
            row = get_interaction_row(df, 'band_c:StimCond')
            if row is None:
                row = get_interaction_row(df, 'StimCondstim:band_c')
            if row is not None and row['p.value'] < 0.05:
                pdf.apa_table(
                    f'Table. Full Model: Coherence {bl} - {display_region(pair)}',
                    ['Predictor', 'OR', '95% CI', 'z', 'p', ''],
                    coef_table_rows(df),
                    col_widths=[42, 18, 40, 15, 18, 8],
                )

    # ── Across-pair coherence ──
    pdf.subsection_title(f'{sec}.2. Across-Region Coherence Models')
    for region_set in ['MTL', 'HPC_subfields']:
        coh_across = []
        for bk, bl in BAND_LABELS.items():
            df = load_coefs(f'coherence_across_{region_set}_{bk}_coefs.csv')
            if df is None:
                continue
            row = get_interaction_row(df, 'StimCondstim:band_c')
            if row is None:
                row = get_interaction_row(df, 'band_c:StimCond')
            if row is not None:
                coh_across.append([
                    bl.split(' (')[0],
                    or_str(row['estimate']),
                    ci_str(row['conf.low'], row['conf.high']),
                    f'{row["statistic"]:.3f}',
                    p_str(row['p.value']),
                    sig_str(row['p.value']),
                ])

        if coh_across:
            pdf.apa_table(
                f'Table. Across-Pair Coherence x StimCond - {COH_SET_LABELS[region_set]}',
                ['Band', 'OR', '95% CI', 'z', 'p', ''],
                coh_across,
                col_widths=[35, 18, 42, 15, 18, 8],
            )

        for bk, bl in BAND_LABELS.items():
            df = load_coefs(f'coherence_across_{region_set}_{bk}_coefs.csv')
            if df is None:
                continue
            row = get_interaction_row(df, 'StimCondstim:band_c')
            if row is None:
                row = get_interaction_row(df, 'band_c:StimCond')
            if row is not None and row['p.value'] < 0.05:
                pdf.apa_table(
                    f'Table. Full Model: Coherence {bl} Across {COH_SET_LABELS[region_set]}',
                    ['Predictor', 'OR', '95% CI', 'z', 'p', ''],
                    coef_table_rows(df),
                    col_widths=[52, 16, 38, 14, 14, 8],
                )

    # ═══════════════════════════════════════════════════════════════════════
    # PAC
    # ═══════════════════════════════════════════════════════════════════════
    sec += 1
    pdf.section_title(f'{sec}. PAC x Stimulation Condition')
    pdf.body_text(
        'PAC measures theta-phase x amplitude coupling. Models pool 3 '
        'amygdala-target pairs per set with Region as a covariate. '
        'PAC predictors are z-scored within the pooled set.'
    )

    for set_name in ['MTL', 'HPC_subfields']:
        pdf.subsection_title(f'{sec}.{1 if set_name == "MTL" else 2}. '
                             f'PAC - {PAC_SET_LABELS[set_name]}')

        pac_summary = []
        for pt, pl in PAC_TYPES.items():
            df = load_coefs(f'pac_{set_name}_{pt}_coefs.csv')
            if df is None:
                continue
            row = get_interaction_row(df, 'pac_z:StimCond')
            if row is None:
                row = get_interaction_row(df, 'StimCondstim:pac_z')
            if row is not None:
                pac_summary.append([
                    pl,
                    or_str(row['estimate']),
                    ci_str(row['conf.low'], row['conf.high']),
                    f'{row["statistic"]:.3f}',
                    p_str(row['p.value']),
                    sig_str(row['p.value']),
                ])

        if pac_summary:
            pdf.apa_table(
                f'Table. PAC x StimCond - {PAC_SET_LABELS[set_name]}',
                ['PAC Type', 'OR', '95% CI', 'z', 'p', ''],
                pac_summary,
                col_widths=[38, 18, 40, 15, 18, 8],
                note='OR per 1 SD increase in z-scored PAC. '
                     '* p < .05, + p < .10.'
            )

        # Full tables for significant PAC models
        for pt, pl in PAC_TYPES.items():
            df = load_coefs(f'pac_{set_name}_{pt}_coefs.csv')
            if df is None:
                continue
            row = get_interaction_row(df, 'pac_z:StimCond')
            if row is None:
                row = get_interaction_row(df, 'StimCondstim:pac_z')
            if row is not None and row['p.value'] < 0.05:
                pdf.apa_table(
                    f'Table. Full Model: {pl} - {PAC_SET_LABELS[set_name]}',
                    ['Predictor', 'OR', '95% CI', 'z', 'p', ''],
                    coef_table_rows(df),
                    col_widths=[48, 16, 40, 14, 16, 8],
                )

    # ═══════════════════════════════════════════════════════════════════════
    # OVERALL SUMMARY TABLE
    # ═══════════════════════════════════════════════════════════════════════
    sec += 1
    pdf.section_title(f'{sec}. Summary of All Band/PAC x StimCond Interactions')

    all_rows = []

    # Per-region power
    for reg in ALL_REGIONS:
        for bk, bl in BAND_LABELS.items():
            df = load_coefs(f'power_region_{reg}_{bk}_coefs.csv')
            if df is None: continue
            row = get_interaction_row(df, 'band_c:StimCond')
            if row is None: row = get_interaction_row(df, 'StimCondstim:band_c')
            if row is not None:
                all_rows.append([
                    'Power', display_region(reg), bl.split(' (')[0],
                    or_str(row['estimate']), p_str(row['p.value']),
                    sig_str(row['p.value']),
                ])

    # Per-pair coherence
    for pair in MTL_COH_PAIRS + HPC_COH_PAIRS:
        for bk, bl in BAND_LABELS.items():
            df = load_coefs(f'coherence_region_{pair}_{bk}_coefs.csv')
            if df is None: continue
            row = get_interaction_row(df, 'band_c:StimCond')
            if row is None: row = get_interaction_row(df, 'StimCondstim:band_c')
            if row is not None:
                all_rows.append([
                    'Coherence', display_region(pair), bl.split(' (')[0],
                    or_str(row['estimate']), p_str(row['p.value']),
                    sig_str(row['p.value']),
                ])

    # PAC
    for set_name in ['MTL', 'HPC_subfields']:
        set_short = 'MTL' if set_name == 'MTL' else 'HPC'
        for pt, pl in PAC_TYPES.items():
            df = load_coefs(f'pac_{set_name}_{pt}_coefs.csv')
            if df is None: continue
            row = get_interaction_row(df, 'pac_z:StimCond')
            if row is None: row = get_interaction_row(df, 'StimCondstim:pac_z')
            if row is not None:
                all_rows.append([
                    'PAC', f'{set_short} set', pl.split(' (')[0],
                    or_str(row['estimate']), p_str(row['p.value']),
                    sig_str(row['p.value']),
                ])

    pdf.apa_table(
        'Table. All Band/PAC x StimCond Interactions (Per-Region/Pair Models)',
        ['Measure', 'Region/Pair', 'Band', 'OR', 'p', ''],
        all_rows,
        col_widths=[22, 30, 26, 20, 20, 8],
        note='Per-region/pair models only. * p < .05, ** p < .01, + p < .10.'
    )

    # ── Output ──
    os.makedirs(os.path.dirname(OUTPUT_PDF), exist_ok=True)
    pdf.output(OUTPUT_PDF)
    print(f'Report saved: {OUTPUT_PDF}')


if __name__ == '__main__':
    build_report()
