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
    'BLA': 'BLA', 'HPC': 'HPC', 'CA': 'CA', 'DG': 'DG',
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

# Section/table label fragment that matches the per-region/per-pair multi-band
# model's number of band predictors.
BAND_GROUP_LABEL = 'Multi-Band' if PHASE == 'encoding' else 'Two-Band'
BAND_GROUP_FORMULA = (
    'theta_c + slow_gamma_c + fast_gamma_c + StimCond + '
    'theta_c:StimCond + slow_gamma_c:StimCond + fast_gamma_c:StimCond'
    if PHASE == 'encoding'
    else 'theta_c + slow_gamma_c + StimCond + '
         'theta_c:StimCond + slow_gamma_c:StimCond'
)

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

# Coherence pairs (BLA-X focus; non-BLA pairs intentionally excluded)
BLA_COH_PAIRS = ['BLA_CA', 'BLA_DG', 'BLA_EC', 'BLA_HPC', 'BLA_PRC']
MTL_COH_PAIRS = BLA_COH_PAIRS
HPC_COH_PAIRS = BLA_COH_PAIRS


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
        'band_c': 'Band',
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


def coef_table_rows_with_bla_reference(df, ref_label='Region [BLA]'):
    """Build coefficient rows and insert an explicit reference row for BLA
    immediately before the first non-BLA Region term."""
    rows = []
    inserted = False
    for _, r in df.iterrows():
        term_label = format_term(r['term'])
        is_region_main = (
            r['term'].startswith('Region')
            and ':' not in r['term']
        )
        if is_region_main and not inserted:
            rows.append([f'{ref_label} (reference)', '1.0000', '-', '-', '-', ''])
            inserted = True
        rows.append([
            term_label,
            or_str(r['estimate']),
            ci_str(r['conf.low'], r['conf.high']),
            f'{r["statistic"]:.3f}',
            p_str(r['p.value']),
            sig_str(r['p.value']),
        ])
    return rows


def fmt_fit(value, decimals=2):
    if pd.isna(value):
        return '-'
    return f'{float(value):.{decimals}f}'


def anova_table_rows(df):
    """Build display rows from a model-build anova dataframe (one row per
    model). Columns produced: Model, Formula, npar, AIC, BIC, logLik, Chisq,
    Df, p."""
    rows = []
    for _, r in df.iterrows():
        rows.append([
            str(r['Model']),
            str(r['Formula']),
            f'{int(r["npar"])}' if not pd.isna(r['npar']) else '-',
            fmt_fit(r['AIC'], 1),
            fmt_fit(r['BIC'], 1),
            fmt_fit(r['logLik'], 1),
            fmt_fit(r['Chisq'], 2),
            fmt_fit(r['Df'], 0),
            p_str(r['p.value']) if not pd.isna(r['p.value']) else '-',
            sig_str(r['p.value']) if not pd.isna(r['p.value']) else '',
        ])
    return rows


def render_anova(pdf, title, anova_filename):
    df = load_coefs(anova_filename)
    if df is None or df.empty:
        return
    pdf.apa_table(
        title,
        ['Model', 'Formula', 'npar', 'AIC', 'BIC', 'logLik', 'Chisq', 'Df', 'p', ''],
        anova_table_rows(df),
        col_widths=[14, 92, 10, 14, 14, 14, 14, 8, 14, 6],
        note='Sequential glmer model build. Chisq, Df, p compare each model '
             'against the previous row (likelihood-ratio test). '
             '* p < .05, ** p < .01, *** p < .001, + p < .10.'
    )


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
        'Region x StimCond models (one per band): '
        'Accuracy ~ Region * StimCond + band_c + (1|Patient). '
        'Fit separately for theta and slow gamma. Tests Region, StimCond, '
        'and Region x StimCond effects across all 6 power regions / all '
        'eligible coherence pairs / all eligible PAC pairs.'
    )
    bands_in_model = (
        'theta, slow gamma, and HFA' if PHASE == 'encoding' else 'theta and slow gamma'
    )
    pdf.body_text(
        f'Per-region {BAND_GROUP_LABEL.lower()} model: '
        f'Accuracy ~ {BAND_GROUP_FORMULA} + (1|Patient). '
        f'Fit separately for each region (or pair) so {bands_in_model} '
        f'effects are tested side-by-side within the same model. The PAC '
        f'analogue substitutes z-scored slow-gamma PAC and HFA PAC.'
    )

    bands_desc = 'theta (4-8 Hz), slow gamma (30-55 Hz), and HFA (70-100 Hz)' if PHASE == 'encoding' \
        else 'theta (4-8 Hz) and slow gamma (30-55 Hz)'
    pdf.body_text(f'Frequency bands analyzed: {bands_desc}.')

    # ═══════════════════════════════════════════════════════════════════════
    # POWER
    # ═══════════════════════════════════════════════════════════════════════
    sec += 1
    pdf.section_title(f'{sec}. Power x Stimulation Condition')

    # ── Power: Region x StimCond per band (across all 6 regions) ──
    sub_idx = 1
    pdf.subsection_title(
        f'{sec}.{sub_idx}. Power: Region x StimCond - All Regions (per band)'
    )
    sub_idx += 1
    pdf.body_text(
        'Model: Accuracy ~ Region * StimCond + band_c + (1|Patient). '
        'Regions: BLA, CA, DG, EC, HPC, PRC. One model per band.'
    )
    for bk, bl in BAND_LABELS.items():
        df = load_coefs(f'power_allregions_regbystim_{bk}_coefs.csv')
        if df is None:
            continue
        pdf.apa_table(
            f'Table. Power - All Regions x StimCond - {bl}',
            ['Predictor', 'OR', '95% CI', 'z', 'p', ''],
            coef_table_rows(df),
            col_widths=[60, 18, 38, 14, 14, 8],
            note='Region [BLA] is the reference category (absorbed into the '
                 'Intercept). Other Region rows are baseline differences vs '
                 'BLA; Region:StimCond[stim] rows are stim-effect differences '
                 'vs BLA.'
        )
        render_anova(
            pdf,
            f'Table. Sequential Model Build - Power All Regions x StimCond - {bl}',
            f'power_allregions_regbystim_{bk}_anova.csv',
        )

    # ── Power: Per-region multi-band model ──
    pdf.subsection_title(
        f'{sec}.{sub_idx}. Power: Per-Region {BAND_GROUP_LABEL} Models'
    )
    sub_idx += 1
    pdf.body_text(
        f'Model (per region): Accuracy ~ {BAND_GROUP_FORMULA} + (1|Patient). '
        f'One full coefficient table per region, even when no terms reach '
        f'significance.'
    )
    for reg in ALL_REGIONS:
        df = load_coefs(f'power_twoband_{reg}_coefs.csv')
        if df is None:
            continue
        pdf.apa_table(
            f'Table. Power {BAND_GROUP_LABEL} Model - {display_region(reg)}',
            ['Predictor', 'OR', '95% CI', 'z', 'p', ''],
            coef_table_rows(df),
            col_widths=[58, 18, 38, 14, 14, 8],
        )
        render_anova(
            pdf,
            f'Table. Sequential Model Build - Power {BAND_GROUP_LABEL} - {display_region(reg)}',
            f'power_twoband_{reg}_anova.csv',
        )

    # ═══════════════════════════════════════════════════════════════════════
    # COHERENCE
    # ═══════════════════════════════════════════════════════════════════════
    sec += 1
    pdf.section_title(f'{sec}. Coherence x Stimulation Condition')

    # ── Coherence: Pair x StimCond per band (across all eligible pairs) ──
    coh_sub_idx = 1
    pdf.subsection_title(
        f'{sec}.{coh_sub_idx}. Coherence: Pair x StimCond - All Pairs (per band)'
    )
    coh_sub_idx += 1
    pdf.body_text(
        'Model: Accuracy ~ band_c + StimCond + Region + Region:StimCond + '
        '(1|Patient), where Region encodes the pair label. Restricted to '
        'BLA-X coherence pairs (BLA-CA, BLA-DG, BLA-EC, BLA-HPC, BLA-PRC).'
    )
    for bk, bl in BAND_LABELS.items():
        df = load_coefs(f'coherence_allpairs_pairbystim_{bk}_coefs.csv')
        if df is None:
            continue
        pdf.apa_table(
            f'Table. Coherence - All Pairs x StimCond - {bl}',
            ['Predictor', 'OR', '95% CI', 'z', 'p', ''],
            coef_table_rows(df),
            col_widths=[60, 18, 38, 14, 14, 8],
            note='Region [BLA-CA] is the reference pair (absorbed into the '
                 'Intercept). Other Region rows are baseline differences vs '
                 'BLA-CA; Region:StimCond[stim] rows are stim-effect '
                 'differences vs BLA-CA.'
        )
        render_anova(
            pdf,
            f'Table. Sequential Model Build - Coherence All Pairs x StimCond - {bl}',
            f'coherence_allpairs_pairbystim_{bk}_anova.csv',
        )

    # ── Coherence: Per-pair multi-band model ──
    pdf.subsection_title(
        f'{sec}.{coh_sub_idx}. Coherence: Per-Pair {BAND_GROUP_LABEL} Models'
    )
    coh_sub_idx += 1
    pdf.body_text(
        f'Model (per pair): Accuracy ~ {BAND_GROUP_FORMULA} + (1|Patient). '
        f'One full coefficient table per pair.'
    )
    for pair in BLA_COH_PAIRS:
        df = load_coefs(f'coherence_twoband_{pair}_coefs.csv')
        if df is None:
            continue
        pdf.apa_table(
            f'Table. Coherence {BAND_GROUP_LABEL} Model - {display_region(pair)}',
            ['Predictor', 'OR', '95% CI', 'z', 'p', ''],
            coef_table_rows(df),
            col_widths=[58, 18, 38, 14, 14, 8],
        )
        render_anova(
            pdf,
            f'Table. Sequential Model Build - Coherence {BAND_GROUP_LABEL} - {display_region(pair)}',
            f'coherence_twoband_{pair}_anova.csv',
        )

    # ═══════════════════════════════════════════════════════════════════════
    # PAC
    # ═══════════════════════════════════════════════════════════════════════
    sec += 1
    pdf.section_title(f'{sec}. PAC x Stimulation Condition')
    pdf.body_text(
        'PAC measures theta-phase x amplitude coupling. PAC values are '
        'z-scored within the modeled pair set.'
    )

    # ── PAC: Pair x StimCond per PAC type (across all eligible pairs) ──
    pac_sub_idx = 1
    pdf.subsection_title(
        f'{sec}.{pac_sub_idx}. PAC: Pair x StimCond - All Pairs (per PAC type)'
    )
    pac_sub_idx += 1
    pdf.body_text(
        'Model: Accuracy ~ pac_z + StimCond + Region + Region:StimCond + '
        '(1|Patient), where Region encodes the pair label. Restricted to '
        'BLA-X PAC pairs (BLA-CA, BLA-DG, BLA-EC, BLA-HPC, BLA-PRC).'
    )
    for pt, pl in PAC_TYPES.items():
        df = load_coefs(f'pac_allpairs_pairbystim_{pt}_coefs.csv')
        if df is None:
            continue
        pdf.apa_table(
            f'Table. PAC ({pl}) - All Pairs x StimCond',
            ['Predictor', 'OR', '95% CI', 'z', 'p', ''],
            coef_table_rows(df),
            col_widths=[60, 18, 38, 14, 14, 8],
            note='Region [BLA-CA] is the reference pair (absorbed into the '
                 'Intercept). Other Region rows are baseline differences vs '
                 'BLA-CA; Region:StimCond[stim] rows are stim-effect '
                 'differences vs BLA-CA.'
        )
        render_anova(
            pdf,
            f'Table. Sequential Model Build - PAC ({pl}) - All Pairs x StimCond',
            f'pac_allpairs_pairbystim_{pt}_anova.csv',
        )

    # ── PAC: Per-pair two-PAC-type model ──
    pdf.subsection_title(
        f'{sec}.{pac_sub_idx}. PAC: Per-Pair Two-PAC-Type Models'
    )
    pac_sub_idx += 1
    pdf.body_text(
        'Model (per pair): Accuracy ~ sg_pac_z + hfa_pac_z + StimCond + '
        'sg_pac_z:StimCond + hfa_pac_z:StimCond + (1|Patient). '
        'PAC values z-scored within each pair. One full coefficient table '
        'per pair.'
    )
    pac_pairs = ['BLA_HPC', 'BLA_EC', 'BLA_PRC', 'BLA_CA', 'BLA_DG']
    for pair in pac_pairs:
        df = load_coefs(f'pac_twotype_{pair}_coefs.csv')
        if df is None:
            continue
        pdf.apa_table(
            f'Table. PAC Two-PAC-Type Model - {display_region(pair)}',
            ['Predictor', 'OR', '95% CI', 'z', 'p', ''],
            coef_table_rows(df),
            col_widths=[58, 18, 38, 14, 14, 8],
        )
        render_anova(
            pdf,
            f'Table. Sequential Model Build - PAC Two-PAC-Type - {display_region(pair)}',
            f'pac_twotype_{pair}_anova.csv',
        )

    # ── Output ──
    os.makedirs(os.path.dirname(OUTPUT_PDF), exist_ok=True)
    pdf.output(OUTPUT_PDF)
    print(f'Report saved: {OUTPUT_PDF}')


if __name__ == '__main__':
    build_report()
