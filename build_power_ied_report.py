#!/usr/bin/env python
"""
Build PDF report: Power x IED Timing Windows MLM (Encoding Only).
Per-region power (BLA, HPC, CA, DG) on IED trials.
"""

import os
import glob
import numpy as np
import pandas as pd
from fpdf import FPDF

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'outputs', 'ied_timing_memory')

REPORT_TITLE = 'Power x IED Effects on Memory'


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


TERM_LABELS = {
    '(Intercept)': 'Intercept',
    'pow_theta_z': 'Theta Power (z)',
    'pow_slow_gamma_z': 'Slow Gamma Power (z)',
    'pow_hfa_z': 'HFA Power (z)',
    'stim': 'Stimulation',
    'ied_before_image': 'IED Before Image',
    'ied_during_image': 'IED During Image',
    'ied_after_image': 'IED After Image',
    'ied_during_stim': 'IED During Stim',
    'n_regions_c': 'Region Spread (centered)',
    'n_channels_c': 'Channel Spread (centered)',
}
# Add interaction terms dynamically
for band in ['theta', 'slow_gamma', 'hfa']:
    bn = {'theta': 'Theta', 'slow_gamma': 'SG', 'hfa': 'HFA'}[band]
    for win in ['ied_before_image', 'ied_during_image', 'ied_after_image', 'ied_during_stim']:
        wn = {'ied_before_image': 'Before', 'ied_during_image': 'During Img',
              'ied_after_image': 'After Img', 'ied_during_stim': 'During Stim'}[win]
        TERM_LABELS[f'pow_{band}_z:{win}'] = f'{bn} Pow x {wn}'

OR_HEADERS = ['Predictor', 'OR', '95% CI', 'p', '']
OR_WIDTHS = [55, 18, 40, 25, 10]


def load_or(filename):
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
                     ci_str(r['conf.low'], r['conf.high']),
                     p_str(r['p.value']), sig_str(r['p.value'])])
    return rows


def build_report():
    pdf = APAReport()
    pdf.add_page()

    pdf.set_font('Helvetica', 'B', 16)
    pdf.cell(0, 10, REPORT_TITLE, new_x="LMARGIN", new_y="NEXT", align='C')
    pdf.set_font('Helvetica', '', 11)
    pdf.cell(0, 6, 'Encoding Phase - IED Trials Only',
             new_x="LMARGIN", new_y="NEXT", align='C')
    pdf.ln(6)

    pdf.section_title('Overview')
    pdf.body_text(
        'This report tests whether trial-level spectral power in individual '
        'brain regions (BLA, HPC, CA, DG) predicts subsequent memory on IED '
        'trials, and whether power interacts with IED timing windows. '
        'Power is baseline-corrected (post minus pre) and band-averaged into '
        'theta (4-8 Hz), slow gamma (30-55 Hz), and HFA (70-100 Hz).'
    )
    pdf.body_text(
        'Part 1: Power -> memory on all encoding trials (stim covariate). '
        'Part 2: IED trials only - power + timing windows (main effects and '
        'interactions). Part 3: Power + IED spread. '
        'All models: GLMM, binomial, bobyqa, random intercept for patient.'
    )

    # Load data for descriptives
    pw_all = pd.read_csv(os.path.join(OUTPUT_DIR, 'power_all.csv'))
    ied_merged = pd.read_csv(os.path.join(OUTPUT_DIR, 'power_ied_merged.csv'))

    tbl_num = 1
    bands = [('theta', 'Theta (4-8 Hz)'),
             ('slow_gamma', 'Slow Gamma (30-55 Hz)'),
             ('hfa', 'HFA (70-100 Hz)')]
    regions = ['BLA', 'HPC', 'CA', 'DG']

    # ═══ PART 1: ALL TRIALS ═══
    pdf.add_page()
    pdf.section_title('Part 1: Power and Memory (All Trials)')

    for reg in regions:
        sub = pw_all[pw_all['region'] == reg]
        if len(sub) == 0: continue

        pdf.subsection_title(f'{reg} ({len(sub):,} trials, {sub["patient_id"].nunique()} patients)')

        for b, bl in bands:
            df = load_or(f'pow_A_{reg}_{b}_odds_ratios.csv')
            if df is not None:
                rows = make_or_rows(df)
                pdf.apa_table(
                    f'Table {tbl_num}. {reg} {bl} (All Trials)',
                    OR_HEADERS, rows, OR_WIDTHS,
                    note=f'memory ~ pow_{b}_z + stim + (1|patient_id). N = {len(sub):,}.'
                )
                tbl_num += 1

    # ═══ PART 2: IED TRIALS - TIMING WINDOWS ═══
    for reg in regions:
        ied_sub = ied_merged[ied_merged['region'] == reg]
        pc = ied_sub.groupby('patient_id').size()
        kp = pc[pc >= 5].index
        ied_filt = ied_sub[ied_sub['patient_id'].isin(kp)]
        if len(ied_filt) < 20: continue

        pdf.add_page()
        pdf.section_title(f'Part 2: {reg} Power x IED Timing (IED Trials)')
        pdf.body_text(
            f'Sample: {len(ied_filt)} IED trials from '
            f'{ied_filt["patient_id"].nunique()} patients (>=5 trials each). '
            f'Memory rate: {ied_filt["memory"].mean()*100:.1f}%.'
        )

        # Timing distribution
        tw_rows = []
        for col, label in [('ied_before_image', 'Before Image'),
                            ('ied_during_image', 'During Image'),
                            ('ied_after_image', 'After Image'),
                            ('ied_during_stim', 'During Stim')]:
            n = int(ied_filt[col].sum())
            tw_rows.append([label, str(n), f'{n/len(ied_filt)*100:.1f}%'])
        pdf.apa_table(
            f'Table {tbl_num}. {reg} IED Timing Window Distribution',
            ['Window', 'N', '%'], tw_rows, [50, 40, 40],
            note=f'N = {len(ied_filt)} IED trials. Windows not mutually exclusive.'
        )
        tbl_num += 1

        pdf.subsection_title('Main Effects: Power + Timing Windows')
        for b, bl in bands:
            df = load_or(f'pow_T_{reg}_{b}_odds_ratios.csv')
            if df is not None:
                rows = make_or_rows(df)
                pdf.apa_table(
                    f'Table {tbl_num}. {reg} {bl} + Timing',
                    OR_HEADERS, rows, OR_WIDTHS,
                    note=f'N = {len(ied_filt)} IED trials, {ied_filt["patient_id"].nunique()} patients.'
                )
                tbl_num += 1

        pdf.subsection_title('Interaction: Power x Timing Windows')
        for b, bl in bands:
            df = load_or(f'pow_TI_{reg}_{b}_odds_ratios.csv')
            if df is not None:
                rows = make_or_rows(df)
                pdf.apa_table(
                    f'Table {tbl_num}. {reg} {bl} x Timing Interactions',
                    OR_HEADERS, rows, OR_WIDTHS,
                    note=f'N = {len(ied_filt)} IED trials.'
                )
                tbl_num += 1

        pdf.subsection_title('Power + IED Spread')
        for b, bl in bands:
            df = load_or(f'pow_SP_{reg}_{b}_odds_ratios.csv')
            if df is not None:
                rows = make_or_rows(df)
                pdf.apa_table(
                    f'Table {tbl_num}. {reg} {bl} + Spread',
                    OR_HEADERS, rows, OR_WIDTHS,
                    note=f'N = {len(ied_filt)} IED trials.'
                )
                tbl_num += 1

    # ═══ SUMMARY ═══
    pdf.add_page()
    pdf.section_title('Summary of Significant Findings')

    sig_findings = []
    for f in sorted(glob.glob(os.path.join(OUTPUT_DIR, 'pow_*_odds_ratios.csv'))):
        df = pd.read_csv(f)
        sig = df[(df['p.value'] < 0.05) & (df['term'] != '(Intercept)')]
        if len(sig) > 0:
            name = os.path.basename(f).replace('_odds_ratios.csv', '')
            for _, r in sig.iterrows():
                label = TERM_LABELS.get(r['term'], r['term'])
                sig_findings.append(
                    f'{name}: {label} OR={r["estimate"]:.2f} '
                    f'{ci_str(r["conf.low"], r["conf.high"])}, '
                    f'p={p_str(r["p.value"])}'
                )

    if sig_findings:
        pdf.body_text('Effects reaching p < .05:')
        for finding in sig_findings:
            pdf.body_text(f'  - {finding}')
    else:
        pdf.body_text('No effects reached p < .05.')

    pdf.body_text(
        'Key patterns: (1) IED timing window effects replicate across power '
        'regions - during-stim and after-image IEDs consistently predict '
        'worse memory (OR ~ 0.57-0.67). (2) BLA power x during-stim IED '
        'interactions: higher theta and slow gamma power in BLA on trials '
        'with during-stim IEDs was associated with worse memory. '
        '(3) IED region spread was a significant negative predictor '
        'in BLA models (OR ~ 0.60, p ~ .03).'
    )

    out_path = os.path.join(OUTPUT_DIR, 'Power_x_IED_Effects_MLM.pdf')
    pdf.output(out_path)
    print(f'Report saved -> {out_path}')


if __name__ == '__main__':
    build_report()
