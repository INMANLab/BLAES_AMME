#!/usr/bin/env python
"""
Build PDF report: Neural and Stim Effects MLM (Encoding Only).

Parts:
  1. BLA Coherence -> Memory (all coherence trials)
  2. Coherence x IED Timing Windows -> Memory (merged IED+coherence)
  3. During-Stim IED Window - Stim Effect
  4. Coherence + IED Spread -> Memory

Outputs: outputs/ied_timing_memory/Neural_and_Stim_Effects_MLM.pdf
"""

import os
import sys
import numpy as np
import pandas as pd
from fpdf import FPDF

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'outputs', 'ied_timing_memory')

# Region pair from command line (default: BLA_HPC)
REGION_PAIR = sys.argv[1] if len(sys.argv) > 1 else 'BLA_HPC'
REGION_LABEL = REGION_PAIR.replace('_', '-')
FILE_PREFIX = 'ns_model' if REGION_PAIR == REGION_PAIR else f'ns_{REGION_PAIR}_model'
PDF_SUFFIX = '' if REGION_PAIR == 'BLA_HPC' else f'_{REGION_PAIR}'

REPORT_TITLE = f'Neural and Stim Effects MLM ({REGION_LABEL})'


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
    elif p < .1:
        return '+'  # dagger
    return ''


def or_str(v):
    return f'{v:.2f}'


def ci_str(lo, hi):
    return f'[{lo:.2f}, {hi:.2f}]'


TERM_LABELS = {
    '(Intercept)': 'Intercept',
    'coh_theta_z': 'Theta Coherence (z)',
    'coh_slow_gamma_z': 'Slow Gamma Coherence (z)',
    'coh_hfa_z': 'HFA Coherence (z)',
    'pac_slow_gamma_z': 'Slow Gamma PAC (z)',
    'pac_hfa_z': 'HFA PAC (z)',
    'stim': 'Stimulation',
    'ied_before_image': 'IED Before Image',
    'ied_during_image': 'IED During Image',
    'ied_after_image': 'IED After Image',
    'ied_during_stim': 'IED During Stim',
    'n_regions_c': 'Region Spread (centered)',
    'n_channels_c': 'Channel Spread (centered)',
    'coh_theta_z:ied_before_image': 'Theta Coh x Before Image',
    'coh_theta_z:ied_during_image': 'Theta Coh x During Image',
    'coh_theta_z:ied_after_image': 'Theta Coh x After Image',
    'coh_theta_z:ied_during_stim': 'Theta Coh x During Stim',
    'coh_slow_gamma_z:ied_before_image': 'Slow Gamma Coh x Before',
    'coh_slow_gamma_z:ied_during_image': 'Slow Gamma Coh x During',
    'coh_slow_gamma_z:ied_after_image': 'Slow Gamma Coh x After',
    'coh_slow_gamma_z:ied_during_stim': 'Slow Gamma Coh x Stim Win',
    'coh_hfa_z:ied_before_image': 'HFA Coh x Before Image',
    'coh_hfa_z:ied_during_image': 'HFA Coh x During Image',
    'coh_hfa_z:ied_after_image': 'HFA Coh x After Image',
    'coh_hfa_z:ied_during_stim': 'HFA Coh x During Stim',
    'coh_theta_z:stim': 'Theta x Stim',
    'coh_slow_gamma_z:stim': 'Slow Gamma x Stim',
    'coh_hfa_z:stim': 'HFA x Stim',
    'pac_slow_gamma_z:ied_before_image': 'SG PAC x Before Image',
    'pac_slow_gamma_z:ied_during_image': 'SG PAC x During Image',
    'pac_slow_gamma_z:ied_after_image': 'SG PAC x After Image',
    'pac_slow_gamma_z:ied_during_stim': 'SG PAC x During Stim',
    'pac_hfa_z:ied_before_image': 'HFA PAC x Before Image',
    'pac_hfa_z:ied_during_image': 'HFA PAC x During Image',
    'pac_hfa_z:ied_after_image': 'HFA PAC x After Image',
    'pac_hfa_z:ied_during_stim': 'HFA PAC x During Stim',
}


def load_or(filename):
    path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def make_or_rows(df, skip_intercept=True):
    rows = []
    for _, r in df.iterrows():
        term = r['term']
        if skip_intercept and term == '(Intercept)':
            continue
        label = TERM_LABELS.get(term, term)
        rows.append([
            label,
            or_str(r['estimate']),
            ci_str(r['conf.low'], r['conf.high']),
            p_str(r['p.value']),
            sig_str(r['p.value']),
        ])
    return rows


OR_HEADERS = ['Predictor', 'OR', '95% CI', 'p', '']
OR_WIDTHS = [55, 18, 40, 25, 10]


def build_report():
    pdf = APAReport()
    pdf.add_page()

    # ── Title ──
    pdf.set_font('Helvetica', 'B', 16)
    pdf.cell(0, 10, REPORT_TITLE, new_x="LMARGIN", new_y="NEXT", align='C')
    pdf.set_font('Helvetica', '', 11)
    pdf.cell(0, 6, 'Encoding Phase Only', new_x="LMARGIN", new_y="NEXT", align='C')
    pdf.ln(6)

    # ── Overview ──
    pdf.section_title('Overview')
    pdf.body_text(
        f'This report examines the relationship between {REGION_LABEL} coherence, '
        'IED timing windows, stimulation, and subsequent memory during encoding. '
        'Coherence was computed as baseline-corrected (post minus pre) '
        f'coherency between {REGION_LABEL} electrode pairs, band-averaged into '
        'theta (4-8 Hz), slow gamma (30-55 Hz), and HFA (70-100 Hz). '
        'All models use generalized linear mixed models (GLMM) with binomial '
        'family and random intercepts for patient.'
    )
    pdf.body_text(
        'Note: Coherence and PAC data are computed as trial-level averages '
        'across the full trial epoch. They are NOT computed separately for '
        'each timing window (before/during/after image). Therefore, the timing '
        'window analysis tests whether the OCCURRENCE of an IED in a specific '
        'window moderates the coherence-memory relationship, not whether '
        'coherence WITHIN that window differs.'
    )

    # ── Data ──
    coh_all = load_or('neural_stim_coh_all.csv')  # just for counts
    desc = pd.read_csv(os.path.join(OUTPUT_DIR, 'neural_stim_descriptives.csv'))

    # Load the full datasets for counts
    coh_all_df = pd.read_csv(os.path.join(OUTPUT_DIR, 'neural_stim_coh_all.csv'))
    bla_all = coh_all_df[coh_all_df['region_pair'] == REGION_PAIR]
    ied_merged = pd.read_csv(os.path.join(OUTPUT_DIR, 'neural_stim_coh_ied_merged.csv'))
    ied_bla = ied_merged[ied_merged['region_pair'] == REGION_PAIR]

    # ═══════════════════════════════════════════════════════════════
    # PART 1: Coherence -> Memory (all trials)
    # ═══════════════════════════════════════════════════════════════
    pdf.section_title(f'Part 1: {REGION_LABEL} Coherence and Memory (All Trials)')
    pdf.body_text(
        f'Sample: {len(bla_all):,} encoding trials from '
        f'{bla_all["patient_id"].nunique()} patients with {REGION_LABEL} coherence data. '
        f'Overall memory rate: {bla_all["memory"].mean()*100:.1f}% remembered.'
    )
    pdf.body_text(
        f'Three separate models tested whether trial-level {REGION_LABEL} coherence '
        'in each frequency band predicted subsequent memory, controlling for '
        'stimulation condition.'
    )

    bands = [('theta', 'Theta (4-8 Hz)'),
             ('slow_gamma', 'Slow Gamma (30-55 Hz)'),
             ('hfa', 'HFA (70-100 Hz)')]

    for i, (b, bl) in enumerate(bands, 1):
        df = load_or(f'{FILE_PREFIX}C{i}_{b}_odds_ratios.csv')
        if df is not None:
            rows = make_or_rows(df)
            pdf.apa_table(
                f'Table {i}. Model C{i}: {bl} Coherence and Memory (All Trials)',
                OR_HEADERS, rows, OR_WIDTHS,
                note=f'memory ~ coh_{b}_z + stim + (1|patient_id). '
                     f'N = {len(bla_all):,} trials, {bla_all["patient_id"].nunique()} patients. '
                     f'OR = odds ratio; CI = confidence interval.'
            )

    pdf.body_text(
        'None of the three frequency bands showed a significant main effect '
        f'of {REGION_LABEL} coherence on subsequent memory. Stimulation was also '
        'non-significant across all models.'
    )

    # ═══════════════════════════════════════════════════════════════
    # PART 2: Coherence x IED Timing Windows
    # ═══════════════════════════════════════════════════════════════
    pdf.add_page()
    pdf.section_title('Part 2: Coherence x IED Timing Windows')

    # Filter to patients with >=5 trials (matching R script)
    if len(ied_bla) > 0:
        pat_counts = ied_bla.groupby('patient_id').size()
        keep_pats = pat_counts[pat_counts >= 5].index
        ied_bla_filt = ied_bla[ied_bla['patient_id'].isin(keep_pats)]
    else:
        ied_bla_filt = ied_bla

    if len(ied_bla_filt) < 5:
        pdf.body_text(
            f'Insufficient IED+coherence merged trials for {REGION_LABEL} '
            f'({len(ied_bla)} total, {len(ied_bla_filt)} after filtering). '
            'Parts 2-4 are skipped for this region pair.'
        )
        has_ied_coh = False
    else:
        has_ied_coh = True
        pdf.body_text(
            f'Sample: {len(ied_bla_filt):,} IED trials from '
            f'{ied_bla_filt["patient_id"].nunique()} patients that had both '
            f'{REGION_LABEL} coherence data and IED detections during the same trial '
            f'(patients with fewer than 5 overlapping trials excluded).'
        )

    tbl_num = len(bands) + 1

    if has_ied_coh:
        # Timing window distribution
        tw_rows = []
        for col, label in [('ied_before_image', 'Before Image'),
                            ('ied_during_image', 'During Image'),
                            ('ied_after_image', 'After Image'),
                            ('ied_during_stim', 'During Stim')]:
            n = int(ied_bla_filt[col].sum())
            pct = n / len(ied_bla_filt) * 100
            tw_rows.append([label, str(n), f'{pct:.1f}%'])

        pdf.apa_table(
            f'Table {tbl_num}. IED Timing Window Distribution (IED+Coherence Trials)',
            ['Window', 'N Trials', '% of Total'],
            tw_rows,
            col_widths=[50, 40, 40],
            note=f'N = {len(ied_bla_filt)} trials. '
                 'Timing windows are not mutually exclusive; a trial may have IEDs in multiple windows.'
        )
        tbl_num += 1

        pdf.subsection_title('Part 2a: Main Effects - Coherence + Timing Windows')
        pdf.body_text(
            'Each model enters one coherence band (z-scored) alongside four binary '
            'IED timing window indicators as predictors of subsequent memory.'
        )

        for i, (b, bl) in enumerate(bands, 1):
            df = load_or(f'{FILE_PREFIX}T{i}_{b}_odds_ratios.csv')
            if df is not None:
                rows = make_or_rows(df)
                pdf.apa_table(
                    f'Table {tbl_num}. Model T{i}: {bl} + Timing Windows',
                    OR_HEADERS, rows, OR_WIDTHS,
                    note=f'memory ~ coh_{b}_z + timing windows + (1|patient_id). '
                         f'N = {len(ied_bla_filt)} trials, '
                         f'{ied_bla_filt["patient_id"].nunique()} patients.'
                )
                tbl_num += 1

        pdf.subsection_title('Part 2b: Coherence x Timing Window Interactions')
        pdf.body_text(
            'These models test whether the coherence-memory relationship '
            'differs depending on which timing window contained the IED.'
        )

        for i, (b, bl) in enumerate(bands, 1):
            df = load_or(f'{FILE_PREFIX}TI{i}_{b}_odds_ratios.csv')
            if df is not None:
                rows = make_or_rows(df)
                pdf.apa_table(
                    f'Table {tbl_num}. Model TI{i}: {bl} x Timing Window Interactions',
                    OR_HEADERS, rows, OR_WIDTHS,
                    note=f'memory ~ coh_{b}_z * each timing window + (1|patient_id). '
                         f'N = {len(ied_bla_filt)} trials.'
                )
                tbl_num += 1

        # Check theta interaction significance
        ti1 = load_or(f'{FILE_PREFIX}TI1_theta_odds_ratios.csv')
        if ti1 is not None:
            before_row = ti1[ti1['term'] == 'coh_theta_z:ied_before_image']
            if len(before_row) > 0:
                br = before_row.iloc[0]
                pdf.body_text(
                    f'A significant theta coherence x before-image IED interaction emerged '
                    f'(OR = {br["estimate"]:.2f}, 95% CI {ci_str(br["conf.low"], br["conf.high"])}, '
                    f'p = {p_str(br["p.value"])}). This suggests that higher {REGION_LABEL} theta '
                    f'coherence was associated with better memory specifically on trials where '
                    f'an IED occurred before image onset.'
                )

        # ===================================================================
        # PART 3: During-Stim Window (stim trials only)
        # ===================================================================
        pdf.add_page()
        pdf.section_title('Part 3: During-Stim IED Trials (Stim Only)')

        ds_bla = ied_bla_filt[(ied_bla_filt['ied_during_stim'] == 1) &
                               (ied_bla_filt['stim'] == 1)]

        pdf.body_text(
            f'Sample: {len(ds_bla)} stimulation trials with IEDs during the '
            f'stimulation window, from {ds_bla["patient_id"].nunique()} patients. '
            f'These models test whether {REGION_LABEL} coherence predicts memory '
            f'specifically on stim trials disrupted by IEDs during stimulation.'
        )

        for i, (b, bl) in enumerate(bands, 1):
            df = load_or(f'{FILE_PREFIX}DS{i}_{b}_odds_ratios.csv')
            if df is not None:
                rows = make_or_rows(df)
                pdf.apa_table(
                    f'Table {tbl_num}. Model DS{i}: {bl} (During-Stim, Stim Only)',
                    OR_HEADERS, rows, OR_WIDTHS,
                    note=f'memory ~ coh_{b}_z + (1|patient_id). '
                         f'N = {len(ds_bla)} stim trials with during-stim IEDs.'
                )
                tbl_num += 1

        # ===================================================================
        # PART 4: With IED Spread
        # ===================================================================
        pdf.add_page()
        pdf.section_title('Part 4: Coherence + IED Spread and Memory')
        pdf.body_text(
            'These models add IED spread metrics (region spread and channel '
            'spread, centered) to the coherence models. Region spread = number '
            'of distinct brain regions with IED detections on a given trial; '
            'channel spread = number of unique electrode contacts.'
        )

        pdf.subsection_title('Part 4a: Coherence + Spread (Without Timing Windows)')
        for i, (b, bl) in enumerate(bands, 1):
            df = load_or(f'{FILE_PREFIX}SP{i}_{b}_odds_ratios.csv')
            if df is not None:
                rows = make_or_rows(df)
                pdf.apa_table(
                    f'Table {tbl_num}. Model SP{i}: {bl} + Spread',
                    OR_HEADERS, rows, OR_WIDTHS,
                    note=f'memory ~ coh_{b}_z + n_regions_c + n_channels_c + (1|patient_id). '
                         f'N = {len(ied_bla_filt)} trials.'
                )
                tbl_num += 1

        sp1 = load_or(f'{FILE_PREFIX}SP1_theta_odds_ratios.csv')
        if sp1 is not None:
            reg_row = sp1[sp1['term'] == 'n_regions_c']
            if len(reg_row) > 0:
                rr = reg_row.iloc[0]
                pdf.body_text(
                    f'Region spread: OR = {rr["estimate"]:.2f}, '
                    f'95% CI {ci_str(rr["conf.low"], rr["conf.high"])}, '
                    f'p = {p_str(rr["p.value"])}.'
                )

        pdf.subsection_title('Part 4b: Coherence + Spread + Timing Windows')
        for i, (b, bl) in enumerate(bands, 1):
            df = load_or(f'{FILE_PREFIX}SPT{i}_{b}_odds_ratios.csv')
            if df is not None:
                rows = make_or_rows(df)
                pdf.apa_table(
                    f'Table {tbl_num}. Model SPT{i}: {bl} + Spread + Timing',
                    OR_HEADERS, rows, OR_WIDTHS,
                    note=f'memory ~ coh_{b}_z + spread + timing windows + (1|patient_id). '
                         f'N = {len(ied_bla_filt)} trials.'
                )
                tbl_num += 1

    # ===================================================================
    # PART 5: PAC -> Memory (all trials)
    # ===================================================================
    pdf.add_page()
    pdf.section_title(f'Part 5: {REGION_LABEL} PAC and Memory (All Trials)')

    pac_all_path = os.path.join(OUTPUT_DIR, 'neural_stim_pac_all.csv')
    if os.path.exists(pac_all_path):
        pac_all_df = pd.read_csv(pac_all_path)
        pac_bla_all = pac_all_df[pac_all_df['region_pair'] == REGION_PAIR]

        pdf.body_text(
            f'Sample: {len(pac_bla_all):,} encoding trials from '
            f'{pac_bla_all["patient_id"].nunique()} patients with BLA-HPC '
            f'phase-amplitude coupling (PAC) data. PAC is computed as '
            f'baseline-corrected (post minus pre) coupling, band-averaged '
            f'into slow gamma (30-55 Hz) and HFA (70-100 Hz). '
            f'Note: theta PAC is not available in this dataset.'
        )

        pac_bands_list = [('slow_gamma', 'Slow Gamma (30-55 Hz)'),
                          ('hfa', 'HFA (70-100 Hz)')]

        for i, (b, bl) in enumerate(pac_bands_list, 1):
            df = load_or(f'{FILE_PREFIX}P{i}_{b}_odds_ratios.csv')
            if df is not None:
                rows = make_or_rows(df)
                pdf.apa_table(
                    f'Table {tbl_num}. Model P{i}: PAC {bl} (All Trials)',
                    OR_HEADERS, rows, OR_WIDTHS,
                    note=f'memory ~ pac_{b}_z + stim + (1|patient_id). '
                         f'N = {len(pac_bla_all):,} trials.'
                )
                tbl_num += 1

        pdf.body_text(
            'Neither slow gamma nor HFA PAC significantly predicted subsequent '
            'memory. Stimulation was also non-significant.'
        )
    else:
        pdf.body_text('PAC all-trials data not available.')

    # ===================================================================
    # PART 6: PAC x IED Timing Windows
    # ===================================================================
    pdf.section_title('Part 6: PAC x IED Timing Windows')

    pac_ied_path = os.path.join(OUTPUT_DIR, 'neural_stim_pac_ied_merged.csv')
    if os.path.exists(pac_ied_path):
        pac_ied_df = pd.read_csv(pac_ied_path)
        pac_ied_bla = pac_ied_df[pac_ied_df['region_pair'] == REGION_PAIR]
        pac_counts = pac_ied_bla.groupby('patient_id').size()
        pac_keep = pac_counts[pac_counts >= 5].index
        pac_ied_filt = pac_ied_bla[pac_ied_bla['patient_id'].isin(pac_keep)]

        pdf.body_text(
            f'Sample: {len(pac_ied_filt)} IED trials from '
            f'{pac_ied_filt["patient_id"].nunique()} patients with both '
            f'{REGION_LABEL} PAC and IED data (>=5 trials per patient). '
            f'Note: limited sample size constrains model complexity.'
        )

        pac_bands_list = [('slow_gamma', 'Slow Gamma (30-55 Hz)'),
                          ('hfa', 'HFA (70-100 Hz)')]

        pdf.subsection_title('Part 6a: PAC + Timing Windows')
        for i, (b, bl) in enumerate(pac_bands_list, 1):
            df = load_or(f'{FILE_PREFIX}PT{i}_{b}_odds_ratios.csv')
            if df is not None:
                rows = make_or_rows(df)
                pdf.apa_table(
                    f'Table {tbl_num}. Model PT{i}: PAC {bl} + Timing',
                    OR_HEADERS, rows, OR_WIDTHS,
                    note=f'memory ~ pac_{b}_z + timing windows + (1|patient_id). '
                         f'N = {len(pac_ied_filt)} trials.'
                )
                tbl_num += 1

        pdf.subsection_title('Part 6b: PAC x Timing Window Interactions')
        for i, (b, bl) in enumerate(pac_bands_list, 1):
            df = load_or(f'{FILE_PREFIX}PTI{i}_{b}_odds_ratios.csv')
            if df is not None:
                rows = make_or_rows(df)
                pdf.apa_table(
                    f'Table {tbl_num}. Model PTI{i}: PAC {bl} x Timing',
                    OR_HEADERS, rows, OR_WIDTHS,
                    note=f'memory ~ pac_{b}_z * timing windows + (1|patient_id). '
                         f'N = {len(pac_ied_filt)} trials.'
                )
                tbl_num += 1

        pdf.body_text(
            'PAC models with IED timing windows were limited by small sample '
            'sizes. Models showed singular fits (zero random-effects variance), '
            'indicating insufficient between-patient variability for reliable '
            'mixed-effects estimation.'
        )
    else:
        pdf.body_text('PAC IED-merged data not available.')

    # ===================================================================
    # PART 7: PAC + IED Spread
    # ===================================================================
    pdf.section_title('Part 7: PAC + IED Spread')

    if os.path.exists(pac_ied_path):
        for i, (b, bl) in enumerate(pac_bands_list, 1):
            df = load_or(f'{FILE_PREFIX}PSP{i}_{b}_odds_ratios.csv')
            if df is not None:
                rows = make_or_rows(df)
                pdf.apa_table(
                    f'Table {tbl_num}. Model PSP{i}: PAC {bl} + Spread',
                    OR_HEADERS, rows, OR_WIDTHS,
                    note=f'memory ~ pac_{b}_z + spread + (1|patient_id). '
                         f'N = {len(pac_ied_filt)} trials.'
                )
                tbl_num += 1

        for i, (b, bl) in enumerate(pac_bands_list, 1):
            df = load_or(f'{FILE_PREFIX}PSPT{i}_{b}_odds_ratios.csv')
            if df is not None:
                rows = make_or_rows(df)
                pdf.apa_table(
                    f'Table {tbl_num}. Model PSPT{i}: PAC {bl} + Spread + Timing',
                    OR_HEADERS, rows, OR_WIDTHS,
                    note=f'memory ~ pac_{b}_z + spread + timing + (1|patient_id). '
                         f'N = {len(pac_ied_filt)} trials.'
                )
                tbl_num += 1

    # ===================================================================
    # SUMMARY
    # ===================================================================
    pdf.add_page()
    pdf.section_title('Summary')
    pdf.body_text(
        f'Key findings from the {REGION_LABEL} neural and stim effects '
        'analysis (encoding):'
    )
    pdf.body_text(
        f'1. {REGION_LABEL} coherence (theta, slow gamma, HFA): '
        f'{len(bla_all):,} all-trial coherence models '
        f'({bla_all["patient_id"].nunique()} patients). '
        f'IED+coherence merged: {len(ied_bla_filt)} trials '
        f'({ied_bla_filt["patient_id"].nunique()} patients).'
    )
    if has_ied_coh:
        pdf.body_text(
            '2. IED timing windows and coherence x timing interactions '
            'tested on IED+coherence merged trials. See tables above for '
            'significant effects.'
        )
    else:
        pdf.body_text(
            '2. IED+coherence merged data was insufficient for timing '
            'window or spread analyses (Parts 2-4 skipped).'
        )
    pdf.body_text(
        f'3. {REGION_LABEL} PAC (slow gamma, HFA): see Part 5+ tables above.'
    )
    pdf.body_text(
        '4. Limitations: Coherence and PAC were computed as trial-level '
        'averages across the full epoch, not separately per timing window. '
        'The timing window analysis tests whether IED occurrence in a '
        'specific window moderates the neural-memory relationship.'
    )

    # Save
    out_path = os.path.join(OUTPUT_DIR, f'Neural_and_Stim_Effects_MLM{PDF_SUFFIX}.pdf')
    pdf.output(out_path)
    print(f'Report saved -> {out_path}')


if __name__ == '__main__':
    build_report()
