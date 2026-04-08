#!/usr/bin/env python
"""
Build PDF report for Hypothesis 3c (All Subjects): IED Laterality and Memory.
Uses all patients (not just bilateral-coverage).

Models:
  Encoding Model 1: memory ~ hemi_cat + stim + (1|patient_id)
  Encoding Model 2: memory ~ hemi_cat * stim + (1|patient_id)
  Encoding Model 3: memory ~ bilateral_ied + stim + (1|patient_id)
  Retrieval Model 4: memory ~ hemi_cat + (1|patient_id)

Outputs: outputs/ied_timing_memory/H3c_allsubjects_laterality_MLM_report.pdf
"""

import os
import numpy as np
import pandas as pd
from fpdf import FPDF

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'outputs', 'ied_timing_memory')


class APAReport(FPDF):
    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=25)

    def header(self):
        self.set_font('Helvetica', 'I', 8)
        self.cell(0, 5, 'Hypothesis 3c (All Subjects): IED Laterality and Memory', align='R')
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
    return ''


def format_or_row(r, label_map=None):
    term = r['term']
    label = label_map.get(term, term) if label_map else term
    p = r['p.value']
    or_val = r['estimate']
    if or_val > 1e6 or or_val < 1e-6:
        return [label, 'n.e.', 'n.e.', f'{r["statistic"]:.2f}', f'{p_str(p)}']
    return [
        label,
        f'{or_val:.3f}',
        f'[{r["conf.low"]:.3f}, {r["conf.high"]:.3f}]',
        f'{r["statistic"]:.2f}',
        f'{p_str(p)}{sig_str(p)}',
    ]


def main():
    pdf = APAReport()
    pdf.add_page()

    # Title
    pdf.set_font('Helvetica', 'B', 16)
    pdf.cell(0, 10, 'Hypothesis 3c (All Subjects):', align='C',
             new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 10, 'IED Laterality and Memory', align='C',
             new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    pdf.italic_text(
        'Frequent bilateral IED occurrence (e.g., both hippocampi at the same time) during '
        'learning or retrieval will predict subsequent memory impairment regardless of BLA '
        'stimulation.'
    )

    # Load data
    enc = pd.read_csv(os.path.join(OUTPUT_DIR, 'h3c_encoding_all.csv'))
    ret = pd.read_csv(os.path.join(OUTPUT_DIR, 'h3c_retrieval_all.csv'))

    m1 = pd.read_csv(os.path.join(OUTPUT_DIR, 'h3c_allsub_model1_odds_ratios.csv'))
    m2 = pd.read_csv(os.path.join(OUTPUT_DIR, 'h3c_allsub_model2_odds_ratios.csv'))
    lrt12 = pd.read_csv(os.path.join(OUTPUT_DIR, 'h3c_allsub_lrt_m1_vs_m2.csv'))

    m3 = pd.read_csv(os.path.join(OUTPUT_DIR, 'h3c_allsub_model3_odds_ratios.csv'))

    m5 = pd.read_csv(os.path.join(OUTPUT_DIR, 'h3c_allsub_model5_odds_ratios.csv'))

    m4_path = os.path.join(OUTPUT_DIR, 'h3c_allsub_model4_odds_ratios.csv')
    m4 = pd.read_csv(m4_path) if os.path.exists(m4_path) else None

    n_enc = len(enc)
    n_enc_pats = enc['patient_id'].nunique()
    n_ret = len(ret)
    n_ret_pats = ret['patient_id'].nunique()

    # ================================================================
    # SAMPLE DESCRIPTION
    # ================================================================
    pdf.section_title('Sample and Data Structure')

    enc_bi_pats = enc[enc['trial_hemisphere'] == 'Bilateral']['patient_id'].unique()
    n_bi_enc = len(enc_bi_pats)
    ret_bi_pats = ret[ret['trial_hemisphere'] == 'Bilateral']['patient_id'].unique()
    n_bi_ret = len(ret_bi_pats)

    pdf.body_text(
        f'This analysis includes all {n_enc_pats} patients at encoding ({n_enc} trials) and '
        f'all {n_ret_pats} patients at retrieval ({n_ret} trials). Each trial is classified '
        f'by the laterality of IEDs detected on that trial: Left (L), Right (R), or Bilateral '
        f'(IEDs detected in both hemispheres simultaneously). '
        f'Bilateral IEDs can only be detected in patients with bilateral electrode coverage '
        f'({n_bi_enc} patients at encoding, {n_bi_ret} at retrieval); patients with unilateral '
        f'coverage contribute only L or R trials by definition.'
    )

    pdf.body_text(
        f'Patients with bilateral IED trials at encoding: {", ".join(sorted(enc_bi_pats))}. '
        f'Patients with bilateral IED trials at retrieval: '
        f'{", ".join(sorted(ret_bi_pats)) if len(ret_bi_pats) > 0 else "none"}.'
    )

    # ================================================================
    # ENCODING DESCRIPTIVES
    # ================================================================
    pdf.section_title('Encoding Phase: Descriptive Statistics')

    # Table 1: Hemisphere distribution
    desc_rows = []
    for h in ['L', 'R', 'Bilateral']:
        sub = enc[enc['trial_hemisphere'] == h]
        n = len(sub)
        nr = int(sub['memory'].sum())
        ns = int(sub['stim'].sum())
        desc_rows.append([h, n, nr, n - nr, f'{nr/n*100:.1f}%', ns, n - ns])
    n_tot = len(enc)
    nr_tot = int(enc['memory'].sum())
    ns_tot = int(enc['stim'].sum())
    desc_rows.append(['Total', n_tot, nr_tot, n_tot - nr_tot,
                      f'{nr_tot/n_tot*100:.1f}%', ns_tot, n_tot - ns_tot])

    pdf.apa_table(
        'Table 1\nEncoding Trial Distribution by IED Laterality (All Patients)',
        ['Hemisphere', 'N Trials', 'Rem', 'Forg', '% Rem', 'Stim', 'NoStim'],
        desc_rows,
        col_widths=[22, 18, 14, 14, 18, 14, 18],
        note=f'N = {n_enc} trials from {n_enc_pats} patients. '
             f'Bilateral = IEDs detected in both hemispheres on the same trial '
             f'(possible only for {n_bi_enc} patients with bilateral electrode coverage).'
    )

    # Per-patient table
    pat_rows = []
    for pat in sorted(enc['patient_id'].unique()):
        sub = enc[enc['patient_id'] == pat]
        n = len(sub)
        n_l = int((sub['trial_hemisphere'] == 'L').sum())
        n_r = int((sub['trial_hemisphere'] == 'R').sum())
        n_bi = int((sub['trial_hemisphere'] == 'Bilateral').sum())
        pct_rem = sub['memory'].mean() * 100
        coverage = 'Bilateral' if n_bi > 0 else ('L' if n_l > 0 and n_r == 0 else
                                                   'R' if n_r > 0 and n_l == 0 else 'L+R')
        pat_rows.append([pat, n, n_l, n_r, n_bi, coverage, f'{pct_rem:.1f}%'])

    pdf.apa_table(
        'Table 2\nPer-Patient IED Laterality Profile (Encoding, All Patients)',
        ['Patient', 'N', 'Left', 'Right', 'Bilat', 'Coverage', '% Rem'],
        pat_rows,
        col_widths=[22, 12, 14, 14, 14, 20, 18],
        note='Coverage = electrode laterality. Bilat = trials with simultaneous L+R IEDs. '
             '% Rem = overall memory rate.'
    )

    # Timing window breakdown
    timing_rows = []
    for h in ['L', 'R', 'Bilateral']:
        sub = enc[enc['trial_hemisphere'] == h]
        n = len(sub)
        for col, label in [('ied_before_image', 'Before Img'),
                            ('ied_during_image', 'During Img'),
                            ('ied_after_image', 'After Img'),
                            ('ied_during_stim', 'During Stim')]:
            nw = int(sub[col].sum())
            timing_rows.append([h, label, nw, f'{nw/n*100:.1f}%'])

    pdf.apa_table(
        'Table 3\nEncoding IED Timing Window Distribution by Hemisphere (All Patients)',
        ['Hemisphere', 'Window', 'N IED+', '% of Trials'],
        timing_rows,
        col_widths=[25, 28, 22, 28],
        note='Percentage of trials within each hemisphere category that had IEDs in each window.'
    )

    # ================================================================
    # ENCODING MODELS
    # ================================================================
    pdf.section_title('Encoding Phase: Mixed-Effects Models')

    # ── Model 1: Main Effects ──
    pdf.subsection_title('Model 1: IED Laterality + Stimulation (Main Effects)')
    pdf.italic_text('memory ~ hemi_cat + stim + (1 | patient_id)')
    pdf.body_text(
        f'This model includes all {n_enc_pats} patients ({n_enc} trials) and tests whether '
        f'IED laterality (Bilateral and Right vs. Left [reference]) and stimulation predict '
        f'memory.'
    )

    label_map_1 = {
        '(Intercept)': '(Intercept)',
        'hemi_catBilateral': 'Bilateral (vs Left)',
        'hemi_catR': 'Right (vs Left)',
        'stim': 'Stimulation',
    }
    rows_1 = [format_or_row(r, label_map_1) for _, r in m1.iterrows()]

    pdf.apa_table(
        'Table 4\nModel 1: IED Laterality Predicting Memory (Encoding, All Patients)',
        ['Predictor', 'OR', '95% CI', 'z', 'p'],
        rows_1,
        col_widths=[38, 16, 35, 16, 22],
        note=f'GLMM with random intercept for patient. N = {n_enc} trials, {n_enc_pats} patients. '
             f'Reference = Left hemisphere IED. OR < 1 = lower odds of remembering. '
             f'*p < .05. **p < .01.'
    )

    bi_or = m1[m1['term'] == 'hemi_catBilateral']['estimate'].values[0]
    bi_p = m1[m1['term'] == 'hemi_catBilateral']['p.value'].values[0]
    r_or = m1[m1['term'] == 'hemi_catR']['estimate'].values[0]
    r_p = m1[m1['term'] == 'hemi_catR']['p.value'].values[0]
    stim_or = m1[m1['term'] == 'stim']['estimate'].values[0]
    stim_p = m1[m1['term'] == 'stim']['p.value'].values[0]

    pdf.body_text(
        f'Bilateral IEDs were significantly associated with reduced memory compared to left '
        f'IEDs (OR = {bi_or:.2f}, p = {p_str(bi_p)}). Trials with bilateral IEDs had '
        f'approximately half the odds of being remembered. Right IEDs were associated with '
        f'numerically higher memory than left IEDs (OR = {r_or:.2f}, p = {p_str(r_p)}), '
        f'but this difference was not significant. Stimulation had no main effect '
        f'(OR = {stim_or:.2f}, p = {p_str(stim_p)}).'
    )

    # Forest plot
    forest_enc = os.path.join(OUTPUT_DIR, 'h3c_allsub_encoding_forest.png')
    if os.path.exists(forest_enc):
        pdf.ln(2)
        pdf.italic_text(
            'Figure 1. Encoding: IED laterality effects on memory '
            '(Model 1, all patients). Reference = Left IED.'
        )
        img_w = pdf.w - pdf.l_margin - pdf.r_margin - 10
        pdf.image(forest_enc, x=pdf.l_margin + 5, w=img_w)
        pdf.ln(3)

    # ── Model 2: Interactions ──
    pdf.subsection_title('Model 2: IED Laterality x Stimulation (Interactions)')
    pdf.italic_text('memory ~ hemi_cat * stim + (1 | patient_id)')
    pdf.body_text(
        f'This model adds laterality x stimulation interactions to test whether the effects '
        f'of bilateral or right IEDs on memory depend on stimulation condition.'
    )

    label_map_2 = {
        '(Intercept)': '(Intercept)',
        'hemi_catBilateral': 'Bilateral (vs Left)',
        'hemi_catR': 'Right (vs Left)',
        'stim': 'Stimulation',
        'hemi_catBilateral:stim': 'Bilateral x Stim',
        'hemi_catR:stim': 'Right x Stim',
    }
    rows_2 = [format_or_row(r, label_map_2) for _, r in m2.iterrows()]

    # Extract LRT values
    lrt_chi = lrt12['Chisq'].dropna().values
    lrt_p = lrt12['Pr(>Chisq)'].dropna().values
    lrt_chi_str = f'{lrt_chi[0]:.2f}' if len(lrt_chi) > 0 else 'n/a'
    lrt_p_str = p_str(lrt_p[0]) if len(lrt_p) > 0 else 'n/a'

    pdf.apa_table(
        'Table 5\nModel 2: IED Laterality x Stimulation Predicting Memory (Encoding, All Patients)',
        ['Predictor', 'OR', '95% CI', 'z', 'p'],
        rows_2,
        col_widths=[38, 16, 35, 16, 22],
        note=f'GLMM with random intercept for patient. N = {n_enc} trials, {n_enc_pats} patients. '
             f'Reference = Left hemisphere IED. '
             f'LRT comparing Model 1 vs Model 2: chi2(2) = {lrt_chi_str}, p = {lrt_p_str}. '
             f'*p < .05. **p < .01.'
    )

    bi2_int_p = m2[m2['term'] == 'hemi_catBilateral:stim']['p.value'].values[0]
    r2_int_p = m2[m2['term'] == 'hemi_catR:stim']['p.value'].values[0]

    pdf.body_text(
        f'Neither the bilateral x stimulation (p = {p_str(bi2_int_p)}) nor the right x '
        f'stimulation (p = {p_str(r2_int_p)}) interaction was significant. The LRT comparing '
        f'Model 1 (main effects) to Model 2 (with interactions) confirmed that adding the '
        f'interaction terms did not improve model fit (chi2(2) = {lrt_chi_str}, '
        f'p = {lrt_p_str}). The laterality effects on memory do not depend on stimulation '
        f'condition.'
    )

    # ── Model 3: Bilateral vs Unilateral ──
    pdf.subsection_title('Model 3: Bilateral vs. Unilateral IEDs')
    pdf.italic_text('memory ~ bilateral_ied + stim + (1 | patient_id)')
    pdf.body_text(
        f'This model collapses L and R into "unilateral" and tests whether bilateral IEDs '
        f'(simultaneously detected in both hemispheres) predict worse memory across all '
        f'{n_enc_pats} patients ({n_enc} trials).'
    )

    label_map_3 = {
        '(Intercept)': '(Intercept)',
        'bilateral_ied': 'Bilateral IED',
        'stim': 'Stimulation',
    }
    rows_3 = [format_or_row(r, label_map_3) for _, r in m3.iterrows()]

    n_bi_trials = len(enc[enc['trial_hemisphere'] == 'Bilateral'])
    n_uni_trials = n_enc - n_bi_trials

    pdf.apa_table(
        'Table 6\nModel 3: Bilateral vs. Unilateral IED Predicting Memory '
        '(Encoding, All Patients)',
        ['Predictor', 'OR', '95% CI', 'z', 'p'],
        rows_3,
        col_widths=[38, 16, 35, 16, 22],
        note=f'GLMM with random intercept for patient. N = {n_enc} trials '
             f'({n_uni_trials} unilateral, {n_bi_trials} bilateral), {n_enc_pats} patients. '
             f'Bilateral IED = 1 if IEDs in both hemispheres on that trial, '
             f'0 = unilateral (L or R). *p < .05. **p < .01.'
    )

    bi3_or = m3[m3['term'] == 'bilateral_ied']['estimate'].values[0]
    bi3_p = m3[m3['term'] == 'bilateral_ied']['p.value'].values[0]
    stim3_p = m3[m3['term'] == 'stim']['p.value'].values[0]

    pdf.body_text(
        f'Bilateral IEDs were significantly associated with reduced memory compared to '
        f'unilateral IEDs (OR = {bi3_or:.2f}, p = {p_str(bi3_p)}). Trials with bilateral '
        f'IEDs had approximately half the odds of being remembered. Stimulation was not '
        f'significant (p = {p_str(stim3_p)}).'
    )

    # ================================================================
    # BILATERAL IED REGION ANALYSIS
    # ================================================================
    pdf.section_title('Regional Origin of Bilateral IEDs')

    pdf.body_text(
        'To characterize the anatomical source of bilateral IEDs, we examined which brain '
        'regions contributed IED detections on bilateral trials. For each bilateral trial, '
        'we identified the region(s) in each hemisphere where IEDs were detected. Regions '
        'containing "Hippocampus" in their label were grouped under "Hippocampus" for this '
        'analysis.'
    )

    # Load raw data for region analysis
    ENCODING_CSV = os.path.join(SCRIPT_DIR, 'IED',
        'AMMEBLAES_IEDs_trial_level_dissertation_study_usethis_cleaned_with_memory.csv')
    RETRIEVAL_CSV = os.path.join(SCRIPT_DIR, 'IED',
        'AMMEBLAES_IEDs_trial_level_dissertation_test_usethis_cleaned_with_memory.csv')

    raw_enc = pd.read_csv(ENCODING_CSV)
    raw_enc = raw_enc[raw_enc['MemoryOutcome'].notna() & raw_enc['StimCond'].isin(['S', 'NS'])].copy()

    def classify_hemi(vals):
        clean = set(vals.dropna()) - {'D'}
        if 'L' in clean and 'R' in clean:
            return 'Bilateral'
        elif 'L' in clean:
            return 'L'
        elif 'R' in clean:
            return 'R'
        return 'None'

    hemi_map = raw_enc.groupby(['Patient', 'Trial']).agg(
        trial_hemisphere=('Hemisphere', classify_hemi)
    ).reset_index()

    bi_trial_keys = hemi_map[hemi_map['trial_hemisphere'] == 'Bilateral'][['Patient', 'Trial']]
    bi_rows = raw_enc.merge(bi_trial_keys, on=['Patient', 'Trial'])

    def simplify_region(r):
        if pd.isna(r):
            return 'Unknown'
        r = str(r)
        if 'Hippocampus' in r:
            return 'Hippocampus'
        if 'Amygdala' in r:
            return 'Amygdala'
        if 'Entorhinal' in r:
            return 'Entorhinal'
        if 'Parahippocampal' in r:
            return 'Parahippocampal'
        return 'Other'

    # Per-trial region involvement
    trial_info = []
    for (pat, trial), grp in bi_rows.groupby(['Patient', 'Trial']):
        l_regs = set(grp[grp['Hemisphere'] == 'L']['Region'].apply(simplify_region))
        r_regs = set(grp[grp['Hemisphere'] == 'R']['Region'].apply(simplify_region))
        hipp_L = 'Hippocampus' in l_regs
        hipp_R = 'Hippocampus' in r_regs
        hipp_both = hipp_L and hipp_R
        hipp_any = hipp_L or hipp_R
        amyg_any = 'Amygdala' in (l_regs | r_regs)
        mem = grp['MemoryOutcome'].iloc[0]
        trial_info.append({
            'Patient': pat, 'Trial': trial,
            'hipp_L': hipp_L, 'hipp_R': hipp_R,
            'hipp_bilateral': hipp_both, 'hipp_any': hipp_any,
            'amyg_any': amyg_any,
            'memory': 1 if mem == 'remembered' else 0,
        })

    ti = pd.DataFrame(trial_info)
    n_bi = len(ti)

    # ── Table: Region counts on bilateral trials ──
    region_counts = bi_rows.groupby(['Region', 'Hemisphere']).size().reset_index(name='count')
    # Simplify and aggregate
    bi_rows['region_simple'] = bi_rows['Region'].apply(simplify_region)
    region_hemi = bi_rows.groupby(['region_simple', 'Hemisphere']).size().unstack(fill_value=0)
    region_hemi['Total'] = region_hemi.sum(axis=1)
    region_hemi = region_hemi.sort_values('Total', ascending=False).reset_index()

    reg_rows = []
    for _, row in region_hemi.iterrows():
        n_l = int(row.get('L', 0))
        n_r = int(row.get('R', 0))
        tot = int(row['Total'])
        reg_rows.append([row['region_simple'], n_l, n_r, tot])

    tbl_num = 7  # next table number after Table 6

    pdf.apa_table(
        f'Table {tbl_num}\nRegion of Origin for IED Detections on Bilateral Trials (Encoding)',
        ['Region', 'Left', 'Right', 'Total'],
        reg_rows,
        col_widths=[40, 20, 20, 20],
        note=f'Counts of IED detections (electrode-level rows) on {n_bi} bilateral trials. '
             f'Regions containing "Hippocampus" grouped together. '
             f'"Other" includes temporal, parietal, frontal, and cingulate cortex.'
    )

    # ── Table: Hippocampal involvement per trial ──
    n_hipp_both = int(ti['hipp_bilateral'].sum())
    n_hipp_any = int(ti['hipp_any'].sum())
    n_hipp_l = int(ti['hipp_L'].sum())
    n_hipp_r = int(ti['hipp_R'].sum())
    n_amyg = int(ti['amyg_any'].sum())

    tbl_num += 1
    inv_rows = [
        ['Hippocampus (both hemispheres)', n_hipp_both, f'{n_hipp_both/n_bi*100:.1f}%'],
        ['Hippocampus (at least one hemisphere)', n_hipp_any, f'{n_hipp_any/n_bi*100:.1f}%'],
        ['Hippocampus (left only)', n_hipp_l, f'{n_hipp_l/n_bi*100:.1f}%'],
        ['Hippocampus (right only)', n_hipp_r, f'{n_hipp_r/n_bi*100:.1f}%'],
        ['Amygdala (either hemisphere)', n_amyg, f'{n_amyg/n_bi*100:.1f}%'],
    ]

    pdf.apa_table(
        f'Table {tbl_num}\nHippocampal Involvement in Bilateral IED Trials (Encoding)',
        ['Region Involvement', 'N Trials', '% of Bilateral'],
        inv_rows,
        col_widths=[55, 22, 30],
        note=f'Of {n_bi} bilateral encoding trials, showing how many involved hippocampal IEDs.'
    )

    pdf.body_text(
        f'The vast majority of bilateral IED trials involved hippocampal IEDs. '
        f'Hippocampal IEDs were detected in both hemispheres simultaneously on '
        f'{n_hipp_both} of {n_bi} bilateral trials ({n_hipp_both/n_bi*100:.1f}%), '
        f'and in at least one hemisphere on {n_hipp_any} trials ({n_hipp_any/n_bi*100:.1f}%). '
        f'Amygdala involvement was present on {n_amyg} bilateral trials '
        f'({n_amyg/n_bi*100:.1f}%).'
    )

    # ── Per-patient table ──
    tbl_num += 1
    pat_reg_rows = []
    for pat in sorted(ti['Patient'].unique()):
        sub = ti[ti['Patient'] == pat]
        n = len(sub)
        n_hb = int(sub['hipp_bilateral'].sum())
        n_ha = int(sub['hipp_any'].sum())
        pct_hb = f'{n_hb/n*100:.0f}%'
        pct_rem = f'{sub["memory"].mean()*100:.1f}%'
        pat_reg_rows.append([pat, n, n_hb, pct_hb, n_ha, pct_rem])

    pdf.apa_table(
        f'Table {tbl_num}\nPer-Patient Hippocampal Involvement in Bilateral Trials (Encoding)',
        ['Patient', 'N Bilat', 'Hipp Both', '% Hipp', 'Hipp Any', '% Rem'],
        pat_reg_rows,
        col_widths=[22, 18, 20, 16, 18, 16],
        note='N Bilat = bilateral IED trials. Hipp Both = hippocampus IED in both hemispheres. '
             'Hipp Any = hippocampus in at least one hemisphere. % Rem = memory rate on bilateral trials.'
    )

    # Memory by hippocampal involvement
    hipp_bi_trials = ti[ti['hipp_bilateral']]
    non_hipp_bi_trials = ti[~ti['hipp_bilateral']]
    n_hb_rem = int(hipp_bi_trials['memory'].sum())
    n_nhb_rem = int(non_hipp_bi_trials['memory'].sum())

    if len(non_hipp_bi_trials) > 0:
        pdf.body_text(
            f'Among bilateral trials with hippocampal IEDs in both hemispheres, '
            f'{n_hb_rem} of {len(hipp_bi_trials)} were remembered '
            f'({hipp_bi_trials["memory"].mean()*100:.1f}%). Among bilateral trials '
            f'without bilateral hippocampal involvement, {n_nhb_rem} of '
            f'{len(non_hipp_bi_trials)} were remembered '
            f'({non_hipp_bi_trials["memory"].mean()*100:.1f}%). This suggests that '
            f'bilateral hippocampal IEDs, while the dominant source of bilateral events, '
            f'are not necessarily more disruptive than bilateral IEDs originating from '
            f'other regions, though the small number of non-hippocampal bilateral trials '
            f'(n = {len(non_hipp_bi_trials)}) limits this comparison.'
        )

    pdf.body_text(
        f'In summary, bilateral IEDs during encoding are overwhelmingly hippocampal in origin. '
        f'On {n_hipp_both/n_bi*100:.0f}% of bilateral trials, hippocampal IEDs were detected '
        f'in both hemispheres simultaneously, consistent with the hypothesis that the bilateral '
        f'IED memory impairment reflects disruption of bilateral hippocampal encoding processes.'
    )

    # ================================================================
    # MODEL 5: TIMING WINDOW ON BILATERAL TRIALS
    # ================================================================
    pdf.section_title('Bilateral IED Timing Window Analysis')

    # Descriptive: timing windows on bilateral trials
    enc_bilat = enc[enc['trial_hemisphere'] == 'Bilateral'].copy()
    n_bilat = len(enc_bilat)
    n_bilat_pats = enc_bilat['patient_id'].nunique()

    pdf.body_text(
        f'To examine whether the timing of bilateral IEDs within a trial modulates their '
        f'impact on memory, we restricted the analysis to the {n_bilat} bilateral IED trials '
        f'from {n_bilat_pats} patients with bilateral electrode coverage. Each trial was coded '
        f'for IED presence in four timing windows: before image onset, during image, after '
        f'image, and during stimulation.'
    )

    # Descriptive table
    timing_desc_rows = []
    for col, label in [('ied_before_image', 'Before Image'),
                        ('ied_during_image', 'During Image'),
                        ('ied_after_image', 'After Image'),
                        ('ied_during_stim', 'During Stimulation')]:
        n1 = int(enc_bilat[col].sum())
        n0 = n_bilat - n1
        sub1 = enc_bilat[enc_bilat[col] == 1]
        sub0 = enc_bilat[enc_bilat[col] == 0]
        rem1 = sub1['memory'].mean() * 100 if len(sub1) > 0 else 0
        rem0 = sub0['memory'].mean() * 100 if len(sub0) > 0 else 0
        timing_desc_rows.append([
            label,
            f'{n1} ({n1/n_bilat*100:.1f}%)',
            f'{n0} ({n0/n_bilat*100:.1f}%)',
            f'{rem1:.1f}%',
            f'{rem0:.1f}%',
        ])

    pdf.apa_table(
        'Table 10\nBilateral IED Timing Window Distribution and Memory (Encoding)',
        ['Window', 'IED+', 'IED-', '% Rem (IED+)', '% Rem (IED-)'],
        timing_desc_rows,
        col_widths=[30, 22, 22, 26, 26],
        note=f'N = {n_bilat} bilateral IED trials from {n_bilat_pats} patients. '
             f'IED+ = bilateral IED detected in that window. % Rem = percentage remembered.'
    )

    # Model 5 table
    pdf.subsection_title('Model 5: Timing Window Effects on Bilateral IED Trials')
    pdf.italic_text(
        'memory ~ ied_before_image + ied_during_image + ied_after_image + '
        'ied_during_stim + (1 | patient_id)'
    )

    label_map_5 = {
        '(Intercept)': '(Intercept)',
        'ied_before_image': 'Before Image',
        'ied_during_image': 'During Image',
        'ied_after_image': 'After Image',
        'ied_during_stim': 'During Stimulation',
    }
    rows_5 = [format_or_row(r, label_map_5) for _, r in m5.iterrows()]

    pdf.apa_table(
        'Table 11\nModel 5: Timing Window Predicting Memory on Bilateral Trials (Encoding)',
        ['Predictor', 'OR', '95% CI', 'z', 'p'],
        rows_5,
        col_widths=[35, 16, 35, 16, 22],
        note=f'GLMM with random intercept for patient. N = {n_bilat} bilateral trials, '
             f'{n_bilat_pats} patients. OR < 1 = lower odds of remembering. *p < .05. **p < .01.'
    )

    # Extract values for text
    before_or = m5[m5['term'] == 'ied_before_image']['estimate'].values[0]
    before_p = m5[m5['term'] == 'ied_before_image']['p.value'].values[0]
    during_or = m5[m5['term'] == 'ied_during_image']['estimate'].values[0]
    during_p = m5[m5['term'] == 'ied_during_image']['p.value'].values[0]
    after_or = m5[m5['term'] == 'ied_after_image']['estimate'].values[0]
    after_p = m5[m5['term'] == 'ied_after_image']['p.value'].values[0]
    stim_or = m5[m5['term'] == 'ied_during_stim']['estimate'].values[0]
    stim_p = m5[m5['term'] == 'ied_during_stim']['p.value'].values[0]

    pdf.body_text(
        f'Among bilateral IED trials, the timing of IEDs within the trial predicted memory. '
        f'Bilateral IEDs occurring during the stimulation window were significantly associated '
        f'with reduced memory (OR = {stim_or:.2f}, p = {p_str(stim_p)}), with approximately '
        f'one-fifth the odds of remembering compared to bilateral trials without IEDs during '
        f'stimulation. IEDs before image onset (OR = {before_or:.2f}, p = {p_str(before_p)}), '
        f'during image (OR = {during_or:.2f}, p = {p_str(during_p)}), and after image '
        f'(OR = {after_or:.2f}, p = {p_str(after_p)}) showed non-significant trends toward '
        f'reduced memory.'
    )

    # Descriptive context: stim condition breakdown
    stim_yes = enc_bilat[enc_bilat['ied_during_stim'] == 1]
    stim_no = enc_bilat[enc_bilat['ied_during_stim'] == 0]

    # Break down by stim condition
    bi_stim = enc_bilat[enc_bilat['stim'] == 1]
    bi_nostim = enc_bilat[enc_bilat['stim'] == 0]
    bi_stim_durY = bi_stim[bi_stim['ied_during_stim'] == 1]
    bi_stim_durN = bi_stim[bi_stim['ied_during_stim'] == 0]

    pdf.subsection_title('Interpreting the During-Stimulation Effect')

    pdf.body_text(
        f'An important consideration is that the during-stimulation timing window only exists '
        f'on stimulation trials -- no stimulation is delivered on NoStim trials, so there is '
        f'no stim epoch during which to detect IEDs. Accordingly, all {len(stim_yes)} bilateral '
        f'during-stim IED trials are stimulation trials, and all {len(bi_nostim)} NoStim '
        f'bilateral trials have ied_during_stim = 0 by definition. The during-stim IED '
        f'predictor is therefore structurally confounded with stimulation condition.'
    )

    pdf.body_text(
        f'However, stimulation alone does not predict memory on bilateral trials: across '
        f'Models 1-3, stimulation was consistently non-significant (all ps > .88). The '
        f'significant during-stim effect in Model 5 (OR = {stim_or:.2f}, p = {p_str(stim_p)}) '
        f'therefore reflects the specific co-occurrence of bilateral IEDs with active '
        f'stimulation delivery, rather than a stimulation main effect. This may suggest that '
        f'bilateral hippocampal IEDs interfere with consolidation processes engaged by BLA '
        f'stimulation, though these results should be interpreted cautiously given the small '
        f'sample ({len(bi_stim)} stimulation bilateral trials from '
        f'{bi_stim["patient_id"].nunique()} patients) and the inherent confound.'
    )

    # ================================================================
    # RETRIEVAL DESCRIPTIVES
    # ================================================================
    pdf.section_title('Retrieval Phase: Descriptive Statistics')

    ret_rows = []
    for h in ['L', 'R', 'Bilateral']:
        sub = ret[ret['trial_hemisphere'] == h]
        n = len(sub)
        if n == 0:
            continue
        nr = int(sub['memory'].sum())
        ret_rows.append([h, n, nr, n - nr, f'{nr/n*100:.1f}%'])
    n_ret_tot = len(ret)
    nr_ret_tot = int(ret['memory'].sum())
    ret_rows.append(['Total', n_ret_tot, nr_ret_tot, n_ret_tot - nr_ret_tot,
                     f'{nr_ret_tot/n_ret_tot*100:.1f}%'])

    pdf.apa_table(
        'Table 12\nRetrieval Trial Distribution by IED Laterality (All Patients)',
        ['Hemisphere', 'N Trials', 'Rem', 'Forg', '% Rem'],
        ret_rows,
        col_widths=[25, 22, 18, 18, 22],
        note=f'N = {n_ret} trials from {n_ret_pats} patients.'
    )

    # ================================================================
    # RETRIEVAL MODEL
    # ================================================================
    pdf.section_title('Retrieval Phase: Mixed-Effects Model')

    pdf.body_text(
        f'Retrieval included {n_ret_pats} patients ({n_ret} trials). Only '
        f'{n_bi_ret} patients had bilateral coverage, contributing '
        f'{len(ret[ret["trial_hemisphere"] == "Bilateral"])} bilateral trials. '
        f'Retrieval has no trial-level stimulation condition.'
    )

    if m4 is not None:
        pdf.subsection_title('Model 4: IED Laterality (Retrieval, All Patients)')
        pdf.italic_text('memory ~ hemi_cat + (1 | patient_id)')

        label_map_4 = {
            '(Intercept)': '(Intercept)',
            'hemi_catBilateral': 'Bilateral (vs Left)',
            'hemi_catR': 'Right (vs Left)',
        }
        rows_4 = [format_or_row(r, label_map_4) for _, r in m4.iterrows()]

        pdf.apa_table(
            'Table 13\nModel 4: IED Laterality Predicting Recall (Retrieval, All Patients)',
            ['Predictor', 'OR', '95% CI', 'z', 'p'],
            rows_4,
            col_widths=[38, 16, 35, 16, 22],
            note=f'N = {n_ret} trials, {n_ret_pats} patients. Reference = Left IED.'
        )

        bi_ret = m4[m4['term'] == 'hemi_catBilateral']
        r_ret = m4[m4['term'] == 'hemi_catR']
        bi_ret_or = bi_ret['estimate'].values[0]
        bi_ret_p = bi_ret['p.value'].values[0]
        r_ret_or = r_ret['estimate'].values[0]
        r_ret_p = r_ret['p.value'].values[0]

        pdf.body_text(
            f'Neither bilateral IEDs (OR = {bi_ret_or:.2f}, p = {p_str(bi_ret_p)}) nor right '
            f'IEDs (OR = {r_ret_or:.2f}, p = {p_str(r_ret_p)}) differed significantly from '
            f'left IEDs in predicting recall. IED laterality during retrieval does not appear '
            f'to affect memory.'
        )

    # Retrieval forest plot
    forest_ret = os.path.join(OUTPUT_DIR, 'h3c_allsub_retrieval_forest.png')
    if os.path.exists(forest_ret):
        pdf.ln(2)
        pdf.italic_text(
            'Figure 2. Retrieval: IED laterality effects on recall '
            '(Model 4, all patients). Reference = Left IED.'
        )
        img_w = pdf.w - pdf.l_margin - pdf.r_margin - 10
        pdf.image(forest_ret, x=pdf.l_margin + 5, w=img_w)
        pdf.ln(3)

    # ================================================================
    # SUMMARY
    # ================================================================
    pdf.section_title('Summary')

    # Compute descriptive percentages
    pct_bi = enc[enc['trial_hemisphere'] == 'Bilateral']['memory'].mean() * 100
    pct_l = enc[enc['trial_hemisphere'] == 'L']['memory'].mean() * 100
    pct_r = enc[enc['trial_hemisphere'] == 'R']['memory'].mean() * 100

    pdf.body_text(
        f'Encoding: Across all {n_enc_pats} patients ({n_enc} trials), IED laterality '
        f'predicted memory performance. Bilateral IEDs were associated with the worst memory '
        f'({pct_bi:.1f}% remembered), compared to left ({pct_l:.1f}%) and right ({pct_r:.1f}%) '
        f'IEDs. In Model 1, bilateral IEDs significantly predicted reduced memory relative to '
        f'left IEDs (OR = {bi_or:.2f}, p = {p_str(bi_p)}), while right IEDs showed numerically '
        f'higher memory (OR = {r_or:.2f}, p = {p_str(r_p)}). Stimulation had no main effect '
        f'(p = {p_str(stim_p)}). Model 2 confirmed that neither laterality x stimulation '
        f'interaction was significant (LRT p = {lrt_p_str}), indicating that the bilateral IED '
        f'impairment occurs regardless of stimulation condition. Model 3, collapsing L and R '
        f'into a single unilateral reference, confirmed the bilateral effect '
        f'(OR = {bi3_or:.2f}, p = {p_str(bi3_p)}).'
    )

    if m4 is not None:
        pdf.body_text(
            f'Retrieval: IED laterality did not predict recall accuracy (Model 4: bilateral '
            f'p = {p_str(bi_ret_p)}, right p = {p_str(r_ret_p)}). This is consistent with '
            f'the hypothesis 3a finding that retrieval-phase IEDs do not disrupt memory.'
        )

    pdf.body_text(
        f'Timing: Among bilateral IED trials (Model 5), bilateral IEDs during the stimulation '
        f'window were the only significant timing predictor (OR = {stim_or:.2f}, '
        f'p = {p_str(stim_p)}). Because this window is only defined on stimulation trials, the '
        f'predictor is structurally confounded with stimulation condition, though stimulation '
        f'alone was non-significant across all models (ps > .88). These bilateral IEDs were '
        f'overwhelmingly hippocampal in origin ({n_hipp_both/n_bi*100:.0f}% of bilateral '
        f'trials had hippocampal IEDs in both hemispheres).'
    )

    pdf.body_text(
        'Interpretation: Bilateral IEDs during encoding significantly impair memory. '
        'By including all patients, the analysis has substantially more power than the '
        'bilateral-coverage-only analysis (802 vs. 328 trials). The bilateral vs. left '
        'contrast reaches significance in the main effects model (Model 1), and the bilateral '
        'vs. unilateral contrast (Model 3) confirms the finding. Simultaneous IED activity '
        'across both hemispheres is particularly disruptive to memory encoding. Region analysis '
        'confirms that bilateral IEDs are overwhelmingly hippocampal in origin, suggesting '
        'disruption of bilateral hippocampal encoding processes. Stimulation does not moderate '
        'the overall bilateral IED effect (Model 2), consistent with the hypothesis that '
        'bilateral IEDs impair memory "regardless of BLA stimulation." However, Model 5 '
        'reveals that when bilateral IEDs co-occur with active stimulation delivery, memory '
        'is most severely impaired, possibly reflecting interference with stimulation-engaged '
        'consolidation. The non-significant trend for right > left IEDs may reflect '
        'material-specific laterality effects (left-hemisphere IEDs disrupting verbal '
        'encoding), but requires further investigation.'
    )

    # Save
    out_path = os.path.join(OUTPUT_DIR, 'H3c_allsubjects_laterality_MLM_report.pdf')
    pdf.output(out_path)
    print(f'Saved: {out_path}')


if __name__ == '__main__':
    main()
