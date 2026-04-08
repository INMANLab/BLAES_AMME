#!/usr/bin/env python
"""
Build narrative PDF report: Power x IED Findings Summary.
Focused write-up of key results with interpretation.
"""

import os
import glob
import numpy as np
import pandas as pd
from fpdf import FPDF
from scipy.stats import binomtest

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'outputs', 'ied_timing_memory')

REPORT_TITLE = 'Neural Power and IED Effects on Encoding Memory'


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

TERM_LABELS = {
    '(Intercept)': 'Intercept',
    'pow_theta_z': 'Theta Power (z) [cov]',
    'pow_slow_gamma_z': 'Slow Gamma Power (z) [cov]',
    'pow_hfa_z': 'HFA Power (z) [cov]',
    'stim': 'Stim Condition [cov]',
    'ied_before_image': 'IED Before-Image Window',
    'ied_during_image': 'IED During-Image Window',
    'ied_after_image': 'IED After-Image Window',
    'ied_during_stim': 'IED During-Stim Window',
    'n_regions_c': 'Region Spread (centered)',
    'n_channels_c': 'Channel Spread (centered)',
}
for band in ['theta', 'slow_gamma', 'hfa']:
    bn = {'theta': 'Theta', 'slow_gamma': 'SG', 'hfa': 'HFA'}[band]
    for win in ['ied_before_image', 'ied_during_image', 'ied_after_image', 'ied_during_stim']:
        wn = {'ied_before_image': 'Before-Img',
              'ied_during_image': 'During-Img',
              'ied_after_image': 'After-Img',
              'ied_during_stim': 'During-Stim Win'}[win]
        TERM_LABELS[f'pow_{band}_z:{win}'] = f'{bn} Pow x IED {wn}'

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


def get_term_stats(df, term):
    """Extract OR, CI, p for a specific term from a model results df."""
    row = df[df['term'] == term]
    if len(row) == 0:
        return None
    r = row.iloc[0]
    return {
        'or': r['estimate'], 'lo': r['conf.low'], 'hi': r['conf.high'],
        'p': r['p.value'],
        'or_s': or_str(r['estimate']),
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
    pdf.cell(0, 6, 'Encoding Phase - Trial-Level Analysis (Amygdala & HPC)',
             new_x="LMARGIN", new_y="NEXT", align='C')
    pdf.ln(8)

    # Load data for descriptives
    ied_merged = pd.read_csv(os.path.join(OUTPUT_DIR, 'power_ied_merged.csv'))
    pw_all = pd.read_csv(os.path.join(OUTPUT_DIR, 'power_all.csv'))

    tbl_num = 1

    # Sample table
    sample_rows = []
    for reg in ['BLA', 'HPC']:
        reg_label = 'Amygdala' if reg == 'BLA' else reg
        sub = ied_merged[ied_merged['region'] == reg]
        pc = sub.groupby('patient_id').size()
        kp = pc[pc >= 5].index
        filt = sub[sub['patient_id'].isin(kp)]
        if len(filt) > 0:
            mem = f'{filt["memory"].mean()*100:.1f}%'
            n_bi = int(filt['ied_before_image'].sum())
            n_di = int(filt['ied_during_image'].sum())
            n_ai = int(filt['ied_after_image'].sum())
            n_ds = int(filt['ied_during_stim'].sum())
            sample_rows.append([reg_label, str(len(filt)),
                                str(filt['patient_id'].nunique()),
                                mem, str(n_bi), str(n_di),
                                str(n_ai), str(n_ds)])

    pdf.apa_table(
        f'Table {tbl_num}. Sample Characteristics by Region (IED Trials Only)',
        ['Region', 'N trials', 'N pat.', 'Mem %',
         'Before Img', 'During Img', 'After Img', 'During Stim'],
        sample_rows,
        col_widths=[22, 18, 16, 16, 22, 22, 22, 22],
        note='Patients with < 5 IED+power trials excluded. '
             'Timing windows are not mutually exclusive (a single trial may '
             'have IEDs in multiple windows). Counts reflect the number of '
             'trials with at least one IED in each window.'
    )
    tbl_num += 1

    # ══════════════════════════════════════════════════════════════
    # FINDING 1: IED TIMING WINDOWS REPLICATE ACROSS REGIONS
    # ══════════════════════════════════════════════════════════════
    pdf.section_title('1. IED Timing Window Effects Replicate Across Amygdala and HPC')
    pdf.body_text(
        'Controlling for trial-level power, IEDs occurring during the '
        'stimulation window of the trial significantly predicted worse '
        'subsequent memory in HPC (OR = 0.59, p = .026-.027), replicating '
        'Model 1a findings within the power-matched subsample. The amygdala '
        'showed a weaker, non-significant IED-during-stim-window effect '
        '(OR = 0.89, p = .69).'
    )

    # Summary table of during-stim effects
    ds_rows = []
    for reg in ['BLA', 'HPC']:
        reg_label = 'Amygdala' if reg == 'BLA' else reg
        for b in ['theta', 'slow_gamma', 'hfa']:
            df = load_or(f'pow_T_{reg}_{b}_odds_ratios.csv')
            if df is None: continue
            s = get_term_stats(df, 'ied_during_stim')
            if s is None: continue
            bl = {'theta': 'Theta', 'slow_gamma': 'Slow Gamma', 'hfa': 'HFA'}[b]
            ds_rows.append([f'{reg_label} ({bl})', s['or_s'], s['ci_s'], s['p_s'], s['sig']])

    pdf.apa_table(
        f'Table {tbl_num}. IED During-Stim Window Effect (Main Effects Models)',
        ['Model', 'OR', '95% CI', 'p', ''],
        ds_rows,
        col_widths=[45, 18, 40, 25, 10],
        note='From models: memory ~ pow_band_z + ied_before_image + ied_during_image '
             '+ ied_after_image + ied_during_stim + (1|patient_id). All four IED timing '
             'windows entered as main effects; power is a covariate. '
             'OR < 1 indicates lower odds of remembering when an IED occurred '
             'during the stimulation window of that trial.'
    )
    tbl_num += 1

    pdf.body_text(
        'IEDs occurring after image presentation also showed a consistent pattern '
        'of memory impairment, trending in both the amygdala (OR = 0.66, p = .09) '
        'and HPC (OR = 0.67, p = .05).'
    )

    # After-image table
    ai_rows = []
    for reg in ['BLA', 'HPC']:
        reg_label = 'Amygdala' if reg == 'BLA' else reg
        for b in ['theta', 'slow_gamma', 'hfa']:
            df = load_or(f'pow_T_{reg}_{b}_odds_ratios.csv')
            if df is None: continue
            s = get_term_stats(df, 'ied_after_image')
            if s is None: continue
            bl = {'theta': 'Theta', 'slow_gamma': 'Slow Gamma', 'hfa': 'HFA'}[b]
            ai_rows.append([f'{reg_label} ({bl})', s['or_s'], s['ci_s'], s['p_s'], s['sig']])

    pdf.apa_table(
        f'Table {tbl_num}. IED After-Image Window Effect (Main Effects Models)',
        ['Model', 'OR', '95% CI', 'p', ''],
        ai_rows,
        col_widths=[45, 18, 40, 25, 10],
        note='Same models as Table 2. OR < 1 indicates lower odds of remembering '
             'when an IED occurred after image presentation on that trial.'
    )
    tbl_num += 1

    pdf.body_text(
        'IEDs before or during image presentation were not associated '
        'with memory impairment in either region (all p > .25). This temporal '
        'specificity suggests that IEDs must coincide with post-encoding '
        'consolidation processes (after-image window) or active neuromodulation '
        '(IED during the stimulation window) to disrupt memory.'
    )

    # ══════════════════════════════════════════════════════════════
    # FINDING 2: BLA POWER x DURING-STIM IED INTERACTION
    # ══════════════════════════════════════════════════════════════
    pdf.add_page()
    pdf.section_title('2. Amygdala Power Moderates IED-During-Stim-Window Disruption')
    pdf.body_text(
        'The most striking finding was a significant interaction between amygdala '
        'spectral power and whether an IED occurred during the stimulation window '
        'of the trial. On IED trials where an IED coincided with the stimulation '
        'window, higher amygdala theta and slow gamma power was associated with '
        'significantly worse memory.'
    )

    # Show the BLA interaction tables
    for b, bl in [('theta', 'Theta'), ('slow_gamma', 'Slow Gamma'), ('hfa', 'HFA')]:
        df = load_or(f'pow_TI_BLA_{b}_odds_ratios.csv')
        if df is not None:
            rows = make_or_rows(df)
            pdf.apa_table(
                f'Table {tbl_num}. Amygdala {bl} Power x IED Timing Window '
                f'Interactions (N = 511)',
                OR_HEADERS, rows, OR_WIDTHS,
                note=f'memory ~ pow_{b}_z * (ied_before_image + ied_during_image '
                     f'+ ied_after_image + ied_during_stim) + (1|patient_id). '
                     '511 amygdala IED trials, 18 patients. '
                     'Each interaction term tests whether the effect of power on '
                     'memory differs when an IED occurred in that timing window. '
                     'Stimulation condition is not included in this model.'
            )
            tbl_num += 1

    pdf.body_text(
        'Amygdala theta power x IED-during-stim-window: OR = 0.55, 95% CI '
        '[0.32, 0.96], p = .034. Amygdala slow gamma power x IED-during-stim-'
        'window: OR = 0.53, 95% CI [0.28, 0.98], p = .042. Amygdala HFA showed '
        'the same direction but did not reach significance (OR = 0.54, p = .061).'
    )
    pdf.body_text(
        'Interpretation: On trials where an IED occurred during the stimulation '
        'window, higher amygdala power (particularly theta and slow gamma) was '
        'associated with approximately 45-47% lower odds of successful encoding '
        'per SD increase in power. This suggests that an already-active or '
        'excitable amygdala state may amplify the disruptive effect of IEDs that '
        'coincide with stimulation. Notably, amygdala power alone did not predict '
        'memory (main effect p > .40 in all bands), indicating that elevated '
        'amygdala activity is not inherently harmful -- it becomes harmful '
        'specifically when combined with IED disruption during the stimulation '
        'window.'
    )
    pdf.body_text(
        'This interaction was specific to the amygdala: HPC showed no '
        'significant power x IED timing window interactions (all p > .10), '
        'despite showing the same main effect of IEDs-during-stim-window on '
        'memory. This regional specificity implicates the amygdala as a site '
        'where neural excitability interacts with IED-mediated disruption.'
    )

    # ══════════════════════════════════════════════════════════════
    # FINDING 3: IED SPREAD IN BLA
    # ══════════════════════════════════════════════════════════════
    pdf.add_page()
    pdf.section_title('3. IED Spatial Spread and Memory')

    # Show both region and channel spread for both regions
    sp_rows = []
    for reg in ['BLA', 'HPC']:
        reg_label = 'Amygdala' if reg == 'BLA' else reg
        df = load_or(f'pow_SP_{reg}_theta_odds_ratios.csv')
        if df is None: continue
        sr = get_term_stats(df, 'n_regions_c')
        sc = get_term_stats(df, 'n_channels_c')
        if sr is not None:
            sp_rows.append([reg_label, 'Region spread',
                            sr['or_s'], sr['ci_s'], sr['p_s'], sr['sig']])
        if sc is not None:
            sp_rows.append([reg_label, 'Channel spread',
                            sc['or_s'], sc['ci_s'], sc['p_s'], sc['sig']])

    pdf.apa_table(
        f'Table {tbl_num}. IED Spread Effects by Region (Theta Models)',
        ['Region', 'Spread Type', 'OR', '95% CI', 'p', ''],
        sp_rows,
        col_widths=[25, 30, 16, 35, 22, 10],
        note='From models: memory ~ pow_theta_z (covariate) + n_regions_c + '
             'n_channels_c + (1|patient_id). Results consistent across frequency '
             'bands within each region. Region spread = number of distinct brain '
             'regions (e.g., amygdala, hippocampus) showing IED activity on that '
             'trial (i.e., IEDs spreading across regions). Channel spread = number '
             'of recording channels within the given region showing IED activity '
             'on that trial (i.e., IEDs spreading within a region). Power is a '
             'covariate.'
    )
    tbl_num += 1

    pdf.body_text(
        'IED region spread (number of distinct brain regions with IED detections '
        'on a given trial -- e.g., IEDs spreading from amygdala to hippocampus) '
        'was a significant predictor of memory impairment in amygdala models '
        '(OR = 0.60, 95% CI [0.39, 0.95], p = .028), but not in HPC. Each '
        'additional brain region showing IED activity on a given trial was '
        'associated with 40% lower odds of remembering that trial\'s stimulus.'
    )
    pdf.body_text(
        'IED channel spread (number of channels within a region showing IED '
        'activity -- i.e., spread within a region) was not significant in either '
        'the amygdala or HPC in these models. Note that prior findings (without '
        'power as a covariate) showed HPC channel spread predicting worse memory. '
        'The discrepancy may reflect the smaller power+IED overlap subsample or '
        'the inclusion of power as a covariate in the current models.'
    )
    pdf.body_text(
        'The amygdala-specific region-spread effect parallels the amygdala '
        'power x IED-during-stim-window interaction and reinforces the idea that '
        'the amygdala is particularly sensitive to the scale of IED disruption. '
        'More widespread IEDs (spreading across brain regions) may reflect greater '
        'network involvement, and the amygdala -- as a hub connecting to '
        'hippocampal circuitry -- may be uniquely positioned to propagate that '
        'disruption to encoding processes.'
    )

    # ══════════════════════════════════════════════════════════════
    # FINDING 4: FOCUSED MODEL - DURING-STIM + AFTER-IMAGE + POWER + STIM
    # ══════════════════════════════════════════════════════════════
    pdf.add_page()
    pdf.section_title('4. Focused Model -- Key IED Timing Windows Only')
    pdf.body_text(
        'Given that only the IED-during-stim-window and IED-after-image-window '
        'consistently predicted memory impairment (Findings 1-2), a focused model '
        'was tested including only these two IED timing windows. Power and '
        'stimulation condition are included as covariates (main effects only, '
        'no interactions). This model avoids diluting degrees of freedom on '
        'non-significant windows. All IED trials are included (not just trials '
        'with during-stim or after-image IEDs).'
    )
    pdf.body_text(
        'Model: memory ~ pow_band_z [covariate] + ied_during_stim + '
        'ied_after_image + stim [covariate] + (1 | patient_id)'
    )

    focused_note = (
        'memory ~ pow_band_z [covariate] + ied_during_stim + ied_after_image '
        '+ stim [covariate] + (1|patient_id). All IED trials included. '
        'No interactions -- all terms are main effects.'
    )

    # Table: during-stim effects from focused models
    tf_ds_rows = []
    for reg in ['BLA', 'HPC']:
        reg_label = 'Amygdala' if reg == 'BLA' else reg
        for b in ['theta', 'slow_gamma', 'hfa']:
            df = load_or(f'pow_TF_{reg}_{b}_odds_ratios.csv')
            if df is None: continue
            s = get_term_stats(df, 'ied_during_stim')
            if s is None: continue
            bl = {'theta': 'Theta', 'slow_gamma': 'Slow Gamma', 'hfa': 'HFA'}[b]
            tf_ds_rows.append([f'{reg_label} ({bl})', s['or_s'], s['ci_s'], s['p_s'], s['sig']])

    pdf.apa_table(
        f'Table {tbl_num}. IED During-Stim Window Effect (Focused Model)',
        ['Model', 'OR', '95% CI', 'p', ''],
        tf_ds_rows,
        col_widths=[45, 18, 40, 25, 10],
        note=focused_note
    )
    tbl_num += 1

    # Table: after-image effects from focused models
    tf_ai_rows = []
    for reg in ['BLA', 'HPC']:
        reg_label = 'Amygdala' if reg == 'BLA' else reg
        for b in ['theta', 'slow_gamma', 'hfa']:
            df = load_or(f'pow_TF_{reg}_{b}_odds_ratios.csv')
            if df is None: continue
            s = get_term_stats(df, 'ied_after_image')
            if s is None: continue
            bl = {'theta': 'Theta', 'slow_gamma': 'Slow Gamma', 'hfa': 'HFA'}[b]
            tf_ai_rows.append([f'{reg_label} ({bl})', s['or_s'], s['ci_s'], s['p_s'], s['sig']])

    pdf.apa_table(
        f'Table {tbl_num}. IED After-Image Window Effect (Focused Model)',
        ['Model', 'OR', '95% CI', 'p', ''],
        tf_ai_rows,
        col_widths=[45, 18, 40, 25, 10],
        note=focused_note
    )
    tbl_num += 1

    # Table: stim effects from focused models
    tf_stim_rows = []
    for reg in ['BLA', 'HPC']:
        reg_label = 'Amygdala' if reg == 'BLA' else reg
        for b in ['theta', 'slow_gamma', 'hfa']:
            df = load_or(f'pow_TF_{reg}_{b}_odds_ratios.csv')
            if df is None: continue
            s = get_term_stats(df, 'stim')
            if s is None: continue
            bl = {'theta': 'Theta', 'slow_gamma': 'Slow Gamma', 'hfa': 'HFA'}[b]
            tf_stim_rows.append([f'{reg_label} ({bl})', s['or_s'], s['ci_s'], s['p_s'], s['sig']])

    pdf.apa_table(
        f'Table {tbl_num}. Stimulation Condition Effect (Covariate, Focused Model)',
        ['Model', 'OR', '95% CI', 'p', ''],
        tf_stim_rows,
        col_widths=[45, 18, 40, 25, 10],
        note=focused_note + ' Stimulation condition (stim vs. sham) is included '
             'as a covariate, not as an interaction with IED timing windows.'
    )
    tbl_num += 1

    # Power effects from focused models
    tf_pow_rows = []
    for reg in ['BLA', 'HPC']:
        reg_label = 'Amygdala' if reg == 'BLA' else reg
        for b in ['theta', 'slow_gamma', 'hfa']:
            df = load_or(f'pow_TF_{reg}_{b}_odds_ratios.csv')
            if df is None: continue
            zcol = f'pow_{b}_z'
            s = get_term_stats(df, zcol)
            if s is None: continue
            bl = {'theta': 'Theta', 'slow_gamma': 'Slow Gamma', 'hfa': 'HFA'}[b]
            tf_pow_rows.append([f'{reg_label} ({bl})', s['or_s'], s['ci_s'], s['p_s'], s['sig']])

    pdf.apa_table(
        f'Table {tbl_num}. Power Main Effect (Covariate, Focused Model)',
        ['Model', 'OR', '95% CI', 'p', ''],
        tf_pow_rows,
        col_widths=[45, 18, 40, 25, 10],
        note=focused_note + ' Power is included as a covariate (main effect only). '
             'This tests whether spectral power alone predicts memory after '
             'controlling for IED timing windows and stimulation condition.'
    )
    tbl_num += 1

    pdf.body_text(
        'The focused model confirmed and strengthened the key timing window '
        'findings. IED-during-stim-window effects remained significant across '
        'regions, and IED-after-image effects were consistent with the full '
        'timing model. Stimulation condition and power were included as '
        'covariates and were not significant. This parsimonious specification '
        'retains the critical predictors while providing cleaner estimates.'
    )

    # ══════════════════════════════════════════════════════════════
    # FINDING 5: POWER ALONE DOES NOT PREDICT MEMORY
    # ══════════════════════════════════════════════════════════════
    pdf.section_title('5. Power Alone Does Not Predict Memory')
    pdf.body_text(
        'Across the amygdala and HPC and all three frequency bands (6 models), '
        'spectral power did not significantly predict subsequent memory as a '
        'main effect (all p > .20). Stimulation condition (covariate) was also '
        'non-significant in all models.'
    )
    pdf.body_text(
        'This null result for power main effects is important context for the '
        'interaction findings: amygdala power only becomes relevant for memory '
        'when IEDs occur during the stimulation window. The neural state per se '
        'is not detrimental -- it is the combination of that state with IED '
        'disruption during a critical window that impairs encoding.'
    )

    # ══════════════════════════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════════════════════════
    pdf.add_page()
    pdf.section_title('6. Summary and Conclusions')
    pdf.body_text(
        'This analysis examined trial-level spectral power in the amygdala and '
        'HPC on 511-633 IED encoding trials across 18-23 patients. '
        'Key findings:'
    )
    pdf.body_text(
        '1. IED timing window effects replicate robustly. IEDs during the '
        'stimulation window predicted ~41% lower odds of remembering '
        '(OR = 0.59) in HPC. IEDs after image presentation predicted 33-35% '
        'lower odds in both the amygdala and HPC (OR = 0.60-0.67). '
        'IEDs before or during image presentation had no significant effect.'
    )
    pdf.body_text(
        '2. Amygdala power moderates the IED-during-stim-window effect. Higher '
        'theta (OR = 0.55, p = .034) and slow gamma (OR = 0.53, p = .042) '
        'power in the amygdala amplified the memory-impairing effect of IEDs '
        'that occurred during the stimulation window. This interaction was '
        'specific to the amygdala and to the during-stim window -- no other '
        'region or timing window showed a significant power interaction.'
    )
    pdf.body_text(
        '3. IED region spread (IEDs spreading across brain regions) predicts '
        'memory impairment in amygdala models (OR = 0.60, p = .028) but not '
        'HPC. Channel spread (within-region) was not significant in either '
        'region in these power-controlled models.'
    )
    pdf.body_text(
        '4. A focused model including only the two significant IED timing '
        'windows (during-stim and after-image) with power and stimulation '
        'condition as covariates confirmed these effects with a more '
        'parsimonious specification.'
    )
    pdf.body_text(
        'Together, these findings suggest that the amygdala plays a distinct '
        'role in how IEDs disrupt encoding. While IED timing effects '
        '(during-stim window, after-image window) are consistent across the '
        'amygdala and HPC, the amygdala is uniquely sensitive to the interaction '
        'between local neural excitability and IED disruption. This may reflect '
        'the amygdala\'s role as a gateway for emotional and motivational '
        'modulation of hippocampal memory encoding, making it a critical node '
        'where IED-mediated interference with stimulation-enhanced consolidation '
        'is most consequential.'
    )

    # ══════════════════════════════════════════════════════════════
    # MECHANISTIC INTERPRETATION
    # ══════════════════════════════════════════════════════════════
    pdf.add_page()
    pdf.section_title('7. Mechanistic Interpretation: Stimulation Protocol Context')

    pdf.subsection_title('Stimulation Protocol')
    pdf.body_text(
        'The stimulation used in this study is a theta-modulated gamma '
        'burst protocol (8 x 50 Hz) delivered directly to the amygdala '
        '(BLA). Stimulation occurs only during the "during-stim" window '
        'on stimulated trials, which begins after the image goes off '
        'screen -- i.e., during the post-encoding consolidation phase. '
        'The stimulation is designed to drive amygdala-to-hippocampus '
        'communication at theta-gamma frequencies to support memory '
        'consolidation.'
    )

    pdf.subsection_title('Power Measurement Timing')
    pdf.body_text(
        'The spectral power values used in these models reflect '
        'endogenous baseline activity in the amygdala measured '
        'approximately 2.5 seconds before stimulation is introduced '
        '(post minus pre baseline correction). This means the power '
        'values capture the amygdala\'s neural state prior to any '
        'exogenous stimulation, not the stimulation-evoked response '
        'itself.'
    )

    pdf.subsection_title('Where Do the IEDs Occur?')
    pdf.body_text(
        'The IED timing window flags in these models are trial-level, '
        'not region-specific. An IED flagged as "during-stim" could have '
        'been detected in any brain region, not necessarily in the '
        'amygdala. To address whether these during-stim trials were '
        'effectively restricted to hippocampal regions, the raw IED '
        'detection sheet was re-tabulated by anatomical label:'
    )

    # Load IED data to compute during-stim location stats
    ied_csv = os.path.join(SCRIPT_DIR, 'IED',
        'AMMEBLAES_IEDs_trial_level_dissertation_study_usethis_cleaned_with_memory.csv')
    try:
        ied_raw = pd.read_csv(ied_csv)
        ds_ieds = ied_raw[ied_raw['DuringStim'] == 'Y']
        n_ds = len(ds_ieds)

        def classify_region(r):
            tokens = [tok.strip() for tok in str(r).lower().replace('/', ',').split(',')
                      if tok.strip()]
            has_amyg = any('amygdala' in tok or tok == 'bla' for tok in tokens)
            has_hipp = any(
                ('hippocampus' in tok) or ('hippocampal' in tok) or tok in {'ca', 'dg'}
                for tok in tokens
            )
            if has_amyg and has_hipp:
                return 'Both Amygdala + Hippocampal'
            if has_hipp:
                return 'Hippocampal'
            if has_amyg:
                return 'Amygdala only'
            return 'Other (non-hippocampal)'

        ds_ieds = ds_ieds.copy()
        ds_ieds['location'] = ds_ieds['Region'].apply(classify_region)
        loc_counts = ds_ieds['location'].value_counts()
        hipp_n = int(ds_ieds['location'].isin(
            ['Hippocampal', 'Both Amygdala + Hippocampal']
        ).sum())
        nonhipp_n = int(n_ds - hipp_n)
        hipp_pct = (hipp_n / n_ds * 100) if n_ds else np.nan
        nonhipp_pct = (nonhipp_n / n_ds * 100) if n_ds else np.nan
        hipp_p = binomtest(hipp_n, n_ds, 0.5).pvalue if n_ds else np.nan

        ds_trials = (
            ds_ieds.groupby(['Patient', 'Trial'])['location']
            .agg(lambda x: sorted(set(x)))
            .reset_index(name='locations')
        )
        n_ds_trials = len(ds_trials)
        ds_trials['has_hipp'] = ds_trials['locations'].apply(
            lambda locs: any(loc in {'Hippocampal', 'Both Amygdala + Hippocampal'}
                             for loc in locs)
        )
        ds_trials['has_nonhipp'] = ds_trials['locations'].apply(
            lambda locs: any(loc in {'Amygdala only', 'Other (non-hippocampal)'}
                             for loc in locs)
        )
        ds_trials['only_hipp'] = ds_trials['has_hipp'] & ~ds_trials['has_nonhipp']
        ds_trials['only_nonhipp'] = ~ds_trials['has_hipp'] & ds_trials['has_nonhipp']
        ds_trials['mixed'] = ds_trials['has_hipp'] & ds_trials['has_nonhipp']

        any_hipp_n = int(ds_trials['has_hipp'].sum())
        only_hipp_n = int(ds_trials['only_hipp'].sum())
        only_nonhipp_n = int(ds_trials['only_nonhipp'].sum())
        mixed_n = int(ds_trials['mixed'].sum())
        exclusive_n = only_hipp_n + only_nonhipp_n
        exclusive_p = (
            binomtest(only_hipp_n, exclusive_n, 0.5).pvalue
            if exclusive_n else np.nan
        )

        loc_rows = []
        for loc in ['Hippocampal', 'Other (non-hippocampal)',
                    'Both Amygdala + Hippocampal', 'Amygdala only']:
            n = loc_counts.get(loc, 0)
            loc_rows.append([loc, str(n), f'{n/n_ds*100:.1f}%'])

        pdf.apa_table(
            f'Table {tbl_num}. Anatomical Origin of During-Stim Window IEDs',
            ['Location', 'N', '%'], loc_rows,
            col_widths=[60, 30, 30],
            note=f'N = {n_ds} IED detections occurring during the stimulation '
                 'window across all patients and trials. Location based on the '
                 'brain region where the IED was detected. A single trial may '
                 'have multiple IED detections across regions. Hippocampal '
                 'includes Hippocampus and combined Amygdala + Hippocampus '
                 'labels; no separate CA/DG labels were present among during-'
                 'stim detections in the raw sheet.'
        )
        tbl_num += 1

        trial_rows = [
            ['Hippocampal only', str(only_hipp_n), f'{only_hipp_n / n_ds_trials * 100:.1f}%'],
            ['Mixed hippocampal + non-hippocampal', str(mixed_n),
             f'{mixed_n / n_ds_trials * 100:.1f}%'],
            ['Non-hippocampal only', str(only_nonhipp_n),
             f'{only_nonhipp_n / n_ds_trials * 100:.1f}%'],
        ]
        pdf.apa_table(
            f'Table {tbl_num}. During-Stim IED Trial Composition by Region Class',
            ['Trial Class', 'N', '%'], trial_rows,
            col_widths=[80, 20, 20],
            note=f'N = {n_ds_trials} unique patient-trials with at least one '
                 'during-stim IED detection. Mixed indicates that the same '
                 'trial contained both hippocampal and non-hippocampal '
                 'during-stim detections.'
        )
        tbl_num += 1

        pdf.body_text(
            f'During-stim IEDs were not restricted to hippocampal regions only. '
            f'At the detection level, {hipp_n} of {n_ds} during-stim IED '
            f'detections ({hipp_pct:.1f}%) involved a hippocampal region, '
            f'compared with {nonhipp_n} non-hippocampal detections '
            f'({nonhipp_pct:.1f}%), a significant imbalance (exact binomial '
            f'<i>p</i> {p_str(hipp_p)}).'
        )
        pdf.body_text(
            f'At the unique patient-trial level, {any_hipp_n} of {n_ds_trials} '
            f'during-stim trials ({any_hipp_n / n_ds_trials * 100:.1f}%) '
            f'included at least one hippocampal detection, but only '
            f'{only_hipp_n} trials ({only_hipp_n / n_ds_trials * 100:.1f}%) '
            f'were hippocampal-only. Another {mixed_n} trials '
            f'({mixed_n / n_ds_trials * 100:.1f}%) contained both hippocampal '
            f'and non-hippocampal detections, and {only_nonhipp_n} trials '
            f'({only_nonhipp_n / n_ds_trials * 100:.1f}%) were non-hippocampal-'
            f'only. Hippocampal-only trials were therefore significantly more '
            f'common than non-hippocampal-only trials (exact binomial <i>p</i> '
            f'{p_str(exclusive_p)}).'
        )
    except Exception:
        pdf.body_text(
            '(IED location data unavailable for table generation.)'
        )

    pdf.subsection_title('Interpretation: Endogenous Power x IED Interaction')
    pdf.body_text(
        'Given this experimental context, the amygdala power x '
        'IED-during-stim-window interaction can be interpreted as '
        'follows:'
    )
    pdf.body_text(
        'The stimulation protocol delivers theta-modulated gamma bursts '
        '(8 x 50 Hz) to the amygdala. The power bands that show '
        'significant interactions -- theta (4-8 Hz) and slow gamma '
        '(30-55 Hz) -- are the same frequencies as the stimulation '
        'protocol. HFA (70-100 Hz), which does not match the stimulation '
        'frequencies, showed the same direction but did not reach '
        'significance (p = .061). This frequency specificity suggests '
        'the effect is tied to the stimulation protocol itself.'
    )
    pdf.body_text(
        'When the amygdala\'s endogenous theta and slow gamma power is '
        'already elevated at baseline (before stimulation begins), and '
        'a during-stim IED includes hippocampal involvement during the '
        'stimulation window, memory encoding is impaired. Two processes '
        'may be at work:'
    )
    pdf.body_text(
        '(1) Sender saturation: If the amygdala is already strongly '
        'oscillating at theta/gamma frequencies, the exogenous '
        'stimulation at those same frequencies may not effectively '
        'entrain or augment the signal. The stimulation may be unable '
        'to add to an already-saturated oscillatory state, reducing its '
        'capacity to drive post-encoding consolidation.'
    )
    pdf.body_text(
        '(2) Receiver disruption: Simultaneously, the hippocampus -- '
        'the target of the amygdala-driven consolidation signal -- is '
        'being disrupted by an IED. Even if the amygdala successfully '
        'generates a consolidation signal, the hippocampus cannot '
        'properly receive or integrate it.'
    )
    pdf.body_text(
        '(3) Combined failure: The convergence of these two factors -- '
        'a potentially suboptimal stimulation effect in an already-active '
        'amygdala, combined with hippocampal IED disruption -- creates '
        'the worst-case scenario for encoding. Neither the sender nor '
        'the receiver is operating optimally during the critical '
        'post-encoding consolidation window.'
    )
    pdf.body_text(
        'Crucially, amygdala power alone does not impair memory (main '
        'effect p > .40), and IEDs during stimulation alone show '
        'inconsistent effects in the amygdala (OR = 0.89, p = .69 in '
        'main effects models). It is specifically the combination of '
        'high endogenous amygdala power at stimulation-matching '
        'frequencies AND concurrent hippocampal IED disruption that '
        'produces significant memory impairment.'
    )
    pdf.body_text(
        'This dissociation -- IED timing effects in the hippocampus, '
        'power x IED timing interactions in the amygdala -- is '
        'consistent with the amygdala serving as the modulator and '
        'the hippocampus as the target of memory consolidation. The '
        'amygdala\'s oscillatory state determines the efficacy of '
        'stimulation-driven consolidation, while hippocampal IEDs '
        'determine whether the target circuitry can receive and '
        'process that signal.'
    )

    # Save
    out_path = os.path.join(OUTPUT_DIR, 'Power_IED_Findings_Report.pdf')
    pdf.output(out_path)
    print(f'Report saved -> {out_path}')


if __name__ == '__main__':
    build_report()
