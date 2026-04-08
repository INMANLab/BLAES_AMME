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
    'stim': 'Stim Condition [cov]',
    'ied_before_image': 'IED Before-Image Window',
    'ied_during_image': 'IED During-Image Window',
    'ied_after_image': 'IED After-Image Window',
    'ied_during_stim': 'IED During-Stim Window',
    'n_regions_c': 'Region Spread (centered)',
    'n_channels_c': 'Channel Spread (centered)',
}
# Add interaction terms dynamically
for band in ['theta', 'slow_gamma', 'hfa']:
    bn = {'theta': 'Theta', 'slow_gamma': 'SG', 'hfa': 'HFA'}[band]
    for win in ['ied_before_image', 'ied_during_image', 'ied_after_image', 'ied_during_stim']:
        wn = {'ied_before_image': 'Before-Img',
              'ied_during_image': 'During-Img',
              'ied_after_image': 'After-Img',
              'ied_during_stim': 'During-Stim Win'}[win]
        TERM_LABELS[f'pow_{band}_z:{win}'] = f'{bn} Pow x IED {wn}'

# Display labels for regions
REGION_LABELS = {
    'BLA': 'Amygdala',
    'HPC': 'Hippocampus (all contacts)',
    'CA': 'CA subfield',
    'DG': 'DG subfield',
}
REGION_SHORT = {
    'BLA': 'Amygdala',
    'HPC': 'HPC',
    'CA': 'CA',
    'DG': 'DG',
}

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
        'brain regions (Amygdala, HPC, CA, DG) predicts subsequent memory on '
        'IED trials, and whether power interacts with IED timing windows. '
        'Power is baseline-corrected (post minus pre) and band-averaged into '
        'theta (4-8 Hz), slow gamma (30-55 Hz), and HFA (70-100 Hz).'
    )
    pdf.body_text(
        'Part 1: Power -> memory on all encoding trials (stimulation condition '
        'as covariate). Part 2: IED trials only - power + IED timing windows '
        '(main effects and interactions). Part 3: Power + IED spread. '
        'All models: GLMM, binomial, bobyqa, random intercept for patient.'
    )

    pdf.subsection_title('Brain Regions')
    pdf.body_text(
        'Amygdala: Contacts localized to the basolateral amygdala (BLA). '
        'The amygdala modulates memory consolidation through emotional '
        'arousal and is thought to gate hippocampal encoding via direct '
        'projections.'
    )
    pdf.body_text(
        'HPC (Hippocampus): All hippocampal contacts pooled. This provides '
        'the broadest hippocampal measure and the largest sample size per '
        'region.'
    )
    pdf.body_text(
        'CA (Cornu Ammonis): Hippocampal subfield involved in pattern '
        'completion -- retrieving a full memory from a partial cue. CA '
        'receives input from DG and projects to subiculum. CA contacts are '
        'a subset of HPC.'
    )
    pdf.body_text(
        'DG (Dentate Gyrus): Hippocampal subfield involved in pattern '
        'separation -- orthogonalizing similar inputs to create distinct '
        'memory traces. DG is the gateway for cortical input into the '
        'hippocampal circuit. DG contacts are a subset of HPC. Note: DG '
        'has the smallest sample (fewest patients with confirmed DG '
        'contacts), so these results should be interpreted with caution.'
    )
    pdf.body_text(
        'Comparing effects across HPC, CA, and DG allows us to test whether '
        'IED-related memory disruption is driven by a specific hippocampal '
        'computation (e.g., pattern separation in DG vs. pattern completion '
        'in CA) or reflects a general hippocampal vulnerability. If effects '
        'are similar across all three, they likely reflect broad hippocampal '
        'disruption rather than subfield-specific mechanisms.'
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
        rl = REGION_LABELS.get(reg, reg)
        rs = REGION_SHORT.get(reg, reg)

        pdf.subsection_title(f'{rl} ({len(sub):,} trials, {sub["patient_id"].nunique()} patients)')

        for b, bl in bands:
            df = load_or(f'pow_A_{reg}_{b}_odds_ratios.csv')
            if df is not None:
                rows = make_or_rows(df)
                pdf.apa_table(
                    f'Table {tbl_num}. {rs} {bl} (All Trials)',
                    OR_HEADERS, rows, OR_WIDTHS,
                    note=f'memory ~ pow_{b}_z + stim [covariate] + (1|patient_id). '
                         f'N = {len(sub):,}.'
                )
                tbl_num += 1

    # ═══ PART 2: IED TRIALS - TIMING WINDOWS ═══

    # Per-region interpretation context
    REGION_CONTEXT = {
        'BLA': (
            'The amygdala is not part of the hippocampal memory circuit per se, '
            'but modulates encoding via emotional arousal pathways. IED effects '
            'here test whether amygdala disruption during encoding impairs the '
            'modulatory signal that normally enhances hippocampal consolidation.'
        ),
        'HPC': (
            'HPC pools all hippocampal contacts, providing the broadest and '
            'most statistically powered test of hippocampal IED disruption. '
            'Effects here reflect general hippocampal vulnerability to IEDs '
            'during encoding, without distinguishing subfield contributions.'
        ),
        'CA': (
            'CA subfield contacts are a subset of HPC. CA is primarily '
            'associated with pattern completion -- retrieving stored memories '
            'from partial cues. During encoding, CA integrates input from DG '
            'and entorhinal cortex. If IED effects differ between CA and DG, '
            'this would suggest subfield-specific vulnerability. Similar '
            'effects to HPC suggest the disruption is not subfield-specific.'
        ),
        'DG': (
            'DG subfield contacts are a subset of HPC. DG performs pattern '
            'separation -- creating distinct, non-overlapping representations '
            'for similar inputs. This computation is critical at encoding for '
            'forming unique memory traces. Caution: DG has the smallest sample '
            '(fewest patients), so null results may reflect insufficient power '
            'rather than true absence of effects.'
        ),
    }

    for reg in regions:
        ied_sub = ied_merged[ied_merged['region'] == reg]
        pc = ied_sub.groupby('patient_id').size()
        kp = pc[pc >= 5].index
        ied_filt = ied_sub[ied_sub['patient_id'].isin(kp)]
        if len(ied_filt) < 20: continue
        rl = REGION_LABELS.get(reg, reg)
        rs = REGION_SHORT.get(reg, reg)

        pdf.add_page()
        pdf.section_title(f'Part 2: {rl} Power x IED Timing (IED Trials)')
        pdf.body_text(
            f'Sample: {len(ied_filt)} IED trials from '
            f'{ied_filt["patient_id"].nunique()} patients (>=5 trials each). '
            f'Memory rate: {ied_filt["memory"].mean()*100:.1f}%.'
        )
        if reg in REGION_CONTEXT:
            pdf.body_text(REGION_CONTEXT[reg])

        # Timing distribution
        tw_rows = []
        for col, label in [('ied_before_image', 'Before Image'),
                            ('ied_during_image', 'During Image'),
                            ('ied_after_image', 'After Image'),
                            ('ied_during_stim', 'During Stim')]:
            n = int(ied_filt[col].sum())
            tw_rows.append([label, str(n), f'{n/len(ied_filt)*100:.1f}%'])
        pdf.apa_table(
            f'Table {tbl_num}. {rs} IED Timing Window Distribution',
            ['Window', 'N', '%'], tw_rows, [50, 40, 40],
            note=f'N = {len(ied_filt)} IED trials. Windows not mutually exclusive.'
        )
        tbl_num += 1

        pdf.subsection_title('Main Effects: Power + IED Timing Windows')
        pdf.body_text(
            'These models test whether each IED timing window independently '
            'predicts memory, with power as a covariate (no interactions). '
            f'Model: memory ~ pow_band_z [covariate] + ied_before_image + '
            'ied_during_image + ied_after_image + ied_during_stim + '
            '(1|patient_id).'
        )
        for b, bl in bands:
            df = load_or(f'pow_T_{reg}_{b}_odds_ratios.csv')
            if df is not None:
                rows = make_or_rows(df)
                pdf.apa_table(
                    f'Table {tbl_num}. {rs} {bl} + IED Timing',
                    OR_HEADERS, rows, OR_WIDTHS,
                    note=f'N = {len(ied_filt)} IED trials, '
                         f'{ied_filt["patient_id"].nunique()} patients. '
                         f'Power is a covariate (main effect only).'
                )
                tbl_num += 1

        pdf.subsection_title('Interaction: Power x IED Timing Windows')
        pdf.body_text(
            'These models test whether the relationship between power and '
            'memory differs depending on IED timing. Each power x IED window '
            'interaction asks: does the effect of power on memory change when '
            'an IED occurred in that window? '
            f'Model: memory ~ pow_band_z * (ied_before_image + ied_during_image '
            '+ ied_after_image + ied_during_stim) + (1|patient_id). '
            'Stimulation condition is NOT in this model.'
        )
        for b, bl in bands:
            df = load_or(f'pow_TI_{reg}_{b}_odds_ratios.csv')
            if df is not None:
                rows = make_or_rows(df)
                pdf.apa_table(
                    f'Table {tbl_num}. {rs} {bl} x IED Timing Interactions',
                    OR_HEADERS, rows, OR_WIDTHS,
                    note=f'N = {len(ied_filt)} IED trials. '
                         'Interaction terms test whether power\'s effect on '
                         'memory is moderated by IED timing window presence.'
                )
                tbl_num += 1

        pdf.subsection_title('Power + IED Spread')
        pdf.body_text(
            'These models test whether the spatial extent of IEDs predicts '
            'memory, with power as a covariate. Region spread = number of '
            'distinct brain regions (e.g., amygdala, hippocampus) showing IED '
            'activity on that trial. Channel spread = number of channels '
            'within the current region showing IED activity (within-region '
            'spread). Model: memory ~ pow_band_z [covariate] + n_regions_c + '
            'n_channels_c + (1|patient_id).'
        )
        for b, bl in bands:
            df = load_or(f'pow_SP_{reg}_{b}_odds_ratios.csv')
            if df is not None:
                rows = make_or_rows(df)
                pdf.apa_table(
                    f'Table {tbl_num}. {rs} {bl} + IED Spread',
                    OR_HEADERS, rows, OR_WIDTHS,
                    note=f'N = {len(ied_filt)} IED trials. '
                         'Power is a covariate.'
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

    pdf.subsection_title('Key Patterns')
    pdf.body_text(
        '(1) IED timing window effects replicate across regions: IEDs '
        'occurring during the stimulation window and after image presentation '
        'consistently predict worse memory (OR ~ 0.57-0.67). IEDs before or '
        'during image presentation do not.'
    )
    pdf.body_text(
        '(2) Amygdala power x IED-during-stim-window interactions: higher '
        'theta and slow gamma power in the amygdala on trials with IEDs '
        'during the stimulation window was associated with worse memory. '
        'This interaction was specific to the amygdala -- HPC, CA, and DG '
        'showed no significant power x timing interactions.'
    )
    pdf.body_text(
        '(3) IED region spread (IEDs spreading across brain regions) was a '
        'significant negative predictor of memory in amygdala models '
        '(OR ~ 0.60, p ~ .03) but not in hippocampal models.'
    )

    pdf.subsection_title('Hippocampal Subfield Comparison: HPC vs CA vs DG')
    pdf.body_text(
        'IED-during-stim-window effects were remarkably consistent across '
        'HPC (OR ~ 0.59, p ~ .026) and CA (OR ~ 0.58, p ~ .030), with '
        'similar effect sizes and significance levels. DG showed the same '
        'direction (OR ~ 0.50) but did not reach significance (p ~ .095), '
        'likely due to the much smaller sample (173 trials, 6 patients vs. '
        '588-633 trials, 21-23 patients for CA/HPC).'
    )
    pdf.body_text(
        'The consistency between HPC and CA suggests that IED disruption '
        'during the stimulation window impairs general hippocampal encoding '
        'rather than targeting a specific subfield computation. CA results '
        'closely mirror pooled HPC results because CA contacts make up the '
        'majority of hippocampal contacts in this sample.'
    )
    pdf.body_text(
        'No power x IED timing interactions were significant in HPC, CA, or '
        'DG (all p > .10). The amygdala was the only region where neural '
        'excitability moderated the impact of IED timing on memory. This '
        'dissociation -- timing effects in hippocampus, power x timing '
        'interactions in amygdala -- suggests that the amygdala\'s state '
        'determines how much an IED disrupts the downstream hippocampal '
        'encoding process.'
    )
    pdf.body_text(
        'IED spread effects were absent in all hippocampal models (HPC, CA, '
        'DG), in contrast to the significant region-spread effect in amygdala '
        'models. This further supports the amygdala as the region most '
        'sensitive to the spatial scale of IED disruption.'
    )

    # ═══ MECHANISTIC INTERPRETATION ═══
    pdf.add_page()
    pdf.section_title('Mechanistic Interpretation: Stimulation Protocol Context')

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
        'amygdala. Examining the raw IED data reveals that the vast '
        'majority of during-stim IEDs originate in the hippocampus:'
    )

    # Load IED data to compute during-stim location stats
    ied_csv = os.path.join(SCRIPT_DIR, 'IED',
        'AMMEBLAES_IEDs_trial_level_dissertation_study_usethis_cleaned_with_memory.csv')
    try:
        ied_raw = pd.read_csv(ied_csv)
        ds_ieds = ied_raw[ied_raw['DuringStim'] == 'Y'].copy()
        n_ds = len(ds_ieds)

        def classify_region(r):
            r = str(r).lower()
            if 'amygdala' in r and 'hippocampus' in r:
                return 'Both Amygdala + Hippocampus'
            elif 'amygdala' in r:
                return 'Amygdala only'
            elif 'hippocampus' in r:
                return 'Hippocampus'
            else:
                return 'Other (cortical)'

        ds_ieds['location'] = ds_ieds['Region'].apply(classify_region)
        loc_counts = ds_ieds['location'].value_counts()

        loc_rows = []
        for loc in ['Hippocampus', 'Other (cortical)',
                     'Both Amygdala + Hippocampus', 'Amygdala only']:
            n = loc_counts.get(loc, 0)
            loc_rows.append([loc, str(n), f'{n/n_ds*100:.1f}%'])

        pdf.apa_table(
            f'Table {tbl_num}. Anatomical Origin of During-Stim Window IEDs',
            ['Location', 'N', '%'], loc_rows,
            col_widths=[60, 30, 30],
            note=f'N = {n_ds} IED detections occurring during the stimulation '
                 'window across all patients and trials. Location based on the '
                 'brain region where the IED was detected. A single trial may '
                 'have multiple IED detections across regions.'
        )
        tbl_num += 1
    except Exception:
        pdf.body_text(
            '(IED location data unavailable for table generation.)'
        )

    pdf.body_text(
        'Approximately 63% of during-stim IEDs are detected in the '
        'hippocampus, with only ~7% involving the amygdala. This means '
        'that on most trials flagged as having a during-stim IED, the '
        'IED is disrupting the hippocampus -- the downstream target of '
        'the amygdala stimulation -- not the amygdala itself.'
    )

    pdf.subsection_title(
        'Interpretation: Endogenous Amygdala Power x Hippocampal IED Interaction'
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
        'an IED simultaneously disrupts the hippocampus during the '
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

    out_path = os.path.join(OUTPUT_DIR, 'Power_x_IED_Effects_MLM.pdf')
    pdf.output(out_path)
    print(f'Report saved -> {out_path}')


if __name__ == '__main__':
    build_report()
