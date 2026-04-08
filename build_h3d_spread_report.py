#!/usr/bin/env python
"""
Build PDF report for Hypothesis 3d — IED Spread, Neural Dynamics, and Memory.
BLA-HPC only.  Four spread metrics x 2 bands (coherence) and x 2 PAC metrics.

Outputs: outputs/ied_timing_memory/H3d_Spread_Neural_MLM_Report.pdf
"""

import os
import numpy as np
import pandas as pd
from fpdf import FPDF

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'outputs', 'ied_timing_memory')

SPREAD_KEYS = [
    ('channel',   'Channel Spread'),
    ('region',    'Region Spread'),
    ('crossreg',  'Cross-Region'),
    ('withinreg', 'Within-Region Spread'),
]
BANDS = [('theta', 'Theta'), ('slow_gamma', 'Slow Gamma')]
PAC_METRICS = [('PACint', 'PAC Stim x Memory'), ('PACrem', 'PAC Stim on Rem')]


class APAReport(FPDF):
    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=25)

    def header(self):
        self.set_font('Helvetica', 'I', 8)
        self.cell(0, 5, 'Hypothesis 3d: IED Spread, Neural Dynamics, and Memory', align='R')
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


def fmt_or(val):
    if pd.isna(val) or abs(val) > 1e6 or abs(val) < 1e-6:
        return 'n.e.'
    return f'{val:.2f}'

def fmt_p(val):
    if pd.isna(val):
        return ''
    return '<.001' if val < .001 else f'{val:.3f}'


def add_model_table(pdf, csv_path, model_label):
    if not os.path.exists(csv_path):
        pdf.body_text(f'  {model_label}: output file not found.')
        return

    df = pd.read_csv(csv_path)
    pdf.subsection_title(model_label)

    col_widths = [60, 20, 20, 20, 20, 20, 22]
    headers = ['Term', 'OR', 'CI Low', 'CI High', 'SE', 'z', 'p']
    pdf.set_font('Helvetica', 'B', 8)
    for w, h in zip(col_widths, headers):
        pdf.cell(w, 6, h, border=1, align='C')
    pdf.ln()

    pdf.set_font('Times', '', 8)
    for _, row in df.iterrows():
        term = str(row.get('term', ''))
        term = (term.replace('pat_channel_spread_z', 'ChanSpread')
                     .replace('pat_region_spread_z', 'RegSpread')
                     .replace('pat_cross_region_z', 'CrossReg')
                     .replace('pat_within_region_z', 'WithinReg')
                     .replace('channel_spread_c', 'ChanSpread(c)')
                     .replace('region_spread_c', 'RegSpread(c)')
                     .replace('within_region_c', 'WithinReg(c)')
                     .replace('cross_region', 'CrossReg')
                     .replace('coh_theta_z', 'CohTheta')
                     .replace('coh_slow_gamma_z', 'CohSlowG')
                     .replace('pac_interaction_z', 'PACint')
                     .replace('pac_stim_rem_z', 'PACrem')
                     .replace('stim:', 'Stim:')
                     .replace('(Intercept)', 'Intercept'))
        if len(term) > 35:
            term = term[:34] + '.'

        p_val = row.get('p.value', np.nan)
        if pd.notna(p_val) and p_val < 0.05 and 'Intercept' not in term:
            pdf.set_font('Times', 'B', 8)
        else:
            pdf.set_font('Times', '', 8)

        vals = [
            term,
            fmt_or(row.get('estimate', np.nan)),
            fmt_or(row.get('conf.low', np.nan)),
            fmt_or(row.get('conf.high', np.nan)),
            f"{row.get('std.error', np.nan):.3f}" if pd.notna(row.get('std.error')) else '',
            f"{row.get('statistic', np.nan):.2f}" if pd.notna(row.get('statistic')) else '',
            fmt_p(p_val),
        ]
        for w, v in zip(col_widths, vals):
            pdf.cell(w, 5.5, v, border=1, align='C')
        pdf.ln()
    pdf.set_font('Times', '', 8)
    pdf.ln(2)


def scan_notable(csv_paths_labels, threshold=0.10):
    notable = []
    for csv_path, label in csv_paths_labels:
        if not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            term = str(row.get('term', ''))
            p = row.get('p.value', 1)
            if '(Intercept)' in term or pd.isna(p):
                continue
            if p < threshold:
                clean_term = (term.replace('pat_channel_spread_z', 'ChanSpread')
                              .replace('pat_region_spread_z', 'RegSpread')
                              .replace('pat_cross_region_z', 'CrossReg')
                              .replace('pat_within_region_z', 'WithinReg')
                              .replace('coh_theta_z', 'CohTheta')
                              .replace('coh_slow_gamma_z', 'CohSlowG')
                              .replace('pac_interaction_z', 'PACint')
                              .replace('pac_stim_rem_z', 'PACrem'))
                star = '*' if p < .05 else '(m)'
                notable.append(
                    f"{label}: {clean_term} OR={fmt_or(row['estimate'])}, "
                    f"p={fmt_p(p)} {star}"
                )
    return notable


def build_report():
    pdf = APAReport()
    pdf.add_page()

    # ══════════════ TITLE ══════════════
    pdf.set_font('Helvetica', 'B', 16)
    pdf.cell(0, 10, 'Hypothesis 3d: IED Spread, Neural Dynamics,', align='C',
             new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 10, 'and Memory', align='C', new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    # ══════════════ RATIONALE ══════════════
    pdf.section_title('Rationale')
    pdf.body_text(
        'Hypothesis 3d posits that IEDs with larger spatial spread during '
        'encoding cause greater disruption of BLA-hippocampal neural dynamics, '
        'leading to more extreme memory modulation. This analysis tests:'
    )
    pdf.body_text(
        '(1) IED spread -> memory: Do 4 spread metrics (channel, region, '
        'cross-region, within-region) predict trial-level memory?\n'
        '(2) Spread x Stim x BLA-HPC coherence -> memory: Three-way '
        'interactions in theta and slow gamma.\n'
        '(3) Spread x Stim x BLA-HPC PAC -> memory: Three-way interactions '
        'with slow gamma PAC metrics.'
    )
    pdf.body_text(
        'All models use GLMM (binary memory DV, random intercepts for patient). '
        'Spread metrics: channel spread (mean contacts/trial), region spread '
        '(mean regions/trial), cross-region (proportion of cross-region trials), '
        'within-region spread (mean leads/trial). Coherence is trial-level, '
        'z-scored. PAC is condition-level (patient moderator), z-scored. '
        'Neural analyses restricted to subjects with IED spread and BLA-HPC data.'
    )

    # ══════════════ PART 1: DESCRIPTIVES ══════════════
    pdf.section_title('Part 1: IED Spread Descriptives')

    spread = pd.read_csv(os.path.join(OUTPUT_DIR, 'h3d_spread_trials.csv'))
    desc = pd.read_csv(os.path.join(OUTPUT_DIR, 'h3d_spread_descriptives.csv'))

    n_cross = spread['cross_region'].sum()
    pdf.body_text(
        f'{len(spread)} encoding IED trials across {spread["patient_id"].nunique()} patients. '
        f'Mean: {spread["n_leads"].mean():.1f} leads, '
        f'{spread["n_channels"].mean():.1f} contacts, '
        f'{spread["n_regions"].mean():.2f} regions per trial. '
        f'{n_cross} trials ({n_cross/len(spread)*100:.1f}%) showed cross-region spread. '
        f'{spread["has_bla"].sum()} ({spread["has_bla"].mean()*100:.1f}%) involved BLA, '
        f'{spread["has_hpc"].sum()} ({spread["has_hpc"].mean()*100:.1f}%) involved HPC.'
    )

    # Per-patient table
    pdf.subsection_title('Table 1: Per-Patient IED Spread Profile')
    cols = [40, 18, 18, 18, 18, 22, 18, 18, 18]
    hdrs = ['Patient', 'N Trials', 'Mean\nLeads', 'Max\nLeads',
            'Mean\nChan', 'Mean\nRegions', 'Max\nRegions', '% Cross\nRegion', '% BLA']
    pdf.set_font('Helvetica', 'B', 7)
    for w, h in zip(cols, hdrs):
        pdf.cell(w, 10, h, border=1, align='C')
    pdf.ln()
    pdf.set_font('Times', '', 7)
    for _, r in desc.iterrows():
        vals = [str(r['patient_id']), str(int(r['n_trials'])),
                f"{r['mean_leads']:.1f}", str(int(r['max_leads'])),
                f"{r['mean_channels']:.1f}", f"{r['mean_regions']:.1f}",
                str(int(r['max_regions'])), f"{r['pct_cross_region']:.0f}%",
                f"{r['pct_bla']:.0f}%"]
        for w, v in zip(cols, vals):
            pdf.cell(w, 5, v, border=1, align='C')
        pdf.ln()
    pdf.ln(3)

    for fig, title in [('h3d_spread_distribution.png', 'Figure 1: IED Spread Distributions'),
                        ('h3d_spread_by_memory.png', 'Figure 2: IED Spread by Memory Outcome'),
                        ('h3d_spread_region_memory.png', 'Figure 3: Memory by IED Region Involvement')]:
        fig_path = os.path.join(OUTPUT_DIR, fig)
        if os.path.exists(fig_path):
            pdf.subsection_title(title)
            pdf.image(fig_path, x=10, w=190)
            pdf.ln(3)

    # ══════════════ PART 2: SPREAD MLM ══════════════
    pdf.add_page()
    pdf.section_title('Part 2: IED Spread and Memory (GLMM)')
    pdf.body_text(
        'Four models test each spread metric x stim interaction on memory. '
        'Continuous predictors centered at the trial level.'
    )

    for suf, label in [('S1_channel',      'Channel Spread x Stim'),
                        ('S2_region',       'Region Spread x Stim'),
                        ('S3_crossregion',  'Cross-Region IED x Stim'),
                        ('S4_withinregion', 'Within-Region Spread x Stim')]:
        add_model_table(pdf, os.path.join(OUTPUT_DIR, f'h3d_model{suf}_odds_ratios.csv'),
                        f'Model {suf}: {label}')

    fig_path = os.path.join(OUTPUT_DIR, 'h3d_spread_forest.png')
    if os.path.exists(fig_path):
        pdf.subsection_title('Figure 4: Spread Predictors Forest Plot')
        pdf.image(fig_path, x=10, w=180)
        pdf.ln(3)

    # ══════════════ PART 3: THREE-WAY COHERENCE (BLA-HPC) ══════════════
    pdf.add_page()
    pdf.section_title('Part 3: Spread x Stim x BLA-HPC Coherence')

    elig = pd.read_csv(os.path.join(OUTPUT_DIR, 'h3d_eligible_subjects.csv'))
    n_coh = elig['has_coherence'].astype(str).str.lower().eq('true').sum()

    coh_data = pd.read_csv(os.path.join(OUTPUT_DIR, 'h3d_coherence_bla_trials.csv'))
    elig_coh_pats = elig[elig['has_coherence'].astype(str).str.lower() == 'true']['patient_id'].tolist()
    coh_hpc = coh_data[(coh_data['patient_id'].isin(elig_coh_pats)) &
                        (coh_data['region_pair'] == 'BLA_HPC')]

    pdf.body_text(
        f'Three-way models test whether patient-level IED spread (z-scored) '
        f'interacts with stimulation and trial-level BLA-HPC coherence to '
        f'predict memory. {n_coh} eligible patients, '
        f'{len(coh_hpc)} BLA-HPC trials from {coh_hpc["patient_id"].nunique()} patients. '
        f'Theta (4-8 Hz) and slow gamma (30-55 Hz) bands tested. '
        f'Model: memory ~ spread_z * stim * coherence_z + (1|patient_id).'
    )

    coh_csv_labels = []
    for band_key, band_label in BANDS:
        pdf.subsection_title(f'BLA-HPC {band_label} Coherence')
        for sp_key, sp_label in SPREAD_KEYS:
            csv = os.path.join(OUTPUT_DIR,
                f'h3d_coh_BLAHPC_{band_key}_{sp_key}_odds_ratios.csv')
            label = f'BLA-HPC {band_label} + {sp_label}'
            add_model_table(pdf, csv, f'{sp_label} x Stim x {band_label} Coherence')
            coh_csv_labels.append((csv, label))

    pdf.subsection_title('Coherence Results Summary')
    notable = scan_notable(coh_csv_labels)
    if notable:
        pdf.body_text('Notable effects (p < .10):')
        for n in notable:
            pdf.body_text(f'  {n}')
    else:
        pdf.body_text('No notable effects found.')

    fig_path = os.path.join(OUTPUT_DIR, 'h3d_coherence_forest.png')
    if os.path.exists(fig_path):
        pdf.subsection_title('Figure 5: Coherence Three-Way Interactions Forest Plot')
        pdf.image(fig_path, x=10, w=190)
        pdf.ln(3)

    # ══════════════ PART 4: THREE-WAY PAC (BLA-HPC) ══════════════
    pdf.add_page()
    pdf.section_title('Part 4: Spread x Stim x BLA-HPC PAC')

    n_pac = elig['has_pac'].astype(str).str.lower().eq('true').sum()
    pdf.body_text(
        f'Three-way models test whether IED spread interacts with stimulation '
        f'and patient-level BLA-HPC slow gamma PAC to predict memory. '
        f'{n_pac} patients with IED spread and PAC data. '
        f'PAC is condition-level (patient moderator), z-scored. '
        f'Two PAC metrics: (1) PAC Interaction = stim x memory interaction, '
        f'(2) PAC Stim on Remembered = stim effect on PAC for remembered items. '
        f'Model: memory ~ spread_z * stim * PAC_z + (1|patient_id).'
    )

    pac_csv_labels = []
    for pac_key, pac_label in PAC_METRICS:
        pdf.subsection_title(f'BLA-HPC {pac_label}')
        for sp_key, sp_label in SPREAD_KEYS:
            csv = os.path.join(OUTPUT_DIR,
                f'h3d_pac_BLAHPC_{pac_key}_{sp_key}_odds_ratios.csv')
            label = f'BLA-HPC {pac_label} + {sp_label}'
            add_model_table(pdf, csv, f'{sp_label} x Stim x {pac_label}')
            pac_csv_labels.append((csv, label))

    pdf.subsection_title('PAC Results Summary')
    notable = scan_notable(pac_csv_labels)
    if notable:
        pdf.body_text('Notable effects (p < .10):')
        for n in notable:
            pdf.body_text(f'  {n}')
    else:
        pdf.body_text('No notable effects found.')

    fig_path = os.path.join(OUTPUT_DIR, 'h3d_pac_forest.png')
    if os.path.exists(fig_path):
        pdf.subsection_title('Figure 6: PAC Three-Way Interactions Forest Plot')
        pdf.image(fig_path, x=10, w=190)
        pdf.ln(3)

    # ══════════════ SUMMARY ══════════════
    pdf.add_page()
    pdf.section_title('Summary')

    pdf.subsection_title('IED Spread and Memory (Part 2)')
    spread_csvs = [(os.path.join(OUTPUT_DIR, f'h3d_model{s}_odds_ratios.csv'), l)
                   for s, l in [('S1_channel', 'ChanSpread'), ('S2_region', 'RegSpread'),
                                ('S3_crossregion', 'CrossReg'), ('S4_withinregion', 'WithinReg')]]
    spread_notable = scan_notable(spread_csvs, threshold=0.05)
    if spread_notable:
        pdf.body_text('Significant spread effects (p < .05):')
        for n in spread_notable:
            pdf.body_text(f'  {n}')
    else:
        pdf.body_text(
            'No spread metric significantly predicted memory at the trial level, '
            'and no spread x stim interactions reached significance.'
        )

    pdf.subsection_title('Spread x Stim x BLA-HPC Coherence (Part 3)')
    coh_notable = scan_notable(coh_csv_labels, threshold=0.05)
    if coh_notable:
        pdf.body_text('Significant coherence findings (p < .05):')
        for n in coh_notable:
            pdf.body_text(f'  {n}')
    else:
        pdf.body_text('No significant effects in the BLA-HPC coherence models.')

    # Check for spread main effects in coherence models
    spread_mains = [s for s in (scan_notable(coh_csv_labels, 0.05))
                    if ('Spread' in s or 'CrossReg' in s or 'WithinReg' in s)
                    and ':' not in s.split('OR=')[0]]
    if spread_mains:
        pdf.body_text(
            'Patient-level IED spread was a significant predictor in some '
            'coherence models, indicating that patients with greater spread '
            'showed reduced memory performance across all their trials.'
        )

    pdf.subsection_title('Spread x Stim x BLA-HPC PAC (Part 4)')
    pac_notable = scan_notable(pac_csv_labels, threshold=0.05)
    if pac_notable:
        pdf.body_text('Significant PAC findings (p < .05):')
        for n in pac_notable:
            pdf.body_text(f'  {n}')
    else:
        pdf.body_text('No significant effects in the BLA-HPC PAC models.')

    pdf.subsection_title('Limitations')
    pdf.body_text(
        '1. Spread is a patient-level variable (not trial-level) since '
        'coherence data lacks trial identifiers to merge with specific IED events.\n'
        '2. PAC is condition-level, limiting precision.\n'
        '3. Three-way interactions require large samples; with 13-14 patients '
        'in neural models, findings should be interpreted cautiously.'
    )

    out_path = os.path.join(OUTPUT_DIR, 'H3d_Spread_Neural_MLM_Report.pdf')
    pdf.output(out_path)
    print(f'Saved: {out_path}')


if __name__ == '__main__':
    build_report()
