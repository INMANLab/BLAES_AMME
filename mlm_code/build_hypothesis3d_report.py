#!/usr/bin/env python
"""
Build APA-format PDF summarizing Hypothesis 3d results:
IED Spread, MTL Coverage, Neural Dynamics, & Memory Modulation.
"""

import os
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                 PageBreak, Table, TableStyle)
from reportlab.lib import colors

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'outputs', 'ied_spread_neural_memory')
PDF_PATH = os.path.join(OUTPUT_DIR, 'Hypothesis_3d_Results.pdf')


def build_styles():
    ss = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'APATitle', parent=ss['Title'],
        fontSize=16, leading=20, alignment=TA_CENTER,
        spaceAfter=6, fontName='Times-Bold')

    heading_style = ParagraphStyle(
        'APAHeading', parent=ss['Heading2'],
        fontSize=13, leading=16, fontName='Times-Bold',
        spaceBefore=18, spaceAfter=6, textColor=colors.black)

    subheading_style = ParagraphStyle(
        'APASubheading', parent=ss['Heading3'],
        fontSize=11, leading=14, fontName='Times-BoldItalic',
        spaceBefore=12, spaceAfter=4, textColor=colors.black)

    body_style = ParagraphStyle(
        'APABody', parent=ss['Normal'],
        fontSize=11, leading=15, fontName='Times-Roman',
        alignment=TA_JUSTIFY, spaceAfter=6,
        firstLineIndent=36)

    body_no_indent = ParagraphStyle(
        'APABodyNoIndent', parent=body_style,
        firstLineIndent=0)

    italic_style = ParagraphStyle(
        'APAItalic', parent=body_style,
        fontName='Times-Italic', firstLineIndent=0)

    note_style = ParagraphStyle(
        'APANote', parent=ss['Normal'],
        fontSize=9, leading=12, fontName='Times-Italic',
        alignment=TA_LEFT, spaceBefore=4, spaceAfter=2)

    return {
        'title': title_style,
        'heading': heading_style,
        'subheading': subheading_style,
        'body': body_style,
        'body_ni': body_no_indent,
        'italic': italic_style,
        'note': note_style,
    }


def build_pdf():
    doc = SimpleDocTemplate(
        PDF_PATH, pagesize=letter,
        topMargin=1 * inch, bottomMargin=1 * inch,
        leftMargin=1 * inch, rightMargin=1 * inch)

    s = build_styles()
    story = []

    # ===== TITLE =====
    story.append(Paragraph(
        'Hypothesis 3d: IED Spread, Brain Coverage,<br/>'
        'Functional Neural Dynamics, and Memory Modulation', s['title']))
    story.append(Spacer(1, 12))

    # ===== HYPOTHESIS =====
    story.append(Paragraph('Hypothesis', s['heading']))
    story.append(Paragraph(
        'IEDs with larger spread and brain coverage (e.g., amygdala [BLA], hippocampus '
        '[HPC, CA, DG], and perirhinal cortex [PRC]) during learning will cause more '
        'interruptions in functional neural dynamics (coherence and phase-amplitude coupling '
        '[PAC] during encoding), leading to more extreme memory modulation results '
        '(impairment or enhancement).',
        s['body']))

    # ===== METHOD =====
    story.append(Paragraph('Analytic Approach', s['heading']))

    story.append(Paragraph(
        'To test this hypothesis, a multi-level correlational approach was employed examining '
        'four links in the hypothesized causal chain: (a) IED spatial spread \u2192 memory '
        'modulation, (b) medial temporal lobe (MTL) region coverage \u2192 memory modulation, '
        '(c) IED spatial spread \u2192 encoding neural dynamics, and (d) encoding neural '
        'dynamics \u2192 memory modulation.',
        s['body']))

    story.append(Paragraph('<b>Participants.</b> '
        'The full IED\u2013behavioral analysis included 32 patients who had both trial-level '
        'IED data and behavioral memory modulation scores (avg_stim_dprime_diff). Of these, '
        '17 patients had encoding coherence data and 16 had encoding PAC data available for '
        'the neural dynamics analyses.',
        s['body']))

    story.append(Paragraph('<b>IED spread metrics.</b> '
        'For each patient, the following IED characteristics were computed across all '
        'encoding trials with memory outcomes: mean and maximum region spread (number of brain '
        'regions simultaneously involved in each IED event), mean and maximum channel spread '
        '(number of iEEG channels involved), and total number of distinct brain regions '
        'exhibiting IED activity.',
        s['body']))

    story.append(Paragraph('<b>MTL region coverage.</b> '
        'Each patient\u2019s IED data were examined for the presence of IEDs in three core MTL '
        'structures: (a) BLA, defined as the \u201cAmygdala\u201d region label; (b) HPC, '
        'defined as any region label containing \u201cHippocampus\u201d (encompassing CA and DG '
        'subfields); and (c) PRC, defined as the \u201cParahippocampal\u201d region label. '
        'A composite MTL coverage score (range: 0\u20133) was computed as the number of these '
        'regions in which at least one IED was detected during encoding.',
        s['body']))

    story.append(Paragraph('<b>Functional neural dynamics.</b> '
        'Encoding-phase coherence was quantified as the mean stim\u2013nostim coherence '
        'difference across MTL region pairs (e.g., BLA\u2013CA, BLA\u2013HPC, HPC\u2013PRC) '
        'in three frequency bands: theta (4\u20138 Hz), slow gamma (30\u201355 Hz), and high '
        'frequency activity (HFA; 70\u2013100 Hz). Phase-amplitude coupling (PAC) was '
        'quantified as the stim \u00d7 memory interaction (i.e., [stim remembered \u2212 '
        'nostim remembered] \u2212 [stim forgotten \u2212 nostim forgotten]) for slow gamma '
        'and HFA bands across MTL region pairs.',
        s['body']))

    story.append(Paragraph('<b>Memory modulation.</b> '
        'The primary outcome was the average stimulation-induced d\u2032 difference '
        '(avg_stim_dprime_diff), reflecting the degree of memory enhancement or impairment '
        'caused by electrical stimulation. Because the hypothesis predicted more '
        '<i>extreme</i> modulation (in either direction), the absolute value of d\u2032 '
        'difference was also examined.',
        s['body']))

    story.append(Paragraph('<b>Statistical analyses.</b> '
        'All pairwise associations were assessed using Spearman rank-order correlations. '
        'Mann-Whitney <i>U</i> tests compared d\u2032 differences between patients with '
        'and without IEDs in each MTL region. All tests were two-tailed with \u03b1 = .05.',
        s['body']))

    # ===== RESULTS =====
    story.append(PageBreak())
    story.append(Paragraph('Results', s['heading']))

    # --- Path A: IED Spread → Memory Modulation ---
    story.append(Paragraph(
        'IED Spatial Spread and Memory Modulation', s['subheading']))

    story.append(Paragraph(
        'None of the IED spatial spread metrics were significantly correlated with '
        'stimulation-induced memory modulation. Mean region spread was not significantly '
        'associated with avg_stim_dprime_diff, \u03c1 = .175, <i>p</i> = .339, <i>n</i> = 32, '
        'nor were max region spread (\u03c1 = .066, <i>p</i> = .721), mean channel spread '
        '(\u03c1 = \u2212.081, <i>p</i> = .659), max channel spread (\u03c1 = \u2212.071, '
        '<i>p</i> = .698), or number of distinct IED regions (\u03c1 = .207, <i>p</i> = .256).',
        s['body']))

    story.append(Paragraph(
        'When the absolute value of d\u2032 difference was examined to capture the '
        '<i>magnitude</i> of memory modulation regardless of direction, mean region spread '
        'showed a trend-level positive association, \u03c1 = .290, <i>p</i> = .107, suggesting '
        'that patients with more spatially widespread IEDs may have experienced more extreme '
        '(though not directionally consistent) memory modulation effects. However, this '
        'association did not reach statistical significance.',
        s['body']))

    # --- Path B: MTL Coverage → Memory Modulation ---
    story.append(Paragraph(
        'MTL Region Coverage and Memory Modulation', s['subheading']))

    story.append(Paragraph(
        'The composite MTL coverage score (number of core MTL regions\u2014BLA, HPC, '
        'PRC\u2014exhibiting IEDs) was not significantly associated with d\u2032 difference, '
        '\u03c1 = .173, <i>p</i> = .344, <i>n</i> = 32. Among the 32 patients, 3 had no MTL '
        'IEDs (coverage = 0), 18 had IEDs in 1 MTL region, 6 had IEDs in 2 regions, and 5 '
        'had IEDs in all 3 regions.',
        s['body']))

    story.append(Paragraph(
        'Region-specific comparisons further confirmed the null result. Mann-Whitney '
        '<i>U</i> tests comparing d\u2032 differences between patients with and without IEDs '
        'in each region were nonsignificant: BLA (<i>U</i> = 136.5, <i>p</i> = .173), '
        'HPC (<i>U</i> = 65.0, <i>p</i> = .917), and PRC (<i>U</i> = 90.5, '
        '<i>p</i> = .562). Notably, HPC was the most commonly affected region (IEDs present '
        'in 29 of 32 patients), leaving very little variability for detecting an effect.',
        s['body']))

    story.append(Paragraph(
        'The MTL coverage score was also not significantly associated with the absolute value '
        'of d\u2032 difference, \u03c1 = \u2212.164, <i>p</i> = .369, providing no evidence '
        'that broader MTL IED coverage was linked to more extreme memory modulation in either '
        'direction.',
        s['body']))

    # --- Path C: IED Spread → Neural Dynamics ---
    story.append(Paragraph(
        'IED Spatial Spread and Encoding Neural Dynamics', s['subheading']))

    story.append(Paragraph(
        'Among the subset of patients with encoding coherence data (<i>n</i> = 17), mean '
        'region spread showed a marginally significant positive correlation with the '
        'stimulation-induced theta coherence effect across MTL region pairs, \u03c1 = .461, '
        '<i>p</i> = .063. This trend suggested that patients with more spatially widespread '
        'IEDs exhibited larger stim-related changes in theta-band (4\u20138 Hz) coherence '
        'during encoding. However, this association did not survive the conventional '
        'significance threshold.',
        s['body']))

    story.append(Paragraph(
        'Mean region spread was not significantly associated with stim-induced coherence '
        'changes in the slow gamma band (\u03c1 = \u2212.091, <i>p</i> = .729) or HFA band '
        '(\u03c1 = \u2212.032, <i>p</i> = .903). Similarly, mean region spread was not '
        'significantly correlated with PAC stim \u00d7 memory interactions in either slow '
        'gamma (\u03c1 = .235, <i>p</i> = .380, <i>n</i> = 16) or HFA (\u03c1 = .247, '
        '<i>p</i> = .356, <i>n</i> = 16).',
        s['body']))

    story.append(Paragraph(
        'The MTL coverage score was not significantly associated with any coherence or PAC '
        'metric (all <i>p</i>s > .38).',
        s['body']))

    # --- Path D: Neural Dynamics → Memory Modulation ---
    story.append(Paragraph(
        'Encoding Neural Dynamics and Memory Modulation', s['subheading']))

    story.append(Paragraph(
        'None of the encoding neural dynamics measures were significantly correlated with '
        'stimulation-induced memory modulation. Theta coherence stim effect was not associated '
        'with d\u2032 difference (\u03c1 = .114, <i>p</i> = .662, <i>n</i> = 17), nor were '
        'slow gamma coherence (\u03c1 = \u2212.061, <i>p</i> = .815) or HFA coherence '
        '(\u03c1 = .026, <i>p</i> = .922). PAC stim \u00d7 memory interactions were also '
        'nonsignificant for both slow gamma (\u03c1 = \u2212.209, <i>p</i> = .437, '
        '<i>n</i> = 16) and HFA (\u03c1 = .262, <i>p</i> = .327, <i>n</i> = 16).',
        s['body']))

    # ===== SUMMARY TABLE =====
    story.append(PageBreak())
    story.append(Paragraph('Summary', s['heading']))

    story.append(Paragraph('Table 1', s['italic']))
    story.append(Paragraph(
        '<i>Spearman Correlations Across the Hypothesized Causal Chain '
        '(IED Spread \u2192 Neural Dynamics \u2192 Memory Modulation)</i>',
        s['note']))
    story.append(Spacer(1, 6))

    table_data = [
        ['Path', 'Predictor', 'Outcome', '\u03c1', '<i>p</i>', '<i>n</i>'],
        ['A', 'Mean Region Spread', 'd\u2032 Diff', '.175', '.339', '32'],
        ['A', 'Max Region Spread', 'd\u2032 Diff', '.066', '.721', '32'],
        ['A', 'Mean Channel Spread', 'd\u2032 Diff', '\u2212.081', '.659', '32'],
        ['A', 'N Distinct Regions', 'd\u2032 Diff', '.207', '.256', '32'],
        ['A', 'Mean Region Spread', '|d\u2032 Diff|', '.290', '.107', '32'],
        ['B', 'N MTL Regions', 'd\u2032 Diff', '.173', '.344', '32'],
        ['B', 'N MTL Regions', '|d\u2032 Diff|', '\u2212.164', '.369', '32'],
        ['C', 'Mean Region Spread', 'Coh \u03b8 Stim', '.461', '.063\u2020', '17'],
        ['C', 'Mean Region Spread', 'Coh S\u03b3 Stim', '\u2212.091', '.729', '17'],
        ['C', 'Mean Region Spread', 'PAC S\u03b3 Int.', '.235', '.380', '16'],
        ['C', 'Mean Region Spread', 'PAC HFA Int.', '.247', '.356', '16'],
        ['D', 'Coh \u03b8 Stim Effect', 'd\u2032 Diff', '.114', '.662', '17'],
        ['D', 'Coh S\u03b3 Stim Effect', 'd\u2032 Diff', '\u2212.061', '.815', '17'],
        ['D', 'PAC S\u03b3 Interaction', 'd\u2032 Diff', '\u2212.209', '.437', '16'],
        ['D', 'PAC HFA Interaction', 'd\u2032 Diff', '.262', '.327', '16'],
    ]

    # Convert italic tags in header
    header_row = []
    for cell in table_data[0]:
        header_row.append(Paragraph(cell, ParagraphStyle(
            'TableHeader', fontName='Times-Bold', fontSize=9, leading=11,
            alignment=TA_CENTER)))
    proc_data = [header_row]
    for row in table_data[1:]:
        proc_row = []
        for cell in row:
            proc_row.append(Paragraph(cell, ParagraphStyle(
                'TableCell', fontName='Times-Roman', fontSize=9, leading=11,
                alignment=TA_CENTER)))
        proc_data.append(proc_row)

    t = Table(proc_data, colWidths=[0.4 * inch, 1.6 * inch, 1.1 * inch,
                                     0.7 * inch, 0.7 * inch, 0.5 * inch])
    t.setStyle(TableStyle([
        ('LINEBELOW', (0, 0), (-1, 0), 1, colors.black),
        ('LINEBELOW', (0, -1), (-1, -1), 1, colors.black),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('ALIGN', (1, 1), (1, -1), 'LEFT'),
        ('ALIGN', (2, 1), (2, -1), 'LEFT'),
    ]))
    story.append(t)

    story.append(Spacer(1, 6))
    story.append(Paragraph(
        '<i>Note.</i> Path A = IED spread \u2192 memory modulation; '
        'Path B = MTL coverage \u2192 memory modulation; '
        'Path C = IED spread \u2192 neural dynamics; '
        'Path D = neural dynamics \u2192 memory modulation. '
        'Coh = coherence; \u03b8 = theta (4\u20138 Hz); S\u03b3 = slow gamma (30\u201355 Hz); '
        'HFA = high frequency activity (70\u2013100 Hz); Int. = stim \u00d7 memory interaction. '
        '\u2020 <i>p</i> &lt; .10.',
        s['note']))

    # ===== DISCUSSION =====
    story.append(Spacer(1, 18))
    story.append(Paragraph('Interpretation', s['heading']))

    story.append(Paragraph(
        'Hypothesis 3d predicted that IEDs with larger spatial spread and greater coverage '
        'of MTL structures during encoding would disrupt functional neural dynamics '
        '(coherence and PAC), leading to more extreme memory modulation outcomes. This '
        'hypothesis was largely <b>not supported</b> by the current data.',
        s['body']))

    story.append(Paragraph(
        'No IED spread metric was significantly associated with the magnitude or direction '
        'of stimulation-induced memory modulation, and MTL region coverage (BLA, HPC, PRC) '
        'similarly showed no significant relationship with d\u2032 differences. The only '
        'notable trend was a marginally significant association between mean region spread '
        'and theta-band coherence stim effects (\u03c1 = .461, <i>p</i> = .063), which '
        'hinted that more widespread IEDs may perturb theta-frequency functional connectivity '
        'during encoding. However, this trend did not survive conventional significance '
        'thresholds, and theta coherence itself was not significantly linked to memory '
        'modulation, breaking the hypothesized causal chain.',
        s['body']))

    story.append(Paragraph(
        'Several factors may have limited the ability to detect effects. First, the sample '
        'sizes for neural dynamics analyses were modest (<i>n</i> = 16\u201317), reducing '
        'statistical power for detecting medium-sized correlations. Second, HPC IEDs were '
        'nearly ubiquitous (present in 29 of 32 patients), creating a ceiling effect that '
        'constrained variability in the MTL coverage predictor. Third, the IED spread metrics '
        'were computed at the patient level, averaging across trials, which may have obscured '
        'trial-level relationships between IED characteristics and memory outcomes. Future '
        'analyses employing trial-level multilevel models may be better positioned to detect '
        'these within-patient effects.',
        s['body']))

    story.append(Paragraph(
        'Despite these null findings at the patient level, the earlier encoding-phase '
        'analyses (Hypotheses 3a\u20133c) demonstrated robust trial-level effects of IED '
        'timing, weighted rate, and multi-window exposure on subsequent memory. This '
        'dissociation suggests that while IED <i>temporal</i> characteristics during encoding '
        'reliably predict memory outcomes at the trial level, IED <i>spatial</i> spread does '
        'not straightforwardly scale up to predict individual differences in overall memory '
        'modulation.',
        s['body']))

    doc.build(story)
    print(f'Saved {PDF_PATH}')


if __name__ == '__main__':
    build_pdf()
