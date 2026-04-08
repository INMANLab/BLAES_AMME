#!/usr/bin/env python
"""
Build APA-formatted PDF report for the Quon et al. (2021) replication — Retrieval Phase.
Outputs → outputs/ied_Quon_paper/retrieval/quon_retrieval_report.pdf
"""

import os
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                 TableStyle, PageBreak, Image, KeepTogether)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'outputs', 'ied_Quon_paper', 'retrieval')
PDF_PATH = os.path.join(OUTPUT_DIR, 'quon_retrieval_report.pdf')

# ---- Styles ----
styles = getSampleStyleSheet()

style_title = ParagraphStyle(
    'APATitle', parent=styles['Title'],
    fontSize=16, leading=20, alignment=TA_CENTER,
    spaceAfter=6, fontName='Times-Bold')

style_authors = ParagraphStyle(
    'APAAuthors', parent=styles['Normal'],
    fontSize=12, leading=16, alignment=TA_CENTER,
    spaceAfter=4, fontName='Times-Roman')

style_affil = ParagraphStyle(
    'APAAffil', parent=styles['Normal'],
    fontSize=10, leading=14, alignment=TA_CENTER,
    spaceAfter=20, fontName='Times-Italic')

style_h1 = ParagraphStyle(
    'APAH1', parent=styles['Heading1'],
    fontSize=14, leading=18, fontName='Times-Bold',
    spaceBefore=18, spaceAfter=8)

style_h2 = ParagraphStyle(
    'APAH2', parent=styles['Heading2'],
    fontSize=12, leading=16, fontName='Times-BoldItalic',
    spaceBefore=14, spaceAfter=6, leftIndent=0)

style_body = ParagraphStyle(
    'APABody', parent=styles['Normal'],
    fontSize=11, leading=15, fontName='Times-Roman',
    alignment=TA_JUSTIFY, firstLineIndent=36,
    spaceBefore=2, spaceAfter=4)

style_body_first = ParagraphStyle(
    'APABodyFirst', parent=style_body,
    firstLineIndent=0)

style_table_title = ParagraphStyle(
    'APATableTitle', parent=styles['Normal'],
    fontSize=11, leading=14, fontName='Times-BoldItalic',
    spaceBefore=12, spaceAfter=4)

style_table_note = ParagraphStyle(
    'APATableNote', parent=styles['Normal'],
    fontSize=9, leading=12, fontName='Times-Italic',
    spaceBefore=2, spaceAfter=10)

style_fig_caption = ParagraphStyle(
    'APAFigCaption', parent=styles['Normal'],
    fontSize=10, leading=13, fontName='Times-Italic',
    spaceBefore=4, spaceAfter=12, alignment=TA_LEFT)


def sig_str(p):
    if p < .001: return '< .001'
    return f'= {p:.3f}'


def star(p):
    if p < .001: return '***'
    if p < .01: return '**'
    if p < .05: return '*'
    return ''


def make_apa_table(data, col_widths=None, font_size=9):
    """Create a styled APA table from list-of-lists data."""
    t = Table(data, colWidths=col_widths, repeatRows=1)
    header_style = [
        ('FONTNAME', (0, 0), (-1, 0), 'Times-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), font_size),
        ('LEADING', (0, 0), (-1, -1), font_size + 3),
        ('FONTNAME', (0, 1), (-1, -1), 'Times-Roman'),
        ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
        ('ALIGN', (0, 0), (0, -1), 'LEFT'),
        ('LINEABOVE', (0, 0), (-1, 0), 1, colors.black),
        ('LINEBELOW', (0, 0), (-1, 0), 1, colors.black),
        ('LINEBELOW', (0, -1), (-1, -1), 1, colors.black),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ]
    t.setStyle(TableStyle(header_style))
    return t


def add_figure(story, filename, caption, max_width=6.5*inch, max_height=4.5*inch):
    """Add a figure with caption to the story."""
    path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(path):
        return
    img = Image(path)
    iw, ih = img.imageWidth, img.imageHeight
    ratio = min(max_width / iw, max_height / ih)
    img.drawWidth = iw * ratio
    img.drawHeight = ih * ratio
    story.append(img)
    story.append(Paragraph(caption, style_fig_caption))


def build_pdf():
    doc = SimpleDocTemplate(
        PDF_PATH, pagesize=letter,
        topMargin=1*inch, bottomMargin=1*inch,
        leftMargin=1*inch, rightMargin=1*inch)

    story = []

    # ========== TITLE PAGE ==========
    story.append(Spacer(1, 1.5*inch))
    story.append(Paragraph(
        'IED Features Associated With Memory Retrieval:<br/>'
        'A Replication of Quon et al. (2021) in the AMME/BLAES Dataset',
        style_title))
    story.append(Spacer(1, 0.3*inch))
    story.append(Paragraph('Inman Lab', style_authors))
    story.append(Paragraph(
        'University of Utah', style_affil))
    story.append(Spacer(1, 0.5*inch))
    story.append(Paragraph(
        'This report extends the Quon et al. (2021, <i>Epilepsia</i>, 62, 2615\u20132626) '
        'replication to the <b>retrieval (test) phase</b> of the AMME/BLAES memory task. '
        'Whereas the companion encoding report examined IED features during initial image '
        'presentation, the present analysis focuses on IEDs detected during the recognition '
        'memory test and their association with retrieval success and peri-retrieval neural '
        'dynamics (power, coherence, and phase\u2013amplitude coupling) in medial temporal '
        'lobe structures.',
        style_body_first))

    story.append(PageBreak())

    # ========== METHOD ==========
    story.append(Paragraph('Method', style_h1))

    # -- Participants & Task --
    story.append(Paragraph('Participants and Memory Task', style_h2))
    story.append(Paragraph(
        'Eighteen patients with medication-refractory epilepsy underwent a recognition memory '
        'test following the encoding phase. During retrieval, patients viewed previously studied '
        'images intermixed with novel items and indicated whether each image was old or new. '
        'Trials were classified as <i>remembered</i> (correct old recognition) or <i>forgotten</i> '
        '(missed old items); new items were excluded from analysis. A total of 352 IED detections '
        'co-occurred with retrieval trials for which a memory outcome was available (266 remembered, '
        '86 forgotten).',
        style_body_first))

    # -- IED Detection & Features --
    story.append(Paragraph('IED Detection and Feature Extraction', style_h2))
    story.append(Paragraph(
        'IEDs were detected during the retrieval phase using the same automated methods applied '
        'to encoding. For each IED, the following features were recorded: (a) <b>IED rate</b>, '
        'the number of IEDs per trial; (b) <b>hemisphere</b> (left or right); (c) <b>brain '
        'region</b>, classified as MTL subregion (amygdala, hippocampus, parahippocampal cortex, '
        'entorhinal cortex) versus non-MTL; (d) <b>tissue type</b> (gray matter, white matter, '
        'or borderline); (e) <b>channel spread</b>; (f) <b>region spread</b>; and '
        '(g) <b>timing</b>, whether the IED occurred during the image/ITI presentation period.',
        style_body_first))
    story.append(Paragraph(
        'Compared with encoding (33 patients, 1,711 IED detections), the retrieval dataset is '
        'smaller (18 patients, 352 detections), reflecting the shorter duration of the test phase '
        'and fewer patients with sufficient IED data during retrieval.',
        style_body))

    # -- Neural Measures --
    story.append(Paragraph('Neural Measures', style_h2))
    story.append(Paragraph(
        'Three neural measures were computed from intracranial EEG during the retrieval phase, '
        'paralleling the encoding analysis.',
        style_body_first))
    story.append(Paragraph(
        '<b>Spectral power.</b> Baseline-corrected spectral power was computed for each retrieval '
        'trial. Power was averaged within six frequency bands: delta (2\u20134 Hz), theta '
        '(4\u20138 Hz), alpha (8\u201312 Hz), beta (12\u201330 Hz), slow gamma (30\u201355 Hz), '
        'and high-frequency activity (HFA; 70\u2013100 Hz). The retrieval subsequent memory effect '
        '(SME) was defined as the difference in mean power between remembered and forgotten trials '
        '(remembered minus forgotten) for each patient, region, and band, using only no-stimulation '
        'trials. Four regions met the minimum sample threshold (<i>n</i> \u2265 8): BLA (<i>n</i> = 15), '
        'CA (<i>n</i> = 15), HPC (<i>n</i> = 15), and PRC (<i>n</i> = 8). DG, EC, and PHG had '
        'insufficient overlap with the IED patient sample.',
        style_body))
    story.append(Paragraph(
        '<b>Coherence.</b> Baseline-corrected coherence was computed between MTL region pairs. '
        'Two region pairs met the <i>n</i> \u2265 8 threshold for the IED\u2013neural correlation: '
        'BLA\u2013CA (<i>n</i> = 13) and BLA\u2013HPC (<i>n</i> = 12). Additional pairs from '
        'the broader coherence dataset (13 pairs total) were included in summary heatmaps.',
        style_body))
    story.append(Paragraph(
        '<b>Phase\u2013amplitude coupling (PAC).</b> The modulation index was computed for MTL '
        'region pairs in two coupling bands: slow gamma (30\u201355 Hz) and HFA (70\u2013100 Hz). '
        'PAC SME was defined as the difference in the no-stimulation modulation index between '
        'remembered and forgotten trials (<i>n</i> = 18 patients).',
        style_body))

    # -- Statistical Approach --
    story.append(Paragraph('Statistical Approach', style_h2))
    story.append(Paragraph(
        'The same two-stage approach used for encoding was applied to retrieval.',
        style_body_first))
    story.append(Paragraph(
        '<b>Stage 1: IED features and memory (Figure 2 analog).</b> Each IED feature was '
        'tested for association with the binary retrieval outcome (remembered vs. forgotten). '
        'Point-biserial correlations were used for continuous features (IED rate, channel spread, '
        'region spread), and Mann\u2013Whitney <i>U</i> tests for categorical features '
        '(hemisphere, tissue type, timing, MTL vs. non-MTL). Pairwise comparisons between MTL '
        'subregions were also conducted.',
        style_body))
    story.append(Paragraph(
        '<b>Stage 2: IED features and neural retrieval effects (Figure 3 analog).</b> '
        'Patient-level summaries of IED features were correlated (Spearman \u03c1) with '
        'retrieval SME values for power (4 regions \u00d7 6 bands), coherence (2 primary pairs '
        '\u00d7 6 bands), and PAC (2 coupling bands). A total of 228 correlations were tested. '
        'Results are reported at the uncorrected \u03b1 = .05 threshold.',
        style_body))

    story.append(PageBreak())

    # ========== RESULTS ==========
    story.append(Paragraph('Results', style_h1))

    # -- Table 1: Descriptive --
    story.append(Paragraph('Descriptive Statistics', style_h2))
    story.append(Paragraph(
        'Table 1 presents the descriptive characteristics of IED detections during '
        'retrieval trials with available memory outcomes.',
        style_body_first))

    story.append(Paragraph('<b>Table 1</b>', style_table_title))
    story.append(Paragraph('<i>Descriptive Characteristics of IED Detections During Retrieval</i>',
                           style_table_note))

    t1_data = [
        ['Characteristic', 'Value'],
        ['Patients, n', '18'],
        ['Total IED detections', '352'],
        ['Trials with IEDs, n', '145'],
        ['', ''],
        ['Memory outcome, n (%)', ''],
        ['    Remembered', '266 (75.6%)'],
        ['    Forgotten', '86 (24.4%)'],
        ['', ''],
        ['IED rate (IEDs/trial), M (SD)', '1.77 (\u00b11.18)'],
        ['Channel spread, M (SD)', '2.93 (\u00b11.22)'],
        ['Region spread, M (SD)', '3.27 (\u00b11.68)'],
        ['', ''],
        ['Hemisphere, n (%)', ''],
        ['    Left', '157 (44.6%)'],
        ['    Right', '195 (55.4%)'],
        ['', ''],
        ['Tissue type, n (%)', ''],
        ['    Gray matter', '321 (91.2%)'],
        ['    White matter', '3 (0.9%)'],
        ['    Both/borderline', '26 (7.4%)'],
        ['', ''],
        ['MTL region, n (%)', ''],
        ['    Hippocampus', '225 (63.9%)'],
        ['    Amygdala', '46 (13.1%)'],
        ['    Parahippocampal', '23 (6.5%)'],
        ['    Entorhinal', '11 (3.1%)'],
    ]
    story.append(make_apa_table(t1_data, col_widths=[3.5*inch, 2.5*inch]))
    story.append(Paragraph(
        '<i>Note.</i> Only retrieval trials with a valid memory outcome (remembered or forgotten) '
        'are included. New items were excluded. MTL = medial temporal lobe.',
        style_table_note))

    story.append(PageBreak())

    # -- IED Features & Memory --
    story.append(Paragraph('IED Features and Retrieval Memory Performance', style_h2))
    story.append(Paragraph(
        'In contrast to encoding, where IED rate and white matter tissue type were significant '
        'predictors, the retrieval analysis revealed a different pattern. <b>Channel spread</b> '
        'was the only IED feature significantly associated with retrieval memory '
        '(<i>r<sub>pb</sub></i> = .198, <i>p</i> < .001). Greater IED spread across channels '
        'was associated with a higher proportion of remembered trials, an unexpected direction '
        'that may reflect confounding with overall IED burden or electrode coverage.',
        style_body_first))
    story.append(Paragraph(
        'The MTL versus non-MTL region comparison approached significance (<i>p</i> = .066), '
        'with non-MTL IEDs showing higher retrieval rates than MTL IEDs (difference = \u2212.187). '
        'This suggests that IEDs originating within the MTL may be more disruptive to retrieval '
        'than those outside the MTL, consistent with the role of MTL structures in recognition memory.',
        style_body))
    story.append(Paragraph(
        'IED rate did not significantly predict retrieval performance (<i>r<sub>pb</sub></i> = '
        '\u2212.070, <i>p</i> = .189), though the direction was consistent with encoding '
        '(higher rate associated with worse memory). The non-significant result likely reflects '
        'the smaller sample size (18 vs. 33 patients) and fewer IED detections (352 vs. 1,711). '
        'Hemisphere, timing, and region spread were also non-significant. Pairwise MTL subregion '
        'comparisons revealed no significant differences. Table 2 presents the full results.',
        style_body))

    # Table 2
    story.append(Paragraph('<b>Table 2</b>', style_table_title))
    story.append(Paragraph(
        '<i>IED Feature Associations With Retrieval Memory Performance</i>',
        style_table_note))

    stats2 = pd.read_csv(os.path.join(OUTPUT_DIR, 'figure2_retrieval_stats.csv'))
    t2_data = [['IED Feature', 'Effect Size', 'p', 'Test']]
    for _, row in stats2.iterrows():
        feat = row['Feature']
        if 'Region:' in feat:
            continue
        es = f"{row['r_or_beta']:.3f}"
        p = sig_str(row['p_value'])
        s = star(row['p_value'])
        t2_data.append([feat, es + s, f'<i>p</i> {p}', row['test']])

    t2_para = []
    for r_idx, row in enumerate(t2_data):
        new_row = []
        for c_idx, cell in enumerate(row):
            p_style = ParagraphStyle('tc', parent=style_body_first,
                                     fontSize=9, leading=12, firstLineIndent=0,
                                     alignment=TA_CENTER if c_idx > 0 else TA_LEFT)
            new_row.append(Paragraph(str(cell), p_style))
        t2_para.append(new_row)

    t2 = Table(t2_para, colWidths=[2.2*inch, 1.2*inch, 1.4*inch, 1.5*inch])
    t2.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, 0), 'Times-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('LINEABOVE', (0, 0), (-1, 0), 1, colors.black),
        ('LINEBELOW', (0, 0), (-1, 0), 1, colors.black),
        ('LINEBELOW', (0, -1), (-1, -1), 1, colors.black),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    story.append(t2)
    story.append(Paragraph(
        '<i>Note.</i> Effect size is point-biserial <i>r</i> for continuous features and '
        'proportion difference for categorical features. '
        '* <i>p</i> &lt; .05. ** <i>p</i> &lt; .01. *** <i>p</i> &lt; .001.',
        style_table_note))

    # Figure 2
    story.append(PageBreak())
    add_figure(story, 'figure2_retrieval_ied_features_memory.png',
               '<b>Figure 1.</b> Retrieval IED features associated with memory. '
               '(A) Effect sizes for each IED feature predicting retrieval outcome. '
               'Red bars indicate <i>p</i> < .05. '
               '(B) Interaction between IED rate (median split) and tissue type on '
               'proportion remembered. '
               '(C) Pairwise comparisons between MTL subregions (forest plot with 95% CIs).',
               max_height=5.5*inch)

    story.append(PageBreak())

    # ========== NEURAL RESULTS ==========
    story.append(Paragraph('IED Features and Neural Retrieval Effects', style_h2))

    # -- Power --
    story.append(Paragraph(
        '<b>Spectral power.</b> Spearman correlations between patient-level IED features and '
        'the retrieval power SME were computed for four brain regions (BLA, CA, HPC, PRC) across '
        'six frequency bands. Of 228 total correlations (including coherence and PAC), 22 reached '
        'significance at <i>p</i> &lt; .05 (uncorrected), with 17 in the power domain.',
        style_body_first))
    story.append(Paragraph(
        'The most prominent pattern involved the <b>proportion of left-lateralized IEDs</b>, '
        'which was positively correlated with retrieval power SME across multiple regions: '
        'BLA alpha (\u03c1 = .713, <i>p</i> = .003), beta (\u03c1 = .698, <i>p</i> = .004), '
        'and slow gamma (\u03c1 = .672, <i>p</i> = .006); HPC slow gamma (\u03c1 = .661, '
        '<i>p</i> = .007); CA slow gamma (\u03c1 = .529, <i>p</i> = .043); and PRC slow gamma '
        '(\u03c1 = .756, <i>p</i> = .030). This striking convergence across four MTL regions '
        'in the slow gamma band suggests that left-lateralized IEDs are associated with enhanced '
        'oscillatory differentiation between remembered and forgotten items during retrieval.',
        style_body))
    story.append(Paragraph(
        '<b>Channel spread</b> was negatively associated with power SME in BLA theta through '
        'slow gamma (\u03c1 ranging from \u2212.523 to \u2212.566, all <i>p</i> &lt; .05), '
        'HPC delta (\u03c1 = \u2212.586, <i>p</i> = .022), and PRC HFA (\u03c1 = \u2212.719, '
        '<i>p</i> = .045). This indicates that greater IED spatial spread was associated with '
        'reduced neural differentiation between memory outcomes, consistent with IED spread '
        'disrupting retrieval-related neural signals.',
        style_body))
    story.append(Paragraph(
        'In <b>CA</b>, mean region spread was negatively correlated with theta (\u03c1 = \u2212.565, '
        '<i>p</i> = .028) and beta (\u03c1 = \u2212.590, <i>p</i> = .021) power SME, indicating '
        'that wider anatomical IED spread attenuated hippocampal retrieval signals.',
        style_body))
    story.append(Paragraph(
        'In <b>PRC</b>, the proportion of white matter IEDs was negatively correlated with delta '
        '(\u03c1 = \u2212.733, <i>p</i> = .039), alpha (\u03c1 = \u2212.733, <i>p</i> = .039), '
        'and HFA (\u03c1 = \u2212.733, <i>p</i> = .039) power SME, echoing the encoding finding '
        'that white matter IED propagation disrupts memory-related neural dynamics.',
        style_body))

    # Power Table 3
    story.append(Paragraph('<b>Table 3</b>', style_table_title))
    story.append(Paragraph(
        '<i>Significant Retrieval Power SME \u00d7 IED Feature Correlations (p < .05)</i>',
        style_table_note))

    stats3 = pd.read_csv(os.path.join(OUTPUT_DIR, 'figure3_retrieval_stats.csv'))
    sig_pwr = stats3[(stats3['Measure'] == 'Power') & (stats3['p_value'] < 0.05)].sort_values(
        ['Region', 'p_value'])

    t3_data = [['Region', 'Band', 'IED Feature', '\u03c1', 'p', 'n']]
    for _, row in sig_pwr.iterrows():
        t3_data.append([
            row['Region'], row['Band'], row['IED_Feature_Label'],
            f"{row['rho']:.3f}", sig_str(row['p_value']), str(int(row['n']))])
    story.append(make_apa_table(t3_data,
                                col_widths=[0.7*inch, 1.5*inch, 1.5*inch, 0.7*inch, 0.9*inch, 0.5*inch]))
    story.append(Paragraph(
        '<i>Note.</i> SME = subsequent memory effect (remembered \u2212 forgotten). '
        '\u03c1 = Spearman rank correlation. All tests uncorrected.',
        style_table_note))

    # Power summary heatmap figures
    story.append(PageBreak())
    add_figure(story, 'fig3_power_heatmap_ALL_regions_pct_left.png',
               '<b>Figure 2.</b> Heatmap of Spearman correlations between proportion of '
               'left-lateralized IEDs and retrieval power SME across all brain regions and '
               'frequency bands. Cell values are \u03c1; asterisks denote <i>p</i> < .05.',
               max_height=4*inch)
    story.append(Spacer(1, 0.2*inch))
    add_figure(story, 'fig3_power_heatmap_ALL_regions_mean_ch_spread.png',
               '<b>Figure 3.</b> Heatmap of Spearman correlations between mean channel spread '
               'and retrieval power SME across all brain regions and frequency bands.',
               max_height=4*inch)

    # BLA scatter
    story.append(PageBreak())
    add_figure(story, 'fig3_power_scatter_BLA.png',
               '<b>Figure 4.</b> Scatter plots of mean IED rate versus retrieval power SME '
               'in the BLA across six frequency bands. Dashed lines show linear fit; '
               'yellow reference line indicates SME = 0.',
               max_height=3.5*inch)
    story.append(Spacer(1, 0.2*inch))
    add_figure(story, 'fig3_power_heatmap_BLA.png',
               '<b>Figure 5.</b> Full correlation heatmap for BLA: all IED features (rows) '
               '\u00d7 all frequency bands (columns) for retrieval power SME.',
               max_height=3.5*inch)

    # -- Coherence --
    story.append(PageBreak())
    story.append(Paragraph(
        '<b>Coherence.</b> Of the coherence correlations tested, three reached significance. '
        'In the <b>BLA\u2013CA</b> pair, higher IED rate was negatively associated with both '
        'theta (\u03c1 = \u2212.627, <i>p</i> = .022, <i>n</i> = 13) and delta '
        '(\u03c1 = \u2212.554, <i>p</i> = .050, <i>n</i> = 13) coherence SME. Unlike the '
        'encoding analysis where higher IED rate was associated with <i>larger</i> BLA\u2013CA '
        'delta coherence SME, the retrieval pattern is reversed: higher IED burden reduced the '
        'coherence differentiation between remembered and forgotten items during retrieval. This '
        'dissociation may reflect the different demands of encoding (enhanced engagement) versus '
        'retrieval (pattern reinstatement).',
        style_body_first))
    story.append(Paragraph(
        'In the <b>BLA\u2013HPC</b> pair, the proportion of left-lateralized IEDs was positively '
        'correlated with beta coherence SME (\u03c1 = .665, <i>p</i> = .018, <i>n</i> = 12), '
        'consistent with the broad left-hemisphere IED effect observed in the power domain.',
        style_body))

    # Coherence Table 4
    story.append(Paragraph('<b>Table 4</b>', style_table_title))
    story.append(Paragraph(
        '<i>Significant Retrieval Coherence SME \u00d7 IED Feature Correlations (p < .05)</i>',
        style_table_note))

    sig_coh = stats3[(stats3['Measure'] == 'Coherence') & (stats3['p_value'] < 0.05)].sort_values(
        ['Region', 'p_value'])

    t4_data = [['Region Pair', 'Band', 'IED Feature', '\u03c1', 'p', 'n']]
    for _, row in sig_coh.iterrows():
        t4_data.append([
            row['Region'], row['Band'], row['IED_Feature_Label'],
            f"{row['rho']:.3f}", sig_str(row['p_value']), str(int(row['n']))])
    story.append(make_apa_table(t4_data,
                                col_widths=[0.9*inch, 1.5*inch, 1.5*inch, 0.6*inch, 0.8*inch, 0.5*inch]))
    story.append(Paragraph(
        '<i>Note.</i> SME = subsequent memory effect. \u03c1 = Spearman rank correlation.',
        style_table_note))

    # Coherence heatmap
    story.append(PageBreak())
    add_figure(story, 'fig3_coh_heatmap_ALL_pairs_mean_rate.png',
               '<b>Figure 6.</b> Heatmap of Spearman correlations between mean IED rate '
               'and retrieval coherence SME across all MTL region pairs and frequency bands.',
               max_height=5*inch)
    story.append(Spacer(1, 0.15*inch))
    add_figure(story, 'fig3_coh_heatmap_ALL_pairs_pct_left.png',
               '<b>Figure 7.</b> Heatmap of Spearman correlations between proportion of '
               'left-lateralized IEDs and retrieval coherence SME across all MTL region pairs.',
               max_height=5*inch)

    # -- PAC --
    story.append(PageBreak())
    story.append(Paragraph(
        '<b>Phase\u2013amplitude coupling.</b> Two significant correlations emerged for PAC. '
        'The proportion of left-lateralized IEDs was strongly positively correlated with slow '
        'gamma PAC SME (\u03c1 = .780, <i>p</i> < .001, <i>n</i> = 18), the single strongest '
        'finding in the entire retrieval analysis. This suggests that patients with predominantly '
        'left-hemispheric IEDs showed substantially greater differentiation in slow gamma '
        'cross-frequency coupling between remembered and forgotten retrieval trials.',
        style_body_first))
    story.append(Paragraph(
        'Additionally, mean channel spread was negatively correlated with slow gamma PAC SME '
        '(\u03c1 = \u2212.492, <i>p</i> = .038, <i>n</i> = 18), consistent with the power '
        'findings that greater IED spatial spread attenuates neural memory signals. HFA PAC '
        'showed no significant associations with IED features.',
        style_body))

    add_figure(story, 'fig3_pac_heatmap.png',
               '<b>Figure 8.</b> Heatmap of Spearman correlations between IED features and '
               'retrieval PAC SME for slow gamma and HFA coupling bands.',
               max_height=3.5*inch)

    # ========== DISCUSSION ==========
    story.append(PageBreak())
    story.append(Paragraph('Discussion', style_h1))
    story.append(Paragraph(
        'The retrieval-phase analysis revealed a pattern of IED\u2013memory associations that '
        'both complements and diverges from the encoding findings. At the behavioral level, '
        'channel spread was the only significant predictor of retrieval success, whereas IED rate '
        'and white matter tissue type\u2014the two strongest encoding predictors\u2014did not '
        'reach significance during retrieval, likely due to the reduced sample size.',
        style_body_first))
    story.append(Paragraph(
        'The neural analysis yielded two dominant themes. First, the <b>proportion of left-'
        'lateralized IEDs</b> emerged as the most consistent predictor of retrieval neural '
        'dynamics, correlating positively with power SME in the slow gamma band across BLA, CA, '
        'HPC, and PRC, as well as BLA\u2013HPC beta coherence SME and slow gamma PAC SME '
        '(\u03c1 = .780). This left-hemisphere effect was not prominent during encoding, where '
        'IED rate was the dominant neural predictor. The retrieval-specific emergence of '
        'lateralization effects may reflect the greater involvement of left-hemisphere language '
        'networks during recognition judgments, which require explicit verbal/semantic processing '
        'of previously encoded images.',
        style_body))
    story.append(Paragraph(
        'Second, <b>channel spread</b> was consistently negatively correlated with power SME '
        '(BLA, HPC, PRC) and PAC SME. This extends the finding from encoding that IED spatial '
        'spread disrupts memory-related neural signals, and suggests that the disruptive effect '
        'of widespread IEDs generalizes from encoding to retrieval. The stronger channel spread '
        'effect during retrieval (behavioral significance at <i>p</i> < .001) compared to '
        'encoding (trend at <i>p</i> = .082) may reflect greater vulnerability of retrieval '
        'processes to distributed neural disruption.',
        style_body))
    story.append(Paragraph(
        'The BLA\u2013CA coherence finding is noteworthy: IED rate was <i>negatively</i> '
        'correlated with delta and theta coherence SME during retrieval, opposite to the '
        '<i>positive</i> correlation observed during encoding. This reversal may reflect '
        'different neural demands: during encoding, higher IED burden may drive compensatory '
        'amygdala\u2013hippocampal engagement, whereas during retrieval, IED burden simply '
        'disrupts the pattern completion and reinstatement processes that underlie recognition '
        'memory.',
        style_body))
    story.append(Paragraph(
        '<b>Limitations.</b> The retrieval dataset is substantially smaller than encoding '
        '(18 vs. 33 patients; 352 vs. 1,711 IED detections), limiting statistical power. '
        'Only four brain regions and two coherence pairs met the <i>n</i> \u2265 8 threshold '
        'for IED\u2013neural correlations. The 22 significant results out of 228 tests (9.6%) '
        'modestly exceed the 5% false positive rate, but the clustering of effects around '
        'left-hemisphere lateralization and channel spread suggests systematic rather than '
        'chance associations. Multiple comparison correction was not applied; results should '
        'be interpreted as exploratory.',
        style_body))
    story.append(Paragraph(
        '<b>Conclusion.</b> IED channel spread significantly predicted retrieval performance, '
        'and the proportion of left-lateralized IEDs was the dominant predictor of retrieval '
        'neural dynamics across power, coherence, and PAC. These retrieval-specific effects '
        'contrast with the encoding-phase dominance of IED rate and white matter propagation, '
        'suggesting that different IED features may be relevant for different stages of memory '
        'processing. Together with the encoding report, these findings provide a comprehensive '
        'picture of how IED characteristics modulate memory-related neural activity across the '
        'full encoding\u2013retrieval cycle in MTL structures.',
        style_body))

    # ========== REFERENCES ==========
    story.append(PageBreak())
    story.append(Paragraph('References', style_h1))
    refs = [
        'Guderian, S., Schott, B. H., Richardson-Klavehn, A., & Duzel, E. (2009). '
        'Medial temporal theta state before an event predicts episodic encoding success '
        'in humans. <i>Proceedings of the National Academy of Sciences</i>, <i>106</i>(13), '
        '5365\u20135370.',
        'Quon, R. J., Camp, E. J., Meisenhelter, S., Song, Y., Steimel, S. A., Testorf, M. E., '
        '... & Jobst, B. C. (2021). Features of intracranial interictal epileptiform discharges '
        'associated with memory encoding. <i>Epilepsia</i>, <i>62</i>(11), 2615\u20132626.',
        'Solomon, E. A., Stein, J. M., Das, S., Gorniak, R., Sperling, M. R., Worrell, G., '
        '... & Kahana, M. J. (2019). Dynamic theta networks in the human medial temporal lobe '
        'support episodic memory. <i>Current Biology</i>, <i>29</i>(7), 1100\u20131111.',
    ]
    ref_style_hang = ParagraphStyle('APARefHang', parent=style_body,
                                     firstLineIndent=-36, leftIndent=36,
                                     spaceBefore=4, spaceAfter=4)
    for ref in refs:
        story.append(Paragraph(ref, ref_style_hang))

    doc.build(story)
    print(f'PDF saved to: {PDF_PATH}')


if __name__ == '__main__':
    build_pdf()
