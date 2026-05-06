#!/usr/bin/env python
"""
Build APA-formatted PDF report for the Quon et al. (2021) replication analysis.
Outputs → outputs/ied_Quon_paper/quon_replication_report.pdf
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
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'outputs', 'ied_Quon_paper')
PDF_PATH = os.path.join(OUTPUT_DIR, 'quon_replication_report.pdf')

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
        'IED Features Associated With Memory Encoding:<br/>'
        'A Replication of Quon et al. (2021) in the AMME/BLAES Dataset',
        style_title))
    story.append(Spacer(1, 0.3*inch))
    story.append(Paragraph('Inman Lab', style_authors))
    story.append(Paragraph(
        'University of Utah', style_affil))
    story.append(Spacer(1, 0.5*inch))
    story.append(Paragraph(
        'This report replicates the analytic approach of Quon et al. (2021, <i>Epilepsia</i>, '
        '62, 2615\u20132626) using interictal epileptiform discharge (IED) data and neural '
        'recordings from the AMME/BLAES intracranial EEG dataset. Whereas Quon et al. '
        'examined lateral temporal cortex, the present analysis focuses on medial temporal '
        'lobe (MTL) structures\u2014amygdala, hippocampus, parahippocampal cortex, and '
        'entorhinal cortex\u2014and extends the neural analysis to include coherence between '
        'region pairs and phase\u2013amplitude coupling (PAC) in addition to spectral power.',
        style_body_first))

    story.append(PageBreak())

    # ========== METHOD ==========
    story.append(Paragraph('Method', style_h1))

    # -- Participants & Task --
    story.append(Paragraph('Participants and Memory Task', style_h2))
    story.append(Paragraph(
        'Thirty-three patients with medication-refractory epilepsy undergoing intracranial '
        'EEG monitoring participated in an image-based memory encoding and retrieval task. '
        'During the encoding phase, patients viewed images while intracranial electrodes '
        'recorded neural activity from medial temporal lobe structures. Memory was subsequently '
        'tested, and each trial was classified as <i>remembered</i> or <i>forgotten</i> based '
        'on the patient\u2019s recognition performance. A total of 1,711 IED detections '
        'co-occurred with trials for which a memory outcome was available (1,214 remembered, '
        '497 forgotten).',
        style_body_first))

    # -- IED Detection & Features --
    story.append(Paragraph('IED Detection and Feature Extraction', style_h2))
    story.append(Paragraph(
        'IEDs were detected during the encoding phase using automated detection methods '
        'applied to the intracranial EEG signal. For each detected IED, the following '
        'features were recorded: (a) <b>IED rate</b>, defined as the number of IEDs per '
        'trial; (b) <b>hemisphere</b> (left or right); (c) <b>brain region</b>, classified '
        'into MTL subregions (amygdala, hippocampus, parahippocampal cortex, entorhinal '
        'cortex) versus non-MTL regions; (d) <b>tissue type</b>, indicating whether the '
        'electrode contact was in gray matter (G), white matter (W), or borderline (B); '
        '(e) <b>channel spread</b>, the number of channels simultaneously showing an IED; '
        '(f) <b>region spread</b>, the number of distinct brain regions simultaneously '
        'involved; and (g) <b>timing</b>, whether the IED occurred during image presentation, '
        'before the image during the inter-trial interval, or after the image.',
        style_body_first))

    # -- Neural Measures --
    story.append(Paragraph('Neural Measures', style_h2))
    story.append(Paragraph(
        'Three neural measures were computed from the intracranial EEG recordings during '
        'encoding: spectral power, inter-regional coherence, and phase\u2013amplitude coupling (PAC).',
        style_body_first))
    story.append(Paragraph(
        '<b>Spectral power.</b> Baseline-corrected spectral power was computed for each '
        'trial at each electrode contact. Power was averaged within six canonical frequency '
        'bands: delta (2\u20134 Hz), theta (4\u20138 Hz), alpha (8\u201312 Hz), beta '
        '(12\u201330 Hz), slow gamma (30\u201355 Hz), and high-frequency activity (HFA; '
        '70\u2013100 Hz). Power was extracted separately for seven brain regions: basolateral '
        'amygdala (BLA), cornu ammonis (CA), dentate gyrus (DG), entorhinal cortex (EC), '
        'hippocampus proper (HPC), parahippocampal gyrus (PHG), and perirhinal cortex (PRC). '
        'The <i>subsequent memory effect</i> (SME) was defined as the difference in mean '
        'power between remembered and forgotten trials (remembered minus forgotten) for each '
        'patient, region, and frequency band, using only no-stimulation trials to isolate '
        'endogenous neural dynamics.',
        style_body))
    story.append(Paragraph(
        '<b>Coherence.</b> Baseline-corrected coherence was computed between all pairs of '
        'MTL region contacts for each trial. Coherence SME was calculated identically to '
        'power SME (remembered minus forgotten) for each patient, region pair, and frequency '
        'band. Thirteen region pairs with at least eight overlapping patients were retained '
        'for analysis (e.g., BLA\u2013CA, BLA\u2013HPC, CA\u2013PRC, EC\u2013HPC).',
        style_body))
    story.append(Paragraph(
        '<b>Phase\u2013amplitude coupling (PAC).</b> The modulation index for PAC was '
        'computed for MTL region pairs in two coupling bands: slow gamma (30\u201355 Hz) '
        'and HFA (70\u2013100 Hz). PAC SME was defined as the difference in the no-stimulation '
        'modulation index between remembered and forgotten trials.',
        style_body))

    # -- Statistical Approach --
    story.append(Paragraph('Statistical Approach', style_h2))
    story.append(Paragraph(
        'Following the approach of Quon et al. (2021), the analysis proceeded in two stages.',
        style_body_first))
    story.append(Paragraph(
        '<b>Stage 1: IED features and memory (Figure 2 analog).</b> Each IED feature was '
        'tested for its association with the binary memory outcome (remembered vs. forgotten). '
        'For continuous features (IED rate, channel spread, region spread), the point-biserial '
        'correlation between the feature and memory outcome was computed. For categorical '
        'features (hemisphere, tissue type, timing, MTL vs. non-MTL region), the difference '
        'in the proportion of remembered trials between groups was calculated and tested with '
        'a Mann\u2013Whitney <i>U</i> test. Pairwise comparisons between MTL subregions '
        'were similarly conducted. An interaction between IED rate and tissue type (gray vs. '
        'white matter) was assessed by splitting trials at the median rate and testing the '
        'rate effect separately within each tissue category.',
        style_body))
    story.append(Paragraph(
        '<b>Stage 2: IED features and neural memory effects (Figure 3 analog).</b> Patient-level '
        'summaries were computed for both IED features (mean rate, mean channel spread, mean '
        'region spread, proportion white matter, proportion left hemisphere, proportion during '
        'image) and neural SME values (per region and frequency band). Spearman rank correlations '
        'were then computed between each IED feature and each neural SME, yielding a correlation '
        'matrix for every combination of brain region (or region pair), frequency band, and IED '
        'feature. This was performed separately for power (5 regions with <i>n</i> \u2265 8), '
        'coherence (13 region pairs with <i>n</i> \u2265 8), and PAC (MTL composite). A total '
        'of 516 correlations were tested. Results are reported at the uncorrected \u03b1 = .05 '
        'threshold; significance should be interpreted cautiously given the number of comparisons.',
        style_body))

    story.append(PageBreak())

    # ========== RESULTS ==========
    story.append(Paragraph('Results', style_h1))

    # -- Table 1: Descriptive --
    story.append(Paragraph('Descriptive Statistics', style_h2))
    story.append(Paragraph(
        'Table 1 presents the descriptive characteristics of the IED detections during '
        'encoding trials with available memory outcomes.',
        style_body_first))

    story.append(Paragraph('<b>Table 1</b>', style_table_title))
    story.append(Paragraph('<i>Descriptive Characteristics of IED Detections During Encoding</i>',
                           style_table_note))

    t1_data = [
        ['Characteristic', 'Value'],
        ['Patients, n', '33'],
        ['Total IED detections', '1,711'],
        ['Trials with IEDs, n', '802'],
        ['', ''],
        ['Memory outcome, n (%)', ''],
        ['    Remembered', '1,214 (71.0%)'],
        ['    Forgotten', '497 (29.0%)'],
        ['', ''],
        ['IED rate (IEDs/trial), M (SD)', '1.65 (\u00b10.99)'],
        ['Channel spread, M (SD)', '2.48 (\u00b11.36)'],
        ['Region spread, M (SD)', '2.75 (\u00b11.38)'],
        ['', ''],
        ['Hemisphere, n (%)', ''],
        ['    Left', '904 (52.8%)'],
        ['    Right', '802 (46.9%)'],
        ['', ''],
        ['Tissue type, n (%)', ''],
        ['    Gray matter', '1,625 (95.0%)'],
        ['    White matter', '50 (2.9%)'],
        ['    Both/borderline', '30 (1.8%)'],
        ['', ''],
        ['MTL region, n (%)', ''],
        ['    Hippocampus', '1,053 (61.5%)'],
        ['    Amygdala', '78 (4.6%)'],
        ['    Parahippocampal', '60 (3.5%)'],
        ['    Entorhinal', '21 (1.2%)'],
        ['', ''],
        ['Timing, n (%)', ''],
        ['    During image', '1,062 (62.1%)'],
        ['    Before image ITI', '687 (40.2%)'],
        ['    After image ITI', '538 (31.4%)'],
        ['    During stimulation', '272 (15.9%)'],
    ]
    story.append(make_apa_table(t1_data, col_widths=[3.5*inch, 2.5*inch]))
    story.append(Paragraph(
        '<i>Note.</i> Timing categories are not mutually exclusive; a single IED may span '
        'multiple time windows. MTL = medial temporal lobe; ITI = inter-trial interval.',
        style_table_note))

    story.append(PageBreak())

    # -- IED Features & Memory --
    story.append(Paragraph('IED Features and Memory Performance', style_h2))
    story.append(Paragraph(
        'Consistent with Quon et al. (2021), higher IED rate was significantly associated '
        'with worse memory encoding (<i>r<sub>pb</sub></i> = \u2212.053, <i>p</i> = .029). '
        'Additionally, IEDs detected in white matter were associated with substantially lower '
        'recall than gray matter IEDs (proportion difference = \u2212.173, <i>p</i> = .008). '
        'No other IED features reached significance at \u03b1 = .05, although channel spread '
        'showed a trend (<i>r<sub>pb</sub></i> = .042, <i>p</i> = .082).',
        style_body_first))
    story.append(Paragraph(
        'Unlike Quon et al., who found that left-lateralized IEDs were more harmful to memory, '
        'hemisphere did not significantly predict memory in the present MTL dataset (<i>p</i> = .294). '
        'This may reflect differences in electrode coverage: the present study focuses on bilateral '
        'MTL structures, whereas Quon et al. examined lateral temporal cortex where left-hemisphere '
        'language regions are critical for the verbal free-recall task they used.',
        style_body))
    story.append(Paragraph(
        'An interaction analysis revealed that the IED rate effect was significant for gray '
        'matter IEDs (high vs. low rate: \u0394 = \u2212.056, <i>p</i> = .014) and showed a '
        'strong trend for white matter IEDs (\u0394 = \u2212.314, <i>p</i> = .055), suggesting '
        'that white matter IEDs may be particularly deleterious at higher rates, paralleling '
        'Quon et al.\u2019s amplitude \u00d7 white matter interaction.',
        style_body))
    story.append(Paragraph(
        'Pairwise comparisons among MTL subregions (amygdala, hippocampus, parahippocampal '
        'cortex, entorhinal cortex) did not yield significant differences, though hippocampus '
        'showed numerically lower recall rates than other MTL subregions. Table 2 presents '
        'the full set of IED feature\u2013memory associations.',
        style_body))

    # Table 2
    story.append(Paragraph('<b>Table 2</b>', style_table_title))
    story.append(Paragraph(
        '<i>IED Feature Associations With Memory Encoding Performance</i>',
        style_table_note))

    stats2 = pd.read_csv(os.path.join(OUTPUT_DIR, 'figure2_stats.csv'))
    t2_data = [['IED Feature', 'Effect Size', 'p', 'Test']]
    for _, row in stats2.iterrows():
        feat = row['Feature']
        if 'Region:' in feat:
            continue  # put pairwise in separate table
        es = f"{row['r_or_beta']:.3f}"
        p = sig_str(row['p_value'])
        s = star(row['p_value'])
        t2_data.append([feat, es + s, f'<i>p</i> {p}', row['test']])

    # Convert to Paragraph objects for italic support
    t2_para = []
    for r_idx, row in enumerate(t2_data):
        new_row = []
        for c_idx, cell in enumerate(row):
            st = style_body_first if r_idx > 0 else style_body_first
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
    add_figure(story, 'figure2_ied_features_memory.png',
               '<b>Figure 1.</b> IED features associated with memory encoding. '
               '(A) Effect sizes for each IED feature predicting memory outcome. '
               'Red bars indicate <i>p</i> < .05. '
               '(B) Interaction between IED rate (median split) and tissue type on '
               'proportion remembered. '
               '(C) Pairwise comparisons between MTL subregions (forest plot with 95% CIs).',
               max_height=5.5*inch)

    story.append(PageBreak())

    # ========== NEURAL RESULTS ==========
    story.append(Paragraph('IED Features and Neural Memory Effects', style_h2))

    # -- Power --
    story.append(Paragraph(
        '<b>Spectral power.</b> Spearman correlations between patient-level IED features and '
        'the encoding power subsequent memory effect (SME) were computed for each brain region '
        'and frequency band. Of 180 power correlations, 12 reached significance at '
        '<i>p</i> &lt; .05 (uncorrected).',
        style_body_first))
    story.append(Paragraph(
        'The strongest and most consistent findings emerged in the <b>basolateral amygdala '
        '(BLA)</b>, where higher IED rate was positively correlated with the power SME in the '
        'delta (\u03c1 = .507, <i>p</i> = .022), theta (\u03c1 = .583, <i>p</i> = .007), '
        'and alpha (\u03c1 = .484, <i>p</i> = .031) bands. This suggests that patients with '
        'higher IED rates showed a <i>larger</i> power difference between remembered and '
        'forgotten trials in the BLA at low frequencies. A similar pattern was observed for '
        'the proportion of left-lateralized IEDs, which correlated with BLA delta (\u03c1 = .610, '
        '<i>p</i> = .004) and theta (\u03c1 = .564, <i>p</i> = .010) SME.',
        style_body))
    story.append(Paragraph(
        'In the <b>entorhinal cortex (EC)</b>, IED rate was positively correlated with alpha '
        '(\u03c1 = .553, <i>p</i> = .050) and HFA (\u03c1 = .611, <i>p</i> = .027) power SME. '
        'In the <b>hippocampus (HPC)</b>, IED rate correlated with beta SME (\u03c1 = .385, '
        '<i>p</i> = .036), and the proportion of IEDs during image presentation was negatively '
        'correlated with alpha SME (\u03c1 = \u2212.380, <i>p</i> = .038). In <b>CA</b> and '
        '<b>PRC</b>, the proportion of white matter IEDs was negatively correlated with HFA '
        'SME (CA: \u03c1 = \u2212.426, <i>p</i> = .021; PRC: \u03c1 = \u2212.597, '
        '<i>p</i> = .015), consistent with Quon et al.\u2019s finding that white matter '
        'propagation disrupts high-frequency neural dynamics supporting memory.',
        style_body))

    # Power Table 3
    story.append(Paragraph('<b>Table 3</b>', style_table_title))
    story.append(Paragraph(
        '<i>Significant Power SME \u00d7 IED Feature Correlations (p < .05)</i>',
        style_table_note))

    stats3 = pd.read_csv(os.path.join(OUTPUT_DIR, 'figure3_stats.csv'))
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

    # Power summary heatmap figure
    story.append(PageBreak())
    add_figure(story, 'fig3_power_heatmap_ALL_regions_rate.png',
               '<b>Figure 2.</b> Heatmap of Spearman correlations between mean IED rate '
               'and encoding power SME across all brain regions and frequency bands. '
               'Cell values are \u03c1; asterisks denote <i>p</i> < .05.',
               max_height=4*inch)
    story.append(Spacer(1, 0.2*inch))
    add_figure(story, 'fig3_power_heatmap_ALL_regions_pct_white.png',
               '<b>Figure 3.</b> Heatmap of Spearman correlations between proportion of '
               'white matter IEDs and encoding power SME across all brain regions and '
               'frequency bands.',
               max_height=4*inch)

    # BLA scatter
    story.append(PageBreak())
    add_figure(story, 'fig3_power_scatter_BLA.png',
               '<b>Figure 4.</b> Scatter plots of mean IED rate versus encoding power SME '
               'in the BLA across six frequency bands. Dashed lines show linear fit; '
               'yellow reference line indicates SME = 0.',
               max_height=3.5*inch)
    story.append(Spacer(1, 0.2*inch))
    add_figure(story, 'fig3_power_heatmap_BLA.png',
               '<b>Figure 5.</b> Full correlation heatmap for BLA: all IED features (rows) '
               '\u00d7 all frequency bands (columns).',
               max_height=3.5*inch)

    # -- Coherence --
    story.append(PageBreak())
    story.append(Paragraph(
        '<b>Coherence.</b> Of 312 coherence correlations (13 region pairs \u00d7 6 bands '
        '\u00d7 ~4 IED features with sufficient data), 10 reached significance at '
        '<i>p</i> &lt; .05. Notable findings included:',
        style_body_first))
    story.append(Paragraph(
        '(a) In the <b>BLA\u2013CA</b> pair, higher IED rate was associated with larger '
        'delta-band coherence SME (\u03c1 = .557, <i>p</i> = .011, <i>n</i> = 20), '
        'suggesting that IED burden enhances the low-frequency coherence difference between '
        'remembered and forgotten trials in amygdala\u2013hippocampal circuitry.',
        style_body))
    story.append(Paragraph(
        '(b) In the <b>BLA\u2013PRC</b> pair, the proportion of white matter IEDs was '
        'negatively correlated with both delta (\u03c1 = \u2212.723, <i>p</i> = .012) and '
        'HFA (\u03c1 = \u2212.682, <i>p</i> = .021) coherence SME, indicating that white '
        'matter IED propagation disrupts the coherence-based memory signal between amygdala '
        'and perirhinal cortex.',
        style_body))
    story.append(Paragraph(
        '(c) In the <b>CA\u2013PRC</b> pair, mean region spread was positively correlated '
        'with theta coherence SME (\u03c1 = .702, <i>p</i> = .007) and negatively with '
        'beta coherence SME (\u03c1 = \u2212.702, <i>p</i> = .007), suggesting that '
        'wider IED spread may differentially modulate oscillatory coupling in hippocampal\u2013'
        'perirhinal circuits.',
        style_body))
    story.append(Paragraph(
        '(d) In the <b>EC\u2013HPC</b> pair, IED rate was positively correlated with '
        'theta coherence SME (\u03c1 = .581, <i>p</i> = .047), consistent with the role '
        'of entorhinal\u2013hippocampal theta connectivity in episodic memory.',
        style_body))

    # Coherence Table 4
    story.append(Paragraph('<b>Table 4</b>', style_table_title))
    story.append(Paragraph(
        '<i>Significant Coherence SME \u00d7 IED Feature Correlations (p < .05)</i>',
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
               'and encoding coherence SME across all MTL region pairs and frequency bands.',
               max_height=5*inch)
    story.append(Spacer(1, 0.15*inch))
    add_figure(story, 'fig3_coh_heatmap_ALL_pairs_pct_white.png',
               '<b>Figure 7.</b> Heatmap of Spearman correlations between proportion of '
               'white matter IEDs and encoding coherence SME across all MTL region pairs.',
               max_height=5*inch)

    # -- PAC --
    story.append(PageBreak())
    story.append(Paragraph(
        '<b>Phase\u2013amplitude coupling.</b> No significant correlations were observed '
        'between IED features and the PAC subsequent memory effect for either slow gamma or '
        'HFA coupling bands (<i>n</i> = 31 patients). This null finding may reflect the '
        'composite nature of the PAC measure (averaged across multiple MTL region pairs) or '
        'insufficient power.',
        style_body_first))

    add_figure(story, 'fig3_pac_heatmap.png',
               '<b>Figure 8.</b> Heatmap of Spearman correlations between IED features and '
               'PAC SME for slow gamma and HFA coupling bands.',
               max_height=3.5*inch)

    # ========== DISCUSSION ==========
    story.append(PageBreak())
    story.append(Paragraph('Discussion', style_h1))
    story.append(Paragraph(
        'The present analysis replicated key findings from Quon et al. (2021) in a medial '
        'temporal lobe dataset. Consistent with the original report, higher IED rate and '
        'white matter IED propagation were significantly associated with poorer memory '
        'encoding. The IED rate effect (<i>r</i> = \u2212.053) was comparable in direction '
        'and magnitude to Quon et al.\u2019s finding of reduced recall with increased IED '
        'rate in the lateral temporal cortex.',
        style_body_first))
    story.append(Paragraph(
        'The white matter finding was notably robust in the present dataset (<i>p</i> = .008), '
        'with white matter IEDs reducing the proportion remembered by approximately 17 '
        'percentage points relative to gray matter IEDs. This extends Quon et al.\u2019s '
        'observation that white matter propagation amplifies the deleterious effect of IEDs, '
        'and suggests that this principle generalizes from lateral to medial temporal structures. '
        'Mechanistically, white matter IEDs may disrupt long-range connectivity between MTL '
        'subregions that is necessary for successful encoding.',
        style_body))
    story.append(Paragraph(
        'The neural analysis revealed region-specific effects of IED features on the encoding '
        'power SME. The BLA emerged as the region most sensitive to IED burden, with IED rate '
        'correlating positively with delta, theta, and alpha power SME. This pattern\u2014'
        'patients with more IEDs showing <i>larger</i> power differences between remembered '
        'and forgotten trials\u2014may reflect a compensatory mechanism whereby the amygdala '
        'increases its engagement to overcome IED-related interference. Alternatively, it may '
        'indicate that patients with more IEDs have a more polarized neural response to '
        'encoding success versus failure.',
        style_body))
    story.append(Paragraph(
        'The coherence results complement the power findings. The BLA\u2013CA pair showed '
        'IED rate effects in the delta band, while the BLA\u2013PRC pair was sensitive to '
        'white matter IED propagation. The CA\u2013PRC theta coherence finding is particularly '
        'noteworthy given the established role of hippocampal\u2013perirhinal theta coupling '
        'in memory encoding. Finally, the EC\u2013HPC theta coherence correlation with IED '
        'rate aligns with the known importance of entorhinal\u2013hippocampal theta for '
        'episodic memory (Guderian et al., 2009; Solomon et al., 2019).',
        style_body))
    story.append(Paragraph(
        '<b>Limitations.</b> Several caveats warrant mention. First, the sample of 33 patients '
        'is substantially smaller than Quon et al.\u2019s 261 participants, limiting statistical '
        'power, particularly for the patient-level neural correlations. Second, the 516 '
        'correlations in the neural analysis were not corrected for multiple comparisons, and '
        'the 22 significant results (4.3%) are only modestly above the 5% rate expected by '
        'chance. Third, the neural data were not time-locked to individual IEDs (as in Quon '
        'et al.\u2019s peri-IED spectral analysis), but instead reflect trial-level encoding '
        'dynamics correlated with patient-level IED burden. Future work with peri-IED time-'
        'locked analysis would more directly replicate Quon et al.\u2019s approach.',
        style_body))
    story.append(Paragraph(
        '<b>Conclusion.</b> IED rate and white matter propagation significantly predicted '
        'worse memory encoding in MTL structures, consistent with Quon et al. (2021). The '
        'BLA was the MTL region most sensitive to IED burden in terms of encoding power '
        'dynamics, and white matter IEDs disrupted coherence-based memory signals in '
        'amygdala\u2013perirhinal and hippocampal\u2013perirhinal circuits. These findings '
        'support the clinical relevance of IED features for understanding cognitive impairment '
        'in epilepsy and extend the Quon et al. framework to medial temporal lobe structures.',
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
    ref_style = ParagraphStyle('APARef', parent=style_body,
                               firstLineIndent=0, leftIndent=36,
                               spaceBefore=4, spaceAfter=4)
    ref_style_hang = ParagraphStyle('APARefHang', parent=ref_style,
                                     firstLineIndent=-36)
    for ref in refs:
        story.append(Paragraph(ref, ref_style_hang))

    doc.build(story)
    print(f'PDF saved to: {PDF_PATH}')


if __name__ == '__main__':
    build_pdf()
