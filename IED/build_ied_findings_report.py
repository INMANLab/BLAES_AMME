#!/usr/bin/env python
"""
Build APA-format PDF summarizing all IED encoding & retrieval findings.
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
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'outputs', 'ied_timing_memory')
PDF_PATH = os.path.join(OUTPUT_DIR, 'IED_Encoding_Retrieval_Findings.pdf')


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
        'Interictal Epileptiform Discharge (IED) Effects on Memory:<br/>'
        'Encoding and Retrieval Phase Findings', s['title']))
    story.append(Spacer(1, 12))

    # ===== OVERVIEW =====
    story.append(Paragraph('Overview', s['heading']))
    story.append(Paragraph(
        'This report summarizes the effects of interictal epileptiform discharges (IEDs) '
        'on subsequent memory performance during both encoding and retrieval phases of a '
        'recognition memory task. IED timing, tissue location, weighted rate, and '
        'multi-window presence were examined as predictors of whether trials were '
        'subsequently remembered or forgotten. All analyses were conducted at the '
        'trial level, with IED presence collapsed across channels within each trial.',
        s['body']))

    story.append(Paragraph(
        'The encoding-phase dataset comprised 802 trials from 33 patients (571 remembered, '
        '231 forgotten). The retrieval-phase dataset comprised 145 trials from 18 patients '
        '(110 remembered, 35 forgotten). Chi-square tests, Spearman correlations, '
        'Mann-Whitney <i>U</i> tests, and point-biserial correlations were used as appropriate.',
        s['body']))

    # ================================================================
    # ENCODING PHASE
    # ================================================================
    story.append(PageBreak())
    story.append(Paragraph('Encoding Phase Results', s['heading']))

    # --- Timing ---
    story.append(Paragraph('IED Timing Window Effects on Subsequent Memory', s['subheading']))
    story.append(Paragraph(
        'Four timing windows were examined during encoding: Before Image (ITI), During Image, '
        'After Image (ITI), and During Stimulation. Chi-square tests assessed whether IED '
        'presence in each window was associated with subsequent memory outcome.',
        s['body']))

    story.append(Paragraph(
        '<b>Before Image.</b> IED presence before image onset was not significantly associated '
        'with subsequent memory, \u03c7\u00b2(1) = 0.81, <i>p</i> = .367. Trials with IEDs '
        'present were remembered at 73.0% (<i>n</i> = 341) compared to 69.8% when absent '
        '(<i>n</i> = 461).',
        s['body']))

    story.append(Paragraph(
        '<b>During Image.</b> IED presence during image presentation was not significantly '
        'associated with subsequent memory, \u03c7\u00b2(1) = 0.20, <i>p</i> = .656. '
        'Remembered rates were 70.6% with IEDs present (<i>n</i> = 527) and 72.4% when '
        'absent (<i>n</i> = 275).',
        s['body']))

    story.append(Paragraph(
        '<b>After Image.</b> IED presence after image offset was significantly associated '
        'with worse subsequent memory, \u03c7\u00b2(1) = 8.82, <i>p</i> = .003. Trials with '
        'IEDs in this window were remembered at only 64.5% (<i>n</i> = 279), compared to '
        '74.8% when IEDs were absent (<i>n</i> = 523).',
        s['body']))

    story.append(Paragraph(
        '<b>During Stimulation.</b> IED presence during stimulation was significantly '
        'associated with worse subsequent memory, \u03c7\u00b2(1) = 7.30, <i>p</i> = .007. '
        'Trials with IEDs during stimulation were remembered at 61.3% (<i>n</i> = 137) '
        'versus 73.2% when absent (<i>n</i> = 665).',
        s['body']))

    story.append(Paragraph(
        'In summary, IEDs occurring <i>after</i> image offset and <i>during stimulation</i> '
        'at encoding significantly impaired subsequent memory, whereas IEDs before or during '
        'image presentation did not.',
        s['body']))

    # --- Gray Matter ---
    story.append(Paragraph('IED Tissue Location', s['subheading']))
    story.append(Paragraph(
        'IED tissue location (gray vs. white matter) was examined as a predictor of subsequent '
        'memory. The majority of encoding trials had IEDs originating in gray matter '
        '(<i>n</i> = 767, 71.2% remembered), with very few white matter trials '
        '(<i>n</i> = 5, 20.0% remembered) and some trials with IEDs in both tissue types '
        '(<i>n</i> = 28, 78.6% remembered).',
        s['body']))

    story.append(Paragraph(
        'A chi-square test comparing gray versus white matter trials was significant, '
        '\u03c7\u00b2(1) = 4.07, <i>p</i> = .044, suggesting that white matter IEDs may be '
        'more detrimental to memory. However, this result should be interpreted with extreme '
        'caution given the very small white matter sample size (<i>n</i> = 5).',
        s['body']))

    # --- Weighted Rate ---
    story.append(Paragraph('Weighted IED Rate', s['subheading']))
    story.append(Paragraph(
        'The weighted IED rate (average across channels per trial) was examined as a continuous '
        'predictor of memory outcome. Forgotten trials had a significantly higher mean weighted '
        'IED rate (<i>M</i> = 1.78) compared to remembered trials (<i>M</i> = 1.56), '
        'Mann-Whitney <i>U</i> = 56,420, <i>p</i> = .001. A Spearman correlation confirmed '
        'a significant negative association between weighted IED rate and remembering, '
        '\u03c1 = \u2212.114, <i>p</i> = .001.',
        s['body']))

    story.append(Paragraph(
        'When binned, trials with a rate of 1.0 showed the highest memory performance (75.6%, '
        '<i>n</i> = 405), declining to 69.3% for rates 1\u20132 (<i>n</i> = 257), 58.2% for '
        'rates 2\u20133 (<i>n</i> = 91), with a partial recovery to 68.2% for rates 3+ '
        '(<i>n</i> = 44).',
        s['body']))

    # --- Multi-Window ---
    story.append(Paragraph('Multi-Window IED Exposure ("Dose-Response")', s['subheading']))
    story.append(Paragraph(
        'The number of timing windows in which an IED was present on a given trial (range: '
        '1\u20134) was examined as a dose-response predictor. A significant negative trend '
        'was observed: memory performance declined from 74.4% for 1-window trials (<i>n</i> = 426) '
        'to 70.7% for 2 windows (<i>n</i> = 283), 60.0% for 3 windows (<i>n</i> = 80), and '
        '46.2% for 4 windows (<i>n</i> = 13).',
        s['body']))

    story.append(Paragraph(
        'Both the Spearman correlation (\u03c1 = \u2212.096, <i>p</i> = .007) and chi-square '
        'test (\u03c7\u00b2 = 11.05, <i>p</i> = .011) were significant, supporting a '
        'dose-response relationship in which greater IED temporal spread during encoding '
        'is associated with worse subsequent memory.',
        s['body']))

    story.append(Paragraph(
        'Among specific timing combinations, trials with IEDs only before the image showed '
        'the best memory (80.6%, <i>n</i> = 124), while trials with IEDs spanning all four '
        'windows showed the worst (46.2%, <i>n</i> = 13). The combination of During + After + '
        'Stim was also notably poor (43.8%, <i>n</i> = 16).',
        s['body']))

    # ================================================================
    # RETRIEVAL PHASE
    # ================================================================
    story.append(PageBreak())
    story.append(Paragraph('Retrieval Phase Results', s['heading']))

    story.append(Paragraph(
        'Retrieval-phase analyses examined whether IEDs present during the recognition test '
        'affected memory accuracy for old items. Only old items (remembered and forgotten) were '
        'included; new items were excluded. Two timing windows were available: Before Image '
        'and During Image.',
        s['body']))

    # --- Timing ---
    story.append(Paragraph('IED Timing Window Effects at Retrieval', s['subheading']))

    story.append(Paragraph(
        '<b>Before Image.</b> IED presence before image onset at retrieval was not associated '
        'with memory accuracy, \u03c7\u00b2(1) = 0.01, <i>p</i> = .909. Hit rates were '
        '74.6% with IEDs present (<i>n</i> = 63) and 76.8% when absent (<i>n</i> = 82).',
        s['body']))

    story.append(Paragraph(
        '<b>During Image.</b> IED presence during image presentation at retrieval was not '
        'associated with memory accuracy, \u03c7\u00b2(1) = 0.00, <i>p</i> = 1.000. Hit '
        'rates were 76.0% with IEDs present (<i>n</i> = 129) and 75.0% when absent '
        '(<i>n</i> = 16).',
        s['body']))

    story.append(Paragraph(
        'Neither retrieval-phase timing window showed a significant effect of IED presence '
        'on recognition accuracy.',
        s['body']))

    # --- Gray Matter ---
    story.append(Paragraph('IED Tissue Location at Retrieval', s['subheading']))
    story.append(Paragraph(
        'At retrieval, nearly all trials had IEDs originating in gray matter (<i>n</i> = 141, '
        '77.3% remembered). Only 3 trials had IEDs in both gray and white matter (0% '
        'remembered), and no trials had exclusively white matter IEDs. The extremely limited '
        'tissue-type variability precluded meaningful comparison.',
        s['body']))

    # --- Weighted Rate ---
    story.append(Paragraph('Weighted IED Rate at Retrieval', s['subheading']))
    story.append(Paragraph(
        'The weighted IED rate at retrieval did not significantly differ between remembered '
        '(<i>M</i> = 1.65) and forgotten trials (<i>M</i> = 1.80), Mann-Whitney '
        '<i>U</i> = 1,778, <i>p</i> = .709. The Spearman correlation was also nonsignificant, '
        '\u03c1 = \u2212.032, <i>p</i> = .708.',
        s['body']))

    story.append(Paragraph(
        'Binned rates showed no systematic pattern: rates of 1 (75.4%), 1\u20132 (79.1%), '
        '2\u20133 (72.7%), and 3+ (77.8%) all showed comparable recognition accuracy.',
        s['body']))

    # --- Multi-Window ---
    story.append(Paragraph('Multi-Window IED Exposure at Retrieval', s['subheading']))
    story.append(Paragraph(
        'With only two possible timing windows at retrieval, trials could have IEDs in 0, 1, '
        'or 2 windows. Trials with 1 window showed 78.1% accuracy (<i>n</i> = 96) and trials '
        'with 2 windows showed 72.9% (<i>n</i> = 48). Neither the Spearman correlation '
        '(\u03c1 = \u2212.030, <i>p</i> = .716) nor the chi-square test (\u03c7\u00b2 = 3.64, '
        '<i>p</i> = .162) was significant.',
        s['body']))

    # ================================================================
    # SUMMARY
    # ================================================================
    story.append(PageBreak())
    story.append(Paragraph('Summary of Findings', s['heading']))

    story.append(Paragraph(
        'The pattern of results indicates that IEDs exert their primary disruptive effect '
        'during encoding rather than retrieval. At encoding, IEDs occurring after image offset '
        'and during electrical stimulation significantly impaired subsequent memory. A '
        'dose-response relationship was observed, with trials experiencing IEDs across more '
        'timing windows showing progressively worse memory. Higher weighted IED rates during '
        'encoding were also associated with poorer memory.',
        s['body']))

    story.append(Paragraph(
        'In contrast, no retrieval-phase IED variable\u2014timing window, tissue location, '
        'weighted rate, or multi-window exposure\u2014was significantly associated with '
        'recognition accuracy. This dissociation suggests that IEDs primarily disrupt memory '
        'formation (encoding and early consolidation processes) rather than memory retrieval.',
        s['body']))

    # Summary table
    story.append(Spacer(1, 12))
    story.append(Paragraph('Table 1', s['italic']))
    story.append(Paragraph(
        '<i>Summary of Significant and Nonsignificant IED Effects on Memory</i>',
        s['note']))
    story.append(Spacer(1, 6))

    table_data = [
        ['Analysis', 'Encoding Phase', 'Retrieval Phase'],
        ['IED Before Image', 'n.s. (p = .367)', 'n.s. (p = .909)'],
        ['IED During Image', 'n.s. (p = .656)', 'n.s. (p = 1.000)'],
        ['IED After Image', 'p = .003 **', 'N/A'],
        ['IED During Stim', 'p = .007 **', 'N/A'],
        ['Weighted IED Rate', 'p = .001 **', 'n.s. (p = .709)'],
        ['Multi-Window Dose', 'p = .007 **', 'n.s. (p = .716)'],
        ['Gray vs. White', 'p = .044 *', 'Insufficient data'],
    ]

    t = Table(table_data, colWidths=[2.2 * inch, 1.8 * inch, 1.8 * inch])
    t.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, 0), 'Times-Bold'),
        ('FONTNAME', (0, 1), (-1, -1), 'Times-Roman'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
        ('ALIGN', (0, 0), (0, -1), 'LEFT'),
        ('LINEBELOW', (0, 0), (-1, 0), 1, colors.black),
        ('LINEBELOW', (0, -1), (-1, -1), 1, colors.black),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(t)

    story.append(Spacer(1, 6))
    story.append(Paragraph(
        '<i>Note.</i> * <i>p</i> &lt; .05. ** <i>p</i> &lt; .01. N/A = timing window not '
        'available in retrieval-phase data.',
        s['note']))

    doc.build(story)
    print(f'Saved {PDF_PATH}')


if __name__ == '__main__':
    build_pdf()
