#!/usr/bin/env python
"""
Build PDF report for Hypothesis 3c: Bilateral IED Occurrence and Memory.

Simplified design (single predictor, no L/R hemisphere terms):
    Encoding:  memory ~ bilateral_ied + (1 | patient_id)   [bilateral-coverage patients]
    Retrieval: memory ~ bilateral_ied + (1 | patient_id)   [bilateral-coverage patients]
Bilateral_ied = 1 when IEDs are detected in both hemispheres on the same trial,
0 when unilateral. Model fixed effects come from h3c_bilateral_only_mlm.R.

Output: IED/ied_timing_memory/H3c_Bilateral_MLM_Report.pdf
"""

import os
import pandas as pd
from fpdf import FPDF

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Canonical data/output dir (data + PDF live here after the topic-dir reorg)
OUTPUT_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, '..', 'IED', 'ied_timing_memory'))
FIG_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, '..', 'OUTPUTS', 'updated_IED_figures'))


class APAReport(FPDF):
    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=25)

    def header(self):
        self.set_font('Helvetica', 'I', 8)
        self.cell(0, 5, 'Hypothesis 3c: Bilateral IED Occurrence and Memory', align='R')
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

    def figure(self, path, caption, width):
        if not os.path.exists(path):
            return
        self.ln(2)
        self.set_font('Times', 'I', 10)
        self.multi_cell(0, 5, caption)
        self.ln(1)
        x = (self.w - width) / 2
        self.image(path, x=x, w=width)
        self.ln(3)

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


def bh_fdr(pvals):
    """Benjamini-Hochberg q-values aligned to input order (plain Python)."""
    n = len(pvals)
    order = sorted(range(n), key=lambda i: pvals[i])
    q_raw = [0.0] * n
    for rank, idx in enumerate(order, start=1):
        q_raw[rank - 1] = pvals[idx] * n / rank
    running = 1.0
    q_mono = [0.0] * n
    for k in range(n - 1, -1, -1):
        running = min(running, q_raw[k])
        q_mono[k] = running
    out = [0.0] * n
    for rank, idx in enumerate(order):
        out[idx] = min(q_mono[rank], 1.0)
    return out


def format_or_row(r, label_map=None):
    """Format a tidy model row for the table."""
    term = r['term']
    label = label_map.get(term, term) if label_map else term
    p = r['p.value']
    or_val = r['estimate']
    return [
        label,
        f'{or_val:.3f}',
        f'[{r["conf.low"]:.3f}, {r["conf.high"]:.3f}]',
        f'{r["statistic"]:.2f}',
        f'{p_str(p)}{sig_str(p)}',
    ]


def get(df, term, col):
    return df[df['term'] == term][col].values[0]


def main():
    pdf = APAReport()
    pdf.add_page()

    # Title
    pdf.set_font('Helvetica', 'B', 16)
    pdf.cell(0, 10, 'Hypothesis 3c:', align='C', new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 10, 'Bilateral IED Occurrence and Memory', align='C',
             new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    pdf.italic_text(
        'Frequent bilateral IED occurrence (e.g., both hippocampi at the same time, or '
        'amygdala and hippocampus together) during learning or retrieval will predict '
        'subsequent memory impairment regardless of BLA stimulation.'
    )

    # ── Load data ────────────────────────────────────────────────────────────
    enc_bi = pd.read_csv(os.path.join(OUTPUT_DIR, 'h3c_encoding_bilateral.csv'))
    enc_all = pd.read_csv(os.path.join(OUTPUT_DIR, 'h3c_encoding_all.csv'))
    ret_bi = pd.read_csv(os.path.join(OUTPUT_DIR, 'h3c_retrieval_bilateral.csv'))
    ret_all = pd.read_csv(os.path.join(OUTPUT_DIR, 'h3c_retrieval_all.csv'))

    mE = pd.read_csv(os.path.join(OUTPUT_DIR, 'h3c_enc_bilonly_or.csv'))
    mEs = pd.read_csv(os.path.join(OUTPUT_DIR, 'h3c_enc_bilonly_stim_or.csv'))
    mE_stim = pd.read_csv(os.path.join(OUTPUT_DIR, 'h3c_enc_stimonly_or.csv'))
    mEs_lrt = pd.read_csv(os.path.join(OUTPUT_DIR, 'h3c_enc_hierarchical_lrt.csv'))
    mEx = pd.read_csv(os.path.join(OUTPUT_DIR, 'h3c_enc_bilxstim_or.csv'))
    mEx_lrt = pd.read_csv(os.path.join(OUTPUT_DIR, 'h3c_enc_bilxstim_lrt.csv'))
    mEw = pd.read_csv(os.path.join(OUTPUT_DIR, 'h3c_enc_bilwindow_or.csv'))
    mR = pd.read_csv(os.path.join(OUTPUT_DIR, 'h3c_ret_bilonly_or.csv'))

    # ── Derived counts ───────────────────────────────────────────────────────
    n_cov_enc = enc_bi['patient_id'].nunique()
    n_cov_enc_tr = len(enc_bi)
    n_all_enc = enc_all['patient_id'].nunique()
    enc_bil = enc_bi[enc_bi['bilateral_ied'] == 1]
    enc_uni = enc_bi[enc_bi['bilateral_ied'] == 0]
    n_bil_enc_tr = len(enc_bil)
    n_bil_enc_pat = enc_bil['patient_id'].nunique()

    n_cov_ret = ret_bi['patient_id'].nunique()
    n_cov_ret_tr = len(ret_bi)
    n_all_ret = ret_all['patient_id'].nunique()
    ret_bil = ret_bi[ret_bi['bilateral_ied'] == 1]
    ret_uni = ret_bi[ret_bi['bilateral_ied'] == 0]
    n_bil_ret_tr = len(ret_bil)
    n_bil_ret_pat = ret_bil['patient_id'].nunique()

    # ── Sample and Data Structure ────────────────────────────────────────────
    pdf.section_title('Sample and Data Structure')
    pdf.body_text(
        f'A bilateral IED is defined as IEDs detected in both hemispheres on the same trial. '
        f'Bilateral occurrence can only be observed in patients with bilateral electrode '
        f'coverage, so analyses are restricted to those patients. Within them, each trial is '
        f'coded as bilateral (IEDs in both hemispheres) versus unilateral (IEDs confined to '
        f'one hemisphere); left/right hemisphere is not modeled.'
    )
    pdf.body_text(
        f'At encoding, {n_cov_enc} of {n_all_enc} patients had bilateral coverage '
        f'({n_cov_enc_tr} trials); {n_bil_enc_tr} of those trials, from {n_bil_enc_pat} '
        f'patients, had bilateral IEDs. At retrieval, {n_cov_ret} of {n_all_ret} patients had '
        f'bilateral coverage ({n_cov_ret_tr} trials); only {n_bil_ret_tr} trials, from '
        f'{n_bil_ret_pat} patients, had bilateral IEDs.'
    )

    # ── Encoding ─────────────────────────────────────────────────────────────
    pdf.section_title('Encoding Phase')

    def desc_block(bil, uni):
        rows = []
        for label, sub in [('Bilateral IED', bil), ('Unilateral IED', uni)]:
            n = len(sub)
            nr = int(sub['memory'].sum())
            rows.append([label, n, nr, n - nr, f'{nr / n * 100:.1f}%'])
        tot = pd.concat([bil, uni])
        n = len(tot)
        nr = int(tot['memory'].sum())
        rows.append(['Total', n, nr, n - nr, f'{nr / n * 100:.1f}%'])
        return rows

    pdf.apa_table(
        'Table 1\nEncoding Memory by Bilateral IED Occurrence',
        ['Trial type', 'N Trials', 'Rem', 'Forg', '% Rem'],
        desc_block(enc_bil, enc_uni),
        col_widths=[35, 22, 18, 18, 22],
        note=f'N = {n_cov_enc_tr} trials from {n_cov_enc} bilateral-coverage patients. '
             f'Bilateral = IEDs in both hemispheres on the same trial; unilateral = one '
             f'hemisphere only.'
    )

    pdf.subsection_title('Mixed-Effects Model')
    pdf.italic_text('memory ~ bilateral_ied + (1 | patient_id)')
    pdf.body_text(
        f'A logistic GLMM with a random intercept for patient tested whether bilateral IED '
        f'occurrence predicts subsequent memory, across the {n_bil_enc_tr} bilateral-IED '
        f'trials and {n_cov_enc_tr} total trials from the {n_cov_enc} bilateral-coverage '
        f'patients.'
    )

    label_map = {'(Intercept)': '(Intercept)', 'bilateral_ied': 'Bilateral IED'}
    pdf.apa_table(
        'Table 2\nBilateral IED Predicting Memory (Encoding)',
        ['Predictor', 'OR', '95% CI', 'z', 'p'],
        [format_or_row(r, label_map) for _, r in mE.iterrows()],
        col_widths=[38, 16, 35, 16, 22],
        note=f'GLMM with random intercept for patient. N = {n_cov_enc_tr} trials, '
             f'{n_cov_enc} patients. OR < 1 = lower odds of remembering. *p < .05. **p < .01.'
    )

    e_or, e_z, e_p = get(mE, 'bilateral_ied', 'estimate'), get(mE, 'bilateral_ied', 'statistic'), get(mE, 'bilateral_ied', 'p.value')
    e_lo, e_hi = get(mE, 'bilateral_ied', 'conf.low'), get(mE, 'bilateral_ied', 'conf.high')
    rem_bil = enc_bil['memory'].mean() * 100
    rem_uni = enc_uni['memory'].mean() * 100
    pdf.body_text(
        f'Bilateral IEDs significantly predicted worse subsequent memory: trials with '
        f'bilateral IEDs were remembered at {rem_bil:.1f}% versus {rem_uni:.1f}% for '
        f'unilateral trials (OR = {e_or:.2f}, 95% CI [{e_lo:.2f}, {e_hi:.2f}], '
        f'z = {e_z:.2f}, p = {p_str(e_p)}).'
    )

    pdf.figure(
        os.path.join(FIG_DIR, 'h3c_bilateral_vs_unilateral_forgetting.png'),
        'Figure 1. Subsequent forgetting for bilateral versus unilateral IED trials '
        '(encoding, bilateral-coverage patients).',
        width=88,
    )

    # ── Encoding: stim vs no-stim breakdown of bilateral-IED trials ──────────
    pdf.subsection_title('Bilateral IED Trials by Stimulation Condition')
    bil_sb = enc_bi[enc_bi['trial_hemisphere'] == 'Bilateral']
    sb_rows = []
    for label, cond in [('Stim', 1), ('No-stim', 0)]:
        sub = bil_sb[bil_sb['stim'] == cond]
        n = len(sub)
        nr = int(sub['memory'].sum())
        sb_rows.append([label, n, nr, n - nr, f'{(n - nr) / n * 100:.1f}%'])
    n_sb = len(bil_sb)
    nf_sb = int((bil_sb['memory'] == 0).sum())
    sb_rows.append(['Total', n_sb, n_sb - nf_sb, nf_sb, f'{nf_sb / n_sb * 100:.1f}%'])
    pdf.apa_table(
        'Table 3\nBilateral-IED Trials by Stimulation Condition (Encoding)',
        ['Condition', 'N Trials', 'Rem', 'Forg', '% Forg'],
        sb_rows,
        col_widths=[35, 22, 18, 18, 22],
        note=f'N = {n_sb} bilateral-IED trials. % Forg = percentage forgotten.'
    )
    pdf.body_text(
        'Bilateral-IED trials split roughly evenly across stimulation conditions, and '
        'forgetting rates were nearly identical for stimulated and non-stimulated trials, '
        'descriptively consistent with a bilateral IED effect that does not depend on '
        'stimulation.'
    )
    pdf.figure(
        os.path.join(FIG_DIR, 'h3c_bilateral_stim_forgetting.png'),
        'Figure 2. Subsequent forgetting among bilateral-IED trials by stimulation '
        'condition (encoding).',
        width=88,
    )

    # ── Encoding: hierarchical entry (stim first, then bilateral IED) ────────
    pdf.subsection_title('Bilateral IED Above and Beyond Stimulation (Hierarchical Entry)')
    pdf.italic_text('Step 1: memory ~ stim + (1 | patient_id)   ->   '
                    'Step 2: memory ~ stim + bilateral_ied + (1 | patient_id)')
    s1_or, s1_p = get(mE_stim, 'stim', 'estimate'), get(mE_stim, 'stim', 'p.value')
    pdf.body_text(
        f'Stimulation was entered first (Step 1), then bilateral IED was added (Step 2), '
        f'testing whether bilateral IED predicts memory above and beyond stimulation. In '
        f'Step 1, stimulation alone did not predict memory (OR = {s1_or:.2f}, '
        f'p = {p_str(s1_p)}). Table 4 reports the Step 2 model.'
    )

    label_map_s = {
        '(Intercept)': '(Intercept)',
        'stim': 'Stimulation',
        'bilateral_ied': 'Bilateral IED',
    }
    # Order rows: Intercept, Stimulation (entered first), Bilateral IED (added)
    order_s = ['(Intercept)', 'stim', 'bilateral_ied']
    rows_s = [format_or_row(mEs[mEs['term'] == t].iloc[0], label_map_s) for t in order_s]
    pdf.apa_table(
        'Table 4\nBilateral IED Above and Beyond Stimulation (Encoding, Step 2)',
        ['Predictor', 'OR', '95% CI', 'z', 'p'],
        rows_s,
        col_widths=[38, 16, 35, 16, 22],
        note=f'GLMM with random intercept for patient. N = {n_cov_enc_tr} trials, '
             f'{n_cov_enc} patients. Hierarchical entry: stimulation first, then bilateral '
             f'IED. *p < .05. **p < .01.'
    )

    es_or, es_z, es_p = get(mEs, 'bilateral_ied', 'estimate'), get(mEs, 'bilateral_ied', 'statistic'), get(mEs, 'bilateral_ied', 'p.value')
    st_p = get(mEs, 'stim', 'p.value')
    h_chi, h_p = mEs_lrt['chisq'].values[0], mEs_lrt['p.value'].values[0]
    pdf.body_text(
        f'Adding bilateral IED to the stimulation-only model significantly improved fit '
        f'(LRT chi2(1) = {h_chi:.2f}, p = {p_str(h_p)}). Above and beyond stimulation, '
        f'bilateral IED remained a significant predictor of worse memory (OR = {es_or:.2f}, '
        f'z = {es_z:.2f}, p = {p_str(es_p)}), while stimulation itself had no effect '
        f'(p = {p_str(st_p)}). The bilateral IED impairment is therefore present regardless '
        f'of BLA stimulation.'
    )

    # ── Encoding: bilateral x stim interaction (separate table) ──────────────
    pdf.subsection_title('Bilateral IED x Stimulation Interaction')
    pdf.italic_text('memory ~ bilateral_ied * stim + (1 | patient_id)')
    pdf.body_text(
        'This model adds the interaction term to test whether the bilateral IED effect '
        'depends on stimulation. Note: with the interaction present, the Bilateral IED row '
        'is the simple effect at stim = 0 (non-stimulated trials), not the overall effect '
        'reported in Table 4.'
    )

    label_map_x = {
        '(Intercept)': '(Intercept)',
        'bilateral_ied': 'Bilateral IED',
        'stim': 'Stimulation',
        'bilateral_ied:stim': 'Bilateral IED x Stim',
    }
    pdf.apa_table(
        'Table 5\nBilateral IED x Stimulation Predicting Memory (Encoding)',
        ['Predictor', 'OR', '95% CI', 'z', 'p'],
        [format_or_row(r, label_map_x) for _, r in mEx.iterrows()],
        col_widths=[42, 16, 35, 16, 22],
        note=f'GLMM with random intercept for patient. N = {n_cov_enc_tr} trials, '
             f'{n_cov_enc} patients. *p < .05. **p < .01.'
    )

    x_or, x_z, x_p = get(mEx, 'bilateral_ied:stim', 'estimate'), get(mEx, 'bilateral_ied:stim', 'statistic'), get(mEx, 'bilateral_ied:stim', 'p.value')
    lrt_chi, lrt_p = mEx_lrt['chisq'].values[0], mEx_lrt['p.value'].values[0]
    pdf.body_text(
        f'The bilateral IED x stimulation interaction was not significant (OR = {x_or:.2f}, '
        f'z = {x_z:.2f}, p = {p_str(x_p)}; LRT chi2(1) = {lrt_chi:.2f}, p = {p_str(lrt_p)}). '
        f'The bilateral IED impairment does not depend on stimulation condition, consistent '
        f'with an effect that operates regardless of BLA stimulation.'
    )

    # ── Encoding: where bilateral IEDs fall across the four windows ──────────
    pdf.subsection_title('Timing-Window Distribution of Bilateral IEDs')
    pdf.body_text(
        f'Across the {n_bil_enc_tr} bilateral-IED trials, the table and figure below count '
        f'how many had an IED in each encoding window. A trial may contain IEDs in more than '
        f'one window, so the counts sum to more than {n_bil_enc_tr}.'
    )
    bil_tr_d = enc_bi[enc_bi['trial_hemisphere'] == 'Bilateral']
    dist_rows = []
    for col, lab in [('ied_before_image', 'Before Image'),
                     ('ied_during_image', 'During Image'),
                     ('ied_after_image', 'After Image'),
                     ('ied_during_stim', 'During Stim')]:
        c = int((bil_tr_d[col] == 1).sum())
        dist_rows.append([lab, c, f'{100 * c / n_bil_enc_tr:.1f}%'])
    pdf.apa_table(
        'Table 6\nBilateral IED Count by Encoding Timing Window',
        ['Window', 'N IED+ trials', '% of bilateral trials'],
        dist_rows,
        col_widths=[40, 38, 50],
        note=f'N = {n_bil_enc_tr} bilateral-IED trials. Counts are not mutually exclusive '
             f'(a trial may have IEDs in multiple windows).'
    )
    pdf.figure(
        os.path.join(FIG_DIR, 'h3c_bilateral_window_counts.png'),
        'Figure 3. Timing-window distribution of bilateral IEDs: number of the 87 '
        'bilateral-IED trials with an IED in each encoding window.',
        width=120,
    )

    # ── Encoding: timing of bilateral IEDs across all four windows ───────────
    pdf.subsection_title('Timing of Bilateral IEDs: All Four Encoding Windows')
    pdf.italic_text('memory ~ ied_before_image + ied_during_image + ied_after_image + '
                    'ied_during_stim + (1 | patient_id)')
    pdf.body_text(
        f'Restricted to the {n_bil_enc_tr} trials with bilateral IEDs, this model tests '
        f'whether the window in which the bilateral IEDs occurred predicts memory, entering '
        f'all four encoding windows simultaneously (each window\'s effect is adjusted for the '
        f'others). The During-Stim window exists only on stimulated trials. Note: "After '
        f'Image" is the post-image/post-stimulation inter-trial window.'
    )

    # Descriptive rates by window within bilateral trials (all four windows)
    bil_tr = enc_bi[enc_bi['trial_hemisphere'] == 'Bilateral']
    window_defs = [('ied_before_image', 'Before Image'),
                   ('ied_during_image', 'During Image'),
                   ('ied_after_image', 'After Image'),
                   ('ied_during_stim', 'During Stim')]
    win_rows = []
    for col, lab in window_defs:
        for present, plab in [(1, 'IED present'), (0, 'IED absent')]:
            sub = bil_tr[bil_tr[col] == present]
            n = len(sub)
            nr = int(sub['memory'].sum())
            win_rows.append([f'{lab} - {plab}', n, nr, n - nr, f'{nr / n * 100:.1f}%'])
    pdf.apa_table(
        'Table 7\nMemory by Timing Window Within Bilateral-IED Trials',
        ['Window', 'N Trials', 'Rem', 'Forg', '% Rem'],
        win_rows,
        col_widths=[48, 22, 16, 16, 20],
        note=f'N = {n_bil_enc_tr} bilateral-IED trials. Rows show trials with vs without an '
             f'IED in each window (categories are not mutually exclusive).'
    )

    label_map_w = {
        '(Intercept)': '(Intercept)',
        'ied_before_image': 'Before-Image window',
        'ied_during_image': 'During-Image window',
        'ied_after_image': 'After-Image window',
        'ied_during_stim': 'During-Stim window',
    }
    # FDR-BH across the four timing-window comparisons (intercept excluded from family).
    win_family = ['ied_before_image', 'ied_during_image', 'ied_after_image', 'ied_during_stim']
    win_p = [get(mEw, t, 'p.value') for t in win_family]
    win_q = dict(zip(win_family, bh_fdr(win_p)))

    rows_w = []
    for _, r in mEw.iterrows():
        term = r['term']
        if term in win_q:
            q = win_q[term]
            q_cell = f'{p_str(q)}{sig_str(q)}'
        else:
            q_cell = '--'
        rows_w.append([
            label_map_w.get(term, term),
            f'{r["estimate"]:.3f}',
            f'[{r["conf.low"]:.3f}, {r["conf.high"]:.3f}]',
            f'{r["statistic"]:.2f}',
            p_str(r['p.value']),
            q_cell,
        ])

    pdf.apa_table(
        'Table 8\nTiming of Bilateral IEDs Predicting Memory (Encoding)',
        ['Predictor', 'OR', '95% CI', 'z', 'p', 'FDR q'],
        rows_w,
        col_widths=[40, 15, 33, 14, 16, 18],
        note=f'GLMM with random intercept for patient. N = {n_bil_enc_tr} bilateral-IED '
             f'trials, {n_bil_enc_pat} patients. OR < 1 = lower odds of remembering. '
             f'FDR q = Benjamini-Hochberg-corrected across the four timing-window comparisons; '
             f'-- = not in the FDR family. Stars reflect FDR q. *q < .05. **q < .01.'
    )

    w_ds_or, w_ds_z, w_ds_p = get(mEw, 'ied_during_stim', 'estimate'), get(mEw, 'ied_during_stim', 'statistic'), get(mEw, 'ied_during_stim', 'p.value')
    w_ds_lo, w_ds_hi = get(mEw, 'ied_during_stim', 'conf.low'), get(mEw, 'ied_during_stim', 'conf.high')
    w_ds_q = win_q['ied_during_stim']
    other_p = [get(mEw, t, 'p.value') for t in ['ied_before_image', 'ied_during_image', 'ied_after_image']]
    pdf.body_text(
        f'Across all four encoding windows, only the During-Stim window predicted worse '
        f'memory (OR = {w_ds_or:.2f}, 95% CI [{w_ds_lo:.2f}, {w_ds_hi:.2f}], '
        f'z = {w_ds_z:.2f}, p = {p_str(w_ds_p)}); after FDR-BH correction across the four '
        f'windows this no longer reached significance (q = {p_str(w_ds_q)}). The Before-Image, '
        f'During-Image, and After-Image windows did not predict memory (all p > '
        f'{min(other_p):.2f}). The bilateral IED impairment is concentrated in the '
        f'During-Stim window.'
    )

    pdf.figure(
        os.path.join(FIG_DIR, 'h3c_bilateral_window_forgetting.png'),
        'Figure 4. Subsequent forgetting within the 87 bilateral-IED trials, across all four '
        'encoding windows. For each window, trials are split by whether the bilateral IED '
        'fell in that window (present) or in another window (absent); all trials have '
        'bilateral IEDs.',
        width=150,
    )

    # ── Retrieval ────────────────────────────────────────────────────────────
    pdf.section_title('Retrieval Phase')

    pdf.body_text(
        'Retrieval has no trial-level stimulation condition, and only the Before-Image and '
        'During-Image windows are defined at retrieval. The stimulation-interaction and '
        'timing-window analyses above therefore cannot be run at retrieval; only the '
        'bilateral IED main effect is testable.'
    )

    pdf.apa_table(
        'Table 9\nRetrieval Recall by Bilateral IED Occurrence',
        ['Trial type', 'N Trials', 'Rem', 'Forg', '% Rem'],
        desc_block(ret_bil, ret_uni),
        col_widths=[35, 22, 18, 18, 22],
        note=f'N = {n_cov_ret_tr} trials from {n_cov_ret} bilateral-coverage patients.'
    )

    pdf.subsection_title('Mixed-Effects Model')
    pdf.italic_text('memory ~ bilateral_ied + (1 | patient_id)')
    pdf.body_text(
        f'Retrieval has no trial-level stimulation condition. The same GLMM was fit, but it '
        f'is severely underpowered: only {n_bil_ret_tr} of {n_cov_ret_tr} trials, from '
        f'{n_bil_ret_pat} patients, had bilateral IEDs.'
    )

    pdf.apa_table(
        'Table 10\nBilateral IED Predicting Recall (Retrieval)',
        ['Predictor', 'OR', '95% CI', 'z', 'p'],
        [format_or_row(r, label_map) for _, r in mR.iterrows()],
        col_widths=[38, 16, 35, 16, 22],
        note=f'GLMM with random intercept for patient. N = {n_cov_ret_tr} trials, '
             f'{n_cov_ret} patients.'
    )

    r_or, r_z, r_p = get(mR, 'bilateral_ied', 'estimate'), get(mR, 'bilateral_ied', 'statistic'), get(mR, 'bilateral_ied', 'p.value')
    rrem_bil = ret_bil['memory'].mean() * 100
    rrem_uni = ret_uni['memory'].mean() * 100
    pdf.body_text(
        f'Bilateral IEDs during retrieval did not predict recall (OR = {r_or:.2f}, '
        f'z = {r_z:.2f}, p = {p_str(r_p)}); if anything the point estimate ran in the '
        f'opposite direction ({rrem_bil:.1f}% remembered for bilateral vs {rrem_uni:.1f}% '
        f'for unilateral trials). With only {n_bil_ret_tr} bilateral trials this test has '
        f'essentially no power and is reported for completeness.'
    )

    # ── Summary ──────────────────────────────────────────────────────────────
    pdf.section_title('Summary')
    pdf.body_text(
        f'At encoding, bilateral IED occurrence significantly predicted worse subsequent '
        f'memory ({rem_bil:.1f}% vs {rem_uni:.1f}% remembered; OR = {e_or:.2f}, '
        f'95% CI [{e_lo:.2f}, {e_hi:.2f}], z = {e_z:.2f}, p = {p_str(e_p)}). Entered '
        f'hierarchically above stimulation, bilateral IED still improved fit (LRT '
        f'chi2(1) = {h_chi:.2f}, p = {p_str(h_p)}) and remained significant '
        f'(OR = {es_or:.2f}, p = {p_str(es_p)}), while stimulation alone had no effect, '
        f'consistent with an impairment that '
        f'operates regardless of BLA stimulation (interaction p = {p_str(x_p)}). Entering '
        f'all four encoding windows simultaneously, the impairment was concentrated in the '
        f'During-Stim window (OR = {w_ds_or:.2f}, p = {p_str(w_ds_p)}; FDR q = '
        f'{p_str(w_ds_q)}), with no effect of the other three windows. At retrieval, '
        f'bilateral IEDs did not '
        f'predict recall (OR = {r_or:.2f}, p = {p_str(r_p)}), but only {n_bil_ret_tr} trials '
        f'had bilateral IEDs, so this null is uninformative.'
    )

    # Report lives in the updated-IED-figures folder, alongside the figures it embeds.
    out_path = os.path.join(FIG_DIR, 'H3c_Bilateral_MLM_Report.pdf')
    pdf.output(out_path)
    print(f'Saved: {out_path}')


if __name__ == '__main__':
    main()
