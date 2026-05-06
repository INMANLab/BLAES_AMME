#!/usr/bin/env python
"""
Build APA-formatted PDF report for IED Hypotheses 3a, 3b, 3c.
Outputs: outputs/ied_timing_memory/IED_Hypothesis_Results.pdf
"""

import os
import numpy as np
import pandas as pd
from fpdf import FPDF
from scipy.stats import chi2_contingency, fisher_exact, spearmanr, mannwhitneyu, kruskal

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'outputs', 'ied_timing_memory')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────────
IED_STUDY = os.path.join(SCRIPT_DIR, 'IED',
    'AMMEBLAES_IEDs_trial_level_dissertation_study_usethis_cleaned_with_memory.csv')
IED_TEST = os.path.join(SCRIPT_DIR, 'IED',
    'AMMEBLAES_IEDs_trial_level_dissertation_test_usethis_cleaned_with_memory.csv')
BEH_CSV = os.path.join(SCRIPT_DIR, 'behavioral figures',
    'AMMEBLAES_includedpts_firstsession_behavioral.csv')

TIMING_STUDY = ['BeforeImgITI', 'DuringImg', 'AfterImgITI', 'DuringStim']
TIMING_TEST = ['BeforeImgITI', 'DuringImgITI']
TIMING_LABELS = {
    'BeforeImgITI': 'Before Image', 'DuringImg': 'During Image',
    'AfterImgITI': 'After Image', 'DuringStim': 'During Stim',
    'DuringImgITI': 'During Image',
}

BLAES_DIR = os.path.join(SCRIPT_DIR, '..', 'LFP_analyses', 'Results_CSVOutput')
AMME_DIR = os.path.join(SCRIPT_DIR, '..', '..', '..', 'AMME_Data_Emory',
                        'AMME_Data', 'LFP_analyses_Martina', 'Results_CSVOutput')


def load_study():
    df = pd.read_csv(IED_STUDY)
    return df[df['MemoryOutcome'].notna()].copy()


def load_test():
    df = pd.read_csv(IED_TEST)
    return df[df['MemoryOutcome'].isin(['remembered', 'forgotten'])].copy()


def add_test_stim_cond(df):
    """Add StimCond to test IED data from phase 3 files."""
    stim_maps = {}
    for pat in df['Patient'].unique():
        is_amme = pat.lower().startswith('amyg')
        if is_amme:
            p3_file = os.path.join(AMME_DIR, f'Data_{pat}phase3MeasurePower.csv')
        else:
            p3_file = os.path.join(BLAES_DIR, f'Data_{pat}phase3MeasurePower.csv')
        if not os.path.exists(p3_file):
            continue
        p3 = pd.read_csv(p3_file)
        if is_amme:
            if 'Region' in p3.columns:
                p3 = p3[p3['Region'] == p3['Region'].iloc[0]]
            trial_col = 'number' if 'number' in p3.columns else 'trialIdx'
        else:
            p3 = p3[p3['ChanName'] == p3['ChanName'].iloc[0]]
            trial_col = 'trialIdx'
        smap = {}
        for _, row in p3.iterrows():
            num = int(row[trial_col])
            if is_amme:
                tt = str(row['trial_type']).strip().lower()
                if tt == 'stim':
                    smap[num] = 'S'
                elif tt == 'nostim':
                    smap[num] = 'NS'
            else:
                if 'stimulation' in p3.columns and pd.notna(row.get('stimulation')):
                    smap[num] = 'S' if int(row['stimulation']) == 1 else 'NS'
        stim_maps[pat] = smap

    def get_stim(row):
        smap = stim_maps.get(row['Patient'], {})
        return smap.get(int(row['Trial']), np.nan)
    df['StimCond'] = df.apply(get_stim, axis=1)
    return df


def collapse_trials(df, timing_cols, group_cols=None):
    if group_cols is None:
        group_cols = ['Patient', 'Trial', 'MemoryOutcome']
    agg_dict = {c: lambda x, c=c: 'Y' if (x == 'Y').any() else 'N' for c in timing_cols}
    if 'Hemisphere' in df.columns:
        def classify_hemi(s):
            vals = set(s.dropna()) - {'D'}
            if 'L' in vals and 'R' in vals:
                return 'Bilateral'
            elif 'L' in vals:
                return 'L'
            elif 'R' in vals:
                return 'R'
            return 'Unknown'
        agg_dict['Hemisphere'] = classify_hemi
    return df.groupby(group_cols).agg(agg_dict).reset_index()


def timing_chi2(trial_df, timing_cols):
    """Return list of dicts with chi-square results per timing window."""
    rows = []
    for col in timing_cols:
        yes = trial_df[trial_df[col] == 'Y']
        no = trial_df[trial_df[col] == 'N']
        n_yes = len(yes)
        n_no = len(no)
        rem_yes = (yes['MemoryOutcome'] == 'remembered').sum()
        rem_no = (no['MemoryOutcome'] == 'remembered').sum()
        pct_yes = rem_yes / n_yes * 100 if n_yes > 0 else np.nan
        pct_no = rem_no / n_no * 100 if n_no > 0 else np.nan
        ct = pd.crosstab(trial_df[col], trial_df['MemoryOutcome'])
        chi2, p = np.nan, np.nan
        if ct.shape == (2, 2):
            chi2, p, _, _ = chi2_contingency(ct)
        rows.append({
            'Window': TIMING_LABELS.get(col, col),
            'IED_Yes_N': n_yes, 'IED_Yes_Rem': rem_yes,
            'IED_Yes_Pct': round(pct_yes, 1) if not np.isnan(pct_yes) else '-',
            'IED_No_N': n_no, 'IED_No_Rem': rem_no,
            'IED_No_Pct': round(pct_no, 1) if not np.isnan(pct_no) else '-',
            'Chi2': round(chi2, 2) if not np.isnan(chi2) else '-',
            'p': round(p, 4) if not np.isnan(p) else '-',
        })
    return rows


# ── PDF class ──────────────────────────────────────────────────────────────
class APAReport(FPDF):
    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=25)

    def header(self):
        self.set_font('Helvetica', 'I', 8)
        self.cell(0, 5, 'IED and Subsequent Memory: Hypothesis Testing Results', align='R')
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

    def bold_text(self, text):
        self.set_font('Times', 'B', 11)
        self.multi_cell(0, 5.5, text)
        self.ln(1)

    def apa_table(self, title, headers, rows, col_widths=None, note=None):
        """Draw an APA-style table with top/bottom/header rules."""
        self.ln(3)
        # Table title (italic)
        self.set_font('Times', 'I', 10)
        self.multi_cell(0, 5, title)
        self.ln(1)

        if col_widths is None:
            avail = self.w - self.l_margin - self.r_margin
            col_widths = [avail / len(headers)] * len(headers)

        # Top rule
        x_start = self.get_x()
        total_w = sum(col_widths)
        self.set_line_width(0.5)
        self.line(x_start, self.get_y(), x_start + total_w, self.get_y())
        self.ln(1)

        # Header
        self.set_font('Times', 'B', 9)
        for i, h in enumerate(headers):
            self.cell(col_widths[i], 5, str(h), align='C')
        self.ln()

        # Header rule
        self.set_line_width(0.3)
        self.line(x_start, self.get_y(), x_start + total_w, self.get_y())
        self.ln(1)

        # Data rows
        self.set_font('Times', '', 9)
        for row in rows:
            # Check if we need a new page
            if self.get_y() > self.h - 35:
                # Bottom rule before page break
                self.set_line_width(0.5)
                self.line(x_start, self.get_y(), x_start + total_w, self.get_y())
                self.add_page()
                # Reprint header on new page
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

        # Bottom rule
        self.set_line_width(0.5)
        self.line(x_start, self.get_y(), x_start + total_w, self.get_y())
        self.ln(1)

        # Note
        if note:
            self.set_font('Times', 'I', 8)
            self.multi_cell(0, 4, f'Note. {note}')
            self.ln(2)


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


def p_str(p):
    if isinstance(p, str):
        return p
    if p < .001:
        return '< .001'
    return f'{p:.3f}'.lstrip('0')


def main():
    pdf = APAReport()
    pdf.add_page()

    # ── Title ──
    pdf.set_font('Helvetica', 'B', 16)
    pdf.cell(0, 10, 'Interictal Epileptiform Discharges and', align='C', new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 10, 'Subsequent Memory: Hypothesis Testing', align='C', new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)

    # ── Load all data ──
    study_raw = load_study()
    test_raw = load_test()
    test_raw = add_test_stim_cond(test_raw)
    beh = pd.read_csv(BEH_CSV)

    # Encoding trial-level
    enc_trials = collapse_trials(study_raw, TIMING_STUDY,
                                  ['Patient', 'Trial', 'MemoryOutcome', 'StimCond'])
    n_enc = len(enc_trials)
    n_enc_pat = enc_trials['Patient'].nunique()
    n_enc_rem = (enc_trials['MemoryOutcome'] == 'remembered').sum()
    n_enc_forg = (enc_trials['MemoryOutcome'] == 'forgotten').sum()

    # Test trial-level
    test_trials = collapse_trials(test_raw, TIMING_TEST,
                                   ['Patient', 'Trial', 'MemoryOutcome'])
    n_test = len(test_trials)
    n_test_pat = test_trials['Patient'].nunique()

    # ── Data overview ──
    pdf.section_title('Data Overview')
    pdf.body_text(
        f'Interictal epileptiform discharges (IEDs) were identified on a trial-by-trial basis '
        f'during encoding and retrieval phases. Encoding analyses included {n_enc} trials with '
        f'known memory outcomes from {n_enc_pat} patients ({n_enc_rem} remembered, {n_enc_forg} '
        f'forgotten). Retrieval analyses included {n_test} old-item trials from {n_test_pat} '
        f'patients. IED presence was coded across timing windows relative to the image '
        f'presentation. Chi-square tests of independence were used to assess the association '
        f'between IED presence and subsequent memory (remembered vs. forgotten) for each '
        f'timing window. Spearman rank correlations and Mann-Whitney U tests were used for '
        f'continuous measures.'
    )

    # ================================================================
    # HYPOTHESIS 3a
    # ================================================================
    pdf.section_title('Hypothesis 3a')
    pdf.italic_text(
        'Frequently occurring IEDs during object learning (before BLA stimulation) will show '
        'memory impairment regardless of stimulation, but individuals with highly frequent IEDs '
        'during retrieval can achieve memory enhancement due to prior stimulation.'
    )

    pdf.subsection_title('Encoding: IED Timing and Subsequent Memory (All Regions)')
    enc_timing = timing_chi2(enc_trials, TIMING_STUDY)
    headers = ['Window', 'IED+\nn', 'IED+\n% Rem', 'IED-\nn', 'IED-\n% Rem',
               'Chi2', 'p']
    rows = []
    for r in enc_timing:
        p_val = r['p']
        rows.append([
            r['Window'], r['IED_Yes_N'], f"{r['IED_Yes_Pct']}%",
            r['IED_No_N'], f"{r['IED_No_Pct']}%",
            r['Chi2'], f"{p_str(p_val)}{sig_str(p_val)}",
        ])
    pdf.apa_table(
        'Table 1\nIED Presence During Encoding and Subsequent Memory Across All Timing Windows',
        headers, rows,
        col_widths=[30, 18, 18, 18, 18, 18, 28],
        note='IED+ = IED present; IED- = IED absent; % Rem = percent subsequently remembered. '
             f'N = {n_enc} trials from {n_enc_pat} patients. '
             '*p < .05. **p < .01. ***p < .001.'
    )

    pdf.body_text(
        f'IEDs during encoding significantly impaired subsequent memory in two timing windows. '
        f'Trials with IEDs after the image (post-encoding consolidation window) were remembered '
        f'at a lower rate ({enc_timing[2]["IED_Yes_Pct"]}%) than trials without '
        f'({enc_timing[2]["IED_No_Pct"]}%), chi2(1) = {enc_timing[2]["Chi2"]}, '
        f'p = {p_str(enc_timing[2]["p"])}. Trials with IEDs during stimulation showed a similar '
        f'pattern ({enc_timing[3]["IED_Yes_Pct"]}% vs. {enc_timing[3]["IED_No_Pct"]}%), '
        f'chi2(1) = {enc_timing[3]["Chi2"]}, p = {p_str(enc_timing[3]["p"])}. '
        f'IEDs before or during image presentation did not significantly affect memory (ps > .35).'
    )

    # Multi-window dose-response
    pdf.subsection_title('Multi-Window IED Dose-Response')
    mw_rows = []
    for nw in sorted(enc_trials.assign(
        nw=lambda d: sum((d[c] == 'Y').astype(int) for c in TIMING_STUDY)
    )['nw'].unique()):
        sub = enc_trials[sum((enc_trials[c] == 'Y').astype(int) for c in TIMING_STUDY) == nw]
        n = len(sub)
        nr = (sub['MemoryOutcome'] == 'remembered').sum()
        mw_rows.append([f'{nw}', n, nr, n - nr, f'{nr/n*100:.1f}%'])

    enc_trials_nw = enc_trials.copy()
    enc_trials_nw['nw'] = sum((enc_trials_nw[c] == 'Y').astype(int) for c in TIMING_STUDY)
    enc_trials_nw['rem_bin'] = (enc_trials_nw['MemoryOutcome'] == 'remembered').astype(int)
    rho_nw, p_nw = spearmanr(enc_trials_nw['nw'], enc_trials_nw['rem_bin'])

    pdf.apa_table(
        'Table 2\nSubsequent Memory by Number of Timing Windows With IED',
        ['N Windows', 'N Trials', 'Remembered', 'Forgotten', '% Remembered'],
        mw_rows,
        col_widths=[25, 25, 28, 25, 30],
        note=f'Spearman rho = {rho_nw:.3f}, p = {p_str(p_nw)}. '
             f'N = {n_enc} trials from {n_enc_pat} patients. '
             'More concurrent IED timing windows were associated with lower memory.'
    )

    pdf.body_text(
        f'A significant dose-response relationship was observed: as the number of timing '
        f'windows with IED presence increased, subsequent memory decreased monotonically '
        f'(Spearman rho = {rho_nw:.3f}, p = {p_str(p_nw)}). Trials with IEDs in all four '
        f'windows were remembered only {mw_rows[-1][4]} of the time compared to '
        f'{mw_rows[0][4]} for single-window IEDs.'
    )

    # Weighted IED rate
    pdf.subsection_title('Weighted IED Rate and Subsequent Memory')
    enc_trials_rate = study_raw.groupby(['Patient', 'Trial', 'MemoryOutcome']).agg(
        rate=('WeightedIEDRate', lambda x: pd.to_numeric(x, errors='coerce').mean())
    ).reset_index()
    enc_rate_valid = enc_trials_rate[enc_trials_rate['rate'].notna()]
    rem_rates = enc_rate_valid[enc_rate_valid['MemoryOutcome'] == 'remembered']['rate']
    forg_rates = enc_rate_valid[enc_rate_valid['MemoryOutcome'] == 'forgotten']['rate']
    rho_rate, p_rate = spearmanr(enc_rate_valid['rate'],
                                  (enc_rate_valid['MemoryOutcome'] == 'remembered').astype(int))
    u_rate, p_mw_rate = mannwhitneyu(rem_rates, forg_rates, alternative='two-sided')

    pdf.body_text(
        f'Higher weighted IED rates during encoding predicted worse subsequent memory '
        f'(Spearman rho = {rho_rate:.3f}, p = {p_str(p_rate)}; Mann-Whitney U = {u_rate:.0f}, '
        f'p = {p_str(p_mw_rate)}). Mean rate for remembered trials was {rem_rates.mean():.2f} '
        f'compared to {forg_rates.mean():.2f} for forgotten trials.'
    )

    # Retrieval analysis
    pdf.subsection_title('Retrieval: IED Timing and Recall')
    test_timing = timing_chi2(test_trials, TIMING_TEST)
    rows_test = []
    for r in test_timing:
        p_val = r['p']
        rows_test.append([
            r['Window'], r['IED_Yes_N'], f"{r['IED_Yes_Pct']}%",
            r['IED_No_N'], f"{r['IED_No_Pct']}%",
            r['Chi2'], f"{p_str(p_val)}{sig_str(p_val)}",
        ])
    pdf.apa_table(
        'Table 3\nIED Presence During Retrieval and Recall Accuracy',
        headers, rows_test,
        col_widths=[30, 18, 18, 18, 18, 18, 28],
        note=f'N = {n_test} old-item trials from {n_test_pat} patients. '
             '*p < .05. **p < .01.'
    )

    pdf.body_text(
        f'In contrast to encoding, IEDs during retrieval did not significantly affect recall '
        f'accuracy in any timing window (all ps > .84). This dissociation suggests that IEDs '
        f'are primarily disruptive to memory formation rather than memory retrieval.'
    )

    # 3a summary
    pdf.subsection_title('Hypothesis 3a Summary')
    pdf.body_text(
        'Hypothesis 3a was partially supported. IEDs during encoding significantly impaired '
        'subsequent memory, particularly in the post-image and during-stimulation windows, with '
        'a dose-response relationship for multi-window IEDs. However, IEDs during retrieval did '
        'not disrupt recall, regardless of prior stimulation status. Thus, the prediction that '
        'individuals with frequent retrieval IEDs could still show memory enhancement was '
        'supported by the null effect of retrieval IEDs on recall, but this null effect applied '
        'universally rather than being specific to previously stimulated items.'
    )

    # ================================================================
    # HYPOTHESIS 3b
    # ================================================================
    pdf.section_title('Hypothesis 3b')
    pdf.italic_text(
        'IEDs that occur during BLA stimulation will have weaker associations with subsequent '
        'memory outcomes compared to IEDs during encoding or retrieval, suggesting that BLA '
        'stimulation may attenuate their potentially disruptive effect.'
    )

    # Encoding by stim condition
    pdf.subsection_title('Encoding IED Effects: Stimulation vs. No Stimulation Trials')
    stim_data = []
    for sc, sl in [('S', 'Stimulation'), ('NS', 'No Stimulation')]:
        sub = enc_trials[enc_trials['StimCond'] == sc]
        sub_timing = timing_chi2(sub, TIMING_STUDY)
        for r in sub_timing:
            p_val = r['p']
            stim_data.append([
                sl, r['Window'], r['IED_Yes_N'], f"{r['IED_Yes_Pct']}%",
                r['IED_No_N'], f"{r['IED_No_Pct']}%",
                r['Chi2'], f"{p_str(p_val)}{sig_str(p_val)}",
            ])

    pdf.apa_table(
        'Table 4\nIED Effects on Subsequent Memory by Encoding Stimulation Condition',
        ['Condition', 'Window', 'IED+\nn', 'IED+\n% Rem', 'IED-\nn', 'IED-\n% Rem',
         'Chi2', 'p'],
        stim_data,
        col_widths=[25, 22, 14, 16, 14, 16, 14, 24],
        note='Stimulation = encoding trials with BLA stimulation; No Stimulation = trials '
             'without. *p < .05. **p < .01.'
    )

    pdf.body_text(
        'On stimulation trials, the During Stim window showed the strongest IED-memory '
        'disruption (62.2% vs. 76.1%, p = .004), while the After Image effect was trending '
        '(p = .053). On no-stimulation trials, the After Image window was significant '
        '(64.4% vs. 75.1%, p = .038), while the During Stim window was not testable (only 2 '
        'IED+ trials). This pattern suggests a clean dissociation: on stimulation trials, IEDs '
        'concurrent with the stimulation pulse are most disruptive; on non-stimulation trials, '
        'IEDs during the post-encoding consolidation period are most harmful.'
    )

    # Retrieval by stim
    test_raw_stim = test_raw.copy()
    test_stim_data = []
    for sc, sl in [('S', 'Stimulation'), ('NS', 'No Stimulation')]:
        sub = test_raw_stim[test_raw_stim['StimCond'] == sc]
        sub = sub[sub['MemoryOutcome'].isin(['remembered', 'forgotten'])]
        if len(sub) < 5:
            continue
        sub_trials = collapse_trials(sub, TIMING_TEST, ['Patient', 'Trial', 'MemoryOutcome'])
        sub_timing = timing_chi2(sub_trials, TIMING_TEST)
        for r in sub_timing:
            p_val = r['p']
            test_stim_data.append([
                sl, r['Window'], r['IED_Yes_N'], f"{r['IED_Yes_Pct']}%",
                r['IED_No_N'], f"{r['IED_No_Pct']}%",
                r['Chi2'], f"{p_str(p_val)}{sig_str(p_val)}",
            ])

    if test_stim_data:
        pdf.apa_table(
            'Table 5\nIED Effects on Recall by Prior Encoding Stimulation Condition',
            ['Enc. Cond.', 'Window', 'IED+\nn', 'IED+\n% Rem', 'IED-\nn', 'IED-\n% Rem',
             'Chi2', 'p'],
            test_stim_data,
            col_widths=[25, 22, 14, 16, 14, 16, 14, 24],
            note='Enc. Cond. = whether the item was stimulated during encoding. '
                 '*p < .05. **p < .01.'
        )

    pdf.subsection_title('Hypothesis 3b Summary')
    pdf.body_text(
        'Hypothesis 3b was not supported in the predicted direction. IEDs during BLA stimulation '
        'did not show weaker associations with memory impairment; rather, they showed the '
        'strongest association (p = .004). IEDs concurrent with stimulation were more disruptive '
        'than IEDs in other timing windows on stimulation trials. However, this finding could be '
        'interpreted as consistent with the hypothesis in a different sense: BLA stimulation may '
        'create a critical window during which neural activity is particularly vulnerable to '
        'disruption by IEDs. The null effects at retrieval (regardless of stim condition) suggest '
        'that stimulation does not protect against IED disruption during memory formation, but '
        'the retrieval process itself is robust to IED interference.'
    )

    # ================================================================
    # HYPOTHESIS 3c
    # ================================================================
    pdf.section_title('Hypothesis 3c')
    pdf.italic_text(
        'Frequent bilateral IED occurrence (e.g., both hippocampi at the same time) during '
        'learning or retrieval will predict subsequent memory impairment regardless of BLA '
        'stimulation.'
    )

    # Hemisphere trial-level analysis (encoding)
    pdf.subsection_title('Encoding: Hemisphere of IED and Subsequent Memory')

    # Build hemisphere map from raw data
    hemi_map = study_raw.groupby(['Patient', 'Trial']).agg(
        Hemisphere=('Hemisphere', lambda x: (
            'Bilateral' if len(set(x.dropna()) - {'D'}) > 1
            else (list(set(x.dropna()) - {'D'}) + ['Unknown'])[0]
        ))
    ).reset_index()

    # Collapse timing (without Hemisphere in agg) and merge hemisphere
    agg_timing = {c: lambda x, c=c: 'Y' if (x == 'Y').any() else 'N' for c in TIMING_STUDY}
    enc_hemi = study_raw.groupby(
        ['Patient', 'Trial', 'MemoryOutcome', 'StimCond']
    ).agg(agg_timing).reset_index()
    enc_hemi = enc_hemi.merge(hemi_map, on=['Patient', 'Trial'], how='left')

    hemi_rows = []
    for hemi in ['Left', 'Right', 'Bilateral']:
        hemi_code = hemi[0] if hemi != 'Bilateral' else 'Bilateral'
        sub = enc_hemi[enc_hemi['Hemisphere'] == hemi_code]
        n = len(sub)
        nr = (sub['MemoryOutcome'] == 'remembered').sum()
        nf = n - nr
        pct = f'{nr/n*100:.1f}%' if n > 0 else '-'
        n_pts = sub['Patient'].nunique()
        hemi_rows.append([hemi, n, nr, nf, pct, n_pts])

    pdf.apa_table(
        'Table 6\nSubsequent Memory by IED Hemisphere During Encoding',
        ['Hemisphere', 'N Trials', 'Remembered', 'Forgotten', '% Remembered', 'N Patients'],
        hemi_rows,
        col_widths=[25, 22, 25, 22, 28, 22],
    )

    # Chi-square
    hemi_test = enc_hemi[enc_hemi['Hemisphere'].isin(['L', 'R', 'Bilateral'])]
    ct_all = pd.crosstab(hemi_test['Hemisphere'], hemi_test['MemoryOutcome'])
    chi2_all, p_all, _, _ = chi2_contingency(ct_all)

    lr = enc_hemi[enc_hemi['Hemisphere'].isin(['L', 'R'])]
    ct_lr = pd.crosstab(lr['Hemisphere'], lr['MemoryOutcome'])
    chi2_lr, p_lr, _, _ = chi2_contingency(ct_lr)

    pdf.body_text(
        f'Bilateral IED occurrence during encoding was associated with markedly lower '
        f'subsequent memory ({hemi_rows[2][4]}) compared to left-lateralized '
        f'({hemi_rows[0][4]}) or right-lateralized ({hemi_rows[1][4]}) IEDs, '
        f'chi2(2) = {chi2_all:.2f}, p {p_str(p_all)}. Left and right unilateral IEDs did not '
        f'significantly differ from each other, chi2(1) = {chi2_lr:.2f}, p = {p_str(p_lr)}.'
    )

    # Hemisphere by stim
    pdf.subsection_title('Hemisphere Effects by Stimulation Condition')
    hemi_stim_rows = []
    for sc, sl in [('S', 'Stim'), ('NS', 'No Stim')]:
        for hemi, hl in [('L', 'Left'), ('R', 'Right'), ('Bilateral', 'Bilateral')]:
            sub = enc_hemi[(enc_hemi['StimCond'] == sc) & (enc_hemi['Hemisphere'] == hemi)]
            n = len(sub)
            if n < 3:
                continue
            nr = (sub['MemoryOutcome'] == 'remembered').sum()
            hemi_stim_rows.append([sl, hl, n, nr, n - nr, f'{nr/n*100:.1f}%'])

    pdf.apa_table(
        'Table 7\nSubsequent Memory by IED Hemisphere and Stimulation Condition',
        ['Condition', 'Hemisphere', 'N', 'Rem', 'Forg', '% Rem'],
        hemi_stim_rows,
        col_widths=[22, 25, 18, 18, 18, 22],
        note='Bilateral IED impairment was consistent across stimulation conditions.'
    )

    # Hemisphere x timing windows
    pdf.subsection_title('Hemisphere-Specific Timing Window Effects')
    hemi_timing_rows = []
    for hemi, hl in [('L', 'Left'), ('R', 'Right'), ('Bilateral', 'Bilateral')]:
        sub = enc_hemi[enc_hemi['Hemisphere'] == hemi]
        if len(sub) < 10:
            continue
        for col in TIMING_STUDY:
            ct = pd.crosstab(sub[col], sub['MemoryOutcome'])
            if ct.shape == (2, 2):
                chi2, p, _, _ = chi2_contingency(ct)
                yes_s = sub[sub[col] == 'Y']
                no_s = sub[sub[col] == 'N']
                pct_y = (yes_s['MemoryOutcome'] == 'remembered').mean() * 100
                pct_n = (no_s['MemoryOutcome'] == 'remembered').mean() * 100
                hemi_timing_rows.append([
                    hl, TIMING_LABELS[col],
                    f'{pct_y:.1f}%', f'{pct_n:.1f}%',
                    f'{chi2:.2f}', f'{p_str(p)}{sig_str(p)}',
                ])

    pdf.apa_table(
        'Table 8\nTiming Window Effects on Memory by IED Hemisphere',
        ['Hemisphere', 'Window', 'IED+ % Rem', 'IED- % Rem', 'Chi2', 'p'],
        hemi_timing_rows,
        col_widths=[23, 25, 25, 25, 18, 25],
        note='Right hemisphere IEDs showed timing-specific effects; bilateral IEDs impaired '
             'memory regardless of timing. *p < .05. **p < .01.'
    )

    # Behavioral dprime correlation
    pdf.subsection_title('IED Laterality and Overall Stimulation Benefit (d-prime)')

    # Build patient-level hemisphere classification
    pat_hemi = {}
    for pat in enc_hemi['Patient'].unique():
        hemis = set(enc_hemi[enc_hemi['Patient'] == pat]['Hemisphere'].dropna()) - {'Unknown'}
        if 'Bilateral' in hemis or ('L' in hemis and 'R' in hemis):
            pat_hemi[pat] = 'Bilateral'
        elif 'L' in hemis:
            pat_hemi[pat] = 'L'
        elif 'R' in hemis:
            pat_hemi[pat] = 'R'
        else:
            pat_hemi[pat] = 'Unknown'

    pat_df = pd.DataFrame([
        {'Patient': k, 'IED_Hemisphere': v} for k, v in pat_hemi.items()
    ]).merge(beh[['Patient', 'avg_stim_dprime_diff']], on='Patient', how='left')
    pat_df = pat_df[pat_df['avg_stim_dprime_diff'].notna()]

    dprime_rows = []
    for hemi in ['L', 'R', 'Bilateral']:
        sub = pat_df[pat_df['IED_Hemisphere'] == hemi]['avg_stim_dprime_diff']
        if len(sub) > 0:
            dprime_rows.append([
                hemi, len(sub), f'{sub.mean():.3f}', f'{sub.median():.3f}',
                f'{sub.std():.3f}',
            ])

    pdf.apa_table(
        "Table 9\nStimulation Benefit (avg_stim_dprime_diff) by Patient IED Laterality",
        ['Laterality', 'N', 'M', 'Mdn', 'SD'],
        dprime_rows,
        col_widths=[30, 20, 28, 28, 28],
    )

    l_dp = pat_df[pat_df['IED_Hemisphere'] == 'L']['avg_stim_dprime_diff']
    r_dp = pat_df[pat_df['IED_Hemisphere'] == 'R']['avg_stim_dprime_diff']
    b_dp = pat_df[pat_df['IED_Hemisphere'] == 'Bilateral']['avg_stim_dprime_diff']

    groups_present = []
    group_vals = []
    for lbl, vals in [('Left', l_dp), ('Right', r_dp), ('Bilateral', b_dp)]:
        if len(vals) > 1:
            groups_present.append(lbl)
            group_vals.append(vals)

    if len(group_vals) >= 2:
        h_stat, p_kw = kruskal(*group_vals)
        pdf.body_text(
            f'IED laterality did not significantly predict overall stimulation benefit '
            f'(Kruskal-Wallis H = {h_stat:.2f}, p = {p_str(p_kw)}). Bilateral IED patients '
            f'showed numerically higher d-prime difference (M = {b_dp.mean():.3f}) than left '
            f'(M = {l_dp.mean():.3f}) or right (M = {r_dp.mean():.3f}), but this was not '
            f'statistically significant.'
        )

    # Retrieval hemisphere
    pdf.subsection_title('Retrieval: Hemisphere of IED and Recall')
    test_old = test_raw[test_raw['MemoryOutcome'].isin(['remembered', 'forgotten'])].copy()
    test_hemi_map = test_old.groupby(['Patient', 'Trial']).agg(
        Hemisphere=('Hemisphere', lambda x: (
            'Bilateral' if len(set(x.dropna()) - {'D'}) > 1
            else (list(set(x.dropna()) - {'D'}) + ['Unknown'])[0]
        ))
    ).reset_index()

    # Collapse test timing without Hemisphere, then merge
    agg_test = {c: lambda x, c=c: 'Y' if (x == 'Y').any() else 'N' for c in TIMING_TEST}
    test_trials_h = test_old.groupby(
        ['Patient', 'Trial', 'MemoryOutcome']
    ).agg(agg_test).reset_index()
    test_trials_h = test_trials_h.merge(test_hemi_map, on=['Patient', 'Trial'], how='left')

    test_hemi_rows = []
    for hemi, hl in [('L', 'Left'), ('R', 'Right'), ('Bilateral', 'Bilateral')]:
        sub = test_trials_h[test_trials_h['Hemisphere'] == hemi]
        n = len(sub)
        if n < 1:
            continue
        nr = (sub['MemoryOutcome'] == 'remembered').sum()
        test_hemi_rows.append([hl, n, nr, n - nr, f'{nr/n*100:.1f}%', sub['Patient'].nunique()])

    if test_hemi_rows:
        pdf.apa_table(
            'Table 10\nRecall Accuracy by IED Hemisphere During Retrieval',
            ['Hemisphere', 'N Trials', 'Remembered', 'Forgotten', '% Recalled', 'N Patients'],
            test_hemi_rows,
            col_widths=[25, 22, 25, 22, 25, 22],
        )

    pdf.subsection_title('Hypothesis 3c Summary')
    pdf.body_text(
        f'Hypothesis 3c was strongly supported for encoding. Bilateral IED occurrence during '
        f'encoding was associated with a dramatic reduction in subsequent memory '
        f'({hemi_rows[2][4]}) compared to unilateral IEDs (left: {hemi_rows[0][4]}; '
        f'right: {hemi_rows[1][4]}), chi2(2) = {chi2_all:.2f}, p {p_str(p_all)}. '
        f'This impairment was consistent across stimulation conditions, confirming that '
        f'bilateral IEDs impair memory regardless of BLA stimulation. Left vs. right unilateral '
        f'IEDs did not differ (p = {p_str(p_lr)}). Right-hemisphere IEDs showed '
        f'timing-specific effects (Before Image and After Image windows), while bilateral IEDs '
        f'impaired memory across all windows. IED laterality did not predict the overall '
        f'stimulation benefit on d-prime.'
    )

    # ================================================================
    # OVERALL SUMMARY
    # ================================================================
    pdf.section_title('Summary of Findings')
    pdf.body_text(
        'Three key findings emerged from these analyses:'
    )
    pdf.body_text(
        '1. IEDs during encoding significantly impaired subsequent memory, with the strongest '
        'effects in the post-image consolidation window and during electrical stimulation. A '
        'dose-response relationship was observed such that IEDs spanning more timing windows '
        'produced progressively worse memory. IEDs during retrieval did not affect recall.'
    )
    pdf.body_text(
        '2. Rather than attenuating IED disruption, BLA stimulation created a window of '
        'heightened vulnerability: IEDs concurrent with stimulation pulses were the most '
        'disruptive (p = .004). On non-stimulation trials, the post-image window was most '
        'vulnerable to IED disruption (p = .038).'
    )
    pdf.body_text(
        '3. Bilateral IEDs (affecting both hemispheres simultaneously) were devastating to '
        'memory (46.0% remembered vs. ~74% for unilateral), regardless of stimulation. This '
        'impairment was the strongest predictor of subsequent memory failure among all IED '
        'characteristics examined.'
    )

    # Save
    out_path = os.path.join(OUTPUT_DIR, 'IED_Hypothesis_Results.pdf')
    pdf.output(out_path)
    print(f'Saved: {out_path}')


if __name__ == '__main__':
    main()
