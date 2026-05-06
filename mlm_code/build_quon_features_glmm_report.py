#!/usr/bin/env python
"""
Build the GLMM version of Quon Figure 2 Panel A (encoding + retrieval IED
features vs. memory) and an APA-style PDF report.

Replaces the Mann-Whitney / point-biserial tests with a single multivariable
trial-level binomial GLMM (random intercept for patient) containing all IED
features simultaneously, restricted to gray-matter IEDs.

Inputs (produced by ied_quon_features_glmm.R):
  outputs/ied_Quon_paper/figure2_glmm_coefs.csv
  outputs/ied_Quon_paper/figure2_glmm_coefs_retrieval.csv

Outputs:
  outputs/ied_Quon_paper/figure2_ied_features_memory_glmm.png
  outputs/ied_Quon_paper/figure2_ied_features_memory_glmm_retrieval.png
  outputs/ied_Quon_paper/figure2_ied_features_memory_glmm_stats.csv
  outputs/ied_Quon_paper/figure2_ied_features_memory_glmm_stats_retrieval.csv
  outputs/ied_Quon_paper/IED_Features_GLMM_Report.pdf
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from fpdf import FPDF

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, 'outputs', 'ied_Quon_paper')

ENC_COEFS_CSV = os.path.join(OUT_DIR, 'figure2_glmm_coefs.csv')
RET_COEFS_CSV = os.path.join(OUT_DIR, 'figure2_glmm_coefs_retrieval.csv')

ENC_FIG = os.path.join(OUT_DIR, 'figure2_ied_features_memory_glmm.png')
RET_FIG = os.path.join(OUT_DIR, 'figure2_ied_features_memory_glmm_retrieval.png')
ENC_STATS_OUT = os.path.join(OUT_DIR, 'figure2_ied_features_memory_glmm_stats.csv')
RET_STATS_OUT = os.path.join(OUT_DIR, 'figure2_ied_features_memory_glmm_stats_retrieval.csv')
ENC_CORR_FIG = os.path.join(OUT_DIR, 'figure2_glmm_feature_correlations_encoding.png')
RET_CORR_FIG = os.path.join(OUT_DIR, 'figure2_glmm_feature_correlations_retrieval.png')
ENC_CORR_CSV = os.path.join(OUT_DIR, 'figure2_glmm_feature_correlations_encoding.csv')
RET_CORR_CSV = os.path.join(OUT_DIR, 'figure2_glmm_feature_correlations_retrieval.csv')
ENC_TRIAL_CSV = os.path.join(OUT_DIR, 'figure2_glmm_trial_features.csv')
RET_TRIAL_CSV = os.path.join(OUT_DIR, 'figure2_glmm_trial_features_retrieval.csv')
PDF_PATH = os.path.join(OUT_DIR, 'IED_Features_GLMM_Report.pdf')

SIG_COLOR = '#C44E52'
NS_COLOR = '#555555'

ENC_FEATURE_ORDER = [
    'IED Rate',
    'Hemisphere (R vs. none)',
    'Channel Spread',
    'Region Spread',
    'Before Image',
    'During Image',
    'After Image',
    'During Stim',
]
RET_FEATURE_ORDER = [
    'IED Rate',
    'Hemisphere (R vs. none)',
    'Channel Spread',
    'Region Spread',
    'Before Image',
    'During Image',
]

PLOT_LABELS = {
    'IED Rate':                 'IED Rate\n(per 1 SD)',
    'Hemisphere (R vs. none)':  'Any Right-\nhem IED\n(Yes vs No)',
    'Channel Spread':           'Channel\nSpread\n(per 1 SD)',
    'Region Spread':            'Region\nSpread\n(per 1 SD)',
    'Before Image':             'Before Image\nYes vs No',
    'During Image':             'During Image\nYes vs No',
    'After Image':              'After Image\nYes vs No',
    'During Stim':              'During Stim\nYes vs No',
}


def sig_star(p):
    if pd.isna(p):
        return ''
    if p < 0.001:
        return '***'
    if p < 0.01:
        return '**'
    if p < 0.05:
        return '*'
    return ''


def load_slopes(path, order):
    df = pd.read_csv(path)
    slopes = df[df['term'] != '(Intercept)'].copy()
    slopes['order'] = slopes['feature'].map({f: i for i, f in enumerate(order)})
    slopes = slopes.sort_values('order').reset_index(drop=True)
    return df, slopes


def build_figure(slopes: pd.DataFrame, out_path: str, title: str):
    """Panel A with GLMM log-odds betas and SE error bars."""
    sns.set_style('ticks')
    sns.set_context('talk', font_scale=0.85)

    fig, ax = plt.subplots(figsize=(max(10, 1.6 * len(slopes)), 6))
    colors = [SIG_COLOR if p < 0.05 else NS_COLOR for p in slopes['p.value']]
    ax.bar(range(len(slopes)), slopes['estimate'],
           color=colors, edgecolor='black', linewidth=0.6)
    # SE error bars (symmetric, +/- 1 SE)
    ax.errorbar(range(len(slopes)), slopes['estimate'],
                yerr=slopes['std.error'],
                fmt='none', ecolor='black', elinewidth=0.9, capsize=3)
    ax.axhline(0, color='black', linestyle='--', linewidth=0.8)

    ax.set_xticks(range(len(slopes)))
    ax.set_xticklabels([PLOT_LABELS[f] for f in slopes['feature']], fontsize=10)
    ax.set_ylabel('GLMM beta (log-odds of remembering)', fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')

    for i, row in slopes.iterrows():
        star = sig_star(row['p.value'])
        if not star:
            continue
        se = row['std.error'] if not pd.isna(row['std.error']) else 0
        top = row['estimate'] + se
        bot = row['estimate'] - se
        y_pos = top + 0.03 if row['estimate'] >= 0 else bot - 0.06
        va = 'bottom' if row['estimate'] >= 0 else 'top'
        ax.text(i, y_pos, star, ha='center', va=va,
                fontsize=14, fontweight='bold', color=SIG_COLOR)

    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Saved {out_path}')


# =========================================================================
#  FEATURE CORRELATION MATRIX
# =========================================================================
# Maps the predictor *names* in the report to the *column names* the R
# script writes to the trial-level CSV.
# For each predictor we record its column name and whether it is binary.
# Binary predictors are 0/1; continuous predictors are z-scored.
ENC_FEATURE_COLS = {
    'IED Rate':                ('ied_rate_z',  False),
    'Hemisphere (R vs. none)': ('any_R_ied',   True),
    'Channel Spread':          ('ch_spread_z', False),
    'Region Spread':           ('reg_spread_z',False),
    'Before Image':            ('before_img',  True),
    'During Image':            ('during_img',  True),
    'After Image':             ('after_img',   True),
    'During Stim':             ('during_stim', True),
}
RET_FEATURE_COLS = {
    'IED Rate':                ('ied_rate_z',  False),
    'Hemisphere (R vs. none)': ('any_R_ied',   True),
    'Channel Spread':          ('ch_spread_z', False),
    'Region Spread':           ('reg_spread_z',False),
    'Before Image':            ('before_img',  True),
    'During Image':            ('during_img',  True),
}


def _pair_association(x: np.ndarray, y: np.ndarray,
                      x_bin: bool, y_bin: bool):
    """Return (assoc, p, metric_label) using the appropriate test for the
    variable types of the pair.

    binary x binary  -> phi coefficient (= Pearson r on 0/1) + chi-square p
    binary x cont    -> point-biserial r (= Pearson r) + its t-test p
    cont   x cont    -> Spearman's rho + its asymptotic p

    Phi is signed so it carries a direction interpretable in the same scale
    as the other two metrics in the heatmap.
    """
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    if x.size < 3 or np.all(x == x[0]) or np.all(y == y[0]):
        return np.nan, np.nan, ''
    if x_bin and y_bin:
        # 2x2 chi-square (no continuity correction so phi^2 = chi2 / n)
        ct = pd.crosstab(x.astype(int), y.astype(int))
        if ct.shape != (2, 2):
            return np.nan, np.nan, 'phi'
        chi2, p, _, _ = stats.chi2_contingency(ct, correction=False)
        phi = np.sqrt(chi2 / ct.values.sum())
        # Sign from Pearson r so direction matches the other cells
        r = np.corrcoef(x, y)[0, 1]
        if r < 0:
            phi = -phi
        return phi, p, 'phi'
    if x_bin or y_bin:
        # Point-biserial correlation
        if x_bin:
            res = stats.pointbiserialr(x.astype(int), y)
        else:
            res = stats.pointbiserialr(y.astype(int), x)
        return res.correlation, res.pvalue, 'r_pb'
    # Continuous-continuous: Spearman
    res = stats.spearmanr(x, y)
    return res.correlation, res.pvalue, 'rho'


def build_correlation_matrix(trial_csv: str, feat_map: dict,
                             out_csv: str, out_fig: str, phase_label: str):
    """Compute pairwise feature associations using the appropriate test per
    variable-type pair and save both the heatmap PNG and a long-format CSV.

    Pair-type rules:
      - binary x binary  -> phi coefficient with chi-square test
      - binary x cont    -> point-biserial correlation
      - cont   x cont    -> Spearman's rho
    """
    if not os.path.exists(trial_csv):
        print(f'  Skipping association matrix: {trial_csv} not found')
        return None
    df = pd.read_csv(trial_csv)

    labels = list(feat_map.keys())
    cols   = [feat_map[l][0] for l in labels]
    is_bin = [feat_map[l][1] for l in labels]

    n = len(labels)
    assoc = pd.DataFrame(np.eye(n), index=labels, columns=labels)
    pmat  = pd.DataFrame(np.zeros((n, n)), index=labels, columns=labels)
    metric = pd.DataFrame('', index=labels, columns=labels)

    long_rows = []
    for i in range(n):
        xi = df[cols[i]].astype(float).to_numpy()
        for j in range(n):
            if i == j:
                metric.iloc[i, j] = ''
                continue
            xj = df[cols[j]].astype(float).to_numpy()
            a, p, mlab = _pair_association(xi, xj, is_bin[i], is_bin[j])
            assoc.iloc[i, j] = a
            pmat.iloc[i, j] = p
            metric.iloc[i, j] = mlab
            if j > i:
                long_rows.append({
                    'feature_1': labels[i],
                    'feature_2': labels[j],
                    'metric': mlab,
                    'assoc': a,
                    'p_value': p,
                })

    long_df = pd.DataFrame(long_rows)
    long_df.to_csv(out_csv, index=False)
    print(f'Saved {out_csv}')

    # Heatmap with annotations including the metric symbol on the lower
    # triangle.  We mask the strictly upper triangle so the matrix is
    # readable, and put a small symbol (* / ** / ***) for significance.
    sns.set_style('white')
    fig, ax = plt.subplots(figsize=(max(10, 1.1 * n + 5),
                                    max(8, 1.0 * n + 4)))
    mask = np.triu(np.ones_like(assoc.values, dtype=bool), k=1)

    # Build annotation grid: value + sig stars
    annot = np.empty_like(assoc.values, dtype=object)
    for i in range(n):
        for j in range(n):
            if mask[i, j]:
                annot[i, j] = ''
                continue
            if i == j:
                annot[i, j] = '1.00'
                continue
            v = assoc.iloc[i, j]
            p = pmat.iloc[i, j]
            star = ''
            if not np.isnan(p):
                if p < 0.001: star = '***'
                elif p < 0.01: star = '**'
                elif p < 0.05: star = '*'
            annot[i, j] = f'{v:.2f}{star}'

    sns.heatmap(assoc, annot=annot, fmt='', cmap='RdBu_r', center=0,
                vmin=-1, vmax=1, mask=mask,
                cbar_kws={'label': 'Association (phi / r_pb / rho)'},
                annot_kws={'fontsize': 14, 'fontweight': 'bold'},
                linewidths=0.5,
                linecolor='lightgray', ax=ax, square=True)
    ax.set_title(
        f'IED feature associations - {phase_label}\n'
        '(phi for binary-binary, point-biserial r for binary-continuous, '
        "Spearman rho for continuous-continuous)",
        fontsize=14, fontweight='bold')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha='right',
                       fontsize=13, fontweight='bold')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0,
                       fontsize=13, fontweight='bold')
    # Bold + larger colorbar label and ticks
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=12)
    cbar.ax.set_ylabel('Association (phi / r_pb / rho)',
                       fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(out_fig, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Saved {out_fig}')
    return long_df


def chi_square_table(trial_csv: str, feat_map: dict):
    """Return a list-of-dicts of chi-square tests of independence for every
    binary x binary pair (used to build the report table)."""
    if not os.path.exists(trial_csv):
        return []
    df = pd.read_csv(trial_csv)
    bin_labels = [l for l, (_, b) in feat_map.items() if b]
    rows = []
    for i in range(len(bin_labels)):
        for j in range(i + 1, len(bin_labels)):
            a, b = bin_labels[i], bin_labels[j]
            ca, cb = feat_map[a][0], feat_map[b][0]
            x = df[ca].astype(int).to_numpy()
            y = df[cb].astype(int).to_numpy()
            ct = pd.crosstab(x, y)
            if ct.shape != (2, 2):
                rows.append({'a': a, 'b': b, 'chi2': np.nan,
                             'df': np.nan, 'p': np.nan, 'phi': np.nan,
                             'n': int(ct.values.sum())})
                continue
            chi2, p, dof, _ = stats.chi2_contingency(ct, correction=False)
            n = ct.values.sum()
            phi = np.sqrt(chi2 / n)
            r = np.corrcoef(x, y)[0, 1]
            if r < 0:
                phi = -phi
            rows.append({'a': a, 'b': b, 'chi2': chi2, 'df': dof,
                         'p': p, 'phi': phi, 'n': int(n)})
    return rows


def write_stats_csv(slopes: pd.DataFrame, out_path: str):
    out = slopes[['feature', 'kind', 'term', 'estimate', 'std.error',
                  'statistic', 'p.value', 'conf.low', 'conf.high',
                  'or', 'or_ci_low', 'or_ci_hi', 'n_yes', 'n_no']].copy()
    out.columns = ['Feature', 'Predictor type', 'Term', 'Beta (log-odds)',
                   'SE', 'z', 'p', 'Beta CI lower', 'Beta CI upper',
                   'OR', 'OR CI lower', 'OR CI upper',
                   'N feature=1 or N total', 'N feature=0']
    out.to_csv(out_path, index=False)
    print(f'Saved {out_path}')


# =========================================================================
#  PDF REPORT  (APA-style tables)
# =========================================================================
def fmt_p(p):
    if pd.isna(p):
        return ''
    if p < 0.001:
        return '< .001'
    return f'= {p:.3f}'


def fmt_num(v, digits=2):
    if pd.isna(v):
        return '-'
    return f'{v:.{digits}f}'


class Report(FPDF):
    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=20)

    def header(self):
        self.set_font('Helvetica', 'I', 8)
        self.cell(0, 5, 'IED Features vs. Memory (joint GLMM) - Quon Replication',
                  align='R')
        self.ln(8)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', align='C')

    def h1(self, text):
        self.set_font('Helvetica', 'B', 15)
        self.ln(2)
        self.cell(0, 9, text, new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def h2(self, text):
        self.set_font('Helvetica', 'B', 12)
        self.ln(2)
        self.cell(0, 7, text, new_x="LMARGIN", new_y="NEXT")

    def body(self, text):
        self.set_font('Times', '', 11)
        self.multi_cell(0, 5.5, text)
        self.ln(1)


def _hline(pdf: Report, width: float, thickness: float = 0.3):
    x = pdf.get_x()
    y = pdf.get_y()
    pdf.set_line_width(thickness)
    pdf.line(x, y, x + width, y)
    pdf.ln(0.8)


_TABLE_COUNTER = {'n': 0}


def apa_table(pdf: Report, title_text: str, note_text: str,
              header_labels, header_italic_mask, widths, row_dicts,
              row_aligns):
    _TABLE_COUNTER['n'] += 1
    tnum = _TABLE_COUNTER['n']

    pdf.ln(2)
    pdf.set_font('Times', 'B', 11)
    pdf.cell(0, 5.5, f'Table {tnum}', new_x="LMARGIN", new_y="NEXT")
    pdf.set_font('Times', 'I', 11)
    pdf.multi_cell(0, 5.5, title_text)
    pdf.ln(1)

    total_w = sum(widths)

    _hline(pdf, total_w, 0.5)
    for label, italic, w, align in zip(header_labels, header_italic_mask,
                                       widths, row_aligns):
        pdf.set_font('Times', 'I' if italic else '', 10.5)
        pdf.cell(w, 5.8, label, align=align)
    pdf.ln(5.8)
    _hline(pdf, total_w, 0.3)
    pdf.ln(0.5)

    pdf.set_font('Times', '', 10.5)
    for row in row_dicts:
        for label, w, align in zip(header_labels, widths, row_aligns):
            pdf.cell(w, 5.2, str(row.get(label, '')), align=align)
        pdf.ln(5.2)

    _hline(pdf, total_w, 0.5)
    pdf.ln(1)

    if note_text:
        pdf.set_font('Times', 'I', 9.5)
        pdf.cell(pdf.get_string_width('Note. '), 4.8, 'Note.')
        pdf.set_font('Times', '', 9.5)
        pdf.multi_cell(0, 4.8, ' ' + note_text)
        pdf.ln(1)


def joint_model_table(pdf: Report, coefs_all: pd.DataFrame, title_text: str,
                      note_text: str):
    """APA table showing intercept + all simultaneous slopes from the joint
    multivariable GLMM."""
    pretty_term = {
        '(Intercept)': 'Intercept',
        'ied_rate_z':  'IED Rate (z)',
        'any_R_ied':   'Any Right-hem IED',
        'ch_spread_z': 'Channel Spread (z)',
        'reg_spread_z':'Region Spread (z)',
        'before_img':  'Before Image (Y)',
        'during_img':  'During Image (Y)',
        'after_img':   'After Image (Y)',
        'during_stim': 'During Stim (Y)',
    }

    headers = ['Term', 'beta', 'SE', 'OR', 'z', 'p']
    italics = [False, True, True, True, True, True]
    widths = [60, 20, 20, 22, 20, 28]
    aligns = ['L', 'R', 'R', 'R', 'R', 'R']

    rows = []
    for _, r in coefs_all.iterrows():
        rows.append({
            'Term': pretty_term.get(r['term'], r['term']),
            'beta': fmt_num(r['estimate'], 2),
            'SE':   fmt_num(r['std.error'], 2),
            'OR':   fmt_num(r['or'], 2),
            'z':    fmt_num(r['statistic'], 2),
            'p':    fmt_p(r['p.value']).lstrip('= ').strip(),
        })

    apa_table(pdf, title_text=title_text, note_text=note_text,
              header_labels=headers, header_italic_mask=italics,
              widths=widths, row_dicts=rows, row_aligns=aligns)


def build_pdf(enc_all, enc_slopes, ret_all, ret_slopes,
              enc_n, enc_npat, enc_rem, enc_forg,
              ret_n, ret_npat, ret_rem, ret_forg):
    pdf = Report()
    pdf.add_page()

    pdf.set_font('Helvetica', 'B', 17)
    pdf.cell(0, 10, 'IED Features vs. Memory - Joint GLMM',
             new_x="LMARGIN", new_y="NEXT")
    pdf.set_font('Helvetica', '', 11)
    pdf.cell(0, 6, 'Quon (2021) Figure 2A analog, trial-level multivariable GLMM',
             new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    pdf.h1('Overview')
    pdf.body(
        'The original Panel A in the Quon replication tested each IED feature '
        'separately using Mann-Whitney U tests (binary features) or point-'
        'biserial correlations (continuous features) on channel-level IED '
        'rows. Those tests treat channel rows as independent and do not '
        'adjust one feature for the others.\n\n'
        'This report replaces that panel with a single trial-level '
        'multivariable binomial GLMM (logit link) per phase. IEDs are '
        'collapsed to one row per (patient, trial) that contained at least '
        'one IED (all tissue types retained). The White-Matter (W vs. G) '
        'feature is dropped from the predictor list per request, but no '
        'tissue-based filtering is applied to the trials themselves. All '
        'remaining features are entered simultaneously:'
    )
    pdf.set_font('Times', 'I', 11)
    pdf.multi_cell(0, 5.5,
                   '    memory ~ ied_rate_z + any_R_ied + ch_spread_z + reg_spread_z\n'
                   '             + before_img + during_img + after_img + during_stim\n'
                   '             + (1 | patient_id)')
    pdf.ln(1)
    pdf.set_font('Times', '', 11)
    pdf.body(
        'Retrieval has only Before-Image and During-Image timing windows; '
        'the After-Image and During-Stim terms are dropped for that phase. '
        'Continuous features (IED rate, channel spread, region spread) are '
        'z-scored so beta is the log-odds change per 1 SD while adjusting '
        'for all other features. Binary features are coded 0/1 so beta is '
        'the log-odds change when the feature is present vs. absent. Odds '
        'ratios (OR = exp(beta)) are reported alongside beta. Error bars '
        'on the figures are +/- 1 SE on the beta scale.'
    )

    pdf.h1('Sample')
    pdf.body(
        f'Encoding: N = {enc_n} trials from {enc_npat} patients '
        f'({enc_rem} remembered, {enc_forg} forgotten).\n'
        f'Retrieval: N = {ret_n} trials from {ret_npat} patients '
        f'({ret_rem} remembered, {ret_forg} forgotten; new items excluded).'
    )

    # ========== ENCODING ==========
    pdf.add_page()
    pdf.h1('Encoding phase')
    pdf.h2('(A) IED features associated with encoding memory')
    if os.path.exists(ENC_FIG):
        pdf.image(ENC_FIG, w=180)
        pdf.set_font('Times', 'I', 9.5)
        pdf.multi_cell(0, 4.8,
                       'Figure 1. Joint GLMM coefficients (beta, log-odds of '
                       'remembering) for each IED feature, encoding. Error '
                       'bars are +/- 1 SE. Red bars indicate p < .05 (Wald). '
                       'All tissues retained; the White-Matter (W-G) feature is dropped from the predictor list.')
        pdf.ln(2)

    joint_model_table(
        pdf, enc_all,
        title_text='Joint multivariable GLMM fixed effects - encoding phase.',
        note_text=(f'Binomial GLMM: memory ~ 8 IED features + (1 | patient_id). '
                   f'N = {enc_n} trials, {enc_npat} patients. beta is on the log-'
                   'odds scale, adjusting for all other features. Continuous '
                   'predictors (IED Rate, Channel Spread, Region Spread) are '
                   'z-scored. OR = exp(beta). p values from Wald z tests.')
    )

    if os.path.exists(ENC_CORR_FIG):
        pdf.add_page()
        pdf.h2('Associations among IED features (encoding)')
        pdf.body(
            'Pairwise associations among the 8 trial-level IED features that '
            'enter the joint GLMM. Because most predictors are binary, an '
            'all-Pearson / all-Spearman matrix is inappropriate; instead, '
            'each cell uses the test that fits the variable types of the '
            'pair: phi coefficient with chi-square test of independence for '
            'binary x binary pairs, point-biserial r for binary x continuous '
            'pairs, and Spearman rho for continuous x continuous pairs. '
            'Strong off-diagonal associations indicate features that share '
            'variance in the multivariable model and therefore compete for '
            'the same beta - the diagnostic for the apparent shrinkage of '
            'individual feature effects in the joint model relative to '
            'single-predictor GLMMs.'
        )
        pdf.image(ENC_CORR_FIG, w=170)
        pdf.set_font('Times', 'I', 9.5)
        pdf.multi_cell(0, 4.8,
                       'Figure 2. Pairwise feature associations at the trial '
                       'level (encoding). Cells use phi (binary-binary), '
                       'point-biserial r (binary-continuous), or Spearman '
                       'rho (continuous-continuous). Asterisks: * p < .05, '
                       '** p < .01, *** p < .001.')
        pdf.ln(2)

        enc_chi = chi_square_table(ENC_TRIAL_CSV, ENC_FEATURE_COLS)
        if enc_chi:
            chi_rows = []
            for r in enc_chi:
                chi_rows.append({
                    'Pair':  f"{r['a']} x {r['b']}",
                    'phi':   fmt_num(r['phi'], 2),
                    'chi2':  fmt_num(r['chi2'], 2),
                    'df':    f"{int(r['df'])}" if not pd.isna(r['df']) else '',
                    'n':     f"{r['n']}",
                    'p':     fmt_p(r['p']).lstrip('= ').strip(),
                })
            apa_table(
                pdf,
                title_text=('Chi-square tests of independence for every '
                            'binary x binary IED-feature pair (encoding).'),
                note_text=('phi is the signed phi coefficient '
                           '(sqrt(chi2 / N), sign taken from Pearson r). '
                           'Tests are 2x2 chi-square without continuity '
                           'correction. p values are not adjusted for '
                           'multiple comparisons.'),
                header_labels=['Pair', 'phi', 'chi2', 'df', 'n', 'p'],
                header_italic_mask=[False, True, True, True, True, True],
                widths=[80, 18, 22, 14, 18, 24],
                row_dicts=chi_rows,
                row_aligns=['L', 'R', 'R', 'R', 'R', 'R'],
            )

    # ========== RETRIEVAL ==========
    pdf.add_page()
    pdf.h1('Retrieval phase')
    pdf.h2('(A) IED features associated with retrieval memory')
    if os.path.exists(RET_FIG):
        pdf.image(RET_FIG, w=180)
        pdf.set_font('Times', 'I', 9.5)
        pdf.multi_cell(0, 4.8,
                       'Figure 2. Joint GLMM coefficients for each IED feature, '
                       'retrieval. Error bars are +/- 1 SE. Red bars indicate '
                       'p < .05 (Wald). Retrieval has only Before-Image and '
                       'During-Image timing windows; only old items '
                       '(remembered / forgotten) are included. Gray-matter '
                       'IEDs only.')
        pdf.ln(2)

    joint_model_table(
        pdf, ret_all,
        title_text='Joint multivariable GLMM fixed effects - retrieval phase.',
        note_text=(f'Binomial GLMM: memory ~ 6 IED features + (1 | patient_id). '
                   f'N = {ret_n} trials, {ret_npat} patients. beta is on the log-'
                   'odds scale, adjusting for all other features. Continuous '
                   'predictors are z-scored. OR = exp(beta). p values from '
                   'Wald z tests. The retrieval sample is much smaller than '
                   'encoding and effects should be interpreted with caution.')
    )

    if os.path.exists(RET_CORR_FIG):
        pdf.add_page()
        pdf.h2('Associations among IED features (retrieval)')
        pdf.body(
            'Pairwise associations among the 6 trial-level IED features in '
            'the retrieval joint GLMM, using the same pair-type rules as the '
            'encoding matrix (phi + chi-square for binary x binary, '
            'point-biserial r for binary x continuous, Spearman rho for '
            'continuous x continuous). Note in particular the Channel '
            'Spread x Region Spread association, which helps explain the '
            'large opposing coefficients those two features received in the '
            'retrieval joint model.'
        )
        pdf.image(RET_CORR_FIG, w=170)
        pdf.set_font('Times', 'I', 9.5)
        pdf.multi_cell(0, 4.8,
                       'Figure 3. Pairwise feature associations at the trial '
                       'level (retrieval). Cells use phi (binary-binary), '
                       'point-biserial r (binary-continuous), or Spearman '
                       'rho (continuous-continuous). Asterisks: * p < .05, '
                       '** p < .01, *** p < .001.')
        pdf.ln(2)

        ret_chi = chi_square_table(RET_TRIAL_CSV, RET_FEATURE_COLS)
        if ret_chi:
            chi_rows = []
            for r in ret_chi:
                chi_rows.append({
                    'Pair':  f"{r['a']} x {r['b']}",
                    'phi':   fmt_num(r['phi'], 2),
                    'chi2':  fmt_num(r['chi2'], 2),
                    'df':    f"{int(r['df'])}" if not pd.isna(r['df']) else '',
                    'n':     f"{r['n']}",
                    'p':     fmt_p(r['p']).lstrip('= ').strip(),
                })
            apa_table(
                pdf,
                title_text=('Chi-square tests of independence for every '
                            'binary x binary IED-feature pair (retrieval).'),
                note_text=('phi is the signed phi coefficient '
                           '(sqrt(chi2 / N), sign taken from Pearson r). '
                           'Tests are 2x2 chi-square without continuity '
                           'correction. p values are not adjusted for '
                           'multiple comparisons.'),
                header_labels=['Pair', 'phi', 'chi2', 'df', 'n', 'p'],
                header_italic_mask=[False, True, True, True, True, True],
                widths=[80, 18, 22, 14, 18, 24],
                row_dicts=chi_rows,
                row_aligns=['L', 'R', 'R', 'R', 'R', 'R'],
            )

    # ========== INTERPRETATION ==========
    pdf.add_page()
    pdf.h1('Interpretation')

    pdf.h2('Encoding')
    enc_bullets = []
    for _, r in enc_slopes.iterrows():
        sig = ('significant' if r['p.value'] < 0.05 else 'not significant')
        direction = 'reduces' if r['estimate'] < 0 else 'increases'
        enc_bullets.append(
            f"- {r['feature']}: beta = {fmt_num(r['estimate'], 2)} "
            f"(SE = {fmt_num(r['std.error'], 2)}), OR = {fmt_num(r['or'], 2)}, "
            f"p {fmt_p(r['p.value'])} - {sig}; the feature {direction} the odds "
            "of remembering when all other features are held constant."
        )
    pdf.set_font('Times', '', 11)
    for b in enc_bullets:
        pdf.multi_cell(0, 5.5, b)
        pdf.ln(0.3)
    pdf.ln(1)
    pdf.body(
        'In the joint encoding model, During-Stim IED presence remains '
        'significantly associated with reduced odds of remembering after '
        'adjusting for all other IED features. After-Image IEDs show a '
        'marginal effect in the same direction. IED Rate is no longer '
        'significant once the timing-window features are included, '
        'consistent with its marginal association being partly explained '
        'by co-occurring After / During-Stim activity.'
    )

    pdf.h2('Retrieval')
    ret_bullets = []
    for _, r in ret_slopes.iterrows():
        sig = ('significant' if r['p.value'] < 0.05 else 'not significant')
        direction = 'reduces' if r['estimate'] < 0 else 'increases'
        ret_bullets.append(
            f"- {r['feature']}: beta = {fmt_num(r['estimate'], 2)} "
            f"(SE = {fmt_num(r['std.error'], 2)}), OR = {fmt_num(r['or'], 2)}, "
            f"p {fmt_p(r['p.value'])} - {sig}."
        )
    pdf.set_font('Times', '', 11)
    for b in ret_bullets:
        pdf.multi_cell(0, 5.5, b)
        pdf.ln(0.3)
    pdf.ln(1)
    pdf.body(
        'The retrieval model is based on 144 trials from 18 patients and has '
        'substantially wider SEs than encoding. Channel Spread and Region '
        'Spread show opposing significant effects in the joint model, which '
        'likely reflects their high correlation at retrieval (wider spread '
        'on one metric is partially captured by the other). These effects '
        'should be interpreted as multicollinearity-influenced estimates '
        'rather than independent biological signals.'
    )

    pdf.output(PDF_PATH)
    print(f'Saved {PDF_PATH}')


def _counts_from_coefs(df):
    """Pull N (total trials) from the Intercept row of the tidy coefs."""
    intercept = df[df['term'] == '(Intercept)']
    if intercept.empty:
        return None
    return int(intercept['n_yes'].iloc[0])


def _phase_counts(phase_trial_csv):
    """Read the trial-level CSV the R script wrote to get remembered/forgotten
    counts and unique patient count."""
    if not os.path.exists(phase_trial_csv):
        return None, None, None, None
    d = pd.read_csv(phase_trial_csv)
    return (len(d), d['patient_id'].nunique(),
            int(d['memory'].sum()), int((1 - d['memory']).sum()))


def main():
    for p in [ENC_COEFS_CSV, RET_COEFS_CSV]:
        if not os.path.exists(p):
            raise SystemExit(
                f'Missing {p}; run Rscript ied_quon_features_glmm.R first.')

    enc_all, enc_slopes = load_slopes(ENC_COEFS_CSV, ENC_FEATURE_ORDER)
    ret_all, ret_slopes = load_slopes(RET_COEFS_CSV, RET_FEATURE_ORDER)

    # Figures
    build_figure(
        enc_slopes, ENC_FIG,
        title='(A) IED features and encoding memory - joint GLMM '
              '(all tissues; W-G feature dropped)'
    )
    build_figure(
        ret_slopes, RET_FIG,
        title='(A) IED features and retrieval memory - joint GLMM '
              '(all tissues; W-G feature dropped)'
    )

    # Stats CSVs
    write_stats_csv(enc_slopes, ENC_STATS_OUT)
    write_stats_csv(ret_slopes, RET_STATS_OUT)

    # Feature association matrices (mixed-metric: phi/chi-sq for
    # binary-binary, point-biserial for binary-continuous, Spearman for
    # continuous-continuous)
    build_correlation_matrix(ENC_TRIAL_CSV, ENC_FEATURE_COLS,
                             ENC_CORR_CSV, ENC_CORR_FIG, 'encoding')
    build_correlation_matrix(RET_TRIAL_CSV, RET_FEATURE_COLS,
                             RET_CORR_CSV, RET_CORR_FIG, 'retrieval')

    # Sample counts for the report
    enc_n, enc_npat, enc_rem, enc_forg = _phase_counts(
        os.path.join(OUT_DIR, 'figure2_glmm_trial_features.csv')
    )
    ret_n, ret_npat, ret_rem, ret_forg = _phase_counts(
        os.path.join(OUT_DIR, 'figure2_glmm_trial_features_retrieval.csv')
    )

    build_pdf(enc_all, enc_slopes, ret_all, ret_slopes,
              enc_n, enc_npat, enc_rem, enc_forg,
              ret_n, ret_npat, ret_rem, ret_forg)


if __name__ == '__main__':
    main()
