#!/Users/martinahollearn/anaconda3/bin/python
"""
Build an inferential PDF report for During-Stimulation-window IEDs and
responder status.

Primary analyses:
1. DS-IED burden model on stimulated IED-positive trials only:
      during_stim ~ responder_status
2. Forgetting model on stimulated IED-positive trials only:
      forgotten ~ during_stim * responder_status

Both models use patient-clustered binomial GEE with exchangeable working
correlation. "Strong responders" is the reference group throughout.

Outputs are written only to outputs/ied_timing_memory/.
"""

from __future__ import annotations

import os
import textwrap
import warnings

ROOT = '/Users/martinahollearn/Library/CloudStorage/Box-Box/InmanLab/BLAES_data/dissertation/AMME_BLAES'
OUTPUT_DIR = os.path.join(ROOT, 'outputs', 'ied_timing_memory')
os.environ.setdefault('MPLCONFIGDIR', os.path.join(ROOT, 'outputs', '.mplconfig'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import fisher_exact, norm


ENC_CSV = os.path.join(
    ROOT, 'IED', 'AMMEBLAES_IEDs_trial_level_dissertation_study_usethis_cleaned_with_memory.csv'
)
RESP_CSV = os.path.join(
    ROOT, 'behavioral figures', 'behavioral_figures_reformatted', 'AMMEBLAES_responder_status.csv'
)
FIG_GROUP = os.path.join(OUTPUT_DIR, 'during_stim_burden_by_responder_group.png')
FIG_PATIENT = os.path.join(OUTPUT_DIR, 'during_stim_forgetting_boost_by_patient.png')

PDF_PATH = os.path.join(OUTPUT_DIR, 'During_Stim_Responder_Inferential_Report.pdf')
BURDEN_CSV = os.path.join(OUTPUT_DIR, 'during_stim_burden_gee_terms.csv')
FORGET_CSV = os.path.join(OUTPUT_DIR, 'during_stim_forgetting_gee_terms.csv')
SIMPLE_EFFECTS_CSV = os.path.join(OUTPUT_DIR, 'during_stim_within_group_contrasts.csv')
POOLED_COUNTS_CSV = os.path.join(OUTPUT_DIR, 'during_stim_group_counts_for_inference.csv')
POOLED_FISHER_CSV = os.path.join(OUTPUT_DIR, 'during_stim_group_pooled_fisher_tests.csv')

RESP_ORDER = [
    'Anti-responders',
    'Non-responders',
    'Moderate responders',
    'Strong responders',
]
PAGE_SIZE = (8.5, 11.0)

TERM_LABELS = {
    'Intercept': 'Intercept',
    "C(responder_status, Treatment(reference='Strong responders'))[T.Anti-responders]":
        'Anti-responders vs Strong responders',
    "C(responder_status, Treatment(reference='Strong responders'))[T.Non-responders]":
        'Non-responders vs Strong responders',
    "C(responder_status, Treatment(reference='Strong responders'))[T.Moderate responders]":
        'Moderate responders vs Strong responders',
    'during_stim': 'During Stimulation window IED (Strong responders)',
    "during_stim:C(responder_status, Treatment(reference='Strong responders'))[T.Anti-responders]":
        'Extra During Stimulation effect in Anti-responders vs Strong responders',
    "during_stim:C(responder_status, Treatment(reference='Strong responders'))[T.Non-responders]":
        'Extra During Stimulation effect in Non-responders vs Strong responders',
    "during_stim:C(responder_status, Treatment(reference='Strong responders'))[T.Moderate responders]":
        'Extra During Stimulation effect in Moderate responders vs Strong responders',
}


def any_yes(series: pd.Series) -> str:
    return 'Y' if (series == 'Y').any() else 'N'


def format_p(value: float) -> str:
    if pd.isna(value):
        return 'NA'
    if value < 0.001:
        return '<0.001'
    return f'{value:.3f}'


def format_block(df: pd.DataFrame, columns: list[str] | None = None) -> str:
    if columns is not None:
        df = df[columns].copy()
    return df.to_string(index=False)


def fit_qic(result) -> tuple[float, float]:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        qic, qicu = result.qic(scale=1.0)
    return float(qic), float(qicu)


def load_trial_level_data() -> pd.DataFrame:
    ied = pd.read_csv(ENC_CSV)
    ied = ied[ied['MemoryOutcome'].isin(['remembered', 'forgotten'])].copy()

    timing_cols = ['BeforeImgITI', 'DuringImg', 'AfterImgITI', 'DuringStim']
    trial = (
        ied.groupby(['Patient', 'Trial', 'MemoryOutcome', 'StimCond'])[timing_cols]
        .agg(any_yes)
        .reset_index()
    )
    trial['forgotten'] = (trial['MemoryOutcome'] == 'forgotten').astype(int)
    trial['during_stim'] = (trial['DuringStim'] == 'Y').astype(int)

    resp = pd.read_csv(RESP_CSV).rename(columns={'Responder status': 'responder_status'})
    trial = trial.merge(resp[['Patient', 'responder_status', 'avg_stim_dprime_diff']],
                        on='Patient', how='left')

    stim_window_patients = [
        patient for patient in sorted(trial['Patient'].dropna().unique())
        if ((trial['Patient'] == patient) & (trial['during_stim'] == 1)).sum() > 0
    ]

    df = trial[
        (trial['Patient'].isin(stim_window_patients)) &
        (trial['StimCond'] == 'S') &
        trial['responder_status'].isin(RESP_ORDER)
    ].copy()

    df['responder_status'] = pd.Categorical(df['responder_status'],
                                            categories=RESP_ORDER, ordered=True)
    return df


def build_group_counts(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group in RESP_ORDER:
        sub = df[df['responder_status'] == group]
        if sub.empty:
            continue
        ds = sub[sub['during_stim'] == 1]
        nods = sub[sub['during_stim'] == 0]
        rows.append({
            'Group': group,
            'n_patients': int(sub['Patient'].nunique()),
            'n_DS_trials': int(len(ds)),
            'n_nonDS_trials': int(len(nods)),
            'forgotten_DS': int(ds['forgotten'].sum()),
            'forgotten_nonDS': int(nods['forgotten'].sum()),
            'pct_forgotten_DS': 100 * ds['forgotten'].mean() if len(ds) else np.nan,
            'pct_forgotten_nonDS': 100 * nods['forgotten'].mean() if len(nods) else np.nan,
            'forgetting_boost_pp': (
                100 * ds['forgotten'].mean() - 100 * nods['forgotten'].mean()
                if len(ds) and len(nods) else np.nan
            ),
            'mean_stim_dprime_diff': sub['avg_stim_dprime_diff'].mean(),
        })
    return pd.DataFrame(rows)


def fit_models(df: pd.DataFrame):
    family = sm.families.Binomial()

    burden_model = smf.gee(
        "during_stim ~ C(responder_status, Treatment(reference='Strong responders'))",
        groups='Patient',
        cov_struct=sm.cov_struct.Exchangeable(),
        family=family,
        data=df,
    ).fit()

    forgetting_model = smf.gee(
        "forgotten ~ during_stim * C(responder_status, Treatment(reference='Strong responders'))",
        groups='Patient',
        cov_struct=sm.cov_struct.Exchangeable(),
        family=family,
        data=df,
    ).fit()

    return burden_model, forgetting_model


def term_table(result, model_name: str) -> pd.DataFrame:
    conf = result.conf_int()
    rows = []
    for term in result.params.index:
        rows.append({
            'model': model_name,
            'term': term,
            'label': TERM_LABELS.get(term, term),
            'B': float(result.params[term]),
            'SE(B)': float(result.bse[term]),
            'z': float(result.tvalues[term]),
            'p': float(result.pvalues[term]),
            'OR': float(np.exp(result.params[term])),
            'CI_low': float(np.exp(conf.loc[term, 0])),
            'CI_high': float(np.exp(conf.loc[term, 1])),
        })
    return pd.DataFrame(rows)


def wald_scalar(result, hypothesis: str) -> tuple[float, int, float]:
    test = result.wald_test(hypothesis, scalar=True)
    return float(test.statistic), int(test.df_denom), float(test.pvalue)


def linear_contrast(result, label: str, weights: dict[str, float]) -> dict[str, float | str]:
    params = result.params
    cov = result.cov_params()
    vec = np.zeros(len(params))
    param_index = {name: idx for idx, name in enumerate(params.index)}
    for term, weight in weights.items():
        vec[param_index[term]] = weight
    beta = float(vec @ params.values)
    se = float(np.sqrt(vec @ cov.values @ vec))
    z_value = beta / se
    p_value = 2 * norm.sf(abs(z_value))
    ci_low = beta - 1.96 * se
    ci_high = beta + 1.96 * se
    return {
        'Group': label,
        'B': beta,
        'SE(B)': se,
        'z': z_value,
        'p': p_value,
        'OR': float(np.exp(beta)),
        'CI_low': float(np.exp(ci_low)),
        'CI_high': float(np.exp(ci_high)),
    }


def within_group_effects(forgetting_model: object, group_counts: pd.DataFrame) -> pd.DataFrame:
    anti_term = (
        "during_stim:C(responder_status, Treatment(reference='Strong responders'))[T.Anti-responders]"
    )
    non_term = (
        "during_stim:C(responder_status, Treatment(reference='Strong responders'))[T.Non-responders]"
    )
    mod_term = (
        "during_stim:C(responder_status, Treatment(reference='Strong responders'))[T.Moderate responders]"
    )

    rows = [
        linear_contrast(forgetting_model, 'Strong responders', {'during_stim': 1.0}),
        linear_contrast(forgetting_model, 'Anti-responders', {'during_stim': 1.0, anti_term: 1.0}),
        linear_contrast(forgetting_model, 'Non-responders', {'during_stim': 1.0, non_term: 1.0}),
        linear_contrast(forgetting_model, 'Moderate responders', {'during_stim': 1.0, mod_term: 1.0}),
    ]
    out = pd.DataFrame(rows)
    out = out.merge(group_counts[['Group', 'n_DS_trials', 'n_nonDS_trials',
                                  'pct_forgotten_DS', 'pct_forgotten_nonDS',
                                  'forgetting_boost_pp']],
                    on='Group', how='left')
    return out


def pooled_fisher_tests(group_counts: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in group_counts.iterrows():
        ds_forg = int(row['forgotten_DS'])
        ds_rem = int(row['n_DS_trials'] - row['forgotten_DS'])
        nods_forg = int(row['forgotten_nonDS'])
        nods_rem = int(row['n_nonDS_trials'] - row['forgotten_nonDS'])
        if row['n_DS_trials'] == 0 or row['n_nonDS_trials'] == 0:
            odds_ratio = np.nan
            p_value = np.nan
        else:
            odds_ratio, p_value = fisher_exact([[ds_forg, ds_rem], [nods_forg, nods_rem]])
        rows.append({
            'Group': row['Group'],
            'pooled_OR': odds_ratio,
            'pooled_p': p_value,
        })
    return pd.DataFrame(rows)


def add_title(fig, title: str, subtitle: str | None = None):
    fig.text(0.05, 0.965, title, fontsize=17, fontweight='bold', va='top')
    if subtitle:
        fig.text(0.05, 0.935, subtitle, fontsize=10.5, va='top')


def add_paragraph(fig, y: float, text: str, width: int = 95, fontsize: float = 10.5) -> float:
    wrapped = textwrap.fill(text, width=width)
    fig.text(0.05, y, wrapped, fontsize=fontsize, va='top')
    lines = wrapped.count('\n') + 1
    return y - lines * 0.028 - 0.012


def add_mono_block(fig, y: float, text: str, fontsize: float = 9.2) -> float:
    fig.text(0.05, y, text, family='monospace', fontsize=fontsize, va='top')
    lines = text.count('\n') + 1
    return y - lines * 0.022 - 0.02


def add_image(fig, path: str, left: float, bottom: float, width: float, height: float):
    if not os.path.exists(path):
        return
    ax = fig.add_axes([left, bottom, width, height])
    ax.imshow(plt.imread(path))
    ax.axis('off')


def build_pdf(df: pd.DataFrame,
              group_counts: pd.DataFrame,
              burden_terms: pd.DataFrame,
              forgetting_terms: pd.DataFrame,
              within_terms: pd.DataFrame,
              pooled_fisher: pd.DataFrame,
              burden_fit: dict[str, float],
              forgetting_fit: dict[str, float]):
    with PdfPages(PDF_PATH) as pdf:
        fig = plt.figure(figsize=PAGE_SIZE)
        fig.patch.set_facecolor('white')
        add_title(
            fig,
            'During Stimulation IEDs and Responder Status: Inferential Report',
            'Stimulated trials only, by design. Strong responders are the reference group.',
        )
        y = 0.89
        y = add_paragraph(
            fig, y,
            f'The analysis cohort contains {df["Patient"].nunique()} patients and {len(df)} '
            'stimulated IED-positive trials. All inferential tests intentionally restrict the '
            'comparison to stimulated trials only. The goal is to ask whether, within the set '
            'of stimulated trials that already contain an IED, IEDs that land in the During '
            'Stimulation time window are associated with responder status and with forgetting.'
        )
        y = add_paragraph(
            fig, y,
            'Primary models use patient-clustered binomial GEE with exchangeable working '
            'correlation. This keeps the trial-level analysis but adjusts for the fact that '
            'many trials come from the same patient. GEE coefficients are reported as B, '
            'SE(B), z, p, and odds ratios (OR). Model fit is summarized with QIC and QICu.'
        )
        y = add_paragraph(
            fig, y,
            'Interpretation target: a significant within-group During Stimulation effect means '
            'that, inside that responder group, stimulated trials with a During Stimulation '
            'window IED are forgotten more often than stimulated trials whose IEDs land only in '
            'other time windows. A significant interaction versus Strong responders would mean '
            'that the size of that effect differs from the Strong responder effect.'
        )
        add_image(fig, FIG_GROUP, 0.09, 0.08, 0.82, 0.42)
        fig.text(0.05, 0.055,
                 'Figure. Existing descriptive figure used as context. Inferential conclusions '
                 'below come from patient-clustered models, not from pooled bars alone.',
                 fontsize=9.5, va='bottom')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        fig = plt.figure(figsize=PAGE_SIZE)
        fig.patch.set_facecolor('white')
        add_title(fig, 'Raw Stimulation-Only Counts')
        y = 0.91
        y = add_paragraph(
            fig, y,
            'This table reproduces the pooled stimulation-only counts behind the bar figure. '
            'These are descriptive counts, not the primary inferential test.'
        )
        raw_table = group_counts.copy()
        raw_table['mean_stim_dprime_diff'] = raw_table['mean_stim_dprime_diff'].map(lambda x: f'{x:+.3f}')
        raw_table['pct_forgotten_DS'] = raw_table['pct_forgotten_DS'].map(lambda x: f'{x:.1f}')
        raw_table['pct_forgotten_nonDS'] = raw_table['pct_forgotten_nonDS'].map(lambda x: f'{x:.1f}')
        raw_table['forgetting_boost_pp'] = raw_table['forgetting_boost_pp'].map(lambda x: f'{x:+.1f}')
        y = add_mono_block(
            fig, y,
            format_block(raw_table, [
                'Group', 'n_patients', 'n_DS_trials', 'n_nonDS_trials',
                'forgotten_DS', 'forgotten_nonDS', 'pct_forgotten_DS',
                'pct_forgotten_nonDS', 'forgetting_boost_pp', 'mean_stim_dprime_diff'
            ]),
            fontsize=9.0,
        )
        y = add_paragraph(
            fig, y,
            'Visible separation in pooled bars can be misleading because pooled bars treat all '
            'trials as independent. The inferential models below downweight that illusion by '
            'accounting for within-patient clustering.'
        )
        add_image(fig, FIG_PATIENT, 0.08, 0.02, 0.84, 0.26)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        fig = plt.figure(figsize=PAGE_SIZE)
        fig.patch.set_facecolor('white')
        add_title(fig, 'Model 1: During Stimulation Window Burden')
        y = 0.91
        y = add_paragraph(
            fig, y,
            'Model formula: during_stim ~ responder_status. Outcome = whether an IED-positive '
            'stimulated trial had an IED in the During Stimulation window. Reference group = '
            'Strong responders.'
        )
        fit_lines = (
            f"Trials = {burden_fit['n_obs']}, patients = {burden_fit['n_clusters']}, "
            f"working correlation alpha = {burden_fit['alpha']:.3f}, "
            f"QIC = {burden_fit['qic']:.2f}, QICu = {burden_fit['qicu']:.2f}, "
            f"omnibus Wald chi2({int(burden_fit['df'])}) = {burden_fit['wald']:.2f}, "
            f"p = {format_p(burden_fit['p'])}"
        )
        y = add_paragraph(fig, y, fit_lines)
        burden_show = burden_terms.copy()
        burden_show = burden_show[burden_show['term'] != 'Intercept'].copy()
        for col in ['B', 'SE(B)', 'z', 'OR', 'CI_low', 'CI_high']:
            burden_show[col] = burden_show[col].map(lambda x: f'{x:.3f}')
        burden_show['p'] = burden_show['p'].map(format_p)
        burden_show['CI'] = burden_show['CI_low'] + ' to ' + burden_show['CI_high']
        y = add_mono_block(
            fig, y,
            format_block(burden_show, ['label', 'B', 'SE(B)', 'z', 'p', 'OR', 'CI']),
            fontsize=9.0,
        )
        y = add_paragraph(
            fig, y,
            'Result: there is no evidence that Anti-, Non-, or Moderate responders carry a '
            'different During Stimulation IED burden than Strong responders once trials are '
            'clustered by patient.'
        )
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        fig = plt.figure(figsize=PAGE_SIZE)
        fig.patch.set_facecolor('white')
        add_title(fig, 'Model 2: Forgetting on Stimulated IED-Positive Trials')
        y = 0.91
        y = add_paragraph(
            fig, y,
            'Model formula: forgotten ~ during_stim * responder_status. Outcome = forgotten '
            '(1) vs remembered (0) on stimulated IED-positive trials only. The during_stim '
            'coefficient is the During Stimulation effect within Strong responders. Interaction '
            'terms are the extra During Stimulation effect relative to Strong responders.'
        )
        fit_lines = (
            f"Trials = {forgetting_fit['n_obs']}, patients = {forgetting_fit['n_clusters']}, "
            f"working correlation alpha = {forgetting_fit['alpha']:.3f}, "
            f"QIC = {forgetting_fit['qic']:.2f}, QICu = {forgetting_fit['qicu']:.2f}, "
            f"interaction Wald chi2({int(forgetting_fit['df'])}) = {forgetting_fit['wald']:.2f}, "
            f"p = {format_p(forgetting_fit['p'])}"
        )
        y = add_paragraph(fig, y, fit_lines)
        forget_show = forgetting_terms.copy()
        for col in ['B', 'SE(B)', 'z', 'OR', 'CI_low', 'CI_high']:
            forget_show[col] = forget_show[col].map(lambda x: f'{x:.3f}')
        forget_show['p'] = forget_show['p'].map(format_p)
        forget_show['CI'] = forget_show['CI_low'] + ' to ' + forget_show['CI_high']
        y = add_mono_block(
            fig, y,
            format_block(forget_show, ['label', 'B', 'SE(B)', 'z', 'p', 'OR', 'CI']),
            fontsize=8.35,
        )
        y = add_paragraph(
            fig, y,
            'Result: the interaction versus Strong responders is not significant overall. '
            'So Anti- and Non-responders do not show a reliably larger During Stimulation '
            'effect than Strong responders, even if their raw pooled percentages look larger.'
        )
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        fig = plt.figure(figsize=PAGE_SIZE)
        fig.patch.set_facecolor('white')
        add_title(fig, 'Within-Group During Stimulation vs Non-During Stimulation Contrasts')
        y = 0.91
        y = add_paragraph(
            fig, y,
            'These are the patient-clustered simple effects from the forgetting model. Each row '
            'tests, within a single responder group, whether stimulated trials with a During '
            'Stimulation window IED are forgotten more often than stimulated trials whose IEDs '
            'occur only in other time windows.'
        )
        simple_show = within_terms.copy()
        for col in ['B', 'SE(B)', 'z', 'OR', 'CI_low', 'CI_high',
                    'pct_forgotten_DS', 'pct_forgotten_nonDS', 'forgetting_boost_pp']:
            simple_show[col] = simple_show[col].map(lambda x: f'{x:.3f}' if 'pct_' not in col and 'boost' not in col else f'{x:.1f}')
        simple_show['p'] = simple_show['p'].map(format_p)
        simple_show['CI'] = simple_show['CI_low'] + ' to ' + simple_show['CI_high']
        y = add_mono_block(
            fig, y,
            format_block(simple_show, [
                'Group', 'n_DS_trials', 'n_nonDS_trials', 'pct_forgotten_DS',
                'pct_forgotten_nonDS', 'forgetting_boost_pp', 'B', 'SE(B)',
                'z', 'p', 'OR', 'CI'
            ]),
            fontsize=8.4,
        )
        pooled_show = pooled_fisher.copy()
        pooled_show['pooled_OR'] = pooled_show['pooled_OR'].map(lambda x: f'{x:.3f}')
        pooled_show['pooled_p'] = pooled_show['pooled_p'].map(format_p)
        y = add_paragraph(
            fig, y,
            'For reference only, the table below gives naive pooled Fisher exact tests that ignore '
            'patient clustering. These are not the primary inferential tests.'
        )
        y = add_mono_block(fig, y, format_block(pooled_show, ['Group', 'pooled_OR', 'pooled_p']), fontsize=9.0)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        fig = plt.figure(figsize=PAGE_SIZE)
        fig.patch.set_facecolor('white')
        add_title(fig, 'Interpretation')
        y = 0.91
        y = add_paragraph(
            fig, y,
            'Your intuition is partly right. Anti-responders and Non-responders each show a '
            'significant within-group During Stimulation effect when the contrast is tested '
            'inside that group. Strong responders also show a significant During Stimulation '
            'effect. Moderate responders do not.'
        )
        y = add_paragraph(
            fig, y,
            'What is not significant is the extra effect relative to Strong responders. In other '
            'words, Anti- and Non-responders have positive During Stimulation forgetting boosts, '
            'but those boosts are not estimated precisely enough to conclude that they are larger '
            'than the Strong-responder boost after patient clustering is taken into account.'
        )
        y = add_paragraph(
            fig, y,
            'That is why the pooled bars can look compelling while the interaction terms remain '
            'non-significant. The pooled bars count trials, but the inferential question depends '
            'on how consistently those effects replicate across patients. With only 5 Anti-, '
            '7 Non-, and 7 Strong-responder patients, the interaction estimates remain wide.'
        )
        y = add_paragraph(
            fig, y,
            'Bottom line: restriction to stimulation trials is statistically coherent and matches '
            'your scientific question. The defensible summary is that During Stimulation window '
            'IEDs are associated with more forgetting within Strong, Anti-, and Non-responder '
            'groups, but the data do not yet support a claim that Anti- or Non-responders differ '
            'significantly from Strong responders in the size of that During Stimulation effect.'
        )
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = load_trial_level_data()
    group_counts = build_group_counts(df)
    burden_model, forgetting_model = fit_models(df)

    burden_terms = term_table(burden_model, 'burden')
    forgetting_terms = term_table(forgetting_model, 'forgetting')
    within_terms = within_group_effects(forgetting_model, group_counts)
    pooled_fisher = pooled_fisher_tests(group_counts)

    burden_terms.to_csv(BURDEN_CSV, index=False)
    forgetting_terms.to_csv(FORGET_CSV, index=False)
    within_terms.to_csv(SIMPLE_EFFECTS_CSV, index=False)
    group_counts.to_csv(POOLED_COUNTS_CSV, index=False)
    pooled_fisher.to_csv(POOLED_FISHER_CSV, index=False)

    burden_qic, burden_qicu = fit_qic(burden_model)
    forgetting_qic, forgetting_qicu = fit_qic(forgetting_model)
    burden_wald, burden_df, burden_p = wald_scalar(
        burden_model,
        "C(responder_status, Treatment(reference='Strong responders'))[T.Anti-responders] = 0, "
        "C(responder_status, Treatment(reference='Strong responders'))[T.Moderate responders] = 0, "
        "C(responder_status, Treatment(reference='Strong responders'))[T.Non-responders] = 0",
    )
    forgetting_wald, forgetting_df, forgetting_p = wald_scalar(
        forgetting_model,
        "during_stim:C(responder_status, Treatment(reference='Strong responders'))[T.Anti-responders] = 0, "
        "during_stim:C(responder_status, Treatment(reference='Strong responders'))[T.Moderate responders] = 0, "
        "during_stim:C(responder_status, Treatment(reference='Strong responders'))[T.Non-responders] = 0",
    )

    burden_fit = {
        'n_obs': int(burden_model.nobs),
        'n_clusters': int(df['Patient'].nunique()),
        'alpha': float(burden_model.cov_struct.dep_params),
        'qic': burden_qic,
        'qicu': burden_qicu,
        'wald': burden_wald,
        'df': burden_df,
        'p': burden_p,
    }
    forgetting_fit = {
        'n_obs': int(forgetting_model.nobs),
        'n_clusters': int(df['Patient'].nunique()),
        'alpha': float(forgetting_model.cov_struct.dep_params),
        'qic': forgetting_qic,
        'qicu': forgetting_qicu,
        'wald': forgetting_wald,
        'df': forgetting_df,
        'p': forgetting_p,
    }

    build_pdf(df, group_counts, burden_terms, forgetting_terms,
              within_terms, pooled_fisher, burden_fit, forgetting_fit)
    print(f'Saved {PDF_PATH}')


if __name__ == '__main__':
    main()
