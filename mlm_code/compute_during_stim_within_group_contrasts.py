#!/Users/martinahollearn/anaconda3/bin/python
"""
Compute the within-responder-group During-Stimulation forgetting contrasts and
write them (with Benjamini-Hochberg FDR q-values) to a CSV that the fpdf report
build_during_stim_responder_report.py can render as an APA table.

This lives in a separate step because statsmodels (GEE) and fpdf are not
installed in the same Python environment: this script runs under anaconda
(statsmodels), the report runs under .venv (fpdf), and they communicate via the
CSV below.

Model (identical to build_during_stim_responder_inferential_report.py):
    forgotten ~ during_stim * C(responder_status, ref='Strong responders')
    patient-clustered binomial GEE, exchangeable working correlation.

Each output row is the SIMPLE EFFECT of a During-Stim-window IED within one
responder group: odds of forgetting on stimulated IED-positive trials whose IED
fell in the During-Stim window vs. that group's stimulated IED trials whose IED
fell only in OTHER time windows (the reference, during_stim = 0).

Output: OUTPUTS/ied_timing_memory/during_stim_within_group_contrasts_fdr.csv
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests

ROOT = '/Users/martinahollearn/Library/CloudStorage/Box-Box/InmanLab/BLAES_data/dissertation/AMME_BLAES'
ENC_CSV = os.path.join(
    ROOT, 'IED', 'AMMEBLAES_IEDs_trial_level_dissertation_study_usethis_cleaned_with_memory.csv')
RESP_CSV = os.path.join(ROOT, 'OUTPUTS', 'csvs', 'AMMEBLAES_responder_status.csv')
OUT_DIR = os.path.join(ROOT, 'IED', 'ied_timing_memory')
OUT_CSV = os.path.join(OUT_DIR, 'during_stim_within_group_contrasts_fdr.csv')

RESP_ORDER = ['Strong responders', 'Anti-responders',
              'Non-responders', 'Moderate responders']
TIMING_COLS = ['BeforeImgITI', 'DuringImg', 'AfterImgITI', 'DuringStim']
REF = "Strong responders"
TERM = "during_stim:C(responder_status, Treatment(reference='Strong responders'))[T.%s]"


def any_yes(s: pd.Series) -> str:
    return 'Y' if (s == 'Y').any() else 'N'


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    ied = pd.read_csv(ENC_CSV)
    ied = ied[ied['MemoryOutcome'].isin(['remembered', 'forgotten'])].copy()
    trial = (ied.groupby(['Patient', 'Trial', 'MemoryOutcome', 'StimCond'])[TIMING_COLS]
                .agg(any_yes).reset_index())
    trial['forgotten'] = (trial['MemoryOutcome'] == 'forgotten').astype(int)
    trial['during_stim'] = (trial['DuringStim'] == 'Y').astype(int)

    resp = pd.read_csv(RESP_CSV).rename(columns={'Responder status': 'responder_status'})
    trial = trial.merge(resp[['Patient', 'responder_status']], on='Patient', how='left')

    stim_window_patients = [
        p for p in trial['Patient'].dropna().unique()
        if ((trial['Patient'] == p) & (trial['during_stim'] == 1)).sum() > 0
    ]
    df = trial[(trial['Patient'].isin(stim_window_patients)) &
               (trial['StimCond'] == 'S') &
               trial['responder_status'].isin(RESP_ORDER)].copy()

    model = smf.gee(
        "forgotten ~ during_stim * C(responder_status, Treatment(reference='Strong responders'))",
        groups='Patient', cov_struct=sm.cov_struct.Exchangeable(),
        family=sm.families.Binomial(), data=df).fit()

    b = model.params
    V = model.cov_params()
    idx = {n: i for i, n in enumerate(b.index)}

    def contrast(weights: dict) -> dict:
        v = np.zeros(len(b))
        for term, w in weights.items():
            v[idx[term]] = w
        est = float(v @ b.values)
        se = float(np.sqrt(v @ V.values @ v))
        z = est / se
        return {
            'OR': float(np.exp(est)),
            'CI_low': float(np.exp(est - 1.96 * se)),
            'CI_high': float(np.exp(est + 1.96 * se)),
            'z': z,
            'p': float(2 * norm.sf(abs(z))),
        }

    specs = {
        'Strong responders':   {'during_stim': 1.0},
        'Anti-responders':     {'during_stim': 1.0, TERM % 'Anti-responders': 1.0},
        'Non-responders':      {'during_stim': 1.0, TERM % 'Non-responders': 1.0},
        'Moderate responders': {'during_stim': 1.0, TERM % 'Moderate responders': 1.0},
    }
    # group-level n's for the table note / sanity
    counts = {g: df[df['responder_status'] == g] for g in RESP_ORDER}

    rows = []
    for g in RESP_ORDER:
        c = contrast(specs[g])
        sub = counts[g]
        c['Group'] = g
        c['n_DS_trials'] = int((sub['during_stim'] == 1).sum())
        c['n_other_trials'] = int((sub['during_stim'] == 0).sum())
        rows.append(c)

    out = pd.DataFrame(rows)
    out['q_fdr'] = multipletests(out['p'].values, method='fdr_bh')[1]
    out['n_trials_total'] = int(model.nobs)
    out['n_patients'] = int(df['Patient'].nunique())
    out = out[['Group', 'OR', 'CI_low', 'CI_high', 'z', 'p', 'q_fdr',
               'n_DS_trials', 'n_other_trials', 'n_trials_total', 'n_patients']]
    out.to_csv(OUT_CSV, index=False)

    print(f'Wrote {OUT_CSV}')
    print(out.to_string(index=False))


if __name__ == '__main__':
    main()
