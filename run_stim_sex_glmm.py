#!/usr/bin/env python
"""
Stim x Sex GLMM Analysis
=========================
Runs trial-level (G)LMMs to test stim x sex interactions on neural measures
(Power, Coherence, PAC) during encoding and retrieval.

For each phase (Encoding, Retrieval), runs 6 model families:
  1. Power   -- per-region models (freq band as factor)
  2. Power   -- per-frequency models (region as factor)
  3. Coherence -- per-region-pair models (freq band as factor)
  4. Coherence -- per-frequency models (region pair as factor)
  5. PAC     -- per-region-pair models (freq band as factor)
  6. PAC     -- per-frequency models (region pair as factor)

Regions restricted to: BLA, HPC, EC, PRC (and pairs thereof).
Frequency bands: Theta (4-8 Hz), Slow gamma (30-55 Hz).
PAC values are z-scored to avoid astronomical ORs.
Groups with < 5 subjects per sex are skipped.

Outputs: PDF reports in outputs/stim_sex_glmm/
"""

import os
import sys
import json
import subprocess
import tempfile
import warnings
import numpy as np
import pandas as pd
from datetime import datetime

warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'outputs', 'stim_sex_glmm')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Target regions
POWER_REGIONS = ['BLA', 'HPC', 'EC', 'PRC']
COH_PAIRS = ['BLA_EC', 'BLA_HPC', 'BLA_PRC', 'EC_HPC', 'EC_PRC', 'HPC_PRC']
PAC_TARGET_REGIONS = {'BLA', 'HPC', 'EC', 'PRC'}

THETA = (4, 8)
SLOW_GAMMA = (30, 55)
BANDS = {'Theta': THETA, 'SlowGamma': SLOW_GAMMA}

MIN_SUBJECTS_PER_SEX = 5

BEHAVIORAL_CSV = os.path.join(
    SCRIPT_DIR, 'behavioral figures',
    'AMMEBLAES_includedpts_firstsession_behavioral.csv',
)


# =========================================================================
#  SEX METADATA
# =========================================================================

def load_sex_map():
    df = pd.read_csv(BEHAVIORAL_CSV)
    df['sex'] = df['sex'].astype(str).str.strip().str.lower()
    return dict(zip(df['Patient'], df['sex']))


# =========================================================================
#  DATA HELPERS
# =========================================================================

def get_freq_cols(df, prefix='diff_Freq_'):
    cols = [c for c in df.columns if c.startswith(prefix)]
    freqs = np.array([float(c.replace(prefix, '')) for c in cols])
    return cols, freqs


def band_mean(df, freq_cols, freqs, lo, hi):
    mask = (freqs >= lo) & (freqs <= hi)
    return df[np.array(freq_cols)[mask]].mean(axis=1)


def build_trial_level_df(mlmr_frames, sex_map, target_regions, is_pair=False):
    """Build long-form trial-level DataFrame with band means and sex."""
    if not mlmr_frames:
        return pd.DataFrame()

    df = pd.concat(mlmr_frames, ignore_index=True)
    freq_cols, freqs = get_freq_cols(df)
    if not freq_cols:
        return pd.DataFrame()

    # Filter to target regions
    if is_pair:
        # Keep pairs where both parts are in target regions
        def pair_ok(r):
            parts = str(r).split('_')
            return len(parts) == 2 and all(p in PAC_TARGET_REGIONS for p in parts)
        df = df[df['Region'].apply(pair_ok)].copy()
    else:
        df = df[df['Region'].isin(target_regions)].copy()

    if df.empty:
        return pd.DataFrame()

    # Add sex
    df['sex'] = df['Patient'].map(sex_map)
    df = df[df['sex'].isin(['male', 'female'])].copy()

    # Add stim indicator
    df['stim'] = (df['trial_type'].str.lower().str.strip() == 'stim').astype(int)

    # Compute band means
    for band_name, (lo, hi) in BANDS.items():
        df[band_name] = band_mean(df, freq_cols, freqs, lo, hi)

    # Melt to long form: one row per trial x band
    keep_cols = ['Patient', 'Region', 'stim', 'sex', 'yes_or_no'] + list(BANDS.keys())
    df = df[keep_cols].copy()

    long = df.melt(
        id_vars=['Patient', 'Region', 'stim', 'sex', 'yes_or_no'],
        value_vars=list(BANDS.keys()),
        var_name='freq_band',
        value_name='neural_value',
    )

    return long


def build_pac_patient_level_df(pac_data, sex_map):
    """Build patient-level long-form DataFrame from PAC dict data.
    PAC data: {region_pair: {patient: spectrum_array}} for each condition.
    """
    freqs = pac_data.get('freqs_diff')
    if freqs is None:
        return pd.DataFrame()

    conditions = {
        ('nostim', 'remembered'): pac_data.get('diff_nostim_rem', {}),
        ('nostim', 'forgotten'): pac_data.get('diff_nostim_forg', {}),
        ('stim', 'remembered'): pac_data.get('diff_stim_rem', {}),
        ('stim', 'forgotten'): pac_data.get('diff_stim_forg', {}),
    }

    rows = []
    for (stim_label, mem_label), region_dict in conditions.items():
        for region, patient_dict in region_dict.items():
            # Filter to target pairs
            parts = region.split('_')
            if len(parts) != 2 or not all(p in PAC_TARGET_REGIONS for p in parts):
                continue
            for patient, spectrum in patient_dict.items():
                sex = sex_map.get(patient)
                if sex not in ('male', 'female'):
                    continue
                spectrum = np.asarray(spectrum, dtype=float)
                for band_name, (lo, hi) in BANDS.items():
                    mask = (freqs >= lo) & (freqs <= hi)
                    if mask.sum() == 0:
                        continue
                    val = float(spectrum[mask].mean())
                    rows.append({
                        'Patient': patient,
                        'Region': region,
                        'stim': 1 if stim_label == 'stim' else 0,
                        'sex': sex,
                        'yes_or_no': 'yes' if mem_label == 'remembered' else 'no',
                        'freq_band': band_name,
                        'neural_value': val,
                    })

    df = pd.DataFrame(rows)
    if not df.empty:
        # Z-score PAC values to avoid astronomical coefficients
        df['neural_value'] = (df['neural_value'] - df['neural_value'].mean()) / df['neural_value'].std()

    return df


def check_min_subjects(df, groupby_cols):
    """Check that each sex has at least MIN_SUBJECTS_PER_SEX unique subjects."""
    for sex in ['male', 'female']:
        n = df[df['sex'] == sex]['Patient'].nunique()
        if n < MIN_SUBJECTS_PER_SEX:
            return False, n
    return True, None


# =========================================================================
#  R MODEL RUNNER
# =========================================================================

def run_lmm_in_r(df, formula, model_label):
    """Run an LMM in R via subprocess. Returns dict with results."""
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False, mode='w') as f:
        df.to_csv(f, index=False)
        csv_path = f.name

    r_script = f"""
library(lme4)
library(lmerTest)
library(car)

d <- read.csv("{csv_path}")
d$sex <- factor(d$sex, levels=c("female","male"))
d$stim <- factor(d$stim, levels=c(0,1), labels=c("nostim","stim"))

if ("freq_band" %in% names(d)) {{
    d$freq_band <- factor(d$freq_band)
}}
if ("Region" %in% names(d)) {{
    d$Region <- factor(d$Region)
}}

tryCatch({{
    m <- lmer({formula}, data=d, REML=FALSE,
              control=lmerControl(optimizer="bobyqa", optCtrl=list(maxfun=50000)))

    s <- summary(m)
    coefs <- as.data.frame(s$coefficients)
    coefs$term <- rownames(coefs)

    # Type III Anova
    a <- tryCatch({{
        a3 <- anova(m, type=3, ddf="Satterthwaite")
        a_df <- as.data.frame(a3)
        a_df$term <- rownames(a_df)
        a_df
    }}, error=function(e) {{
        data.frame(term="error", msg=as.character(e))
    }})

    n_obs <- nrow(d)
    n_patients <- length(unique(d$Patient))
    n_male <- length(unique(d$Patient[d$sex=="male"]))
    n_female <- length(unique(d$Patient[d$sex=="female"]))

    result <- list(
        coefs = coefs,
        anova = a,
        n_obs = n_obs,
        n_patients = n_patients,
        n_male = n_male,
        n_female = n_female,
        formula = "{formula}",
        converged = TRUE
    )

    cat(jsonlite::toJSON(result, auto_unbox=TRUE, pretty=TRUE))
}}, error=function(e) {{
    cat(jsonlite::toJSON(list(
        error = as.character(e),
        converged = FALSE,
        formula = "{formula}"
    ), auto_unbox=TRUE, pretty=TRUE))
}})
"""

    with tempfile.NamedTemporaryFile(suffix='.R', delete=False, mode='w') as f:
        f.write(r_script)
        r_path = f.name

    try:
        result = subprocess.run(
            ['Rscript', r_path],
            capture_output=True, text=True, timeout=120,
        )
        os.unlink(csv_path)
        os.unlink(r_path)

        if result.returncode != 0:
            return {'error': result.stderr, 'converged': False, 'label': model_label}

        # Parse JSON output
        output = result.stdout.strip()
        parsed = json.loads(output)
        parsed['label'] = model_label

        # jsonlite toJSON serializes data frames as list-of-row-dicts;
        # convert to dict-of-column-lists for consistent access downstream.
        def rows_to_cols(obj):
            if isinstance(obj, list) and len(obj) > 0 and isinstance(obj[0], dict):
                keys = list(obj[0].keys())
                return {k: [row.get(k) for row in obj] for k in keys}
            return obj

        if 'anova' in parsed:
            parsed['anova'] = rows_to_cols(parsed['anova'])
        if 'coefs' in parsed:
            parsed['coefs'] = rows_to_cols(parsed['coefs'])

        return parsed

    except Exception as e:
        try:
            os.unlink(csv_path)
            os.unlink(r_path)
        except:
            pass
        return {'error': str(e), 'converged': False, 'label': model_label}


# =========================================================================
#  MODEL SPECIFICATIONS
# =========================================================================

def run_per_region_models(df, measure_name, regions_list):
    """Model 1: For each region, freq_band x stim x sex, random intercept for Patient.
    DV = neural_value ~ stim * sex * freq_band + (1|Patient)
    """
    results = []
    for region in regions_list:
        rd = df[df['Region'] == region].copy()
        if rd.empty:
            continue
        ok, n = check_min_subjects(rd, [])
        if not ok:
            results.append({
                'label': f'{measure_name} -- {region}',
                'skipped': True,
                'reason': f'Only {n} subjects in one sex group (min={MIN_SUBJECTS_PER_SEX})',
            })
            continue

        formula = "neural_value ~ stim * sex * freq_band + (1|Patient)"
        label = f'{measure_name} -- {region} (freq band x stim x sex)'
        result = run_lmm_in_r(rd, formula, label)
        results.append(result)
        print(f"    {label}: {'OK' if result.get('converged') else 'FAILED'}")

    return results


def run_per_freq_models(df, measure_name, regions_list):
    """Model 2: For each freq band, region x stim x sex, random intercept for Patient.
    DV = neural_value ~ stim * sex * Region + (1|Patient)
    """
    results = []
    for band_name in BANDS.keys():
        bd = df[df['freq_band'] == band_name].copy()
        # Also restrict to target regions
        bd = bd[bd['Region'].isin(regions_list)].copy()
        if bd.empty:
            continue
        ok, n = check_min_subjects(bd, [])
        if not ok:
            results.append({
                'label': f'{measure_name} -- {band_name}',
                'skipped': True,
                'reason': f'Only {n} subjects in one sex group (min={MIN_SUBJECTS_PER_SEX})',
            })
            continue

        formula = "neural_value ~ stim * sex * Region + (1|Patient)"
        label = f'{measure_name} -- {band_name} (region x stim x sex)'
        result = run_lmm_in_r(bd, formula, label)
        results.append(result)
        print(f"    {label}: {'OK' if result.get('converged') else 'FAILED'}")

    return results


# =========================================================================
#  PDF REPORT GENERATION
# =========================================================================

def generate_pdf_report(phase_label, all_model_results, output_path):
    """Generate a detailed PDF report of all model results."""
    from fpdf import FPDF

    class PDF(FPDF):
        def header(self):
            self.set_font('Helvetica', 'B', 11)
            self.cell(0, 8, f'Stim x Sex GLMM Report -- {phase_label}', 0, 1, 'C')
            self.ln(2)

        def footer(self):
            self.set_y(-15)
            self.set_font('Helvetica', 'I', 8)
            self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    pdf = PDF(orientation='L', unit='mm', format='A4')
    pdf.set_auto_page_break(auto=True, margin=20)

    # Helper: safe multi_cell that always starts from left margin
    def safe_multi_cell(w, h, txt, **kwargs):
        pdf.set_x(pdf.l_margin)
        safe_txt = str(txt).encode('latin-1', errors='replace').decode('latin-1')
        pdf.multi_cell(pdf.epw if w == 0 else w, h, safe_txt, **kwargs)

    pdf.add_page()

    # Title
    pdf.set_font('Helvetica', 'B', 16)
    pdf.cell(0, 10, f'Stim x Sex Interaction GLMM Analysis -- {phase_label}', 0, 1, 'C')
    pdf.ln(3)

    # Overview
    pdf.set_font('Helvetica', '', 10)
    overview = (
        f'Report generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}\n'
        f'Phase: {phase_label}\n'
        f'Regions (Power): BLA, HPC, EC, PRC\n'
        f'Region pairs (Coherence): BLA_EC, BLA_HPC, BLA_PRC, EC_HPC, EC_PRC, HPC_PRC\n'
        f'PAC pairs: directed pairs among BLA, HPC, EC, PRC\n'
        f'Frequency bands: Theta (4-8 Hz), Slow gamma (30-55 Hz)\n'
        f'DV: Baseline-corrected neural measure (PAC z-scored)\n'
        f'Model: neural_value ~ stim * sex * [freq_band or Region] + (1|Patient)\n'
        f'Minimum subjects per sex: {MIN_SUBJECTS_PER_SEX}\n'
        f'Key interaction of interest: stim x sex'
    )
    for line in overview.split('\n'):
        pdf.cell(0, 5, line, 0, 1)
    pdf.ln(5)

    # Models
    for model_family_label, results_list in all_model_results:
        pdf.set_font('Helvetica', 'B', 13)
        pdf.set_fill_color(220, 220, 240)
        pdf.cell(0, 8, model_family_label, 0, 1, 'L', fill=True)
        pdf.ln(2)

        for result in results_list:
            label = result.get('label', 'Unknown')
            pdf.set_font('Helvetica', 'B', 11)
            pdf.cell(0, 7, label, 0, 1)

            if result.get('skipped'):
                pdf.set_font('Helvetica', 'I', 9)
                pdf.cell(0, 5, f"  SKIPPED: {result.get('reason', 'Unknown')}", 0, 1)
                pdf.ln(3)
                continue

            if not result.get('converged'):
                pdf.set_font('Helvetica', 'I', 9)
                err = str(result.get('error', 'Unknown error'))[:200]
                pdf.cell(0, 5, f"  MODEL FAILED: {err}", 0, 1)
                pdf.ln(3)
                continue

            # Sample info
            pdf.set_font('Helvetica', '', 9)
            n_obs = result.get('n_obs', '?')
            n_pat = result.get('n_patients', '?')
            n_m = result.get('n_male', '?')
            n_f = result.get('n_female', '?')
            pdf.cell(0, 5, f"  N obs={n_obs}, Patients={n_pat} (Male={n_m}, Female={n_f})", 0, 1)
            pdf.cell(0, 5, f"  Formula: {result.get('formula', '?')}", 0, 1)
            pdf.ln(1)

            # Type III ANOVA table
            anova = result.get('anova', {})
            if isinstance(anova, dict) and 'term' in anova:
                # Single term dict
                terms = [anova['term']] if isinstance(anova['term'], str) else anova['term']
                pdf.set_font('Helvetica', 'B', 9)
                pdf.cell(0, 5, '  Type III ANOVA (Satterthwaite):', 0, 1)
                pdf.set_font('Courier', '', 8)

                # Header
                pdf.cell(80, 5, '  Term', 0, 0)
                pdf.cell(20, 5, 'Sum Sq', 0, 0, 'R')
                pdf.cell(20, 5, 'Mean Sq', 0, 0, 'R')
                pdf.cell(15, 5, 'NumDF', 0, 0, 'R')
                pdf.cell(20, 5, 'DenDF', 0, 0, 'R')
                pdf.cell(20, 5, 'F', 0, 0, 'R')
                pdf.cell(25, 5, 'p-value', 0, 1, 'R')

                if isinstance(terms, list):
                    n_terms = len(terms)
                    for i in range(n_terms):
                        term = terms[i]
                        # Highlight stim:sex interactions
                        is_interaction = 'stim' in str(term).lower() and 'sex' in str(term).lower()

                        def get_val(key, idx):
                            v = anova.get(key)
                            if v is None:
                                return '--'
                            if isinstance(v, list):
                                return v[idx] if idx < len(v) else '--'
                            return v

                        ss = get_val('Sum Sq', i)
                        ms = get_val('Mean Sq', i)
                        ndf = get_val('NumDF', i)
                        ddf = get_val('DenDF', i)
                        fval = get_val('F value', i)
                        # Try different p-value column names
                        pval = get_val('Pr(>F)', i)
                        if pval == '--':
                            pval = get_val('Pr..F.', i)

                        if is_interaction:
                            pdf.set_font('Courier', 'B', 8)
                        else:
                            pdf.set_font('Courier', '', 8)

                        def fmt(v, decimals=3):
                            if isinstance(v, (int, float)):
                                if decimals == 4:
                                    return f'{v:.4f}'
                                return f'{v:.3f}'
                            return str(v)[:8]

                        pdf.cell(80, 5, f'  {str(term)[:38]}', 0, 0)
                        pdf.cell(20, 5, fmt(ss), 0, 0, 'R')
                        pdf.cell(20, 5, fmt(ms), 0, 0, 'R')
                        pdf.cell(15, 5, fmt(ndf, 0), 0, 0, 'R')
                        pdf.cell(20, 5, fmt(ddf), 0, 0, 'R')
                        pdf.cell(20, 5, fmt(fval), 0, 0, 'R')

                        p_str = fmt(pval, 4)
                        try:
                            p_float = float(pval)
                            if p_float < 0.001:
                                p_str = f'{p_float:.1e}'
                            elif p_float < 0.05:
                                p_str = f'{p_float:.4f} *'
                        except:
                            pass

                        pdf.cell(25, 5, p_str, 0, 1, 'R')

                pdf.set_font('Courier', '', 8)
            pdf.set_x(pdf.l_margin)
            pdf.ln(2)

            # Fixed effects coefficients table
            coefs = result.get('coefs', {})
            if isinstance(coefs, dict) and 'term' in coefs:
                terms = coefs['term']
                if isinstance(terms, str):
                    terms = [terms]

                pdf.set_font('Helvetica', 'B', 9)
                pdf.cell(0, 5, '  Fixed Effects Coefficients:', 0, 1)
                pdf.set_font('Courier', '', 8)

                pdf.cell(80, 5, '  Term', 0, 0)
                pdf.cell(22, 5, 'Estimate', 0, 0, 'R')
                pdf.cell(22, 5, 'Std.Err', 0, 0, 'R')
                pdf.cell(18, 5, 'df', 0, 0, 'R')
                pdf.cell(18, 5, 't', 0, 0, 'R')
                pdf.cell(25, 5, 'p-value', 0, 1, 'R')

                n_terms = len(terms) if isinstance(terms, list) else 1

                for i in range(n_terms):
                    term = terms[i] if isinstance(terms, list) else terms
                    is_interaction = 'stim' in str(term).lower() and 'sex' in str(term).lower()

                    def get_val(key, idx):
                        v = coefs.get(key)
                        if v is None:
                            return '--'
                        if isinstance(v, list):
                            return v[idx] if idx < len(v) else '--'
                        return v

                    est = get_val('Estimate', i)
                    se = get_val('Std. Error', i)
                    df_val = get_val('df', i)
                    t_val = get_val('t value', i)
                    pval = get_val('Pr(>|t|)', i)
                    if pval == '--':
                        pval = get_val('Pr...t..', i)

                    if is_interaction:
                        pdf.set_font('Courier', 'B', 8)
                    else:
                        pdf.set_font('Courier', '', 8)

                    def fmt(v, decimals=3):
                        if isinstance(v, (int, float)):
                            if decimals == 4:
                                return f'{v:.4f}'
                            return f'{v:.3f}'
                        return str(v)[:10]

                    pdf.cell(80, 5, f'  {str(term)[:38]}', 0, 0)
                    pdf.cell(22, 5, fmt(est), 0, 0, 'R')
                    pdf.cell(22, 5, fmt(se), 0, 0, 'R')
                    pdf.cell(18, 5, fmt(df_val), 0, 0, 'R')
                    pdf.cell(18, 5, fmt(t_val), 0, 0, 'R')

                    p_str = fmt(pval, 4)
                    try:
                        p_float = float(pval)
                        if p_float < 0.001:
                            p_str = f'{p_float:.1e}'
                        elif p_float < 0.05:
                            p_str = f'{p_float:.4f} *'
                    except:
                        pass

                    pdf.cell(25, 5, p_str, 0, 1, 'R')

            pdf.ln(3)

            # Interpretation
            pdf.set_font('Helvetica', 'I', 9)
            pdf.set_x(pdf.l_margin)
            interpret = interpret_model(result)
            for line in interpret:
                safe_multi_cell(0, 4.5, line)
            pdf.ln(4)

        # Page break between model families
        pdf.add_page()

    # Summary page
    pdf.set_font('Helvetica', 'B', 14)
    pdf.cell(0, 10, 'Summary of Significant Stim x Sex Interactions', 0, 1, 'C')
    pdf.ln(3)

    sig_findings = []
    for family_label, results_list in all_model_results:
        for result in results_list:
            if result.get('skipped') or not result.get('converged'):
                continue
            # Check anova for stim:sex terms
            anova = result.get('anova', {})
            terms = anova.get('term', [])
            if isinstance(terms, str):
                terms = [terms]
            for i, term in enumerate(terms):
                if 'stim' in str(term).lower() and 'sex' in str(term).lower():
                    pvals = anova.get('Pr(>F)', anova.get('Pr..F.', []))
                    if isinstance(pvals, list) and i < len(pvals):
                        try:
                            p = float(pvals[i])
                            if p < 0.05:
                                fvals = anova.get('F value', [])
                                f = fvals[i] if isinstance(fvals, list) and i < len(fvals) else '?'
                                sig_findings.append({
                                    'model': result.get('label', '?'),
                                    'term': term,
                                    'F': f,
                                    'p': p,
                                })
                        except:
                            pass

    pdf.set_font('Helvetica', '', 10)
    pdf.set_x(pdf.l_margin)
    if sig_findings:
        for sf in sig_findings:
            p_str = f"{sf['p']:.1e}" if sf['p'] < 0.001 else f"{sf['p']:.4f}"
            txt = (f"* {sf['model']}: {sf['term']}, F={sf['F']:.2f}, p={p_str}"
                if isinstance(sf['F'], float) else
                f"* {sf['model']}: {sf['term']}, F={sf['F']}, p={p_str}")
            safe_multi_cell(0, 6, txt)
    else:
        pdf.cell(0, 6, 'No significant (p < .05) stim x sex interactions found.', 0, 1)

    pdf.ln(5)
    pdf.set_font('Helvetica', 'I', 9)
    safe_multi_cell(0, 5,
        'Note: All models use linear mixed-effects models (lmer) with '
        'Satterthwaite degrees of freedom. DV is baseline-corrected neural measure. '
        'PAC values were z-scored before modeling. '
        'Random effects: (1|Patient). Fixed effects vary by model type. '
        f'Groups with fewer than {MIN_SUBJECTS_PER_SEX} subjects per sex were excluded. '
        'Reference levels: sex=female, stim=nostim. '
        'Bold terms in tables indicate stim x sex interactions.'
    )

    pdf.output(output_path)
    print(f"\n  PDF saved: {output_path}")


def interpret_model(result):
    """Generate interpretation lines for a model result."""
    lines = []
    if not result.get('converged'):
        return ['Model did not converge.']

    anova = result.get('anova', {})
    terms = anova.get('term', [])
    if isinstance(terms, str):
        terms = [terms]

    # Look for stim:sex interactions
    for i, term in enumerate(terms):
        if 'stim' in str(term).lower() and 'sex' in str(term).lower():
            pvals = anova.get('Pr(>F)', anova.get('Pr..F.', []))
            if isinstance(pvals, list) and i < len(pvals):
                try:
                    p = float(pvals[i])
                    fvals = anova.get('F value', [])
                    f = fvals[i] if isinstance(fvals, list) and i < len(fvals) else '?'
                    if p < 0.05:
                        lines.append(
                            f'=> SIGNIFICANT {term}: F={f:.2f}, p={p:.4f}. '
                            f'Stimulation effects differ between men and women.'
                        )
                    elif p < 0.10:
                        lines.append(
                            f'=> Trend-level {term}: F={f:.2f}, p={p:.4f}.'
                        )
                    else:
                        lines.append(
                            f'=> Non-significant {term}: F={f:.2f}, p={p:.4f}. '
                            f'No evidence that stimulation effects differ by sex.'
                        )
                except:
                    pass

    # Main effect of stim
    for i, term in enumerate(terms):
        if str(term).lower().strip() == 'stim':
            pvals = anova.get('Pr(>F)', anova.get('Pr..F.', []))
            if isinstance(pvals, list) and i < len(pvals):
                try:
                    p = float(pvals[i])
                    fvals = anova.get('F value', [])
                    f = fvals[i] if isinstance(fvals, list) and i < len(fvals) else '?'
                    sig = 'significant' if p < 0.05 else 'non-significant'
                    lines.append(f'   Main effect of stim: F={f:.2f}, p={p:.4f} ({sig}).')
                except:
                    pass

    # Main effect of sex
    for i, term in enumerate(terms):
        if str(term).lower().strip() == 'sex':
            pvals = anova.get('Pr(>F)', anova.get('Pr..F.', []))
            if isinstance(pvals, list) and i < len(pvals):
                try:
                    p = float(pvals[i])
                    fvals = anova.get('F value', [])
                    f = fvals[i] if isinstance(fvals, list) and i < len(fvals) else '?'
                    sig = 'significant' if p < 0.05 else 'non-significant'
                    lines.append(f'   Main effect of sex: F={f:.2f}, p={p:.4f} ({sig}).')
                except:
                    pass

    if not lines:
        lines.append('Could not extract interaction terms from model output.')

    return lines


# =========================================================================
#  DATA LOADING (reuse from combined scripts)
# =========================================================================

def load_all_retrieval_power():
    from combined_retrieval_power import (
        load_blaes_retrieval, load_amme_retrieval, build_all_retrieval_data,
    )
    blaes = load_blaes_retrieval()
    amme = load_amme_retrieval()
    return build_all_retrieval_data(blaes, amme)


def load_all_encoding_power():
    from combined_encoding_power import (
        load_blaes_encoding, load_amme_encoding, build_all_encoding_data,
    )
    blaes = load_blaes_encoding()
    amme = load_amme_encoding()
    return build_all_encoding_data(blaes, amme)


def load_all_retrieval_coherence():
    from combined_retrieval_coherence import (
        load_blaes_retrieval as lb, load_amme_retrieval as la,
    )
    blaes = lb()
    amme = la()
    # Merge
    def md(*ds):
        m = {}
        for d in ds:
            if not d:
                continue
            for k, v in d.items():
                m.setdefault(k, {}).update(v)
        return m

    fp = blaes.get('freqs_post') if blaes.get('freqs_post') is not None else amme.get('freqs_post')
    fd = blaes.get('freqs_diff') if blaes.get('freqs_diff') is not None else amme.get('freqs_diff')
    return {
        'freqs_post': fp, 'freqs_diff': fd,
        'mlmr_export_frames': blaes.get('mlmr_export_frames', []) + amme.get('mlmr_export_frames', []),
        'bc_nostim_rem': md(blaes.get('bc_nostim_rem', {}), amme.get('bc_nostim_rem', {})),
        'bc_nostim_forg': md(blaes.get('bc_nostim_forg', {}), amme.get('bc_nostim_forg', {})),
        'bc_stim_rem': md(blaes.get('bc_stim_rem', {}), amme.get('bc_stim_rem', {})),
        'bc_stim_forg': md(blaes.get('bc_stim_forg', {}), amme.get('bc_stim_forg', {})),
    }


def load_all_encoding_coherence():
    from combined_encoding_coherence import (
        load_blaes_encoding as lb, load_amme_encoding as la,
    )
    blaes = lb()
    amme = la()
    def md(*ds):
        m = {}
        for d in ds:
            if not d:
                continue
            for k, v in d.items():
                m.setdefault(k, {}).update(v)
        return m

    fp = blaes.get('freqs_post') if blaes.get('freqs_post') is not None else amme.get('freqs_post')
    fd = blaes.get('freqs_diff') if blaes.get('freqs_diff') is not None else amme.get('freqs_diff')
    return {
        'freqs_post': fp, 'freqs_diff': fd,
        'mlmr_export_frames': blaes.get('mlmr_export_frames', []) + amme.get('mlmr_export_frames', []),
    }


def load_all_retrieval_pac():
    from combined_retrieval_pac import load_grouped_retrieval_pac_data
    return load_grouped_retrieval_pac_data()['all']


def load_all_encoding_pac():
    from combined_encoding_pac import load_grouped_encoding_pac_data
    return load_grouped_encoding_pac_data()['all']


# =========================================================================
#  MAIN ANALYSIS PIPELINE
# =========================================================================

def run_phase_analysis(phase_label, power_data, coherence_data, pac_data, sex_map):
    """Run all 6 model families for one phase. Return list of (label, results)."""
    all_results = []

    # --- 1. Power: per-region ---
    print(f"\n  [1/6] Power -- per-region models...")
    power_df = build_trial_level_df(
        power_data.get('mlmr_export_frames', []),
        sex_map, POWER_REGIONS, is_pair=False,
    )
    if power_df.empty:
        print("    WARNING: No power trial-level data.")
        all_results.append(('1. Power -- Per-Region (freq band x stim x sex)', []))
    else:
        print(f"    Power trial-level: {len(power_df)} rows, "
              f"{power_df['Patient'].nunique()} patients")
        r1 = run_per_region_models(power_df, 'Power', POWER_REGIONS)
        all_results.append(('1. Power -- Per-Region (freq band x stim x sex)', r1))

    # --- 2. Power: per-frequency ---
    print(f"\n  [2/6] Power -- per-frequency models...")
    if power_df.empty:
        all_results.append(('2. Power -- Per-Frequency (region x stim x sex)', []))
    else:
        r2 = run_per_freq_models(power_df, 'Power', POWER_REGIONS)
        all_results.append(('2. Power -- Per-Frequency (region x stim x sex)', r2))

    # --- 3. Coherence: per-region-pair ---
    print(f"\n  [3/6] Coherence -- per-region-pair models...")
    coh_df = build_trial_level_df(
        coherence_data.get('mlmr_export_frames', []),
        sex_map, COH_PAIRS, is_pair=True,
    )
    if coh_df.empty:
        print("    WARNING: No coherence trial-level data.")
        all_results.append(('3. Coherence -- Per-Region-Pair (freq band x stim x sex)', []))
    else:
        print(f"    Coherence trial-level: {len(coh_df)} rows, "
              f"{coh_df['Patient'].nunique()} patients")
        r3 = run_per_region_models(coh_df, 'Coherence', COH_PAIRS)
        all_results.append(('3. Coherence -- Per-Region-Pair (freq band x stim x sex)', r3))

    # --- 4. Coherence: per-frequency ---
    print(f"\n  [4/6] Coherence -- per-frequency models...")
    if coh_df.empty:
        all_results.append(('4. Coherence -- Per-Frequency (region pair x stim x sex)', []))
    else:
        r4 = run_per_freq_models(coh_df, 'Coherence', COH_PAIRS)
        all_results.append(('4. Coherence -- Per-Frequency (region pair x stim x sex)', r4))

    # --- 5. PAC: per-region-pair ---
    print(f"\n  [5/6] PAC -- per-region-pair models...")
    pac_df = build_pac_patient_level_df(pac_data, sex_map)
    if pac_df.empty:
        print("    WARNING: No PAC data.")
        all_results.append(('5. PAC -- Per-Region-Pair (freq band x stim x sex)', []))
    else:
        # Filter to target pairs
        pac_pairs_avail = sorted(pac_df['Region'].unique())
        target_pac_pairs = [p for p in pac_pairs_avail
                            if all(part in PAC_TARGET_REGIONS for part in p.split('_'))]
        print(f"    PAC patient-level (z-scored): {len(pac_df)} rows, "
              f"{pac_df['Patient'].nunique()} patients, "
              f"{len(target_pac_pairs)} target pairs")
        r5 = run_per_region_models(pac_df, 'PAC', target_pac_pairs)
        all_results.append(('5. PAC -- Per-Region-Pair (freq band x stim x sex)', r5))

    # --- 6. PAC: per-frequency ---
    print(f"\n  [6/6] PAC -- per-frequency models...")
    if pac_df.empty:
        all_results.append(('6. PAC -- Per-Frequency (region pair x stim x sex)', []))
    else:
        r6 = run_per_freq_models(pac_df, 'PAC', target_pac_pairs)
        all_results.append(('6. PAC -- Per-Frequency (region pair x stim x sex)', r6))

    return all_results


def main():
    print("=" * 60)
    print("Stim x Sex GLMM Analysis")
    print("=" * 60)

    sex_map = load_sex_map()
    n_m = sum(1 for v in sex_map.values() if v == 'male')
    n_f = sum(1 for v in sex_map.values() if v == 'female')
    print(f"Sex metadata: {n_m} male, {n_f} female")

    # === RETRIEVAL ===
    print("\n" + "=" * 60)
    print("RETRIEVAL PHASE")
    print("=" * 60)

    print("\n  Loading retrieval data...")
    ret_power = load_all_retrieval_power()
    ret_coh = load_all_retrieval_coherence()
    ret_pac = load_all_retrieval_pac()

    ret_results = run_phase_analysis('Retrieval', ret_power, ret_coh, ret_pac, sex_map)
    ret_pdf = os.path.join(OUTPUT_DIR, 'stim_x_sex_glmm_retrieval.pdf')
    generate_pdf_report('Retrieval', ret_results, ret_pdf)

    # === ENCODING ===
    print("\n" + "=" * 60)
    print("ENCODING PHASE")
    print("=" * 60)

    print("\n  Loading encoding data...")
    enc_power = load_all_encoding_power()
    enc_coh = load_all_encoding_coherence()
    enc_pac = load_all_encoding_pac()

    enc_results = run_phase_analysis('Encoding', enc_power, enc_coh, enc_pac, sex_map)
    enc_pdf = os.path.join(OUTPUT_DIR, 'stim_x_sex_glmm_encoding.pdf')
    generate_pdf_report('Encoding', enc_results, enc_pdf)

    print("\n" + "=" * 60)
    print(f"Done! Reports saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == '__main__':
    main()
