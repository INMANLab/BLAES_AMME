#!/usr/bin/env python
"""
run_all_analyses.py
====================
Standalone orchestrator that runs every cell of the dissertation analysis
matrix in parallel and post-processes the resulting trial-level CSVs to
mark Benjamini-Hochberg FDR-significant comparisons.

Matrix
------
    phase     ∈ {encoding, retrieval}
    subset    ∈ {balanced_memory, onesecstim}
    analysis  ∈ {power, coherence, pac}
    band      ∈ {theta = 4–8 Hz, slow_gamma = 30–55 Hz}     # labels used in plots: "theta" and "slow gamma"

The (encoding, balanced_memory, *) cell is provided by
``balanced_memory_code/balanced_memory_trials.py``; the (*, onesecstim, *)
cells are provided by ``onesecstim_code/onesecstim_trials.py``. Both
runners load via the existing pipelines, apply their respective trial
filters, and emit BC trial-level MLMR CSVs into:
    outputs/<subset>_trials/csvs/

This script also runs ``permutation/build_onesec_filtered_csvs.py`` once
to ensure the onesec-filtered CSVs exist before the onesecstim runner is
called (the runner re-aggregates from raw via patched pd.read_csv, but
the CSVs are useful for downstream MLM/permutation pipelines).

Region exclusions (applied EVERYWHERE)
--------------------------------------
    EXCLUDED_REGIONS = {'MTL', 'PHG'}  + any region containing 'PNAS'
All other ROIs are kept (BLA, ALLHPC, CA, DG, HPC, EC, PRC, …).

FDR families (BH applied independently within each)
---------------------------------------------------
    F1 (BLA → others)   : BLA_ALLHPC, BLA_EC, BLA_PRC, BLA_BLA           # 4 comparisons
    F2 (cortical MTL)   : ALLHPC, EC, PRC                                # 3 comparisons
    F3 (HPC subfields+BLA): CA, DG, HPC, BLA                             # 4 comparisons
    F4 (HPC subfields+cortex): CA, DG, HPC, EC, PRC                      # 5 comparisons

Each family is evaluated SEPARATELY for every (phase × subset × analysis
× band × test_type) cell, where test_type ∈ {stim_vs_nostim,
remembered_vs_forgotten}. The marks placed on the resulting summary
table are:
    *  → BH-FDR q < 0.05 (within the family)
    +  → uncorrected p < 0.05 (and not *)
    (blank) for p ≥ 0.05

Plot palette (used downstream — informational only here)
--------------------------------------------------------
    endogenous memory  : remembered = '#FFD43B' (yellow),
                         forgotten  = '#9C7AC9' (purple)
    stim vs nostim     : stim       = '#E63946' (red),
                         nostim     = '#1D3557' (blue)

Usage
-----
    # Run everything (recommended; uses subprocess parallelism)
    python run_all_analyses.py

    # Run a subset
    python run_all_analyses.py --subset onesecstim --analysis power
    python run_all_analyses.py --phase encoding --subset balanced_memory

    # Just the FDR post-processing (re-mark significance from existing CSVs)
    python run_all_analyses.py --post-only

    # Assemble PDFs from existing figures + significance table
    python run_all_analyses.py --build-pdfs
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# ===========================================================================
# Configuration (single source of truth — edit here, not in jobs)
# ===========================================================================

PHASES = ('encoding', 'retrieval')
SUBSETS = ('balanced_memory', 'onesecstim')
ANALYSES = ('power', 'coherence', 'pac')

# Band labels and ranges. We say "slow gamma" everywhere; the underlying
# pipelines already use 30–55 Hz under that label.
BANDS = {
    'theta':      (4.0,  8.0),
    'slow_gamma': (30.0, 55.0),
}

# Region exclusions — applied to the post-processor and respected by the
# runner scripts. NEVER include MTL, PHG, or any PNAS variant in any analysis.
EXCLUDED_REGIONS = {'MTL', 'PHG'}
EXCLUDED_REGION_SUBSTRINGS = ('PNAS',)

# FDR families (BH applied separately within each). Each entry lists the
# Region values to include from the MLMR CSV. Pair regions (e.g. BLA_ALLHPC)
# are matched as-is for coherence/PAC; single-region names match power and
# the diagonal of pair tables (BLA_BLA, etc.).
FDR_FAMILIES = {
    'F1_bla_to_others':       ['BLA_ALLHPC', 'BLA_EC', 'BLA_PRC', 'BLA_BLA'],
    'F2_cortical_mtl':        ['ALLHPC', 'EC', 'PRC'],
    'F3_hpc_subfields_bla':   ['CA', 'DG', 'HPC', 'BLA'],
    'F4_hpc_subfields_cortex': ['CA', 'DG', 'HPC', 'EC', 'PRC'],
}
FDR_ALPHA = 0.05

# Plot color palette (referenced by downstream plot scripts; declared here
# so the orchestrator is the single source of palette truth).
COLORS = {
    'remembered': '#FFD43B',  # yellow
    'forgotten':  '#9C7AC9',  # purple
    'stim':       '#E63946',  # red
    'nostim':     '#1D3557',  # blue
}

OUTPUTS_DIR = os.path.join(PROJECT_ROOT, 'outputs')
SUMMARY_OUT = os.path.join(OUTPUTS_DIR, 'significance_summary.csv')
SUMMARY_TXT = os.path.join(OUTPUTS_DIR, 'significance_summary.txt')

RUNNERS = {
    'balanced_memory': os.path.join(
        PROJECT_ROOT, 'balanced_memory_code', 'balanced_memory_trials.py'),
    'onesecstim': os.path.join(
        PROJECT_ROOT, 'onesecstim_code', 'onesecstim_trials.py'),
}
ONESEC_CSV_BUILDER = os.path.join(
    PROJECT_ROOT, 'permutation', 'build_onesec_filtered_csvs.py')


# ===========================================================================
# Job dispatch — each cell runs the appropriate runner as a subprocess
# ===========================================================================

def _job_label(subset, analysis):
    return f'{subset}/{analysis}'


def _build_subprocess_cmd(subset, analysis):
    """Return the subprocess command for a (subset, analysis) cell.

    Both runners drive all phases (encoding+retrieval) inside a single
    invocation. To keep granularity at the (subset, analysis) level we
    set MEASURES_TO_RUN env var that the runners may honour, then fall
    back to invoking the full runner if that env hook is absent. The
    onesecstim runner already supports this; balanced_memory_trials.py
    does not, so we invoke it once for the full power+coherence+pac
    pipeline (see _ensure_balanced_runs_once).
    """
    runner = RUNNERS[subset]
    return [sys.executable, runner], {
        'MEASURE_TO_RUN': analysis,  # consumed by onesecstim runner if set
    }


def _run_balanced_full():
    """The balanced_memory runner is monolithic — invoke it once."""
    print('\n>>> balanced_memory: launching full pipeline (power+coherence+pac)')
    rc = subprocess.call([sys.executable, RUNNERS['balanced_memory']])
    return ('balanced_memory', 'all', rc)


def _run_onesec_cell(analysis):
    """Run a single onesecstim measure (power|coherence|pac) in isolation."""
    print(f'\n>>> onesecstim/{analysis}: launching')
    env = os.environ.copy()
    env['MEASURE_TO_RUN'] = analysis
    cmd = [sys.executable, '-c',
           'import sys; sys.path.insert(0, '
           f'{repr(os.path.dirname(RUNNERS["onesecstim"]))}); '
           'from onesecstim_trials import main; '
           f'main(measures=({repr(analysis)},))']
    rc = subprocess.call(cmd, env=env)
    return ('onesecstim', analysis, rc)


def _ensure_onesec_csvs():
    """Build the trial-filtered onesec CSVs if they don't already exist."""
    sentinel = os.path.join(
        OUTPUTS_DIR, 'csvs', 'onesec',
        'combined_encoding_power_onesec_mlmr_input.csv')
    if os.path.exists(sentinel):
        print('  onesec MLMR CSVs already present — skipping rebuild.')
        return 0
    print('\n>>> Building onesec-filtered MLMR CSVs (one-time prerequisite)...')
    return subprocess.call([sys.executable, ONESEC_CSV_BUILDER])


def run_jobs(phases, subsets, analyses, max_workers=None):
    """Dispatch all selected (subset, analysis) cells in parallel."""
    # The balanced runner is monolithic; if it's selected at all we run it
    # once. The onesecstim runner is per-measure-parallelizable.
    tasks = []
    if 'balanced_memory' in subsets:
        tasks.append(('balanced_memory', 'all'))
    if 'onesecstim' in subsets:
        _ensure_onesec_csvs()
        for a in analyses:
            tasks.append(('onesecstim', a))

    if not tasks:
        print('No jobs to run.')
        return []

    max_workers = max_workers or min(len(tasks), os.cpu_count() or 2)
    print(f'\nLaunching {len(tasks)} job(s) with {max_workers} parallel workers')
    print('Phases dispatched:', ', '.join(phases))

    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = {}
        for subset, analysis in tasks:
            if subset == 'balanced_memory':
                fut = ex.submit(_run_balanced_full)
            else:
                fut = ex.submit(_run_onesec_cell, analysis)
            futures[fut] = (subset, analysis)

        for fut in as_completed(futures):
            subset, analysis = futures[fut]
            try:
                result = fut.result()
                results.append(result)
                status = 'ok' if result[2] == 0 else f'rc={result[2]}'
                print(f'<<< {_job_label(subset, analysis)} done [{status}]')
            except Exception as exc:  # noqa: BLE001
                print(f'!!! {_job_label(subset, analysis)} failed: {exc}')
                results.append((subset, analysis, 1))
    return results


# ===========================================================================
# FDR post-processor — scans MLMR CSVs and marks * / + on per-region tests
# ===========================================================================

def _is_excluded_region(roi):
    s = str(roi)
    if s in EXCLUDED_REGIONS:
        return True
    if any(sub in s.upper() for sub in EXCLUDED_REGION_SUBSTRINGS):
        return True
    if '_' in s:
        for part in s.split('_'):
            if part in EXCLUDED_REGIONS:
                return True
            if any(sub in part.upper() for sub in EXCLUDED_REGION_SUBSTRINGS):
                return True
    return False


def _band_average(df, freq_lo, freq_hi):
    """Mean across diff_Freq_* columns within [lo, hi] Hz."""
    diff_cols = [c for c in df.columns if c.startswith('diff_Freq_')]
    if not diff_cols:
        return None
    freqs = np.array([float(c.replace('diff_Freq_', '')) for c in diff_cols])
    mask = (freqs >= freq_lo) & (freqs <= freq_hi)
    if not mask.any():
        return None
    cols = [c for c, m in zip(diff_cols, mask) if m]
    return df[cols].mean(axis=1)


def _paired_test(group_a, group_b):
    """Two-sided paired t-test. Returns (n, t, p) or (0, nan, nan)."""
    from scipy import stats
    common = set(group_a.index) & set(group_b.index)
    if len(common) < 3:
        return (len(common), float('nan'), float('nan'))
    a = group_a.loc[sorted(common)].to_numpy()
    b = group_b.loc[sorted(common)].to_numpy()
    valid = ~(np.isnan(a) | np.isnan(b))
    a, b = a[valid], b[valid]
    if len(a) < 3:
        return (len(a), float('nan'), float('nan'))
    res = stats.ttest_rel(a, b)
    return (len(a), float(res.statistic), float(res.pvalue))


def _per_region_tests(df_band):
    """For one (cohort × band) frame, run per-region paired t-tests.

    Returns a long-format DataFrame with columns:
        Region, test_type, n, t, p_raw
    """
    rows = []
    for region, sub in df_band.groupby('Region'):
        if _is_excluded_region(region):
            continue
        # subject-level mean of band-averaged BC values per (Patient, trial_type, yes_or_no)
        agg = (sub.groupby(['Patient', 'trial_type', 'yes_or_no'])['band_value']
               .mean().reset_index())

        # Test 1: stim vs nostim (paired by Patient, collapsed across yes/no).
        t1 = (agg.groupby(['Patient', 'trial_type'])['band_value']
              .mean().unstack('trial_type'))
        if {'stim', 'nostim'}.issubset(t1.columns):
            n, t, p = _paired_test(t1['stim'], t1['nostim'])
            rows.append({'Region': region, 'test_type': 'stim_vs_nostim',
                         'n': n, 't': t, 'p_raw': p})

        # Test 2: remembered vs forgotten within nostim only (paired by Patient).
        nostim = agg[agg['trial_type'] == 'nostim']
        t2 = (nostim.groupby(['Patient', 'yes_or_no'])['band_value']
              .mean().unstack('yes_or_no'))
        if {'yes', 'no'}.issubset(t2.columns):
            n, t, p = _paired_test(t2['yes'], t2['no'])
            rows.append({'Region': region, 'test_type': 'remembered_vs_forgotten',
                         'n': n, 't': t, 'p_raw': p})

    return pd.DataFrame(rows)


def _bh_fdr(pvals, alpha=FDR_ALPHA):
    """Benjamini-Hochberg adjusted q-values for a vector of p-values.

    Returns q-values aligned to the input. NaNs are passed through.
    """
    p = np.asarray(pvals, dtype=float)
    n = np.sum(~np.isnan(p))
    if n == 0:
        return np.full_like(p, np.nan)
    order = np.argsort(np.where(np.isnan(p), 1.0, p))
    ranks = np.empty(len(p), dtype=int)
    ranks[order] = np.arange(1, len(p) + 1)
    q_raw = p * n / ranks
    q = np.full_like(p, np.nan)
    sorted_q = q_raw[order]
    # enforce monotonicity
    for i in range(len(sorted_q) - 2, -1, -1):
        if not np.isnan(sorted_q[i]) and not np.isnan(sorted_q[i + 1]):
            sorted_q[i] = min(sorted_q[i], sorted_q[i + 1])
    sorted_q = np.minimum(sorted_q, 1.0)
    q[order] = sorted_q
    return q


def _mark(p_raw, q_fdr):
    if np.isnan(p_raw):
        return ''
    if not np.isnan(q_fdr) and q_fdr < FDR_ALPHA:
        return '*'
    if p_raw < FDR_ALPHA:
        return '+'
    return ''


def _classify_family(region):
    """Return the FDR family key whose region list contains this region.
    Returns None if the region isn't in any family.
    """
    for fam_key, members in FDR_FAMILIES.items():
        if region in members:
            return fam_key
    return None


def _csv_metadata(path):
    """Parse subset/phase/analysis/cohort from the MLMR filename stem."""
    stem = os.path.basename(path).replace('_mlmr_input.csv', '')
    parts = stem.split('_')
    # Expected stems:
    #   balanced_<phase>_<analysis>_<cohort>
    #   onesec_<phase>_<analysis>_<cohort>
    if not parts:
        return None
    if parts[0] == 'balanced':
        subset = 'balanced_memory'
        rest = parts[1:]
    elif parts[0] == 'onesec':
        subset = 'onesecstim'
        rest = parts[1:]
    else:
        return None
    if len(rest) < 3:
        return None
    phase, analysis, cohort = rest[0], rest[1], '_'.join(rest[2:])
    return {'subset': subset, 'phase': phase, 'analysis': analysis,
            'cohort': cohort, 'path': path}


def _discover_mlmr_csvs():
    paths = []
    for subset_dir in ('balanced_memory_trials', 'onesecstim_trials'):
        d = os.path.join(OUTPUTS_DIR, subset_dir, 'csvs')
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if fn.endswith('_mlmr_input.csv'):
                paths.append(os.path.join(d, fn))
    return paths


def post_process_fdr():
    """Read every MLMR CSV, run per-region tests for each band, BH-correct
    per FDR family, and write the master significance summary.
    """
    print('\n' + '=' * 60)
    print('FDR Post-Processor (BH within each family)')
    print('=' * 60)

    csv_paths = _discover_mlmr_csvs()
    if not csv_paths:
        print('  No MLMR CSVs found — run analyses first.')
        return None

    all_rows = []
    for path in csv_paths:
        meta = _csv_metadata(path)
        if not meta:
            print(f'  [skip] could not parse {os.path.basename(path)}')
            continue
        df = pd.read_csv(path)
        if df.empty or 'Region' not in df.columns:
            continue
        df = df[~df['Region'].apply(_is_excluded_region)].copy()
        if df.empty:
            continue
        if 'yes_or_no' not in df.columns or 'trial_type' not in df.columns:
            continue

        for band, (lo, hi) in BANDS.items():
            ba = _band_average(df, lo, hi)
            if ba is None:
                continue
            df_band = df[['Patient', 'Region', 'trial_type', 'yes_or_no']].copy()
            df_band['band_value'] = ba

            tests = _per_region_tests(df_band)
            if tests.empty:
                continue
            tests['family'] = tests['Region'].apply(_classify_family)
            tests = tests[tests['family'].notna()].copy()
            if tests.empty:
                continue

            # BH-FDR within each (family × test_type)
            tests['q_fdr'] = np.nan
            for (fam, ttype), grp in tests.groupby(['family', 'test_type']):
                q = _bh_fdr(grp['p_raw'].to_numpy())
                tests.loc[grp.index, 'q_fdr'] = q

            tests['mark'] = [
                _mark(p, q) for p, q in zip(tests['p_raw'], tests['q_fdr'])
            ]

            tests.insert(0, 'band',     band)
            tests.insert(0, 'cohort',   meta['cohort'])
            tests.insert(0, 'analysis', meta['analysis'])
            tests.insert(0, 'phase',    meta['phase'])
            tests.insert(0, 'subset',   meta['subset'])
            all_rows.append(tests)

    if not all_rows:
        print('  No significance rows generated.')
        return None

    summary = pd.concat(all_rows, ignore_index=True)
    summary = summary[[
        'subset', 'phase', 'analysis', 'cohort', 'band', 'family',
        'test_type', 'Region', 'n', 't', 'p_raw', 'q_fdr', 'mark',
    ]]
    summary.to_csv(SUMMARY_OUT, index=False)
    print(f'  wrote {SUMMARY_OUT}  ({len(summary)} rows)')

    # Compact text view: one significant row per line, grouped by family.
    sig_only = summary[summary['mark'] != ''].copy()
    with open(SUMMARY_TXT, 'w', encoding='utf-8') as h:
        h.write('Significance summary (BH-FDR within family; * q<.05, + raw p<.05)\n')
        h.write('=' * 80 + '\n\n')
        if sig_only.empty:
            h.write('No comparisons met p<.05.\n')
        else:
            keys = ['subset', 'phase', 'analysis', 'cohort', 'band', 'family',
                    'test_type']
            for vals, grp in sig_only.groupby(keys):
                h.write(' | '.join(f'{k}={v}' for k, v in zip(keys, vals)) + '\n')
                for _, row in grp.iterrows():
                    h.write(f"    {row['mark']:>1}  {row['Region']:<14}"
                            f"  n={int(row['n']):>3}  t={row['t']:+.3f}"
                            f"  p={row['p_raw']:.4f}  q={row['q_fdr']:.4f}\n")
                h.write('\n')
    print(f'  wrote {SUMMARY_TXT}')
    return summary


# ===========================================================================
# PDF assembly — one PDF per (subset × phase × analysis × cohort) cell.
# Bundles every PNG in that cell's output folder and appends the cell's
# significance table at the end. Triggered by --build-pdfs.
# ===========================================================================

PDF_FIGURE_EXTS = ('.png',)


def _cell_output_dir(subset, phase, analysis, cohort):
    """Map a matrix cell to its existing figure folder."""
    subset_dir = (
        'balanced_memory_trials' if subset == 'balanced_memory'
        else 'onesecstim_trials'
    )
    return os.path.join(
        OUTPUTS_DIR, subset_dir, f'{phase}_{analysis}', cohort)


def _collect_pngs(cell_dir):
    """Return PNGs in cell_dir + cell_dir/endogenous_only, sorted."""
    pngs = []
    for root in (cell_dir, os.path.join(cell_dir, 'endogenous_only')):
        if not os.path.isdir(root):
            continue
        for fn in sorted(os.listdir(root)):
            if fn.lower().endswith(PDF_FIGURE_EXTS):
                pngs.append(os.path.join(root, fn))
    return pngs


def _safe_pdf_text(s):
    """fpdf's core fonts are latin-1 only — strip anything outside it."""
    return str(s).encode('latin-1', errors='replace').decode('latin-1')


def _add_significance_pages(pdf, summary_df, subset, phase, analysis, cohort):
    """Append a table of marked rows for this cell."""
    if summary_df is None or summary_df.empty:
        return
    sub = summary_df[
        (summary_df['subset'] == subset)
        & (summary_df['phase'] == phase)
        & (summary_df['analysis'] == analysis)
        & (summary_df['cohort'] == cohort)
    ].copy()
    if sub.empty:
        return
    pdf.add_page()
    pdf.set_font('Helvetica', 'B', 13)
    pdf.cell(0, 8, _safe_pdf_text(
        f'Significance — {subset} | {phase} | {analysis} | {cohort}'),
        ln=1)
    pdf.set_font('Helvetica', '', 9)
    pdf.cell(0, 5, '* = BH-FDR q<.05  +  = uncorrected p<.05  (within family)',
             ln=1)
    pdf.ln(2)
    pdf.set_font('Courier', 'B', 8)
    headers = ['band', 'family', 'test_type', 'Region', 'n', 't', 'p_raw',
               'q_fdr', 'mark']
    widths = [22, 26, 32, 26, 10, 14, 16, 16, 12]
    for h, w in zip(headers, widths):
        pdf.cell(w, 5, h, border=1)
    pdf.ln()
    pdf.set_font('Courier', '', 8)
    for _, row in sub.sort_values(
            ['band', 'family', 'test_type', 'Region']).iterrows():
        cells = [
            row['band'], row['family'], row['test_type'], row['Region'],
            f"{int(row['n'])}",
            f"{row['t']:+.3f}" if pd.notna(row['t']) else 'nan',
            f"{row['p_raw']:.4f}" if pd.notna(row['p_raw']) else 'nan',
            f"{row['q_fdr']:.4f}" if pd.notna(row['q_fdr']) else 'nan',
            row['mark'],
        ]
        for v, w in zip(cells, widths):
            pdf.cell(w, 4.6, _safe_pdf_text(v)[:max(1, int(w / 1.6))],
                     border=1)
        pdf.ln()


def _build_one_pdf(subset, phase, analysis, cohort, summary_df, out_dir):
    from fpdf import FPDF

    cell_dir = _cell_output_dir(subset, phase, analysis, cohort)
    pngs = _collect_pngs(cell_dir)
    if not pngs:
        return None

    pdf = FPDF(orientation='L', unit='mm', format='Letter')
    pdf.set_auto_page_break(auto=True, margin=8)

    # Cover
    pdf.add_page()
    pdf.set_font('Helvetica', 'B', 22)
    pdf.cell(0, 14, _safe_pdf_text(
        f'{subset.replace("_", " ").title()} — {phase.title()} {analysis.title()}'),
        ln=1, align='C')
    pdf.set_font('Helvetica', '', 13)
    pdf.cell(0, 8, _safe_pdf_text(f'Cohort: {cohort.upper()}'),
             ln=1, align='C')
    pdf.cell(0, 6, _safe_pdf_text(
        f'Bands: {", ".join(BANDS)}   Excluded regions: MTL, PHG, *PNAS*'),
        ln=1, align='C')
    pdf.cell(0, 6, _safe_pdf_text(f'Source folder: {cell_dir}'),
             ln=1, align='C')
    pdf.ln(4)
    pdf.set_font('Helvetica', '', 10)
    pdf.cell(0, 5, _safe_pdf_text(f'{len(pngs)} figures included'),
             ln=1, align='C')

    # One figure per page, scaled to fit page minus margins.
    page_w = pdf.w - 16
    page_h = pdf.h - 24
    for png in pngs:
        pdf.add_page()
        pdf.set_font('Helvetica', '', 9)
        pdf.cell(0, 5, _safe_pdf_text(os.path.basename(png)), ln=1)
        try:
            pdf.image(png, x=8, y=15, w=page_w, h=page_h - 6)
        except Exception as exc:  # noqa: BLE001
            pdf.set_font('Helvetica', '', 10)
            pdf.cell(0, 6, _safe_pdf_text(f'[image failed: {exc}]'), ln=1)

    _add_significance_pages(pdf, summary_df, subset, phase, analysis, cohort)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(
        out_dir, f'{phase}_{analysis}_{cohort}.pdf')
    pdf.output(out_path)
    return out_path


def build_pdfs(subsets=SUBSETS, phases=PHASES, analyses=ANALYSES,
               cohorts=('blaes', 'amme', 'all')):
    """Assemble per-cell PDFs. Reads significance_summary.csv (re-running
    the post-processor first if it's missing) so each PDF carries its own
    BH-FDR-marked table."""
    print('\n' + '=' * 60)
    print('PDF Assembly')
    print('=' * 60)

    if not os.path.exists(SUMMARY_OUT):
        print('  significance_summary.csv missing — running post-processor first')
        post_process_fdr()
    summary_df = (pd.read_csv(SUMMARY_OUT) if os.path.exists(SUMMARY_OUT)
                  else None)

    written = []
    for subset, phase, analysis, cohort in product(
            subsets, phases, analyses, cohorts):
        subset_dir = ('balanced_memory_trials' if subset == 'balanced_memory'
                      else 'onesecstim_trials')
        out_dir = os.path.join(OUTPUTS_DIR, subset_dir, 'pdfs')
        try:
            path = _build_one_pdf(subset, phase, analysis, cohort,
                                  summary_df, out_dir)
        except Exception as exc:  # noqa: BLE001
            print(f'  [fail] {subset}/{phase}/{analysis}/{cohort}: {exc}')
            continue
        if path:
            print(f'  wrote {path}')
            written.append(path)
        else:
            print(f'  [skip] {subset}/{phase}/{analysis}/{cohort} '
                  f'(no figures yet)')
    print(f'  total PDFs written: {len(written)}')
    return written


# ===========================================================================
# Entry point
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--phase', choices=PHASES, action='append',
                   help='Restrict to one phase (default: both)')
    p.add_argument('--subset', choices=SUBSETS, action='append',
                   help='Restrict to one subset (default: both)')
    p.add_argument('--analysis', choices=ANALYSES, action='append',
                   help='Restrict to one analysis (default: all)')
    p.add_argument('--workers', type=int, default=None,
                   help='Parallel worker count (default: min(jobs, ncpu))')
    p.add_argument('--post-only', action='store_true',
                   help='Skip job dispatch; only run FDR post-processor')
    p.add_argument('--no-post', action='store_true',
                   help='Run jobs but skip FDR post-processor')
    p.add_argument('--build-pdfs', action='store_true',
                   help='Assemble per-cell PDFs from existing figures + '
                        'significance table (can be combined with any other '
                        'flag, or used standalone after the main pipeline)')
    return p.parse_args()


def main():
    args = parse_args()
    phases = tuple(args.phase) if args.phase else PHASES
    subsets = tuple(args.subset) if args.subset else SUBSETS
    analyses = tuple(args.analysis) if args.analysis else ANALYSES

    print('=' * 60)
    print('AMME_BLAES — Master Analysis Orchestrator')
    print('=' * 60)
    print(f'Phases:    {", ".join(phases)}')
    print(f'Subsets:   {", ".join(subsets)}')
    print(f'Analyses:  {", ".join(analyses)}')
    print(f'Bands:     {", ".join(BANDS)}')
    print(f'Excluded regions: MTL, PHG, *PNAS*  (all others kept)')
    print(f'Output root: {OUTPUTS_DIR}')

    only_pdfs = args.build_pdfs and args.post_only
    if not args.post_only and not only_pdfs:
        run_jobs(phases, subsets, analyses, max_workers=args.workers)

    if not args.no_post:
        post_process_fdr()

    if args.build_pdfs:
        build_pdfs(subsets=subsets, phases=phases, analyses=analyses)

    print('\nAll done.')


if __name__ == '__main__':
    main()
