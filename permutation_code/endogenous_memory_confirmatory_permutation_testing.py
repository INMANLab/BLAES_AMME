#!/usr/bin/env python
"""Confirmatory paired permutation tests for endogenous memory at retrieval.

Same paired sign-flip permutation procedure as the stim-vs-no-stim analysis,
but the metric is the per-subject endogenous memory contrast:
    diff = (rem - forg)  averaged across stim and no-stim
    rem  = mean(bc_stim_rem,  bc_nostim_rem)
    forg = mean(bc_stim_forg, bc_nostim_forg)

For each modality (power, coherence, PAC):
1. Per subject, compute the within-subject (rem - forg) of the band-averaged
   baseline-corrected value within Theta (4-8 Hz) or Slow gamma (35-50 Hz).
2. Compute the observed group t-statistic per (region, band).
3. Run 1000 sign-flip permutations: each subject independently has the sign of
   its (rem - forg) flipped (Bernoulli 1/2). Recompute group t.
4. Family-wise correction across all (region x band) tests uses max-statistic
   permutation: max |t| across the family at each iteration forms the null.

Outputs:
  outputs/parametric retrieval test/endogenous memory/confirmatory permutation testing/
    power/permutation.pdf, permutation.csv
    coherence/permutation.pdf, permutation.csv
    PAC/permutation.pdf, permutation.csv

Regions containing 'PNAS' are excluded.
"""

from pathlib import Path
import shutil
import textwrap

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from endogenous_memory_paired_ttests_report import (
    PAC_KEYS,
    POWER_COH_KEYS,
    collect_endogenous_subject_band,
    is_excluded_region,
    load_coherence_data,
    load_pac_data,
    load_power_data,
)


matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_BASE = SCRIPT_DIR / "outputs" / "parametric retrieval test" / "endogenous memory" / "confirmatory permutation testing"

N_PERMUTATIONS = 10000
RNG_SEED = 20260429

BAND_SPECS = {
    "Theta": (4.0, 8.0),
    "Slow gamma": (35.0, 50.0),
}

plt.rcParams.update({"font.family": "serif", "font.size": 10})


def reset_dir(path: Path) -> Path:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def p_str(value):
    if pd.isna(value):
        return ""
    if value < 0.001:
        return "< .001"
    return f"{value:.3f}".lstrip("0")


def sig_label(value):
    if pd.isna(value):
        return ""
    if value < 0.001:
        return "***"
    if value < 0.01:
        return "**"
    if value < 0.05:
        return "*"
    return "n.s."


def fmt_num(value, decimals=3):
    if pd.isna(value):
        return ""
    return f"{float(value):.{decimals}f}"


def collect_paired_diffs(data, keys, freqs, low, high, roi):
    subjects, rem_vals, forg_vals = collect_endogenous_subject_band(data, keys, freqs, low, high, roi)
    return subjects, rem_vals - forg_vals


def run_permutation_for_modality(modality_label, data, keys, freq_key, band_specs,
                                 n_perm=N_PERMUTATIONS, seed=RNG_SEED):
    freqs = data.get(freq_key)
    if freqs is None:
        raise RuntimeError(f"{modality_label}: missing {freq_key}.")
    freqs = np.asarray(freqs, dtype=float)

    rng = np.random.default_rng(seed)

    all_regions = set()
    for k in keys:
        all_regions |= set(data.get(k, {}).keys())
    rois = sorted(r for r in all_regions if not is_excluded_region(r))

    eligible_tests = []
    skipped_rows = []
    for roi in rois:
        for band_label, (low, high) in band_specs.items():
            if not np.any((freqs >= low) & (freqs <= high)):
                continue
            subjects, diffs = collect_paired_diffs(data, keys, freqs, low, high, roi)
            if len(diffs) < 2:
                skipped_rows.append(
                    {
                        "Region": roi,
                        "Band": band_label,
                        "N": int(len(diffs)),
                        "Mean difference (remembered - forgotten)": float(np.nanmean(diffs)) if len(diffs) else np.nan,
                        "t": np.nan,
                        "p_uncorrected": np.nan,
                        "p_fwe_maxstat": np.nan,
                        "Sig (uncorrected)": "",
                        "Sig (FWE)": "",
                        "Subjects": ", ".join(subjects),
                    }
                )
                continue
            eligible_tests.append((roi, band_label, subjects, diffs))

    rows = list(skipped_rows)
    if not eligible_tests:
        return pd.DataFrame(rows)

    from retrieval_confirmatory_permutation_testing import family_max_stat_unique
    observed_t, observed_mean, observed_n, p_uncorrected, p_fwe, n_actual, n_union = \
        family_max_stat_unique(eligible_tests, n_perm, rng)

    for idx, (roi, band_label, subjects, _diffs) in enumerate(eligible_tests):
        rows.append(
            {
                "Region": roi,
                "Band": band_label,
                "N": int(observed_n[idx]),
                "Mean difference (remembered - forgotten)": float(observed_mean[idx]),
                "t": float(observed_t[idx]),
                "p_uncorrected": float(p_uncorrected[idx]),
                "p_fwe_maxstat": float(p_fwe[idx]),
                "Sig (uncorrected)": sig_label(p_uncorrected[idx]),
                "Sig (FWE)": sig_label(p_fwe[idx]),
                "n_iterations": int(n_actual),
                "n_union_subjects": int(n_union),
                "Subjects": ", ".join(subjects),
            }
        )
    return pd.DataFrame(rows)


def make_title_page(modality_label, n_perm, freqs, band_specs):
    fig = plt.figure(figsize=(8.5, 11))
    ax = fig.add_axes([0.07, 0.06, 0.86, 0.88])
    ax.axis("off")

    y = 0.97
    ax.text(0, y, f"Endogenous Memory: Paired Permutation Tests | {modality_label}", fontsize=15, fontweight="bold", va="top")
    y -= 0.06
    ax.text(0, y, "Remembered vs Forgotten retrieval trials (averaged across stim and no-stim)", fontsize=12, va="top")
    y -= 0.05

    freq_min = float(np.min(freqs)) if freqs is not None and len(freqs) else float("nan")
    freq_max = float(np.max(freqs)) if freqs is not None and len(freqs) else float("nan")
    bands_str = ", ".join(f"{name} = {lo:g}-{hi:g} Hz" for name, (lo, hi) in band_specs.items())

    paragraphs = [
        f"Method: paired sign-flip permutation test with {n_perm} unique iterations.",
        "Uniqueness: each iteration corresponds to a distinct sign-flip pattern over the family's subject union. When 2^N_union <= n_perm, the test enumerates all 2^N_union patterns exactly; otherwise n_perm patterns are drawn without replacement using numpy.random.Generator.choice(replace=False). Per-test signs are subset from the family-wide pattern, so within an iteration all tests share the same global random state.",
        "Per subject, compute (rem - forg) of band-averaged baseline-corrected value, where rem = mean(stim_rem, nostim_rem) and forg = mean(stim_forg, nostim_forg). Each permutation flips the sign of (rem - forg) for a subset of subjects according to the iteration's pattern; group mean and t-statistic are recomputed.",
        "Family-wise correction uses max-statistic permutation: at each iteration the maximum |t| across all region-band tests forms the null against which observed |t| is compared.",
        f"Bands: {bands_str}.",
        f"Frequencies in this modality: {freq_min:.2f}-{freq_max:.2f} Hz. Bands without spectral samples are omitted.",
        "Regions containing 'PNAS' are excluded.",
        "p-values use the (count + 1) / (n_perm + 1) convention.",
    ]
    for paragraph in paragraphs:
        wrapped = textwrap.fill(paragraph, width=92)
        ax.text(0, y, wrapped, fontsize=10.3, va="top")
        y -= 0.04 * (wrapped.count("\n") + 1) + 0.02

    return fig


def make_summary_page(stats_df):
    sig_unc = stats_df[stats_df["p_uncorrected"].notna() & (stats_df["p_uncorrected"] < 0.05)].copy()
    sig_fwe = stats_df[stats_df["p_fwe_maxstat"].notna() & (stats_df["p_fwe_maxstat"] < 0.05)].copy()

    fig = plt.figure(figsize=(8.5, 11))
    ax = fig.add_axes([0.07, 0.06, 0.86, 0.88])
    ax.axis("off")
    ax.text(0, 0.97, "Significant Results", fontsize=16, fontweight="bold", va="top")

    y = 0.91
    ax.text(0, y, "Family-wise corrected (max-stat, p_fwe < .05):", fontsize=12, fontweight="bold", va="top")
    y -= 0.04
    if sig_fwe.empty:
        ax.text(0, y, "  (none)", fontsize=10.5, va="top")
        y -= 0.03
    else:
        for _, row in sig_fwe.sort_values(["p_fwe_maxstat", "Region", "Band"]).iterrows():
            line = (
                f"  {row['Region']} | {row['Band']}: "
                f"t = {fmt_num(row['t'])}, p_unc = {p_str(row['p_uncorrected'])}, "
                f"p_fwe = {p_str(row['p_fwe_maxstat'])}."
            )
            wrapped = textwrap.fill(line, width=92)
            ax.text(0, y, wrapped, fontsize=10, va="top")
            y -= 0.035 * (wrapped.count("\n") + 1) + 0.005

    y -= 0.02
    ax.text(0, y, "Uncorrected (p_unc < .05):", fontsize=12, fontweight="bold", va="top")
    y -= 0.04
    if sig_unc.empty:
        ax.text(0, y, "  (none)", fontsize=10.5, va="top")
    else:
        for _, row in sig_unc.sort_values(["p_uncorrected", "Region", "Band"]).iterrows():
            line = (
                f"  {row['Region']} | {row['Band']}: "
                f"t = {fmt_num(row['t'])}, p_unc = {p_str(row['p_uncorrected'])}, "
                f"p_fwe = {p_str(row['p_fwe_maxstat'])}."
            )
            wrapped = textwrap.fill(line, width=92)
            ax.text(0, y, wrapped, fontsize=9.5, va="top")
            y -= 0.03 * (wrapped.count("\n") + 1) + 0.004
            if y < 0.05:
                break

    return fig


def chunk_dataframe(df, chunk_size=24):
    if df.empty:
        return []
    return [df.iloc[i : i + chunk_size].reset_index(drop=True) for i in range(0, len(df), chunk_size)]


def make_table_page(title, df):
    fig = plt.figure(figsize=(11, 8.5))
    ax = fig.add_axes([0.04, 0.06, 0.92, 0.88])
    ax.axis("off")
    ax.text(0, 1.02, title, fontsize=15, fontweight="bold", va="bottom")

    display_df = df[
        ["Region", "N", "Mean difference (remembered - forgotten)", "t",
         "p_uncorrected", "p_fwe_maxstat", "Sig (uncorrected)", "Sig (FWE)"]
    ].copy()
    display_df["Mean difference (remembered - forgotten)"] = display_df["Mean difference (remembered - forgotten)"].map(fmt_num)
    display_df["t"] = display_df["t"].map(fmt_num)
    display_df["p_uncorrected"] = display_df["p_uncorrected"].map(p_str)
    display_df["p_fwe_maxstat"] = display_df["p_fwe_maxstat"].map(p_str)

    table = ax.table(
        cellText=display_df.values.tolist(),
        colLabels=list(display_df.columns),
        loc="upper left",
        cellLoc="left",
        colLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.4)
    table.scale(1, 1.25)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#404040" if row == 0 else "#b0b0b0")
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#e8e8e8")
        else:
            cell.set_facecolor("white")
            sig_unc_val = display_df.iloc[row - 1]["Sig (uncorrected)"]
            sig_fwe_val = display_df.iloc[row - 1]["Sig (FWE)"]
            if col == 7 and sig_fwe_val and sig_fwe_val != "n.s.":
                cell.set_facecolor("#ffd9a8")
                cell.set_text_props(weight="bold")
            elif col == 6 and sig_unc_val and sig_unc_val != "n.s.":
                cell.set_facecolor("#fff4d6")
                cell.set_text_props(weight="bold")

    ax.text(
        0,
        -0.03,
        textwrap.fill(
            "Note. Significance codes: * p < .05, ** p < .01, *** p < .001. "
            "p_uncorrected: per-test permutation p. p_fwe_maxstat: max-stat family-wise corrected p across all region-band tests.",
            width=120,
        ),
        fontsize=8.6,
        style="italic",
        va="top",
    )
    return fig


def write_modality_outputs(modality_label, modality_subdir, freqs, stats_df, band_order):
    out_dir = OUTPUT_BASE / modality_subdir
    reset_dir(out_dir)

    csv_path = out_dir / "permutation.csv"
    pdf_path = out_dir / "permutation.pdf"

    stats_df = stats_df.sort_values(["Region", "Band"]).reset_index(drop=True)
    stats_df.to_csv(csv_path, index=False)

    with PdfPages(pdf_path) as pdf:
        pdf.savefig(make_title_page(modality_label, N_PERMUTATIONS, freqs, BAND_SPECS))
        plt.close("all")

        for band_label in band_order:
            band_df = stats_df[stats_df["Band"] == band_label].copy()
            if band_df.empty:
                continue
            chunks = chunk_dataframe(band_df, chunk_size=24)
            total = len(chunks)
            for idx, chunk in enumerate(chunks, start=1):
                title = f"{modality_label} | {band_label}"
                if total > 1:
                    title += f" ({idx}/{total})"
                pdf.savefig(make_table_page(title, chunk))
                plt.close("all")

        pdf.savefig(make_summary_page(stats_df))
        plt.close("all")

    print(f"  -> {pdf_path}")
    print(f"  -> {csv_path}")


def main():
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    band_order = list(BAND_SPECS.keys())

    print("[1/3] Power...")
    power_data = load_power_data()
    power_stats = run_permutation_for_modality("Power", power_data, POWER_COH_KEYS, "freqs_diff", BAND_SPECS)
    write_modality_outputs("Power", "power", np.asarray(power_data["freqs_diff"], dtype=float), power_stats, band_order)

    print("[2/3] Coherence...")
    coh_data = load_coherence_data()
    coh_stats = run_permutation_for_modality("Coherence", coh_data, POWER_COH_KEYS, "freqs_diff", BAND_SPECS)
    write_modality_outputs("Coherence", "coherence", np.asarray(coh_data["freqs_diff"], dtype=float), coh_stats, band_order)

    print("[3/3] PAC...")
    pac_data = load_pac_data()
    pac_stats = run_permutation_for_modality("PAC", pac_data, PAC_KEYS, "freqs_diff", BAND_SPECS)
    write_modality_outputs("PAC", "PAC", np.asarray(pac_data["freqs_diff"], dtype=float), pac_stats, band_order)

    print("Done.")


if __name__ == "__main__":
    main()
