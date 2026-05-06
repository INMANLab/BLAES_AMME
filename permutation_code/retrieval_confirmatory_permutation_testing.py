#!/usr/bin/env python
"""Confirmatory paired permutation tests for retrieval power, coherence, and PAC.

For each modality (power, coherence, PAC) and each memory condition (remembered,
forgotten), this script:

1. Computes, for every subject, the within-subject mean difference
   (stim - no stim) of the baseline-corrected value averaged within Theta
   (4-8 Hz) or Slow gamma (35-50 Hz).
2. Computes the observed group-level mean difference per (region, band).
3. Runs 1000 sign-flip permutations: at each iteration, the (stim, no-stim)
   labels are randomly flipped within each subject independently, and the new
   group mean difference is recorded.
4. Family-wise (across regions and bands within a memory condition + modality)
   p-values use max-statistic correction: at each permutation, the maximum
   |statistic| across all (region, band) tests is retained, and observed |t|
   is compared to that null distribution.
5. Both uncorrected and FWE-corrected (max-stat) p-values are reported.

Outputs (one PDF + one CSV per modality):
  outputs/parametric retrieval test/confirmatory permutation testing/
    power/permutation.pdf, permutation.csv
    coherence/permutation.pdf, permutation.csv
    PAC/permutation.pdf, permutation.csv

Regions containing "PNAS" are excluded.
"""

# --- AMME_BLAES sys.path bootstrap ---
import sys as _sys
from pathlib import Path as _Path
_root = _Path(__file__).resolve().parent.parent
for _p in (_root, _root / 'encoding', _root / 'retrieval', _root / 'endogenous_memory',
          _root / 'regressions', _root / 'behavioral', _root / 'balanced_memory'):
    if str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))
# --- end bootstrap ---

from pathlib import Path
import shutil
import sys
import textwrap

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from combined_retrieval_power import (
    build_all_retrieval_data,
    load_amme_retrieval as load_amme_power,
    load_blaes_retrieval as load_blaes_power,
)
from combined_retrieval_coherence import (
    augment_with_allhpc_pair_composites,
    canonicalize_coherence_data,
    load_amme_retrieval as load_amme_coh,
    load_blaes_retrieval as load_blaes_coh,
    merge_dicts,
    merge_sets,
)
from combined_retrieval_pac import load_grouped_retrieval_pac_data


matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_BASE = SCRIPT_DIR / "outputs" / "parametric retrieval test" / "stim vs no stim" / "confirmatory permutation testing"

N_PERMUTATIONS = 10000
RNG_SEED = 20260429

BAND_SPECS = {
    "Theta": (4.0, 8.0),
    "Slow gamma": (35.0, 50.0),
}

MEMORY_KEYS_POWER_COH = {
    "Remembered": ("bc_stim_rem", "bc_nostim_rem"),
    "Forgotten": ("bc_stim_forg", "bc_nostim_forg"),
}
MEMORY_KEYS_PAC = {
    "Remembered": ("diff_stim_rem", "diff_nostim_rem"),
    "Forgotten": ("diff_stim_forg", "diff_nostim_forg"),
}

plt.rcParams.update({"font.family": "serif", "font.size": 10})


def reset_dir(path: Path) -> Path:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


EXCLUDED_REGION_TOKENS = {"PNAS", "ALLHPC", "MTL", "PHG"}


def is_excluded_region(roi):
    s = str(roi)
    if "PNAS" in s:
        return True
    tokens = set(s.split("_"))
    return bool(tokens & EXCLUDED_REGION_TOKENS)


def band_mean(values, freqs, low, high):
    values = np.asarray(values, dtype=float)
    mask = (freqs >= low) & (freqs <= high)
    if not np.any(mask):
        return np.nan
    band_values = values[mask]
    if np.all(np.isnan(band_values)):
        return np.nan
    return float(np.nanmean(band_values))


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


def collect_paired_diffs(stim_dict, nostim_dict, freqs, low, high):
    """Return (subjects, diffs) for one ROI x band."""
    paired_subjects = sorted(set(stim_dict) & set(nostim_dict))
    subjects = []
    diffs = []
    for subject in paired_subjects:
        stim_value = band_mean(stim_dict[subject], freqs, low, high)
        nostim_value = band_mean(nostim_dict[subject], freqs, low, high)
        if np.isfinite(stim_value) and np.isfinite(nostim_value):
            subjects.append(subject)
            diffs.append(stim_value - nostim_value)
    return subjects, np.asarray(diffs, dtype=float)


def unique_signflip_codes(n_subjects, n_perm, rng):
    """Return (n_actual, codes) of unique sign-flip patterns.

    Each integer code in [0, 2^n_subjects) encodes a sign-flip pattern (bit s
    = 0 means -1, bit s = 1 means +1 for subject s). When 2^n_subjects <=
    n_perm we enumerate exhaustively (n_actual = 2^n_subjects, fully unique
    by construction). Otherwise we draw n_perm distinct codes without
    replacement using numpy.random.Generator.choice(replace=False).
    """
    if n_subjects >= 64:
        raise NotImplementedError(
            f"n_subjects={n_subjects} >= 64; int64 codes cannot index sign patterns."
        )
    max_unique = 1 << n_subjects
    if max_unique <= n_perm:
        return int(max_unique), np.arange(max_unique, dtype=np.int64)
    if max_unique < (1 << 31):
        codes = rng.choice(int(max_unique), size=n_perm, replace=False)
        return n_perm, codes.astype(np.int64)
    # Universe too large for choice(replace=False); use set-based dedup.
    seen = set()
    codes = np.empty(n_perm, dtype=np.int64)
    i = 0
    while i < n_perm:
        need = (n_perm - i) * 2
        cands = rng.integers(0, max_unique, size=need, dtype=np.int64)
        for c in cands:
            ic = int(c)
            if ic in seen:
                continue
            seen.add(ic)
            codes[i] = ic
            i += 1
            if i >= n_perm:
                break
    return n_perm, codes


def codes_to_signs_subset(codes, subject_indices):
    """Convert codes to (n_actual, n_test) sign matrix (+/-1 float64) using
    only the bits at subject_indices."""
    codes = np.asarray(codes, dtype=np.int64)
    indices = np.asarray(subject_indices, dtype=np.int64)
    bits = (codes[:, None] >> indices[None, :]) & 1
    return (bits.astype(np.int8) * 2 - 1).astype(np.float64)


def _vectorised_t_stats(permuted, n):
    """Compute one-sample t-stats per row of permuted (shape (k, n))."""
    if n <= 1:
        return np.zeros(permuted.shape[0])
    means = permuted.mean(axis=1)
    diff_sq = (permuted - means[:, None]) ** 2
    var = diff_sq.sum(axis=1) / (n - 1)
    sds = np.sqrt(var)
    ts = np.zeros_like(means)
    mask = sds > 0
    ts[mask] = means[mask] / (sds[mask] / np.sqrt(n))
    return ts


def family_max_stat_unique(eligible_tests, n_perm, rng):
    """Run paired sign-flip permutation with family-wide unique iterations.

    Each iteration corresponds to a unique sign-flip pattern over the family's
    subject union. Per-test signs are subset from that pattern, so within a
    single iteration all tests share the same global random state.

    Returns observed t/mean/n arrays, p_uncorrected, p_fwe_maxstat, and the
    actual number of unique iterations executed (may be smaller than n_perm
    when 2^n_union <= n_perm).
    """
    union_subjects = sorted(set().union(*[set(s) for (_, _, s, _) in eligible_tests]))
    subj_to_bit = {s: i for i, s in enumerate(union_subjects)}
    n_union = len(union_subjects)
    n_actual, codes = unique_signflip_codes(n_union, n_perm, rng)

    n_tests = len(eligible_tests)
    observed_t = np.empty(n_tests)
    observed_mean = np.empty(n_tests)
    observed_n = np.empty(n_tests, dtype=int)
    for k, (_, _, _, diffs) in enumerate(eligible_tests):
        n = len(diffs)
        m = float(np.mean(diffs))
        sd = float(np.std(diffs, ddof=1)) if n > 1 else 0.0
        observed_mean[k] = m
        observed_n[k] = n
        observed_t[k] = m / (sd / np.sqrt(n)) if sd > 0 else 0.0
    abs_obs = np.abs(observed_t)

    max_null = np.zeros(n_actual)
    unc_count = np.zeros(n_tests, dtype=np.int64)
    for k, (_, _, subjects, diffs) in enumerate(eligible_tests):
        n = len(diffs)
        idx = np.array([subj_to_bit[s] for s in subjects], dtype=np.int64)
        signs = codes_to_signs_subset(codes, idx)         # (n_actual, n)
        permuted = signs * diffs[None, :]
        ts = _vectorised_t_stats(permuted, n)
        abs_ts = np.abs(ts)
        unc_count[k] = int(np.sum(abs_ts >= abs_obs[k]))
        np.maximum(max_null, abs_ts, out=max_null)

    p_uncorrected = (unc_count + 1) / (n_actual + 1)
    p_fwe = (np.sum(max_null[None, :] >= abs_obs[:, None], axis=1) + 1) / (n_actual + 1)

    return (
        observed_t, observed_mean, observed_n,
        p_uncorrected, p_fwe,
        n_actual, n_union,
    )


def run_permutation_for_modality(modality_label, data, memory_keys, freq_key, n_perm=N_PERMUTATIONS, seed=RNG_SEED):
    """Run sign-flip paired permutation tests with max-stat FWE.

    Per memory condition, all (region x band) tests form a single family for
    max-stat correction.
    """
    freqs = data.get(freq_key)
    if freqs is None:
        raise RuntimeError(f"{modality_label}: missing {freq_key} frequency array.")
    freqs = np.asarray(freqs, dtype=float)

    rng = np.random.default_rng(seed)
    rows_all = []

    for memory_label, (stim_key, nostim_key) in memory_keys.items():
        stim_block = data.get(stim_key, {})
        nostim_block = data.get(nostim_key, {})
        roi_names = sorted(set(stim_block) | set(nostim_block))
        roi_names = [r for r in roi_names if not is_excluded_region(r)]

        eligible_tests = []
        for roi in roi_names:
            stim_subjects = stim_block.get(roi, {})
            nostim_subjects = nostim_block.get(roi, {})
            for band_label, (low, high) in BAND_SPECS.items():
                if not np.any((freqs >= low) & (freqs <= high)):
                    continue
                subjects, diffs = collect_paired_diffs(stim_subjects, nostim_subjects, freqs, low, high)
                if len(diffs) < 2:
                    rows_all.append(
                        {
                            "Memory": memory_label,
                            "Region": roi,
                            "Band": band_label,
                            "N": int(len(diffs)),
                            "Mean difference (stim - no stim)": float(np.nanmean(diffs)) if len(diffs) else np.nan,
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

        if not eligible_tests:
            continue

        observed_t, observed_mean, observed_n, p_uncorrected, p_fwe, n_actual, n_union = \
            family_max_stat_unique(eligible_tests, n_perm, rng)

        for idx, (roi, band_label, subjects, _diffs) in enumerate(eligible_tests):
            rows_all.append(
                {
                    "Memory": memory_label,
                    "Region": roi,
                    "Band": band_label,
                    "N": int(observed_n[idx]),
                    "Mean difference (stim - no stim)": float(observed_mean[idx]),
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

    return pd.DataFrame(rows_all)


def make_title_page(modality_label, n_perm, freqs):
    fig = plt.figure(figsize=(8.5, 11))
    ax = fig.add_axes([0.07, 0.06, 0.86, 0.88])
    ax.axis("off")

    y = 0.97
    ax.text(0, y, f"Retrieval {modality_label}: Paired Permutation Tests", fontsize=17, fontweight="bold", va="top")
    y -= 0.06
    ax.text(0, y, "Stim vs No Stim, remembered and forgotten trials separately", fontsize=12, va="top")
    y -= 0.05

    freq_min = float(np.min(freqs)) if freqs is not None and len(freqs) else float("nan")
    freq_max = float(np.max(freqs)) if freqs is not None and len(freqs) else float("nan")

    paragraphs = [
        f"Method: paired sign-flip permutation test with {n_perm} unique iterations.",
        "Uniqueness: each iteration corresponds to a distinct sign-flip pattern over the family's subject union. When 2^N_union <= n_perm, the test enumerates all 2^N_union patterns exactly; otherwise n_perm patterns are drawn without replacement using numpy.random.Generator.choice(replace=False). Per-test signs are subset from the family-wide pattern, so within an iteration all tests in the family share the same global random state.",
        "For each subject, compute the within-subject mean difference (stim - no stim) of the baseline-corrected value averaged in the band. Each permutation flips the sign of (stim - no stim) for a subset of subjects according to the iteration's pattern; the group mean difference and t-statistic are recomputed.",
        "Family-wise correction uses max-statistic permutation: at each iteration, the maximum |t| across all region-band tests within a memory condition forms the null distribution against which observed |t| is compared.",
        "Bands: Theta = 4-8 Hz, Slow gamma = 35-50 Hz.",
        f"Frequencies available in this modality: {freq_min:.2f}-{freq_max:.2f} Hz. Bands without any spectral samples in that range are omitted.",
        "Regions containing 'PNAS' are excluded.",
        "p-values use the (count + 1) / (n_perm + 1) convention.",
    ]
    for paragraph in paragraphs:
        wrapped = textwrap.fill(paragraph, width=92)
        ax.text(0, y, wrapped, fontsize=10.5, va="top")
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
        for _, row in sig_fwe.sort_values(["p_fwe_maxstat", "Memory", "Region", "Band"]).iterrows():
            line = (
                f"  {row['Memory']} | {row['Region']} | {row['Band']}: "
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
        for _, row in sig_unc.sort_values(["p_uncorrected", "Memory", "Region", "Band"]).iterrows():
            line = (
                f"  {row['Memory']} | {row['Region']} | {row['Band']}: "
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
        ["Region", "N", "Mean difference (stim - no stim)", "t", "p_uncorrected", "p_fwe_maxstat", "Sig (uncorrected)", "Sig (FWE)"]
    ].copy()
    display_df["Mean difference (stim - no stim)"] = display_df["Mean difference (stim - no stim)"].map(fmt_num)
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
            "p_uncorrected: per-test permutation p. p_fwe_maxstat: max-statistic family-wise corrected p across all region-band tests within this memory condition.",
            width=120,
        ),
        fontsize=8.6,
        style="italic",
        va="top",
    )
    return fig


def write_modality_outputs(modality_label, modality_subdir, freqs, stats_df):
    out_dir = OUTPUT_BASE / modality_subdir
    reset_dir(out_dir)

    csv_path = out_dir / "permutation.csv"
    pdf_path = out_dir / "permutation.pdf"

    stats_df = stats_df.sort_values(["Memory", "Region", "Band"]).reset_index(drop=True)
    stats_df.to_csv(csv_path, index=False)

    with PdfPages(pdf_path) as pdf:
        pdf.savefig(make_title_page(modality_label, N_PERMUTATIONS, freqs))
        plt.close("all")

        for memory_label in ["Remembered", "Forgotten"]:
            memory_df = stats_df[stats_df["Memory"] == memory_label].copy()
            for band_label in ["Theta", "Slow gamma"]:
                band_df = memory_df[memory_df["Band"] == band_label].copy()
                if band_df.empty:
                    continue
                chunks = chunk_dataframe(band_df, chunk_size=24)
                total = len(chunks)
                for idx, chunk in enumerate(chunks, start=1):
                    title = f"{modality_label} | {memory_label} Trials | {band_label}"
                    if total > 1:
                        title += f" ({idx}/{total})"
                    pdf.savefig(make_table_page(title, chunk))
                    plt.close("all")

        pdf.savefig(make_summary_page(stats_df))
        plt.close("all")

    print(f"  -> {pdf_path}")
    print(f"  -> {csv_path}")


def load_power_data():
    blaes = load_blaes_power()
    amme = load_amme_power()
    return build_all_retrieval_data(blaes, amme)


def load_coherence_data():
    blaes = augment_with_allhpc_pair_composites(load_blaes_coh())
    amme = augment_with_allhpc_pair_composites(load_amme_coh())

    if blaes["freqs_diff"] is not None and amme["freqs_diff"] is not None:
        if np.array_equal(blaes["freqs_diff"], amme["freqs_diff"]):
            all_freqs_diff = blaes["freqs_diff"]
        else:
            all_freqs_diff = blaes["freqs_diff"]
    else:
        all_freqs_diff = blaes["freqs_diff"] if blaes["freqs_diff"] is not None else amme["freqs_diff"]

    all_data = {
        "freqs_diff": all_freqs_diff,
        "bc_stim_rem": merge_dicts(blaes.get("bc_stim_rem", {}), amme.get("bc_stim_rem", {})),
        "bc_stim_forg": merge_dicts(blaes.get("bc_stim_forg", {}), amme.get("bc_stim_forg", {})),
        "bc_nostim_rem": merge_dicts(blaes.get("bc_nostim_rem", {}), amme.get("bc_nostim_rem", {})),
        "bc_nostim_forg": merge_dicts(blaes.get("bc_nostim_forg", {}), amme.get("bc_nostim_forg", {})),
        "bc_plot_exclude_substrings": merge_sets(
            blaes.get("bc_plot_exclude_substrings"),
            amme.get("bc_plot_exclude_substrings"),
        ),
    }
    return augment_with_allhpc_pair_composites(canonicalize_coherence_data(all_data))


def load_pac_data():
    grouped = load_grouped_retrieval_pac_data()
    return grouped["all"]


def main():
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    print("[1/3] Loading power data...")
    power_data = load_power_data()
    print("[1/3] Running paired sign-flip permutation tests for power...")
    power_stats = run_permutation_for_modality("Power", power_data, MEMORY_KEYS_POWER_COH, "freqs_diff")
    write_modality_outputs("Power", "power", np.asarray(power_data["freqs_diff"], dtype=float), power_stats)

    print("[2/3] Loading coherence data...")
    coh_data = load_coherence_data()
    print("[2/3] Running paired sign-flip permutation tests for coherence...")
    coh_stats = run_permutation_for_modality("Coherence", coh_data, MEMORY_KEYS_POWER_COH, "freqs_diff")
    write_modality_outputs("Coherence", "coherence", np.asarray(coh_data["freqs_diff"], dtype=float), coh_stats)

    print("[3/3] Loading PAC data...")
    pac_data = load_pac_data()
    print("[3/3] Running paired sign-flip permutation tests for PAC...")
    pac_stats = run_permutation_for_modality("PAC", pac_data, MEMORY_KEYS_PAC, "freqs_diff")
    write_modality_outputs("PAC", "PAC", np.asarray(pac_data["freqs_diff"], dtype=float), pac_stats)

    print("Done.")


if __name__ == "__main__":
    main()
