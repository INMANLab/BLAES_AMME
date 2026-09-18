#!/usr/bin/env python
"""Paired within-subject t-tests for endogenous memory at retrieval.

For each modality (power, coherence, PAC), this report contrasts remembered vs
forgotten retrieval trials within subject. The metric is the per-subject
band-averaged baseline-corrected value:
    rem  = mean(bc_*_rem  across stim and no-stim)
    forg = mean(bc_*_forg across stim and no-stim)

The paired t-test is on (rem - forg) within Theta (4-8 Hz) and Slow gamma
(35-50 Hz). Regions containing 'PNAS' are excluded.

Outputs:
  outputs/parametric retrieval test/endogenous memory/t-tests/
    power/t-tests.pdf, t-tests.csv
    coherence/t-tests.pdf, t-tests.csv
    PAC/t-tests.pdf, t-tests.csv
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
import re
import shutil
import sys
import textwrap

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import ttest_rel

_ALLHPC_RE = re.compile(r"(?<![A-Za-z])ALLHPC(?![A-Za-z])")
_HPC_RE = re.compile(r"(?<![A-Za-z])HPC(?![A-Za-z])")
_MARKER = "\x00__MACRO_HPC__\x00"


def pretty_in_text(text):
    if text is None:
        return text
    s = str(text)
    s = _ALLHPC_RE.sub(_MARKER, s)
    s = _HPC_RE.sub("SUB", s)
    s = s.replace(_MARKER, "HPC")
    return s

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
OUTPUT_BASE = SCRIPT_DIR / "outputs" / "parametric retrieval test" / "endogenous memory" / "t-tests"

BAND_SPECS = {
    "Theta": (4.0, 8.0),
    "Slow gamma": (35.0, 50.0),
}

POWER_COH_KEYS = ("bc_stim_rem", "bc_nostim_rem", "bc_stim_forg", "bc_nostim_forg")
PAC_KEYS = ("diff_stim_rem", "diff_nostim_rem", "diff_stim_forg", "diff_nostim_forg")

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


def collect_endogenous_subject_band(data, keys, freqs, low, high, roi):
    """Per-subject band-averaged endogenous memory contrast (rem - forg).

    keys = (stim_rem_key, nostim_rem_key, stim_forg_key, nostim_forg_key).
    Per subject: rem = mean of available rem values; forg = mean of available
    forg values; subject contributes only if both rem and forg are finite.
    """
    stim_rem_key, nostim_rem_key, stim_forg_key, nostim_forg_key = keys
    stim_rem = data.get(stim_rem_key, {}).get(roi, {})
    nostim_rem = data.get(nostim_rem_key, {}).get(roi, {})
    stim_forg = data.get(stim_forg_key, {}).get(roi, {})
    nostim_forg = data.get(nostim_forg_key, {}).get(roi, {})

    subjects = sorted(set(stim_rem) | set(nostim_rem) | set(stim_forg) | set(nostim_forg))
    rem_vals = []
    forg_vals = []
    used = []
    for subject in subjects:
        rem_pieces = []
        for d in (stim_rem, nostim_rem):
            if subject in d:
                v = band_mean(d[subject], freqs, low, high)
                if np.isfinite(v):
                    rem_pieces.append(v)
        forg_pieces = []
        for d in (stim_forg, nostim_forg):
            if subject in d:
                v = band_mean(d[subject], freqs, low, high)
                if np.isfinite(v):
                    forg_pieces.append(v)
        if rem_pieces and forg_pieces:
            rem_vals.append(float(np.mean(rem_pieces)))
            forg_vals.append(float(np.mean(forg_pieces)))
            used.append(subject)
    return used, np.asarray(rem_vals, dtype=float), np.asarray(forg_vals, dtype=float)


def paired_band_stats(data, keys, freq_key):
    freqs = data.get(freq_key)
    if freqs is None:
        raise RuntimeError(f"missing {freq_key} frequency array")
    freqs = np.asarray(freqs, dtype=float)

    all_regions = set()
    for k in keys:
        all_regions |= set(data.get(k, {}).keys())
    rois = sorted(r for r in all_regions if not is_excluded_region(r))

    rows = []
    for roi in rois:
        for band_label, (low, high) in BAND_SPECS.items():
            if not np.any((freqs >= low) & (freqs <= high)):
                continue

            subjects, rem_vals, forg_vals = collect_endogenous_subject_band(
                data, keys, freqs, low, high, roi
            )
            if len(rem_vals) >= 2:
                t_stat, p_value = ttest_rel(rem_vals, forg_vals, nan_policy="omit")
            else:
                t_stat, p_value = np.nan, np.nan

            mean_diff = float(np.nanmean(rem_vals - forg_vals)) if len(rem_vals) else np.nan
            rows.append(
                {
                    "Region": roi,
                    "Band": band_label,
                    "N": int(len(rem_vals)),
                    "Mean difference (remembered - forgotten)": mean_diff,
                    "t": float(t_stat) if pd.notna(t_stat) else np.nan,
                    "p": float(p_value) if pd.notna(p_value) else np.nan,
                    "Sig": sig_label(p_value),
                    "Subjects": ", ".join(subjects),
                }
            )
    return pd.DataFrame(rows)


def make_title_page(modality_label, freqs):
    fig = plt.figure(figsize=(8.5, 11))
    ax = fig.add_axes([0.07, 0.06, 0.86, 0.88])
    ax.axis("off")

    y = 0.97
    ax.text(0, y, f"Endogenous Memory Paired T-Tests | {modality_label}", fontsize=17, fontweight="bold", va="top")
    y -= 0.06
    ax.text(0, y, "Remembered vs Forgotten retrieval trials (averaged across stim and no-stim)", fontsize=12, va="top")
    y -= 0.05

    freq_min = float(np.min(freqs)) if freqs is not None and len(freqs) else float("nan")
    freq_max = float(np.max(freqs)) if freqs is not None and len(freqs) else float("nan")

    paragraphs = [
        "Tests are paired within-subject t-tests on (remembered - forgotten) of the baseline-corrected value averaged within band.",
        "Per subject: rem = mean(stim_rem, nostim_rem); forg = mean(stim_forg, nostim_forg); subject contributes if both rem and forg are finite.",
        "Bands: Theta = 4-8 Hz, Slow gamma = 35-50 Hz.",
        f"Frequencies in this modality: {freq_min:.2f}-{freq_max:.2f} Hz. Bands without spectral samples are omitted.",
        "Regions containing 'PNAS' are excluded.",
        "No family-wise correction was applied.",
    ]
    for paragraph in paragraphs:
        wrapped = textwrap.fill(paragraph, width=92)
        ax.text(0, y, wrapped, fontsize=10.5, va="top")
        y -= 0.04 * (wrapped.count("\n") + 1) + 0.02

    return fig


def make_summary_page(stats_df):
    sig_df = stats_df[stats_df["p"].notna() & (stats_df["p"] < 0.05)].copy()
    fig = plt.figure(figsize=(8.5, 11))
    ax = fig.add_axes([0.07, 0.06, 0.86, 0.88])
    ax.axis("off")
    ax.text(0, 0.97, "Significant Results (p < .05, uncorrected)", fontsize=15, fontweight="bold", va="top")

    y = 0.91
    if sig_df.empty:
        ax.text(0, y, "(none)", fontsize=10.5, va="top")
    else:
        for _, row in sig_df.sort_values(["p", "Region", "Band"]).iterrows():
            line = (
                f"{pretty_in_text(row['Region'])} | {row['Band']}: "
                f"t = {fmt_num(row['t'])}, p = {p_str(row['p'])} ({row['Sig']})."
            )
            wrapped = textwrap.fill(line, width=94)
            ax.text(0, y, wrapped, fontsize=10, va="top")
            y -= 0.035 * (wrapped.count("\n") + 1) + 0.005
            if y < 0.05:
                break
    return fig


def chunk_dataframe(df, chunk_size=28):
    if df.empty:
        return []
    return [df.iloc[i : i + chunk_size].reset_index(drop=True) for i in range(0, len(df), chunk_size)]


def make_table_page(title, df):
    fig = plt.figure(figsize=(11, 8.5))
    ax = fig.add_axes([0.04, 0.06, 0.92, 0.88])
    ax.axis("off")
    ax.text(0, 1.02, title, fontsize=15, fontweight="bold", va="bottom")

    display_df = df[["Region", "N", "Mean difference (remembered - forgotten)", "t", "p", "Sig"]].copy()
    display_df["Region"] = display_df["Region"].map(pretty_in_text)
    display_df["Mean difference (remembered - forgotten)"] = display_df["Mean difference (remembered - forgotten)"].map(fmt_num)
    display_df["t"] = display_df["t"].map(fmt_num)
    display_df["p"] = display_df["p"].map(p_str)

    table = ax.table(
        cellText=display_df.values.tolist(),
        colLabels=list(display_df.columns),
        loc="upper left",
        cellLoc="left",
        colLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.6)
    table.scale(1, 1.25)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#404040" if row == 0 else "#b0b0b0")
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#e8e8e8")
        else:
            cell.set_facecolor("white")
            if col == 5 and display_df.iloc[row - 1]["Sig"] != "n.s.":
                cell.set_facecolor("#fff4d6")
                cell.set_text_props(weight="bold")

    ax.text(
        0,
        -0.03,
        textwrap.fill("Note. Significance codes: * p < .05, ** p < .01, *** p < .001.", width=110),
        fontsize=8.8,
        style="italic",
        va="top",
    )
    return fig


def write_modality(modality_label, modality_subdir, freqs, stats_df):
    out_dir = OUTPUT_BASE / modality_subdir
    reset_dir(out_dir)
    csv_path = out_dir / "t-tests.csv"
    pdf_path = out_dir / "t-tests.pdf"

    stats_df = stats_df.sort_values(["Region", "Band"]).reset_index(drop=True)
    stats_df.to_csv(csv_path, index=False)

    with PdfPages(pdf_path) as pdf:
        pdf.savefig(make_title_page(modality_label, freqs))
        plt.close("all")
        for band_label in ["Theta", "Slow gamma"]:
            band_df = stats_df[stats_df["Band"] == band_label].copy()
            if band_df.empty:
                continue
            chunks = chunk_dataframe(band_df, chunk_size=28)
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


def load_power_data():
    blaes = load_blaes_power()
    amme = load_amme_power()
    return build_all_retrieval_data(blaes, amme)


def load_coherence_data():
    blaes = augment_with_allhpc_pair_composites(load_blaes_coh())
    amme = augment_with_allhpc_pair_composites(load_amme_coh())
    if blaes["freqs_diff"] is not None and amme["freqs_diff"] is not None and np.array_equal(blaes["freqs_diff"], amme["freqs_diff"]):
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
            blaes.get("bc_plot_exclude_substrings"), amme.get("bc_plot_exclude_substrings")
        ),
    }
    return augment_with_allhpc_pair_composites(canonicalize_coherence_data(all_data))


def load_pac_data():
    grouped = load_grouped_retrieval_pac_data()
    return grouped["all"]


def main():
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    print("[1/3] Power...")
    power_data = load_power_data()
    power_stats = paired_band_stats(power_data, POWER_COH_KEYS, "freqs_diff")
    write_modality("Power", "power", np.asarray(power_data["freqs_diff"], dtype=float), power_stats)

    print("[2/3] Coherence...")
    coh_data = load_coherence_data()
    coh_stats = paired_band_stats(coh_data, POWER_COH_KEYS, "freqs_diff")
    write_modality("Coherence", "coherence", np.asarray(coh_data["freqs_diff"], dtype=float), coh_stats)

    print("[3/3] PAC...")
    pac_data = load_pac_data()
    pac_stats = paired_band_stats(pac_data, PAC_KEYS, "freqs_diff")
    write_modality("PAC", "PAC", np.asarray(pac_data["freqs_diff"], dtype=float), pac_stats)

    print("Done.")


if __name__ == "__main__":
    main()
