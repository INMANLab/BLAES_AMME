#!/usr/bin/env python
"""Paired within-subject t-tests for retrieval PAC.

This report tests stim vs no-stim retrieval PAC (modulation index) separately
for remembered and forgotten trials, using Theta (4-8 Hz) and Slow gamma
(35-50 Hz) band averages of the baseline-corrected (post - pre) MI spectra.

Note: PAC TG spectra cover gamma amplitude frequencies (~30-100 Hz). The Theta
band is reported only for ROIs whose stored frequency arrays span 4-8 Hz; in
the standard PAC TG output (30-100 Hz) the Theta rows will be skipped because
no spectral samples fall inside the band.
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

from combined_retrieval_pac import load_grouped_retrieval_pac_data

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


def running_in_notebook():
    try:
        from IPython import get_ipython

        shell = get_ipython()
        return shell is not None and shell.__class__.__name__ == "ZMQInteractiveShell"
    except Exception:
        return False


IN_NOTEBOOK = running_in_notebook()
if not IN_NOTEBOOK:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "outputs" / "parametric retrieval test" / "stim vs no stim" / "t-tests" / "PAC"
PDF_PATH = OUTPUT_DIR / "t-tests.pdf"
CSV_PATH = OUTPUT_DIR / "t-tests.csv"

BAND_SPECS = {
    "Theta": (4.0, 8.0),
    "Slow gamma": (35.0, 50.0),
}

MEMORY_KEYS = {
    "Remembered": ("diff_stim_rem", "diff_nostim_rem"),
    "Forgotten": ("diff_stim_forg", "diff_nostim_forg"),
}

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 10,
    }
)


def reset_dir(path: Path) -> Path:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def band_mean(values, freqs, low, high):
    values = np.asarray(values, dtype=float)
    mask = (freqs >= low) & (freqs <= high)
    if not np.any(mask):
        return np.nan
    band_values = values[mask]
    if np.all(np.isnan(band_values)):
        return np.nan
    return float(np.nanmean(band_values))


def roi_sort_key(roi):
    return (str(roi),)


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


def fmt_num(value, decimals=4):
    if pd.isna(value):
        return ""
    return f"{float(value):.{decimals}f}"


def load_retrieval_data():
    grouped = load_grouped_retrieval_pac_data()
    return grouped["all"]


def paired_band_stats(data):
    freqs = data.get("freqs_diff")
    if freqs is None:
        raise RuntimeError("Retrieval PAC diff frequencies were not available.")
    freqs = np.asarray(freqs, dtype=float)

    rows = []
    for memory_label, (stim_key, nostim_key) in MEMORY_KEYS.items():
        stim_dict = data.get(stim_key, {})
        nostim_dict = data.get(nostim_key, {})
        roi_names = sorted(set(stim_dict) | set(nostim_dict), key=roi_sort_key)

        for roi in roi_names:
            roi_str = str(roi)
            if "PNAS" in roi_str:
                continue
            if set(roi_str.split("_")) & {"ALLHPC", "MTL", "PHG"}:
                continue
            stim_subjects = stim_dict.get(roi, {})
            nostim_subjects = nostim_dict.get(roi, {})
            paired_subjects = sorted(set(stim_subjects) & set(nostim_subjects))

            for band_label, (low, high) in BAND_SPECS.items():
                if not np.any((freqs >= low) & (freqs <= high)):
                    continue

                stim_values = []
                nostim_values = []
                subject_rows = []

                for subject in paired_subjects:
                    stim_value = band_mean(stim_subjects[subject], freqs, low, high)
                    nostim_value = band_mean(nostim_subjects[subject], freqs, low, high)
                    if np.isfinite(stim_value) and np.isfinite(nostim_value):
                        stim_values.append(stim_value)
                        nostim_values.append(nostim_value)
                        subject_rows.append((subject, stim_value, nostim_value))

                if len(stim_values) >= 2:
                    t_stat, p_value = ttest_rel(stim_values, nostim_values, nan_policy="omit")
                else:
                    t_stat, p_value = np.nan, np.nan

                stim_values = np.asarray(stim_values, dtype=float)
                nostim_values = np.asarray(nostim_values, dtype=float)
                mean_diff = float(np.nanmean(stim_values - nostim_values)) if len(stim_values) else np.nan

                rows.append(
                    {
                        "Memory": memory_label,
                        "Region": roi,
                        "Band": band_label,
                        "N": int(len(stim_values)),
                        "Mean difference (stim - no stim)": mean_diff,
                        "t": float(t_stat) if pd.notna(t_stat) else np.nan,
                        "p": float(p_value) if pd.notna(p_value) else np.nan,
                        "Sig": sig_label(p_value),
                        "Subjects": ", ".join(subject for subject, _, _ in subject_rows),
                    }
                )

    return pd.DataFrame(rows)


def make_title_page(freqs):
    fig = plt.figure(figsize=(8.5, 11))
    ax = fig.add_axes([0.07, 0.06, 0.86, 0.88])
    ax.axis("off")

    y = 0.97
    ax.text(0, y, "Retrieval PAC Paired T-Tests", fontsize=18, fontweight="bold", va="top")
    y -= 0.06
    ax.text(0, y, "Stim vs No Stim, remembered and forgotten trials separately", fontsize=12, va="top")
    y -= 0.05

    freq_min = float(np.min(freqs)) if freqs is not None and len(freqs) else float("nan")
    freq_max = float(np.max(freqs)) if freqs is not None and len(freqs) else float("nan")

    paragraphs = [
        "Tests are paired within-subject t-tests comparing stim and no-stim retrieval PAC (post - pre modulation index).",
        "Each value is the mean baseline-corrected MI within the selected band for a given region pair and patient.",
        "Band definitions: Theta = 4-8 Hz, Slow gamma = 35-50 Hz.",
        f"PAC TG spectra cover {freq_min:.1f}-{freq_max:.1f} Hz; bands without any frequency samples in that range are omitted.",
        "Remembered and forgotten trials are analyzed separately, matching the panel structure in the retrieval figure.",
        "No family-wise correction was applied.",
    ]
    for paragraph in paragraphs:
        wrapped = textwrap.fill(paragraph, width=92)
        ax.text(0, y, wrapped, fontsize=10.5, va="top")
        y -= 0.04 * (wrapped.count("\n") + 1) + 0.02

    return fig


def make_summary_page(summary_df):
    sig_df = summary_df[summary_df["p"].notna() & (summary_df["p"] < 0.05)].copy()
    lines = []
    if sig_df.empty:
        lines.append("No region-band comparison reached p < .05.")
    else:
        for _, row in sig_df.sort_values(["p", "Memory", "Region", "Band"]).iterrows():
            lines.append(
                f"{row['Memory']} | {pretty_in_text(row['Region'])} | {row['Band']}: "
                f"t = {fmt_num(row['t'], decimals=3)}, p = {p_str(row['p'])} ({row['Sig']})."
            )

    fig = plt.figure(figsize=(8.5, 11))
    ax = fig.add_axes([0.07, 0.06, 0.86, 0.88])
    ax.axis("off")
    ax.text(0, 0.97, "Significant Results", fontsize=16, fontweight="bold", va="top")
    y = 0.91
    for line in lines:
        wrapped = textwrap.fill(line, width=94)
        ax.text(0, y, wrapped, fontsize=10.5, va="top")
        y -= 0.04 * (wrapped.count("\n") + 1) + 0.02
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

    display_df = df[["Region", "N", "Mean difference (stim - no stim)", "t", "p", "Sig"]].copy()
    display_df["Region"] = display_df["Region"].map(pretty_in_text)
    display_df["Mean difference (stim - no stim)"] = display_df["Mean difference (stim - no stim)"].map(fmt_num)
    display_df["t"] = display_df["t"].map(lambda x: fmt_num(x, decimals=3))
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


def main():
    reset_dir(OUTPUT_DIR)
    all_data = load_retrieval_data()
    stats_df = paired_band_stats(all_data)
    if stats_df.empty:
        raise RuntimeError("No paired retrieval PAC statistics could be computed.")

    stats_df = stats_df.sort_values(["Memory", "Region", "Band"]).reset_index(drop=True)
    stats_df.to_csv(CSV_PATH, index=False)

    freqs = np.asarray(all_data.get("freqs_diff"), dtype=float)

    with PdfPages(PDF_PATH) as pdf:
        pdf.savefig(make_title_page(freqs))
        plt.close("all")

        for memory_label in ["Remembered", "Forgotten"]:
            memory_df = stats_df[stats_df["Memory"] == memory_label].copy()
            for band_label in ["Theta", "Slow gamma"]:
                band_df = memory_df[memory_df["Band"] == band_label].copy()
                if band_df.empty:
                    continue
                chunks = chunk_dataframe(band_df, chunk_size=28)
                total = len(chunks)
                for idx, chunk in enumerate(chunks, start=1):
                    title = f"{memory_label} Trials | {band_label}"
                    if total > 1:
                        title += f" ({idx}/{total})"
                    pdf.savefig(make_table_page(title, chunk))
                    plt.close("all")

        pdf.savefig(make_summary_page(stats_df))
        plt.close("all")

    print(f"Wrote paired retrieval PAC tests to: {PDF_PATH}")
    print(f"Wrote tabular statistics to: {CSV_PATH}")


if __name__ == "__main__":
    main()
