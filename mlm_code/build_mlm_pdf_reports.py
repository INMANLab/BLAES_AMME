#!/usr/bin/env python3
"""Generate APA-formatted PDF reports for each MLM analysis."""

import os
import csv
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = Path(__file__).resolve().parent
STATS = BASE / "outputs" / "stats"

# Analysis configs: (stats_dir_name, display_title, output_folder, phase, measure, is_band_by_region)
ANALYSES = [
    # --- Power: MTL cortical (BLA, HPC, EC, PRC) ---
    ("enc_power_mtl_rbband",  "Encoding Power (BLA, HPC, EC, PRC) \u2013 Region-by-Band",   "encoding_power",   "Encoding",  "Power", False),
    ("enc_power_mtl_bbreg",   "Encoding Power (BLA, HPC, EC, PRC) \u2013 Band-by-Region",   "encoding_power",   "Encoding",  "Power", True),
    ("ret_power_mtl_rbband",  "Retrieval Power (BLA, HPC, EC, PRC) \u2013 Region-by-Band",  "retrieval_power",  "Retrieval", "Power", False),
    ("ret_power_mtl_bbreg",   "Retrieval Power (BLA, HPC, EC, PRC) \u2013 Band-by-Region",  "retrieval_power",  "Retrieval", "Power", True),
    # --- Power: HPC subfields (BLA, CA, DG) ---
    ("enc_power_hpc_rbband",  "Encoding Power (BLA, CA, DG) \u2013 Region-by-Band",         "encoding_power",   "Encoding",  "Power", False),
    ("enc_power_hpc_bbreg",   "Encoding Power (BLA, CA, DG) \u2013 Band-by-Region",         "encoding_power",   "Encoding",  "Power", True),
    ("ret_power_hpc_rbband",  "Retrieval Power (BLA, CA, DG) \u2013 Region-by-Band",        "retrieval_power",  "Retrieval", "Power", False),
    ("ret_power_hpc_bbreg",   "Retrieval Power (BLA, CA, DG) \u2013 Band-by-Region",        "retrieval_power",  "Retrieval", "Power", True),
    # --- Coherence: MTL cortical ---
    ("enc_coh_mtl_rbband",    "Encoding Coherence (BLA, HPC, EC, PRC) \u2013 Region-by-Band",  "encoding_coherence",  "Encoding",  "Coherence", False),
    ("enc_coh_mtl_bbreg",     "Encoding Coherence (BLA, HPC, EC, PRC) \u2013 Band-by-Region",  "encoding_coherence",  "Encoding",  "Coherence", True),
    ("ret_coh_mtl_rbband",    "Retrieval Coherence (BLA, HPC, EC, PRC) \u2013 Region-by-Band", "retrieval_coherence", "Retrieval", "Coherence", False),
    ("ret_coh_mtl_bbreg",     "Retrieval Coherence (BLA, HPC, EC, PRC) \u2013 Band-by-Region", "retrieval_coherence", "Retrieval", "Coherence", True),
    # --- Coherence: HPC subfields ---
    ("enc_coh_hpc_rbband",    "Encoding Coherence (BLA, CA, DG) \u2013 Region-by-Band",        "encoding_coherence",  "Encoding",  "Coherence", False),
    ("enc_coh_hpc_bbreg",     "Encoding Coherence (BLA, CA, DG) \u2013 Band-by-Region",        "encoding_coherence",  "Encoding",  "Coherence", True),
    ("ret_coh_hpc_rbband",    "Retrieval Coherence (BLA, CA, DG) \u2013 Region-by-Band",       "retrieval_coherence", "Retrieval", "Coherence", False),
    ("ret_coh_hpc_bbreg",     "Retrieval Coherence (BLA, CA, DG) \u2013 Band-by-Region",       "retrieval_coherence", "Retrieval", "Coherence", True),
    # --- PAC ---
    ("enc_pac_rbband",        "Encoding PAC \u2013 Region-by-Band",                              "encoding_pac",  "Encoding",  "PAC", False),
    ("enc_pac_bbreg",         "Encoding PAC \u2013 Band-by-Region",                              "encoding_pac",  "Encoding",  "PAC", True),
    ("ret_pac_rbband",        "Retrieval PAC \u2013 Region-by-Band",                             "retrieval_pac", "Retrieval", "PAC", False),
    ("ret_pac_bbreg",         "Retrieval PAC \u2013 Band-by-Region",                             "retrieval_pac", "Retrieval", "PAC", True),
]

SIG_THRESHOLD = 0.05

# ---------------------------------------------------------------------------
# APA style settings for matplotlib
# ---------------------------------------------------------------------------
APA_FONT = {"family": "serif", "size": 11}
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "text.usetex": False,
})

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def read_csv_safe(path):
    """Read a CSV, return list of dicts."""
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)

def safe_float(val, default=None):
    try:
        return float(val)
    except (ValueError, TypeError):
        return default

def fmt_p(p):
    if p is None:
        return ""
    if p < 0.001:
        return "< .001"
    return f"{p:.3f}".lstrip("0")

def fmt_val(v, decimals=2):
    if v is None:
        return ""
    return f"{v:.{decimals}f}"

def is_sig(p):
    return p is not None and p < SIG_THRESHOLD

def display_region(name):
    # Check if it's a band name first
    if name in BAND_DISPLAY_NAMES:
        return BAND_DISPLAY_NAMES[name]
    return name.replace("_", " \u2013 ") if "_" in name else name


# ---------------------------------------------------------------------------
# Discover subunits in a stats dir
# ---------------------------------------------------------------------------
def discover_subunits(stats_dir, is_band_by_region):
    """Return list of subunit names (regions or bands) found."""
    subunits = set()
    for level in ("patient_level", "trial_level"):
        level_dir = stats_dir / level
        if level_dir.is_dir():
            for child in sorted(level_dir.iterdir()):
                if child.is_dir() and (child / "coefficients.csv").exists():
                    subunits.add(child.name)
    return sorted(subunits)


# ---------------------------------------------------------------------------
# Build overview fit summary table figure
# ---------------------------------------------------------------------------
def fig_fit_summary(stats_dir, is_band_by_region):
    """Create a figure with the overview fit summary tables."""
    tables_data = []

    if is_band_by_region:
        # Band-by-region has per-band overview files
        for level in ("patient_level", "trial_level"):
            for band in ("theta", "slow_gamma", "fast_gamma"):
                fname = f"{level}_{band}_fit_summary.csv"
                rows = read_csv_safe(stats_dir / "overview" / fname)
                if rows:
                    band_label = band.replace("_", " ").title()
                    tables_data.append((f"{level.replace('_', ' ').title()} - {band_label}", rows))
    else:
        for level in ("patient_level", "trial_level"):
            fname = f"{level}_fit_summary.csv"
            rows = read_csv_safe(stats_dir / "overview" / fname)
            if rows:
                tables_data.append((level.replace("_", " ").title(), rows))

    if not tables_data:
        return None

    n_tables = len(tables_data)
    fig_height = max(3, 1.5 * n_tables + 1)
    fig, axes = plt.subplots(n_tables, 1, figsize=(7.5, fig_height))
    if n_tables == 1:
        axes = [axes]
    fig.subplots_adjust(hspace=0.6, top=0.92, bottom=0.05)

    for ax, (title, rows) in zip(axes, tables_data):
        ax.axis("off")
        cell_data = [[r.get("Metric", ""), r.get("Value", "")] for r in rows]
        col_labels = ["Metric", "Value"]

        tbl = ax.table(cellText=cell_data, colLabels=col_labels,
                       loc="center", cellLoc="left")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1, 1.3)

        # APA table styling
        for (row, col), cell in tbl.get_celld().items():
            cell.set_edgecolor("white")
            cell.set_linewidth(0)
            if row == 0:
                cell.set_text_props(weight="bold")
                cell.set_facecolor("#f0f0f0")
            else:
                cell.set_facecolor("white")

        # Add top and bottom rules
        ax.plot([0.05, 0.95], [1.02, 1.02], transform=ax.transAxes,
                color="black", linewidth=1.2, clip_on=False)
        ax.plot([0.05, 0.95], [-0.02, -0.02], transform=ax.transAxes,
                color="black", linewidth=1.2, clip_on=False)

        ax.set_title(f"Table. Model Fit Summary \u2013 {title}",
                     fontsize=10, fontstyle="italic", loc="left", pad=8)

    return fig


# ---------------------------------------------------------------------------
# Build coefficients table figure for a subunit
# ---------------------------------------------------------------------------
def fig_coefficients_table(rows, title, level_label):
    """APA-style coefficients table as a figure."""
    if not rows:
        return None

    est_label = rows[0].get("estimate_label", "Estimate")

    headers = ["Predictor", est_label, "SE", "95% CI", "Statistic", "p"]
    cell_data = []
    bold_rows = []
    for i, r in enumerate(rows):
        p = safe_float(r.get("p_value"))
        est = safe_float(r.get("estimate"))
        se = safe_float(r.get("std_error"))
        cl = safe_float(r.get("conf_low"))
        ch = safe_float(r.get("conf_high"))
        stat = safe_float(r.get("statistic"))
        predictor = r.get("Predictor", "")

        ci_str = f"[{fmt_val(cl)}, {fmt_val(ch)}]" if cl is not None else ""
        p_str = fmt_p(p)
        if is_sig(p):
            p_str += "*"
            bold_rows.append(i)

        cell_data.append([predictor, fmt_val(est), fmt_val(se), ci_str,
                          fmt_val(stat), p_str])

    n_rows = len(cell_data)
    fig_height = max(2, 0.35 * n_rows + 1.2)
    fig, ax = plt.subplots(figsize=(7.5, fig_height))
    ax.axis("off")

    tbl = ax.table(cellText=cell_data, colLabels=headers,
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1, 1.4)

    # Column widths
    col_widths = [0.28, 0.12, 0.10, 0.20, 0.14, 0.10]
    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor("white")
        cell.set_linewidth(0)
        cell.set_width(col_widths[col])
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#f0f0f0")
        else:
            cell.set_facecolor("white")
            if col == 0:
                cell.set_text_props(ha="left")
            # Bold significant rows
            if (row - 1) in bold_rows:
                cell.set_text_props(weight="bold")

    # APA horizontal rules
    ax.plot([0.02, 0.98], [1.05, 1.05], transform=ax.transAxes,
            color="black", linewidth=1.2, clip_on=False)
    ax.plot([0.02, 0.98], [-0.05, -0.05], transform=ax.transAxes,
            color="black", linewidth=1.2, clip_on=False)

    ax.set_title(f"Table. {level_label} Fixed Effects \u2013 {title}",
                 fontsize=10, fontstyle="italic", loc="left", pad=10)

    # Note
    fig.text(0.05, 0.01, "Note. * p < .05. CI = 95% Wald confidence interval.",
             fontsize=8, fontstyle="italic")

    fig.tight_layout(rect=[0, 0.04, 1, 0.98])
    return fig


# ---------------------------------------------------------------------------
# Build model comparison table figure
# ---------------------------------------------------------------------------
MODEL_FORMULAS_POWER_COH = {
    "m0": "Accuracy ~ 1 + (1 | Patient)",
    "m1": "Accuracy ~ StimCond + (1 | Patient)",
    "m2": "Accuracy ~ StimCond + Theta(4\u20138Hz)_c + (1 | Patient)",
    "m3": "Accuracy ~ StimCond + SlowGamma(30\u201355Hz)_c + (1 | Patient)",
    "m4": "Accuracy ~ StimCond + Theta_c + SlowGamma_c + HFA(55\u2013100Hz)_c + (1 | Patient)",
    "m_full": "Accuracy ~ (Theta_c + SlowGamma_c + HFA_c) * StimCond + (1 | Patient)",
}

MODEL_FORMULAS_PAC_ENC = {
    "m0": "Accuracy ~ 1 + (1 | Patient)",
    "m1": "Accuracy ~ StimCond + (1 | Patient)",
    "m2": "Accuracy ~ StimCond + SlowGammaPAC(30\u201350Hz)_c + (1 | Patient)",
    "m3": "Accuracy ~ StimCond + HFA_PAC(55\u2013100Hz)_c + (1 | Patient)",
    "m4": "Accuracy ~ StimCond + SlowGammaPAC_c + HFA_PAC_c + (1 | Patient)",
    "m_full": "Accuracy ~ (SlowGammaPAC_c + HFA_PAC_c) * StimCond + (1 | Patient)",
}

MODEL_FORMULAS_PAC_RET = {
    "m0": "Accuracy ~ 1 + (1 | Patient)",
    "m1": "Accuracy ~ StimCond + (1 | Patient)",
    "m2": "Accuracy ~ StimCond + SlowGammaPAC(30\u201350Hz)_c + (1 | Patient)",
    "m_full": "Accuracy ~ SlowGammaPAC_c * StimCond + (1 | Patient)",
}

# Band name lookup for band-by-region subunit names
BAND_DISPLAY_NAMES = {
    "theta": "Theta (4\u20138 Hz)",
    "slow_gamma": "Slow Gamma (30\u201355 Hz)",
    "fast_gamma": "HFA (55\u2013100 Hz)",
    "slow_gamma_pac": "Theta Phase \u00d7 Slow Gamma Amp (30\u201350 Hz)",
    "hfa_pac": "Theta Phase \u00d7 HFA Amp (55\u2013100 Hz)",
}

def fig_model_comparison(rows, title, level_label, measure="Power", phase="Encoding"):
    """APA model comparison table."""
    if not rows:
        return None

    headers = ["Model", "npar", "AIC", "BIC", "logLik", "\u03c7\u00b2", "df", "p"]
    cell_data = []
    bold_rows = []
    for i, r in enumerate(rows):
        p = safe_float(r.get("p_value"))
        chisq = safe_float(r.get("Chisq"))
        df = safe_float(r.get("Df"))
        p_str = fmt_p(p)
        if is_sig(p):
            p_str += "*"
            bold_rows.append(i)

        cell_data.append([
            r.get("Model", ""),
            fmt_val(safe_float(r.get("npar")), 0),
            fmt_val(safe_float(r.get("AIC")), 1),
            fmt_val(safe_float(r.get("BIC")), 1),
            fmt_val(safe_float(r.get("logLik")), 1),
            fmt_val(chisq, 2) if chisq is not None else "",
            fmt_val(df, 0) if df is not None else "",
            p_str,
        ])

    n_rows = len(cell_data)
    fig_height = max(2.5, 0.35 * n_rows + 2.2)
    fig, ax = plt.subplots(figsize=(7.5, fig_height))
    ax.axis("off")

    tbl = ax.table(cellText=cell_data, colLabels=headers,
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1, 1.4)

    col_widths = [0.10, 0.08, 0.13, 0.13, 0.13, 0.13, 0.08, 0.12]
    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor("white")
        cell.set_linewidth(0)
        cell.set_width(col_widths[col])
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#f0f0f0")
        else:
            cell.set_facecolor("white")
            if (row - 1) in bold_rows:
                cell.set_text_props(weight="bold")

    ax.plot([0.02, 0.98], [1.05, 1.05], transform=ax.transAxes,
            color="black", linewidth=1.2, clip_on=False)
    ax.plot([0.02, 0.98], [-0.05, -0.05], transform=ax.transAxes,
            color="black", linewidth=1.2, clip_on=False)

    ax.set_title(f"Table. {level_label} Model Comparison \u2013 {title}",
                 fontsize=10, fontstyle="italic", loc="left", pad=10)

    # Pick the right formula set
    if measure == "PAC":
        formulas = MODEL_FORMULAS_PAC_ENC if phase == "Encoding" else MODEL_FORMULAS_PAC_RET
    else:
        formulas = MODEL_FORMULAS_POWER_COH

    # Only show formulas for models present in the data
    model_names_in_data = {r.get("Model", "") for r in rows}
    formula_lines = []
    for model_name, formula in formulas.items():
        if model_name in model_names_in_data:
            formula_lines.append(f"{model_name}: {formula}")
    note_text = "Note. * p < .05.\n" + "\n".join(formula_lines)
    fig.text(0.03, 0.01, note_text, fontsize=6.5, fontstyle="italic",
             verticalalignment="bottom", family="monospace")

    fig.tight_layout(rect=[0, 0.16, 1, 0.98])
    return fig


# ---------------------------------------------------------------------------
# Forest plot of fixed effects
# ---------------------------------------------------------------------------
def fig_forest_plot(coef_rows, title, level_label):
    """Forest plot showing effect sizes with CIs."""
    if not coef_rows:
        return None

    # Skip intercept
    predictors = []
    estimates = []
    ci_low = []
    ci_high = []
    p_vals = []
    for r in coef_rows:
        pred = r.get("Predictor", "")
        if "Intercept" in pred:
            continue
        est = safe_float(r.get("estimate"))
        cl = safe_float(r.get("conf_low"))
        ch = safe_float(r.get("conf_high"))
        p = safe_float(r.get("p_value"))
        if est is None:
            continue
        predictors.append(pred)
        estimates.append(est)
        ci_low.append(cl if cl is not None else est)
        ci_high.append(ch if ch is not None else est)
        p_vals.append(p)

    if not predictors:
        return None

    n = len(predictors)
    fig_height = max(2.5, 0.45 * n + 1.0)
    fig, ax = plt.subplots(figsize=(7.5, fig_height))

    y_pos = np.arange(n)
    est_label = coef_rows[0].get("estimate_label", "Estimate")
    is_odds = "Odds" in est_label
    ref_line = 1.0 if is_odds else 0.0

    colors = []
    for p in p_vals:
        if is_sig(p):
            colors.append("#d62728")  # red for significant
        else:
            colors.append("#1f77b4")  # blue for non-significant

    xerr_low = [est - cl for est, cl in zip(estimates, ci_low)]
    xerr_high = [ch - est for est, ch in zip(estimates, ci_high)]

    for i in range(n):
        ax.errorbar(estimates[i], y_pos[i],
                    xerr=[[xerr_low[i]], [xerr_high[i]]],
                    fmt="none", ecolor=colors[i], elinewidth=1.5, capsize=3)
        marker = "D" if is_sig(p_vals[i]) else "o"
        ax.plot(estimates[i], y_pos[i], marker, color=colors[i],
                markersize=7, zorder=5)

    ax.axvline(ref_line, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(predictors, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel(est_label, fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_title(f"Figure. {level_label} Fixed Effects Forest Plot \u2013 {title}",
                 fontsize=10, fontstyle="italic", loc="left", pad=10)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="D", color="w", markerfacecolor="#d62728",
               markersize=7, label="p < .05"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#1f77b4",
               markersize=7, label="p >= .05"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8,
              frameon=True, edgecolor="gray")

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Summary of significant findings
# ---------------------------------------------------------------------------
def collect_significant_findings(stats_dir, is_band_by_region):
    """Collect all significant p < .05 fixed effects across subunits."""
    findings = []
    subunits = discover_subunits(stats_dir, is_band_by_region)

    for level in ("patient_level", "trial_level"):
        for sub in subunits:
            coef_path = stats_dir / level / sub / "coefficients.csv"
            rows = read_csv_safe(coef_path)
            for r in rows:
                p = safe_float(r.get("p_value"))
                if is_sig(p):
                    pred = r.get("Predictor", "")
                    if "Intercept" in pred:
                        continue
                    est = safe_float(r.get("estimate"))
                    est_label = r.get("estimate_label", "Estimate")
                    findings.append({
                        "level": level.replace("_", " ").title(),
                        "subunit": display_region(sub),
                        "predictor": pred,
                        "estimate": est,
                        "estimate_label": est_label,
                        "p": p,
                    })

    # Also check model comparisons
    mc_findings = []
    for level in ("patient_level", "trial_level"):
        for sub in subunits:
            mc_path = stats_dir / level / sub / "model_comparison.csv"
            rows = read_csv_safe(mc_path)
            for r in rows:
                p = safe_float(r.get("p_value"))
                if is_sig(p):
                    mc_findings.append({
                        "level": level.replace("_", " ").title(),
                        "subunit": display_region(sub),
                        "model": r.get("Model", ""),
                        "chisq": safe_float(r.get("Chisq")),
                        "df": safe_float(r.get("Df")),
                        "p": p,
                    })

    return findings, mc_findings


def fig_significant_summary(findings, mc_findings, analysis_title):
    """Create a summary figure of all significant findings."""
    if not findings and not mc_findings:
        fig, ax = plt.subplots(figsize=(7.5, 2))
        ax.axis("off")
        ax.text(0.5, 0.5, "No significant effects found at p < .05.",
                ha="center", va="center", fontsize=12, fontstyle="italic",
                transform=ax.transAxes)
        ax.set_title(f"Summary of Significant Findings \u2013 {analysis_title}",
                     fontsize=12, weight="bold", loc="left")
        fig.tight_layout()
        return fig

    pages = []

    # Fixed effects summary
    if findings:
        headers = ["Level", "Region/Band", "Predictor", "Est. Label", "Estimate", "p"]
        cell_data = []
        for f in findings:
            cell_data.append([
                f["level"], f["subunit"], f["predictor"],
                f["estimate_label"], fmt_val(f["estimate"]), fmt_p(f["p"]) + "*"
            ])

        n_rows = len(cell_data)
        fig_height = max(3, 0.35 * n_rows + 1.5)
        fig, ax = plt.subplots(figsize=(7.5, fig_height))
        ax.axis("off")

        tbl = ax.table(cellText=cell_data, colLabels=headers,
                       loc="center", cellLoc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        tbl.scale(1, 1.4)

        col_widths = [0.14, 0.16, 0.26, 0.14, 0.14, 0.10]
        for (row, col), cell in tbl.get_celld().items():
            cell.set_edgecolor("white")
            cell.set_linewidth(0)
            cell.set_width(col_widths[col])
            if row == 0:
                cell.set_text_props(weight="bold")
                cell.set_facecolor("#f0f0f0")
            else:
                cell.set_facecolor("white")
                cell.set_text_props(weight="bold")

        ax.plot([0.02, 0.98], [1.05, 1.05], transform=ax.transAxes,
                color="black", linewidth=1.2, clip_on=False)
        ax.plot([0.02, 0.98], [-0.05, -0.05], transform=ax.transAxes,
                color="black", linewidth=1.2, clip_on=False)

        ax.set_title(f"Table. Significant Fixed Effects (p < .05) \u2013 {analysis_title}",
                     fontsize=10, fontstyle="italic", loc="left", pad=10)
        fig.text(0.05, 0.01, "Note. * p < .05. Only predictors with p < .05 shown (excluding intercept).",
                 fontsize=8, fontstyle="italic")
        fig.tight_layout(rect=[0, 0.04, 1, 0.98])
        pages.append(fig)

    # Model comparison summary
    if mc_findings:
        headers = ["Level", "Region/Band", "Model Step", "\u03c7\u00b2", "df", "p"]
        cell_data = []
        for f in mc_findings:
            cell_data.append([
                f["level"], f["subunit"], f["model"],
                fmt_val(f["chisq"]), fmt_val(f["df"], 0), fmt_p(f["p"]) + "*"
            ])

        n_rows = len(cell_data)
        fig_height = max(3, 0.35 * n_rows + 1.5)
        fig, ax = plt.subplots(figsize=(7.5, fig_height))
        ax.axis("off")

        tbl = ax.table(cellText=cell_data, colLabels=headers,
                       loc="center", cellLoc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        tbl.scale(1, 1.4)

        col_widths = [0.15, 0.18, 0.18, 0.15, 0.10, 0.12]
        for (row, col), cell in tbl.get_celld().items():
            cell.set_edgecolor("white")
            cell.set_linewidth(0)
            cell.set_width(col_widths[col])
            if row == 0:
                cell.set_text_props(weight="bold")
                cell.set_facecolor("#f0f0f0")
            else:
                cell.set_facecolor("white")
                cell.set_text_props(weight="bold")

        ax.plot([0.02, 0.98], [1.05, 1.05], transform=ax.transAxes,
                color="black", linewidth=1.2, clip_on=False)
        ax.plot([0.02, 0.98], [-0.05, -0.05], transform=ax.transAxes,
                color="black", linewidth=1.2, clip_on=False)

        ax.set_title(f"Table. Significant Model Comparison Steps (p < .05) \u2013 {analysis_title}",
                     fontsize=10, fontstyle="italic", loc="left", pad=10)
        fig.text(0.05, 0.01, "Note. * p < .05. Only model steps with significant improvement shown.",
                 fontsize=8, fontstyle="italic")
        fig.tight_layout(rect=[0, 0.04, 1, 0.98])
        pages.append(fig)

    return pages


# ---------------------------------------------------------------------------
# Title page
# ---------------------------------------------------------------------------
def fig_title_page(title, phase, measure, n_patients, n_regions, n_obs):
    fig, ax = plt.subplots(figsize=(7.5, 10))
    ax.axis("off")

    ax.text(0.5, 0.75, title, ha="center", va="center",
            fontsize=18, weight="bold", family="serif")

    # Method description
    is_bbr = "Band-by-Region" in title
    if measure == "PAC":
        if phase == "Encoding":
            band_desc = ("theta-phase \u00d7 slow gamma amplitude (30\u201350 Hz) and "
                        "theta-phase \u00d7 HFA amplitude (55\u2013100 Hz)")
        else:
            band_desc = "theta-phase \u00d7 slow gamma amplitude (30\u201350 Hz)"

        if is_bbr:
            method_text = (
                f"Multilevel models (MLMs) examined whether phase-amplitude coupling (PAC) "
                f"predicted memory accuracy during {phase.lower()}. "
                f"PAC measures: {band_desc}. "
                f"For each coupling measure, a model was fit with the PAC value, "
                f"stimulation condition, and region pair as fixed effects, with random intercepts "
                f"for patients.\n\n"
                f"Patient-level: lmer (continuous accuracy). "
                f"Trial-level: glmer (binomial, binary accuracy). "
                f"All PAC values were baseline-corrected (post \u2013 pre)."
            )
        else:
            method_text = (
                f"Multilevel models (MLMs) examined whether phase-amplitude coupling (PAC) "
                f"predicted memory accuracy during {phase.lower()}. "
                f"PAC measures: {band_desc}. "
                f"Nested models were fit per region pair, adding PAC predictors incrementally.\n\n"
                f"Patient-level: lmer (continuous accuracy). "
                f"Trial-level: glmer (binomial, binary accuracy). "
                f"All PAC values were baseline-corrected (post \u2013 pre)."
            )
    elif is_bbr:
        method_text = (
            f"Multilevel models (MLMs) examined whether {measure.lower()} "
            f"in individual frequency bands predicted memory accuracy during {phase.lower()}. "
            f"For each band (Theta 4\u20138 Hz, Slow Gamma 30\u201355 Hz, HFA 55\u2013100 Hz), "
            f"a model was fit with the band value, stimulation condition, and region as "
            f"fixed effects, with random intercepts for patients.\n\n"
            f"Patient-level: lmer (continuous accuracy). "
            f"Trial-level: glmer (binomial, binary accuracy)."
        )
    else:
        method_text = (
            f"Multilevel models (MLMs) examined whether {measure.lower()} "
            f"in three frequency bands (Theta 4\u20138 Hz, Slow Gamma 30\u201355 Hz, HFA "
            f"55\u2013100 Hz) predicted memory accuracy during {phase.lower()}. "
            f"Nested models were fit per brain region, adding predictors incrementally.\n\n"
            f"Patient-level: lmer (continuous accuracy). "
            f"Trial-level: glmer (binomial, binary accuracy). "
            f"Random intercepts for patients."
        )

    ax.text(0.5, 0.45, method_text, ha="center", va="center",
            fontsize=9.5, family="serif", wrap=True,
            transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="gray", alpha=0.8),
            multialignment="left",
            fontdict={"linespacing": 1.5})

    # Sample info
    sample_text = (
        f"N patients = {n_patients}    |    "
        f"N regions = {n_regions}    |    "
        f"Total observations = {n_obs}"
    )
    ax.text(0.5, 0.12, sample_text, ha="center", va="center",
            fontsize=10, family="serif")

    ax.text(0.5, 0.05, "Significance threshold: p < .05",
            ha="center", va="center", fontsize=10, fontstyle="italic", family="serif")

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Interpretation text page
# ---------------------------------------------------------------------------
def fig_interpretation_page(findings, mc_findings, analysis_title, phase, measure):
    """Generate a page with written interpretation of results."""
    fig, ax = plt.subplots(figsize=(7.5, 10))
    ax.axis("off")

    ax.text(0.5, 0.97, f"Results Interpretation \u2013 {analysis_title}",
            ha="center", va="top", fontsize=14, weight="bold", family="serif")

    lines = []

    if not findings and not mc_findings:
        lines.append(
            f"No significant effects were found in the {measure.lower()} MLM analyses "
            f"during {phase.lower()} at the p < .05 threshold. Neither the frequency-band "
            f"predictors (theta, slow gamma, fast gamma) nor stimulation condition, nor their "
            f"interactions, significantly predicted memory accuracy in any brain region at "
            f"either the patient or trial level."
        )
    else:
        lines.append(f"Significant effects (p < .05) in the {phase.lower()} {measure.lower()} MLM analyses:")
        lines.append("")

        # Group findings by level
        for level_name in ("Patient Level", "Trial Level"):
            level_findings = [f for f in findings if f["level"] == level_name]
            if level_findings:
                lines.append(f"{level_name}:")
                for f in level_findings:
                    est = f["estimate"]
                    est_label = f["estimate_label"]
                    direction = ""
                    if "Odds" in est_label:
                        if est > 1:
                            direction = "increased odds of remembering"
                        elif est < 1:
                            direction = "decreased odds of remembering"
                    else:
                        if est > 0:
                            direction = "positive association with accuracy"
                        elif est < 0:
                            direction = "negative association with accuracy"

                    lines.append(
                        f"  \u2022 {f['subunit']}: {f['predictor']} "
                        f"({est_label} = {fmt_val(est)}, p = {fmt_p(f['p'])}): "
                        f"{direction}."
                    )
                lines.append("")

        # Model comparison findings
        level_mc = {}
        for f in mc_findings:
            level_mc.setdefault(f["level"], []).append(f)

        if level_mc:
            lines.append("Significant model improvement steps:")
            lines.append("")
            for level_name, mc_list in level_mc.items():
                lines.append(f"{level_name}:")
                for f in mc_list:
                    chisq_str = fmt_val(f["chisq"]) if f["chisq"] is not None else "N/A"
                    df_str = fmt_val(f["df"], 0) if f["df"] is not None else "N/A"
                    lines.append(
                        f"  \u2022 {f['subunit']}: Adding {f['model']} significantly improved fit "
                        f"(\u03c7\u00b2({df_str}) = {chisq_str}, p = {fmt_p(f['p'])})."
                    )
                lines.append("")

    text = "\n".join(lines)
    ax.text(0.05, 0.90, text, ha="left", va="top",
            fontsize=9.5, family="serif", transform=ax.transAxes,
            linespacing=1.6, verticalalignment="top")

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main: build PDF for one analysis
# ---------------------------------------------------------------------------
def build_pdf(stats_dir_name, display_title, output_folder, phase, measure, is_band_by_region):
    stats_dir = STATS / stats_dir_name
    if not stats_dir.is_dir():
        print(f"  SKIP: {stats_dir} does not exist")
        return

    subunits = discover_subunits(stats_dir, is_band_by_region)
    if not subunits:
        print(f"  SKIP: No subunits found in {stats_dir}")
        return

    # Get overview info for title page
    n_patients = n_regions = n_obs = "N/A"
    # Try to read from any overview file
    for level in ("trial_level", "patient_level"):
        if is_band_by_region:
            fname = f"{level}_theta_fit_summary.csv"
        else:
            fname = f"{level}_fit_summary.csv"
        overview_rows = read_csv_safe(stats_dir / "overview" / fname)
        for r in overview_rows:
            if r.get("Metric") == "N patients":
                n_patients = r.get("Value", "N/A")
            elif r.get("Metric") == "N regions":
                n_regions = r.get("Value", "N/A")
            elif r.get("Metric") == "Observations":
                n_obs = r.get("Value", "N/A")
        if n_patients != "N/A":
            break

    # Determine output path
    # Use a filename based on the stats dir to avoid collisions
    pdf_name = f"{stats_dir_name}_mlm_report.pdf"
    out_dir = BASE / "outputs" / output_folder
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / pdf_name

    print(f"  Building: {pdf_path.relative_to(BASE)}")

    with PdfPages(str(pdf_path)) as pdf:
        # Title page
        fig = fig_title_page(display_title, phase, measure, n_patients, n_regions, n_obs)
        pdf.savefig(fig)
        plt.close(fig)

        # Overview fit summary
        fig = fig_fit_summary(stats_dir, is_band_by_region)
        if fig:
            pdf.savefig(fig)
            plt.close(fig)

        # Per-subunit results
        for sub in subunits:
            sub_display = display_region(sub)

            for level in ("patient_level", "trial_level"):
                level_label = level.replace("_", " ").title()

                # Coefficients table
                coef_rows = read_csv_safe(stats_dir / level / sub / "coefficients.csv")
                if coef_rows:
                    fig = fig_coefficients_table(coef_rows, sub_display, level_label)
                    if fig:
                        pdf.savefig(fig)
                        plt.close(fig)

                    # Forest plot
                    fig = fig_forest_plot(coef_rows, sub_display, level_label)
                    if fig:
                        pdf.savefig(fig)
                        plt.close(fig)

                # Model comparison table
                mc_rows = read_csv_safe(stats_dir / level / sub / "model_comparison.csv")
                if mc_rows:
                    fig = fig_model_comparison(mc_rows, sub_display, level_label, measure=measure, phase=phase)
                    if fig:
                        pdf.savefig(fig)
                        plt.close(fig)

                # Embed existing summary panel if available
                panel_path = stats_dir / level / sub / "summary_panel.png"
                if panel_path.exists():
                    try:
                        img = plt.imread(str(panel_path))
                        h, w = img.shape[:2]
                        aspect = h / w
                        fig_w = 7.5
                        fig_h = min(10, fig_w * aspect)
                        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
                        ax.imshow(img)
                        ax.axis("off")
                        ax.set_title(
                            f"Figure. {level_label} Summary Panel \u2013 {sub_display}",
                            fontsize=10, fontstyle="italic", loc="left", pad=5)
                        fig.tight_layout()
                        pdf.savefig(fig)
                        plt.close(fig)
                    except Exception:
                        pass

        # Significant findings summary
        findings, mc_findings = collect_significant_findings(stats_dir, is_band_by_region)
        sig_figs = fig_significant_summary(findings, mc_findings, display_title)
        if isinstance(sig_figs, list):
            for fig in sig_figs:
                pdf.savefig(fig)
                plt.close(fig)
        elif sig_figs is not None:
            pdf.savefig(sig_figs)
            plt.close(sig_figs)

        # Interpretation page
        fig = fig_interpretation_page(findings, mc_findings, display_title, phase, measure)
        pdf.savefig(fig)
        plt.close(fig)

    print(f"  Done: {pdf_path.relative_to(BASE)}")


# ---------------------------------------------------------------------------
# Run all
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("Generating APA-formatted MLM PDF reports")
    print("=" * 60)

    for stats_dir_name, title, out_folder, phase, measure, is_bbr in ANALYSES:
        print(f"\n--- {title} ---")
        build_pdf(stats_dir_name, title, out_folder, phase, measure, is_bbr)

    print("\n" + "=" * 60)
    print("All reports complete.")
    print("=" * 60)
