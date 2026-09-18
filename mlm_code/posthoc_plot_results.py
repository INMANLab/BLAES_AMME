#!/usr/bin/env python
"""Plot post-hoc results: random-slope-vs-intercept comparison, Johnson-Neyman
regions, and per-panel stability summaries (LOSO + deltas + downsample).

One stability diagnostic figure per trending panel:
  <out_dir>/stability_<measure>_<unit>_<band>_diagnostics.png

One JN figure per trending interaction panel x model (RI/RS):
  <out_dir>/jn_<measure>_<unit>_<band>_<model>.png

One random-slope comparison summary figure per analysis type:
  <out_dir>/random_slope_comparison.png
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parent.parent
ENC = REPO / "OUTPUTS" / "encoding_memory_reports"

ANALYSES = {
    "alltrials":                          {"is_interaction": False, "contrast_label": "All trials"},
    "endogenous_memory_effect":           {"is_interaction": False, "contrast_label": "Endogenous (no-stim)"},
    "stim_effect":                        {"is_interaction": False, "contrast_label": "Stim only"},
    "GLMM_interactions_model_variations": {"is_interaction": True,  "contrast_label": "band_c x StimCond"},
}


def pretty_label(name):
    parts = []
    for p in str(name).split("_"):
        if p == "ALLHPC":
            parts.append("HPC")
        elif p == "HPC":
            parts.append("SUB")
        else:
            parts.append(p)
    return "-".join(parts)


def out_dir(analysis):
    d = ENC / analysis / "post_hoc_testing"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ── Stability summary figure ───────────────────────────────────────────────

def plot_stability(analysis, panel_row):
    od = out_dir(analysis)
    m, u, b = panel_row["measure"], panel_row["unit"], panel_row["band"]
    loso_csv  = od / f"stability_{m}_{u}_{b}_loso.csv"
    delta_csv = od / f"stability_{m}_{u}_{b}_deltas.csv"
    ds_csv    = od / f"stability_{m}_{u}_{b}_downsample.csv"
    if not (loso_csv.exists() and delta_csv.exists() and ds_csv.exists()):
        return None
    loso = pd.read_csv(loso_csv)
    deltas = pd.read_csv(delta_csv)
    ds = pd.read_csv(ds_csv)

    full = loso[loso["excluded"] == "<none>"].iloc[0]
    loso_rest = loso[loso["excluded"] != "<none>"].sort_values("OR")
    valid_p = loso["p"].dropna()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # LOSO
    ax = axes[0]
    y = np.arange(len(loso_rest))
    ax.errorbar(loso_rest["OR"], y,
                xerr=[loso_rest["OR"] - loso_rest["lo"],
                      loso_rest["hi"] - loso_rest["OR"]],
                fmt="o", color="#2b2b2b", ecolor="#888888",
                capsize=3, ms=5)
    ax.axvline(1.0, color="0.5", ls=":", lw=1.0, label="OR = 1")
    ax.axvline(full["OR"], color="#c0392b", ls="--", lw=1.5,
               label=f"Full OR = {full['OR']:.3g}")
    ax.set_yticks(y)
    ax.set_yticklabels(loso_rest["excluded"], fontsize=8)
    ax.set_xlabel("LOSO OR (95% CI)", fontsize=10, fontweight="bold")
    ax.set_title(
        f"LOSO  ({(loso_rest['p'] < 0.05).sum()}/{len(loso_rest)} p<.05)",
        fontsize=11, fontweight="bold",
    )
    if full["OR"] > 0 and not np.isnan(full["OR"]):
        try:
            ax.set_xscale("log")
        except Exception:
            pass
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.25, axis="x")

    # Subject deltas
    ax = axes[1]
    if "interaction_delta" in deltas.columns:
        col = "interaction_delta"
        ylab = ("Interaction delta\n"
                "(stim memory effect - nostim memory effect per subj)\n"
                "memory effect = mean(band | remembered) - mean(band | forgotten)")
    else:
        col = "delta"
        ylab = "mean(band | remembered) - mean(band | forgotten)"
    d = deltas[deltas[col].notna()].copy().sort_values(col)
    colors = ["#a0a0a0" if v < 0 else "#2b2b2b" for v in d[col]]
    ax.bar(range(len(d)), d[col], color=colors)
    ax.axhline(0, color="0.4", lw=1)
    if len(d):
        ax.axhline(d[col].mean(), color="#c0392b", ls="--", lw=1.5,
                   label=f"mean = {d[col].mean():+.4g}")
    ax.set_xticks(range(len(d)))
    ax.set_xticklabels(d["Patient"], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(ylab, fontsize=9, fontweight="bold")
    pos = int((d[col] > 0).sum()); neg = int((d[col] < 0).sum())
    ax.set_title(f"Subject deltas  ({pos} pos / {neg} neg)",
                 fontsize=11, fontweight="bold")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.25, axis="y")

    # Downsample
    ax = axes[2]
    ds_clean = ds.dropna(subset=["OR", "p"])
    n_ds = len(ds_clean)
    if n_ds > 0:
        median_or = float(ds_clean["OR"].median())
        ax.hist(ds_clean["p"], bins=np.linspace(0, 1, 41),
                color="#a0a0a0", edgecolor="white")
        ax.axvline(0.05, color="#c0392b", ls="--", lw=1.5, label="p = .05")
        ax.axvline(float(full["p"]), color="#1f77b4", ls=":", lw=1.5,
                   label=f"Full p = {full['p']:.3f}")
        sig_frac = (ds_clean["p"] < 0.05).mean()
        ax.set_title(
            f"Downsample p ({int(sig_frac * n_ds)}/{n_ds} < .05)\n"
            f"DS median OR = {median_or:.3g}",
            fontsize=11, fontweight="bold",
        )
    ax.set_xlabel("p-value", fontsize=10, fontweight="bold")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.25, axis="x")

    info = ANALYSES[analysis]
    fig.suptitle(
        f"{pretty_label(u)} {b.replace('_',' ')} {m} | "
        f"{info['contrast_label']}  (p_obs = {panel_row['p_value']:.4f})",
        fontsize=12, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    png = od / f"stability_{m}_{u}_{b}_diagnostics.png"
    fig.savefig(png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return png


# ── JN figure ──────────────────────────────────────────────────────────────

def plot_jn(analysis, measure, unit, band, model_tag):
    if not ANALYSES[analysis]["is_interaction"]:
        return None
    od = out_dir(analysis)
    csv = od / f"jn_{measure}_{unit}_{band}_{model_tag}.csv"
    summary_csv = od / "jn_summary.csv"
    if not csv.exists():
        return None
    grid = pd.read_csv(csv)
    summary = pd.read_csv(summary_csv) if summary_csv.exists() else pd.DataFrame()
    sr = summary[(summary["measure"] == measure) &
                 (summary["unit"] == unit) &
                 (summary["band"] == band) &
                 (summary["model"] == model_tag)] if not summary.empty else pd.DataFrame()

    fig, ax = plt.subplots(figsize=(10, 6))
    sig = grid["significant"].astype(bool)
    ax.fill_between(grid["band_c"],
                    grid["simple_slope"] - 1.96 * grid["se"],
                    grid["simple_slope"] + 1.96 * grid["se"],
                    color="0.85", alpha=0.6, lw=0, label="95% CI")
    ax.plot(grid["band_c"], grid["simple_slope"],
            color="#2b2b2b", lw=2.0, label="Simple slope of Stim - No-stim")
    if sig.any():
        ax.plot(grid["band_c"][sig], grid["simple_slope"][sig],
                color="#c0392b", lw=3.0,
                label=f"Significant (p<.05): {int(sig.sum())} of {len(sig)} grid pts")
    ax.axhline(0, ls=":", color="0.4", lw=1)

    if not sr.empty:
        for col, c in (("jn_root_lo", "#1f77b4"), ("jn_root_hi", "#ff7f0e")):
            v = sr[col].iloc[0]
            if pd.notna(v):
                ax.axvline(v, ls="--", color=c, lw=1.4,
                           label=f"JN root: band_c = {v:.4g}")
    model_lbl = ("Random intercept only" if model_tag == "ri"
                 else "+ random StimCond slope")
    ax.set_xlabel(f"band_c value ({band.replace('_', ' ')} {measure})",
                  fontsize=11, fontweight="bold")
    ax.set_ylabel("Logit-scale simple slope (Stim - No-stim)",
                  fontsize=11, fontweight="bold")
    ax.set_title(
        f"Johnson-Neyman | {pretty_label(unit)} {band.replace('_',' ')} {measure} | "
        f"{model_lbl}",
        fontsize=12, fontweight="bold",
    )
    ax.legend(loc="best", fontsize=9, framealpha=0.92)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    png = od / f"jn_{measure}_{unit}_{band}_{model_tag}.png"
    fig.savefig(png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return png


# ── Random-slope comparison summary figure ─────────────────────────────────

def plot_random_slope_summary(analysis):
    od = out_dir(analysis)
    csv = od / "random_slope_comparison.csv"
    if not csv.exists():
        return None
    df = pd.read_csv(csv)
    if df.empty:
        return None

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # 1. Delta AIC distribution
    ax = axes[0]
    dAIC = df["dAIC"].dropna()
    if len(dAIC):
        ax.hist(dAIC, bins=20, color="#2b2b2b", alpha=0.75, edgecolor="white")
        ax.axvline(0, color="#c0392b", lw=1.5, ls="--",
                   label="Delta AIC = 0 (no improvement)")
        ax.set_xlabel("Delta AIC (RS - RI)", fontsize=10, fontweight="bold")
        ax.set_ylabel("count", fontsize=10, fontweight="bold")
        n_better = int((dAIC < 0).sum()); n_total = len(dAIC)
        ax.set_title(
            f"Random slope vs RI: AIC delta\n{n_better}/{n_total} panels favor RS",
            fontsize=11, fontweight="bold",
        )
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.25)

    # 2. LRT p-value
    ax = axes[1]
    lrt = df["LRT_p"].dropna()
    if len(lrt):
        ax.hist(lrt, bins=np.linspace(0, 1, 41),
                color="#a0a0a0", edgecolor="white")
        ax.axvline(0.05, color="#c0392b", ls="--", lw=1.5, label="p = .05")
        n_sig = int((lrt < 0.05).sum())
        ax.set_title(
            f"LRT (RS vs RI): {n_sig}/{len(lrt)} panels p < .05",
            fontsize=11, fontweight="bold",
        )
    ax.set_xlabel("LRT p-value", fontsize=10, fontweight="bold")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.25)

    # 3. Focal-term p comparison RI vs RS
    ax = axes[2]
    paired = df.dropna(subset=["ri_focal_p", "rs_focal_p"])
    if len(paired):
        ax.scatter(paired["ri_focal_p"], paired["rs_focal_p"],
                   s=18, color="#2b2b2b", alpha=0.7)
        lim = max(paired["ri_focal_p"].max(), paired["rs_focal_p"].max(), 1.0)
        ax.plot([0, lim], [0, lim], color="0.5", lw=1, ls=":")
        ax.axvline(0.05, color="#c0392b", ls="--", lw=1.0)
        ax.axhline(0.05, color="#c0392b", ls="--", lw=1.0)
    ax.set_xlabel("Focal p, RI", fontsize=10, fontweight="bold")
    ax.set_ylabel("Focal p, RS", fontsize=10, fontweight="bold")
    ax.set_title("Focal-term p: RI vs RS", fontsize=11, fontweight="bold")
    ax.grid(alpha=0.25)

    fig.suptitle(f"Random-slope variant comparison: {analysis}",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    png = od / "random_slope_comparison.png"
    fig.savefig(png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return png


def main():
    # NOTE: Johnson-Neyman PNGs are written directly by
    # posthoc_johnson_neyman_v2.R using interactions::sim_slopes() (cleaner
    # output, proper cyan/pink significance regions). We do NOT regenerate
    # them here -- doing so would overwrite the sim_slopes ggplot with a
    # rougher matplotlib version. Stability + RS-comparison only below.
    for analysis, info in ANALYSES.items():
        plot_random_slope_summary(analysis)

        trending_csv = out_dir(analysis) / "trending_panels.csv"
        if not trending_csv.exists():
            continue
        trending = pd.read_csv(trending_csv)

        for _, r in trending.iterrows():
            plot_stability(analysis, r)

        print(f"Plotted: {analysis} ({len(trending)} trending panels)")


if __name__ == "__main__":
    main()
