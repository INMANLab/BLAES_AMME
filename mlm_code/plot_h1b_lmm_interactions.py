#!/usr/bin/env python
"""
Plot the significant H1b LMM interactions:
  - Continuous figure: one subplot per panel showing predicted feature vs.
    mem_mod_z, two lines (no-stim, stim), 95% CI shading + per-subject dots.
  - Categorical (quad) figure: one subplot per panel showing predicted
    feature for each responder group x stim condition with 95% CI bars.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "outputs" / "h1b_subregions_quad_lmm_imbalanced_retrieval_mlm"
OUT_DIR = DATA_DIR

# Panels with significant Stim x mem_mod_z interaction (continuous moderator).
CONT_PANELS = [
    ("coherence", "BLA_EC", "slow_gamma", "BLA-EC slow gamma coherence",  0.016, "*"),
    ("coherence", "BLA_CA", "slow_gamma", "BLA-CA slow gamma coherence",  0.030, "*"),
    ("coherence", "BLA_CA", "theta",      "BLA-CA theta coherence",        0.030, "*"),
    ("coherence", "EC_HPC", "theta",      "EC-HPC theta coherence",        0.040, "*"),
    ("coherence", "CA_PRC", "theta",      "CA-PRC theta coherence",        0.054, "+"),
]

# Panels with at least one significant Stim x QuadResponderGroup contrast.
CAT_PANELS = [
    ("pac",       "BLA_PRC", "slow_gamma", "BLA-PRC slow gamma PAC",
        ["AntiResp", "Moderate", "Strong"]),
    ("coherence", "BLA_EC",  "slow_gamma", "BLA-EC slow gamma coherence",
        ["AntiResp"]),
    ("coherence", "HPC_PRC", "slow_gamma", "HPC-PRC slow gamma coherence",
        ["AntiResp"]),
    ("coherence", "BLA_CA",  "theta",      "BLA-CA theta coherence",
        ["AntiResp"]),
    ("power",     "DG",      "theta",      "DG theta power",
        ["Strong"]),
]

COND_COLOR = {"nostim": "#1f77b4", "stim": "#d62728"}
COND_LABEL = {"nostim": "No stim", "stim": "Stim"}
GROUP_ORDER = ["AntiResp", "NonResp", "Moderate", "Strong"]
GROUP_LABEL = {"AntiResp": "Anti", "NonResp": "Non", "Moderate": "Moderate", "Strong": "Strong"}
GROUP_X = {g: i for i, g in enumerate(GROUP_ORDER)}


def panel_tag(modality, unit, band):
    return f"{modality}_{unit}_{band}"


def load_panel(modality, unit, band):
    tag = panel_tag(modality, unit, band)
    pred_cont = DATA_DIR / f"plot_cont_pred_{tag}.csv"
    pred_cat = DATA_DIR / f"plot_cat_pred_{tag}.csv"
    subj = DATA_DIR / f"plot_subjects_{tag}.csv"
    return (
        pd.read_csv(pred_cont) if pred_cont.exists() else None,
        pd.read_csv(pred_cat) if pred_cat.exists() else None,
        pd.read_csv(subj) if subj.exists() else None,
    )


# ── Continuous figure ─────────────────────────────────────────────────────

def plot_continuous_panel(ax, modality, unit, band, label, sig_p, sig_marker):
    pred_cont, _, subj = load_panel(modality, unit, band)
    if pred_cont is None or subj is None:
        ax.set_visible(False); return

    for cond in ["nostim", "stim"]:
        sub = pred_cont[pred_cont["StimCond"] == cond]
        ax.fill_between(sub["mem_mod_z"], sub["lo"], sub["hi"],
                        color=COND_COLOR[cond], alpha=0.18)
        ax.plot(sub["mem_mod_z"], sub["yhat"],
                color=COND_COLOR[cond], lw=2.0, label=COND_LABEL[cond])

    for cond in ["nostim", "stim"]:
        s = subj[subj["StimCond"] == cond]
        ax.scatter(s["mem_mod_z"], s["feature"],
                   c=COND_COLOR[cond], alpha=0.5, s=18,
                   edgecolors="white", linewidths=0.4)

    ax.axhline(0, ls=":", color="0.5", lw=0.7)
    ax.axvline(0, ls=":", color="0.5", lw=0.7)
    n_subj = subj["Patient"].nunique()
    ax.set_title(
        f"{label}\n"
        f"(Stim x mem_mod_z p = {sig_p:.3f} {sig_marker}, N = {n_subj} subjects)",
        fontsize=10, fontweight="bold")
    ax.set_xlabel("mem_mod_z (subj memory modulation)", fontsize=9)
    ax.set_ylabel("Feature (band-avg)", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.2)


def _layout_2x2_plus1(fig_w_per_col=4.0, fig_h_per_row=3.6):
    """3-row x 4-col gridspec: top two rows hold a 2x2 grid (each cell spans
    2 columns); the 5th panel sits centered on the bottom row."""
    fig = plt.figure(figsize=(2 * fig_w_per_col, 3 * fig_h_per_row))
    gs = fig.add_gridspec(3, 4, hspace=0.55, wspace=0.35)
    axes = [
        fig.add_subplot(gs[0, 0:2]),
        fig.add_subplot(gs[0, 2:4]),
        fig.add_subplot(gs[1, 0:2]),
        fig.add_subplot(gs[1, 2:4]),
        fig.add_subplot(gs[2, 1:3]),
    ]
    return fig, axes


def fig_continuous():
    fig, axes = _layout_2x2_plus1(fig_w_per_col=4.5, fig_h_per_row=3.8)
    panels = CONT_PANELS[:5]
    for ax, panel in zip(axes, panels):
        plot_continuous_panel(ax, *panel)
    for j in range(len(panels), len(axes)):
        axes[j].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", fontsize=10, framealpha=0.9,
               bbox_to_anchor=(0.99, 0.99))
    fig.suptitle("H1b: Stim x continuous memory modulation interactions (LMM)\n"
                 "lines = LMM-predicted feature; shading = 95% CI; dots = per-subject means",
                 fontsize=12, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = OUT_DIR / "h1b_lmm_continuous_interactions.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ── Categorical figure ────────────────────────────────────────────────────

def plot_categorical_panel(ax, modality, unit, band, label, sig_groups):
    _, pred_cat, subj = load_panel(modality, unit, band)
    if pred_cat is None or subj is None:
        ax.set_visible(False); return

    for cond, dx in [("nostim", -0.18), ("stim", +0.18)]:
        rows = []
        for g in GROUP_ORDER:
            r = pred_cat[(pred_cat["Group"] == g) & (pred_cat["StimCond"] == cond)]
            if r.empty: continue
            rows.append((GROUP_X[g] + dx, r.iloc[0]["yhat"], r.iloc[0]["lo"], r.iloc[0]["hi"]))
        if not rows: continue
        xs = [r[0] for r in rows]; ys = [r[1] for r in rows]
        los = [r[2] for r in rows]; his = [r[3] for r in rows]
        yerr = [[y - lo for y, lo in zip(ys, los)],
                [hi - y for hi, y in zip(his, ys)]]
        ax.errorbar(xs, ys, yerr=yerr,
                    fmt="o", color=COND_COLOR[cond], ms=6, capsize=3,
                    lw=1.4, label=COND_LABEL[cond])

    # Per-subject dots, jittered within group bin and condition.
    rng = np.random.RandomState(0)
    for cond, dx in [("nostim", -0.30), ("stim", +0.30)]:
        s = subj[subj["StimCond"] == cond]
        for g in GROUP_ORDER:
            ss = s[s["QuadResponderGroup"] == g]
            if ss.empty: continue
            jitter = (rng.rand(len(ss)) - 0.5) * 0.18
            ax.scatter(GROUP_X[g] + dx + jitter, ss["feature"],
                       c=COND_COLOR[cond], alpha=0.30, s=14,
                       edgecolors="none")

    ax.set_xticks(list(GROUP_X.values()))
    xlabels = []
    for g in GROUP_ORDER:
        marks = []
        for tag in sig_groups:
            base = tag.split(" ")[0]
            if base == g:
                marks.append("*" if "(+)" not in tag else "+")
        n_g = subj.loc[subj["QuadResponderGroup"] == g, "Patient"].nunique()
        label_g = f"{GROUP_LABEL[g]}\nn={n_g}"
        if marks:
            label_g += "\n" + "".join(marks)
        xlabels.append(label_g)
    ax.set_xticklabels(xlabels, fontsize=9)
    n_subj = subj["Patient"].nunique()
    ax.set_title(f"{label}\n(N = {n_subj} subjects)",
                 fontsize=10, fontweight="bold")
    ax.set_xlabel("Responder group (vs Non = ref)", fontsize=9)
    ax.set_ylabel("Feature (band-avg)", fontsize=9)
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(alpha=0.2, axis="y")


def fig_categorical():
    fig, axes = _layout_2x2_plus1(fig_w_per_col=5.0, fig_h_per_row=3.8)
    panels = CAT_PANELS[:5]
    for ax, panel in zip(axes, panels):
        plot_categorical_panel(ax, *panel)
    for j in range(len(panels), len(axes)):
        axes[j].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", fontsize=10, framealpha=0.9,
               bbox_to_anchor=(0.99, 0.99))
    fig.suptitle("H1b: Stim x quad responder group interactions (LMM)\n"
                 "points = LMM-predicted group means; whiskers = 95% CI; "
                 "translucent dots = per-subject means; * = sig vs Non",
                 fontsize=12, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = OUT_DIR / "h1b_lmm_categorical_interactions.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def main():
    plt.rcParams.update({"font.family": "serif"})
    fig_continuous()
    fig_categorical()


if __name__ == "__main__":
    main()
