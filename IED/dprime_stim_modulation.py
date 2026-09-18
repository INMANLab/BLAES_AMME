"""Within-subjects stim vs no-stim memory modulation (d-prime) for the IED cohorts.

Behavioral source: AMMEBLAES_includedpts_firstsession_behavioral.csv
  nostim                = d' for no-stim items
  avg_stim              = d' for stim items
  avg_stim_dprime_diff  = d' difference (stim - no-stim)

Cohorts (matched to behavioral subjects by name):
  Encoding  = patients with encoding IEDs (33; 32 matched, BJH032 not in behavioral file)
  Retrieval = patients with retrieval IEDs on targets (18; 17 matched)

Per cohort:
  - connected dot plot of d' (no-stim -> stim), with swarm
  - swarm of the d' difference score (mean +/- 1 SE)
  - paired t-test (stim vs no-stim) and one-sample t-test (difference vs 0)

Outputs -> OUTPUTS/updated_IED_figures/
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

plt.rcParams["figure.dpi"] = 140
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.family"] = "Helvetica"

# Match behavioral_sex_connected_dotplots.py style
NOSTIM_C = "blue"      # no-stim points
STIM_C = "red"         # stim points
DIFF_C = "0.40"        # neutral grey for the difference swarm

AMME_BLAES = Path(__file__).resolve().parent.parent
BEHAV = AMME_BLAES / "behavioral" / "behavioral figures" / "AMMEBLAES_includedpts_firstsession_behavioral.csv"
IED_ENC = AMME_BLAES / "IED" / "AMMEBLAES_IEDs_trial_level_dissertation_study_usethis_cleaned_with_memory.csv"
IED_RET = AMME_BLAES / "IED" / "AMMEBLAES_IEDs_trial_level_dissertation_test_usethis_cleaned_with_memory.csv"
OUT_DIR = AMME_BLAES / "OUTPUTS" / "updated_IED_figures"

rng = np.random.default_rng(0)


def cohort_patients():
    e = pd.read_csv(IED_ENC)
    r = pd.read_csv(IED_RET)
    enc = sorted(e["Patient"].astype(str).unique())
    ret = sorted(r[r["MemoryOutcome"].isin(["remembered", "forgotten"])]["Patient"].astype(str).unique())
    return enc, ret


def stats_block(df):
    nostim = df["nostim"].to_numpy(float)
    stim = df["avg_stim"].to_numpy(float)
    diff = stim - nostim                              # primary: stim - no-stim
    diff_col = df["avg_stim_dprime_diff"].to_numpy(float)  # provided difference column
    n = len(df)

    # primary within-subjects test (identical to one-sample t on diff = stim - no-stim)
    t_paired, p_paired = stats.ttest_rel(stim, nostim)
    try:
        w_stat, w_p = stats.wilcoxon(stim, nostim)
    except ValueError:
        w_stat, w_p = np.nan, np.nan

    md, sd = diff.mean(), diff.std(ddof=1)
    se = sd / np.sqrt(n)
    tcrit = stats.t.ppf(0.975, n - 1)
    ci = (md - tcrit * se, md + tcrit * se)
    dz = md / sd if sd else np.nan

    # secondary: the provided avg_stim_dprime_diff column, tested vs 0
    t_col, p_col = stats.ttest_1samp(diff_col, 0.0)
    md_col, se_col = diff_col.mean(), diff_col.std(ddof=1) / np.sqrt(n)

    # patients where the provided column diverges from stim - no-stim
    mism = df.loc[np.abs(diff_col - diff) > 0.01, "Patient"].tolist()

    return {
        "n": n, "mean_nostim": nostim.mean(), "sd_nostim": nostim.std(ddof=1),
        "mean_stim": stim.mean(), "sd_stim": stim.std(ddof=1),
        "mean_diff": md, "sd_diff": sd, "se_diff": se, "ci": ci,
        "t_paired": t_paired, "df": n - 1, "p_paired": p_paired, "dz": dz,
        "w_stat": w_stat, "w_p": w_p,
        "mean_diff_col": md_col, "se_diff_col": se_col, "t_col": t_col, "p_col": p_col,
        "mismatch": mism,
        "nostim": nostim, "stim": stim, "diff": diff,
    }


def fmt_p(p):
    return "p < .001" if p < 0.001 else f"p = {p:.3f}"


def write_stats(blocks, missing):
    lines = []
    lines.append("Within-subjects stim vs no-stim memory modulation (d-prime)")
    lines.append("Source: AMMEBLAES_includedpts_firstsession_behavioral.csv")
    lines.append("=" * 70)
    for label, s, miss in blocks:
        lines.append(f"\n### {label}  (n = {s['n']}; not in behavioral file: {miss or 'none'})")
        lines.append(f"  d' no-stim : M = {s['mean_nostim']:.3f}, SD = {s['sd_nostim']:.3f}")
        lines.append(f"  d' stim    : M = {s['mean_stim']:.3f}, SD = {s['sd_stim']:.3f}")
        lines.append(f"  d' diff (stim - no-stim): M = {s['mean_diff']:.3f}, SD = {s['sd_diff']:.3f}, "
                     f"SE = {s['se_diff']:.3f}, 95% CI [{s['ci'][0]:.3f}, {s['ci'][1]:.3f}]")
        lines.append(f"  PRIMARY -- Paired t-test (stim vs no-stim): t({s['df']}) = {s['t_paired']:.3f}, "
                     f"{fmt_p(s['p_paired'])}  |  Cohen's dz = {s['dz']:.3f}")
        lines.append(f"            Wilcoxon signed-rank: W = {s['w_stat']:.1f}, {fmt_p(s['w_p'])}")
        lines.append(f"  SECONDARY -- provided 'avg_stim_dprime_diff' column vs 0: "
                     f"M = {s['mean_diff_col']:.3f}, SE = {s['se_diff_col']:.3f}, "
                     f"t({s['df']}) = {s['t_col']:.3f}, {fmt_p(s['p_col'])}")
        if s["mismatch"]:
            lines.append(f"  NOTE: avg_stim_dprime_diff != (avg_stim - nostim) for: {s['mismatch']} "
                         f"-- this is why the two tests differ.")
    txt = "\n".join(lines)
    (OUT_DIR / "dprime_stim_modulation_stats.txt").write_text(txt + "\n")
    print(txt)


def draw_connected(ax, s):
    x0, x1 = 0.0, 1.0
    # individual connecting lines (no jitter -> dots align exactly with line ends)
    for a, b in zip(s["nostim"], s["stim"]):
        ax.plot([x0, x1], [a, b], color="black", linewidth=1.25, alpha=0.45, zorder=1)
    ax.scatter([x0] * s["n"], s["nostim"], s=72, color=NOSTIM_C, zorder=3)
    ax.scatter([x1] * s["n"], s["stim"], s=72, color=STIM_C, zorder=3)
    ax.set_xticks([x0, x1])
    ax.set_xticklabels(["No-stim", "Stim"], fontsize=13, fontweight="bold")
    ax.set_xlim(-0.3, 1.3)
    ax.set_ylabel("d'", fontsize=13)
    # one line of stats above the connected dot plot
    ax.set_title(f"t({s['df']}) = {s['t_paired']:.2f}, {fmt_p(s['p_paired'])}, "
                 f"Cohen's d = {s['dz']:.2f}", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def draw_diff_swarm(ax, s):
    jx = 0.10
    xs = rng.uniform(-jx, jx, s["n"])
    ax.axhline(0, color="0.6", linestyle="--", linewidth=1.0, zorder=1)
    ax.scatter(xs, s["diff"], s=48, color=DIFF_C, edgecolor="black", linewidth=0.4,
               alpha=0.85, zorder=3)
    ax.set_xticks([0])
    ax.set_xticklabels(["Δd' (stim − no-stim)"], fontsize=12)
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylabel("Δd'", fontsize=13)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    b = pd.read_csv(BEHAV)
    b["Patient"] = b["Patient"].astype(str)
    enc_pts, ret_pts = cohort_patients()

    cohorts = []
    for label, pts in [("Encoding", enc_pts), ("Retrieval", ret_pts)]:
        sub = b[b["Patient"].isin(pts)].copy()
        miss = [p for p in pts if p not in set(b["Patient"])]
        cohorts.append((label, stats_block(sub), miss))

    write_stats(cohorts, None)

    fig = plt.figure(figsize=(10.5, 9.8))
    subfigs = fig.subfigures(2, 1, hspace=0.10)
    for sf, (label, s, _) in zip(subfigs, cohorts):
        sf.suptitle(label, fontsize=19, fontweight="bold")  # cohort label, centered, large bold
        axes = sf.subplots(1, 2)
        draw_connected(axes[0], s)
        draw_diff_swarm(axes[1], s)
    png = OUT_DIR / "dprime_stim_modulation.png"
    fig.savefig(png, bbox_inches="tight")
    print(f"\nSaved figure: {png}")
    print(f"Saved stats:  {OUT_DIR / 'dprime_stim_modulation_stats.txt'}")


if __name__ == "__main__":
    main()
