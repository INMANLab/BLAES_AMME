#!/usr/bin/env python
"""
Cluster-based permutation test (Maris & Oostenveld, 2007) of stim vs no-stim
trials at ENCODING.

Same methodology as run_permutation_stim_nostim.py (which is for retrieval);
this version reads encoding inputs and writes to the encoding/ subfolder.

Usage:
  python run_permutation_encoding_stim_nostim.py
  python run_permutation_encoding_stim_nostim.py power
"""
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path("/Users/martinahollearn/Library/CloudStorage/Box-Box/InmanLab/"
            "BLAES_data/dissertation/AMME_BLAES")
INPUT_DIR = ROOT / "outputs" / "csvs"

_raw_args = sys.argv[1:]
NO_BLA = "--no-bla" in _raw_args
BALANCED = "--balanced" in _raw_args
_pos_args = [a for a in _raw_args if not a.startswith("--")]

_branch = ("encoding_balanced" if (BALANCED and not NO_BLA)
           else "encoding_balanced_no_bla" if (BALANCED and NO_BLA)
           else "encoding_no_bla" if NO_BLA
           else "encoding")
OUT_BASE = (ROOT / "outputs" / "PermutationOutputsAlireza"
            / _branch / "stim_nostim")
OUT_BASE.mkdir(parents=True, exist_ok=True)

N_PERMUTATIONS = 5000
SEED = 100
ALPHA = 0.05
MIN_SUBJECTS = 5
MIN_TRIALS_BALANCED = 10
EXCLUDE_TOKENS = ("PHG", "BLA") if NO_BLA else ("PHG",)
EXCLUDE_PAIRS = {
    frozenset({"CA", "DG"}),
    frozenset({"CA", "HPC"}),
    frozenset({"DG", "HPC"}),
}


def is_excluded_region(r):
    if any(tok in r for tok in EXCLUDE_TOKENS):
        return True
    parts = r.split("_")
    if len(parts) == 2 and frozenset(parts) in EXCLUDE_PAIRS:
        return True
    return False

MODALITY_TO_FILE = {
    "power":     "combined_encoding_power_amme_timing_filtered_mlmr_input.csv",
    "coherence": "combined_encoding_coherence_amme_timing_filtered_mlmr_input.csv",
    "pac":       "combined_encoding_pac_amme_timing_filtered_mlmr_input.csv",
}
MODALITY_YLABEL = {
    "power":     "Power (dB, baseline-subtracted)",
    "coherence": "Coherence (baseline-subtracted)",
    "pac":       "PAC (baseline-subtracted)",
}

modalities = _pos_args if _pos_args else ["power", "coherence", "pac"]


def _balanced_keep(df):
    counts = (df.groupby("Patient")
                .agg(n_rem=("yes_or_no", lambda s: (s == "yes").sum()),
                     n_forg=("yes_or_no", lambda s: (s == "no").sum()),
                     n_regions=("Region", lambda s: s.nunique()))
                .reset_index())
    counts["rem_per_reg"]  = (counts["n_rem"]  / counts["n_regions"]).round()
    counts["forg_per_reg"] = (counts["n_forg"] / counts["n_regions"]).round()
    return set(counts.loc[(counts["rem_per_reg"]  >= MIN_TRIALS_BALANCED) &
                          (counts["forg_per_reg"] >= MIN_TRIALS_BALANCED),
                          "Patient"].tolist())


def load_modality(modality):
    df = pd.read_csv(INPUT_DIR / MODALITY_TO_FILE[modality])
    df.loc[df["Patient"] == "BJH033", "Patient"] = "BJH032"
    df = df[df["yes_or_no"].isin(["yes", "no"])]
    df = df[df["trial_type"].notna()]
    if BALANCED:
        keep = _balanced_keep(df)
        before = df["Patient"].nunique()
        df = df[df["Patient"].isin(keep)]
        print(f"  [balanced] {modality}: kept "
              f"{df['Patient'].nunique()}/{before} patients")
    df["StimCond"] = np.where(df["trial_type"] == "nostim", "Nostim", "Stim")

    freq_cols = [c for c in df.columns if "_Freq_" in c]
    diff_cols = [c for c in freq_cols if c.startswith("diff_")]
    freqs = [float(c.split("_Freq_")[1]) for c in diff_cols]
    return df, diff_cols, np.array(freqs)


def per_subject_means(df, diff_cols):
    return df.groupby(["Patient", "Region", "StimCond"],
                      as_index=False)[diff_cols].mean()


def paired_arrays(grouped, region, freq_cols):
    sub = grouped[grouped["Region"] == region]
    pivoted = sub.pivot_table(index="Patient", columns="StimCond",
                              values=freq_cols, aggfunc="mean")
    if pivoted.empty:
        return None, None, None
    has_both = pivoted.dropna(how="any").index.tolist()
    if len(has_both) < MIN_SUBJECTS:
        return None, None, None
    pivoted = pivoted.loc[has_both]
    if "Stim" not in pivoted.columns.get_level_values(1) or \
       "Nostim" not in pivoted.columns.get_level_values(1):
        return None, None, None
    X_S = np.column_stack([pivoted[(c, "Stim")].values   for c in freq_cols])
    X_N = np.column_stack([pivoted[(c, "Nostim")].values for c in freq_cols])
    return X_S, X_N, has_both


def find_clusters(t_vals, t_thresh):
    sign = np.zeros_like(t_vals, dtype=int)
    sign[t_vals >  t_thresh] =  1
    sign[t_vals < -t_thresh] = -1
    clusters = []
    i, n = 0, len(t_vals)
    while i < n:
        if sign[i] != 0:
            j = i
            while j + 1 < n and sign[j + 1] == sign[i]:
                j += 1
            mass = float(np.sum(t_vals[i:j + 1]))
            clusters.append((i, j, int(sign[i]), mass))
            i = j + 1
        else:
            i += 1
    return clusters


def cluster_perm_paired(X_A, X_B, n_perm=N_PERMUTATIONS, alpha=ALPHA, seed=SEED):
    n_subj = X_A.shape[0]
    diff = X_A - X_B
    df = n_subj - 1
    t_thresh = float(stats.t.ppf(1 - alpha, df=df))

    mean = diff.mean(axis=0); sd = diff.std(axis=0, ddof=1)
    se = sd / np.sqrt(n_subj); se = np.where(se == 0, np.nan, se)
    t_obs = np.where(np.isnan(se), 0, mean / se)

    obs_clusters = find_clusters(t_obs, t_thresh)
    rng = np.random.default_rng(seed)
    null_max_mass = np.zeros(n_perm)
    for p in range(n_perm):
        signs = rng.choice([-1, 1], size=n_subj)
        flipped = diff * signs[:, None]
        m = flipped.mean(axis=0)
        s = flipped.std(axis=0, ddof=1)
        se_p = s / np.sqrt(n_subj)
        se_p = np.where(se_p == 0, np.nan, se_p)
        t_p = np.where(np.isnan(se_p), 0, m / se_p)
        cls = find_clusters(t_p, t_thresh)
        null_max_mass[p] = max((abs(c[3]) for c in cls), default=0.0)

    cluster_records = []
    for (i0, i1, sgn, mass) in obs_clusters:
        cluster_records.append({
            "start_idx": int(i0), "end_idx": int(i1),
            "n_freqs": int(i1 - i0 + 1),
            "direction": "Stim>Nostim" if sgn > 0 else "Stim<Nostim",
            "cluster_mass": float(mass),
            "p_value": float(np.mean(null_max_mass >= abs(mass))),
        })
    return {"t_obs": t_obs, "t_thresh": t_thresh, "n_subj": n_subj,
            "obs_clusters": cluster_records,
            "null_max_mass": null_max_mass,
            "X_S": X_A, "X_N": X_B}


def plot_region_panel(ax, freqs, X_S, X_N, clusters, region, ylabel):
    RED  = "#d62728"
    BLUE = "#1f77b4"
    DARK_YELLOW       = "#b8860b"
    DARK_YELLOW_LIGHT = "#daa520"

    mean_S = X_S.mean(axis=0); sem_S = X_S.std(axis=0, ddof=1) / np.sqrt(X_S.shape[0])
    mean_N = X_N.mean(axis=0); sem_N = X_N.std(axis=0, ddof=1) / np.sqrt(X_N.shape[0])

    ax.fill_between(freqs, mean_S - sem_S, mean_S + sem_S,
                    color=RED, alpha=0.18, lw=0, zorder=1)
    ax.fill_between(freqs, mean_N - sem_N, mean_N + sem_N,
                    color=BLUE, alpha=0.18, lw=0, zorder=1)

    sig_text_y_max = -np.inf
    for c in clusters:
        i0, i1 = c["start_idx"], c["end_idx"]
        sig = c["p_value"] < .05
        s = slice(i0, i1 + 1)
        fill_color = DARK_YELLOW if sig else DARK_YELLOW_LIGHT
        alpha_fill = 0.85 if sig else 0.30
        ax.fill_between(freqs[s], mean_S[s], mean_N[s],
                        color=fill_color, alpha=alpha_fill, lw=0,
                        zorder=2, interpolate=True)
        if sig:
            f_lo, f_hi = freqs[i0], freqs[i1]
            y_top = max(np.max(mean_S[s] + sem_S[s]),
                        np.max(mean_N[s] + sem_N[s]))
            sig_text_y_max = max(sig_text_y_max, y_top)
            ax.text((f_lo + f_hi) / 2, y_top,
                    f"p = {c['p_value']:.3f}",
                    ha="center", va="bottom", fontsize=8,
                    color="black", fontweight="bold", zorder=4)

    ax.plot(freqs, mean_S, color=RED, lw=1.6, label="Stim", zorder=3)
    ax.plot(freqs, mean_N, color=BLUE, lw=1.6, label="No stim", zorder=3)
    ax.axhline(0, color="0.7", lw=0.4, ls="--", zorder=0)

    ymin, ymax = ax.get_ylim()
    full_range = ymax - ymin
    if np.isfinite(sig_text_y_max):
        target_top = max(ymax, sig_text_y_max + 0.10 * full_range)
        ax.set_ylim(ymin, target_top + 0.08 * full_range)

    title = f"{region} (n={X_S.shape[0]})"
    if any(c["p_value"] < .05 for c in clusters):
        title += " *"
    ax.set_title(title, fontsize=10)
    ax.tick_params(labelsize=8)


def run_modality(modality):
    print(f"\n========== {modality} ==========")
    df, diff_cols, freqs = load_modality(modality)
    if df.empty:
        print("  No trials found."); return
    print(f"  {len(diff_cols)} freq bins, {df['Patient'].nunique()} patients in raw data")
    grouped = per_subject_means(df, diff_cols)

    out_dir = OUT_BASE / modality
    out_dir.mkdir(parents=True, exist_ok=True)

    regions = sorted(grouped["Region"].unique())
    excluded = [r for r in regions if is_excluded_region(r)]
    if excluded:
        print(f"  Excluding: {', '.join(excluded)}")
    regions = [r for r in regions if r not in excluded]

    panel_results = []; cluster_table = []
    for region in regions:
        X_S, X_N, _ = paired_arrays(grouped, region, diff_cols)
        if X_S is None:
            print(f"  - {region}: skipped (n < {MIN_SUBJECTS})")
            continue
        print(f"  - {region}: n={X_S.shape[0]} subjects, {len(diff_cols)} freqs ...", flush=True)
        res = cluster_perm_paired(X_S, X_N)
        for c in res["obs_clusters"]:
            cluster_table.append({
                "modality": modality, "region": region,
                "n_subjects": res["n_subj"],
                "freq_lo_hz": float(freqs[c["start_idx"]]),
                "freq_hi_hz": float(freqs[c["end_idx"]]),
                "n_freqs": c["n_freqs"], "direction": c["direction"],
                "cluster_mass_t": c["cluster_mass"],
                "t_threshold": res["t_thresh"],
                "p_value": c["p_value"],
                "significant": bool(c["p_value"] < .05),
            })
        panel_results.append({"region": region, "X_S": X_S, "X_N": X_N,
                              "clusters": res["obs_clusters"],
                              "null_max_mass": res["null_max_mass"]})

    pd.DataFrame(cluster_table).to_csv(
        out_dir / f"{modality}_cluster_stats.csv", index=False)
    print(f"  Saved cluster stats: {out_dir / f'{modality}_cluster_stats.csv'}")

    if not panel_results:
        return
    n = len(panel_results)
    ncols = min(4, max(2, int(np.ceil(np.sqrt(n)))))
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 4.0, nrows * 3.2),
                             squeeze=False)
    axes_flat = axes.flatten()
    for ax, panel in zip(axes_flat, panel_results):
        plot_region_panel(ax, freqs,
                          panel["X_S"], panel["X_N"],
                          panel["clusters"], panel["region"],
                          MODALITY_YLABEL[modality])
    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", fontsize=10,
               frameon=True, bbox_to_anchor=(0.99, 0.99))
    fig.suptitle(
        f"Encoding | Stim vs No-stim - {modality.upper()}\n"
        f"Encoding trials, paired cluster permutation "
        f"({N_PERMUTATIONS} samples)",
        fontsize=12, fontweight="bold")
    fig.supxlabel("Frequency (Hz)", fontsize=11, fontweight="bold")
    fig.supylabel(MODALITY_YLABEL[modality], fontsize=11, fontweight="bold")
    fig.tight_layout(rect=[0.02, 0.02, 1, 0.94])
    fig.savefig(out_dir / f"{modality}_stim_nostim.png",
                dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved figure: {out_dir / f'{modality}_stim_nostim.png'}")

    fig, ax = plt.subplots(figsize=(8, 5))
    all_max = np.concatenate([p["null_max_mass"] for p in panel_results])
    ax.hist(all_max, bins=60, color="0.7", edgecolor="0.4")
    ax.set_title(
        f"Null distribution of max |cluster mass| - {modality.upper()} (Encoding)\n"
        f"({N_PERMUTATIONS} permutations x {n} regions, pooled)", fontsize=11)
    ax.set_xlabel("Max |cluster mass|"); ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(out_dir / f"{modality}_null_distribution.png",
                dpi=160, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    for m in modalities:
        run_modality(m)
    print("\n===== ALL DONE =====")
