#!/usr/bin/env python
"""Raw encoding PAC across the full frequency range for stim-only trials,
contrasting remembered vs forgotten, for the EC-PRC pair.

Mirrors plot_stim_memory_effect_hpc_prc.py but on the encoding PAC CSV.
EC_PRC is a raw pair in the input (no ALLHPC averaging needed).

Output:
  OUTPUTS/encoding_memory_reports/stim_effect/
    EC_PRC_encoding_pac_stim_remembered_vs_forgotten.png
    EC_PRC_encoding_pac_stim_remembered_vs_forgotten.csv
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
CSV_DIR = REPO_ROOT / "OUTPUTS" / "csvs"
OUT_DIR = REPO_ROOT / "OUTPUTS" / "encoding_memory_reports" / "stim_effect"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PAIR = "EC_PRC"


def main():
    csv_path = CSV_DIR / "combined_encoding_pac_all_mlmr_input.csv"
    df = pd.read_csv(csv_path)
    df = df[(df["yes_or_no"].isin(["yes", "no"])) &
            (df["trial_type"].isin(["nostim", "stim"]))]
    df.loc[df["Patient"] == "BJH033", "Patient"] = "BJH032"

    freq_cols = sorted([c for c in df.columns if c.startswith("diff_Freq_")],
                       key=lambda c: float(c.replace("diff_Freq_", "")))
    freqs = np.array([float(c.replace("diff_Freq_", "")) for c in freq_cols])

    pair = df[df["Region"] == PAIR].copy()
    if pair.empty:
        raise RuntimeError(f"No rows found for Region == {PAIR}.")

    stim = pair[pair["trial_type"] == "stim"].copy()
    if stim.empty:
        raise RuntimeError(f"No stim trials for {PAIR}.")

    subj_means = (stim.groupby(["Patient", "yes_or_no"], as_index=False)
                       [freq_cols].mean(numeric_only=True))
    remembered = subj_means[subj_means["yes_or_no"] == "yes"]
    forgotten  = subj_means[subj_means["yes_or_no"] == "no"]

    def group_stats(d):
        vals = d[freq_cols].to_numpy(dtype=float)
        n = vals.shape[0]
        m = np.nanmean(vals, axis=0)
        sem = (np.nanstd(vals, axis=0, ddof=1) / np.sqrt(n)
               if n > 1 else np.zeros_like(m))
        return m, sem, n

    rem_mean, rem_sem, rem_n = group_stats(remembered)
    forg_mean, forg_sem, forg_n = group_stats(forgotten)

    rem_subjects = set(remembered["Patient"])
    forg_subjects = set(forgotten["Patient"])
    both_n = len(rem_subjects & forg_subjects)

    n_stim_trials = len(stim)
    n_rem_trials = int((stim["yes_or_no"] == "yes").sum())
    n_forg_trials = int((stim["yes_or_no"] == "no").sum())

    rem_color = "#2b2b2b"
    forg_color = "#a0a0a0"

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.fill_between(freqs, forg_mean - forg_sem, forg_mean + forg_sem,
                    color=forg_color, alpha=0.30, lw=0)
    ax.plot(freqs, forg_mean, color=forg_color, lw=2.2,
            label=f"Stim forgotten  (N = {forg_n} subj, {n_forg_trials} trials)")

    ax.fill_between(freqs, rem_mean - rem_sem, rem_mean + rem_sem,
                    color=rem_color, alpha=0.25, lw=0)
    ax.plot(freqs, rem_mean, color=rem_color, lw=2.2,
            label=f"Stim remembered  (N = {rem_n} subj, {n_rem_trials} trials)")

    ax.axhline(0, color="0.5", ls=":", lw=0.8)
    ax.set_xlim(freqs.min(), freqs.max())
    ax.set_xlabel("Frequency (Hz)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Encoding PAC (stim - baseline)",
                  fontsize=12, fontweight="bold")
    ax.set_title(
        "EC-PRC encoding PAC: stim remembered vs forgotten\n"
        f"Stim trials only | {both_n} subjects contribute to both conditions | "
        f"{n_stim_trials} stim trials total",
        fontsize=12, fontweight="bold",
    )
    ax.legend(loc="upper right", framealpha=0.9, fontsize=11)
    ax.grid(alpha=0.25)

    fig.tight_layout()
    out_png = OUT_DIR / "EC_PRC_encoding_pac_stim_remembered_vs_forgotten.png"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_png}")

    summary = pd.DataFrame({
        "freq_hz": freqs,
        "remembered_mean": rem_mean,
        "remembered_sem": rem_sem,
        "forgotten_mean": forg_mean,
        "forgotten_sem": forg_sem,
    })
    out_csv = OUT_DIR / "EC_PRC_encoding_pac_stim_remembered_vs_forgotten.csv"
    summary.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()
