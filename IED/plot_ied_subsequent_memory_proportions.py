"""Subsequent-memory proportions (remembered vs forgotten) in patients with IEDs.

Cohorts:
  - Encoding: 802 trials, 33 patients (valid MemoryOutcome, collapsed to Patient x Trial)
  - Retrieval: 145 trials, 18 patients (old items only: remembered/forgotten)

For each cohort we compute the per-patient proportion of IED trials that were
remembered vs forgotten, then plot the across-patient distribution (swarm) with
the across-patient mean +/- SEM. Pooled trial-level proportions are annotated.

Output: outputs/IED_subsequent_memory/ied_subsequent_memory_proportions.{png,pdf}
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["figure.dpi"] = 140
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.family"] = "Helvetica"

# Canonical dissertation memory palette (run_all_analyses.py)
REM_COLOR = "#FFD43B"   # yellow
FORG_COLOR = "#9C7AC9"  # purple

BASE_DIR = Path(__file__).resolve().parent
ENC_CSV = BASE_DIR / "AMMEBLAES_IEDs_trial_level_dissertation_study_usethis_cleaned_with_memory.csv"
RET_CSV = BASE_DIR / "AMMEBLAES_IEDs_trial_level_dissertation_test_usethis_cleaned_with_memory.csv"
OUT_DIR = BASE_DIR.parent / "OUTPUTS" / "updated_IED_figures"


def trial_level(csv_path: Path) -> pd.DataFrame:
    """Collapse row-level IED detections to unique Patient x Trial with an
    old-item memory outcome (remembered/forgotten)."""
    df = pd.read_csv(csv_path)
    df = df[df["MemoryOutcome"].isin(["remembered", "forgotten"])].copy()
    # one row per trial; take the trial's outcome (constant within a trial)
    trials = (
        df.groupby(["Patient", "Trial"])["MemoryOutcome"].first().reset_index()
    )
    return trials


def per_patient_props(trials: pd.DataFrame) -> pd.DataFrame:
    """Per-patient proportion remembered / forgotten of IED trials."""
    g = trials.groupby("Patient")["MemoryOutcome"]
    out = pd.DataFrame({
        "n_trials": g.size(),
        "prop_remembered": g.apply(lambda s: (s == "remembered").mean()),
    })
    out["prop_forgotten"] = 1.0 - out["prop_remembered"]
    return out.reset_index()


def summarize(trials: pd.DataFrame, pp: pd.DataFrame, label: str) -> dict:
    n_rem = int((trials["MemoryOutcome"] == "remembered").sum())
    n_forg = int((trials["MemoryOutcome"] == "forgotten").sum())
    n_tot = len(trials)
    return {
        "label": label,
        "n_patients": trials["Patient"].nunique(),
        "n_trials": n_tot,
        "n_rem": n_rem,
        "n_forg": n_forg,
        "pooled_prop_rem": n_rem / n_tot,
        "pooled_prop_forg": n_forg / n_tot,
        "mean_prop_rem": pp["prop_remembered"].mean(),
        "sem_prop_rem": pp["prop_remembered"].sem(),
        "mean_prop_forg": pp["prop_forgotten"].mean(),
        "sem_prop_forg": pp["prop_forgotten"].sem(),
        "pp": pp,
    }


def jitter(n: int, width: float = 0.06) -> np.ndarray:
    # deterministic, symmetric spread (no RNG)
    if n == 1:
        return np.array([0.0])
    return np.linspace(-width, width, n)


def draw_cohort(ax, stats: dict, x_rem: float, x_forg: float):
    pp = stats["pp"]
    for x, key, color in [
        (x_rem, "prop_remembered", REM_COLOR),
        (x_forg, "prop_forgotten", FORG_COLOR),
    ]:
        # bar = across-patient mean proportion; error bar = +/- 1 SE of the mean
        mean = pp[key].mean()
        se = pp[key].sem()
        ax.bar(x, mean, width=0.55, color=color, edgecolor="black",
               linewidth=1.0, zorder=2)
        ax.errorbar(x, mean, yerr=se, fmt="none", ecolor="black",
                    elinewidth=1.3, capsize=4, zorder=4)
        ax.text(x, mean + se + 0.03, f"{mean:.0%}", ha="center", va="bottom", fontsize=10)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    enc_trials = trial_level(ENC_CSV)
    ret_trials = trial_level(RET_CSV)
    enc_pp = per_patient_props(enc_trials)
    ret_pp = per_patient_props(ret_trials)

    enc = summarize(enc_trials, enc_pp, "Encoding")
    ret = summarize(ret_trials, ret_pp, "Retrieval")

    for s in (enc, ret):
        print(f"{s['label']}: {s['n_trials']} trials, {s['n_patients']} patients "
              f"| remembered {s['n_rem']} ({s['pooled_prop_rem']:.1%}), "
              f"forgotten {s['n_forg']} ({s['pooled_prop_forg']:.1%}) "
              f"| per-patient prop. remembered "
              f"{s['mean_prop_rem']:.2f} +/- {s['sem_prop_rem']:.2f} SEM")

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 5.4), sharey=True)
    cohorts = [enc, ret]
    x_positions = [(0.0, 1.0), (0.0, 1.0)]

    for ax, s, (xr, xf) in zip(axes, cohorts, x_positions):
        draw_cohort(ax, s, xr, xf)
        ax.set_xticks([xr, xf])
        ax.set_xticklabels(["Remembered", "Forgotten"])
        ax.set_ylim(0, 1.0)
        ax.set_title(
            f"{s['label']}\n{s['n_trials']} IED target trials, {s['n_patients']} patients",
            fontsize=12, fontweight="bold",
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.axhline(0.5, color="0.7", linestyle="--", linewidth=0.8, zorder=0)

    axes[0].set_ylabel("Proportion of IED target trials", fontsize=12)
    fig.suptitle(
        "Subsequent-memory proportions of IED target trials (remembered vs forgotten)",
        fontsize=13, fontweight="bold", y=1.02,
    )
    fig.text(0.5, -0.04,
             "Bars = across-patient mean proportion; error bars = ±1 SE of the mean.",
             ha="center", fontsize=9, color="0.3")

    fig.tight_layout()
    png = OUT_DIR / "ied_subsequent_memory_proportions.png"
    fig.savefig(png, bbox_inches="tight")
    print(f"\nSaved:\n  {png}")


if __name__ == "__main__":
    main()
