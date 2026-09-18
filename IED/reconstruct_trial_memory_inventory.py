"""Reconstruct the full trial-level memory inventory (all trials, IED + no-IED)
from the raw LFP Results_CSVOutput files, for the IED-cohort patients.

Memory-labeling rules mirror encoding_code/combined_encoding_power.py:
  ENCODING (phase1):
    - AMME (file has 'number','test_yes_or_no'): subsequent memory = test_yes_or_no
      (yes -> remembered, no -> forgotten). All studied items are "old".
    - BLAES (file has 'trialIdx','imagename','stimulus_code'): subsequent memory
      joined from the phase3 test response for the same image
      (recognized old -> remembered, called new -> forgotten).
  RETRIEVAL (phase3):
    - AMME: trial_type in {stim,nostim} are OLD items; yes_or_no (yes->remembered,
      no->forgotten). trial_type == 'new' are lures.
    - BLAES: trial_type == 'old' are OLD items; response old->remembered,
      response new->forgotten. trial_type == 'new' are lures.

Then each trial is flagged IED-positive (present in the IED detection file for that
patient+trial) vs no-IED, and remembered:forgotten ratios are compared.

Diagnostic targets: encoding 4,149 trials / 33 patients; retrieval 1,880 / 18 patients.
"""

from pathlib import Path
import glob
import numpy as np
import pandas as pd

AMME_BLAES = Path(__file__).resolve().parent.parent
RAW = Path("/Users/martinahollearn/Library/CloudStorage/Box-Box/InmanLab/"
           "BLAES_data/dissertation/LFP_analyses/Results_CSVOutput")
IED_ENC = AMME_BLAES / "IED" / "AMMEBLAES_IEDs_trial_level_dissertation_study_usethis_cleaned_with_memory.csv"
IED_RET = AMME_BLAES / "IED" / "AMMEBLAES_IEDs_trial_level_dissertation_test_usethis_cleaned_with_memory.csv"

OLD_RESP = {"78", "66", "37", "old", "yes"}   # recognized as old
NEW_RESP = {"67", "86", "39", "new", "no"}    # called new


def nonfreq_cols(f):
    h = pd.read_csv(f, nrows=0).columns
    return [c for c in h if not c.startswith(("pre_Freq_", "post_Freq_", "diff_Freq_"))]


def norm_resp(x):
    if pd.isna(x):
        return None
    s = str(x).strip().lower()
    try:
        s = str(int(float(s)))
    except Exception:
        pass
    if s in OLD_RESP:
        return "old"
    if s in NEW_RESP:
        return "new"
    return None


def load_phase(phase, patients):
    """Return per-trial DataFrame: patient, trial, memory_str (remembered/forgotten/None),
    is_old (bool). Collapsed to unique trials per patient."""
    files = [f for f in glob.glob(str(RAW / f"*{phase}*MeasurePower.csv")) if "AllPatients" not in f]
    out = []
    for f in files:
        cols = nonfreq_cols(f)
        df = pd.read_csv(f, usecols=cols)
        pat = df["Patient"].iloc[0]
        if pat not in patients:
            continue
        is_amme = "number" in df.columns and "trialIdx" not in df.columns
        tid = "number" if is_amme else "trialIdx"
        u = df.drop_duplicates(subset=[tid]).copy()
        rec = pd.DataFrame({"patient": pat, "trial": u[tid].astype(int)})

        if phase == "phase1":  # ENCODING -- all studied items are old
            rec["is_old"] = True
            if is_amme:
                yn = u["test_yes_or_no"].astype(str).str.strip().str.lower()
                rec["memory_str"] = np.where(yn == "yes", "remembered",
                                     np.where(yn == "no", "forgotten", None))
            else:  # BLAES: memory comes from phase3 join, filled later
                rec["memory_str"] = None
                rec["stimulus_code"] = pd.to_numeric(u["stimulus_code"], errors="coerce").values
                rec["imagename"] = u["imagename"].values
        else:  # phase3 RETRIEVAL
            if is_amme:
                tt = u["trial_type"].astype(str).str.strip().str.lower()
                rec["is_old"] = tt.isin(["stim", "nostim"]).values
                yn = u["yes_or_no"].astype(str).str.strip().str.lower()
                rec["memory_str"] = np.where(yn == "yes", "remembered",
                                     np.where(yn == "no", "forgotten", None))
            else:  # BLAES
                tt = u["trial_type"].astype(str).str.strip().str.lower()
                rec["is_old"] = (tt == "old").values
                resp = u["response"].apply(norm_resp)
                rec["memory_str"] = np.where(resp == "old", "remembered",
                                     np.where(resp == "new", "forgotten", None))
        rec["cohort"] = "AMME" if is_amme else "BLAES"
        out.append(rec)
    return pd.concat(out, ignore_index=True)


def fill_blaes_encoding_memory(enc):
    """For BLAES encoding trials, join phase3 test response by image to get
    subsequent memory."""
    files = [f for f in glob.glob(str(RAW / "*phase3*MeasurePower.csv")) if "AllPatients" not in f]
    parts = []
    for f in files:
        cols = nonfreq_cols(f)
        df = pd.read_csv(f, usecols=cols)
        if "trialIdx" not in df.columns or "response" not in df.columns:
            continue  # AMME or non-matching
        if not {"stimulus_code", "full_im_name", "trial_type"} & set(df.columns):
            pass
        img_col = next((c for c in ["full_im_name", "full_img_name", "imagename"] if c in df.columns), None)
        if img_col is None or "stimulus_code" not in df.columns:
            continue
        sub = df.drop_duplicates(subset=["stimulus_code", img_col]).copy()
        sub["resp"] = sub["response"].apply(norm_resp)
        tt = sub["trial_type"].astype(str).str.strip().str.lower()
        sub = sub[(tt == "old").values]  # only old items have a remembered/forgotten label
        sub["memory_str"] = np.where(sub["resp"] == "old", "remembered",
                             np.where(sub["resp"] == "new", "forgotten", None))
        sub["patient"] = df["Patient"].iloc[0]
        sub["stimulus_code"] = pd.to_numeric(sub["stimulus_code"], errors="coerce")
        parts.append(sub[["patient", "stimulus_code", "memory_str"]].rename(
            columns={"memory_str": "mem_lookup"}))
    lookup = pd.concat(parts, ignore_index=True).dropna(subset=["mem_lookup"])
    lookup = lookup.drop_duplicates(subset=["patient", "stimulus_code"])

    mask = enc["cohort"] == "BLAES"
    merged = enc[mask].merge(lookup, on=["patient", "stimulus_code"], how="left")
    enc.loc[mask, "memory_str"] = merged["mem_lookup"].values
    return enc


def summarize(df, label, old_only=True):
    use = df[df["is_old"]] if old_only else df
    valid = use[use["memory_str"].isin(["remembered", "forgotten"])]
    print(f"\n=== {label} ===")
    print(f"  patients: {df['patient'].nunique()}")
    print(f"  all unique trials: {len(df)}  | old-item trials: {int(df['is_old'].sum())}")
    print(f"  trials with rem/forg label: {len(valid)}")
    return valid


def ied_flag(valid, ied_csv):
    ied = pd.read_csv(ied_csv)
    ied_trials = ied[ied["MemoryOutcome"].isin(["remembered", "forgotten"])]
    keys = set(zip(ied_trials["Patient"], ied_trials["Trial"].astype(int)))
    valid = valid.copy()
    valid["ied"] = [(p, t) in keys for p, t in zip(valid["patient"], valid["trial"])]
    return valid


def ratio_table(valid, label):
    print(f"\n--- {label}: remembered vs forgotten by IED status ---")
    for grp, name in [(True, "IED trials"), (False, "no-IED trials")]:
        sub = valid[valid["ied"] == grp]
        n_rem = (sub["memory_str"] == "remembered").sum()
        n_forg = (sub["memory_str"] == "forgotten").sum()
        n = len(sub)
        ratio = (n_rem / n_forg) if n_forg else float("nan")
        prop = (n_rem / n) if n else float("nan")
        print(f"  {name:14s}: n={n:5d} | rem={n_rem:5d} forg={n_forg:5d} "
              f"| rem:forg={ratio:.2f} | %rem={prop:.1%}")


def main():
    enc_ied = pd.read_csv(IED_ENC)
    ret_ied = pd.read_csv(IED_RET)
    enc_pts = set(enc_ied["Patient"].unique())
    ret_pts = set(ret_ied[ret_ied["MemoryOutcome"].isin(["remembered", "forgotten"])]["Patient"].unique())

    enc = load_phase("phase1", enc_pts)
    enc = fill_blaes_encoding_memory(enc)
    ret = load_phase("phase3", ret_pts)

    enc_valid = summarize(enc, "ENCODING (target 4,149 all / 33 pts)", old_only=False)
    ret_all = summarize(ret, "RETRIEVAL (target 1,880 / 18 pts)", old_only=False)

    enc_valid = ied_flag(enc_valid, IED_ENC)
    ret_valid = ied_flag(ret[ret["is_old"] & ret["memory_str"].isin(["remembered", "forgotten"])], IED_RET)

    ratio_table(enc_valid, "ENCODING")
    ratio_table(ret_valid, "RETRIEVAL (old items)")


if __name__ == "__main__":
    main()
