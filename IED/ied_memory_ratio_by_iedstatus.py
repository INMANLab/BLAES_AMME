"""Remembered:forgotten ratio in IED vs no-IED trials, encoding & retrieval.

Strategy (robust to the messy raw data):
  - IED old-item remembered/forgotten counts come straight from the IED detection
    files (authoritative): these are the "IED trials".
  - TOTAL old-item remembered/forgotten counts are reconstructed from the raw LFP
    MeasurePower files (all trials, collapsed across channels, old items only).
  - no-IED = TOTAL - IED.

Old item = studied/target item that received a remembered-or-forgotten outcome.
  Lures are excluded: trial_type in {new, foil, lure}.

Memory labeling (mirrors encoding_code/combined_encoding_power.py):
  ENCODING (phase1):
    - AMME file (MartinaChannels amyg, has test_yes_or_no): subsequent memory =
      test_yes_or_no (yes->remembered, no->forgotten).
    - BLAES (has imagename/stimulus_code, no memory col): subsequent memory joined
      from the phase3 test response for the same stimulus (Targ/old recognized
      'old' -> remembered, called 'new' -> forgotten).
  RETRIEVAL (phase3), old items only:
    - AMME: yes_or_no (yes->remembered, no->forgotten).
    - BLAES: response old->remembered, new->forgotten.

File locations:
  - BLAES (BJH/UIC/SLCH) + MartinaChannels amyg phase1: BLAES_DIR
  - amyg phase3: AMME_PARENT (Data_amyg{N}phase3MeasurePower.csv)
"""

from pathlib import Path
import glob
import numpy as np
import pandas as pd

AMME_BLAES = Path(__file__).resolve().parent.parent
BLAES_DIR = Path("/Users/martinahollearn/Library/CloudStorage/Box-Box/InmanLab/"
                 "BLAES_data/dissertation/LFP_analyses/Results_CSVOutput")
AMME_PARENT = Path("/Users/martinahollearn/Library/CloudStorage/Box-Box/InmanLab/"
                   "AMME_Data_Emory/AMME_Data/LFP_analyses_Martina/Results_CSVOutput")
IED_ENC = AMME_BLAES / "IED" / "AMMEBLAES_IEDs_trial_level_dissertation_study_usethis_cleaned_with_memory.csv"
IED_RET = AMME_BLAES / "IED" / "AMMEBLAES_IEDs_trial_level_dissertation_test_usethis_cleaned_with_memory.csv"

NEW_TYPES = {"new", "foil", "lure"}          # not old items
OLD_RESP = {"78", "66", "37", "old", "yes"}  # recognized as old
NEW_RESP = {"67", "86", "39", "new", "no"}   # called new


def nonfreq(f):
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
    return "old" if s in OLD_RESP else ("new" if s in NEW_RESP else None)


def norm_img(x):
    if pd.isna(x):
        return np.nan
    import os
    return os.path.basename(str(x).strip().replace("\\", "/")).lower()


def resolve_file(pat, phase):
    is_amme = pat.lower().startswith("amyg")
    if not is_amme:
        f = BLAES_DIR / f"Data_{pat}{phase}MeasurePower.csv"
        return (f, is_amme) if f.exists() else (None, is_amme)
    # amyg
    if phase == "phase1":
        hits = glob.glob(str(BLAES_DIR / f"*amyg*{pat.replace('amyg','')}*phase1*MeasurePower.csv"))
        hits = [h for h in hits if f"amyg{pat.replace('amyg','')}phase1" in h.replace("MartinaChannels", "")]
    else:
        hits = glob.glob(str(AMME_PARENT / f"Data_{pat}{phase}MeasurePower.csv"))
    return (Path(hits[0]), is_amme) if hits else (None, is_amme)


def trials_for_patient(pat, phase):
    """Return DataFrame of unique trials with columns: trial, memory (rem/forg/None)."""
    f, is_amme = resolve_file(pat, phase)
    if f is None:
        return None
    df = pd.read_csv(f, usecols=nonfreq(f))
    tid = "number" if is_amme else "trialIdx"
    if tid not in df.columns:
        tid = "trialIdx" if "trialIdx" in df.columns else "number"
    u = df.drop_duplicates(subset=[tid]).copy()

    # old-item mask via trial_type (exclude lures); encoding studied items are all old
    ttcol = next((c for c in ["trial_type", "test_trial_type"] if c in u.columns), None)
    if phase == "phase1":
        is_old = pd.Series(True, index=u.index)
    else:
        tt = u[ttcol].astype(str).str.strip().str.lower() if ttcol else pd.Series("", index=u.index)
        is_old = ~tt.isin(NEW_TYPES)

    # memory labeling
    if is_amme:
        col = "test_yes_or_no" if phase == "phase1" else "yes_or_no"
        yn = u[col].astype(str).str.strip().str.lower() if col in u.columns else pd.Series("", index=u.index)
        mem = np.where(yn == "yes", "remembered", np.where(yn == "no", "forgotten", None))
    else:
        if phase == "phase3":
            resp = u["response"].apply(norm_resp)
            mem = np.where(resp == "old", "remembered", np.where(resp == "new", "forgotten", None))
        else:
            mem = np.array([None] * len(u), dtype=object)  # BLAES encoding filled via phase3 lookup

    out = pd.DataFrame({"trial": u[tid].astype(int).values, "memory": mem, "is_old": is_old.values})
    if phase == "phase1":
        imgcol = next((c for c in ["imagename", "full_im_name"] if c in u.columns), None)
        out["img"] = u[imgcol].apply(norm_img).values if imgcol else np.nan
    out["patient"] = pat
    out["cohort"] = "AMME" if is_amme else "BLAES"
    return out[out["is_old"]].copy()


def phase3_image_memory_lookup():
    """(patient, normalized image name) -> remembered/forgotten, from every phase3
    power file (both BLAES and AMME dirs), old/Targ items only. Used to recover the
    subsequent-memory outcome of encoded items from the recognition test."""
    parts = []
    for d in (BLAES_DIR, AMME_PARENT):
        for f in glob.glob(str(d / "*phase3*MeasurePower.csv")):
            if "AllPatients" in f:
                continue
            df = pd.read_csv(f, usecols=nonfreq(f))
            imgcol = next((c for c in ["full_im_name", "imagename"] if c in df.columns), None)
            if imgcol is None:
                continue
            pat = df["Patient"].iloc[0]
            tt = df["trial_type"].astype(str).str.strip().str.lower() if "trial_type" in df else pd.Series("", index=df.index)
            old = df[~tt.isin(NEW_TYPES)].copy()
            if "response" in old.columns:
                mem = old["response"].apply(norm_resp).map({"old": "remembered", "new": "forgotten"})
            elif "yes_or_no" in old.columns:
                yn = old["yes_or_no"].astype(str).str.strip().str.lower()
                mem = np.where(yn == "yes", "remembered", np.where(yn == "no", "forgotten", None))
            else:
                continue
            old = old.assign(patient=pat, img=old[imgcol].apply(norm_img), memory=mem)
            parts.append(old[["patient", "img", "memory"]].dropna(subset=["img", "memory"]))
    return (pd.concat(parts, ignore_index=True).drop_duplicates(subset=["patient", "img"])
            if parts else pd.DataFrame(columns=["patient", "img", "memory"]))


def reconstruct(phase, patients):
    frames = [trials_for_patient(p, phase) for p in patients]
    frames = [f for f in frames if f is not None]
    allt = pd.concat(frames, ignore_index=True)
    if phase == "phase1":
        # Fill subsequent-memory outcome from the recognition test, by image name.
        # AMME keeps its inline test_yes_or_no where present; everything else (BLAES,
        # and AMME NaNs) is recovered from the phase3 image lookup.
        lk = phase3_image_memory_lookup().set_index(["patient", "img"])["memory"]
        keys = list(zip(allt["patient"], allt["img"]))
        recovered = pd.Series([lk.get(k, np.nan) for k in keys], index=allt.index)
        allt["memory"] = allt["memory"].where(allt["memory"].notna(), recovered)
    return allt


def counts(df):
    rem = int((df["memory"] == "remembered").sum())
    forg = int((df["memory"] == "forgotten").sum())
    return rem, forg


def ied_counts(csv):
    d = pd.read_csv(csv)
    d = d[d["MemoryOutcome"].isin(["remembered", "forgotten"])]
    t = d.groupby(["Patient", "Trial"])["MemoryOutcome"].first()
    return int((t == "remembered").sum()), int((t == "forgotten").sum())


def report(label, total_rem, total_forg, ied_rem, ied_forg):
    no_rem, no_forg = total_rem - ied_rem, total_forg - ied_forg
    print(f"\n===== {label} =====")
    def line(name, r, fg):
        n = r + fg
        ratio = r / fg if fg else float("nan")
        prop = r / n if n else float("nan")
        print(f"  {name:16s}: n={n:5d} | remembered={r:5d} forgotten={fg:5d} "
              f"| rem:forg={ratio:.2f} | %rem={prop:.1%}")
    line("TOTAL target", total_rem, total_forg)
    line("IED trials", ied_rem, ied_forg)
    line("no-IED trials", no_rem, no_forg)
    if no_rem < 0 or no_forg < 0:
        print("  *** WARNING: IED count exceeds reconstructed total -> raw labeling is incomplete.")
    return {"total": (total_rem, total_forg), "ied": (ied_rem, ied_forg), "noied": (no_rem, no_forg)}


def main():
    ied_e = pd.read_csv(IED_ENC)
    ied_r = pd.read_csv(IED_RET)
    enc_pts = sorted(ied_e["Patient"].unique())
    ret_pts = sorted(ied_r[ied_r["MemoryOutcome"].isin(["remembered", "forgotten"])]["Patient"].unique())

    enc = reconstruct("phase1", enc_pts)
    ret = reconstruct("phase3", ret_pts)

    e_rem, e_forg = counts(enc)
    r_rem, r_forg = counts(ret)
    ie_rem, ie_forg = ied_counts(IED_ENC)
    ir_rem, ir_forg = ied_counts(IED_RET)

    print(f"Reconstructed target trials with rem/forg label: "
          f"encoding={e_rem + e_forg} ({enc['patient'].nunique()} pts), "
          f"retrieval={r_rem + r_forg} ({ret['patient'].nunique()} pts)")

    res = {
        "Encoding": report("ENCODING", e_rem, e_forg, ie_rem, ie_forg),
        "Retrieval": report("RETRIEVAL", r_rem, r_forg, ir_rem, ir_forg),
    }
    return res


if __name__ == "__main__":
    main()
