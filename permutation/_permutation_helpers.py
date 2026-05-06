"""Shared helpers for cluster-based permutation scripts.

Provides:
  - add_allhpc_region: pools CA/DG/HPC -> ALLHPC for power, and
    CA_X/DG_X/HPC_X -> ALLHPC_X for coherence/PAC pairs.
  - FREQ_FAMILIES + family_masks: defines confirmatory (theta 4-8 Hz,
    slow gamma 35-50 Hz) and exploratory frequency masks.
  - run_family_permutations: cluster-based permutation run separately
    per frequency family with its own null distribution / FWE.
  - plot_region_panel_dual: panel plotter that shades confirmatory
    significant clusters dark, exploratory significant clusters light,
    and leaves non-significant ranges unshaded.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


# --- Frequency families ------------------------------------------------------

THETA_LO, THETA_HI = 4.0, 8.0
SLOWGAMMA_LO, SLOWGAMMA_HI = 35.0, 50.0

FREQ_FAMILIES = ("theta", "slow_gamma", "exploratory")
CONFIRMATORY_FAMILIES = ("theta", "slow_gamma")


def family_masks(freqs: np.ndarray) -> dict[str, np.ndarray]:
    freqs = np.asarray(freqs, dtype=float)
    theta = (freqs >= THETA_LO) & (freqs <= THETA_HI)
    sgamma = (freqs >= SLOWGAMMA_LO) & (freqs <= SLOWGAMMA_HI)
    explor = ~(theta | sgamma)
    return {"theta": theta, "slow_gamma": sgamma, "exploratory": explor}


def is_confirmatory(family: str) -> bool:
    return family in CONFIRMATORY_FAMILIES


# --- ALLHPC region pooling ---------------------------------------------------

HPC_SUBREGIONS = ("CA", "DG", "HPC")
ALLHPC_PARTNERS = ("BLA", "EC", "PRC")  # excludes PHG (already excluded)


def _pool_pair_name(allhpc_label: str, partner: str) -> str:
    a, b = sorted([allhpc_label, partner])
    return f"{a}_{b}"


def add_allhpc_region(
    grouped: pd.DataFrame,
    diff_cols: list[str],
    cond_col,
    modality: str,
) -> pd.DataFrame:
    """Append ALLHPC rows to a per-subject-mean dataframe.

    `grouped` columns: Patient, Region, <cond_col(s)>, <diff_cols...>.
    `cond_col` can be a string or list of strings (all condition columns
    that should be preserved when pooling across hippocampal subregions).
    For power: averages across {CA, DG, HPC} -> Region='ALLHPC'.
    For coherence/PAC: averages across {CA_X, DG_X, HPC_X} -> 'ALLHPC_X'
    for X in {BLA, EC, PRC}.
    Patients with at least one contributing subregion are included.
    """
    cond_cols = [cond_col] if isinstance(cond_col, str) else list(cond_col)
    group_cols = ["Patient"] + cond_cols

    if modality == "power":
        sub = grouped[grouped["Region"].isin(HPC_SUBREGIONS)].copy()
        if sub.empty:
            return grouped
        pooled = (sub.groupby(group_cols, as_index=False)[diff_cols].mean())
        pooled["Region"] = "ALLHPC"
        return pd.concat([grouped, pooled], ignore_index=True)

    pooled_frames = []
    for partner in ALLHPC_PARTNERS:
        member_pairs = [_pool_pair_name(s, partner) for s in HPC_SUBREGIONS]
        sub = grouped[grouped["Region"].isin(member_pairs)].copy()
        if sub.empty:
            continue
        pooled = (sub.groupby(group_cols, as_index=False)[diff_cols].mean())
        pooled["Region"] = _pool_pair_name("ALLHPC", partner)
        pooled_frames.append(pooled)
    if not pooled_frames:
        return grouped
    return pd.concat([grouped] + pooled_frames, ignore_index=True)


# --- Cluster permutation per family ------------------------------------------

def _find_clusters(t_vals: np.ndarray, t_thresh: float) -> list[tuple]:
    """Contiguous same-sign supra-threshold runs in t_vals (1-D)."""
    sign = np.zeros_like(t_vals, dtype=int)
    sign[t_vals > t_thresh] = 1
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


def _cluster_perm_paired_on_mask(
    diff: np.ndarray,
    mask: np.ndarray,
    n_perm: int,
    alpha: float,
    seed: int,
) -> dict:
    """Cluster permutation restricted to frequency indices where mask is True.

    Cluster detection only sees the masked freqs (indices are remapped to
    the masked subset, then mapped back). Null max-mass distribution is
    computed from the same masked freqs, so the FWE correction is local
    to this family.
    """
    n_subj = diff.shape[0]
    df = n_subj - 1
    t_thresh = float(stats.t.ppf(1 - alpha, df=df))

    masked_indices = np.where(mask)[0]
    if masked_indices.size == 0:
        return {"t_obs_full": np.zeros(diff.shape[1]),
                "t_thresh": t_thresh, "n_subj": n_subj,
                "obs_clusters": [], "null_max_mass": np.zeros(n_perm)}

    diff_m = diff[:, masked_indices]
    mean = diff_m.mean(axis=0)
    sd = diff_m.std(axis=0, ddof=1)
    se = sd / np.sqrt(n_subj)
    se = np.where(se == 0, np.nan, se)
    t_obs_m = np.where(np.isnan(se), 0, mean / se)

    obs_clusters_local = _find_clusters(t_obs_m, t_thresh)

    rng = np.random.default_rng(seed)
    null_max_mass = np.zeros(n_perm)
    for p in range(n_perm):
        signs = rng.choice([-1, 1], size=n_subj)
        flipped = diff_m * signs[:, None]
        m = flipped.mean(axis=0)
        s = flipped.std(axis=0, ddof=1)
        se_p = s / np.sqrt(n_subj)
        se_p = np.where(se_p == 0, np.nan, se_p)
        t_p = np.where(np.isnan(se_p), 0, m / se_p)
        cls = _find_clusters(t_p, t_thresh)
        null_max_mass[p] = max((abs(c[3]) for c in cls), default=0.0)

    t_obs_full = np.zeros(diff.shape[1])
    t_obs_full[masked_indices] = t_obs_m

    obs_clusters = []
    for (i0_loc, i1_loc, sgn, mass) in obs_clusters_local:
        i0_global = int(masked_indices[i0_loc])
        i1_global = int(masked_indices[i1_loc])
        p_val = float(np.mean(null_max_mass >= abs(mass)))
        obs_clusters.append({
            "start_idx": i0_global,
            "end_idx": i1_global,
            "n_freqs": int(i1_loc - i0_loc + 1),
            "sign": int(sgn),
            "cluster_mass": float(mass),
            "p_value": p_val,
        })

    return {"t_obs_full": t_obs_full, "t_thresh": t_thresh,
            "n_subj": n_subj, "obs_clusters": obs_clusters,
            "null_max_mass": null_max_mass}


def run_family_permutations(
    X_A: np.ndarray,
    X_B: np.ndarray,
    freqs: np.ndarray,
    n_perm: int,
    alpha: float,
    seed: int,
    direction_labels: tuple[str, str],
) -> dict:
    """Run cluster permutation separately for each frequency family.

    Returns dict with:
      - "n_subj"
      - "t_thresh"
      - "clusters": flat list of clusters with 'family' tag
      - "null_by_family": {family: null_max_mass array}
    """
    diff = X_A - X_B
    masks = family_masks(freqs)
    pos_label, neg_label = direction_labels
    all_clusters = []
    null_by_family = {}
    t_thresh = None
    n_subj = X_A.shape[0]
    # Use a different seed per family so the three nulls aren't correlated.
    family_seed_offsets = {"theta": 0, "slow_gamma": 1, "exploratory": 2}
    for family in FREQ_FAMILIES:
        res = _cluster_perm_paired_on_mask(
            diff, masks[family], n_perm, alpha,
            seed + family_seed_offsets[family],
        )
        t_thresh = res["t_thresh"]
        null_by_family[family] = res["null_max_mass"]
        for c in res["obs_clusters"]:
            c2 = dict(c)
            c2["family"] = family
            c2["confirmatory"] = is_confirmatory(family)
            c2["direction"] = pos_label if c["sign"] > 0 else neg_label
            all_clusters.append(c2)
    return {"n_subj": n_subj, "t_thresh": t_thresh,
            "clusters": all_clusters, "null_by_family": null_by_family}


# --- Plotting ----------------------------------------------------------------

def plot_region_panel_dual(
    ax,
    freqs: np.ndarray,
    X_A: np.ndarray,
    X_B: np.ndarray,
    clusters: list[dict],
    region: str,
    line_color_A: str,
    line_color_B: str,
    label_A: str,
    label_B: str,
    fill_color: str = "#c0392b",
    xlim: tuple[float, float] | None = None,
):
    """Plot a single region panel using the dual confirmatory/exploratory
    shading scheme:
      - confirmatory significant cluster -> dark fill (alpha 0.85)
      - exploratory significant cluster  -> light fill (alpha 0.30)
      - non-significant cluster          -> NO fill at all
    """
    mean_A = X_A.mean(axis=0)
    sem_A = X_A.std(axis=0, ddof=1) / np.sqrt(X_A.shape[0])
    mean_B = X_B.mean(axis=0)
    sem_B = X_B.std(axis=0, ddof=1) / np.sqrt(X_B.shape[0])

    ax.fill_between(freqs, mean_A - sem_A, mean_A + sem_A,
                    color=line_color_A, alpha=0.18, lw=0, zorder=1)
    ax.fill_between(freqs, mean_B - sem_B, mean_B + sem_B,
                    color=line_color_B, alpha=0.18, lw=0, zorder=1)

    sig_clusters = [c for c in clusters if c["p_value"] < 0.05]
    # Confirmatory rendered AFTER exploratory so dark sits on top of light.
    sig_clusters.sort(key=lambda c: (c["confirmatory"], -c["n_freqs"]))

    for c in sig_clusters:
        i0, i1 = c["start_idx"], c["end_idx"]
        s = slice(i0, i1 + 1)
        alpha_fill = 0.85 if c["confirmatory"] else 0.30
        zfill = 3 if c["confirmatory"] else 2
        ax.fill_between(freqs[s], mean_A[s], mean_B[s],
                        color=fill_color, alpha=alpha_fill, lw=0,
                        zorder=zfill, interpolate=True)

    # Stack p-value annotations vertically to avoid overlap.
    ymin, ymax = ax.get_ylim()
    full_range = ymax - ymin
    annotations = []
    for c in sig_clusters:
        i0, i1 = c["start_idx"], c["end_idx"]
        s = slice(i0, i1 + 1)
        f_mid = (freqs[i0] + freqs[i1]) / 2
        y_top = max(np.max(mean_A[s] + sem_A[s]),
                    np.max(mean_B[s] + sem_B[s]))
        annotations.append((f_mid, y_top, c))

    # Sort by x-position then assign vertical slots so overlapping labels
    # don't collide. Each row is one full_range step apart.
    annotations.sort(key=lambda a: a[0])
    occupied = []  # list of (x_lo, x_hi, slot)
    sig_text_y_max = -np.inf
    for f_mid, y_top, c in annotations:
        # Approximate label half-width in data units: 6% of x-range.
        x_range = freqs[-1] - freqs[0]
        label_hw = 0.06 * x_range
        x_lo, x_hi = f_mid - label_hw, f_mid + label_hw
        slot = 0
        while any(slot == s_used and not (x_hi < xl or x_lo > xh)
                  for xl, xh, s_used in occupied):
            slot += 1
        occupied.append((x_lo, x_hi, slot))
        y_label = y_top + (0.02 + 0.07 * slot) * full_range
        sig_text_y_max = max(sig_text_y_max, y_label)
        tag = "C" if c["confirmatory"] else "E"
        weight = "bold" if c["confirmatory"] else "normal"
        ax.text(f_mid, y_label,
                f"{tag} p={c['p_value']:.3f}",
                ha="center", va="bottom", fontsize=7,
                color="black", fontweight=weight, zorder=5)

    ax.plot(freqs, mean_A, color=line_color_A, lw=1.6, label=label_A, zorder=3)
    ax.plot(freqs, mean_B, color=line_color_B, lw=1.6, label=label_B, zorder=3)
    ax.axhline(0, color="0.7", lw=0.4, ls="--", zorder=0)

    # Light vertical guides at the confirmatory band edges.
    for x in (THETA_LO, THETA_HI, SLOWGAMMA_LO, SLOWGAMMA_HI):
        ax.axvline(x, color="0.85", lw=0.4, ls=":", zorder=0)

    ymin, ymax = ax.get_ylim()
    full_range = ymax - ymin
    if np.isfinite(sig_text_y_max):
        target_top = max(ymax, sig_text_y_max + 0.06 * full_range)
        ax.set_ylim(ymin, target_top)

    if xlim is not None:
        ax.set_xlim(*xlim)

    title = f"{region} (n={X_A.shape[0]})"
    if any(c["p_value"] < .05 and c["confirmatory"] for c in clusters):
        title += " **"
    elif any(c["p_value"] < .05 for c in clusters):
        title += " *"
    ax.set_title(title, fontsize=10)
    ax.tick_params(labelsize=8)


# --- Band-wise permutation ---------------------------------------------------
# Bands per user spec (gaps allowed between exploratory bands).
# Confirmatory: theta (4-8), slow_gamma (30-55).
# Exploratory: delta (~1.95-3), alpha (9-13), beta (14-29), HFA (70-100).

BANDS = {
    "delta":      (1.95, 3.0),    # exploratory
    "theta":      (4.0,  8.0),    # confirmatory
    "alpha":      (9.0,  13.0),   # exploratory
    "beta":       (14.0, 29.0),   # exploratory
    "slow_gamma": (30.0, 55.0),   # confirmatory
    "HFA":        (70.0, 100.0),  # exploratory
}
CONFIRMATORY_BANDS = ("theta", "slow_gamma")
EXPLORATORY_BANDS = ("delta", "alpha", "beta", "HFA")
BAND_FAMILIES = {
    "confirmatory": CONFIRMATORY_BANDS,
    "exploratory": EXPLORATORY_BANDS,
}


def band_masks(freqs: np.ndarray) -> dict[str, np.ndarray]:
    """Return per-band boolean masks over the freq axis."""
    freqs = np.asarray(freqs, dtype=float)
    return {name: (freqs >= lo) & (freqs <= hi)
            for name, (lo, hi) in BANDS.items()}


def _paired_t(diff_band_avg: np.ndarray) -> float:
    """Paired t-stat for a 1-D vector of subject-level differences."""
    n = len(diff_band_avg)
    if n < 2:
        return 0.0
    m = float(np.mean(diff_band_avg))
    s = float(np.std(diff_band_avg, ddof=1))
    if s == 0 or not np.isfinite(s):
        return 0.0
    return m / (s / np.sqrt(n))


def run_band_permutations(
    X_A: np.ndarray,
    X_B: np.ndarray,
    freqs: np.ndarray,
    n_perm: int,
    alpha: float,
    seed: int,
    direction_labels: tuple[str, str],
) -> dict:
    """Band-averaged paired permutation test with family-wise FWE.

    For each band defined in BANDS, average the per-subject differences
    (X_A - X_B) over the band's freq mask, then compute a paired t-stat.
    Per family ('confirmatory' = {theta, slow_gamma}; 'exploratory' =
    {delta, alpha, beta, HFA}), build a null distribution of the max |t|
    across the family's bands by sign-flipping subjects. Each band's
    p-value = proportion of family null max |t| >= observed |t|.

    Returns dict with:
      - "n_subj"
      - "bands": list of band records (one per band): {band, family,
        confirmatory, freq_lo_hz, freq_hi_hz, t_stat, p_value, direction}
      - "null_by_family": {family: array of null max |t|}
    """
    diff = X_A - X_B  # (n_subj, n_freqs)
    n_subj = diff.shape[0]
    masks = band_masks(freqs)
    pos_label, neg_label = direction_labels

    # Subject-level band averages.
    band_diff_avg = {}  # name -> (n_subj,) vector
    for name, mask in masks.items():
        if not mask.any():
            band_diff_avg[name] = np.zeros(n_subj)
        else:
            band_diff_avg[name] = diff[:, mask].mean(axis=1)

    # Observed t per band.
    t_obs = {name: _paired_t(band_diff_avg[name]) for name in BANDS}

    # Permutation: sign-flip subjects, recompute t per band, take family max |t|.
    rng = np.random.default_rng(seed)
    null_by_family = {fam: np.zeros(n_perm) for fam in BAND_FAMILIES}
    for p in range(n_perm):
        signs = rng.choice([-1, 1], size=n_subj)
        # Recompute per-band t with flipped signs (only affects sign of mean).
        t_p = {}
        for name in BANDS:
            v = band_diff_avg[name] * signs
            t_p[name] = _paired_t(v)
        for fam, members in BAND_FAMILIES.items():
            mx = max((abs(t_p[b]) for b in members), default=0.0)
            null_by_family[fam][p] = mx

    band_records = []
    for name, (lo, hi) in BANDS.items():
        confirmatory = name in CONFIRMATORY_BANDS
        family = "confirmatory" if confirmatory else "exploratory"
        t = t_obs[name]
        null_mx = null_by_family[family]
        p_val = float(np.mean(null_mx >= abs(t)))
        band_records.append({
            "band": name,
            "family": family,
            "confirmatory": confirmatory,
            "freq_lo_hz": float(lo),
            "freq_hi_hz": float(hi),
            "t_stat": float(t),
            "p_value": p_val,
            "direction": pos_label if t > 0 else neg_label,
        })
    return {"n_subj": n_subj,
            "bands": band_records,
            "null_by_family": null_by_family}


def plot_region_panel_bandwise(
    ax,
    freqs: np.ndarray,
    X_A: np.ndarray,
    X_B: np.ndarray,
    bands: list[dict],
    region: str,
    line_color_A: str,
    line_color_B: str,
    label_A: str,
    label_B: str,
    fill_color: str = "#c0392b",
    xlim: tuple[float, float] | None = None,
):
    """Panel plot with per-frequency lines + per-band shading.

    Bands with p<.05 get vertical-strip shading spanning their freq range:
      - confirmatory significant -> dark fill (alpha 0.85)
      - exploratory significant -> light fill (alpha 0.30)
      - non-significant -> no shading (band freq range left clear)
    """
    mean_A = X_A.mean(axis=0)
    sem_A = X_A.std(axis=0, ddof=1) / np.sqrt(X_A.shape[0])
    mean_B = X_B.mean(axis=0)
    sem_B = X_B.std(axis=0, ddof=1) / np.sqrt(X_B.shape[0])

    ax.fill_between(freqs, mean_A - sem_A, mean_A + sem_A,
                    color=line_color_A, alpha=0.18, lw=0, zorder=1)
    ax.fill_between(freqs, mean_B - sem_B, mean_B + sem_B,
                    color=line_color_B, alpha=0.18, lw=0, zorder=1)

    # Significant band shading + p-value annotations.
    # Fill between the two mean lines (not full vertical axvspan) within
    # each significant band's frequency range.
    sig_bands = [b for b in bands if b["p_value"] < 0.05]
    sig_bands.sort(key=lambda b: (b["confirmatory"], -(b["freq_hi_hz"] - b["freq_lo_hz"])))

    annotations = []
    freqs_arr = np.asarray(freqs, dtype=float)
    for b in sig_bands:
        lo, hi = b["freq_lo_hz"], b["freq_hi_hz"]
        band_mask = (freqs_arr >= lo) & (freqs_arr <= hi)
        if not band_mask.any():
            continue
        alpha_fill = 0.85 if b["confirmatory"] else 0.30
        zfill = 3 if b["confirmatory"] else 2
        ax.fill_between(freqs_arr[band_mask],
                        mean_A[band_mask], mean_B[band_mask],
                        color=fill_color, alpha=alpha_fill, lw=0,
                        zorder=zfill, interpolate=True)
        annotations.append(((lo + hi) / 2, b))

    # Stack p-value annotations vertically when bands overlap horizontally.
    annotations.sort(key=lambda a: a[0])
    occupied = []
    sig_text_y_max = -np.inf
    ymin, ymax = ax.get_ylim()
    full_range = ymax - ymin
    for f_mid, b in annotations:
        x_range = freqs[-1] - freqs[0]
        label_hw = 0.05 * x_range
        x_lo, x_hi = f_mid - label_hw, f_mid + label_hw
        slot = 0
        while any(slot == s_used and not (x_hi < xl or x_lo > xh)
                  for xl, xh, s_used in occupied):
            slot += 1
        occupied.append((x_lo, x_hi, slot))
        # Place near the top of the panel.
        y_label = ymax + (0.02 + 0.07 * slot) * full_range
        sig_text_y_max = max(sig_text_y_max, y_label)
        tag = "C" if b["confirmatory"] else "E"
        weight = "bold" if b["confirmatory"] else "normal"
        ax.text(f_mid, y_label,
                f"{tag} {b['band']} p={b['p_value']:.3f}",
                ha="center", va="bottom", fontsize=7,
                color="black", fontweight=weight, zorder=5)

    ax.plot(freqs, mean_A, color=line_color_A, lw=1.6, label=label_A, zorder=4)
    ax.plot(freqs, mean_B, color=line_color_B, lw=1.6, label=label_B, zorder=4)
    ax.axhline(0, color="0.7", lw=0.4, ls="--", zorder=0)

    # Light vertical guides at all band edges.
    for name, (lo, hi) in BANDS.items():
        ax.axvline(lo, color="0.85", lw=0.4, ls=":", zorder=0)
        ax.axvline(hi, color="0.85", lw=0.4, ls=":", zorder=0)

    if xlim is not None:
        ax.set_xlim(*xlim)

    if np.isfinite(sig_text_y_max):
        target_top = max(ymax, sig_text_y_max + 0.06 * full_range)
        ax.set_ylim(ymin, target_top)

    title = f"{region} (n={X_A.shape[0]})"
    if any(b["p_value"] < .05 and b["confirmatory"] for b in bands):
        title += " **"
    elif any(b["p_value"] < .05 for b in bands):
        title += " *"
    ax.set_title(title, fontsize=10)
    ax.tick_params(labelsize=8)
