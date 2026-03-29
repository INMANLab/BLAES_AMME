# AMME-BLAES Analysis Documentation: Averaging & Calculation Pipelines

This document describes how each analysis script loads, averages, and prepares data for plotting. It covers Power, Coherence, and PAC for both Encoding and Retrieval, and flags any differences between the two phases.

---

## Table of Contents

1. [Shared Concepts](#shared-concepts)
2. [Power Analysis](#power-analysis)
3. [Coherence Analysis](#coherence-analysis)
4. [PAC Analysis](#pac-analysis)
5. [Encoding vs Retrieval Consistency Check](#encoding-vs-retrieval-consistency-check)

---

## 1. Shared Concepts

### Data Sources
All analyses read **pre-computed** spectral values from CSV files produced upstream (typically in MATLAB). No raw signal processing occurs in any of these Python scripts. Each CSV row represents a single trial for one patient and one region (or region pair), with columns for frequency-bin values.

### Two Column Families
- **`post_Freq_X`**: Post-stimulus values at frequency X Hz (raw post-stimulus spectra).
- **`diff_Freq_X`** (Power/Coherence) or **`pre_Freq_X`** (PAC): Used for baseline correction. For Power and Coherence, `diff_Freq_X` arrives pre-computed (post minus baseline). For PAC, `pre_Freq_X` is the pre-stimulus MI and baseline correction is computed in-script as `post - pre`.

### Two Study Cohorts
- **BLAES**: Patient IDs like BJH0XX.
- **AMME**: Patient IDs starting with `amyg`.
- Combined ("All") merges both after independent loading.

### Averaging Hierarchy (applies to all three measures)
1. **Trial averaging**: Arithmetic mean across trials within each (Patient, Region, Condition) cell via `pandas groupby().mean()`.
2. **Composite ROI construction**: Arithmetic mean across contributing ROIs per patient via `np.nanmean(np.vstack(...), axis=0)`.
3. **Band-mean extraction** (for bar plots): Arithmetic mean of spectral values within a frequency band via `spectrum[mask].mean()`.
4. **Cross-subject averaging** (at plot time): Arithmetic mean across patients for the group line/bar, with SD or SEM for error.

**All means are arithmetic. No medians, log transforms, z-scores, or normalization are applied anywhere.**

### Error Bars / Shading
- **Raw spectral line plots**: mean +/- 1 SD (numpy default ddof=0).
- **Baseline-corrected spectral line plots**: mean +/- 1 SEM (std(ddof=0) / sqrt(N)).
- **Bar plots**: SEM, either via seaborn `errorbar='se'` or manually with `std(ddof=1) / sqrt(N)`.

### Region Pair Directionality: Coherence vs PAC

**This is a critical distinction between the two pairwise measures.**

- **Coherence is symmetric.** The coherence between region A and region B is the same regardless of which is listed first. Both coherence scripts (`combined_encoding_coherence.py`, `combined_retrieval_coherence.py`) enforce this by **alphabetically sorting** the two parts of each region pair label via `normalize_region_label()` (e.g., `HPC_BLA` and `BLA_HPC` both become `BLA_HPC`). If a patient has data under both orderings, the spectra are averaged together with `np.nanmean`. This is correct behavior for a symmetric measure.

- **PAC is directional.** The coupling from region A's phase to region B's amplitude is NOT the same as B's phase to A's amplitude. Both PAC scripts (`BLAES_Group_PAC_analyses_from_matlab_encoding.py`, `BLAES_Group_PAC_analyses_from_matlab_retrieval.py`) preserve the original ordering via `clean_region_label()`, which only renames `ER` → `EC` but does **not** sort the parts. `EC_PRC` (EC phase → PRC amplitude) and `PRC_EC` (PRC phase → EC amplitude) remain as separate, distinct labels throughout the entire pipeline. This is consistent between encoding and retrieval.

- **PAC composite ROIs are also directional.** For example, `BLA_ALLHPC` (BLA phase → hippocampal amplitude) and `ALLHPC_BLA` (hippocampal phase → BLA amplitude) are built and plotted as separate composites. The composite logic text in `combined_pac_common.py` explicitly states: *"preserve source-to-target PAC direction for each original region pair."*

- **Power uses single regions**, so directionality is not applicable.

### Statistical Tests
**None of these scripts perform formal statistical tests.** They are purely visualization and summary-export pipelines. Trial-level data is exported as MLMR CSVs for statistical modeling elsewhere.

---

## 2. Power Analysis

**Scripts**: `combined_encoding_power.py`, `combined_retrieval_power.py`

### 2a. Data Loading

| | Encoding | Retrieval |
|---|---|---|
| **BLAES source** | `*phase1*Power*.csv` | `*phase3*Power*.csv` |
| **AMME source** | `*Power*.csv` from Phase1/ | `*Power*.csv` from Phase2/ |
| **Columns used** | `post_Freq_*`, `diff_Freq_*` | `post_Freq_*`, `diff_Freq_*` |

**Encoding-specific**: BLAES encoding trials do not carry their own stim/memory labels. A **retrieval lookup** is built from phase-3 CSVs to determine (a) whether each encoding trial's image was later stimulated or not, and (b) whether it was later remembered or forgotten (hit vs miss). Only encoding trials with valid subsequent-memory labels are kept.

**Retrieval-specific**: BLAES retrieval CSVs contain `stimulation` and `response`/`trial_type` columns directly. Memory is determined by whether old items were correctly identified (hit = remembered) or missed (forgotten). Only old-item trials are used; correct rejections and false alarms are dropped.

**AMME** (both phases): `test_trial_type` (stim/nostim) and `test_yes_or_no` (yes/no) columns are read directly from the CSV.

### 2b. Subject/Region Exclusions

| | Encoding | Retrieval |
|---|---|---|
| **BLAES BLA** | BJH042, BJH029 excluded | (none noted) |
| **AMME BLA** | amyg016, amyg046, amyg057, amyg037 | amyg057 |
| **AMME HPC** | (none) | amyg030 |
| **AMME CA** | (none) | amyg034 |

### 2c. Frequency Bands

- **Theta**: 4-8 Hz
- **Slow Gamma**: 30-55 Hz

### 2d. Averaging Pipeline (step by step)

1. **Load CSV**: One row per trial with `post_Freq_X` (raw power in dB) and `diff_Freq_X` (baseline-corrected power in dB, pre-computed upstream).

2. **Trial-to-patient mean**: `df.groupby(['Patient', 'Region', 'stimulation'])[freq_cols].mean()` — arithmetic mean across all trials for that patient/region/condition. Produces one spectrum per patient per ROI per condition.

3. **Memory split** (when applicable): Further groupby adds `memory_cond`, yielding spectra for stim-remembered, stim-forgotten, nostim-remembered, nostim-forgotten.

4. **Composite ROI construction** (from `combined_power_common.py`):
   - **ALLHPC** = mean of that patient's CA, DG, HPC spectra (whichever available): `np.nanmean(np.vstack([CA, DG, HPC]), axis=0)`.
   - **MTL** = mean of CA, DG, HPC, EC, PHG, PRC spectra.
   - Each constituent ROI contributes equally regardless of trial count.

5. **"Overall" spectrum for spectral plots**:
   - When `use_memory_collapsed_overall=True`: averages the 4 condition spectra (stim-rem, stim-forg, nostim-rem, nostim-forg) per patient per ROI with `np.nanmean`. This equally weights the 4 cells rather than weighting by trial count.
   - When `use_memory_collapsed_overall=False`: uses `group_all_power` (straight mean of ALL trials).
   - **Encoding BLAES**: False (uses straight trial mean). **Encoding AMME**: True. **Retrieval (both)**: True.

6. **Band-mean extraction** (for bar plots): Boolean mask selects frequencies in the band, then `spectrum[mask].mean()`. For Stim-minus-NoStim difference bars: `stim_band_mean - nostim_band_mean` per patient.

7. **Cross-subject bar**: `seaborn.barplot(errorbar='se')` computes arithmetic mean and SEM of per-patient values. Individual patient dots overlaid.

8. **Cross-subject spectral lines**: Stack all patients' spectra → `mat.mean(axis=0)` for the line, `mat.std(axis=0)` for SD shading (raw) or `std/sqrt(N)` for SEM shading (baseline-corrected).

### 2e. Data Structure (dictionary keys)

All stored as `{ROI_string: {Patient_string: np.array_of_spectrum}}`:

| Key | Description |
|---|---|
| `group_all_power` | Post-stimulus spectrum, all trials |
| `stim_power` / `nostim_power` | Post-stimulus, by stim condition |
| `bc_stim` / `bc_nostim` | Baseline-corrected, by stim condition |
| `stim_rem` / `stim_forg` / `nostim_rem` / `nostim_forg` | Post-stimulus, by stim x memory |
| `bc_stim_rem` / `bc_stim_forg` / `bc_nostim_rem` / `bc_nostim_forg` | Baseline-corrected, by stim x memory |
| `freqs_post` | Frequency axis for post columns (np.array) |
| `freqs_diff` | Frequency axis for diff columns (np.array) |

### 2f. Plot Types

| Plot | X-axis | Y-axis | Grouping | Error |
|---|---|---|---|---|
| Spectral by ROI | Freq (Hz) | Power (dB) | One line per ROI | +/- SD |
| Spectral by patient | Freq (Hz) | Power (mean across ROIs) | One line per patient | none |
| Stim vs NoStim spectral | Freq (Hz) | Power (dB) | Two panels, lines per ROI | +/- SD |
| BC per ROI | Freq (Hz) | BC Power (dB) | One subplot per ROI, stim vs nostim | +/- SEM |
| BC bar (Stim-NoStim diff) | ROI | Band mean diff | Bars per ROI, dots per patient | SEM |
| Grouped condition bars | ROI | BC band mean | Paired bars (nostim/stim), connected dots | SEM |
| Quadrant (stim x memory) | Freq (Hz) | Power (dB) | 2x2 grid, lines per ROI | +/- SD |
| Connected dots | ROI | BC band mean | Paired bars, split by memory | SEM |

---

## 3. Coherence Analysis

**Scripts**: `combined_encoding_coherence.py`, `combined_retrieval_coherence.py`

### 3a. Data Loading

| | Encoding | Retrieval |
|---|---|---|
| **BLAES source** | `*phase1*Coherency*.csv` (+ phase3 for memory lookup) | `*phase3*Coherency*.csv` |
| **AMME source** | `*Coherency*.csv` from Phase1/ | `*Coherency*.csv` from Phase2/ |
| **Columns used** | `post_Freq_*`, `diff_Freq_*` | `post_Freq_*`, `diff_Freq_*` |

**Encoding-specific**: Same as Power — BLAES encoding uses a phase-3 retrieval lookup to determine subsequent memory and stim condition for each encoding trial.

**Region normalization**: `ER` renamed to `EC`. Two-part region pair names are **alphabetically sorted** (e.g., `HPC_BLA` becomes `BLA_HPC`) because coherence is a symmetric measure — the coherence between A and B is the same as between B and A. If alphabetical sorting causes duplicates for the same patient (i.e., a patient had data under both `HPC_BLA` and `BLA_HPC`), those spectra are averaged together with `np.nanmean`. This collapsing is intentional and correct for coherence.

### 3b. Subject/Region Exclusions

| | Encoding | Retrieval |
|---|---|---|
| **BLAES BLA** | BJH042, BJH029 | (none noted) |
| **AMME BLA** | amyg016, amyg046, amyg057, amyg037 | amyg057 |
| **AMME HPC** | (none) | amyg030 |
| **AMME CA pairs** | amyg034 from CA_DG, CA_HPC | amyg034 from CA_DG, CA_HPC |

### 3c. Frequency Bands

- **Theta**: 4-8 Hz
- **Slow Gamma**: 30-55 Hz

### 3d. Averaging Pipeline

Identical to Power (see section 2d), except:
- Regions are **pairs** (e.g., BLA_CA, BLA_HPC) rather than single regions.
- Composite ROIs are:
  - **BLA_ALLHPC**: mean of BLA_CA, BLA_DG, BLA_HPC per patient.
  - **BLA_MTL**: mean of all available BLA-to-non-BLA pairs per patient.
  - **ALLHPC_EC**: mean of CA_EC, DG_EC, EC_HPC per patient.
  - **ALLHPC_PRC**: mean of CA_PRC, DG_PRC, HPC_PRC per patient.
- `use_memory_collapsed_overall`: **Encoding BLAES**: False. **Encoding AMME**: True. **Retrieval (both)**: True.

All other steps (trial mean → composite → band extraction → cross-subject) are the same arithmetic mean pipeline as Power.

### 3e. Data Structure

Same key structure as Power (section 2e), with the same nested `{region_pair: {patient: spectrum}}` format.

### 3f. Plot Types

Same plot types as Power (section 2f), with "ROI" replaced by "ROI pair" on bar plot x-axes and "Coherency (dB)" on y-axes.

---

## 4. PAC Analysis

**Scripts**: `combined_encoding_pac.py`, `combined_retrieval_pac.py`
**Underlying modules**: `to_combine/BLAES_Group_PAC_analyses_from_matlab_encoding.py`, `to_combine/BLAES_Group_PAC_analyses_from_matlab_retrieval.py`
**Shared logic**: `combined_pac_common.py`

### 4a. Data Loading

| | Encoding | Retrieval |
|---|---|---|
| **Source** | `*phase1*PAC_TG*.csv` | `*phase3*PAC_TG*.csv` |
| **Columns** | `pre_Freq_*`, `post_Freq_*` | `pre_Freq_*`, `post_Freq_*` |

"TG" in filenames stands for Theta-Gamma (theta phase, gamma amplitude).

**Key difference from Power/Coherence**: PAC CSVs have `pre_Freq_*` and `post_Freq_*` columns containing Modulation Index (MI) values at each amplitude frequency. The MI values are pre-computed in MATLAB. **Baseline correction is computed in-script** as `diff_Freq_X = post_Freq_X - pre_Freq_X` (simple subtraction per trial per frequency bin).

### 4b. Memory Condition Assignment

**Encoding**: Memory is derived from either `ret_response` (mapped: Old=remembered, New=forgotten) or `test_yes_or_no` (yes=remembered, no=forgotten).

**Retrieval BLAES**: Cross-references `trial_type_raw` (old/targ vs new/foil) with participant response to classify as hit (remembered) or miss (forgotten). Only old-item trials kept.

**Retrieval AMME**: `trial_type` → stim/nostim; `yes_or_no` → remembered/forgotten.

### 4c. Subject/Region Exclusions and Region Handling

- Encoding: `amyg066` excluded from all BLA-containing regions.
- Same-region pairs (e.g., BLA_BLA) dropped in both phases.
- Region pairs containing `PNAS` dropped in both phases.
- `ER` renamed to `EC` via `clean_region_label()`.
- **Direction is preserved**: Unlike coherence, PAC region pairs are NOT alphabetically sorted. `EC_PRC` (EC phase → PRC amplitude) and `PRC_EC` (PRC phase → EC amplitude) are kept as separate labels and plotted independently. This is consistent between encoding and retrieval.

### 4d. Frequency Band

- **Slow Gamma**: **35-50 Hz** (NOTE: different from Power/Coherence which use 30-55 Hz)

### 4e. Averaging Pipeline

1. **Per-trial MI values** from MATLAB CSVs as `pre_Freq_X` and `post_Freq_X`.

2. **Baseline correction per trial** (computed in-script): `diff_Freq_X = post_Freq_X - pre_Freq_X`.

3. **Trial-to-patient mean**: `df.groupby(['Patient', 'Region'])[freq_cols].mean()` — arithmetic mean across trials within each (Patient, Region, Condition) cell. Produces one MI vector per patient per directed region pair per condition.

4. **Composite ROI construction** (from `combined_pac_common.py`):
   - PAC is **directional** (source→target), so composites preserve direction.
   - **BLA_ALLHPC**: mean of BLA→CA, BLA→DG, BLA→HPC MI spectra per patient.
   - **BLA_MTL**: mean of all BLA→X MI spectra per patient.
   - Additional directional composites: ALLHPC_BLA, MTL_BLA, ALLHPC_EC, EC_ALLHPC, ALLHPC_PRC, PRC_ALLHPC.
   - All via `np.nanmean(np.vstack(vectors), axis=0)`.

5. **Band-mean extraction** (for bar plots): `np.nanmean(values[mask])` where mask selects 35-50 Hz.

6. **Condition difference**: `stim_vector - nostim_vector` element-wise per patient, then band mean of the difference.

7. **Cross-subject bar**: `seaborn.barplot(errorbar='se')` or manual `mean()` and `std(ddof=1)/sqrt(N)`.

8. **Cross-subject spectral lines**: `mat.mean(axis=0)` for line, `mat.std(axis=0)` for SD shading (raw) or SEM shading (baseline-corrected).

### 4f. Data Structure

| Key | Structure | Description |
|---|---|---|
| `post_all` | `{region: {patient: np.array}}` | Post-stimulus MI, all trials |
| `post_stim` / `post_nostim` | same | Post MI by stim condition |
| `post_stim_rem` / `post_stim_forg` / `post_nostim_rem` / `post_nostim_forg` | same | Post MI by stim x memory |
| `diff_stim` / `diff_nostim` | same | Baseline-corrected MI by stim |
| `diff_stim_rem` / `diff_stim_forg` / `diff_nostim_rem` / `diff_nostim_forg` | same | Baseline-corrected MI by stim x memory |
| `freqs_post` | np.array | Amplitude frequency axis |
| `freqs_diff` | np.array | Same as freqs_post |
| `trial_count_rows` | list of DataFrames | Trial counts by Patient/Region/condition |

**Note**: PAC does not have separate `group_all_power` vs memory-collapsed overall logic. The `post_all` key is a straight mean of all trials.

### 4g. Plot Types

| Plot | X-axis | Y-axis | Grouping | Error |
|---|---|---|---|---|
| PAC by ROI | Amplitude freq (Hz) | Post-stim MI (mean) | One line per ROI | +/- SD |
| PAC by patient | Amplitude freq (Hz) | MI (mean across ROIs) | One line per patient | none |
| Stim vs NoStim | Amplitude freq (Hz) | MI | Two panels, lines per ROI | +/- SD |
| BC per ROI | Amplitude freq (Hz) | BC MI | One subplot per ROI, stim vs nostim | +/- SEM |
| MI Difference per ROI | Amplitude freq (Hz) | Stim-NoStim MI diff | Per ROI, panels for Rem/Forg | +/- SEM |
| BC bar (Stim-NoStim diff) | ROI | Band mean MI diff | Bars per ROI, dots per patient | SEM |
| Quadrant (stim x memory) | Amplitude freq (Hz) | Raw MI | 2x2 grid, lines per ROI | +/- SD |
| Connected dots | ROI | BC band mean MI | Paired bars, split by memory | SEM |

### 4h. BLA Composite Logic (from `combined_pac_common.py`)

The composite pipeline is documented in a text note saved with outputs:

> 1. Preserve source-to-target PAC direction for each original region pair.
> 2. Average PAC across trials within each patient, ordered region pair, and condition to form one spectrum per directed pair.
> 3. For composite ROIs (BLA_ALLHPC, ALLHPC_BLA, BLA_MTL, MTL_BLA, etc.), average the already-averaged spectra across the matching directed source pairs within each patient and condition.
> 4. Compute plotted band means or contrasts from the resulting directed composite spectrum.

---

## 5. Encoding vs Retrieval Consistency Check

### Things That ARE Consistent

| Aspect | Encoding | Retrieval | Match? |
|---|---|---|---|
| **Trial averaging method** | `groupby().mean()` (arithmetic) | `groupby().mean()` (arithmetic) | YES |
| **Composite ROI method** | `np.nanmean(np.vstack(...), axis=0)` | `np.nanmean(np.vstack(...), axis=0)` | YES |
| **Band-mean method** | `spectrum[mask].mean()` | `spectrum[mask].mean()` | YES |
| **Cross-subject mean** | `mat.mean(axis=0)` | `mat.mean(axis=0)` | YES |
| **SD for spectral shading** | numpy default (ddof=0) | numpy default (ddof=0) | YES |
| **SEM for bar plots** | `std(ddof=1)/sqrt(N)` or seaborn `se` | `std(ddof=1)/sqrt(N)` or seaborn `se` | YES |
| **Power/Coherence frequency bands** | Theta 4-8, Slow gamma 30-55 | Theta 4-8, Slow gamma 30-55 | YES |
| **PAC frequency band** | Slow gamma 35-50 | Slow gamma 35-50 | YES |
| **PAC baseline correction** | In-script: post - pre | In-script: post - pre | YES |
| **Power/Coherence baseline correction** | Pre-computed diff_Freq columns | Pre-computed diff_Freq columns | YES |
| **Coherence region pair handling** | Alphabetically sorted (symmetric) | Alphabetically sorted (symmetric) | YES |
| **PAC region pair handling** | Direction preserved (not sorted) | Direction preserved (not sorted) | YES |
| **PAC directional composites** | BLA_ALLHPC and ALLHPC_BLA kept separate | BLA_ALLHPC and ALLHPC_BLA kept separate | YES |
| **Composite ROI names (Power)** | ALLHPC, MTL | ALLHPC, MTL | YES |
| **Composite ROI names (Coherence)** | BLA_ALLHPC, BLA_MTL, ALLHPC_EC, ALLHPC_PRC | BLA_ALLHPC, BLA_MTL, ALLHPC_EC, ALLHPC_PRC | YES |
| **Composite ROI names (PAC)** | BLA_ALLHPC, BLA_MTL + directional group composites | BLA_ALLHPC, BLA_MTL + directional group composites | YES |
| **PAC data structure keys** | post_all, post_stim, diff_stim, etc. | post_all, post_stim, diff_stim, etc. | YES |
| **Power/Coherence data structure keys** | group_all_power, stim_power, bc_stim, etc. | group_all_power, stim_power, bc_stim, etc. | YES |

### Differences to Be Aware Of

#### 1. "Overall" Spectral Plot: Memory-Collapsed vs Straight Trial Mean
- **Encoding BLAES** (Power & Coherence): `use_memory_collapsed_overall = False`. The overall spectral plot uses `group_all_power`, which is a straight mean of ALL trials. Conditions with more trials have proportionally more influence.
- **Encoding AMME** (Power & Coherence): `use_memory_collapsed_overall = True`.
- **Retrieval (both cohorts, Power & Coherence)**: `use_memory_collapsed_overall = True`. The overall spectral plot averages the 4 condition cells (stim-rem, stim-forg, nostim-rem, nostim-forg) equally, regardless of trial count.
- **PAC (both phases)**: Uses `post_all` (straight trial mean) — no memory-collapsed option exists.
- **Impact**: For encoding BLAES Power/Coherence, the "overall" spectral line weights by trial count. For everything else, the 4 conditions are equally weighted. This could matter if trial counts are unbalanced across conditions.

#### 2. Subject/Region Exclusions Differ Between Phases
- Encoding and retrieval apply **different exclusion lists** (see sections 2b, 3b, 4c). This is expected since exclusion criteria are phase-specific (e.g., a noisy electrode during encoding may be fine during retrieval, or vice versa).

#### 3. Memory Condition Assignment Differs by Phase
- **Encoding**: Memory is "subsequent memory" — whether the item encoded during this trial was later remembered at retrieval. For BLAES, this requires a phase-3 lookup.
- **Retrieval**: Memory is "retrieval accuracy" — whether the participant correctly identified an old item as old (hit) during this trial. Only old-item trials are used.
- This is a fundamental design difference (subsequent memory effect vs retrieval success), not an inconsistency.

#### 4. PAC Slow Gamma Band Differs from Power/Coherence
- **Power & Coherence**: Slow gamma = 30-55 Hz
- **PAC**: Slow gamma = 35-50 Hz
- This is a narrower band for PAC. Whether this is intentional or an inconsistency depends on the scientific rationale.

#### 5. PAC Retrieval Outputs All 3 Subgroups; Encoding Outputs Only "All"
- `combined_retrieval_pac.py` generates outputs for BLAES, AMME, and All subgroups.
- `combined_encoding_pac.py` generates outputs only for the combined "All" group.
- Power and Coherence generate all subgroups for both phases.

#### 6. No Formal Statistical Tests in Any Script
- All six scripts are visualization-only. Statistical modeling is performed elsewhere using the exported MLMR CSVs (Power/Coherence) or the summary tables (PAC).
