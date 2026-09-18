#!/usr/bin/env python
"""
Permutation-balanced visualization for unbalanced memory conditions.

For subjects with fewer than 10 remembered or forgotten trials overall within a
phase/metric, this script balances remembered vs forgotten spectra by repeated
subsampling of the larger memory condition within each stimulation condition.
The smaller memory condition is kept intact on every iteration.

Outputs:
    outputs/permutation_testing_unbalanced_memory_conditions/
"""

# --- AMME_BLAES sys.path bootstrap ---
import sys as _sys
from pathlib import Path as _Path
_root = _Path(__file__).resolve().parent.parent
for _p in (_root, _root / 'encoding', _root / 'retrieval', _root / 'endogenous_memory',
          _root / 'regressions', _root / 'behavioral', _root / 'balanced_memory'):
    if str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))
# --- end bootstrap ---

from __future__ import annotations

import glob
import os
import re
import sys
import warnings
from pathlib import Path

_ALLHPC_RE = re.compile(r"(?<![A-Za-z])ALLHPC(?![A-Za-z])")
_HPC_RE = re.compile(r"(?<![A-Za-z])HPC(?![A-Za-z])")
_MARKER = "\x00__MACRO_HPC__\x00"


def pretty_in_text(text):
    if text is None:
        return text
    s = str(text)
    s = _ALLHPC_RE.sub(_MARKER, s)
    s = _HPC_RE.sub("SUB", s)
    s = s.replace(_MARKER, "HPC")
    return s

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from combined_encoding_power import (
    AMME_ENCODING_REGION_EXCLUSIONS as POWER_AMME_ENCODING_EXCLUSIONS,
    BLAES_ENCODING_REGION_EXCLUSIONS as POWER_BLAES_ENCODING_EXCLUSIONS,
    apply_subject_region_exclusions as apply_power_encoding_exclusions,
    fix_region as fix_power_encoding_region,
    normalize_image_name,
    normalize_memory_condition_blaes,
    normalize_response_blaes,
    normalize_trial_type_amme as normalize_power_encoding_trial_type_amme,
)
from combined_retrieval_power import (
    AMME_RETRIEVAL_REGION_EXCLUSIONS as POWER_AMME_RETRIEVAL_EXCLUSIONS,
    add_memory_condition_blaes as add_power_retrieval_memory_condition_blaes,
    apply_subject_region_exclusions as apply_power_retrieval_exclusions,
    fix_region as fix_power_retrieval_region,
    normalize_stimulation_blaes as normalize_power_retrieval_stimulation_blaes,
    normalize_trial_type_amme as normalize_power_retrieval_trial_type_amme,
)
from combined_encoding_coherence import (
    AMME_ENCODING_REGION_EXCLUSIONS as COH_AMME_ENCODING_EXCLUSIONS,
    BLAES_ENCODING_REGION_EXCLUSIONS as COH_BLAES_ENCODING_EXCLUSIONS,
    apply_subject_region_exclusions as apply_coherence_encoding_exclusions,
    fix_region as fix_coherence_encoding_region,
)
from combined_retrieval_coherence import (
    AMME_RETRIEVAL_REGION_EXCLUSIONS as COH_AMME_RETRIEVAL_EXCLUSIONS,
    add_memory_condition_blaes as add_coherence_retrieval_memory_condition_blaes,
    apply_subject_region_exclusions as apply_coherence_retrieval_exclusions,
    fix_region as fix_coherence_retrieval_region,
    normalize_stimulation_blaes as normalize_coherence_retrieval_stimulation_blaes,
    normalize_trial_type_amme as normalize_coherence_retrieval_trial_type_amme,
)
from to_combine.BLAES_Group_PAC_analyses_from_matlab_encoding import (
    ENCODING_SUBJECT_REGION_EXCLUSIONS as PAC_ENCODING_SUBJECT_REGION_EXCLUSIONS,
    PAC_FILES as ENCODING_PAC_FILES,
    apply_subject_region_exclusions as apply_pac_encoding_exclusions,
    clean_region_label as clean_pac_encoding_region_label,
    has_excluded_region_token as pac_has_excluded_region_token,
    is_same_region_comparison as pac_is_same_region_comparison,
    normalize_trial_type_amme as normalize_pac_encoding_trial_type_amme,
)
from to_combine.BLAES_Group_PAC_analyses_from_matlab_retrieval import (
    PAC_FILES as RETRIEVAL_PAC_FILES,
    add_memory_condition_blaes as add_pac_retrieval_memory_condition_blaes,
    clean_region_label as clean_pac_retrieval_region_label,
    has_excluded_region_token as pac_retrieval_has_excluded_region_token,
    is_same_region_comparison as pac_retrieval_is_same_region_comparison,
    normalize_stimulation_blaes as normalize_pac_retrieval_stimulation_blaes,
    normalize_trial_type_amme as normalize_pac_retrieval_trial_type_amme,
)

warnings.filterwarnings('ignore')


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = SCRIPT_DIR / 'outputs' / 'permutation_testing_unbalanced_memory_conditions'

POWER_BANDS = {
    'Theta': (4, 8),
    'Slow gamma': (30, 55),
}
PAC_BANDS = {
    'Slow gamma': (35, 50),
}

LOW_TRIAL_THRESHOLD = 10
N_ITERATIONS = 1000
MAX_EXAMPLE_SUBJECTS = 3
MAX_STABILITY_CURVES = 120
BASE_RANDOM_SEED = 20260331

STIM_ORDER = ['nostim', 'stim']
MEMORY_ORDER = ['remembered', 'forgotten']
STIM_LABELS = {'nostim': 'No Stim', 'stim': 'Stim'}
MEMORY_LABELS = {'remembered': 'Remembered', 'forgotten': 'Forgotten'}
STIM_COLORS = {'nostim': '#1f77b4', 'stim': '#d62728'}
MEMORY_COLORS = {'remembered': '#7B2D8E', 'forgotten': '#DAA520'}


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def finalize_figure(fig):
    plt.close(fig)


def slugify(value: str) -> str:
    return ''.join(ch.lower() if ch.isalnum() else '_' for ch in str(value)).strip('_')


def sorted_freq_cols(df: pd.DataFrame, prefix: str) -> list[str]:
    cols = [col for col in df.columns if col.startswith(prefix)]
    return sorted(cols, key=lambda col: float(col.split(prefix, 1)[1]))


def freq_values(freq_cols: list[str], prefix: str) -> np.ndarray:
    return np.array([float(col.split(prefix, 1)[1]) for col in freq_cols], dtype=float)


def sem(matrix: np.ndarray) -> np.ndarray:
    if matrix.shape[0] <= 1:
        return np.zeros(matrix.shape[1], dtype=float)
    return np.nanstd(matrix, axis=0, ddof=1) / np.sqrt(matrix.shape[0])


def compute_band_mean(curve: np.ndarray, freqs: np.ndarray, band_range: tuple[float, float]) -> float:
    mask = (freqs >= band_range[0]) & (freqs <= band_range[1])
    if not mask.any():
        return np.nan
    return float(np.nanmean(curve[mask]))


def build_trial_id(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    id_cols = [col for col in ['trialIdx', 'number', 'stimulus_code', 'start_time'] if col in df.columns]
    image_col = next((col for col in ['imagename', 'full_im_name', 'full_img_name'] if col in df.columns), None)
    if image_col is not None:
        id_cols.append(image_col)
    if not id_cols:
        df['trial_id'] = np.arange(len(df)).astype(str)
        return df
    parts = []
    for col in id_cols:
        values = df[col].fillna('NA').astype(str).str.strip()
        if 'image' in col.lower():
            values = values.str.replace('\\', '/', regex=False).str.lower().str.split('/').str[-1]
        parts.append(values)
    df['trial_id'] = pd.concat(parts, axis=1).agg('|'.join, axis=1)
    return df


def add_group_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['Group'] = np.where(df['Patient'].astype(str).str.startswith('amyg'), 'AMME', 'BLAES')
    return df


def standardize_trial_frame(
    df: pd.DataFrame,
    *,
    freq_cols: list[str],
    trial_type_col: str,
    region_col: str = 'Region',
) -> pd.DataFrame:
    keep_cols = ['Patient', region_col, 'trial_id', trial_type_col, 'memory_cond', 'Group'] + freq_cols
    trial_df = df[keep_cols].copy()
    trial_df = trial_df.rename(columns={region_col: 'Region', trial_type_col: 'trial_type'})
    trial_df['Patient'] = trial_df['Patient'].astype(str)
    trial_df['Region'] = trial_df['Region'].astype(str)
    trial_df['trial_type'] = trial_df['trial_type'].astype(str)
    trial_df['memory_cond'] = trial_df['memory_cond'].astype(str)
    return trial_df


def build_blaes_encoding_lookup(file_pattern: str) -> pd.DataFrame:
    lookup_parts = []
    for file_path in glob.glob(file_pattern):
        df = pd.read_csv(file_path)
        required = {'Patient', 'stimulus_code', 'stimulation', 'response', 'trial_type'}
        if not required.issubset(df.columns):
            continue
        img_col = next((col for col in ['full_im_name', 'full_img_name', 'imagename'] if col in df.columns), None)
        if img_col is None:
            continue
        sub = df[['Patient', 'stimulus_code', img_col, 'stimulation', 'response', 'trial_type']].copy()
        sub['img_key'] = sub[img_col].apply(normalize_image_name)
        sub['memory_condition'] = sub['trial_type'].apply(normalize_memory_condition_blaes)
        sub['normalized_response'] = sub['response'].apply(normalize_response_blaes)
        sub['yes_or_no'] = np.where(
            sub['normalized_response'] == 'old',
            'yes',
            np.where(sub['normalized_response'] == 'new', 'no', np.nan),
        )
        sub['memory_cond'] = np.select(
            [
                (sub['memory_condition'] == 'old') & (sub['normalized_response'] == 'old'),
                (sub['memory_condition'] == 'old') & (sub['normalized_response'] == 'new'),
            ],
            ['remembered', 'forgotten'],
            default=np.nan,
        )
        sub = sub.dropna(subset=['Patient', 'stimulus_code', 'img_key'])
        sub = sub[sub['memory_cond'].isin(['remembered', 'forgotten'])]
        sub = sub.drop_duplicates(subset=['Patient', 'stimulus_code', 'img_key'])
        lookup_parts.append(sub[['Patient', 'stimulus_code', 'img_key', 'stimulation', 'yes_or_no']])
    if not lookup_parts:
        return pd.DataFrame(columns=['Patient', 'stimulus_code', 'img_key', 'stimulation', 'yes_or_no'])
    return pd.concat(lookup_parts, ignore_index=True).drop_duplicates(
        subset=['Patient', 'stimulus_code', 'img_key']
    )


def load_power_encoding_trials() -> tuple[pd.DataFrame, np.ndarray]:
    blaes_project = '/Users/martinahollearn/Library/CloudStorage/Box-Box/InmanLab/BLAES_data/dissertation/LFP_analyses'
    amme_project = '/Users/martinahollearn/Library/CloudStorage/Box-Box/InmanLab/AMME_Data_Emory/AMME_Data/LFP_analyses_Martina'
    blaes_files = sorted(glob.glob(os.path.join(blaes_project, 'Results_CSVOutput', '*phase1*Power*.csv')))
    amme_files = sorted(glob.glob(os.path.join(amme_project, 'Results_CSVOutput', 'Phase1', '*Power*.csv')))
    phase3_lookup = build_blaes_encoding_lookup(os.path.join(blaes_project, 'Results_CSVOutput', '*phase3*Power*.csv'))
    phase3_lookup['stimulus_code'] = pd.to_numeric(phase3_lookup['stimulus_code'], errors='coerce')

    frames = []
    freqs = None

    for file_path in blaes_files:
        df = fix_power_encoding_region(pd.read_csv(file_path))
        if 'Patient' not in df.columns or 'Region' not in df.columns:
            continue
        if 'imagename' not in df.columns or 'stimulus_code' not in df.columns:
            continue
        df['img_key'] = df['imagename'].apply(normalize_image_name)
        df['stimulus_code'] = pd.to_numeric(df['stimulus_code'], errors='coerce')
        df = df.merge(phase3_lookup, on=['Patient', 'stimulus_code', 'img_key'], how='left')
        df['stimulation'] = pd.to_numeric(df['stimulation'], errors='coerce')
        df = df[df['stimulation'].isin([0, 1])].copy()
        df = apply_power_encoding_exclusions(df, POWER_BLAES_ENCODING_EXCLUSIONS)
        df = df[df['yes_or_no'].isin(['yes', 'no'])].copy()
        if df.empty:
            continue
        df['memory_cond'] = np.where(df['yes_or_no'] == 'yes', 'remembered', 'forgotten')
        df['trial_type'] = np.where(df['stimulation'] == 1, 'stim', 'nostim')
        df = build_trial_id(df)
        df = add_group_column(df)
        freq_cols = sorted_freq_cols(df, 'post_Freq_')
        if not freq_cols:
            continue
        if freqs is None:
            freqs = freq_values(freq_cols, 'post_Freq_')
        frames.append(standardize_trial_frame(df, freq_cols=freq_cols, trial_type_col='trial_type'))

    for file_path in amme_files:
        df = pd.read_csv(file_path)
        if 'Patient' not in df.columns or 'Region' not in df.columns:
            continue
        df = apply_power_encoding_exclusions(df, POWER_AMME_ENCODING_EXCLUSIONS)
        if 'test_trial_type' not in df.columns or 'test_yes_or_no' not in df.columns:
            continue
        df['test_trial_type'] = df['test_trial_type'].apply(normalize_power_encoding_trial_type_amme)
        df = df[df['test_trial_type'].notnull()].copy()
        df = df[df['test_yes_or_no'].isin(['yes', 'no'])].copy()
        if df.empty:
            continue
        df['memory_cond'] = np.where(df['test_yes_or_no'] == 'yes', 'remembered', 'forgotten')
        df = build_trial_id(df)
        df = add_group_column(df)
        freq_cols = sorted_freq_cols(df, 'post_Freq_')
        if not freq_cols:
            continue
        if freqs is None:
            freqs = freq_values(freq_cols, 'post_Freq_')
        frames.append(standardize_trial_frame(df, freq_cols=freq_cols, trial_type_col='test_trial_type'))

    return pd.concat(frames, ignore_index=True), freqs


def load_power_retrieval_trials() -> tuple[pd.DataFrame, np.ndarray]:
    blaes_project = '/Users/martinahollearn/Library/CloudStorage/Box-Box/InmanLab/BLAES_data/dissertation/LFP_analyses'
    amme_project = '/Users/martinahollearn/Library/CloudStorage/Box-Box/InmanLab/AMME_Data_Emory/AMME_Data/LFP_analyses_Martina'
    blaes_files = sorted(glob.glob(os.path.join(blaes_project, 'Results_CSVOutput', '*phase3*Power*.csv')))
    amme_files = sorted(glob.glob(os.path.join(amme_project, 'Results_CSVOutput', 'Phase2', '*Power*.csv')))

    frames = []
    freqs = None

    for file_path in blaes_files:
        df = fix_power_retrieval_region(pd.read_csv(file_path))
        if 'Patient' not in df.columns or 'Region' not in df.columns or 'stimulation' not in df.columns:
            continue
        if 'trial_type' in df.columns:
            df['trial_type_raw'] = df['trial_type']
        df['trial_type'] = df['stimulation'].apply(normalize_power_retrieval_stimulation_blaes)
        df = df[df['trial_type'].notnull()].copy()
        if df.empty or df['trial_type'].nunique() < 2:
            continue
        df = add_power_retrieval_memory_condition_blaes(df)
        if df.empty:
            continue
        df = build_trial_id(df)
        df = add_group_column(df)
        freq_cols = sorted_freq_cols(df, 'post_Freq_')
        if not freq_cols:
            continue
        if freqs is None:
            freqs = freq_values(freq_cols, 'post_Freq_')
        frames.append(standardize_trial_frame(df, freq_cols=freq_cols, trial_type_col='trial_type'))

    for file_path in amme_files:
        df = pd.read_csv(file_path)
        if 'Patient' not in df.columns or 'Region' not in df.columns or 'trial_type' not in df.columns:
            continue
        df = apply_power_retrieval_exclusions(df, POWER_AMME_RETRIEVAL_EXCLUSIONS)
        df['trial_type'] = df['trial_type'].apply(normalize_power_retrieval_trial_type_amme)
        df = df[df['trial_type'].notnull()].copy()
        if df.empty:
            continue
        df = df[df['yes_or_no'].isin(['yes', 'no'])].copy()
        if df.empty:
            continue
        df['memory_cond'] = np.where(df['yes_or_no'] == 'yes', 'remembered', 'forgotten')
        df = build_trial_id(df)
        df = add_group_column(df)
        freq_cols = sorted_freq_cols(df, 'post_Freq_')
        if not freq_cols:
            continue
        if freqs is None:
            freqs = freq_values(freq_cols, 'post_Freq_')
        frames.append(standardize_trial_frame(df, freq_cols=freq_cols, trial_type_col='trial_type'))

    return pd.concat(frames, ignore_index=True), freqs


def load_coherence_encoding_trials() -> tuple[pd.DataFrame, np.ndarray]:
    blaes_project = '/Users/martinahollearn/Library/CloudStorage/Box-Box/InmanLab/BLAES_data/dissertation/LFP_analyses'
    amme_project = '/Users/martinahollearn/Library/CloudStorage/Box-Box/InmanLab/AMME_Data_Emory/AMME_Data/LFP_analyses_Martina'
    blaes_files = sorted(glob.glob(os.path.join(blaes_project, 'Results_CSVOutput', '*phase1*Coherency*.csv')))
    amme_files = sorted(glob.glob(os.path.join(amme_project, 'Results_CSVOutput', 'Phase1', '*Coherency*.csv')))
    phase3_lookup = build_blaes_encoding_lookup(os.path.join(blaes_project, 'Results_CSVOutput', '*phase3*Coherency*.csv'))
    phase3_lookup['stimulus_code'] = pd.to_numeric(phase3_lookup['stimulus_code'], errors='coerce')

    frames = []
    freqs = None

    for file_path in blaes_files:
        df = fix_coherence_encoding_region(pd.read_csv(file_path))
        if 'Patient' not in df.columns or 'Region' not in df.columns:
            continue
        if 'imagename' not in df.columns or 'stimulus_code' not in df.columns:
            continue
        df['img_key'] = df['imagename'].apply(normalize_image_name)
        df['stimulus_code'] = pd.to_numeric(df['stimulus_code'], errors='coerce')
        df = df.merge(phase3_lookup, on=['Patient', 'stimulus_code', 'img_key'], how='left')
        df['stimulation'] = pd.to_numeric(df['stimulation'], errors='coerce')
        df = df[df['stimulation'].isin([0, 1])].copy()
        df = apply_coherence_encoding_exclusions(df, COH_BLAES_ENCODING_EXCLUSIONS)
        df = df[df['yes_or_no'].isin(['yes', 'no'])].copy()
        if df.empty:
            continue
        df['memory_cond'] = np.where(df['yes_or_no'] == 'yes', 'remembered', 'forgotten')
        df['trial_type'] = np.where(df['stimulation'] == 1, 'stim', 'nostim')
        df = build_trial_id(df)
        df = add_group_column(df)
        freq_cols = sorted_freq_cols(df, 'post_Freq_')
        if not freq_cols:
            continue
        if freqs is None:
            freqs = freq_values(freq_cols, 'post_Freq_')
        frames.append(standardize_trial_frame(df, freq_cols=freq_cols, trial_type_col='trial_type'))

    for file_path in amme_files:
        df = fix_coherence_encoding_region(pd.read_csv(file_path))
        if 'Patient' not in df.columns or 'Region' not in df.columns:
            continue
        df = apply_coherence_encoding_exclusions(df, COH_AMME_ENCODING_EXCLUSIONS)
        if 'test_trial_type' not in df.columns or 'test_yes_or_no' not in df.columns:
            continue
        df['test_trial_type'] = df['test_trial_type'].apply(normalize_power_encoding_trial_type_amme)
        df = df[df['test_trial_type'].notnull()].copy()
        df = df[df['test_yes_or_no'].isin(['yes', 'no'])].copy()
        if df.empty:
            continue
        df['memory_cond'] = np.where(df['test_yes_or_no'] == 'yes', 'remembered', 'forgotten')
        df = build_trial_id(df)
        df = add_group_column(df)
        freq_cols = sorted_freq_cols(df, 'post_Freq_')
        if not freq_cols:
            continue
        if freqs is None:
            freqs = freq_values(freq_cols, 'post_Freq_')
        frames.append(standardize_trial_frame(df, freq_cols=freq_cols, trial_type_col='test_trial_type'))

    return pd.concat(frames, ignore_index=True), freqs


def load_coherence_retrieval_trials() -> tuple[pd.DataFrame, np.ndarray]:
    blaes_project = '/Users/martinahollearn/Library/CloudStorage/Box-Box/InmanLab/BLAES_data/dissertation/LFP_analyses'
    amme_project = '/Users/martinahollearn/Library/CloudStorage/Box-Box/InmanLab/AMME_Data_Emory/AMME_Data/LFP_analyses_Martina'
    blaes_files = sorted(glob.glob(os.path.join(blaes_project, 'Results_CSVOutput', '*phase3*Coherency*.csv')))
    amme_files = sorted(glob.glob(os.path.join(amme_project, 'Results_CSVOutput', 'Phase2', '*Coherency*.csv')))

    frames = []
    freqs = None

    for file_path in blaes_files:
        df = fix_coherence_retrieval_region(pd.read_csv(file_path))
        if 'Patient' not in df.columns or 'Region' not in df.columns or 'stimulation' not in df.columns:
            continue
        if 'trial_type' in df.columns:
            df['trial_type_raw'] = df['trial_type']
        df['trial_type'] = df['stimulation'].apply(normalize_coherence_retrieval_stimulation_blaes)
        df = df[df['trial_type'].notnull()].copy()
        if df.empty or df['trial_type'].nunique() < 2:
            continue
        df = add_coherence_retrieval_memory_condition_blaes(df)
        if df.empty:
            continue
        df = build_trial_id(df)
        df = add_group_column(df)
        freq_cols = sorted_freq_cols(df, 'post_Freq_')
        if not freq_cols:
            continue
        if freqs is None:
            freqs = freq_values(freq_cols, 'post_Freq_')
        frames.append(standardize_trial_frame(df, freq_cols=freq_cols, trial_type_col='trial_type'))

    for file_path in amme_files:
        df = fix_coherence_retrieval_region(pd.read_csv(file_path))
        if 'Patient' not in df.columns or 'Region' not in df.columns or 'trial_type' not in df.columns:
            continue
        df = apply_coherence_retrieval_exclusions(df, COH_AMME_RETRIEVAL_EXCLUSIONS)
        df['trial_type'] = df['trial_type'].apply(normalize_coherence_retrieval_trial_type_amme)
        df = df[df['trial_type'].notnull()].copy()
        if df.empty:
            continue
        df = df[df['yes_or_no'].isin(['yes', 'no'])].copy()
        if df.empty:
            continue
        df['memory_cond'] = np.where(df['yes_or_no'] == 'yes', 'remembered', 'forgotten')
        df = build_trial_id(df)
        df = add_group_column(df)
        freq_cols = sorted_freq_cols(df, 'post_Freq_')
        if not freq_cols:
            continue
        if freqs is None:
            freqs = freq_values(freq_cols, 'post_Freq_')
        frames.append(standardize_trial_frame(df, freq_cols=freq_cols, trial_type_col='trial_type'))

    return pd.concat(frames, ignore_index=True), freqs


def load_pac_encoding_trials() -> tuple[pd.DataFrame, np.ndarray]:
    frames = []
    freqs = None

    for file_path in ENCODING_PAC_FILES:
        df = pd.read_csv(file_path)
        required = {'Patient', 'Region', 'stimulation'}
        if not required.issubset(df.columns):
            continue
        df = df.copy()
        df['Region'] = df['Region'].map(clean_pac_encoding_region_label)
        df['stimulation'] = pd.to_numeric(df['stimulation'], errors='coerce')
        df = df[df['stimulation'].isin([0, 1])].copy()
        if 'ret_response' in df.columns:
            df['memory_cond'] = df['ret_response'].map({'Old': 'remembered', 'New': 'forgotten'})
            df = df[df['memory_cond'].isin(['remembered', 'forgotten'])].copy()
        elif 'test_yes_or_no' in df.columns:
            if 'test_trial_type' in df.columns:
                df['test_trial_type'] = df['test_trial_type'].apply(normalize_pac_encoding_trial_type_amme)
            df = df[df['test_yes_or_no'].isin(['yes', 'no'])].copy()
            df['memory_cond'] = np.where(df['test_yes_or_no'] == 'yes', 'remembered', 'forgotten')
        else:
            continue
        df = df[~df['Region'].map(pac_is_same_region_comparison)].copy()
        df = df[~df['Region'].map(pac_has_excluded_region_token)].copy()
        df = apply_pac_encoding_exclusions(df, PAC_ENCODING_SUBJECT_REGION_EXCLUSIONS)
        if df.empty:
            continue
        df['trial_type'] = np.where(df['stimulation'] == 1, 'stim', 'nostim')
        df = build_trial_id(df)
        df = add_group_column(df)
        freq_cols = sorted_freq_cols(df, 'post_Freq_')
        if not freq_cols:
            continue
        if freqs is None:
            freqs = freq_values(freq_cols, 'post_Freq_')
        frames.append(standardize_trial_frame(df, freq_cols=freq_cols, trial_type_col='trial_type'))

    return pd.concat(frames, ignore_index=True), freqs


def load_pac_retrieval_trials() -> tuple[pd.DataFrame, np.ndarray]:
    frames = []
    freqs = None

    for file_path in RETRIEVAL_PAC_FILES:
        df = pd.read_csv(file_path)
        required = {'Patient', 'Region'}
        if not required.issubset(df.columns):
            continue
        df = df.copy()
        df['Region'] = df['Region'].map(clean_pac_retrieval_region_label)

        if 'stimulation' in df.columns and 'response' in df.columns:
            if 'trial_type' in df.columns:
                df['trial_type_raw'] = df['trial_type']
            df['trial_type'] = df['stimulation'].apply(normalize_pac_retrieval_stimulation_blaes)
            df = df[df['trial_type'].notnull()].copy()
            if df.empty or df['trial_type'].nunique() < 2:
                continue
            df = add_pac_retrieval_memory_condition_blaes(df)
            if df.empty:
                continue
        elif 'trial_type' in df.columns and 'yes_or_no' in df.columns:
            df['trial_type'] = df['trial_type'].apply(normalize_pac_retrieval_trial_type_amme)
            df = df[df['trial_type'].notnull()].copy()
            if df.empty or df['trial_type'].nunique() < 2:
                continue
            df = df[df['yes_or_no'].isin(['yes', 'no'])].copy()
            if df.empty:
                continue
            df['memory_cond'] = np.where(df['yes_or_no'] == 'yes', 'remembered', 'forgotten')
        else:
            continue

        df = df[~df['Region'].map(pac_retrieval_is_same_region_comparison)].copy()
        df = df[~df['Region'].map(pac_retrieval_has_excluded_region_token)].copy()
        if df.empty:
            continue
        df = build_trial_id(df)
        df = add_group_column(df)
        freq_cols = sorted_freq_cols(df, 'post_Freq_')
        if not freq_cols:
            continue
        if freqs is None:
            freqs = freq_values(freq_cols, 'post_Freq_')
        frames.append(standardize_trial_frame(df, freq_cols=freq_cols, trial_type_col='trial_type'))

    return pd.concat(frames, ignore_index=True), freqs


def compute_subject_selection(df: pd.DataFrame) -> pd.DataFrame:
    trial_df = df[['Patient', 'Group', 'trial_id', 'memory_cond']].drop_duplicates().copy()
    counts = (
        trial_df.groupby(['Patient', 'Group', 'memory_cond'])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=MEMORY_ORDER, fill_value=0)
        .reset_index()
    )
    counts['n_min'] = counts[['remembered', 'forgotten']].min(axis=1)
    counts['selected_for_permutation'] = (counts['n_min'] > 0) & (counts['n_min'] < LOW_TRIAL_THRESHOLD)
    counts['missing_memory_condition'] = counts['n_min'] == 0
    counts = counts.sort_values(['selected_for_permutation', 'n_min', 'Patient'], ascending=[False, True, True])
    return counts


def choose_example_subjects(region_df: pd.DataFrame) -> set[str]:
    counts = (
        region_df.groupby(['Patient', 'trial_type', 'memory_cond'])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=MEMORY_ORDER, fill_value=0)
        .reset_index()
    )
    counts['n_min'] = counts[['remembered', 'forgotten']].min(axis=1)
    counts = counts[counts['n_min'] > 0].sort_values(['n_min', 'Patient', 'trial_type'])
    ordered_patients = []
    for patient in counts['Patient']:
        if patient not in ordered_patients:
            ordered_patients.append(patient)
    return set(ordered_patients[:MAX_EXAMPLE_SUBJECTS])


def sample_mean_curves(matrix: np.ndarray, n_take: int, n_iterations: int, rng: np.random.Generator) -> np.ndarray:
    if matrix.shape[0] == 0:
        raise ValueError('Cannot sample from an empty matrix.')
    if matrix.shape[0] == n_take:
        return np.repeat(matrix.mean(axis=0, keepdims=True), n_iterations, axis=0)
    curves = np.empty((n_iterations, matrix.shape[1]), dtype=float)
    for idx in range(n_iterations):
        sample_idx = rng.choice(matrix.shape[0], size=n_take, replace=False)
        curves[idx] = matrix[sample_idx].mean(axis=0)
    return curves


def compute_subject_balanced_curves(
    remembered_matrix: np.ndarray,
    forgotten_matrix: np.ndarray,
    *,
    freqs: np.ndarray,
    band_ranges: dict[str, tuple[float, float]],
    rng: np.random.Generator,
    store_samples: bool = False,
) -> dict[str, object] | None:
    n_rem = remembered_matrix.shape[0]
    n_forg = forgotten_matrix.shape[0]
    n_min = min(n_rem, n_forg)
    if n_min == 0:
        return None

    remembered_mean = remembered_matrix.mean(axis=0)
    forgotten_mean = forgotten_matrix.mean(axis=0)
    remembered_sd = np.zeros_like(remembered_mean)
    forgotten_sd = np.zeros_like(forgotten_mean)
    diff_samples = None
    sample_payload = None
    resampled_condition = 'none'

    if n_rem < n_forg:
        resampled_condition = 'forgotten'
        forgotten_samples = sample_mean_curves(forgotten_matrix, n_min, N_ITERATIONS, rng)
        forgotten_mean = forgotten_samples.mean(axis=0)
        forgotten_sd = np.nanstd(forgotten_samples, axis=0, ddof=1)
        diff_samples = remembered_mean[None, :] - forgotten_samples
        if store_samples:
            sample_payload = {
                'memory_cond': 'forgotten',
                'curves': forgotten_samples,
                'mean_curve': forgotten_mean,
                'n_small': n_rem,
                'n_large': n_forg,
                'n_min': n_min,
            }
    elif n_forg < n_rem:
        resampled_condition = 'remembered'
        remembered_samples = sample_mean_curves(remembered_matrix, n_min, N_ITERATIONS, rng)
        remembered_mean = remembered_samples.mean(axis=0)
        remembered_sd = np.nanstd(remembered_samples, axis=0, ddof=1)
        diff_samples = remembered_samples - forgotten_mean[None, :]
        if store_samples:
            sample_payload = {
                'memory_cond': 'remembered',
                'curves': remembered_samples,
                'mean_curve': remembered_mean,
                'n_small': n_forg,
                'n_large': n_rem,
                'n_min': n_min,
            }
    else:
        diff_samples = np.repeat((remembered_mean - forgotten_mean)[None, :], N_ITERATIONS, axis=0)

    diff_mean = diff_samples.mean(axis=0)
    diff_sd = np.nanstd(diff_samples, axis=0, ddof=1) if diff_samples.shape[0] > 1 else np.zeros_like(diff_mean)

    band_rows = []
    for band, band_range in band_ranges.items():
        if not ((freqs >= band_range[0]) & (freqs <= band_range[1])).any():
            continue
        band_rows.extend([
            {
                'memory_cond': 'remembered',
                'band': band,
                'band_mean': compute_band_mean(remembered_mean, freqs, band_range),
                'resample_sd': float(np.nanmean(remembered_sd[(freqs >= band_range[0]) & (freqs <= band_range[1])])) if remembered_sd.size else 0.0,
            },
            {
                'memory_cond': 'forgotten',
                'band': band,
                'band_mean': compute_band_mean(forgotten_mean, freqs, band_range),
                'resample_sd': float(np.nanmean(forgotten_sd[(freqs >= band_range[0]) & (freqs <= band_range[1])])) if forgotten_sd.size else 0.0,
            },
        ])

    return {
        'remembered_mean': remembered_mean,
        'remembered_sd': remembered_sd,
        'forgotten_mean': forgotten_mean,
        'forgotten_sd': forgotten_sd,
        'diff_mean': diff_mean,
        'diff_sd': diff_sd,
        'n_remembered': n_rem,
        'n_forgotten': n_forg,
        'n_min': n_min,
        'resampled_condition': resampled_condition,
        'sample_payload': sample_payload,
        'band_rows': band_rows,
    }


def plot_main_curves(
    condition_subject_curves: dict[tuple[str, str], dict[str, np.ndarray]],
    freqs: np.ndarray,
    out_path: Path,
    title_prefix: str,
    ylabel: str,
):
    if not any(condition_subject_curves.get((stim, mem)) for stim in STIM_ORDER for mem in MEMORY_ORDER):
        return
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharex=True, sharey=True)
    for ax, memory_cond in zip(axes, MEMORY_ORDER):
        for trial_type in STIM_ORDER:
            subject_curves = condition_subject_curves.get((trial_type, memory_cond), {})
            if not subject_curves:
                continue
            matrix = np.vstack(list(subject_curves.values()))
            mean_curve = np.nanmean(matrix, axis=0)
            sem_curve = sem(matrix)
            label = f"{STIM_LABELS[trial_type]} (N={matrix.shape[0]})"
            color = STIM_COLORS[trial_type]
            ax.plot(freqs, mean_curve, color=color, linewidth=2.5, label=label)
            ax.fill_between(freqs, mean_curve - sem_curve, mean_curve + sem_curve, color=color, alpha=0.18)
        ax.set_title(MEMORY_LABELS[memory_cond], fontsize=15, fontweight='bold')
        ax.set_xlabel('Frequency (Hz)', fontsize=13, fontweight='bold')
        ax.axhline(0, color='#888888', linewidth=0.8, alpha=0.5)
        ax.tick_params(axis='both', labelsize=11)
    axes[0].set_ylabel(ylabel, fontsize=13, fontweight='bold')
    axes[1].legend(frameon=False, fontsize=10, loc='best')
    fig.suptitle(f'{title_prefix} Main Balanced Curves', fontsize=17, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    finalize_figure(fig)


def plot_difference_curves(
    diff_subject_curves: dict[str, dict[str, np.ndarray]],
    freqs: np.ndarray,
    out_path: Path,
    title_prefix: str,
    ylabel: str,
):
    if not any(diff_subject_curves.get(trial_type) for trial_type in STIM_ORDER):
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    for trial_type in STIM_ORDER:
        subject_curves = diff_subject_curves.get(trial_type, {})
        if not subject_curves:
            continue
        matrix = np.vstack(list(subject_curves.values()))
        mean_curve = np.nanmean(matrix, axis=0)
        sem_curve = sem(matrix)
        color = STIM_COLORS[trial_type]
        ax.plot(freqs, mean_curve, color=color, linewidth=2.5, label=f"{STIM_LABELS[trial_type]} (N={matrix.shape[0]})")
        ax.fill_between(freqs, mean_curve - sem_curve, mean_curve + sem_curve, color=color, alpha=0.18)
    ax.axhline(0, color='#555555', linewidth=1.0)
    ax.set_xlabel('Frequency (Hz)', fontsize=13, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=13, fontweight='bold')
    ax.set_title(f'{title_prefix} Difference Curves (Remembered - Forgotten)', fontsize=16, fontweight='bold')
    ax.tick_params(axis='both', labelsize=11)
    ax.legend(frameon=False, fontsize=10, loc='best')
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    finalize_figure(fig)


def plot_resampling_stability(
    stability_rows: list[dict[str, object]],
    freqs: np.ndarray,
    out_path: Path,
    title_prefix: str,
    ylabel: str,
):
    if not stability_rows:
        return
    n_rows = len(stability_rows)
    fig, axes = plt.subplots(n_rows, 1, figsize=(11, max(3.5 * n_rows, 4.5)), sharex=True, squeeze=False)
    for ax, row in zip(axes[:, 0], stability_rows):
        curves = np.asarray(row['curves'], dtype=float)
        if curves.shape[0] > MAX_STABILITY_CURVES:
            step = max(1, curves.shape[0] // MAX_STABILITY_CURVES)
            curves = curves[::step][:MAX_STABILITY_CURVES]
        color = STIM_COLORS[row['trial_type']]
        for curve in curves:
            ax.plot(freqs, curve, color=color, alpha=0.08, linewidth=0.8)
        ax.plot(freqs, row['mean_curve'], color=color, linewidth=2.5)
        ax.set_title(
            f"{row['Patient']} | {STIM_LABELS[row['trial_type']]} | Resampled {MEMORY_LABELS[row['memory_cond']]} "
            f"(n_small={row['n_small']}, n_large={row['n_large']}, n_min={row['n_min']})",
            fontsize=11,
            fontweight='bold',
        )
        ax.set_ylabel(ylabel, fontsize=10, fontweight='bold')
        ax.tick_params(axis='both', labelsize=10)
    axes[-1, 0].set_xlabel('Frequency (Hz)', fontsize=12, fontweight='bold')
    fig.suptitle(f'{title_prefix} Resampling Stability', fontsize=16, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    finalize_figure(fig)


def plot_band_summary(
    band_df: pd.DataFrame,
    out_path: Path,
    title_prefix: str,
    ylabel: str,
):
    if band_df.empty:
        return
    categories = [
        ('nostim', 'remembered'),
        ('nostim', 'forgotten'),
        ('stim', 'remembered'),
        ('stim', 'forgotten'),
    ]
    category_labels = [
        'No Stim\nRemembered',
        'No Stim\nForgotten',
        'Stim\nRemembered',
        'Stim\nForgotten',
    ]
    band_order = [band for band in ['Theta', 'Slow gamma'] if band in band_df['band'].unique()]
    fig, axes = plt.subplots(1, len(band_order), figsize=(6.5 * len(band_order), 5.5), squeeze=False)
    rng = np.random.default_rng(17)
    for ax, band in zip(axes[0], band_order):
        subset = band_df[band_df['band'] == band].copy()
        for idx, (trial_type, memory_cond) in enumerate(categories):
            cat_df = subset[(subset['trial_type'] == trial_type) & (subset['memory_cond'] == memory_cond)]
            if cat_df.empty:
                continue
            jitter = rng.uniform(-0.10, 0.10, len(cat_df))
            ax.scatter(
                np.full(len(cat_df), idx) + jitter,
                cat_df['band_mean'],
                color=STIM_COLORS[trial_type],
                alpha=0.65,
                s=36,
                edgecolors='none',
            )
            mean_value = cat_df['band_mean'].mean()
            sem_value = cat_df['band_mean'].sem() if len(cat_df) > 1 else 0.0
            ax.errorbar(idx, mean_value, yerr=sem_value, color='black', linewidth=1.5, capsize=4)
        ax.set_title(band, fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(categories)))
        ax.set_xticklabels(category_labels, fontsize=10)
        ax.tick_params(axis='y', labelsize=10)
        ax.axhline(0, color='#888888', linewidth=0.8, alpha=0.5)
    axes[0, 0].set_ylabel(ylabel, fontsize=12, fontweight='bold')
    fig.suptitle(f'{title_prefix} Band Summary', fontsize=16, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    finalize_figure(fig)


def plot_memory_percent_across_subjects(selection_df: pd.DataFrame, out_path: Path, title_prefix: str):
    if selection_df.empty:
        return
    plot_df = selection_df.copy()
    totals = plot_df['remembered'] + plot_df['forgotten']
    plot_df = plot_df[totals > 0].copy()
    if plot_df.empty:
        return
    plot_df['remembered_pct'] = plot_df['remembered'] / totals * 100.0
    plot_df['forgotten_pct'] = plot_df['forgotten'] / totals * 100.0

    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    categories = ['remembered_pct', 'forgotten_pct']
    labels = ['Remembered', 'Forgotten']
    rng = np.random.default_rng(29)

    for idx, (category, label) in enumerate(zip(categories, labels)):
        values = plot_df[category].to_numpy(dtype=float)
        mean_value = float(np.nanmean(values))
        sem_value = float(pd.Series(values).sem()) if len(values) > 1 else 0.0
        ax.bar(
            idx,
            mean_value,
            yerr=sem_value,
            color=MEMORY_COLORS[category.replace('_pct', '')],
            edgecolor='black',
            linewidth=1.2,
            width=0.5,
            capsize=4,
        )
        jitter = rng.uniform(-0.08, 0.08, len(values))
        ax.scatter(
            np.full(len(values), idx) + jitter,
            values,
            color='black',
            alpha=0.35,
            s=34,
        )
        ax.text(idx, mean_value + sem_value + 1.8, f'{mean_value:.1f}%', ha='center', va='bottom',
                fontsize=17, fontweight='bold')

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=18)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Mean % of Each Subject's Total Trials", fontsize=21)
    ax.set_title(f'{title_prefix} Across Subjects (±SEM)', fontsize=24, fontweight='bold')
    ax.tick_params(axis='y', labelsize=15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    finalize_figure(fig)


def plot_memory_percent_by_subject(selection_df: pd.DataFrame, out_path: Path, title_prefix: str):
    if selection_df.empty:
        return
    plot_df = selection_df.copy()
    totals = plot_df['remembered'] + plot_df['forgotten']
    plot_df = plot_df[totals > 0].copy()
    if plot_df.empty:
        return
    plot_df['remembered_pct'] = plot_df['remembered'] / totals * 100.0
    plot_df['forgotten_pct'] = plot_df['forgotten'] / totals * 100.0
    plot_df = plot_df.sort_values(['forgotten_pct', 'Patient']).reset_index(drop=True)

    x = np.arange(len(plot_df))
    fig, ax = plt.subplots(figsize=(max(14, len(plot_df) * 0.45), 7.5))
    ax.bar(
        x,
        plot_df['remembered_pct'],
        color=MEMORY_COLORS['remembered'],
        label='Remembered',
    )
    ax.bar(
        x,
        plot_df['forgotten_pct'],
        bottom=plot_df['remembered_pct'],
        color=MEMORY_COLORS['forgotten'],
        label='Forgotten',
    )
    ax.set_ylim(0, 100)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df['Patient'], rotation=90, fontsize=10)
    ax.set_ylabel("% of Each Subject's Total Trials", fontsize=18)
    ax.set_xlabel('Subject', fontsize=18)
    ax.set_title(f'{title_prefix} by Subject', fontsize=24, fontweight='bold')
    ax.tick_params(axis='y', labelsize=13)
    ax.legend(frameon=True, fontsize=14, loc='upper left')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    finalize_figure(fig)


def save_summary_text(
    selection_df: pd.DataFrame,
    region_stats_df: pd.DataFrame,
    out_path: Path,
    *,
    metric_label: str,
    phase_label: str,
    pac_theta_note: bool = False,
):
    selected = selection_df[selection_df['selected_for_permutation']].copy()
    missing = selection_df[selection_df['missing_memory_condition']].copy()
    lines = [
        f'{metric_label} {phase_label} permutation-balanced unbalanced memory summary',
        '=' * 80,
        f'Subject criterion: overall remembered or forgotten trial count < {LOW_TRIAL_THRESHOLD}',
        'Trial counts are deduplicated within subject using trial identifiers before selection.',
        f'Resampling iterations per subject-condition: {N_ITERATIONS}',
        '',
        f'Selected subjects: {len(selected)}',
    ]
    if not selected.empty:
        for _, row in selected.sort_values(['n_min', 'Patient']).iterrows():
            lines.append(
                f"  {row['Patient']} ({row['Group']}): remembered={int(row['remembered'])}, "
                f"forgotten={int(row['forgotten'])}, n_min={int(row['n_min'])}"
            )
    else:
        lines.append('  None')
    lines.append('')
    lines.append(f'Subjects skipped because one memory condition was absent: {len(missing)}')
    if not missing.empty:
        for _, row in missing.sort_values('Patient').iterrows():
            lines.append(
                f"  {row['Patient']} ({row['Group']}): remembered={int(row['remembered'])}, "
                f"forgotten={int(row['forgotten'])}"
            )
    else:
        lines.append('  None')
    lines.append('')
    lines.append('Region-level subject Ns after permutation balancing:')
    if region_stats_df.empty:
        lines.append('  None')
    else:
        for region, region_df in region_stats_df.groupby('Region'):
            lines.append(f'  {pretty_in_text(region)}')
            for _, row in region_df.sort_values(['trial_type', 'memory_cond']).iterrows():
                lines.append(
                    f"    {STIM_LABELS[row['trial_type']]} {MEMORY_LABELS[row['memory_cond']]}: "
                    f"N={int(row['n_subjects'])}"
                )
    if pac_theta_note:
        lines.extend([
            '',
            'PAC note:',
            '  Theta band summary is omitted because the available PAC amplitude-frequency bins start at 30 Hz.',
        ])
    out_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def analyze_metric_phase(
    metric_label: str,
    phase_label: str,
    df: pd.DataFrame,
    freqs: np.ndarray,
    out_dir: Path,
    *,
    band_ranges: dict[str, tuple[float, float]],
    ylabel: str,
):
    ensure_dir(out_dir)
    csv_dir = ensure_dir(out_dir / 'csvs')

    selection_df = compute_subject_selection(df)
    selected_subjects = set(selection_df.loc[selection_df['selected_for_permutation'], 'Patient'])
    df = df[df['Patient'].isin(selected_subjects)].copy()

    all_curve_rows = []
    all_band_rows = []
    region_stats_rows = []

    if df.empty:
        selection_df.to_csv(csv_dir / 'subject_selection.csv', index=False)
        save_summary_text(
            selection_df,
            pd.DataFrame(),
            out_dir / 'summary.txt',
            metric_label=metric_label,
            phase_label=phase_label,
            pac_theta_note=(metric_label == 'PAC'),
        )
        return

    freq_cols = sorted_freq_cols(df, 'post_Freq_')
    rng_master = np.random.default_rng(BASE_RANDOM_SEED)

    for region in sorted(df['Region'].unique()):
        region_df = df[df['Region'] == region].copy()
        if region_df.empty:
            continue

        example_subjects = choose_example_subjects(region_df)
        condition_subject_curves = {(trial_type, memory_cond): {} for trial_type in STIM_ORDER for memory_cond in MEMORY_ORDER}
        diff_subject_curves = {trial_type: {} for trial_type in STIM_ORDER}
        stability_rows = []
        region_band_rows = []

        for patient in sorted(region_df['Patient'].unique()):
            patient_df = region_df[region_df['Patient'] == patient]
            patient_group = patient_df['Group'].iloc[0]
            for trial_type in STIM_ORDER:
                cond_df = patient_df[patient_df['trial_type'] == trial_type]
                if cond_df.empty:
                    continue
                remembered_matrix = cond_df.loc[cond_df['memory_cond'] == 'remembered', freq_cols].to_numpy(dtype=float)
                forgotten_matrix = cond_df.loc[cond_df['memory_cond'] == 'forgotten', freq_cols].to_numpy(dtype=float)
                if remembered_matrix.shape[0] == 0 or forgotten_matrix.shape[0] == 0:
                    continue

                rng = np.random.default_rng(rng_master.integers(0, 2**32 - 1))
                result = compute_subject_balanced_curves(
                    remembered_matrix,
                    forgotten_matrix,
                    freqs=freqs,
                    band_ranges=band_ranges,
                    rng=rng,
                    store_samples=(patient in example_subjects),
                )
                if result is None:
                    continue

                for memory_cond in MEMORY_ORDER:
                    mean_curve = np.asarray(result[f'{memory_cond}_mean'], dtype=float)
                    sd_curve = np.asarray(result[f'{memory_cond}_sd'], dtype=float)
                    condition_subject_curves[(trial_type, memory_cond)][patient] = mean_curve
                    for freq, curve_value, sd_value in zip(freqs, mean_curve, sd_curve):
                        all_curve_rows.append({
                            'metric': metric_label,
                            'phase': phase_label,
                            'Region': region,
                            'Patient': patient,
                            'Group': patient_group,
                            'trial_type': trial_type,
                            'memory_cond': memory_cond,
                            'frequency_hz': freq,
                            'balanced_mean': curve_value,
                            'resample_sd': sd_value,
                            'n_remembered': result['n_remembered'],
                            'n_forgotten': result['n_forgotten'],
                            'n_min': result['n_min'],
                            'resampled_condition': result['resampled_condition'],
                            'n_iterations': N_ITERATIONS,
                        })

                diff_subject_curves[trial_type][patient] = np.asarray(result['diff_mean'], dtype=float)

                for band_row in result['band_rows']:
                    row = {
                        'metric': metric_label,
                        'phase': phase_label,
                        'Region': region,
                        'Patient': patient,
                        'Group': patient_group,
                        'trial_type': trial_type,
                        'memory_cond': band_row['memory_cond'],
                        'band': band_row['band'],
                        'band_mean': band_row['band_mean'],
                        'resample_sd': band_row['resample_sd'],
                        'n_remembered': result['n_remembered'],
                        'n_forgotten': result['n_forgotten'],
                        'n_min': result['n_min'],
                        'resampled_condition': result['resampled_condition'],
                        'n_iterations': N_ITERATIONS,
                    }
                    all_band_rows.append(row)
                    region_band_rows.append(row)

                sample_payload = result['sample_payload']
                if sample_payload is not None:
                    stability_rows.append({
                        'Patient': patient,
                        'trial_type': trial_type,
                        'memory_cond': sample_payload['memory_cond'],
                        'curves': sample_payload['curves'],
                        'mean_curve': sample_payload['mean_curve'],
                        'n_small': sample_payload['n_small'],
                        'n_large': sample_payload['n_large'],
                        'n_min': sample_payload['n_min'],
                    })

        if not any(condition_subject_curves.values()):
            continue

        for trial_type, memory_cond in [(s, m) for s in STIM_ORDER for m in MEMORY_ORDER]:
            region_stats_rows.append({
                'Region': region,
                'trial_type': trial_type,
                'memory_cond': memory_cond,
                'n_subjects': len(condition_subject_curves.get((trial_type, memory_cond), {})),
            })

        region_slug = slugify(region)
        title_prefix = (
            f'{metric_label} | {pretty_in_text(region)} | {phase_label} | '
            f'Unbalanced Memory Subjects (<{LOW_TRIAL_THRESHOLD} overall remembered or forgotten)'
        )
        plot_main_curves(
            condition_subject_curves,
            freqs,
            out_dir / f'{metric_label.lower()}_{region_slug}_{phase_label.lower()}_balanced_remembered_forgotten_main_curves.png',
            title_prefix,
            ylabel,
        )
        plot_difference_curves(
            diff_subject_curves,
            freqs,
            out_dir / f'{metric_label.lower()}_{region_slug}_{phase_label.lower()}_balanced_remembered_forgotten_difference_curves.png',
            title_prefix,
            ylabel,
        )
        plot_resampling_stability(
            stability_rows,
            freqs,
            out_dir / f'{metric_label.lower()}_{region_slug}_{phase_label.lower()}_balanced_remembered_forgotten_resampling_stability.png',
            title_prefix,
            ylabel,
        )
        plot_band_summary(
            pd.DataFrame(region_band_rows),
            out_dir / f'{metric_label.lower()}_{region_slug}_{phase_label.lower()}_balanced_remembered_forgotten_band_summary.png',
            title_prefix,
            ylabel,
        )

    selection_df.to_csv(csv_dir / 'subject_selection.csv', index=False)
    if all_curve_rows:
        pd.DataFrame(all_curve_rows).to_csv(csv_dir / 'subject_curve_means_and_variability.csv', index=False)
    if all_band_rows:
        pd.DataFrame(all_band_rows).to_csv(csv_dir / 'subject_band_means_and_variability.csv', index=False)
    region_stats_df = pd.DataFrame(region_stats_rows)
    if not region_stats_df.empty:
        region_stats_df.to_csv(csv_dir / 'region_subject_counts.csv', index=False)
    save_summary_text(
        selection_df,
        region_stats_df,
        out_dir / 'summary.txt',
        metric_label=metric_label,
        phase_label=phase_label,
        pac_theta_note=(metric_label == 'PAC'),
    )


def load_metric_phase(metric_label: str, phase_label: str) -> tuple[pd.DataFrame, np.ndarray]:
    if metric_label == 'Power' and phase_label == 'Encoding':
        return load_power_encoding_trials()
    if metric_label == 'Power' and phase_label == 'Retrieval':
        return load_power_retrieval_trials()
    if metric_label == 'Coherence' and phase_label == 'Encoding':
        return load_coherence_encoding_trials()
    if metric_label == 'Coherence' and phase_label == 'Retrieval':
        return load_coherence_retrieval_trials()
    if metric_label == 'PAC' and phase_label == 'Encoding':
        return load_pac_encoding_trials()
    if metric_label == 'PAC' and phase_label == 'Retrieval':
        return load_pac_retrieval_trials()
    raise ValueError(f'Unsupported metric/phase combination: {metric_label} {phase_label}')


def main():
    print('=' * 80)
    print('Permutation Testing for Unbalanced Memory Conditions')
    print('=' * 80)
    print(f'Output root: {OUTPUT_ROOT}')
    print(f'Subject threshold: overall remembered or forgotten < {LOW_TRIAL_THRESHOLD}')
    print(f'Resampling iterations: {N_ITERATIONS}')

    ensure_dir(OUTPUT_ROOT)

    analysis_specs = [
        ('Power', 'Encoding', POWER_BANDS, 'Power (dB)'),
        ('Power', 'Retrieval', POWER_BANDS, 'Power (dB)'),
        ('Coherence', 'Encoding', POWER_BANDS, 'Coherence'),
        ('Coherence', 'Retrieval', POWER_BANDS, 'Coherence'),
        ('PAC', 'Encoding', PAC_BANDS, 'PAC'),
        ('PAC', 'Retrieval', PAC_BANDS, 'PAC'),
    ]

    for metric_label, phase_label, band_ranges, ylabel in analysis_specs:
        print('\n' + '-' * 80)
        print(f'Loading {metric_label} {phase_label} trials...')
        df, freqs = load_metric_phase(metric_label, phase_label)
        print(f'  Loaded rows: {len(df)}')
        print(f'  Unique regions: {df["Region"].nunique() if not df.empty else 0}')
        print(f'  Unique subjects: {df["Patient"].nunique() if not df.empty else 0}')
        analyze_metric_phase(
            metric_label,
            phase_label,
            df,
            freqs,
            ensure_dir(OUTPUT_ROOT / f'{phase_label.lower()}_{metric_label.lower()}'),
            band_ranges=band_ranges,
            ylabel=ylabel,
        )

    print('\n' + '=' * 80)
    print(f'Done. Outputs saved to: {OUTPUT_ROOT}')
    print('=' * 80)


if __name__ == '__main__':
    main()
