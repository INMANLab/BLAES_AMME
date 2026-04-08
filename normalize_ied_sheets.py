#!/usr/bin/env python
"""
Normalize IED Excel Sheets (Study & Test)
==========================================
Re-reads the raw Excel files and produces cleaned CSVs with:
  - Normalized Region names
  - Normalized GrayMatter → 'Gray', 'White', 'Both', or NaN
  - ChannelSpread: number of channels per row (underscore-separated names)
  - RegionSpread: number of rows per Patient+Trial (IED detected across regions)
  - WeightedIEDRate numeric
  - IEDRateAvg numeric

Outputs:
  IED/AMMEBLAES_IEDs_trial_level_dissertation_study_usethis_cleaned.csv
  IED/AMMEBLAES_IEDs_trial_level_dissertation_test_usethis_cleaned.csv
"""

import os
import re
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IED_DIR = os.path.join(SCRIPT_DIR, 'IED')

STUDY_XLSX = os.path.join(IED_DIR, 'AMMEBLAES_IEDs_trial_level_dissertation_study.xlsx')
TEST_XLSX = os.path.join(IED_DIR, 'AMMEBLAES_IEDs_trial_level_dissertation_test.xlsx')
STUDY_CSV = os.path.join(IED_DIR, 'AMMEBLAES_IEDs_trial_level_dissertation_study_usethis_cleaned.csv')
TEST_CSV = os.path.join(IED_DIR, 'AMMEBLAES_IEDs_trial_level_dissertation_test_usethis_cleaned.csv')


# ---------------------------------------------------------------------------
# Region normalization
# ---------------------------------------------------------------------------
# Canonical region map: lowercased key -> canonical name
REGION_MAP = {
    'amygdala': 'Amygdala',
    'hippocampus': 'Hippocampus',
    'hippocampal': 'Hippocampus',
    'hippocampalgrey': 'Hippocampus',
    'entorhinal': 'Entorhinal',
    'entorhinal area': 'Entorhinal',
    'entorhinal cortex': 'Entorhinal',
    'parahippocampal': 'Parahippocampal',
    'parahippocampal gyrus': 'Parahippocampal',
    'parahippocampus': 'Parahippocampal',
    'parahippocampul gyrus': 'Parahippocampal',
    'fusiform': 'Fusiform Gyrus',
    'fusiform gyrus': 'Fusiform Gyrus',
    'occipital fusiform gyrus': 'Fusiform Gyrus',
    'cortex (ih) fusiform': 'Fusiform Gyrus',
    'inferior temporal': 'Inferior Temporal Gyrus',
    'inferior temporal gyrus': 'Inferior Temporal Gyrus',
    'interior temporal gyrus': 'Inferior Temporal Gyrus',
    'middle temporal': 'Middle Temporal Gyrus',
    'middle temporal gyrus': 'Middle Temporal Gyrus',
    'superior temporal': 'Superior Temporal Gyrus',
    'superior temporal gyrus': 'Superior Temporal Gyrus',
    'superior-temporal': 'Superior Temporal Gyrus',
    'temporal pole': 'Temporal Pole',
    'temporal lobe': 'Temporal Lobe',
    'insula': 'Insula',
    'thalamus': 'Thalamus',
    'precuneus': 'Precuneus',
    'supramarginal': 'Supramarginal Gyrus',
    'supramarginal gyrus': 'Supramarginal Gyrus',
    'posterior cingulate gyrus': 'Posterior Cingulate',
    'posterior cingulate': 'Posterior Cingulate',
    'posterior congukate': 'Posterior Cingulate',
    'middle cingulate gyrus': 'Middle Cingulate',
    'ithsmus cingulate': 'Isthmus Cingulate',
    'frontal cortex': 'Frontal Cortex',
    'orbito-frontal': 'Orbitofrontal',
    'medial-orbito-frontal': 'Orbitofrontal',
    'superior-frontal': 'Superior Frontal',
    'superiorparietal': 'Superior Parietal',
    'inferior parietal': 'Inferior Parietal',
    'inferior-parietal': 'Inferior Parietal',
    'cerebellum exterior': 'Cerebellum',
    'cerebral white matter': 'Cerebral White Matter',
    'left ventral dc': 'Ventral DC',
    'planum palore': 'Planum Polare',
    'posterior insula': 'Posterior Insula',
    # Full compound names that should NOT be split on dash
    'orbito frontal': 'Orbitofrontal',
    'medial orbito frontal': 'Orbitofrontal',
    'superior temporal': 'Superior Temporal Gyrus',
    'superior frontal': 'Superior Frontal',
    'inferior temporal': 'Inferior Temporal Gyrus',
    'middle temporal': 'Middle Temporal Gyrus',
    'inferior parietal': 'Inferior Parietal',
    'inferior middle temporal': 'Inferior/Middle Temporal Gyrus',
    'inferior temporal fusiform': 'Inferior Temporal Gyrus, Fusiform Gyrus',
    'inferior temporal middle temporal': 'Inferior/Middle Temporal Gyrus',
    'insula middle temporal': 'Insula, Middle Temporal Gyrus',
    'amygdala hippocampal': 'Amygdala, Hippocampus',
    'hippocampal entorhinal': 'Hippocampus, Entorhinal',
    'hippocampal inferior middle temporal': 'Hippocampus, Inferior/Middle Temporal Gyrus',
    'hippocampal parahippocampus': 'Hippocampus, Parahippocampal',
    'parahippocampal hippocampal': 'Parahippocampal, Hippocampus',
    'cerebellum exterior, fusiform gyrus': 'Cerebellum, Fusiform Gyrus',
    'cerebellum exterior, fusiform gyrus, interior temporal gyrus': 'Cerebellum, Fusiform Gyrus, Inferior Temporal Gyrus',
    'hippocampus, fusiform gyrus': 'Hippocampus, Fusiform Gyrus',
    'hippocampus, amydala': 'Amygdala, Hippocampus',
    'hippocampus, amygdala': 'Amygdala, Hippocampus',
    'parahippocampal gyrus, hippocampus': 'Parahippocampal, Hippocampus',
    'parahippocampul gyrus, fusiform gyrus': 'Parahippocampal, Fusiform Gyrus',
    'inferior temporal gyrus ': 'Inferior Temporal Gyrus',
    'posterior congukate ': 'Posterior Cingulate',
    'posterior cingulate ': 'Posterior Cingulate',
    'cerebral white matter hippocampus': 'Cerebral White Matter, Hippocampus',
    'parahippocampal gyrus entorhinal area hippocampus': 'Parahippocampal, Entorhinal, Hippocampus',
    'parahippocampal gyrus fusiform gyrus': 'Parahippocampal, Fusiform Gyrus',
    'parahippocampal gyrus fusiform gyrus inferior temporal gyrus': 'Parahippocampal, Fusiform Gyrus, Inferior Temporal Gyrus',
    'parahippocampal gyrus hippocampus': 'Parahippocampal, Hippocampus',
    'hippocampus amygdala cerebral white matter': 'Hippocampus, Amygdala, Cerebral White Matter',
    'left ventral dc hippocampus cerebral white matter middle temporal gyrus': 'Ventral DC, Hippocampus, Cerebral White Matter, Middle Temporal Gyrus',
    'middle cingulate gyrus, cerebral white matter': 'Middle Cingulate, Cerebral White Matter',
    'entorhinal area inferior temporal gyrus': 'Entorhinal, Inferior Temporal Gyrus',
    'entorhinal area posterior insula planum palore': 'Entorhinal, Posterior Insula, Planum Polare',
    'grey white': 'Both',
}


def normalize_single_region(token):
    """Normalize a single region token."""
    token = token.strip()
    # Remove trailing special characters
    token = re.sub(r'[￼\s]+$', '', token)
    low = token.lower().strip()

    if not low or low == '?':
        return None

    # Direct lookup
    if low in REGION_MAP:
        return REGION_MAP[low]

    # Check if starts with 'hippocamp' but NOT 'parahippocamp'
    if low.startswith('hippocamp') and not low.startswith('parahippocamp'):
        return 'Hippocampus'

    # Check partial matches
    for key, val in REGION_MAP.items():
        if low == key:
            return val

    return token.strip().title()


def normalize_region(raw):
    """Normalize a region string that may contain comma or dash separators
    indicating multiple regions."""
    if pd.isna(raw) or str(raw).strip() in ('', '?'):
        return None

    raw_str = str(raw).strip()
    # Remove trailing special chars
    raw_str = re.sub(r'[￼]+', '', raw_str)

    # First try the full string as a direct match (collapse separators to space)
    low = raw_str.lower().strip()
    # Remove trailing special unicode chars
    low = re.sub(r'[￼]+', '', low).strip()
    low_norm = re.sub(r'[\s\-_,]+', ' ', low).strip()
    if low_norm in REGION_MAP:
        return REGION_MAP[low_norm]
    # Also try with commas preserved for exact matches
    low_comma = re.sub(r'[\s]+', ' ', low).strip()
    if low_comma in REGION_MAP:
        return REGION_MAP[low_comma]
    # Check hippocamp prefix on full string (but not parahippocamp)
    if low_norm.startswith('hippocamp') and not low_norm.startswith('parahippocamp'):
        return 'Hippocampus'

    # Split on comma or dash, but be careful with compound region names
    # like "Inferior Temporal Gyrus" — hyphens between words in a single
    # region name (e.g., "orbito-frontal") should resolve via REGION_MAP
    # before we try splitting into multiple regions.
    # Strategy: first try the whole string via REGION_MAP (done above),
    # then split on , and - and normalize each token.
    parts = re.split(r'[,\-]', raw_str)
    parts = [p.strip() for p in parts if p.strip()]

    if len(parts) == 1:
        return normalize_single_region(parts[0])

    # Normalize each part
    normalized_parts = []
    for p in parts:
        norm = normalize_single_region(p)
        if norm and norm not in normalized_parts:
            normalized_parts.append(norm)

    if not normalized_parts:
        return None
    if len(normalized_parts) == 1:
        return normalized_parts[0]

    return ', '.join(sorted(normalized_parts))


# ---------------------------------------------------------------------------
# GrayMatter normalization
# ---------------------------------------------------------------------------
def normalize_graymatter(raw):
    """Normalize gray matter values.
    G, Grey, Gray, grey -> Gray
    W, White -> White
    Comma or dash between -> Both
    CSF -> CSF
    """
    if pd.isna(raw) or str(raw).strip() in ('', '?'):
        return None

    s = str(raw).strip().lower()
    # Remove trailing whitespace already done

    # Check for separators indicating both
    has_separator = ',' in s or '-' in s

    tokens = re.split(r'[,\-\s]+', s)
    tokens = [t.strip() for t in tokens if t.strip()]

    has_gray = any(t in ('g', 'grey', 'gray') for t in tokens)
    has_white = any(t in ('w', 'white') for t in tokens)
    has_csf = any(t == 'csf' for t in tokens)

    if has_gray and has_white:
        return 'B'
    if has_separator and has_gray and has_csf:
        return 'G'  # CSF + Gray, treat as Gray
    if has_gray:
        return 'G'
    if has_white:
        return 'W'
    if has_csf:
        return 'CSF'

    return None


# ---------------------------------------------------------------------------
# Channel and Region spread
# ---------------------------------------------------------------------------
def count_channels(channel_name):
    """Count channels from underscore-separated channel names."""
    if pd.isna(channel_name):
        return 0
    return len(str(channel_name).strip().split('_'))


# ---------------------------------------------------------------------------
# Process a single sheet
# ---------------------------------------------------------------------------
def process_sheet(df, sheet_name, trial_col='Trial'):
    """Normalize and enrich a single IED sheet."""
    print(f"\n{'='*60}")
    print(f"Processing {sheet_name}: {len(df)} rows")
    print(f"{'='*60}")

    # Drop unnamed columns
    df = df[[c for c in df.columns if not c.startswith('Unnamed')]].copy()

    # Normalize Region
    df['Region_raw'] = df['Region'].copy()
    df['Region'] = df['Region'].apply(normalize_region)

    print(f"\nRegion normalization ({df['Region'].nunique()} unique):")
    for r in sorted(df['Region'].dropna().unique()):
        count = (df['Region'] == r).sum()
        print(f"  {r}: {count}")
    n_null = df['Region'].isna().sum()
    if n_null:
        print(f"  (null): {n_null}")

    # Normalize GrayMatter
    df['GrayMatter_raw'] = df['GrayMatter'].copy()
    df['GrayMatter'] = df['GrayMatter'].apply(normalize_graymatter)

    print(f"\nGrayMatter normalization:")
    print(f"  {df['GrayMatter'].value_counts(dropna=False).to_dict()}")

    # Channel spread (per row)
    df['ChannelSpread'] = df['ChannelName'].apply(count_channels)

    print(f"\nChannelSpread distribution:")
    print(f"  {df['ChannelSpread'].value_counts().sort_index().to_dict()}")

    # Region spread (rows per Patient+Trial)
    region_spread = df.groupby(['Patient', trial_col]).size().reset_index(name='RegionSpread')
    df = df.merge(region_spread, on=['Patient', trial_col], how='left')

    print(f"\nRegionSpread distribution:")
    print(f"  {df['RegionSpread'].value_counts().sort_index().to_dict()}")

    # Numeric conversions
    df['WeightedIEDRate'] = pd.to_numeric(df['WeightedIEDRate'], errors='coerce')
    df['IEDRateAvg'] = pd.to_numeric(df['IEDRateAvg'], errors='coerce')
    df['Rate(within trials)'] = pd.to_numeric(df['Rate(within trials)'], errors='coerce')

    # Summary
    print(f"\nWeightedIEDRate: mean={df['WeightedIEDRate'].mean():.2f}, "
          f"non-null={df['WeightedIEDRate'].notna().sum()}/{len(df)}")
    print(f"IEDRateAvg: mean={df['IEDRateAvg'].mean():.2f}, "
          f"non-null={df['IEDRateAvg'].notna().sum()}/{len(df)}")

    # Drop raw columns
    df = df.drop(columns=['Region_raw', 'GrayMatter_raw'])

    print(f"\nFinal columns: {list(df.columns)}")
    print(f"Final shape: {df.shape}")

    return df


def main():
    # Read Excel files
    study = pd.read_excel(STUDY_XLSX)
    test = pd.read_excel(TEST_XLSX)

    # Process
    study_clean = process_sheet(study, 'STUDY', trial_col='Trial')
    test_clean = process_sheet(test, 'TEST', trial_col='Trial')

    # Save
    study_clean.to_csv(STUDY_CSV, index=False)
    print(f"\nSaved: {STUDY_CSV}")

    test_clean.to_csv(TEST_CSV, index=False)
    print(f"Saved: {TEST_CSV}")

    # Quick patient summary
    print(f"\n{'='*60}")
    print("STUDY patients: " + ', '.join(sorted(study_clean['Patient'].unique())))
    print(f"TEST patients: " + ', '.join(sorted(test_clean['Patient'].unique())))


if __name__ == '__main__':
    main()
