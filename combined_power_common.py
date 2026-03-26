#!/usr/bin/env python
"""Shared helpers for combined power encoding/retrieval analyses."""

from copy import deepcopy

import numpy as np


ALLHPC = 'ALLHPC'
MTL = 'MTL'
COMPOSITE_ROIS = [ALLHPC, MTL]
ALLHPC_REGIONS = {'CA', 'DG', 'HPC'}
MTL_REGIONS = {'CA', 'DG', 'HPC', 'EC', 'PHG', 'PRC'}
COMPOSITE_LOGIC_TEXT = (
    'Composite logic: average power spectra within each patient and original ROI first, '
    'then build ALLHPC by averaging the patient mean CA, DG, and HPC spectra and build MTL '
    'by averaging the patient mean non-BLA MTL spectra (CA, DG, HPC, EC, PHG, PRC). '
    'Band means and plotted contrasts are computed after the composite spectrum is formed.'
)


def is_region_subject_dict(value):
    if not isinstance(value, dict):
        return False
    if not value:
        return True
    return all(isinstance(subvalue, dict) for subvalue in value.values())


def merge_subject_dicts(first, second):
    merged = {}
    for source in [first or {}, second or {}]:
        for region, subject_dict in source.items():
            target = merged.setdefault(region, {})
            for subject, values in subject_dict.items():
                target[str(subject)] = np.asarray(values, dtype=np.float64)
    return merged


def build_power_composite_region_dict(region_dict):
    composites = {roi: {} for roi in COMPOSITE_ROIS}
    for region, subject_dict in (region_dict or {}).items():
        if region in COMPOSITE_ROIS or str(region).startswith('PNAS'):
            continue
        for subject, values in subject_dict.items():
            arr = np.asarray(values, dtype=np.float64)
            if region in MTL_REGIONS:
                composites[MTL].setdefault(str(subject), []).append(arr)
            if region in ALLHPC_REGIONS:
                composites[ALLHPC].setdefault(str(subject), []).append(arr)

    collapsed = {}
    for composite_roi, subject_dict in composites.items():
        for subject, vectors in subject_dict.items():
            if not vectors:
                continue
            collapsed.setdefault(composite_roi, {})[subject] = np.nanmean(np.vstack(vectors), axis=0)
    return collapsed


def augment_with_power_composites(data):
    augmented = {}
    for key, value in data.items():
        if is_region_subject_dict(value):
            augmented[key] = merge_subject_dicts(value, build_power_composite_region_dict(value))
        elif isinstance(value, np.ndarray):
            augmented[key] = np.asarray(value, dtype=np.float64).copy()
        elif isinstance(value, list):
            augmented[key] = deepcopy(value)
        else:
            augmented[key] = value
    return augmented
