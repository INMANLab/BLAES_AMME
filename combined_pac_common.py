#!/usr/bin/env python
"""Shared helpers for combined PAC encoding/retrieval analyses."""

from copy import deepcopy

import numpy as np
import pandas as pd


BLA_ALLHPC = 'BLA_ALLHPC'
BLA_MTL = 'BLA_MTL'
COMPOSITE_ROIS = [BLA_ALLHPC, BLA_MTL]
ALLHPC_PARTNERS = {'CA', 'DG', 'HPC'}
ALLHPC_EC = 'ALLHPC_EC'
ALLHPC_PRC = 'ALLHPC_PRC'
ALLHPC_PAIR_COMPOSITE_ROIS = [ALLHPC_EC, ALLHPC_PRC]
ALLHPC_PAIR_SOURCES = {
    ALLHPC_EC: {'CA_EC', 'DG_EC', 'EC_HPC'},
    ALLHPC_PRC: {'CA_PRC', 'DG_PRC', 'HPC_PRC'},
}
COMPOSITE_LOGIC_TEXT = (
    'Composite logic: 1) average PAC across trials within each patient, original region pair, '
    'and condition to form one spectrum per region pair. 2) Build BLA_ALLHPC by averaging the '
    'already-averaged BLA_CA, BLA_DG, and BLA_HPC spectra within each patient and condition. '
    '3) Build BLA_MTL by averaging the already-averaged BLA-to-non-BLA spectra within each patient '
    'and condition. 4) Build ALLHPC_EC and ALLHPC_PRC by averaging the already-averaged CA/DG/HPC-to-EC '
    'or -PRC spectra within each patient and condition. 5) Compute plotted band means or contrasts from '
    'the resulting composite spectrum.'
)


def is_amme_patient(patient):
    return str(patient).startswith('amyg')


def is_region_subject_dict(value):
    if not isinstance(value, dict):
        return False
    if not value:
        return True
    return all(isinstance(subvalue, dict) for subvalue in value.values())


def subset_region_dict(region_dict, predicate):
    subset = {}
    for region, subject_dict in (region_dict or {}).items():
        kept = {
            str(subject): np.asarray(values, dtype=np.float64)
            for subject, values in subject_dict.items()
            if predicate(str(subject))
        }
        if kept:
            subset[region] = kept
    return subset


def subset_trial_count_rows(rows, predicate):
    filtered = []
    for frame in rows or []:
        df = frame.copy()
        if 'Patient' in df.columns:
            df = df[df['Patient'].astype(str).map(predicate)].copy()
        if not df.empty:
            filtered.append(df)
    return filtered


def subset_pac_data(data, predicate):
    subset = {}
    for key, value in data.items():
        if is_region_subject_dict(value):
            subset[key] = subset_region_dict(value, predicate)
        elif key == 'trial_count_rows':
            subset[key] = subset_trial_count_rows(value, predicate)
        elif isinstance(value, np.ndarray):
            subset[key] = np.asarray(value, dtype=np.float64).copy()
        elif isinstance(value, list):
            subset[key] = deepcopy(value)
        else:
            subset[key] = value
    return subset


def merge_subject_dicts(*sources):
    merged = {}
    for source in sources:
        source = source or {}
        for region, subject_dict in source.items():
            region_target = merged.setdefault(region, {})
            for subject, values in subject_dict.items():
                region_target[str(subject)] = np.asarray(values, dtype=np.float64)
    return merged


def merge_trial_count_rows(first_rows, second_rows):
    rows = []
    rows.extend([frame.copy() for frame in first_rows or [] if not frame.empty])
    rows.extend([frame.copy() for frame in second_rows or [] if not frame.empty])
    return rows


def merge_pac_data(first, second):
    merged = {}
    all_keys = sorted(set(first) | set(second))
    for key in all_keys:
        first_value = first.get(key)
        second_value = second.get(key)
        if is_region_subject_dict(first_value) or is_region_subject_dict(second_value):
            merged[key] = merge_subject_dicts(first_value, second_value)
        elif key == 'trial_count_rows':
            merged[key] = merge_trial_count_rows(first_value, second_value)
        elif isinstance(first_value, np.ndarray) or isinstance(second_value, np.ndarray):
            base = first_value if first_value is not None else second_value
            merged[key] = np.asarray(base, dtype=np.float64).copy() if base is not None else None
        elif isinstance(first_value, list) or isinstance(second_value, list):
            base = first_value if first_value is not None else second_value
            merged[key] = deepcopy(base)
        else:
            merged[key] = first_value if first_value is not None else second_value
    return merged


def bla_partner_region(region):
    parts = [part.strip() for part in str(region).split('_') if part.strip()]
    if len(parts) != 2 or 'BLA' not in parts:
        return None
    return parts[0] if parts[1] == 'BLA' else parts[1]


def build_bla_composite_region_dict(region_dict):
    composites = {roi: {} for roi in COMPOSITE_ROIS}
    for region, subject_dict in (region_dict or {}).items():
        if region in COMPOSITE_ROIS:
            continue
        partner = bla_partner_region(region)
        if partner is None:
            continue
        for subject, values in subject_dict.items():
            arr = np.asarray(values, dtype=np.float64)
            composites[BLA_MTL].setdefault(str(subject), []).append(arr)
            if partner in ALLHPC_PARTNERS:
                composites[BLA_ALLHPC].setdefault(str(subject), []).append(arr)

    collapsed = {}
    for composite_roi, subject_dict in composites.items():
        for subject, vectors in subject_dict.items():
            if not vectors:
                continue
            collapsed.setdefault(composite_roi, {})[subject] = np.nanmean(np.vstack(vectors), axis=0)
    return collapsed

def build_allhpc_pair_composite_region_dict(region_dict):
    composites = {roi: {} for roi in ALLHPC_PAIR_COMPOSITE_ROIS}
    for region, subject_dict in (region_dict or {}).items():
        if region in COMPOSITE_ROIS or region in ALLHPC_PAIR_COMPOSITE_ROIS:
            continue
        for composite_roi, source_rois in ALLHPC_PAIR_SOURCES.items():
            if region not in source_rois:
                continue
            for subject, values in subject_dict.items():
                arr = np.asarray(values, dtype=np.float64)
                composites[composite_roi].setdefault(str(subject), []).append(arr)

    collapsed = {}
    for composite_roi, subject_dict in composites.items():
        for subject, vectors in subject_dict.items():
            if not vectors:
                continue
            collapsed.setdefault(composite_roi, {})[subject] = np.nanmean(np.vstack(vectors), axis=0)
    return collapsed


def build_bla_composite_data(data):
    composite_data = {}
    for key, value in data.items():
        if is_region_subject_dict(value):
            composite_data[key] = build_bla_composite_region_dict(value)
        elif key == 'trial_count_rows':
            composite_data[key] = []
        elif isinstance(value, np.ndarray):
            composite_data[key] = np.asarray(value, dtype=np.float64).copy()
        elif isinstance(value, list):
            composite_data[key] = deepcopy(value)
        else:
            composite_data[key] = value
    return composite_data

def build_allhpc_pair_composite_data(data):
    composite_data = {}
    for key, value in data.items():
        if is_region_subject_dict(value):
            composite_data[key] = build_allhpc_pair_composite_region_dict(value)
        elif key == 'trial_count_rows':
            composite_data[key] = []
        elif isinstance(value, np.ndarray):
            composite_data[key] = np.asarray(value, dtype=np.float64).copy()
        elif isinstance(value, list):
            composite_data[key] = deepcopy(value)
        else:
            composite_data[key] = value
    return composite_data


def augment_with_bla_composites(data):
    augmented = {}
    composite_data = build_bla_composite_data(data)
    pair_composite_data = build_allhpc_pair_composite_data(data)
    for key, value in data.items():
        if is_region_subject_dict(value):
            augmented[key] = merge_subject_dicts(
                value,
                composite_data.get(key, {}),
                pair_composite_data.get(key, {}),
            )
        elif key == 'trial_count_rows':
            augmented[key] = deepcopy(value)
        elif isinstance(value, np.ndarray):
            augmented[key] = np.asarray(value, dtype=np.float64).copy()
        elif isinstance(value, list):
            augmented[key] = deepcopy(value)
        else:
            augmented[key] = value
    return augmented


def split_full_pac_data(full_data):
    blaes = subset_pac_data(full_data, lambda patient: not is_amme_patient(patient))
    amme = subset_pac_data(full_data, is_amme_patient)
    all_data = merge_pac_data(blaes, amme)
    return {'blaes': blaes, 'amme': amme, 'all': all_data}
