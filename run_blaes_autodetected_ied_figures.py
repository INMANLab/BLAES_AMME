from __future__ import annotations

import argparse
import os
from collections import defaultdict
from pathlib import Path
import re
import warnings

REPO_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = REPO_ROOT / "outputs"
OUTPUT_ROOT.mkdir(exist_ok=True)
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
os.environ.setdefault("MPLCONFIGDIR", str(OUTPUT_ROOT / ".mplconfig"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(OUTPUT_ROOT / ".numba_cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(OUTPUT_ROOT / ".cache"))
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import antropy as ant
import joblib
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import scipy.stats as sp_stats
from mne_features.feature_extraction import extract_features


warnings.filterwarnings(
    "ignore",
    message="invalid value encountered in divide",
    category=RuntimeWarning,
)
warnings.filterwarnings(
    "ignore",
    message="Precision loss occurred in moment calculation due to catastrophic cancellation.*",
    category=RuntimeWarning,
)


DATA_ROOT = REPO_ROOT.parent / "BLAES_IED_detection"
NORMALIZED_EPOCHS_DIR = DATA_ROOT / "normalized_epochs"
FIGURE_OUTPUT_DIR = OUTPUT_ROOT / "autodetected_IEDs_BLAES"
MODEL_PATH = Path(
    "/Users/martinahollearn/Library/CloudStorage/Box-Box/InmanLab/AMME_Data_Emory/AMME_Data/IED_checking/iEEG_ied_detection/lgbm_model.joblib"
)
CONFIDENCE_THRESHOLD = 0.98
WINDOW_MS = 250
FEAT_TO_CHOOSE = [
    "teager_kaiser_energy_1_std",
    "teager_kaiser_energy_2_std",
    "chan_ptp",
    "ptp_amp",
    "hjorth_mobility",
    "hjorth_complexity",
    "teager_kaiser_energy_3_std",
    "chan_kurt",
    "teager_kaiser_energy_5_mean",
    "teager_kaiser_energy_0_std",
    "kurtosis",
    "teager_kaiser_energy_1_mean",
    "teager_kaiser_energy_5_std",
    "samp_entropy",
]
INPUT_PATTERN = re.compile(
    r"(?P<subject>.+)_(?P<exp>[A-Z]+)_(?P<session>[^-]+)-epo\.fif$"
)
CHANNEL_PATTERN = re.compile(r"^(?P<lead>\D+)(?P<digit>\d+)$")


def format_detection_times(detections_ms: list[float], max_items: int = 6) -> str:
    formatted = [f"{int(t) if float(t).is_integer() else round(t, 1)} ms" for t in detections_ms]
    if len(formatted) <= max_items:
        return ", ".join(formatted)
    shown = ", ".join(formatted[:max_items])
    return f"{shown}, +{len(formatted) - max_items} more"


def channel_to_lead(chan: str) -> str:
    match = CHANNEL_PATTERN.match(chan)
    return match.group("lead") if match else chan


def iter_input_records() -> list[dict[str, Path | str]]:
    records = []
    for fif_path in sorted(NORMALIZED_EPOCHS_DIR.glob("*-epo.fif")):
        match = INPUT_PATTERN.match(fif_path.name)
        if not match:
            continue
        subject = match.group("subject")
        exp = match.group("exp")
        session = match.group("session")
        prefix = fif_path.name.removesuffix("-epo.fif")
        stats_path = NORMALIZED_EPOCHS_DIR / f"{prefix}-epo_stats.csv"
        sfreq_path = NORMALIZED_EPOCHS_DIR / f"{prefix}-sfreq.txt"
        if not stats_path.exists() or not sfreq_path.exists():
            raise FileNotFoundError(f"Missing sidecar file(s) for {fif_path.name}")
        records.append(
            {
                "subject": subject,
                "exp": exp,
                "session": session,
                "epochs_path": fif_path,
                "stats_path": stats_path,
                "sfreq_path": sfreq_path,
            }
        )
    return records


def extract_epoch_features(epoch_windows: list[np.ndarray], subj_epoch: str, sfreq: float) -> pd.DataFrame:
    mobility, complexity = ant.hjorth_params(epoch_windows, axis=1)
    features = {
        "subj": np.full(len(epoch_windows), subj_epoch),
        "epoch_id": np.arange(len(epoch_windows)),
        "kurtosis": sp_stats.kurtosis(epoch_windows, axis=1),
        "hjorth_mobility": mobility,
        "hjorth_complexity": complexity,
        "ptp_amp": np.ptp(epoch_windows, axis=1),
        "samp_entropy": np.apply_along_axis(ant.sample_entropy, axis=1, arr=epoch_windows),
    }
    energy = extract_features(
        np.asarray(epoch_windows)[:, np.newaxis, :],
        sfreq,
        ["teager_kaiser_energy"],
        return_as_df=True,
    )
    energy.columns = [name[0] + "_" + name[1].replace("ch0_", "") for name in energy.columns]
    return pd.concat([pd.DataFrame(features), energy], axis=1)


def get_epoch_feature_frame(
    epoch_data: np.ndarray,
    ch_names: list[str],
    chan_to_idx: dict[str, int],
    stats_df: pd.DataFrame,
    sfreq: float,
    subj_epoch: str,
) -> pd.DataFrame:
    window_size = int((WINDOW_MS / 1000.0) * sfreq)
    channel_frames = []
    for chan in ch_names:
        chan_norm = epoch_data[chan_to_idx[chan], :].ravel()
        chan_windows = [
            chan_norm[i : i + window_size]
            for i in range(1, len(chan_norm) - window_size, window_size)
        ]
        curr_feat = extract_epoch_features(chan_windows, subj_epoch, sfreq)
        curr_feat["chan_name"] = chan
        curr_feat["chan_ptp"] = stats_df.loc[chan, "chan_ptp"]
        curr_feat["chan_kurt"] = stats_df.loc[chan, "chan_kurt"]
        curr_feat["epoch"] = list(chan_windows)
        channel_frames.append(curr_feat)
    return pd.concat(channel_frames, axis=0, ignore_index=True)


def predict_epoch_detections(
    epoch_data: np.ndarray,
    subject: str,
    epoch_num: int,
    ch_names: list[str],
    chan_to_idx: dict[str, int],
    stats_df: pd.DataFrame,
    sfreq: float,
    tmin: float,
    model,
) -> dict[str, list[float]]:
    frame = get_epoch_feature_frame(
        epoch_data=epoch_data,
        ch_names=ch_names,
        chan_to_idx=chan_to_idx,
        stats_df=stats_df,
        sfreq=sfreq,
        subj_epoch=f"{subject}_{epoch_num}",
    )
    probabilities = model.predict_proba(frame[FEAT_TO_CHOOSE])[:, 1]
    detections = np.asarray(probabilities > CONFIDENCE_THRESHOLD, dtype=bool)

    n_channels = len(ch_names)
    n_rows = len(detections)
    if n_rows % n_channels != 0:
        raise ValueError("Row count is not divisible by number of channels.")
    n_windows = n_rows // n_channels
    detections_by_channel = detections.reshape(n_channels, n_windows)

    window_len_s = WINDOW_MS / 1000.0
    start_s = tmin + np.arange(n_windows) * window_len_s
    start_ms = start_s * 1000.0

    detections_ms = {
        ch_names[ch_idx]: start_ms[detections_by_channel[ch_idx]].tolist()
        for ch_idx in range(n_channels)
    }
    return {chan: timestamps for chan, timestamps in detections_ms.items() if timestamps}


def make_lead_detections(
    detections_ms: dict[str, list[float]],
    round_ms: int | None = 1,
    dedup_tol_ms: float | None = 10.0,
) -> dict[str, list[float]]:
    lead_to_channels: dict[str, list[str]] = defaultdict(list)
    for channel in detections_ms:
        lead_to_channels[channel_to_lead(channel)].append(channel)

    lead_detections: dict[str, list[float]] = {}
    for lead, channels in lead_to_channels.items():
        sorted_channels = sorted(channels, key=lambda value: (len(value), value))
        lead_key = "_".join(sorted_channels)

        timestamps: list[float] = []
        for channel in sorted_channels:
            channel_timestamps = detections_ms.get(channel, [])
            if round_ms is not None:
                channel_timestamps = [round(float(t), round_ms) for t in channel_timestamps]
            timestamps.extend(channel_timestamps)

        if not timestamps:
            continue

        merged_timestamps = np.array(sorted(set(timestamps)), dtype=float)
        if dedup_tol_ms is not None and len(merged_timestamps) > 1:
            deduped = [merged_timestamps[0]]
            for timestamp in merged_timestamps[1:]:
                if (timestamp - deduped[-1]) > dedup_tol_ms:
                    deduped.append(timestamp)
            merged_timestamps = np.array(deduped, dtype=float)

        lead_detections[lead_key] = merged_timestamps.tolist()

    return lead_detections


def plot_epoch_with_lead_detections(
    epoch_data: np.ndarray,
    chan_to_idx: dict[str, int],
    times: np.ndarray,
    subject: str,
    exp: str,
    session: str,
    trial_num: int,
    lead_key: str,
    lead_detections: list[float],
    output_path: Path,
    zscore_each_channel: bool = True,
    spacing: float = 5.0,
    shade_alpha: float = 0.15,
) -> None:
    picks = lead_key.split("_")
    if not picks:
        raise ValueError("Lead key has no channels.")

    raise_if_missing = [channel for channel in picks if channel not in chan_to_idx]
    if raise_if_missing:
        raise ValueError(f"Missing channels for plotting: {raise_if_missing}")

    picked_indices = [chan_to_idx[channel] for channel in picks]
    lead_data = epoch_data[picked_indices, :].copy()

    if zscore_each_channel:
        mu = lead_data.mean(axis=1, keepdims=True)
        sd = lead_data.std(axis=1, keepdims=True)
        lead_data = (lead_data - mu) / (sd + 1e-12)

    offsets = np.arange(lead_data.shape[0])[::-1] * spacing
    stacked = lead_data + offsets[:, None]

    fig, ax = plt.subplots(figsize=(14, 6))
    for detection_ms in lead_detections:
        start_s = detection_ms / 1000.0
        end_s = start_s + (WINDOW_MS / 1000.0)
        ax.axvspan(start_s, end_s, alpha=shade_alpha)

    for idx, channel in enumerate(picks):
        ax.plot(times, stacked[idx], linewidth=0.8)
        ax.text(times[0], offsets[idx], channel, va="center", fontsize=9)

    title = (
        f"{subject} {exp} {session} | Trial {trial_num} | {channel_to_lead(picks[0])} | "
        f"IEDs at {format_detection_times(lead_detections)}"
    )
    ax.set_title(title)
    ax.set_xlabel("Time (s) relative to event")
    ax.set_ylabel("Channels (stacked)")
    ax.set_yticks([])
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def sanitize_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-")


def run_subject(record: dict[str, Path | str], model) -> list[dict[str, object]]:
    subject = str(record["subject"])
    exp = str(record["exp"])
    session = str(record["session"])
    epochs_path = Path(record["epochs_path"])
    stats_path = Path(record["stats_path"])
    sfreq_path = Path(record["sfreq_path"])

    with sfreq_path.open("r", encoding="utf-8") as handle:
        sfreq = float(handle.read().strip())

    print(f"Running {subject} {exp} {session}")
    epochs_mne = mne.read_epochs(epochs_path, preload=True, verbose="error")
    stats_df = pd.read_csv(stats_path).set_index("chan_name")
    epochs_data = epochs_mne.get_data(copy=True)
    ch_names = epochs_mne.ch_names
    chan_to_idx = {channel: idx for idx, channel in enumerate(ch_names)}
    tmin = float(epochs_mne.tmin)
    times = epochs_mne.times

    summary_rows: list[dict[str, object]] = []
    subject_output_dir = FIGURE_OUTPUT_DIR / f"{subject}_{exp}_{session}"
    subject_output_dir.mkdir(parents=True, exist_ok=True)

    for epoch_num in range(len(epochs_mne)):
        if epoch_num % 10 == 0:
            print(f"  epoch {epoch_num + 1}/{len(epochs_mne)}")
        epoch_predictions = predict_epoch_detections(
            epoch_data=epochs_data[epoch_num],
            subject=subject,
            epoch_num=epoch_num,
            ch_names=ch_names,
            chan_to_idx=chan_to_idx,
            stats_df=stats_df,
            sfreq=sfreq,
            tmin=tmin,
            model=model,
        )
        lead_detections = make_lead_detections(epoch_predictions)
        trial_num = epoch_num + 1

        for lead_key, detections_ms in lead_detections.items():
            lead_label = sanitize_filename(channel_to_lead(lead_key.split("_")[0]))
            output_path = subject_output_dir / (
                f"{subject}_{exp}_{session}_trial{trial_num:03d}_{lead_label}.png"
            )
            plot_epoch_with_lead_detections(
                epoch_data=epochs_data[epoch_num],
                chan_to_idx=chan_to_idx,
                times=times,
                subject=subject,
                exp=exp,
                session=session,
                trial_num=trial_num,
                lead_key=lead_key,
                lead_detections=detections_ms,
                output_path=output_path,
            )
            summary_rows.append(
                {
                    "subject": subject,
                    "exp": exp,
                    "session": session,
                    "trial_num": trial_num,
                    "lead_name": lead_key,
                    "detection_times_ms": ";".join(str(timestamp) for timestamp in detections_ms),
                    "figure_path": str(output_path.relative_to(REPO_ROOT)),
                }
            )

    return summary_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run autodetected BLAES IED figures for normalized epoch files."
    )
    parser.add_argument(
        "--subjects",
        nargs="*",
        help="Optional subset of subject IDs to process.",
    )
    parser.add_argument(
        "--summary-name",
        default="autodetected_ied_figures_summary.csv",
        help="Summary CSV filename written inside outputs/autodetected_IEDs_BLAES.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    FIGURE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    records = iter_input_records()
    if not records:
        raise FileNotFoundError(f"No epoch files found in {NORMALIZED_EPOCHS_DIR}")
    if args.subjects:
        requested = set(args.subjects)
        records = [record for record in records if str(record["subject"]) in requested]
        if not records:
            raise ValueError(f"No matching subject runs found for: {sorted(requested)}")

    model = joblib.load(MODEL_PATH)
    all_rows: list[dict[str, object]] = []
    for record in records:
        all_rows.extend(run_subject(record, model))

    summary_df = pd.DataFrame(all_rows)
    summary_path = FIGURE_OUTPUT_DIR / args.summary_name
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved {len(summary_df)} figure rows to {summary_path}")


if __name__ == "__main__":
    main()
