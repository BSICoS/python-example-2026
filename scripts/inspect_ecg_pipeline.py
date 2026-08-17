#!/usr/bin/env python
"""Interactively inspect the current ECG processing on real training records."""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import edfio
import matplotlib
import numpy as np


def _select_matplotlib_backend():
    """Select a GUI backend before pyplot creates its first figure."""

    if "--no-show" in sys.argv:
        matplotlib.use("Agg", force=True)
        return "Agg"

    requested_backend = os.environ.get("MPLBACKEND")
    if requested_backend:
        matplotlib.use(requested_backend, force=True)
        return requested_backend

    if any(
        importlib.util.find_spec(module_name) is not None
        for module_name in ("PyQt6", "PySide6")
    ):
        matplotlib.use("QtAgg", force=True)
        return "QtAgg"

    if importlib.util.find_spec("tkinter") is not None:
        matplotlib.use("TkAgg", force=True)
        return "TkAgg"

    matplotlib.use("Agg", force=True)
    return None


INTERACTIVE_BACKEND = _select_matplotlib_backend()

import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RadioButtons, RangeSlider, Slider


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.ecg_processing import ECG_SEGMENT_FEATURE_LENGTH
from src.lib.ecg_inspection import CurrentEcgTrace, inspect_current_ecg_features
from src.resp_processing import (
    get_respiration_feature_group,
    select_best_respiration_signal,
)
from src.pipeline.config import (
    DEFAULT_CSV_PATH,
    SEGMENT_DURATION_SECONDS,
    SEGMENT_STRIDE_SECONDS,
)


DEFAULT_SUBJECTS = (
    "sub-I0002150001401_ses-2",
    "sub-I0002150005420_ses-1",
    "sub-I0002150021789_ses-1",
    "sub-I0002150024833_ses-1",
    "sub-I0006179009733_ses-2",
)
ECG_TOKENS = ("ecg", "ekg")


@dataclass(frozen=True)
class SignalChannel:
    label: str
    data: np.ndarray
    sampling_frequency: float


@dataclass(frozen=True)
class SubjectRecord:
    subject_id: str
    edf_path: Path


@dataclass(frozen=True)
class LoadedSubject:
    record: SubjectRecord
    ecg: SignalChannel
    respiration_channels: tuple[SignalChannel, ...]

    @property
    def duration_seconds(self) -> float:
        return len(self.ecg.data) / self.ecg.sampling_frequency

    @property
    def window_starts(self) -> np.ndarray:
        last_start = self.duration_seconds - SEGMENT_DURATION_SECONDS
        if last_start < 0:
            return np.array([], dtype=float)
        return np.arange(
            0.0,
            last_start + 1e-9,
            SEGMENT_STRIDE_SECONDS,
            dtype=float,
        )


def _find_records(data_folder: Path, subject_ids) -> list[SubjectRecord]:
    physiological_root = data_folder / "physiological_data"
    records = []
    for subject_id in subject_ids:
        matches = list(physiological_root.glob(f"*/{subject_id}.edf"))
        if len(matches) != 1:
            raise FileNotFoundError(
                f"Expected one EDF for {subject_id}, found {len(matches)}."
            )
        records.append(SubjectRecord(subject_id, matches[0]))
    return records


def _load_subject(record: SubjectRecord) -> LoadedSubject:
    edf = edfio.read_edf(record.edf_path, lazy_load_data=True)
    ecg_signal = next(
        (
            signal
            for signal in edf.signals
            if any(token in signal.label.lower() for token in ECG_TOKENS)
        ),
        None,
    )
    if ecg_signal is None:
        raise ValueError(f"No ECG channel found in {record.edf_path}.")

    respiration_channels = tuple(
        SignalChannel(
            signal.label,
            np.asarray(signal.data, dtype=float),
            float(signal.sampling_frequency),
        )
        for signal in edf.signals
        if get_respiration_feature_group(signal.label, DEFAULT_CSV_PATH)
        is not None
    )

    return LoadedSubject(
        record,
        SignalChannel(
            ecg_signal.label,
            np.asarray(ecg_signal.data, dtype=float),
            float(ecg_signal.sampling_frequency),
        ),
        respiration_channels,
    )


def _slice_channel(channel: SignalChannel, start_seconds: float) -> np.ndarray:
    start = int(round(start_seconds * channel.sampling_frequency))
    stop = int(
        round(
            (start_seconds + SEGMENT_DURATION_SECONDS)
            * channel.sampling_frequency
        )
    )
    return np.asarray(channel.data[start:stop], dtype=float)


class EcgInspectionViewer:
    def __init__(
        self,
        records: list[SubjectRecord],
        *,
        initial_window: int,
        display_seconds: float,
    ):
        self.records = records
        self.subject_index = 0
        self.window_index = max(0, initial_window)
        self.loaded_subject: LoadedSubject | None = None
        self.trace: CurrentEcgTrace | None = None
        self._updating_controls = False

        self.figure, axes = plt.subplots(
            6,
            1,
            figsize=(16, 15),
            gridspec_kw={
                "height_ratios": (2.0, 1.0, 1.0, 1.0, 1.0, 1.0)
            },
        )
        self.axes = list(axes)
        self.figure.subplots_adjust(
            left=0.22,
            right=0.98,
            top=0.95,
            bottom=0.08,
            hspace=0.48,
        )

        radio_axis = self.figure.add_axes((0.015, 0.56, 0.18, 0.34))
        labels = tuple(record.subject_id.replace("sub-", "") for record in records)
        self.subject_selector = RadioButtons(radio_axis, labels, active=0)
        self.subject_selector.on_clicked(self._select_subject)

        previous_subject_axis = self.figure.add_axes((0.015, 0.49, 0.08, 0.045))
        next_subject_axis = self.figure.add_axes((0.11, 0.49, 0.08, 0.045))
        self.previous_subject_button = Button(previous_subject_axis, "Subject -")
        self.next_subject_button = Button(next_subject_axis, "Subject +")
        self.previous_subject_button.on_clicked(
            lambda _event: self._step_subject(-1)
        )
        self.next_subject_button.on_clicked(
            lambda _event: self._step_subject(1)
        )

        previous_window_axis = self.figure.add_axes((0.015, 0.42, 0.08, 0.045))
        next_window_axis = self.figure.add_axes((0.11, 0.42, 0.08, 0.045))
        self.previous_window_button = Button(previous_window_axis, "Window -")
        self.next_window_button = Button(next_window_axis, "Window +")
        self.previous_window_button.on_clicked(
            lambda _event: self._step_window(-1)
        )
        self.next_window_button.on_clicked(
            lambda _event: self._step_window(1)
        )

        window_slider_axis = self.figure.add_axes((0.24, 0.075, 0.70, 0.025))
        self.window_slider = Slider(
            window_slider_axis,
            "Window",
            0,
            1,
            valinit=0,
            valstep=1,
        )
        self.window_slider.on_changed(self._select_window)

        view_slider_axis = self.figure.add_axes((0.24, 0.035, 0.70, 0.025))
        visible_end = min(float(display_seconds), SEGMENT_DURATION_SECONDS)
        self.view_slider = RangeSlider(
            view_slider_axis,
            "Visible seconds",
            0.0,
            float(SEGMENT_DURATION_SECONDS),
            valinit=(0.0, visible_end),
            valstep=1.0,
        )
        self.view_slider.on_changed(self._set_visible_range)

        self._load_selected_subject()
        self._draw()

    def _load_selected_subject(self):
        self.loaded_subject = _load_subject(self.records[self.subject_index])
        starts = self.loaded_subject.window_starts
        if starts.size == 0:
            raise ValueError(
                f"{self.loaded_subject.record.subject_id} is shorter than "
                f"{SEGMENT_DURATION_SECONDS} seconds."
            )
        self.window_index = min(self.window_index, starts.size - 1)
        self.window_slider.valmax = max(0, starts.size - 1)
        self.window_slider.ax.set_xlim(0, max(1, starts.size - 1))
        self._updating_controls = True
        self.window_slider.set_val(self.window_index)
        self._updating_controls = False

    def _select_subject(self, selected_label):
        target = next(
            index
            for index, record in enumerate(self.records)
            if record.subject_id.replace("sub-", "") == selected_label
        )
        if target == self.subject_index:
            return
        self.subject_index = target
        self.window_index = 0
        self._load_selected_subject()
        self._draw()

    def _step_subject(self, step):
        target = (self.subject_index + step) % len(self.records)
        self.subject_selector.set_active(target)

    def _select_window(self, value):
        if self._updating_controls:
            return
        target = int(round(value))
        if target == self.window_index:
            return
        self.window_index = target
        self._draw()

    def _step_window(self, step):
        assert self.loaded_subject is not None
        last = len(self.loaded_subject.window_starts) - 1
        target = min(max(self.window_index + step, 0), last)
        self.window_slider.set_val(target)

    def _set_visible_range(self, value):
        start, stop = value
        if stop <= start:
            return
        for axis in self.axes[:2]:
            axis.set_xlim(start, stop)
        self.figure.canvas.draw_idle()

    def _draw(self):
        assert self.loaded_subject is not None
        start_seconds = self.loaded_subject.window_starts[self.window_index]
        ecg_window = _slice_channel(self.loaded_subject.ecg, start_seconds)
        segment_data = {self.loaded_subject.ecg.label: ecg_window}
        segment_fs = {
            self.loaded_subject.ecg.label:
            self.loaded_subject.ecg.sampling_frequency
        }
        for channel in self.loaded_subject.respiration_channels:
            segment_data[channel.label] = _slice_channel(
                channel, start_seconds
            )
            segment_fs[channel.label] = channel.sampling_frequency
        selected_respiration = select_best_respiration_signal(
            segment_data,
            segment_fs,
            DEFAULT_CSV_PATH,
        )
        self.trace = inspect_current_ecg_features(
            ecg_window,
            self.loaded_subject.ecg.sampling_frequency,
            ECG_SEGMENT_FEATURE_LENGTH,
            respiration_signal=(
                selected_respiration.resampled_signal
                if selected_respiration is not None
                else None
            ),
            respiration_sampling_frequency=(
                selected_respiration.resampled_frequency
                if selected_respiration is not None
                else None
            ),
        )

        for axis in self.axes:
            axis.clear()
            axis.grid(True, alpha=0.25)

        quality = self.trace.quality
        if quality is None:
            quality_status = (
                "REJECTED" if self.trace.failure_reason else "NOT EVALUATED"
            )
            quality_color = (
                "tab:red" if self.trace.failure_reason else "0.4"
            )
        else:
            quality_status = "ACCEPTED" if quality.is_valid else "REJECTED"
            quality_color = (
                "tab:green" if quality.is_valid else "tab:red"
            )

        detector = self.trace.detector
        if detector is not None and self.trace.processed_fs is not None:
            sampling_frequency = self.trace.processed_fs
            clean_signal = self.trace.lowpass_filtered_signal
            clean_time = np.arange(clean_signal.size) / sampling_frequency
            self.axes[0].plot(
                clean_time,
                clean_signal,
                color="black",
                linewidth=0.8,
                label="Clean ECG",
            )

            r_indices = detector.r_locations
            r_times = r_indices / sampling_frequency
            valid_mask = (
                (r_indices >= 0)
                & (r_indices < clean_signal.size)
            )
            removed_mask = (
                self.trace.removed_detection_mask
                if self.trace.removed_detection_mask.size == r_times.size
                else np.zeros(r_times.size, dtype=bool)
            )
            kept_mask = valid_mask & ~removed_mask
            if np.any(kept_mask):
                kept_indices = r_indices[kept_mask]
                self.axes[0].scatter(
                    r_times[kept_mask],
                    clean_signal[kept_indices],
                    s=18,
                    color="tab:red",
                    label=(
                        "Refined R waves kept "
                        f"({np.count_nonzero(kept_mask)})"
                    ),
                    zorder=3,
                )
            visible_removed = valid_mask & removed_mask
            if np.any(visible_removed):
                removed_indices = r_indices[visible_removed]
                self.axes[0].scatter(
                    r_times[visible_removed],
                    clean_signal[removed_indices],
                    s=45,
                    color="tab:orange",
                    marker="x",
                    label=(
                        "Removed by removefp "
                        f"({np.count_nonzero(visible_removed)})"
                    ),
                    zorder=4,
                )

            if self.trace.raw_intervals.size:
                raw_interval_times = r_times[
                    1 : 1 + self.trace.raw_intervals.size
                ]
                self.axes[1].plot(
                    raw_interval_times,
                    self.trace.raw_intervals,
                    ".-",
                    color="0.55",
                    label="Raw RR",
                )
            if self.trace.intervals_after_removefp.size:
                cleaned_interval_times = self.trace.cleaned_event_times[1:]
                intervals_used_for_td = (
                    self.trace.intervals_after_removefp.copy()
                )
                intervals_used_for_td[
                    self.trace.interval_outlier_mask
                ] = np.nan
                self.axes[1].plot(
                    cleaned_interval_times,
                    intervals_used_for_td,
                    ".-",
                    color="tab:green",
                    label="RR used for TD metrics",
                )
                if np.any(self.trace.interval_outlier_mask):
                    self.axes[1].scatter(
                        cleaned_interval_times[
                            self.trace.interval_outlier_mask
                        ],
                        self.trace.intervals_after_removefp[
                            self.trace.interval_outlier_mask
                        ],
                        color="tab:orange",
                        marker="x",
                        s=45,
                        label=(
                            "Removed by medfilt_threshold "
                            f"({np.count_nonzero(self.trace.interval_outlier_mask)})"
                        ),
                        zorder=4,
                    )
        else:
            self.axes[0].text(
                0.5,
                0.5,
                "No valid cleaned ECG is available for this window.",
                transform=self.axes[0].transAxes,
                ha="center",
                va="center",
                color="tab:red",
            )

        frequency_domain = self.trace.frequency_domain
        if frequency_domain is not None:
            interval_times = frequency_domain.filled_event_times[1:]
            actual_intervals = np.diff(
                frequency_domain.filled_event_times
            )
            resolved = np.isfinite(frequency_domain.filled_intervals)
            self.axes[2].plot(
                interval_times[resolved],
                frequency_domain.filled_intervals[resolved],
                ".-",
                color="tab:blue",
                label="RR after fillgaps",
            )
            if np.any(~resolved):
                self.axes[2].scatter(
                    interval_times[~resolved],
                    actual_intervals[~resolved],
                    marker="x",
                    s=45,
                    color="tab:red",
                    label="Unresolved gap > 10 s",
                    zorder=4,
                )
            selected_start = frequency_domain.selected_event_times[0]
            selected_stop = frequency_domain.selected_event_times[-1]
            self.axes[2].axvspan(
                selected_start,
                selected_stop,
                color="tab:green",
                alpha=0.12,
                label="Longest continuous segment",
            )

            def standardize(values):
                values = np.asarray(values, dtype=float)
                scale = np.std(values)
                if not np.isfinite(scale) or scale == 0:
                    return values - np.mean(values)
                return (values - np.mean(values)) / scale

            self.axes[3].plot(
                frequency_domain.sample_times,
                standardize(frequency_domain.modulation),
                color="black",
                linewidth=0.8,
                label="IPFM modulation",
            )
            self.axes[3].plot(
                frequency_domain.sample_times,
                standardize(frequency_domain.respiration),
                color="tab:blue",
                linewidth=0.8,
                alpha=0.8,
                label=(
                    "Respiration: "
                    f"{selected_respiration.label if selected_respiration is not None else 'sloperange'}"
                ),
            )

            frequency_mask = frequency_domain.frequencies <= 1.0
            hrv_spectrum = frequency_domain.spectrum.copy()
            respiration_spectrum = (
                frequency_domain.respiration_spectrum.copy()
            )
            hrv_max = np.max(hrv_spectrum)
            respiration_max = np.max(respiration_spectrum)
            if hrv_max > 0:
                hrv_spectrum /= hrv_max
            if respiration_max > 0:
                respiration_spectrum /= respiration_max
            self.axes[4].plot(
                frequency_domain.frequencies[frequency_mask],
                hrv_spectrum[frequency_mask],
                color="black",
                label="HRV PSD (normalized)",
            )
            self.axes[4].plot(
                frequency_domain.frequencies[frequency_mask],
                respiration_spectrum[frequency_mask],
                color="tab:blue",
                alpha=0.8,
                label="Respiration PSD (normalized)",
            )
            self.axes[4].axvspan(
                0.04, 0.15, color="tab:orange", alpha=0.15, label="LF"
            )
            self.axes[4].axvspan(
                max(
                    0.0,
                    frequency_domain.respiration_frequency - 0.125,
                ),
                frequency_domain.respiration_frequency + 0.125,
                color="tab:green",
                alpha=0.12,
                label="Respiration-centered HF",
            )
            self.axes[4].axvline(
                frequency_domain.respiration_frequency,
                color="tab:blue",
                linestyle="--",
                linewidth=1,
            )

            osp_mask = frequency_domain.related_frequencies <= 1.0
            self.axes[5].plot(
                frequency_domain.related_frequencies[osp_mask],
                frequency_domain.related_spectrum[osp_mask],
                color="tab:green",
                label="Respiration-related PSD",
            )
            self.axes[5].plot(
                frequency_domain.related_frequencies[osp_mask],
                frequency_domain.unrelated_spectrum[osp_mask],
                color="tab:purple",
                label="Respiration-unrelated PSD",
            )
            self.axes[5].axvspan(
                0.04,
                0.15,
                color="tab:orange",
                alpha=0.15,
                label="Unrelated LF integration",
            )
        else:
            reason = (
                self.trace.frequency_failure_reason
                or "Frequency-domain processing was not available."
            )
            for axis in self.axes[2:]:
                axis.text(
                    0.5,
                    0.5,
                    reason,
                    transform=axis.transAxes,
                    ha="center",
                    va="center",
                    color="tab:red",
                )
        quality_lines = [f"ECG QUALITY: {quality_status}"]
        if quality is not None:
            quality_lines.extend(
                (
                    f"R detections: {quality.raw_detection_count} -> "
                    f"{quality.cleaned_detection_count}",
                    f"Removed FP: {quality.removed_fraction:.1%} "
                    f"(maximum {quality.maximum_removed_fraction:.1%})",
                    f"RR outliers: "
                    f"{np.count_nonzero(self.trace.interval_outlier_mask)}/"
                    f"{self.trace.intervals_after_removefp.size}",
                    f"Total RR excluded: "
                    f"{self.trace.removed_rr_percentage:.1f}%",
                    f"Amplitude spread: {quality.amplitude_spread_ratio:.2f} "
                    f"(maximum "
                    f"{quality.maximum_amplitude_spread_ratio:.2f})",
                )
            )
        if self.trace.failure_reason:
            quality_lines.append(self.trace.failure_reason)
        self.axes[0].text(
            0.99,
            0.97,
            "\n".join(quality_lines),
            transform=self.axes[0].transAxes,
            ha="right",
            va="top",
            color=quality_color,
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "0.8"},
            wrap=True,
        )

        for axis in self.axes:
            handles, _ = axis.get_legend_handles_labels()
            if handles:
                axis.legend(loc="upper left", fontsize=8)

        self.axes[0].set_ylabel("ECG")
        self.axes[0].set_title(
            "1. Clean ECG with refined R-wave detections"
        )
        self.axes[1].set_ylabel("RR (s)")
        self.axes[1].set_title("2. RR cleaning: removefp + medfilt_threshold")
        self.axes[2].set_ylabel("RR (s)")
        self.axes[2].set_title(
            "3. FD preparation: fillgaps and longest segment without >10 s gaps"
        )
        self.axes[3].set_ylabel("Standardized")
        self.axes[3].set_title("4. Evenly sampled IPFM and respiration")
        self.axes[3].set_xlabel("Seconds inside the processing window")
        self.axes[4].set_ylabel("Normalized PSD")
        self.axes[4].set_title(
            "5. Welch: 120 s Hamming windows, 50% overlap"
        )
        self.axes[4].set_xlabel("Frequency (Hz)")
        self.axes[4].set_xlim(0.0, 1.0)
        self.axes[5].set_ylabel("PSD")
        self.axes[5].set_title("6. OSP-related and unrelated spectra")
        self.axes[5].set_xlabel("Frequency (Hz)")
        self.axes[5].set_xlim(0.0, 1.0)
        if frequency_domain is not None:
            selected_duration = (
                frequency_domain.selected_event_times[-1]
                - frequency_domain.selected_event_times[0]
            )
            respiration_label = (
                selected_respiration.label
                if selected_respiration is not None
                else "sloperange"
            )
            self.axes[2].set_title(
                "3. FD preparation: fillgaps maxgap=10 s | "
                f"longest segment {selected_duration:.1f} s"
            )
            self.axes[3].set_title(
                "4. Evenly sampled IPFM and respiration | "
                f"source {respiration_label}"
            )
            self.axes[4].set_title(
                "5. Welch: 120 s Hamming, 50% overlap | "
                f"{frequency_domain.welch_window_count} windows"
            )
            self.axes[5].set_title(
                "6. OSP spectra | "
                f"UrLF={frequency_domain.metrics['URLF']:.4g}, "
                f"Re={frequency_domain.metrics['RE']:.4g}, "
                f"R={frequency_domain.metrics['R']:.3f}"
            )
        self.figure.suptitle(
            f"{self.loaded_subject.record.subject_id} | "
            f"window {self.window_index + 1}/"
            f"{len(self.loaded_subject.window_starts)} "
            f"at {start_seconds / 60:.1f} min | "
            f"ECG {self.loaded_subject.ecg.label} "
            f"{self.loaded_subject.ecg.sampling_frequency:g} Hz | "
            f"ECG quality {quality_status}",
            fontsize=12,
        )
        self._set_visible_range(self.view_slider.val)
        self.figure.canvas.draw_idle()


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-folder",
        type=Path,
        default=REPOSITORY_ROOT / "data" / "training_set",
    )
    parser.add_argument(
        "--subject",
        action="append",
        dest="subjects",
        help="EDF stem to include; repeat for multiple subjects.",
    )
    parser.add_argument("--window-index", type=int, default=0)
    parser.add_argument("--display-seconds", type=float, default=30.0)
    parser.add_argument("--save", type=Path)
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Render once without opening a GUI, for smoke tests.",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    if not args.no_show and INTERACTIVE_BACKEND is None:
        raise SystemExit(
            "No interactive Matplotlib backend is installed. Run "
            "'.venv\\Scripts\\python.exe -m pip install "
            "-r requirements-visualization.txt' and launch the viewer again."
        )

    subjects = tuple(args.subjects or DEFAULT_SUBJECTS)
    records = _find_records(args.data_folder.resolve(), subjects)
    viewer = EcgInspectionViewer(
        records,
        initial_window=args.window_index,
        display_seconds=args.display_seconds,
    )
    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        viewer.figure.savefig(args.save, dpi=150)
        print(f"Saved {args.save}")
    if args.no_show:
        plt.close(viewer.figure)
    else:
        plt.show()


if __name__ == "__main__":
    main()
