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

from src.ecg_processing import ECG_SEGMENT_FEATURE_LENGTH, ECG_SEGMENT_FEATURE_NAMES
from src.lib.ecg_inspection import CurrentEcgTrace, inspect_current_ecg_features
from src.pipeline.config import (
    DEFAULT_CSV_PATH,
    SEGMENT_DURATION_SECONDS,
    SEGMENT_STRIDE_SECONDS,
)
from src.resp_processing import (
    get_respiration_feature_group,
    select_best_respiration_signal,
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


def _scaled(signal: np.ndarray) -> np.ndarray:
    finite = signal[np.isfinite(signal)]
    if finite.size == 0:
        return np.full(signal.shape, np.nan, dtype=float)
    center = np.median(finite)
    scale = np.percentile(np.abs(finite - center), 95)
    if not np.isfinite(scale) or scale == 0:
        scale = 1.0
    return (signal - center) / scale


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
            figsize=(16, 12),
            gridspec_kw={"height_ratios": (1.0, 1.0, 1.0, 1.0, 1.0, 0.8)},
        )
        self.axes = list(axes)
        self.figure.subplots_adjust(
            left=0.22,
            right=0.98,
            top=0.95,
            bottom=0.13,
            hspace=0.42,
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
        for axis in self.axes[:5]:
            axis.set_xlim(start, stop)
        self.figure.canvas.draw_idle()

    def _draw(self):
        assert self.loaded_subject is not None
        start_seconds = self.loaded_subject.window_starts[self.window_index]
        ecg_window = _slice_channel(self.loaded_subject.ecg, start_seconds)
        self.trace = inspect_current_ecg_features(
            ecg_window,
            self.loaded_subject.ecg.sampling_frequency,
            ECG_SEGMENT_FEATURE_LENGTH,
        )

        respiration_data = {
            channel.label: _slice_channel(channel, start_seconds)
            for channel in self.loaded_subject.respiration_channels
        }
        respiration_fs = {
            channel.label: channel.sampling_frequency
            for channel in self.loaded_subject.respiration_channels
        }
        selected_respiration = select_best_respiration_signal(
            respiration_data,
            respiration_fs,
            DEFAULT_CSV_PATH,
        )

        for axis in self.axes:
            axis.clear()
            axis.grid(True, alpha=0.25)

        raw_time = (
            np.arange(ecg_window.size)
            / self.loaded_subject.ecg.sampling_frequency
        )
        self.axes[0].plot(
            raw_time,
            _scaled(ecg_window),
            color="black",
            linewidth=0.8,
            label="Raw ECG",
        )
        if selected_respiration is not None:
            resp_time = (
                np.arange(selected_respiration.signal.size)
                / selected_respiration.sampling_frequency
            )
            self.axes[0].plot(
                resp_time,
                _scaled(selected_respiration.signal),
                color="tab:blue",
                linewidth=0.8,
                alpha=0.75,
                label=(
                    f"Resp: {selected_respiration.label} "
                    f"({selected_respiration.group}, "
                    f"quality {selected_respiration.quality:.3g})"
                ),
            )
        else:
            self.axes[0].text(
                0.99,
                0.88,
                "No eligible good respiration: sloperange fallback",
                transform=self.axes[0].transAxes,
                ha="right",
                va="top",
                color="tab:red",
            )
        self.axes[0].set_ylabel("Scaled")
        self.axes[0].set_title("1. Raw signals")
        self.axes[0].legend(loc="upper left", ncols=2)

        if self.trace.processed_fs is not None:
            processed_time = (
                np.arange(self.trace.resampled_signal.size)
                / self.trace.processed_fs
            )
            filter_series = (
                ("Resampled", self.trace.resampled_signal, "0.65"),
                ("60 Hz notch", self.trace.notch_filtered_signal, "tab:blue"),
                ("0.5 Hz high-pass", self.trace.highpass_filtered_signal, "tab:orange"),
                ("50 Hz low-pass", self.trace.lowpass_filtered_signal, "tab:red"),
            )
            for label, signal, color in filter_series:
                if signal.size:
                    self.axes[1].plot(
                        processed_time[: signal.size],
                        _scaled(signal),
                        linewidth=0.8,
                        alpha=0.8,
                        label=label,
                        color=color,
                    )
            self.axes[1].legend(loc="upper left", ncols=4, fontsize=8)
        self.axes[1].set_ylabel("Scaled")
        self.axes[1].set_title("2. Legacy preprocessing filters")

        detector = self.trace.detector
        if detector is not None:
            detector_time = (
                np.arange(detector.ecg_bandpassed.size)
                / self.trace.processed_fs
            )
            self.axes[2].plot(
                detector_time,
                _scaled(detector.ecg_bandpassed),
                label="5-12 Hz",
                linewidth=0.8,
            )
            self.axes[2].plot(
                detector_time,
                _scaled(detector.derivative),
                label="Derivative",
                linewidth=0.7,
            )
            self.axes[2].plot(
                detector_time,
                _scaled(detector.envelope),
                label="Integrated envelope",
                linewidth=1.0,
            )
            self.axes[2].legend(loc="upper left", ncols=3, fontsize=8)
            self.axes[2].set_ylabel("Scaled")
            self.axes[2].set_title("3. Legacy Pan-Tompkins internal signals")

            r_indices = detector.r_locations
            r_times = r_indices / self.trace.processed_fs
            detection_signal = detector.ecg_centered
            self.axes[3].plot(
                detector_time,
                detection_signal,
                color="black",
                linewidth=0.8,
            )
            unrefined_indices = detector.unrefined_r_locations
            valid_unrefined = (
                (unrefined_indices >= 0)
                & (unrefined_indices < detection_signal.size)
            )
            self.axes[3].scatter(
                unrefined_indices[valid_unrefined]
                / self.trace.processed_fs,
                detection_signal[unrefined_indices[valid_unrefined]],
                s=16,
                facecolors="none",
                edgecolors="tab:blue",
                label="Before snap_to_peak",
                zorder=2,
            )
            valid_mask = (
                (r_indices >= 0)
                & (r_indices < detection_signal.size)
            )
            valid_indices = r_indices[valid_mask]
            valid_times = r_times[valid_mask]
            self.axes[3].scatter(
                valid_times,
                detection_signal[valid_indices],
                s=18,
                color="tab:red",
                label=f"Refined R waves ({valid_indices.size})",
                zorder=3,
            )
            removed_mask = (
                self.trace.removed_detection_mask
                if self.trace.removed_detection_mask.size == r_times.size
                else np.zeros(r_times.size, dtype=bool)
            )
            visible_removed = removed_mask & valid_mask
            if np.any(visible_removed):
                removed_indices = r_indices[visible_removed]
                self.axes[3].scatter(
                    r_times[visible_removed],
                    detection_signal[removed_indices],
                    s=45,
                    color="tab:orange",
                    marker="x",
                    label=(
                        "Removed by removefp "
                        f"({np.count_nonzero(removed_mask)})"
                    ),
                    zorder=4,
                )
            self.axes[3].legend(loc="upper left")
            self.axes[3].set_ylabel("ECG")
            self.axes[3].set_title("4. Legacy refined R-wave detections")
            if self.trace.raw_intervals.size:
                raw_interval_times = r_times[
                    1 : 1 + self.trace.raw_intervals.size
                ]
                self.axes[4].plot(
                    raw_interval_times,
                    self.trace.raw_intervals,
                    ".-",
                    color="0.55",
                    label="Raw RR",
                )
            if self.trace.cleaned_intervals.size:
                cleaned_interval_times = self.trace.cleaned_event_times[1:]
                self.axes[4].plot(
                    cleaned_interval_times,
                    self.trace.cleaned_intervals,
                    ".-",
                    color="tab:green",
                    label="RR after removefp",
                )
            if (
                self.trace.raw_intervals.size
                or self.trace.cleaned_intervals.size
            ):
                self.axes[4].legend(loc="upper left", ncols=2, fontsize=8)
        self.axes[4].set_ylabel("RR (s)")
        self.axes[4].set_title("5. Biosigpy removefp (no fillgaps)")
        self.axes[4].set_xlabel("Seconds inside the processing window")

        quality = self.trace.quality
        if quality is None:
            quality_status = (
                "REJECTED" if self.trace.failure_reason else "NOT EVALUATED"
            )
            quality_color = "tab:red" if self.trace.failure_reason else "0.4"
        else:
            quality_status = "ACCEPTED" if quality.is_valid else "REJECTED"
            quality_color = "tab:green" if quality.is_valid else "tab:red"

        metric_lines = [f"ECG QUALITY: {quality_status}"]
        if quality is not None:
            metric_lines.extend(
                (
                    f"R detections: {quality.raw_detection_count} raw -> "
                    f"{quality.cleaned_detection_count} after removefp "
                    f"[{quality.minimum_detections}, "
                    f"{quality.maximum_detections}]",
                    f"Removed detections: "
                    f"{quality.removed_detection_count}/"
                    f"{quality.raw_detection_count} "
                    f"({quality.removed_fraction:.1%}; "
                    f"maximum {quality.maximum_removed_fraction:.1%})",
                    f"ECG amplitude spread: "
                    f"{quality.amplitude_spread_ratio:.2f} "
                    f"(maximum "
                    f"{quality.maximum_amplitude_spread_ratio:.2f})",
                )
            )
        if self.trace.features is not None:
            metric_lines.extend(
                f"{name}: {value:.5g}"
                for name, value in zip(
                    ECG_SEGMENT_FEATURE_NAMES,
                    self.trace.features,
                )
            )
        metric_lines.extend(
            (
                f"Removed by removefp: "
                f"{self.trace.removed_percentage:.2f}%",
                "TD units: AVNN s; SDNN/RMSSD ms",
            )
        )
        if self.trace.failure_reason:
            metric_lines.append(f"Diagnostic: {self.trace.failure_reason}")
        self.axes[5].axis("off")
        self.axes[5].text(
            0.01,
            0.98,
            "\n".join(metric_lines),
            transform=self.axes[5].transAxes,
            va="top",
            family="monospace",
            fontsize=9,
        )
        self.axes[5].set_title(
            "6. Current per-window feature values",
            color=quality_color,
        )

        respiration_label = (
            (
                f"{selected_respiration.label}/"
                f"{selected_respiration.group}"
            )
            if selected_respiration is not None
            else "sloperange fallback"
        )
        self.figure.suptitle(
            f"{self.loaded_subject.record.subject_id} | "
            f"window {self.window_index + 1}/{len(self.loaded_subject.window_starts)} "
            f"at {start_seconds / 60:.1f} min | "
            f"ECG {self.loaded_subject.ecg.label} "
            f"{self.loaded_subject.ecg.sampling_frequency:g} Hz | "
            f"respiration {respiration_label} | "
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
