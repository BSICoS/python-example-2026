from unittest.mock import patch

import numpy as np
from scipy.signal import welch

from src.lib.ecg_frequency_features import (
    FILLGAPS_MAX_GAP_SECONDS,
    MAX_SPECTRUM_FREQUENCY_HZ,
    RESPIRATORY_HALF_BANDWIDTH_HZ,
    WELCH_OVERLAP_FRACTION,
    WELCH_NFFT,
    WELCH_WINDOW_SECONDS,
    compute_frequency_domain_hrv,
)


def _signals(duration_seconds=300, ecg_fs=200, respiration_fs=10):
    ecg_time = np.arange(duration_seconds * ecg_fs) / ecg_fs
    respiration_time = (
        np.arange(duration_seconds * respiration_fs) / respiration_fs
    )
    respiration = np.sin(2 * np.pi * 0.25 * respiration_time)
    ecg = np.sin(2 * np.pi * 1.0 * ecg_time) * (
        1.0 + 0.2 * np.sin(2 * np.pi * 0.25 * ecg_time)
    )
    event_indices = np.arange(1, duration_seconds - 1, dtype=float)
    event_times = event_indices + 0.04 * np.sin(
        2 * np.pi * 0.25 * event_indices
    )
    return ecg, respiration, event_times


def test_frequency_domain_flow_fills_gaps_only_here_and_uses_direct_respiration():
    ecg, respiration, event_times = _signals()
    event_times = np.delete(event_times, 100)

    with patch(
        "src.lib.ecg_frequency_features.fillgaps",
        wraps=__import__("biosigpy.hrv", fromlist=["fillgaps"]).fillgaps,
    ) as fill, patch(
        "src.lib.ecg_frequency_features.welch",
        wraps=welch,
    ) as spectrum:
        result = compute_frequency_domain_hrv(
            event_times,
            ecg,
            200,
            respiration_signal=respiration,
            respiration_sampling_frequency=10,
        )

    fill.assert_called_once()
    assert fill.call_args.kwargs["max_gap_duration"] == 10.0
    assert FILLGAPS_MAX_GAP_SECONDS == 10.0
    assert spectrum.call_count == 4
    for call in spectrum.call_args_list:
        assert call.kwargs["nperseg"] == 120 * 4
        assert call.kwargs["noverlap"] == 60 * 4
        assert call.kwargs["nfft"] == 4096
        assert call.kwargs["window"].size == 120 * 4
    assert result.filled_event_times.size > event_times.size
    assert result.respiration_source == "direct"
    assert np.isclose(result.respiration_frequency, 0.25)
    assert set(result.metrics) == {
        "LF",
        "HF_RESP",
        "LFN_RESP",
        "LFHF_RESP",
        "URLF",
        "RE",
        "R",
    }
    assert RESPIRATORY_HALF_BANDWIDTH_HZ == 0.125
    assert WELCH_WINDOW_SECONDS == 120.0
    assert WELCH_OVERLAP_FRACTION == 0.5
    assert WELCH_NFFT == 4096
    assert MAX_SPECTRUM_FREQUENCY_HZ == 1.0
    assert result.frequencies.size == 1025
    assert result.frequencies[0] == 0.0
    assert result.frequencies[-1] == 1.0


def test_frequency_domain_flow_uses_sloperange_only_without_direct_respiration():
    ecg, _respiration, event_times = _signals()

    with patch(
        "src.lib.ecg_frequency_features.sloperange",
        wraps=__import__("biosigpy.ecg", fromlist=["sloperange"]).sloperange,
    ) as derive:
        result = compute_frequency_domain_hrv(event_times, ecg, 200)

    derive.assert_called_once()
    assert result.respiration_source == "sloperange"


def test_frequency_domain_uses_longest_segment_around_unresolved_large_gap():
    ecg, respiration, _event_times = _signals()
    event_times = np.concatenate(
        (np.arange(1.0, 81.0), np.arange(96.0, 299.0))
    )

    result = compute_frequency_domain_hrv(
        event_times,
        ecg,
        200,
        respiration_signal=respiration,
        respiration_sampling_frequency=10,
    )

    assert np.any(~np.isfinite(result.filled_intervals))
    assert result.selected_event_times[0] == 96.0
    assert result.selected_event_times[-1] == 298.0
    assert result.welch_window_count == 2
