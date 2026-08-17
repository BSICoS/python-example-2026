"""Frequency-domain HRV features using the Biosigpy reconstruction flow."""

import numpy as np
from biosigpy.ecg import sloperange
from biosigpy.hrv import fdmetrics, fillgaps, ipfm, osp
from biosigpy.tools import lpd_filter, nan_filter
from scipy.integrate import trapezoid
from scipy.interpolate import PchipInterpolator
from scipy.signal import detrend, welch


IPFM_SAMPLING_FREQUENCY = 4.0
RESPIRATORY_HALF_BANDWIDTH_HZ = 0.125
MIN_RESPIRATORY_FREQUENCY_HZ = 0.1
MAX_RESPIRATORY_FREQUENCY_HZ = 0.5
FILLGAPS_MAX_GAP_SECONDS = 10.0
WELCH_WINDOW_SECONDS = 120.0
WELCH_OVERLAP_FRACTION = 0.5
WELCH_NFFT = 4096
MAX_SPECTRUM_FREQUENCY_HZ = 1.0
FREQUENCY_DOMAIN_METRIC_NAMES = (
    "LF",
    "HF_RESP",
    "LFN_RESP",
    "LFHF_RESP",
    "URLF",
    "RE",
    "R",
)


def _power_spectrum(signal, sampling_frequency):
    values = np.asarray(signal, dtype=float).flatten()
    window_length = int(round(WELCH_WINDOW_SECONDS * sampling_frequency))
    if values.size < window_length:
        raise ValueError(
            "At least 120 seconds are required for a Welch spectrum."
        )
    overlap = int(round(window_length * WELCH_OVERLAP_FRACTION))
    frequencies, spectrum = welch(
        values,
        fs=sampling_frequency,
        window=np.hamming(window_length),
        nperseg=window_length,
        noverlap=overlap,
        nfft=WELCH_NFFT,
        detrend=False,
    )
    selected = frequencies <= MAX_SPECTRUM_FREQUENCY_HZ
    return frequencies[selected], spectrum[selected]


def _largest_contiguous_segment(event_times, intervals):
    unresolved = np.flatnonzero(~np.isfinite(intervals))
    starts = np.concatenate(([0], unresolved + 1))
    stops = np.concatenate((unresolved + 1, [event_times.size]))
    candidates = [
        event_times[start:stop]
        for start, stop in zip(starts, stops)
        if stop - start >= 3
    ]
    if not candidates:
        raise ValueError("No contiguous R-wave segment contains three events.")
    return max(
        candidates,
        key=lambda values: (values[-1] - values[0], values.size),
    )


def _align_signal(signal, sampling_frequency, sample_times):
    values = np.asarray(signal, dtype=float).flatten()
    if values.size < 2 or sampling_frequency <= 0:
        raise ValueError("Respiration must contain at least two samples.")
    times = np.arange(values.size, dtype=float) / float(sampling_frequency)
    finite = np.isfinite(values)
    if np.count_nonzero(finite) < 2:
        raise ValueError("Respiration must contain at least two finite samples.")
    aligned = PchipInterpolator(
        times[finite], detrend(values[finite]), extrapolate=False
    )(sample_times)
    if np.any(~np.isfinite(aligned)):
        raise ValueError("Respiration does not cover the complete HRV grid.")
    return np.asarray(aligned, dtype=float)


def _derive_respiration(
    ecg_signal,
    ecg_sampling_frequency,
    event_times,
    sample_times,
):
    derivative_filter, _ = lpd_filter(
        ecg_sampling_frequency, 50.0, order=4
    )
    derivative_ecg = nan_filter(
        derivative_filter, [1.0], ecg_signal, max_gap=0
    )
    edr = sloperange(
        derivative_ecg, event_times, ecg_sampling_frequency
    ).edr
    finite = np.isfinite(edr)
    if np.count_nonzero(finite) < 2:
        raise ValueError("Slope-range produced too few finite samples.")
    aligned = PchipInterpolator(
        event_times[finite], detrend(edr[finite]), extrapolate=False
    )(sample_times)
    if np.any(~np.isfinite(aligned)):
        raise ValueError("Slope-range does not cover the complete HRV grid.")
    return np.asarray(aligned, dtype=float)


def _dominant_respiration_frequency(frequencies, spectrum):
    band = (
        (frequencies >= MIN_RESPIRATORY_FREQUENCY_HZ)
        & (frequencies <= MAX_RESPIRATORY_FREQUENCY_HZ)
    )
    if not np.any(band) or not np.any(spectrum[band] > 0):
        raise ValueError("No respiratory spectral peak was found.")
    band_indices = np.flatnonzero(band)
    return float(frequencies[band_indices[np.argmax(spectrum[band])]])


def _integrate_band(spectrum, frequencies, lower, upper):
    lower = max(float(lower), float(frequencies[0]))
    upper = min(float(upper), float(frequencies[-1]))
    if upper <= lower:
        return np.nan
    interior = (frequencies > lower) & (frequencies < upper)
    band_frequencies = np.concatenate(
        ([lower], frequencies[interior], [upper])
    )
    band_spectrum = np.interp(band_frequencies, frequencies, spectrum)
    return float(trapezoid(band_spectrum, band_frequencies))


def compute_frequency_domain_hrv(
    event_times,
    ecg_signal,
    ecg_sampling_frequency,
    *,
    respiration_signal=None,
    respiration_sampling_frequency=None,
):
    """Compute respiration-guided and OSP HRV metrics after gap filling."""

    cleaned_event_times = np.asarray(event_times, dtype=float).flatten()
    if cleaned_event_times.size < 3:
        raise ValueError("At least three cleaned R-wave times are required.")

    filled = fillgaps(
        cleaned_event_times,
        max_gap_duration=FILLGAPS_MAX_GAP_SECONDS,
    )
    selected_event_times = _largest_contiguous_segment(
        filled.tn, filled.dtn
    )
    modulation = ipfm(
        selected_event_times, IPFM_SAMPLING_FREQUENCY, return_m=True
    ).m
    sample_times = selected_event_times[0] + (
        np.arange(modulation.size, dtype=float) / IPFM_SAMPLING_FREQUENCY
    )

    if respiration_signal is None:
        respiration = _derive_respiration(
            np.asarray(ecg_signal, dtype=float),
            float(ecg_sampling_frequency),
            cleaned_event_times,
            sample_times,
        )
    else:
        respiration = _align_signal(
            respiration_signal,
            float(respiration_sampling_frequency),
            sample_times,
        )

    frequencies, spectrum = _power_spectrum(
        modulation, IPFM_SAMPLING_FREQUENCY
    )
    respiration_frequencies, respiration_spectrum = _power_spectrum(
        respiration, IPFM_SAMPLING_FREQUENCY
    )
    if not np.array_equal(respiration_frequencies, frequencies):
        raise ValueError("HRV and respiration spectra use different grids.")

    respiration_frequency = _dominant_respiration_frequency(
        respiration_frequencies, respiration_spectrum
    )
    lf = _integrate_band(spectrum, frequencies, 0.04, 0.15)
    hf = _integrate_band(
        spectrum,
        frequencies,
        respiration_frequency - RESPIRATORY_HALF_BANDWIDTH_HZ,
        respiration_frequency + RESPIRATORY_HALF_BANDWIDTH_HZ,
    )
    lfn = lf / (lf + hf) if lf > 0 and hf > 0 else np.nan
    lfhf = lf / hf if lf > 0 and hf > 0 else np.nan

    decomposition = osp(
        modulation,
        respiration,
        respiration_spectrum,
        frequencies,
        IPFM_SAMPLING_FREQUENCY,
    )
    related_frequencies, related_spectrum = _power_spectrum(
        decomposition.m_resp, IPFM_SAMPLING_FREQUENCY
    )
    unrelated_frequencies, unrelated_spectrum = _power_spectrum(
        decomposition.m_unrelated, IPFM_SAMPLING_FREQUENCY
    )
    if not np.array_equal(related_frequencies, unrelated_frequencies):
        raise ValueError("OSP component spectra use different grids.")
    separated = fdmetrics(
        f=related_frequencies,
        related_pxx=related_spectrum,
        unrelated_pxx=unrelated_spectrum,
    )

    return {
        "LF": float(lf),
        "HF_RESP": float(hf),
        "LFN_RESP": float(lfn),
        "LFHF_RESP": float(lfhf),
        "URLF": float(separated.urlf),
        "RE": float(separated.re),
        "R": float(separated.r),
    }
