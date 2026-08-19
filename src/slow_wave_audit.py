"""Development-only methodological audit of the production slow-wave detector."""

from __future__ import annotations

import argparse
from pathlib import Path

import edfio
import numpy as np
import pandas as pd
from tqdm import tqdm

from helper_code import HEADERS
from src.eeg_processing import (
    EEG_CHANNEL_SPECS,
    _get_eeg_aliases,
    get_eeg_channel_source_labels,
    prepare_slow_wave_detector_input,
)
from src.lib import eeg_features
from src.pipeline.config import (
    DEFAULT_CSV_PATH,
    SEGMENT_DURATION_SECONDS,
    SEGMENT_STRIDE_SECONDS,
)

STAGE_NAMES = {1: 'N3', 2: 'N2', 3: 'N1', 4: 'REM', 5: 'Wake', 9: 'unavailable'}
STAGE_ORDER = ('N3', 'N2', 'N1', 'REM', 'Wake', 'unavailable')
PROBABILITY_LABELS = (
    'caisr_prob_n3', 'caisr_prob_n2', 'caisr_prob_n1',
    'caisr_prob_r', 'caisr_prob_w',
)
SOFT_STAGE_PROBABILITIES = {
    'N3': 'caisr_prob_n3',
    'N2': 'caisr_prob_n2',
    'N1': 'caisr_prob_n1',
    'REM': 'caisr_prob_r',
    'Wake': 'caisr_prob_w',
}
SOFT_STAGE_ORDER = ('N3', 'N2', 'N1', 'REM', 'Wake', 'NREM')
WEIGHT_COLUMNS = tuple(f'weight_{stage}' for stage in SOFT_STAGE_ORDER)
EVENT_COLUMNS = (
    'patient_id', 'bids_folder', 'site_id', 'session_id', 'channel',
    'segment_start_seconds', 'segment_end_seconds', 'down_crossing_seconds',
    'trough_seconds', 'up_crossing_seconds', 'stage_at_trough',
    *PROBABILITY_LABELS, *WEIGHT_COLUMNS,
    'negative_peak_amplitude', 'peak_to_peak_amplitude',
    'negative_slope', 'positive_slope', 'negative_half_wave_duration_seconds',
    'detector_amplitude_threshold', 'detector_data_deviation',
    'detector_slope_threshold', 'source_labels', 'sampling_frequency',
)
SEGMENT_COLUMNS = (
    'patient_id', 'bids_folder', 'site_id', 'session_id', 'channel',
    'segment_start_seconds', 'segment_end_seconds', 'fraction_Wake', 'fraction_N1',
    'fraction_N2', 'fraction_N3', 'fraction_REM', 'fraction_unavailable',
    'TotalSW', 'SWdensity', 'detector_amplitude_threshold',
    'detector_data_deviation', 'detector_slope_threshold',
    *tuple(f'weighted_minutes_{stage}' for stage in SOFT_STAGE_ORDER),
    *tuple(f'weighted_SW_count_{stage}' for stage in ('N3', 'N2', 'NREM', 'REM', 'Wake')),
    *tuple(f'weighted_SW_per_min_{stage}' for stage in ('N3', 'N2', 'NREM', 'REM', 'Wake')),
)


def translate_stage_code(value):
    """Translate CAISR's numeric stage code without inventing missing stages."""
    try:
        code = int(float(value))
    except (TypeError, ValueError, OverflowError):
        return 'unavailable'
    return STAGE_NAMES.get(code, 'unavailable')


def build_segment_intervals(recording_duration_seconds, stride_seconds=SEGMENT_STRIDE_SECONDS):
    """Return full production-length intervals at the requested development stride."""
    last_start = float(recording_duration_seconds) - SEGMENT_DURATION_SECONDS
    if last_start < 0:
        return []
    starts = np.arange(0.0, last_start + 1e-9, float(stride_seconds))
    return [(float(start), float(start + SEGMENT_DURATION_SECONDS)) for start in starts]


def _empty_annotation():
    return {'available': False, 'fs': 1.0 / 30.0, 'stage': np.array([], dtype=float),
            'probabilities': {name: np.array([], dtype=float) for name in PROBABILITY_LABELS}}


def load_caisr_annotation(annotation_path):
    """Read stage_caisr and its five probabilities from the real annotation EDF."""
    if annotation_path is None or not Path(annotation_path).is_file():
        return _empty_annotation()
    try:
        edf = edfio.read_edf(annotation_path, lazy_load_data=True)
        signals = {signal.label.lower().strip(): signal for signal in edf.signals}
        stage_signal = signals.get('stage_caisr')
        if stage_signal is None:
            return _empty_annotation()
        stage = np.asarray(stage_signal.data, dtype=float)
        probabilities = {}
        for label in PROBABILITY_LABELS:
            signal = signals.get(label)
            probabilities[label] = (
                np.asarray(signal.data, dtype=float) if signal is not None
                else np.full(stage.size, np.nan, dtype=float)
            )
        return {'available': True, 'fs': float(stage_signal.sampling_frequency),
                'stage': stage, 'probabilities': probabilities}
    except Exception:
        return _empty_annotation()


def annotation_at_time(annotation, time_seconds):
    """Return CAISR stage/probabilities for the epoch containing an event time."""
    result = {'stage_at_trough': 'unavailable'}
    result.update({label: np.nan for label in PROBABILITY_LABELS})
    result.update({label: np.nan for label in WEIGHT_COLUMNS})
    if not annotation['available'] or time_seconds < 0:
        return result
    index = int(np.floor(float(time_seconds) * annotation['fs']))
    if index < 0 or index >= annotation['stage'].size:
        return result
    stage_name = translate_stage_code(annotation['stage'][index])
    result['stage_at_trough'] = stage_name
    for label, values in annotation['probabilities'].items():
        if index < values.size:
            probability = float(values[index])
            if np.isfinite(probability) and 0.0 <= probability <= 1.0:
                result[label] = probability
    for stage, label in SOFT_STAGE_PROBABILITIES.items():
        result[f'weight_{stage}'] = result[label]
    n2 = result['weight_N2']
    n3 = result['weight_N3']
    result['weight_NREM'] = n2 + n3 if np.isfinite(n2) and np.isfinite(n3) else np.nan
    return result


def stage_minutes_in_interval(annotation, start_seconds, end_seconds):
    """Measure actual CAISR epoch overlap with an arbitrary time interval."""
    minutes = {name: 0.0 for name in STAGE_ORDER}
    if end_seconds <= start_seconds:
        return minutes
    if not annotation['available']:
        minutes['unavailable'] = (end_seconds - start_seconds) / 60.0
        return minutes
    epoch_seconds = 1.0 / annotation['fs']
    first = max(0, int(np.floor(start_seconds / epoch_seconds)))
    last = min(annotation['stage'].size, int(np.ceil(end_seconds / epoch_seconds)))
    covered = 0.0
    for index in range(first, last):
        overlap = max(0.0, min(end_seconds, (index + 1) * epoch_seconds)
                      - max(start_seconds, index * epoch_seconds))
        if overlap:
            minutes[translate_stage_code(annotation['stage'][index])] += overlap / 60.0
            covered += overlap
    missing = max(0.0, end_seconds - start_seconds - covered)
    minutes['unavailable'] += missing / 60.0
    return minutes


def weighted_stage_minutes_in_interval(annotation, start_seconds, end_seconds):
    """Integrate CAISR probabilities over an interval, including partial epochs."""
    minutes = {stage: 0.0 for stage in SOFT_STAGE_ORDER}
    if end_seconds <= start_seconds or not annotation['available']:
        return minutes
    epoch_seconds = 1.0 / annotation['fs']
    first = max(0, int(np.floor(start_seconds / epoch_seconds)))
    last = min(annotation['stage'].size, int(np.ceil(end_seconds / epoch_seconds)))
    for index in range(first, last):
        overlap_minutes = max(
            0.0,
            min(end_seconds, (index + 1) * epoch_seconds)
            - max(start_seconds, index * epoch_seconds),
        ) / 60.0
        if overlap_minutes == 0:
            continue
        epoch_probabilities = {}
        for stage, label in SOFT_STAGE_PROBABILITIES.items():
            values = annotation['probabilities'].get(label, np.array([], dtype=float))
            probability = float(values[index]) if index < values.size else np.nan
            if np.isfinite(probability) and 0.0 <= probability <= 1.0:
                minutes[stage] += probability * overlap_minutes
                epoch_probabilities[stage] = probability
        if 'N2' in epoch_probabilities and 'N3' in epoch_probabilities:
            minutes['NREM'] += (
                epoch_probabilities['N2'] + epoch_probabilities['N3']
            ) * overlap_minutes
    return minutes


def caisr_uncertainty_metrics(annotation):
    """Summarize CAISR confidence without changing or renormalizing probabilities."""
    missing = {
        'caisr_probability_epochs': 0,
        'median_max_stage_probability': np.nan,
        'P10_max_stage_probability': np.nan,
        'fraction_epochs_max_probability_below_0_5': np.nan,
        'fraction_epochs_max_probability_below_0_7': np.nan,
        'mean_stage_entropy': np.nan,
    }
    if not annotation['available']:
        return missing
    columns = []
    for label in PROBABILITY_LABELS:
        values = annotation['probabilities'].get(label)
        if values is None:
            return missing
        columns.append(np.asarray(values, dtype=float))
    if not columns:
        return missing
    length = min(len(values) for values in columns)
    probabilities = np.column_stack([values[:length] for values in columns])
    valid = np.all(np.isfinite(probabilities) & (probabilities >= 0) & (probabilities <= 1), axis=1)
    probabilities = probabilities[valid]
    if probabilities.size == 0:
        return missing
    maxima = np.max(probabilities, axis=1)
    log_probabilities = np.zeros_like(probabilities)
    positive = probabilities > 0
    log_probabilities[positive] = np.log(probabilities[positive])
    entropy_terms = probabilities * log_probabilities
    return {
        'caisr_probability_epochs': int(len(probabilities)),
        'median_max_stage_probability': float(np.median(maxima)),
        'P10_max_stage_probability': float(np.percentile(maxima, 10)),
        'fraction_epochs_max_probability_below_0_5': float(np.mean(maxima < 0.5)),
        'fraction_epochs_max_probability_below_0_7': float(np.mean(maxima < 0.7)),
        'mean_stage_entropy': float(np.mean(-np.sum(entropy_terms, axis=1))),
    }


def _safe_rate(count, minutes):
    return float(count / minutes) if minutes > 0 else np.nan


def _scalar(value):
    values = np.asarray(value, dtype=float).reshape(-1)
    finite = values[np.isfinite(values)]
    return float(finite[0]) if finite.size else np.nan


def physical_signal_statistics(signal):
    """Return robust statistics in the EDF signal's original physical domain."""
    finite = np.asarray(signal, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {name: np.nan for name in
                ('signal_P1', 'signal_median', 'signal_P99', 'signal_P99_minus_P1')}
    p1, median, p99 = np.percentile(finite, [1, 50, 99])
    return {'signal_P1': float(p1), 'signal_median': float(median),
            'signal_P99': float(p99), 'signal_P99_minus_P1': float(p99 - p1)}


def prepare_audit_slow_wave_detector_input(signal, fs):
    """Delegate audit preparation to the exact production SW input function."""
    return prepare_slow_wave_detector_input(signal, fs)


def detect_audited_slow_waves(signal, fs):
    """Prepare and run the detector while retaining its exact input signal."""
    prepared = prepare_audit_slow_wave_detector_input(signal, fs)
    if prepared is None:
        return None
    detector_signal, detector_fs = prepared
    detection = eeg_features.detect_slow_waves(detector_signal, detector_fs)
    return detection, detector_signal, detector_fs


def _event_window(filtered_signal, trough_index, fs, half_seconds=2.0):
    half_samples = int(round(half_seconds * fs))
    start = int(trough_index) - half_samples
    end = int(trough_index) + half_samples + 1
    if start < 0 or end > len(filtered_signal):
        return None
    return np.asarray(filtered_signal[start:end], dtype=float)


class TriggeredWaveforms:
    """Exact running mean plus bounded deterministic samples for quantiles."""

    def __init__(self, max_quantile_samples=20000, waveform_domain='sanitized_resampled_eeg'):
        self.max_quantile_samples = int(max_quantile_samples)
        self.groups = {}
        self.weighted_nrem_groups = {}
        self.rng = np.random.default_rng(20260819)
        self.waveform_domain = str(waveform_domain)

    def add(self, key, waveform):
        waveform = np.asarray(waveform, dtype=float)
        group = self.groups.setdefault(key, {'n': 0, 'sum': np.zeros_like(waveform), 'sample': []})
        group['n'] += 1
        group['sum'] += waveform
        if len(group['sample']) < self.max_quantile_samples:
            group['sample'].append(waveform.copy())
        else:
            replacement = int(self.rng.integers(0, group['n']))
            if replacement < self.max_quantile_samples:
                group['sample'][replacement] = waveform.copy()

    def add_weighted_nrem(self, key, waveform, weight):
        """Accumulate an NREM probability-weighted detector-input waveform."""
        if not np.isfinite(weight) or weight <= 0:
            return
        waveform = np.asarray(waveform, dtype=float)
        group = self.weighted_nrem_groups.setdefault(
            key, {'weight': 0.0, 'weighted_sum': np.zeros_like(waveform)})
        group['weight'] += float(weight)
        group['weighted_sum'] += float(weight) * waveform

    def save(self, path, fs=200.0):
        payload = {
            'time_seconds': np.arange(-int(2 * fs), int(2 * fs) + 1) / fs,
            'waveform_domain': np.asarray(self.waveform_domain),
        }
        for (site, channel, stage), group in sorted(self.groups.items()):
            prefix = f'{site}__{channel.replace("-", "_")}__{stage}'
            sample = np.asarray(group['sample'], dtype=float)
            payload[f'{prefix}__mean'] = group['sum'] / group['n']
            payload[f'{prefix}__median'] = np.median(sample, axis=0)
            payload[f'{prefix}__p25'] = np.percentile(sample, 25, axis=0)
            payload[f'{prefix}__p75'] = np.percentile(sample, 75, axis=0)
            payload[f'{prefix}__n_events'] = np.asarray(group['n'])
            payload[f'{prefix}__quantile_sample_size'] = np.asarray(len(sample))
        for (site, channel), group in sorted(self.weighted_nrem_groups.items()):
            prefix = f'{site}__{channel.replace("-", "_")}__weighted_NREM'
            payload[f'{prefix}__mean_waveform'] = group['weighted_sum'] / group['weight']
            payload[f'{prefix}__total_effective_weight'] = np.asarray(group['weight'])
        np.savez_compressed(path, **payload)


def _distribution(values):
    values = np.asarray(list(values), dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {'N': 0, 'median': np.nan, 'IQR': np.nan, 'P10': np.nan, 'P90': np.nan}
    p10, p25, median, p75, p90 = np.percentile(values, [10, 25, 50, 75, 90])
    return {'N': int(values.size), 'median': float(median), 'IQR': float(p75 - p25),
            'P10': float(p10), 'P90': float(p90)}


def weighted_event_metrics(events, weighted_minutes):
    """Calculate probability-weighted event counts and densities by soft stage."""
    metrics = {}
    for stage in ('N3', 'N2', 'NREM', 'REM', 'Wake'):
        weights = np.asarray(
            [event.get(f'weight_{stage}', np.nan) for event in events], dtype=float)
        count = float(np.sum(weights[np.isfinite(weights)]))
        minutes = float(weighted_minutes.get(stage, 0.0))
        metrics[f'weighted_SW_count_{stage}'] = count
        metrics[f'weighted_SW_per_min_{stage}'] = _safe_rate(count, minutes)
    return metrics


def _weighted_mean(values, weights):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights >= 0)
    total_weight = float(np.sum(weights[valid]))
    if total_weight <= 0:
        return np.nan, 0.0
    return float(np.sum(values[valid] * weights[valid]) / total_weight), total_weight


def _find_annotation_path(data_folder, site_id, bids_folder, session_id):
    return Path(data_folder, 'algorithmic_annotations', site_id,
                f'{bids_folder}_ses-{session_id}_caisr_annotations.edf')


def _find_physiological_path(data_folder, site_id, bids_folder, session_id):
    return Path(data_folder, 'physiological_data', site_id,
                f'{bids_folder}_ses-{session_id}.edf')


def _selected_signal(edf, channel, eeg_aliases):
    labels = [signal.label.lower().strip() for signal in edf.signals]
    signals = dict(zip(labels, edf.signals))
    frequencies = {label: float(signal.sampling_frequency) for label, signal in signals.items()}
    source_labels = get_eeg_channel_source_labels(channel, labels, frequencies, eeg_aliases)
    if source_labels is None:
        return None
    first = signals[source_labels[0]]
    data = np.asarray(first.data, dtype=float)
    unit = str(getattr(first, 'physical_dimension', '') or '')
    if len(source_labels) == 2:
        second = signals[source_labels[1]]
        data = data - np.asarray(second.data, dtype=float)
        second_unit = str(getattr(second, 'physical_dimension', '') or '')
        if second_unit != unit:
            unit = f'{unit}-{second_unit}'
    return data, frequencies[source_labels[0]], unit, ';'.join(source_labels)


def _record_duration(edf):
    """Read EDF duration metadata without loading unrelated signal arrays."""
    return float(edf.duration)


def _subject_rows_for_channel(record, channel, annotation, duration, intervals, events):
    total = stage_minutes_in_interval(annotation, 0.0, duration)
    total_weighted = weighted_stage_minutes_in_interval(annotation, 0.0, duration)
    analyzed = {name: 0.0 for name in STAGE_ORDER}
    analyzed_weighted = {name: 0.0 for name in SOFT_STAGE_ORDER}
    for start, end in intervals:
        exposure = stage_minutes_in_interval(annotation, start, end)
        weighted_exposure = weighted_stage_minutes_in_interval(annotation, start, end)
        for name in STAGE_ORDER:
            analyzed[name] += exposure[name]
        for name in SOFT_STAGE_ORDER:
            analyzed_weighted[name] += weighted_exposure[name]
    counts = {name: 0 for name in STAGE_ORDER}
    for event in events:
        counts[event['stage_at_trough']] += 1
    row = {
        **record, 'channel': channel, 'annotation_available': bool(annotation['available']),
        'number_of_detected_slow_waves': len(events),
        'total_recording_minutes': duration / 60.0,
        'total_N2_minutes': total['N2'], 'total_N3_minutes': total['N3'],
        'total_N2_N3_minutes': total['N2'] + total['N3'],
        'analyzed_minutes': sum(analyzed.values()),
        'analyzed_N2_minutes': analyzed['N2'], 'analyzed_N3_minutes': analyzed['N3'],
        'analyzed_N2_N3_minutes': analyzed['N2'] + analyzed['N3'],
        'total_weighted_NREM_minutes': total_weighted['NREM'],
        'analyzed_weighted_NREM_minutes': analyzed_weighted['NREM'],
    }
    row['analyzed_N2_N3_fraction'] = (
        row['analyzed_N2_N3_minutes'] / row['total_N2_N3_minutes']
        if row['total_N2_N3_minutes'] > 0 else np.nan)
    row['analyzed_weighted_NREM_fraction'] = (
        row['analyzed_weighted_NREM_minutes'] / row['total_weighted_NREM_minutes']
        if row['total_weighted_NREM_minutes'] > 0 else np.nan)
    for name in STAGE_ORDER:
        row[f'number_in_{name}'] = counts[name]
        row[f'analyzed_minutes_{name}'] = analyzed[name]
        row[f'SW_per_min_{name}'] = _safe_rate(counts[name], analyzed[name])
    row['SW_per_min_N3'] = _safe_rate(counts['N3'], analyzed['N3'])
    row['SW_per_min_N2'] = _safe_rate(counts['N2'], analyzed['N2'])
    row['SW_per_min_N2_N3'] = _safe_rate(counts['N2'] + counts['N3'], analyzed['N2'] + analyzed['N3'])
    row['SW_per_min_REM'] = _safe_rate(counts['REM'], analyzed['REM'])
    row['SW_per_min_Wake'] = _safe_rate(counts['Wake'], analyzed['Wake'])
    for name in SOFT_STAGE_ORDER:
        row[f'weighted_minutes_{name}'] = analyzed_weighted[name]
    row.update(weighted_event_metrics(events, analyzed_weighted))
    return row


def _build_stage_summary(events_df, subjects_df):
    rows = []
    morphology = {
        'peak_to_peak_amplitude': 'peak_to_peak_amplitude',
        'negative_peak_amplitude': 'negative_peak_amplitude',
        'negative_slope': 'negative_slope', 'positive_slope': 'positive_slope',
        'negative_half_wave_duration': 'negative_half_wave_duration_seconds',
        'detector_amplitude_threshold': 'detector_amplitude_threshold',
        'detector_slope_threshold': 'detector_slope_threshold',
    }
    sites = sorted(set(subjects_df.get('site_id', [])))
    for site in sites:
        for channel in EEG_CHANNEL_SPECS:
            subject_group = subjects_df[(subjects_df.site_id == site) & (subjects_df.channel == channel)]
            for stage in STAGE_ORDER:
                event_group = events_df[(events_df.site_id == site) & (events_df.channel == channel)
                                        & (events_df.stage_at_trough == stage)]
                rate_column = f'SW_per_min_{stage}'
                for metric, column in [('SW_per_min', rate_column), *morphology.items()]:
                    values = subject_group[rate_column] if metric == 'SW_per_min' else event_group.get(column, [])
                    rows.append({'site_id': site, 'channel': channel, 'stage': stage,
                                 'staging_method': 'hard',
                                 'metric': metric, **_distribution(values)})
            soft_morphology = {
                'peak_to_peak_amplitude': 'peak_to_peak_amplitude',
                'negative_slope': 'negative_slope',
                'positive_slope': 'positive_slope',
                'negative_half_wave_duration': 'negative_half_wave_duration_seconds',
            }
            soft_events = events_df[(events_df.site_id == site) & (events_df.channel == channel)]
            for stage in ('N2', 'N3', 'NREM', 'REM', 'Wake'):
                rate_values = subject_group.get(f'weighted_SW_per_min_{stage}', [])
                rows.append({'site_id': site, 'channel': channel, 'stage': stage,
                             'staging_method': 'soft', 'metric': 'weighted_SW_per_min',
                             **_distribution(rate_values)})
                weights = soft_events.get(f'weight_{stage}', pd.Series(dtype=float))
                for metric, column in soft_morphology.items():
                    weighted_mean, total_weight = _weighted_mean(
                        soft_events.get(column, []), weights)
                    rows.append({'site_id': site, 'channel': channel, 'stage': stage,
                                 'staging_method': 'soft', 'metric': metric,
                                 'N': int(np.sum(np.isfinite(np.asarray(weights, dtype=float)))),
                                 'total_effective_weight': total_weight,
                                 'weighted_mean': weighted_mean})
    return pd.DataFrame(rows)


def _aggregate_summary(subjects_df, events_df, segments_df, group_columns):
    rows = []
    if subjects_df.empty:
        return pd.DataFrame()
    for keys, group in subjects_df.groupby(group_columns, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_columns, keys))
        record_group = group.drop_duplicates(['patient_id', 'session_id'])
        row['records'] = int(len(record_group))
        row['annotation_coverage'] = float(record_group.annotation_available.mean())
        for stage in ('N2', 'N3', 'REM', 'Wake'):
            count = group[f'number_in_{stage}'].sum()
            channel_minutes = group[f'analyzed_minutes_{stage}'].sum()
            row[f'analyzed_{stage}_minutes'] = float(
                record_group[f'analyzed_minutes_{stage}'].sum())
            row[f'SW_per_min_{stage}'] = _safe_rate(count, channel_minutes)
        nrem_count = group.number_in_N2.sum() + group.number_in_N3.sum()
        channel_nrem_minutes = group.analyzed_minutes_N2.sum() + group.analyzed_minutes_N3.sum()
        row['total_N2_N3_minutes'] = float(
            record_group.get('total_N2_N3_minutes', pd.Series(dtype=float)).sum())
        row['analyzed_N2_N3_minutes'] = float(
            record_group.analyzed_minutes_N2.sum() + record_group.analyzed_minutes_N3.sum())
        row['analyzed_N2_N3_fraction'] = _safe_rate(
            row['analyzed_N2_N3_minutes'], row['total_N2_N3_minutes'])
        row['SW_per_min_N2_N3'] = _safe_rate(nrem_count, channel_nrem_minutes)
        row['total_weighted_NREM_minutes'] = float(record_group.get(
            'total_weighted_NREM_minutes', pd.Series(dtype=float)).sum())
        row['analyzed_weighted_NREM_minutes'] = float(record_group.get(
            'analyzed_weighted_NREM_minutes', pd.Series(dtype=float)).sum())
        row['analyzed_weighted_NREM_fraction'] = _safe_rate(
            row['analyzed_weighted_NREM_minutes'], row['total_weighted_NREM_minutes'])
        for stage in ('N2', 'N3', 'NREM', 'REM', 'Wake'):
            weighted_count = float(group.get(
                f'weighted_SW_count_{stage}', pd.Series(dtype=float)).sum())
            weighted_minutes = float(group.get(
                f'weighted_minutes_{stage}', pd.Series(dtype=float)).sum())
            row[f'weighted_SW_count_{stage}'] = weighted_count
            row[f'weighted_SW_per_min_{stage}'] = _safe_rate(
                weighted_count, weighted_minutes)
        for metric in (
            'median_max_stage_probability', 'P10_max_stage_probability',
            'fraction_epochs_max_probability_below_0_5',
            'fraction_epochs_max_probability_below_0_7', 'mean_stage_entropy',
        ):
            values = record_group.get(metric, pd.Series(dtype=float))
            finite_values = values[np.isfinite(values)]
            row[metric] = float(np.median(finite_values)) if len(finite_values) else np.nan
        event_group = events_df
        segment_group = segments_df
        for column, value in row.items():
            if column in group_columns:
                event_group = event_group[event_group[column] == value]
                segment_group = segment_group[segment_group[column] == value]
        row['median_P2P_amplitude'] = _distribution(event_group.get('peak_to_peak_amplitude', []))['median']
        row['median_duration'] = _distribution(event_group.get('negative_half_wave_duration_seconds', []))['median']
        row['median_amp_threshold'] = _distribution(event_group.get('detector_amplitude_threshold', []))['median']
        finite = segment_group[['TotalSW', 'SWdensity']].dropna() if not segment_group.empty else pd.DataFrame()
        row['TotalSW_SWdensity_correlation'] = (
            float(finite.corr().iloc[0, 1]) if len(finite) > 1 and finite.TotalSW.nunique() > 1 else np.nan)
        if len(finite) > 1 and finite.TotalSW.nunique() > 1:
            slope, intercept = np.polyfit(finite.TotalSW, finite.SWdensity, 1)
            row['SWdensity_linear_slope'] = float(slope)
            row['SWdensity_linear_intercept'] = float(intercept)
        else:
            row['SWdensity_linear_slope'] = 1.0 / (SEGMENT_DURATION_SECONDS / 60.0)
            row['SWdensity_linear_intercept'] = 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def _eeg_units_summary(unit_rows):
    rows = []
    frame = pd.DataFrame(unit_rows)
    if frame.empty:
        return frame
    for keys, group in frame.groupby(['site_id', 'channel', 'physical_dimension'], dropna=False):
        row = dict(zip(('site_id', 'channel', 'physical_dimension'), keys))
        row['records'] = len(group)
        row['sampling_frequency_median'] = float(group.sampling_frequency.median())
        for metric in ('signal_median', 'signal_P1', 'signal_P99', 'signal_P99_minus_P1'):
            stats = _distribution(group[metric])
            for stat_name, value in stats.items():
                row[f'{metric}_{stat_name}'] = value
        rows.append(row)
    return pd.DataFrame(rows)


def _extreme_mask(series, factor=3.0):
    finite = series[np.isfinite(series)]
    if finite.empty:
        return pd.Series(False, index=series.index)
    q1, q3 = finite.quantile([0.25, 0.75])
    iqr = q3 - q1
    if iqr == 0:
        low, high = finite.quantile([0.01, 0.99])
    else:
        low, high = q1 - factor * iqr, q3 + factor * iqr
    return (series < low) | (series > high)


def _suspicious_cases(subjects_df, events_df, segments_df):
    rows = []
    def add_candidates(frame, mask, case_type, value_column):
        for _, candidate in frame[mask].iterrows():
            rows.append({key: candidate.get(key, np.nan) for key in
                         ('patient_id', 'bids_folder', 'site_id', 'session_id', 'channel',
                          'segment_start_seconds', 'segment_end_seconds')} |
                        {'case_type': case_type, 'value': candidate.get(value_column, np.nan)})
    if not subjects_df.empty:
        rates = subjects_df.SW_per_min_N2_N3
        finite = rates[np.isfinite(rates)]
        if not finite.empty:
            add_candidates(subjects_df, rates >= finite.quantile(.95), 'highest_SW_per_min_N2_N3', 'SW_per_min_N2_N3')
            positive = finite[finite > 0]
            if not positive.empty:
                add_candidates(subjects_df, (rates > 0) & (rates <= positive.quantile(.05)),
                               'lowest_nonzero_SW_per_min_N2_N3', 'SW_per_min_N2_N3')
        n3_cut = subjects_df.analyzed_N3_minutes.quantile(.75)
        add_candidates(subjects_df, (subjects_df.number_in_N3 == 0) &
                       (subjects_df.analyzed_N3_minutes >= n3_cut) & (subjects_df.analyzed_N3_minutes > 0),
                       'no_SW_despite_substantial_N3', 'analyzed_N3_minutes')
        for stage in ('Wake', 'REM'):
            values = subjects_df[f'SW_per_min_{stage}']
            finite = values[np.isfinite(values)]
            if not finite.empty:
                add_candidates(subjects_df, (values > 0) & (values >= finite.quantile(.95)),
                               f'many_detections_during_{stage}', f'SW_per_min_{stage}')
        pivot = subjects_df.pivot_table(index=['patient_id', 'site_id', 'session_id'],
                                        columns='channel', values='SW_per_min_N2_N3')
        disagreement = pivot.max(axis=1) - pivot.min(axis=1)
        if disagreement.notna().any():
            cutoff = disagreement.quantile(.95)
            for index, value in disagreement[disagreement >= cutoff].items():
                rows.append({'patient_id': index[0], 'site_id': index[1], 'session_id': index[2],
                             'case_type': 'largest_disagreement_between_EEG_channels', 'value': value})
    for column, name in (
        ('peak_to_peak_amplitude', 'extreme_amplitude'),
        ('negative_half_wave_duration_seconds', 'extreme_duration'),
    ):
        if column in events_df:
            add_candidates(events_df, _extreme_mask(events_df[column]), name, column)
    for column, name in (
        ('detector_amplitude_threshold', 'extreme_detector_amplitude_threshold'),
        ('detector_slope_threshold', 'extreme_detector_slope_threshold'),
    ):
        if column in segments_df:
            add_candidates(segments_df, _extreme_mask(segments_df[column]), name, column)
    return pd.DataFrame(rows)


def _print_site_summary(site_summary, channel_summary, units_summary):
    print('\n' + '=' * 60)
    print('SLOW-WAVE AUDIT BY SITE')
    print('=' * 60)
    if site_summary.empty:
        print('No auditable records were found.')
        return
    indexed = site_summary.set_index('site_id')
    overview = ['records', 'annotation_coverage', 'analyzed_N2_N3_fraction',
                'analyzed_weighted_NREM_fraction']
    print(indexed[overview].T.to_string(float_format=lambda x: f'{x:.4g}'))
    print('\nHARD')
    hard = ['SW_per_min_N2', 'SW_per_min_N3', 'SW_per_min_N2_N3',
            'SW_per_min_Wake', 'SW_per_min_REM']
    print(indexed[hard].T.to_string(float_format=lambda x: f'{x:.4g}'))
    print('\nSOFT')
    soft = ['weighted_SW_per_min_N2', 'weighted_SW_per_min_N3',
            'weighted_SW_per_min_NREM', 'weighted_SW_per_min_Wake',
            'weighted_SW_per_min_REM']
    print(indexed[soft].T.to_string(float_format=lambda x: f'{x:.4g}'))
    print('\n' + '=' * 60)
    print('SLOW-WAVE AUDIT BY SITE AND CHANNEL')
    print('=' * 60)
    print(channel_summary.to_string(index=False, float_format=lambda x: f'{x:.4g}'))
    print('\n' + '=' * 60)
    print('EEG PHYSICAL UNITS BY SITE AND CHANNEL')
    print('=' * 60)
    physical_columns = ['site_id', 'channel', 'physical_dimension', 'records',
                        'sampling_frequency_median', 'signal_P99_minus_P1_median']
    if units_summary.empty:
        print('No physical EEG signals were summarized.')
    else:
        print(units_summary[physical_columns].to_string(
            index=False, float_format=lambda x: f'{x:.4g}'))


def run_audit(data_folder, output_folder, channel_table=DEFAULT_CSV_PATH, max_records=None):
    data_folder = Path(data_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    demographics = pd.read_csv(data_folder / 'demographics.csv')
    if max_records is not None:
        demographics = demographics.head(int(max_records))
    eeg_aliases = _get_eeg_aliases(channel_table)
    event_rows, segment_rows, subject_rows, unit_rows, caisr_rows = [], [], [], [], []
    triggered = TriggeredWaveforms()

    for _, demographic in tqdm(demographics.iterrows(), total=len(demographics), desc='Auditing records'):
        site = str(demographic[HEADERS['site_id']])
        patient_id = str(demographic[HEADERS['patient_id']])
        bids_folder = str(demographic[HEADERS['bids_folder']])
        session = str(demographic[HEADERS['session_id']])
        record = {'patient_id': patient_id, 'bids_folder': bids_folder,
                  'site_id': site, 'session_id': session}
        physiological_path = _find_physiological_path(data_folder, site, bids_folder, session)
        annotation = load_caisr_annotation(_find_annotation_path(data_folder, site, bids_folder, session))
        uncertainty = caisr_uncertainty_metrics(annotation)
        subject_record = {**record, **uncertainty}
        caisr_rows.append(subject_record)
        if not physiological_path.is_file():
            continue
        try:
            edf = edfio.read_edf(physiological_path, lazy_load_data=True)
            duration = _record_duration(edf)
            intervals = build_segment_intervals(duration)
            for channel in EEG_CHANNEL_SPECS:
                selected = _selected_signal(edf, channel, eeg_aliases)
                if selected is None:
                    continue
                raw_signal, raw_fs, unit, source_labels = selected
                physical_statistics = physical_signal_statistics(raw_signal)
                if np.isfinite(physical_statistics['signal_median']):
                    unit_rows.append({**record, 'channel': channel, 'source_labels': source_labels,
                                      'physical_dimension': unit, 'sampling_frequency': raw_fs,
                                      **physical_statistics})
                channel_events = []
                channel_intervals = []
                for start, end in intervals:
                    start_index, end_index = int(round(start * raw_fs)), int(round(end * raw_fs))
                    if end_index > raw_signal.size:
                        continue
                    try:
                        audited_detection = detect_audited_slow_waves(
                            raw_signal[start_index:end_index], raw_fs)
                    except Exception:
                        continue
                    if audited_detection is None:
                        continue
                    detection, detector_signal, fs = audited_detection
                    info = detection['info']
                    amp_threshold = _scalar(info['Parameters'].get('Ref_AmplitudeAbsolute'))
                    channel_intervals.append((start, end))
                    data_deviation = _scalar(info['Recording'].get('Data_Deviation'))
                    slope_threshold = _scalar(info['Recording'].get('Slope_Threshold'))
                    exposure = stage_minutes_in_interval(annotation, start, end)
                    weighted_exposure = weighted_stage_minutes_in_interval(
                        annotation, start, end)
                    fractions = {name: exposure[name] / (SEGMENT_DURATION_SECONDS / 60.0) for name in STAGE_ORDER}
                    aggregates = eeg_features.summarize_slow_waves(
                        detection['events'], fs, SEGMENT_DURATION_SECONDS)
                    segment_event_rows = []
                    for event in detection['events']:
                        down = start + _scalar(event.get('Ref_DownInd')) / fs
                        trough = start + _scalar(event.get('Ref_PeakInd')) / fs
                        up = start + _scalar(event.get('Ref_UpInd')) / fs
                        stage = annotation_at_time(annotation, trough)
                        event_row = {**record, 'channel': channel,
                                     'segment_start_seconds': start, 'segment_end_seconds': end,
                                     'down_crossing_seconds': down, 'trough_seconds': trough,
                                     'up_crossing_seconds': up, **stage,
                                     'negative_peak_amplitude': _scalar(event.get('Ref_PeakAmp')),
                                     'peak_to_peak_amplitude': _scalar(event.get('Ref_P2PAmp')),
                                     'negative_slope': _scalar(event.get('Ref_NegSlope')),
                                     'positive_slope': _scalar(event.get('Ref_PosSlope')),
                                     'negative_half_wave_duration_seconds': up - down,
                                     'detector_amplitude_threshold': amp_threshold,
                                     'detector_data_deviation': data_deviation,
                                     'detector_slope_threshold': slope_threshold,
                                     'source_labels': source_labels, 'sampling_frequency': fs}
                        event_rows.append(event_row)
                        channel_events.append(event_row)
                        segment_event_rows.append(event_row)
                        waveform = _event_window(
                            detector_signal, int(round((trough - start) * fs)), fs)
                        if stage['stage_at_trough'] in ('N2', 'N3'):
                            if waveform is not None:
                                triggered.add((site, channel, stage['stage_at_trough']), waveform)
                        if waveform is not None:
                            triggered.add_weighted_nrem(
                                (site, channel), waveform, event_row['weight_NREM'])
                    segment_rows.append({
                        **record, 'channel': channel,
                        'segment_start_seconds': start, 'segment_end_seconds': end,
                        **{f'fraction_{name}': fractions[name] for name in STAGE_ORDER},
                        **{f'weighted_minutes_{name}': weighted_exposure[name]
                           for name in SOFT_STAGE_ORDER},
                        **weighted_event_metrics(segment_event_rows, weighted_exposure),
                        'TotalSW': aggregates['TotalSW'], 'SWdensity': aggregates['SWdensity'],
                        'detector_amplitude_threshold': amp_threshold,
                        'detector_data_deviation': data_deviation,
                        'detector_slope_threshold': slope_threshold,
                    })
                subject_rows.append(_subject_rows_for_channel(
                    subject_record, channel, annotation, duration,
                    channel_intervals, channel_events))
        except Exception as error:
            tqdm.write(f'Skipping {bids_folder} session {session}: {error}')

    events_df = pd.DataFrame(event_rows, columns=EVENT_COLUMNS)
    segments_df = pd.DataFrame(segment_rows, columns=SEGMENT_COLUMNS)
    subjects_df = pd.DataFrame(subject_rows)
    site_summary = _aggregate_summary(subjects_df, events_df, segments_df, ['site_id'])
    channel_summary = _aggregate_summary(subjects_df, events_df, segments_df, ['site_id', 'channel'])
    stage_summary = _build_stage_summary(events_df, subjects_df)
    units_summary = _eeg_units_summary(unit_rows)
    caisr_summary = pd.DataFrame(caisr_rows)
    suspicious = _suspicious_cases(subjects_df, events_df, segments_df)
    outputs = {
        'events.csv': events_df, 'segments.csv': segments_df, 'subjects.csv': subjects_df,
        'site_summary.csv': site_summary, 'channel_summary.csv': channel_summary,
        'stage_summary.csv': stage_summary, 'eeg_units_summary.csv': units_summary,
        'suspicious_cases.csv': suspicious, 'caisr_summary.csv': caisr_summary,
    }
    for filename, frame in outputs.items():
        frame.to_csv(output_folder / filename, index=False)
    triggered.save(output_folder / 'event_triggered_average.npz')
    _print_site_summary(site_summary, channel_summary, units_summary)
    print(f'\nDiagnostic files written to {output_folder.resolve()}')
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-folder', default='training_data')
    parser.add_argument('--output-folder', default='.feature_cache/diagnostics/slow_waves')
    parser.add_argument('--channel-table', default=DEFAULT_CSV_PATH)
    parser.add_argument('--max-records', type=int, default=None,
                        help='Development smoke limit; omit for the complete training set.')
    args = parser.parse_args()
    run_audit(args.data_folder, args.output_folder, args.channel_table, args.max_records)


if __name__ == '__main__':
    main()
