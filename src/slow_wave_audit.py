"""Development-only methodological audit of the production slow-wave detector."""

from __future__ import annotations

import argparse
from pathlib import Path

import edfio
import numpy as np
import pandas as pd
from tqdm import tqdm

from helper_code import HEADERS
from src.common.signal_utils import resample_signal
from src.eeg_processing import (
    EEG_CHANNEL_SPECS,
    _get_eeg_aliases,
    get_eeg_channel_source_labels,
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
EVENT_COLUMNS = (
    'patient_id', 'bids_folder', 'site_id', 'session_id', 'channel',
    'segment_start_seconds', 'segment_end_seconds', 'down_crossing_seconds',
    'trough_seconds', 'up_crossing_seconds', 'stage_at_trough',
    *PROBABILITY_LABELS, 'negative_peak_amplitude', 'peak_to_peak_amplitude',
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
)


def translate_stage_code(value):
    """Translate CAISR's numeric stage code without inventing missing stages."""
    try:
        code = int(float(value))
    except (TypeError, ValueError, OverflowError):
        return 'unavailable'
    return STAGE_NAMES.get(code, 'unavailable')


def build_segment_intervals(recording_duration_seconds):
    """Return the exact 5-minute/15-minute intervals used by production."""
    last_start = float(recording_duration_seconds) - SEGMENT_DURATION_SECONDS
    if last_start < 0:
        return []
    starts = np.arange(0.0, last_start + 1e-9, SEGMENT_STRIDE_SECONDS)
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
    if not annotation['available'] or time_seconds < 0:
        return result
    index = int(np.floor(float(time_seconds) * annotation['fs']))
    if index < 0 or index >= annotation['stage'].size:
        return result
    stage_name = translate_stage_code(annotation['stage'][index])
    result['stage_at_trough'] = stage_name
    if stage_name == 'unavailable':
        return result
    for label, values in annotation['probabilities'].items():
        if index < values.size and np.isfinite(values[index]) and values[index] != 9:
            result[label] = float(values[index])
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


def _safe_rate(count, minutes):
    return float(count / minutes) if minutes > 0 else np.nan


def _scalar(value):
    values = np.asarray(value, dtype=float).reshape(-1)
    finite = values[np.isfinite(values)]
    return float(finite[0]) if finite.size else np.nan


def _event_window(filtered_signal, trough_index, fs, half_seconds=2.0):
    half_samples = int(round(half_seconds * fs))
    start = int(trough_index) - half_samples
    end = int(trough_index) + half_samples + 1
    if start < 0 or end > len(filtered_signal):
        return None
    return np.asarray(filtered_signal[start:end], dtype=float)


class TriggeredWaveforms:
    """Exact running mean plus bounded deterministic samples for quantiles."""

    def __init__(self, max_quantile_samples=20000):
        self.max_quantile_samples = int(max_quantile_samples)
        self.groups = {}
        self.rng = np.random.default_rng(20260819)

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

    def save(self, path, fs=200.0):
        payload = {'time_seconds': np.arange(-int(2 * fs), int(2 * fs) + 1) / fs}
        for (site, channel, stage), group in sorted(self.groups.items()):
            prefix = f'{site}__{channel.replace("-", "_")}__{stage}'
            sample = np.asarray(group['sample'], dtype=float)
            payload[f'{prefix}__mean'] = group['sum'] / group['n']
            payload[f'{prefix}__median'] = np.median(sample, axis=0)
            payload[f'{prefix}__p25'] = np.percentile(sample, 25, axis=0)
            payload[f'{prefix}__p75'] = np.percentile(sample, 75, axis=0)
            payload[f'{prefix}__n_events'] = np.asarray(group['n'])
            payload[f'{prefix}__quantile_sample_size'] = np.asarray(len(sample))
        np.savez_compressed(path, **payload)


def _distribution(values):
    values = np.asarray(list(values), dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {'N': 0, 'median': np.nan, 'IQR': np.nan, 'P10': np.nan, 'P90': np.nan}
    p10, p25, median, p75, p90 = np.percentile(values, [10, 25, 50, 75, 90])
    return {'N': int(values.size), 'median': float(median), 'IQR': float(p75 - p25),
            'P10': float(p10), 'P90': float(p90)}


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
    analyzed = {name: 0.0 for name in STAGE_ORDER}
    for start, end in intervals:
        exposure = stage_minutes_in_interval(annotation, start, end)
        for name in STAGE_ORDER:
            analyzed[name] += exposure[name]
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
    }
    row['analyzed_N2_N3_fraction'] = (
        row['analyzed_N2_N3_minutes'] / row['total_N2_N3_minutes']
        if row['total_N2_N3_minutes'] > 0 else np.nan)
    for name in STAGE_ORDER:
        row[f'number_in_{name}'] = counts[name]
        row[f'analyzed_minutes_{name}'] = analyzed[name]
        row[f'SW_per_min_{name}'] = _safe_rate(counts[name], analyzed[name])
    row['SW_per_min_N3'] = _safe_rate(counts['N3'], analyzed['N3'])
    row['SW_per_min_N2'] = _safe_rate(counts['N2'], analyzed['N2'])
    row['SW_per_min_N2_N3'] = _safe_rate(counts['N2'] + counts['N3'], analyzed['N2'] + analyzed['N3'])
    row['SW_per_min_REM'] = _safe_rate(counts['REM'], analyzed['REM'])
    row['SW_per_min_Wake'] = _safe_rate(counts['Wake'], analyzed['Wake'])
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
                                 'metric': metric, **_distribution(values)})
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
        row['analyzed_N2_N3_minutes'] = float(
            record_group.analyzed_minutes_N2.sum() + record_group.analyzed_minutes_N3.sum())
        row['SW_per_min_N2_N3'] = _safe_rate(nrem_count, channel_nrem_minutes)
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


def _print_site_summary(site_summary, channel_summary):
    print('\n' + '=' * 60)
    print('SLOW-WAVE AUDIT BY SITE')
    print('=' * 60)
    if site_summary.empty:
        print('No auditable records were found.')
        return
    display_columns = ['records', 'annotation_coverage', 'analyzed_N2_N3_minutes',
                       'SW_per_min_N2', 'SW_per_min_N3', 'SW_per_min_N2_N3',
                       'SW_per_min_Wake', 'SW_per_min_REM', 'median_P2P_amplitude',
                       'median_duration', 'median_amp_threshold']
    print(site_summary.set_index('site_id')[display_columns].T.to_string(float_format=lambda x: f'{x:.4g}'))
    print('\n' + '=' * 60)
    print('SLOW-WAVE AUDIT BY SITE AND CHANNEL')
    print('=' * 60)
    print(channel_summary.to_string(index=False, float_format=lambda x: f'{x:.4g}'))


def run_audit(data_folder, output_folder, channel_table=DEFAULT_CSV_PATH, max_records=None):
    data_folder = Path(data_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    demographics = pd.read_csv(data_folder / 'demographics.csv')
    if max_records is not None:
        demographics = demographics.head(int(max_records))
    eeg_aliases = _get_eeg_aliases(channel_table)
    event_rows, segment_rows, subject_rows, unit_rows = [], [], [], []
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
                finite_raw = raw_signal[np.isfinite(raw_signal)]
                if finite_raw.size:
                    p1, median, p99 = np.percentile(finite_raw, [1, 50, 99])
                    unit_rows.append({**record, 'channel': channel, 'source_labels': source_labels,
                                      'physical_dimension': unit, 'sampling_frequency': raw_fs,
                                      'signal_median': median, 'signal_P1': p1, 'signal_P99': p99,
                                      'signal_P99_minus_P1': p99 - p1})
                channel_events = []
                channel_intervals = []
                for start, end in intervals:
                    start_index, end_index = int(round(start * raw_fs)), int(round(end * raw_fs))
                    if end_index > raw_signal.size:
                        continue
                    signal = np.nan_to_num(raw_signal[start_index:end_index], nan=0.0, posinf=0.0, neginf=0.0)
                    fs = raw_fs
                    if fs != 200:
                        signal, fs = resample_signal(signal, fs, 200)
                    try:
                        detection = eeg_features.detect_slow_waves(signal, fs)
                    except Exception:
                        continue
                    info = detection['info']
                    amp_threshold = _scalar(info['Parameters'].get('Ref_AmplitudeAbsolute'))
                    channel_intervals.append((start, end))
                    data_deviation = _scalar(info['Recording'].get('Data_Deviation'))
                    slope_threshold = _scalar(info['Recording'].get('Slope_Threshold'))
                    exposure = stage_minutes_in_interval(annotation, start, end)
                    fractions = {name: exposure[name] / (SEGMENT_DURATION_SECONDS / 60.0) for name in STAGE_ORDER}
                    aggregates = eeg_features.summarize_slow_waves(
                        detection['events'], fs, SEGMENT_DURATION_SECONDS)
                    segment_rows.append({**record, 'channel': channel,
                                         'segment_start_seconds': start, 'segment_end_seconds': end,
                                         **{f'fraction_{name}': fractions[name] for name in STAGE_ORDER},
                                         'TotalSW': aggregates['TotalSW'], 'SWdensity': aggregates['SWdensity'],
                                         'detector_amplitude_threshold': amp_threshold,
                                         'detector_data_deviation': data_deviation,
                                         'detector_slope_threshold': slope_threshold})
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
                        if stage['stage_at_trough'] in ('N2', 'N3'):
                            waveform = _event_window(detection['filtered_signal'],
                                                     int(round((trough - start) * fs)), fs)
                            if waveform is not None:
                                triggered.add((site, channel, stage['stage_at_trough']), waveform)
                subject_rows.append(_subject_rows_for_channel(
                    record, channel, annotation, duration, channel_intervals, channel_events))
        except Exception as error:
            tqdm.write(f'Skipping {bids_folder} session {session}: {error}')

    events_df = pd.DataFrame(event_rows, columns=EVENT_COLUMNS)
    segments_df = pd.DataFrame(segment_rows, columns=SEGMENT_COLUMNS)
    subjects_df = pd.DataFrame(subject_rows)
    site_summary = _aggregate_summary(subjects_df, events_df, segments_df, ['site_id'])
    channel_summary = _aggregate_summary(subjects_df, events_df, segments_df, ['site_id', 'channel'])
    stage_summary = _build_stage_summary(events_df, subjects_df)
    units_summary = _eeg_units_summary(unit_rows)
    suspicious = _suspicious_cases(subjects_df, events_df, segments_df)
    outputs = {
        'events.csv': events_df, 'segments.csv': segments_df, 'subjects.csv': subjects_df,
        'site_summary.csv': site_summary, 'channel_summary.csv': channel_summary,
        'stage_summary.csv': stage_summary, 'eeg_units_summary.csv': units_summary,
        'suspicious_cases.csv': suspicious,
    }
    for filename, frame in outputs.items():
        frame.to_csv(output_folder / filename, index=False)
    triggered.save(output_folder / 'event_triggered_average.npz')
    _print_site_summary(site_summary, channel_summary)
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
