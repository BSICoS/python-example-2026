"""Development-only comparison of current and NREM-aware slow-wave methods."""

from __future__ import annotations

import argparse
from pathlib import Path

import edfio
import numpy as np
import pandas as pd
from tqdm import tqdm

from helper_code import HEADERS
from src.eeg_processing import EEG_CHANNEL_SPECS, _get_eeg_aliases, prepare_slow_wave_detector_input
from src.lib import eeg_features
from src.pipeline.config import DEFAULT_CSV_PATH, SEGMENT_DURATION_SECONDS, SEGMENT_STRIDE_SECONDS
from src.slow_wave_audit import (
    SOFT_STAGE_ORDER, STAGE_ORDER, TriggeredWaveforms, _aggregate_summary, _event_window,
    _find_annotation_path, _find_physiological_path, _record_duration, _scalar, _selected_signal,
    _safe_rate, _subject_rows_for_channel, annotation_at_time, build_segment_intervals,
    caisr_uncertainty_metrics, load_caisr_annotation, stage_minutes_in_interval,
    weighted_event_metrics, weighted_stage_minutes_in_interval,
)

METHODS = {
    'current': {'stride_seconds': SEGMENT_STRIDE_SECONDS, 'nrem_aware': False},
    'nrem_sampled': {'stride_seconds': SEGMENT_STRIDE_SECONDS, 'nrem_aware': True},
    'nrem_full': {'stride_seconds': SEGMENT_DURATION_SECONDS, 'nrem_aware': True},
}


def caisr_stages_to_detector_samples(annotation, start_seconds, n_samples, fs):
    """Expand hard 30-second CAISR epochs by floor assignment, never interpolating."""
    output = np.full(int(n_samples), 9, dtype=int)
    if not annotation['available'] or n_samples <= 0:
        return output
    sample_times = float(start_seconds) + np.arange(int(n_samples)) / float(fs)
    indices = np.floor(sample_times * float(annotation['fs'])).astype(int)
    valid = (indices >= 0) & (indices < len(annotation['stage']))
    values = np.asarray(annotation['stage'], dtype=float)
    finite = valid.copy()
    finite[valid] &= np.isfinite(values[indices[valid]])
    output[finite] = values[indices[finite]].astype(int)
    return output


def _event_row(record, method, channel, start, end, event, fs, annotation, thresholds):
    down = start + _scalar(event.get('Ref_DownInd')) / fs
    trough = start + _scalar(event.get('Ref_PeakInd')) / fs
    up = start + _scalar(event.get('Ref_UpInd')) / fs
    stage = annotation_at_time(annotation, trough)
    return {
        **record, 'variant': method, 'channel': channel,
        'segment_start_seconds': start, 'segment_end_seconds': end,
        'down_crossing_seconds': down, 'trough_seconds': trough, 'up_crossing_seconds': up,
        **stage,
        'negative_peak_amplitude': _scalar(event.get('Ref_PeakAmp')),
        'peak_to_peak_amplitude': _scalar(event.get('Ref_P2PAmp')),
        'negative_slope': _scalar(event.get('Ref_NegSlope')),
        'positive_slope': _scalar(event.get('Ref_PosSlope')),
        'negative_half_wave_duration_seconds': up - down,
        **thresholds,
    }


def _method_summary(subjects, events, segments, group_columns):
    summary = _aggregate_summary(subjects, events, segments, group_columns)
    if summary.empty:
        return summary
    for keys, group in segments.groupby(group_columns, dropna=False):
        if not isinstance(keys, tuple): keys = (keys,)
        mask = np.ones(len(summary), dtype=bool)
        for key, value in zip(group_columns, keys): mask &= summary[key].eq(value)
        for metric in ('detector_amplitude_threshold', 'detector_slope_threshold', 'detector_data_deviation'):
            values = group[metric].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            summary.loc[mask, f'{metric}_median'] = np.median(values) if values.size else np.nan
            summary.loc[mask, f'{metric}_IQR'] = np.subtract(*np.percentile(values, [75, 25])) if values.size else np.nan
            summary.loc[mask, f'{metric}_CV'] = np.std(values) / np.mean(values) if values.size and np.mean(values) else np.nan
    return summary


def _print_comparison(comparison):
    print('\n' + '=' * 60 + '\nSLOW-WAVE METHOD COMPARISON BY SITE\n' + '=' * 60)
    if comparison.empty:
        print('No auditable records were found.')
        return
    fields = ['SW_per_min_N2', 'SW_per_min_N3', 'SW_per_min_N2_N3', 'SW_per_min_Wake',
              'weighted_SW_per_min_NREM', 'analyzed_N2_N3_fraction',
              'analyzed_weighted_NREM_fraction']
    print(comparison.set_index(['variant', 'site_id'])[fields].to_string(float_format=lambda x: f'{x:.4g}'))


def run_method_comparison(data_folder, output_folder, channel_table=DEFAULT_CSV_PATH, max_records=None):
    """Run three development-only detector variants without changing production features."""
    data_folder, output_folder = Path(data_folder), Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    demographics = pd.read_csv(data_folder / 'demographics.csv')
    if max_records is not None: demographics = demographics.head(int(max_records))
    aliases = _get_eeg_aliases(channel_table)
    event_rows, segment_rows, subject_rows = [], [], []
    triggered = TriggeredWaveforms(waveform_domain='internal_slow_wave_detector_filtered_signal')
    for _, demographic in tqdm(demographics.iterrows(), total=len(demographics), desc='Comparing methods'):
        record = {'patient_id': str(demographic[HEADERS['patient_id']]),
                  'bids_folder': str(demographic[HEADERS['bids_folder']]),
                  'site_id': str(demographic[HEADERS['site_id']]),
                  'session_id': str(demographic[HEADERS['session_id']])}
        annotation = load_caisr_annotation(_find_annotation_path(data_folder, record['site_id'], record['bids_folder'], record['session_id']))
        path = _find_physiological_path(data_folder, record['site_id'], record['bids_folder'], record['session_id'])
        if not path.is_file(): continue
        try:
            edf = edfio.read_edf(path, lazy_load_data=True); duration = _record_duration(edf)
            for method, config in METHODS.items():
                intervals = build_segment_intervals(duration, config['stride_seconds'])
                for channel in EEG_CHANNEL_SPECS:
                    selected = _selected_signal(edf, channel, aliases)
                    if selected is None: continue
                    raw, raw_fs, _, _ = selected
                    channel_events, used_intervals = [], []
                    for start, end in intervals:
                        lo, hi = int(round(start * raw_fs)), int(round(end * raw_fs))
                        if hi > raw.size: continue
                        prepared = prepare_slow_wave_detector_input(raw[lo:hi], raw_fs)
                        if prepared is None: continue
                        signal, fs = prepared
                        kwargs = {}
                        if config['nrem_aware']:
                            kwargs = {'sleep_stages': caisr_stages_to_detector_samples(annotation, start, len(signal), fs),
                                      'allowed_stages': (1, 2)}
                        detection = eeg_features.detect_slow_waves(signal, fs, **kwargs)
                        used_intervals.append((start, end))
                        info = detection['info']
                        thresholds = {'detector_amplitude_threshold': _scalar(info['Parameters'].get('Ref_AmplitudeAbsolute')),
                                      'detector_data_deviation': _scalar(info['Recording'].get('Data_Deviation')),
                                      'detector_slope_threshold': _scalar(info['Recording'].get('Slope_Threshold'))}
                        exposure = stage_minutes_in_interval(annotation, start, end)
                        weighted = weighted_stage_minutes_in_interval(annotation, start, end)
                        segment_events = []
                        for event in detection['events']:
                            row = _event_row(record, method, channel, start, end, event, fs, annotation, thresholds)
                            event_rows.append(row); channel_events.append(row); segment_events.append(row)
                            waveform = _event_window(detection['filtered_signal'], int(round((row['trough_seconds'] - start) * fs)), fs)
                            if waveform is not None and row['stage_at_trough'] in ('N2', 'N3'):
                                triggered.add((record['site_id'], f'{method}--{channel}', row['stage_at_trough']), waveform)
                            if waveform is not None: triggered.add_weighted_nrem((record['site_id'], f'{method}--{channel}'), waveform, row['weight_NREM'])
                        aggregate = eeg_features.summarize_slow_waves(detection['events'], fs, SEGMENT_DURATION_SECONDS)
                        segment_rows.append({**record, 'variant': method, 'channel': channel,
                            'segment_start_seconds': start, 'segment_end_seconds': end,
                            'N2_minutes': exposure['N2'], 'N3_minutes': exposure['N3'],
                            'NREM_minutes': exposure['N2'] + exposure['N3'],
                            'weighted_NREM_minutes': weighted['NREM'],
                            'fraction_NREM_in_segment': (exposure['N2'] + exposure['N3']) / (SEGMENT_DURATION_SECONDS / 60),
                            **{f'weighted_minutes_{x}': weighted[x] for x in SOFT_STAGE_ORDER},
                            **weighted_event_metrics(segment_events, weighted), **aggregate, **thresholds})
                    base = _subject_rows_for_channel({**record, 'variant': method, **caisr_uncertainty_metrics(annotation)}, channel, annotation, duration, used_intervals, channel_events)
                    subject_rows.append(base)
        except Exception as error:
            tqdm.write(f"Skipping {record['bids_folder']} session {record['session_id']}: {error}")
    events, segments, subjects = pd.DataFrame(event_rows), pd.DataFrame(segment_rows), pd.DataFrame(subject_rows)
    site = _method_summary(subjects, events, segments, ['variant', 'site_id'])
    channel = _method_summary(subjects, events, segments, ['variant', 'site_id', 'channel'])
    comparison = site.copy()
    for name, frame in {'method_events.csv': events, 'method_segments.csv': segments,
                        'method_site_summary.csv': site, 'method_channel_summary.csv': channel,
                        'method_comparison.csv': comparison}.items(): frame.to_csv(output_folder / name, index=False)
    triggered.save(output_folder / 'method_event_triggered_average.npz')
    _print_comparison(comparison)
    print(f'\nMethod diagnostics written to {output_folder.resolve()}')
    return {'events': events, 'segments': segments, 'subjects': subjects, 'site_summary': site,
            'channel_summary': channel, 'comparison': comparison}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-folder', default='training_data')
    parser.add_argument('--output-folder', default='.feature_cache/diagnostics/slow_wave_methods')
    parser.add_argument('--channel-table', default=DEFAULT_CSV_PATH)
    parser.add_argument('--max-records', type=int, default=None)
    args = parser.parse_args()
    run_method_comparison(args.data_folder, args.output_folder, args.channel_table, args.max_records)


if __name__ == '__main__': main()
