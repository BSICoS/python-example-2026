"""Minimal CAISR primitives used by production slow-wave extraction."""

from pathlib import Path

import edfio
import numpy as np


def unavailable_annotation():
    return {'available': False, 'fs': 1.0 / 30.0, 'stage': np.array([], dtype=float),
            'p_n2': np.array([], dtype=float), 'p_n3': np.array([], dtype=float),
            'arousal': np.array([], dtype=float), 'respiratory': np.array([], dtype=float),
            'limb_movement': np.array([], dtype=float)}


def find_annotation_path(data_folder, site_id, bids_folder, session_id):
    return Path(data_folder, 'algorithmic_annotations', str(site_id),
                f'{bids_folder}_ses-{session_id}_caisr_annotations.edf')


def load_annotation(data_folder, site_id, bids_folder, session_id):
    """Load hard CAISR stages and N2/N3 probabilities without raising normally."""
    path = find_annotation_path(data_folder, site_id, bids_folder, session_id)
    if not path.is_file():
        return unavailable_annotation()
    try:
        edf = edfio.read_edf(path, lazy_load_data=True)
        signals = {signal.label.lower().strip(): signal for signal in edf.signals}
        stage = signals.get('stage_caisr')
        n2, n3 = signals.get('caisr_prob_n2'), signals.get('caisr_prob_n3')
        if stage is None:
            return unavailable_annotation()
        return {'available': True, 'fs': float(stage.sampling_frequency),
                'stage': np.asarray(stage.data, dtype=float),
            'p_n2': np.asarray(n2.data, dtype=float) if n2 is not None else np.array([], dtype=float),
            'p_n3': np.asarray(n3.data, dtype=float) if n3 is not None else np.array([], dtype=float),
            'arousal': np.asarray(signals.get('arousal_caisr').data, dtype=float)
            if signals.get('arousal_caisr') is not None else np.array([], dtype=float),
            'respiratory': np.asarray(signals.get('resp_caisr').data, dtype=float)
            if signals.get('resp_caisr') is not None else np.array([], dtype=float),
            'limb_movement': np.asarray(signals.get('limb_caisr').data, dtype=float)
            if signals.get('limb_caisr') is not None else np.array([], dtype=float)}
    except Exception:
        return unavailable_annotation()


def _count_events(values):
    active = np.isfinite(values) & (np.asarray(values, dtype=float) > 0)
    return int(np.count_nonzero(active & np.concatenate(([True], ~active[:-1])))) if active.size else 0


def get_sleep_architecture_features(annotation):
    """Return CAISR sleep architecture and event-rate features in production order."""
    if not annotation.get('available'):
        return np.full(12, np.nan, dtype=np.float32)

    stages = np.asarray(annotation.get('stage', []), dtype=float)
    valid = np.isin(stages, (1, 2, 3, 4, 5))
    if not np.any(valid):
        return np.full(12, np.nan, dtype=np.float32)

    epoch_minutes = 1.0 / (float(annotation['fs']) * 60.0)
    valid_stages = stages[valid].astype(int)
    fractions = [float(np.mean(valid_stages == stage)) for stage in (5, 3, 2, 1, 4)]
    sleep_mask = np.isin(stages, (1, 2, 3, 4))
    sleep_minutes = float(np.count_nonzero(sleep_mask) * epoch_minutes)
    sleep_efficiency = float(np.count_nonzero(sleep_mask) / np.count_nonzero(valid))

    transitions = sum(
        stages[index] != stages[index + 1]
        for index in range(len(stages) - 1)
        if valid[index] and valid[index + 1]
    )
    valid_hours = np.count_nonzero(valid) * epoch_minutes / 60.0
    transitions_per_hour = float(transitions / valid_hours) if valid_hours else np.nan

    sleep_indices = np.flatnonzero(sleep_mask)
    sleep_onset = int(sleep_indices[0]) if sleep_indices.size else None
    if sleep_onset is None:
        waso_minutes = rem_latency_minutes = np.nan
    else:
        waso_minutes = float(np.count_nonzero(stages[sleep_onset:] == 5) * epoch_minutes)
        rem_indices = np.flatnonzero((stages == 4) & (np.arange(len(stages)) >= sleep_onset))
        rem_latency_minutes = (
            float((rem_indices[0] - sleep_onset) * epoch_minutes) if rem_indices.size else np.nan
        )

    sleep_hours = sleep_minutes / 60.0
    event_rates = [
        float(_count_events(np.asarray(annotation.get(name, []))) / sleep_hours)
        if sleep_hours else np.nan
        for name in ('respiratory', 'arousal', 'limb_movement')
    ]
    return np.asarray(
        [*fractions, sleep_efficiency, transitions_per_hour, waso_minutes, rem_latency_minutes, *event_rates],
        dtype=np.float32,
    )


def expand_stages_to_samples(annotation, start_seconds, n_samples, fs):
    """Assign each detector sample to its CAISR epoch by floor, no interpolation."""
    stages = np.full(int(n_samples), 9, dtype=int)
    if not annotation.get('available') or n_samples <= 0:
        return stages
    indices = np.floor((float(start_seconds) + np.arange(int(n_samples)) / float(fs))
                       * float(annotation['fs'])).astype(int)
    valid = (indices >= 0) & (indices < len(annotation['stage']))
    values = np.asarray(annotation['stage'], dtype=float)
    valid[valid] &= np.isfinite(values[indices[valid]])
    stages[valid] = values[indices[valid]].astype(int)
    return stages


def p_nrem_at_time(annotation, time_seconds):
    """Return p(N2)+p(N3), or NaN when either CAISR probability is unavailable."""
    if not annotation.get('available') or time_seconds < 0:
        return np.nan
    index = int(np.floor(float(time_seconds) * float(annotation['fs'])))
    if index < 0 or index >= len(annotation['p_n2']) or index >= len(annotation['p_n3']):
        return np.nan
    n2, n3 = float(annotation['p_n2'][index]), float(annotation['p_n3'][index])
    return n2 + n3 if np.isfinite(n2) and np.isfinite(n3) else np.nan


def weighted_nrem_minutes(annotation, start_seconds, end_seconds):
    """Integrate p(N2)+p(N3) across an arbitrary interval including overlaps."""
    if not annotation.get('available') or end_seconds <= start_seconds:
        return 0.0
    epoch_seconds = 1.0 / float(annotation['fs'])
    first = max(0, int(np.floor(start_seconds / epoch_seconds)))
    last = min(len(annotation['stage']), int(np.ceil(end_seconds / epoch_seconds)))
    minutes = 0.0
    for index in range(first, last):
        if index >= len(annotation['p_n2']) or index >= len(annotation['p_n3']):
            continue
        n2, n3 = annotation['p_n2'][index], annotation['p_n3'][index]
        if not (np.isfinite(n2) and np.isfinite(n3)):
            continue
        overlap = max(0.0, min(end_seconds, (index + 1) * epoch_seconds)
                      - max(start_seconds, index * epoch_seconds))
        minutes += float(n2 + n3) * overlap / 60.0
    return minutes
