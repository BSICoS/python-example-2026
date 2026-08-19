"""Minimal CAISR primitives used by production slow-wave extraction."""

from pathlib import Path

import edfio
import numpy as np


def unavailable_annotation():
    return {'available': False, 'fs': 1.0 / 30.0, 'stage': np.array([], dtype=float),
            'p_n2': np.array([], dtype=float), 'p_n3': np.array([], dtype=float)}


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
        if stage is None or n2 is None or n3 is None:
            return unavailable_annotation()
        return {'available': True, 'fs': float(stage.sampling_frequency),
                'stage': np.asarray(stage.data, dtype=float),
                'p_n2': np.asarray(n2.data, dtype=float),
                'p_n3': np.asarray(n3.data, dtype=float)}
    except Exception:
        return unavailable_annotation()


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
