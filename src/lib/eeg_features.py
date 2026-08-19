"""EEG background feature helpers used by the active submission pipeline."""

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, welch

def _safe_sqrt_variance_ratio(numerator_signal, denominator_signal):
    numerator_var = np.var(numerator_signal)
    denominator_var = np.var(denominator_signal)
    if denominator_var <= 0 or not np.isfinite(denominator_var):
        return 0.0
    ratio = numerator_var / denominator_var
    if ratio <= 0 or not np.isfinite(ratio):
        return 0.0
    return float(np.sqrt(ratio))


def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='bandpass')
    y = filtfilt(b, a, data) 
    return y


def create_epochs(data, fs, epoch_duration=30):
    samples_per_epoch = int(fs * epoch_duration)
    num_epochs = len(data) // samples_per_epoch

    data_trimmed = data[:num_epochs * samples_per_epoch]
    epochs = data_trimmed.reshape(num_epochs, samples_per_epoch)
    return epochs


def extract_band_powers(epochs, fs, win_len=2):
    features = []
    complexities = []
    bands = {
        'Delta': (0.5, 4),
        'Theta': (4, 8),
        'Alpha': (8, 12),
        'Sigma': (11, 16),
        'Beta': (12, 30)
    }

    for epoch in epochs:
        freqs, psd = welch(epoch, fs, nperseg=fs*30)
        epoch_features = {}
        for band_name, (low, high) in bands.items():
            idx_band = np.logical_and(freqs >= low, freqs <= high)
            epoch_features[band_name] = np.mean(psd[idx_band])

        features.append(epoch_features)

        diff = np.diff(epoch)
        mobility = _safe_sqrt_variance_ratio(diff, epoch)
        diff2 = np.diff(diff)
        mobility_diff = _safe_sqrt_variance_ratio(diff2, diff)
        complexity = mobility_diff / mobility if mobility > 0 else 0
        complexities.append({'Hjorth_Mobility': mobility, 'Hjorth_Complexity': complexity})

    return pd.DataFrame(features), pd.DataFrame(complexities)


def get_patient_profile(df_features):
    total_power = df_features.sum(axis=1)
    avg_p = df_features.mean()
    total_avg_p = avg_p.sum()

    # Relative Powers
    rel_powers = df_features.div(total_power, axis=0).mean()
    rel_delta = avg_p['Delta'] / total_avg_p
    rel_beta = rel_powers['Beta']

    # Cross-frequency ratios
    tar = avg_p['Theta'] / avg_p['Alpha']
    tbr = avg_p['Theta'] / avg_p['Beta']

    # Variability (Coefficient of Variation)
    var_delta = df_features['Delta'].std() / df_features['Delta'].mean() if df_features['Delta'].mean() > 0 else np.nan

    metrics = {
        'Relative_Delta_Power': rel_delta,
        'Theta_Alpha_Ratio': tar,
        'Theta_Beta_Ratio': tbr,
        'Rel_Beta': rel_beta,
        'variability_Delta': var_delta,
    }

    return pd.Series(metrics)
