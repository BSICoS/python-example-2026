import os
import torch
import numpy as np

from .resnet import ResNet1d_mse

# Configuración de Rutas para los pesos (.pth)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHT_PATH = os.path.join(CURRENT_DIR, 'model.pth')

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_LEADS = 1
SEQ_LEN = 2000  # Fijo para PROPHECG (10s a 200 Hz)
BLOCKS = [(32, 2000), (64, 400), (128, 80), (256, 16), (512, 4)]
KERNEL = 7
DROPOUT = 0.4
SCALE = 5.0

def load_prophecg_model():
    """Carga y retorna el modelo en memoria."""
    model = ResNet1d_mse(
        input_dim=(N_LEADS, SEQ_LEN),
        blocks_dim=BLOCKS,
        n_classes=1,
        kernel_size=KERNEL,
        dropout_rate=DROPOUT
    )
    checkpoint = torch.load(WEIGHT_PATH, map_location=DEVICE)
    state_dict = checkpoint.get('model', checkpoint)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model

# Inicializamos el modelo de forma global
MODEL = load_prophecg_model()

def compute_ecgage(ecg_signal):
    # 1) Asegurar que la longitud sea exactamente 2000 muestras
    if len(ecg_signal) > SEQ_LEN:
        ecg_signal = ecg_signal[:SEQ_LEN]
    elif len(ecg_signal) < SEQ_LEN:
        ecg_signal = np.pad(ecg_signal, (0, SEQ_LEN - len(ecg_signal)), 'edge')
    # 2) Preprocesamiento de la señal ECG
    def normalize(x: np.ndarray) -> np.ndarray:
        """3) Min–Max 정규화 (0–1)"""
        mn, mx = x.min(), x.max()
        return (x - mn) / (mx - mn) if mx != mn else np.zeros_like(x)

    ecg_proc = normalize(ecg_signal)

    # 2) Dispositivo e Hiperparámetros
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 3) Convertir la señal procesada a Tensor
    x = torch.from_numpy(ecg_proc.astype('float32')).unsqueeze(0).unsqueeze(0).to(DEVICE)

    # 4) Inferencia
    with torch.no_grad():
        output = MODEL(x).item()

    # 10) Resultado final
    predicted_age = output * SCALE

    return float(predicted_age)