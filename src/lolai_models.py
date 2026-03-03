import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from captum.attr import (
    IntegratedGradients,
    LayerGradCam,
    LayerAttribution,
    Occlusion,
    GradientShap
)

from typing import Dict, List, Tuple, Union, Optional


class CNN_LSTM_Classifier(nn.Module):
    def __init__(self, input_channels=3, hidden_dim=64, num_classes=3, dropout=0.3):
        super(CNN_LSTM_Classifier, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(input_channels, 6, kernel_size=5, padding=2),
            nn.BatchNorm1d(6),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(dropout),

            nn.Conv1d(6, 9, kernel_size=3, padding=1),
            nn.BatchNorm1d(9),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(dropout),


            nn.Conv1d(9, 18, kernel_size=3, padding=1),
            nn.BatchNorm1d(18),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(dropout)
        )

        self.lstm = nn.LSTM(
            input_size=18,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True  # Use bidirectional LSTM for better context
        )

        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden_dim, 64),  # Adjust for bidirectional LSTM
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x: (batch, channels, time)
        x = self.cnn(x)  # (batch, features, time)
        x = x.permute(0, 2, 1)  # (batch, time, features)
        _, (h_n, _) = self.lstm(x)  # h_n: (num_layers * num_directions, batch, hidden_dim)
        h_n = torch.cat((h_n[-2], h_n[-1]), dim=1)  # Concatenate forward and backward states
        out = self.classifier(h_n)  # (batch, num_classes)
        return out

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_LSTM_Classifier_XAI(nn.Module):
    def __init__(self, input_channels=3, hidden_dim=32, num_classes=3, dropout=0.4):
        super(CNN_LSTM_Classifier_XAI, self).__init__()
        
        self.cnn_activations = []
        self.lstm_activations = None
        self.attention_weights = None
        self.gradients = None
        self.last_cnn_output = None
        self.input = None

        # CNN
        self.conv1 = nn.Conv1d(input_channels, 16, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.dropout = nn.Dropout(dropout)

        # LSTM
        self.lstm = nn.LSTM(input_size=64, hidden_size=hidden_dim,
                            batch_first=True, bidirectional=True)

        # Atención
        self.attention = nn.Linear(2 * hidden_dim, 1)

        # Clasificador
        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes)
        )

    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x, return_attention=False, track_gradients=False):
        self.input = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        self.cnn_activations.append(x.detach())
        x = self.pool(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        self.cnn_activations.append(x.detach())
        x = self.pool(x)
        x = self.dropout(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        cnn_output = x

        if track_gradients and cnn_output.requires_grad:
            cnn_output.register_hook(self.activations_hook)

        self.last_cnn_output = cnn_output  # necesario para Grad-CAM
        self.cnn_activations.append(cnn_output.detach())

        x = self.pool(cnn_output)
        x = self.dropout(x)

        x = x.permute(0, 2, 1)  # (batch, time, features)
        lstm_out, _ = self.lstm(x)
        self.lstm_activations = lstm_out.detach()

        attention_scores = self.attention(lstm_out).squeeze(-1)
        attention_weights = F.softmax(attention_scores, dim=1)
        self.attention_weights = attention_weights.detach()

        context_vector = torch.bmm(attention_weights.unsqueeze(1), lstm_out).squeeze(1)
        out = self.classifier(context_vector)

        if return_attention:
            return out, attention_weights
        return out

    def reset_activation_storage(self):
        self.cnn_activations = []
        self.lstm_activations = None
        self.attention_weights = None
        self.gradients = None
        self.last_cnn_output = None
        self.input = None

    def interpret(self, x, class_idx=None):
        self.reset_activation_storage()

        was_training = self.training
        lstm_was_training = self.lstm.training

        self.eval()
        self.lstm.train()  # necesario para CuDNN backward

        x.requires_grad_()
        logits, attention = self.forward(x, return_attention=True, track_gradients=True)
        pred = torch.softmax(logits, dim=1)

        if class_idx is None:
            class_idx = pred.argmax(dim=1)

        for i in range(x.shape[0]):
            pred[i, class_idx[i]].backward(retain_graph=True)

        self.train(was_training)
        self.lstm.train(lstm_was_training)

        feature_importance = self.get_feature_importance()
        temporal_channel_importance = self.get_temporal_channel_importance()
        channel_imp=self.get_channel_importance()

        self.input = None  # limpieza para evitar problemas de memoria
        torch.cuda.empty_cache()

        return {
            'prediction': pred.detach(),
            'class_idx': class_idx,
            'attention_weights': self.attention_weights,
            'feature_importance': feature_importance,
            'cnn_activations': self.cnn_activations,
            'temporal_channel_importance': temporal_channel_importance,
            'channel_importance': channel_imp
        }

    def get_feature_importance(self):
        """
        Grad-CAM temporal sobre la salida del último bloque CNN.
        Devuelve tensor (batch, time)
        """
        if self.gradients is None or self.last_cnn_output is None:
            return None

        pooled_gradients = torch.mean(self.gradients, dim=[0, 2])  # (channels,)
        cam = self.last_cnn_output.clone()

        for i in range(cam.shape[1]):
            cam[:, i, :] *= pooled_gradients[i]

        heatmap = torch.mean(cam, dim=1).detach()  # (batch, time)
        return heatmap

    def get_channel_importance(self):
        """
        Importancia por canal: (batch, channels)
        """
        if self.input.grad is None:
            raise ValueError("Gradientes de la entrada no están disponibles. Llama primero a interpret().")
        return self.input.grad.abs().mean(dim=2)

    def get_temporal_channel_importance(self):
        """
        Importancia canal-temporal: (batch, channels, time)
        """
        if self.input.grad is None:
            raise ValueError("Gradients of the input are not available. Call interpret() first.")
        return self.input.grad.abs().detach()






class ContrastiveVAE(nn.Module):
    def __init__(self, in_channels=4, latent_dim=32, lstm_hidden=64, n_classes=3, use_classifier=False):
        super().__init__()
        self.use_classifier = use_classifier

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.global_pool = nn.AdaptiveAvgPool1d(1)  # for VAE path
        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)

        # Decoder (for reconstruction)
        self.decoder_input = nn.Linear(latent_dim, 64)
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(32, in_channels, kernel_size=3, padding=1)
        )

        # LSTM + Classifier always initialized (but optionally used)
        self.lstm = nn.LSTM(input_size=64, hidden_size=lstm_hidden, batch_first=True, bidirectional=True)
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes)
        )

    def encode(self, x):
        h = self.encoder(x)  # (B, 64, T)
        pooled = self.global_pool(h).squeeze(-1)  # (B, 64)
        mu = self.fc_mu(pooled)
        logvar = self.fc_logvar(pooled)
        return mu, logvar, h  # h is (B, 64, T)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, length):
        # Mejora la reconstrucción con una capa de proyección inicial
        h = self.decoder_input(z).unsqueeze(-1)  # (B, 64, 1)
        # Usar interpolación para un escalado más suave en lugar de expand
        h = F.interpolate(h, size=length, mode='linear', align_corners=False)
        x_recon = self.decoder(h)
        return x_recon

    def forward(self, x):
        B, C, T = x.shape
        mu, logvar, features = self.encode(x)  # features: (B, 64, T)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z, T)

        logits = None
        if self.use_classifier:
            features_t = features.permute(0, 2, 1)  # (B, T, 64)
            lstm_out, _ = self.lstm(features_t)     # (B, T, 2*hidden)
            lstm_feat = lstm_out.mean(dim=1)        # (B, 2*hidden)
            logits = self.classifier(lstm_feat)     # (B, n_classes)

        return x_recon, mu, logvar, z, logits

    def get_latents(self, x, use_mean=True):
        mu, logvar, _ = self.encode(x)
        return mu if use_mean else self.reparameterize(mu, logvar)

    def classify(self, x):
        """Forward through the classifier only (requires use_classifier = True)."""
        assert self.use_classifier, "Classifier is not enabled. Set model.use_classifier = True before calling classify."
        _, _, features = self.encode(x)
        features_t = features.permute(0, 2, 1)
        lstm_out, _ = self.lstm(features_t)
        lstm_feat = lstm_out.mean(dim=1)
        return self.classifier(lstm_feat)


def vae_loss(recon_x, x, mu, logvar):
    recon_loss = F.mse_loss(recon_x, x, reduction='mean')
    kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_div, recon_loss, kl_div


def contrastive_loss(z, ids, temperature=0.1):
    z = F.normalize(z, dim=1)
    sim = torch.mm(z, z.T) / temperature
    labels = ids.view(-1, 1)
    mask = torch.eq(labels, labels.T).float().to(z.device)
    mask = mask - torch.eye(len(z), device=z.device)
    exp_sim = torch.exp(sim) * (1 - torch.eye(len(z), device=z.device))
    log_prob = sim - torch.log(exp_sim.sum(1, keepdim=True) + 1e-8)
    mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)
    return -mean_log_prob_pos.mean()


def intra_patient_loss(latents, patient_ids):
    """Versión optimizada que evita bucles explícitos"""
    # Convertir IDs a tensor si no lo son ya
    if not isinstance(patient_ids, torch.Tensor):
        patient_ids = torch.tensor(patient_ids, device=latents.device)
    
    # Crear matriz de similaridad de pacientes (1 donde son iguales)
    patient_sim = (patient_ids.unsqueeze(1) == patient_ids.unsqueeze(0)).float()
    # Quitar diagonal (mismo ejemplo)
    mask = patient_sim - torch.eye(len(latents), device=latents.device)
    # Calcular distancias entre latentes
    latent_dists = torch.cdist(latents, latents, p=2)
    # Aplicar máscara y promediar
    valid_pairs = mask.sum()
    if valid_pairs > 0:
        return (mask * latent_dists).sum() / valid_pairs
    return torch.tensor(0.0, device=latents.device)


def training_step(model, batch1, batch2, patient_ids, optimizer, alpha=0.1, beta=1.0):
    model.train()
    x1, x2 = batch1, batch2
    x1_recon, mu1, logvar1, z1, _ = model(x1)
    x2_recon, mu2, logvar2, z2, _ = model(x2)

    recon1, r1, kl1 = vae_loss(x1_recon, x1, mu1, logvar1)
    recon2, r2, kl2 = vae_loss(x2_recon, x2, mu2, logvar2)
    vae_total = (recon1 + recon2) / 2

    z_all = torch.cat([z1, z2], dim=0)
    # Asegurar que IDs sean tensores
    ids = torch.arange(len(z1), device=z1.device).repeat(2)
    contrastive = contrastive_loss(z_all, ids)

    # Extender patient_ids correctamente
    if isinstance(patient_ids, torch.Tensor):
        p_ids = torch.cat([patient_ids, patient_ids], dim=0)
    else:
        p_ids = patient_ids + patient_ids  # Si es una lista
    
    patient_reg = intra_patient_loss(z_all, p_ids)

    total = vae_total + alpha * contrastive + beta * patient_reg

    optimizer.zero_grad()
    total.backward()
    optimizer.step()

    return {
        'total_loss': total.item(),
        'recon_loss': vae_total.item(),
        'contrastive': contrastive.item(),
        'patient_reg': patient_reg.item(),
        'kl_loss': (kl1 + kl2).item() / 2
    }


def fine_tune_step(model, x, y, optimizer, criterion=nn.CrossEntropyLoss()):
    model.train()
    model.use_classifier = True
    logits = model.classify(x)
    loss = criterion(logits, y)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    preds = torch.argmax(logits, dim=1)
    acc = (preds == y).float().mean().item()

    return {
        'classification_loss': loss.item(),
        'accuracy': acc
    }

class ImprovedPainClassifier(nn.Module):
    def __init__(self, input_channels=3, hidden_dim=128, num_classes=3, dropout=0.4):
        super(ImprovedPainClassifier, self).__init__()
        
        # Increased regularization and feature extraction for small datasets
        self.cnn = nn.Sequential(
            # Layer 1: More filters to capture diverse patterns
            nn.Conv1d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),  # LeakyReLU helps with gradient flow
            nn.MaxPool1d(kernel_size=2),
            
            # Layer 2: Increased complexity
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(dropout),
            
            # Layer 3: Additional layer for better feature extraction
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout)
        )
        
        # Attention mechanism to focus on important temporal patterns
        self.attention = nn.Sequential(
            nn.Linear(256, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
        # Bidirectional LSTM with residual connections
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_dim,
            num_layers=2,  # Multiple layers for complex temporal patterns
            batch_first=True,
            bidirectional=True,
            dropout=dropout  # Apply dropout between LSTM layers
        )
        
        # Classifier with additional regularization
        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),  # Normalize activations
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
    def forward(self, x):
        # x: (batch, channels, time) - BVP, EDA, and respiratory signals
        
        # Extract features with CNN
        cnn_out = self.cnn(x)  # (batch, 128, time')
        
        # Reshape for LSTM
        cnn_out = cnn_out.permute(0, 2, 1)  # (batch, time', 128)
        
        # Process with LSTM
        lstm_out, (h_n, _) = self.lstm(cnn_out)  # lstm_out: (batch, time', 2*hidden_dim)
        
        # Apply attention to focus on relevant parts of the signal
        attn_weights = self.attention(lstm_out).softmax(dim=1)  # (batch, time', 1)
        context = torch.sum(attn_weights * lstm_out, dim=1)  # (batch, 2*hidden_dim)
        
        # Alternative: Use concatenated hidden states from both directions
        # h_n = torch.cat((h_n[-2], h_n[-1]), dim=1)  # (batch, 2*hidden_dim)
        
        # Classify
        out = self.classifier(context)  # (batch, num_classes)
        return out


import matplotlib.pyplot as plt
import numpy as np
import torch

class ExplainabilityVisualizer:
    def __init__(self, channel_names=None):
        """
        channel_names: lista opcional con nombres de los canales de entrada
        """
        self.channel_names = channel_names

    def plot_attention_weights(self, attention_weights, title="Atención temporal"):
        attention = attention_weights.squeeze().cpu().numpy()
        plt.figure(figsize=(10, 2))
        plt.plot(attention)
        plt.title(title)
        plt.xlabel("Timestep")
        plt.ylabel("Weight")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_gradcam_heatmap(self, heatmap, title="Grad-CAM temporal"):
        heat = heatmap.squeeze().cpu().numpy()
        plt.figure(figsize=(10, 2))
        plt.plot(heat)
        plt.title(title)
        plt.xlabel("Timestep")
        plt.ylabel("Importance")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_channel_importance(self, channel_importance, title="Importancia por canal"):
        values = channel_importance.squeeze().cpu().numpy()
        channels = self.channel_names if self.channel_names else [f"Channel {i}" for i in range(len(values))]
        plt.figure(figsize=(6, 3))
        plt.bar(channels, values)
        plt.title(title)
        plt.ylabel("Importancia media")
        plt.xticks(rotation=45)
        plt.grid(axis='y')
        plt.tight_layout()
        plt.show()

    def plot_signals_with_attention_highlight(self, x, importance, threshold=0.85, title="Señal con zonas de atención"):
        """
        Dibuja las señales multicanal y sombreado rojo donde la importancia temporal supera el umbral.

        Args:
            x: Tensor (channels, time)
            importance: Tensor (time,)
            threshold: percentil (0-1) o valor absoluto
            title: título del gráfico
        """
        x = x.detach().cpu().numpy()
        importance = importance.detach().cpu().numpy()
        time = np.arange(x.shape[1])
        n_channels = x.shape[0]

        if threshold <= 1.0:
            threshold_value = np.quantile(importance, threshold)
        else:
            threshold_value = threshold

        high_attention_mask = importance >= threshold_value

        fig, axs = plt.subplots(n_channels, 1, figsize=(12, 2.5 * n_channels), sharex=True)
        if n_channels == 1:
            axs = [axs]

        for i in range(n_channels):
            axs[i].plot(time, x[i], label=self.channel_names[i] if self.channel_names else f"Canal {i}", color="black")
            axs[i].set_ylabel("Valor")
            axs[i].grid(True)

            in_high = False
            start = 0
            for t in range(len(high_attention_mask)):
                if high_attention_mask[t] and not in_high:
                    start = t
                    in_high = True
                elif not high_attention_mask[t] and in_high:
                    axs[i].axvspan(start, t, color='red', alpha=0.25)
                    in_high = False
            if in_high:
                axs[i].axvspan(start, len(high_attention_mask), color='red', alpha=0.25)

            axs[i].legend(loc="upper right")

        axs[-1].set_xlabel("Tiempo (muestras)")
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()



class FocalLoss(nn.Module):
    """
    Focal Loss para clasificación binaria y multiclase.
    
    Parámetros:
    - alpha: Factor de ponderación para manejar desequilibrio de clases.
             Puede ser un escalar (mismo valor para todas las clases) o
             un tensor (valores específicos por clase).
    - gamma: Factor de modulación para enfocar en ejemplos difíciles (>= 0).
    - reduction: 'none' | 'mean' | 'sum'
    - eps: Pequeño valor para estabilidad numérica
    
    Referencias:
    - Paper original: "Focal Loss for Dense Object Detection" por Lin et al.
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean', eps=1e-6):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.eps = eps
        
    def forward(self, inputs, targets):
        """
        Args:
            inputs: Logits de forma [B, C] donde B es el tamaño del batch y C es el número de clases.
                   Para clasificación binaria, C puede ser 1.
            targets: Etiquetas de objetivos de forma [B] para multiclase o [B, 1] para binaria.
                    Valores enteros para multiclase (clases indexadas desde 0 a C-1).
                    Valores continuos entre 0 y 1 para binaria.
        """
        # Determinar si es clasificación binaria o multiclase
        if inputs.shape[1] == 1 or inputs.shape[1] == 2:  # Binaria
            # Aplicar sigmoide para obtener probabilidades
            probs = torch.sigmoid(inputs.view(-1))
            targets = targets.view(-1)
            
            # Calcular pt (probabilidad del objetivo correcto)
            pt = probs * targets + (1 - probs) * (1 - targets)
            
            # Aplicar factores de ponderación
            if isinstance(self.alpha, (float, int)):
                alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            else:
                # Si alpha es un tensor, usar indexación
                alpha_t = self.alpha if self.alpha is not None else torch.ones_like(pt)
            
            # Calcular la focal loss
            focal_weight = (1 - pt).pow(self.gamma)
            loss = -alpha_t * focal_weight * torch.log(pt.clamp(min=self.eps))
            
        else:  # Multiclase
            # Convertir logits a distribución de probabilidad
            log_softmax = F.log_softmax(inputs, dim=1)
            
            # Obtener log probabilidad para las clases objetivo
            targets = targets.view(-1, 1)
            log_pt = log_softmax.gather(1, targets).view(-1)
            pt = log_pt.exp()  # Obtener probabilidades
            
            # Aplicar factores de ponderación
            if isinstance(self.alpha, (list, tuple, torch.Tensor)):
                # Si alpha es específico por clase
                alpha = torch.tensor(self.alpha, device=inputs.device)
                alpha_t = alpha.gather(0, targets.view(-1))
            else:
                alpha_t = self.alpha if self.alpha is not None else 1.0
            
            # Calcular focal loss
            focal_weight = (1 - pt).pow(self.gamma)
            loss = -alpha_t * focal_weight * log_pt
        
        # Aplicar reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss
        



class CNN_LSTM_Classifier_XAI_2(nn.Module):
    def __init__(self, input_channels=3, hidden_dim=32, num_classes=3, dropout=0.1):
        super(CNN_LSTM_Classifier_XAI_2, self).__init__()
        
        self.cnn_activations = []
        self.lstm_activations = None
        self.attention_weights = None
        self.gradients = None
        self.last_cnn_output = None
        self.input = None
        self.input_channels = input_channels

        # CNN
        self.conv1 = nn.Conv1d(input_channels, 16, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.dropout = nn.Dropout(dropout)

        # LSTM
        self.lstm = nn.LSTM(input_size=64, hidden_size=hidden_dim,
                            batch_first=True, bidirectional=True)

        # Attention
        self.attention = nn.Linear(2 * hidden_dim, 1)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes)
        )

    def activations_hook(self, grad):
        self.gradients = grad

    

    def forward(self, x, return_attention=False, track_gradients=False):
        # Reset activation storage at the beginning of each forward pass
        self.reset_activation_storage()
        
        self.input = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        self.cnn_activations.append(x.detach())
        x = self.pool(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        self.cnn_activations.append(x.detach())
        x = self.pool(x)
        x = self.dropout(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        cnn_output = x

        if track_gradients and cnn_output.requires_grad:
            cnn_output.register_hook(self.activations_hook)

        self.last_cnn_output = cnn_output  # needed for Grad-CAM
        self.cnn_activations.append(cnn_output.detach())

        x = self.pool(cnn_output)
        x = self.dropout(x)

        x = x.permute(0, 2, 1)  # (batch, time, features)
        lstm_out, (h_n, c_n) = self.lstm(x)
        self.lstm_activations = lstm_out.detach()

        attention_scores = self.attention(lstm_out).squeeze(-1)
        attention_weights = F.softmax(attention_scores, dim=1)
        self.attention_weights = attention_weights.detach()

        context_vector = torch.bmm(attention_weights.unsqueeze(1), lstm_out).squeeze(1)
        out = self.classifier(context_vector)

        if return_attention:
            return out, attention_weights
        return out

    def reset_activation_storage(self):
        self.cnn_activations = []
        self.lstm_activations = None
        self.attention_weights = None
        self.gradients = None
        self.last_cnn_output = None

    def interpret(self, x, class_idx=None, methods=None):
        """
        Enhanced interpretation method with multiple explainability techniques
        
        Args:
            x: Input data tensor
            class_idx: Target class indices to explain (defaults to predicted class)
            methods: List of methods to use, options: ['gradcam', 'integrated_gradients', 
                     'occlusion', 'shap', 'feature_ablation', 'all']
                      
        Returns:
            Dictionary with various interpretability outputs
        """
        if methods is None:
            methods = ['gradcam', 'attention']  # Default methods
        if 'all' in methods:
            methods = ['gradcam', 'integrated_gradients', 'occlusion', 'shap', 
                      'feature_ablation', 'attention', 'layer_importance']
        
        # Store original training state
        was_training = self.training
        lstm_was_training = self.lstm.training

        # Set model to evaluation mode for interpretability
        self.eval()
        self.lstm.train()  # needed for CuDNN backward compatibility
        
        # Base prediction
        x.requires_grad_()
        self.input = x  # Store input for interpretability methods
        
        logits, attention = self.forward(x, return_attention=True, track_gradients=True)
        pred = torch.softmax(logits, dim=1)

        if class_idx is None:
            class_idx = pred.argmax(dim=1)
        
        # Initialize results dictionary
        results = {
            'prediction': pred.detach(),
            'class_idx': class_idx,
            'attention_weights': self.attention_weights,
        }
        
        # Apply selected interpretability methods
        if 'gradcam' in methods:
            for i in range(x.shape[0]):
                pred[i, class_idx[i]].backward(retain_graph=True if i < x.shape[0]-1 else False)
            
            results['feature_importance'] = self.get_feature_importance()
            results['temporal_channel_importance'] = self.get_temporal_channel_importance()
            results['channel_importance'] = self.get_channel_importance()
            results['cnn_activations'] = self.cnn_activations
            
        # Integrated Gradients
        if 'integrated_gradients' in methods:
            ig = IntegratedGradients(self.forward_wrapper)
            results['integrated_gradients'] = self._compute_integrated_gradients(
                ig, x, class_idx)
        
        # Occlusion analysis
        if 'occlusion' in methods:
            occlusion = Occlusion(self.forward_wrapper)
            results['occlusion'] = self._compute_occlusion(occlusion, x, class_idx)
        
        # SHAP (GradientSHAP implementation)
        if 'shap' in methods:
            gradient_shap = GradientShap(self.forward_wrapper)
            results['gradient_shap'] = self._compute_gradient_shap(gradient_shap, x, class_idx)
        
        # Feature ablation (sensitivity analysis)
        if 'feature_ablation' in methods:
            results['feature_ablation'] = self._feature_ablation_analysis(x, class_idx)
        
        # Layer importance analysis
        if 'layer_importance' in methods:
            results['layer_importance'] = self._compute_layer_importance(x, class_idx)
            
        # Restore original training states
        self.train(was_training)
        self.lstm.train(lstm_was_training)
        
        # Clean up to avoid memory issues
        self.input = None
        torch.cuda.empty_cache()
        
        return results

    def forward_wrapper(self, x):
        """Wrapper for Captum compatibility"""
        return self.forward(x)
    
    def get_feature_importance(self):
        """
        Grad-CAM temporal over the output of the last CNN block.
        Returns tensor (batch, time)
        """
        if self.gradients is None or self.last_cnn_output is None:
            return None

        pooled_gradients = torch.mean(self.gradients, dim=[0, 2])  # (channels,)
        cam = self.last_cnn_output.clone()

        for i in range(cam.shape[1]):
            cam[:, i, :] *= pooled_gradients[i]

        heatmap = torch.mean(cam, dim=1).detach()  # (batch, time)
        
        # Apply ReLU to highlight only positive influences
        heatmap = F.relu(heatmap)
        
        # Normalize heatmap for better visualization
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
            
        return heatmap

    def get_channel_importance(self):
        """
        Channel importance: (batch, channels)
        """
        if self.input is None or self.input.grad is None:
            raise ValueError("Input gradients not available. Call interpret() first.")
        return self.input.grad.abs().mean(dim=2).detach()

    def get_temporal_channel_importance(self):
        """
        Temporal-channel importance: (batch, channels, time)
        """
        if self.input is None or self.input.grad is None:
            raise ValueError("Input gradients not available. Call interpret() first.")
        return self.input.grad.abs().detach()
    
    def _compute_integrated_gradients(self, ig, x, class_idx):
        """Compute integrated gradients attribution"""
        batch_size = x.shape[0]
        attributions = []
        
        for i in range(batch_size):
            baseline = torch.zeros_like(x[i:i+1])
            attr = ig.attribute(
                x[i:i+1], baseline, target=class_idx[i].item(), n_steps=50
            )
            attributions.append(attr)
            
        return torch.cat(attributions).detach()
    
    def _compute_occlusion(self, occlusion_algo, x, class_idx):
        """Compute occlusion-based feature attribution"""
        batch_size = x.shape[0]
        attributions = []
        
        # Define sliding window parameters for temporal data
        window_size = min(5, x.shape[2] // 4)  # Adapt window size to input length
        
        for i in range(batch_size):
            attr = occlusion_algo.attribute(
                x[i:i+1], 
                sliding_window_shapes=(1, window_size),
                target=class_idx[i].item(),
                strides=(1, max(1, window_size // 2))
            )
            attributions.append(attr)
            
        return torch.cat(attributions).detach()
    
    def _compute_gradient_shap(self, shap_algo, x, class_idx):
        """Compute GradientSHAP attributions"""
        batch_size = x.shape[0]
        attributions = []
        
        for i in range(batch_size):
            # Create random baselines (typically 10-50 for good estimates)
            baselines = torch.randn(10, *x[i:i+1].shape[1:]) * 0.001
            
            # Ensure baselines device matches input
            baselines = baselines.to(x.device)
            
            attr = shap_algo.attribute(
                x[i:i+1], baselines=baselines, target=class_idx[i].item()
            )
            attributions.append(attr)
            
        return torch.cat(attributions).detach()
    
    def _feature_ablation_analysis(self, x, class_idx):
        """Analyze model by systematically ablating input features"""
        batch_size = x.shape[0]
        results = []
        
        for i in range(batch_size):
            # Store original prediction
            with torch.no_grad():
                orig_output = self.forward(x[i:i+1])
                orig_prob = torch.softmax(orig_output, dim=1)[0, class_idx[i]].item()
            
            # Test ablation of each channel
            channel_importance = []
            for c in range(self.input_channels):
                # Create ablated input (zero out one channel)
                ablated_input = x[i:i+1].clone()
                ablated_input[:, c, :] = 0
                
                # Get prediction on ablated input
                with torch.no_grad():
                    ablated_output = self.forward(ablated_input)
                    ablated_prob = torch.softmax(ablated_output, dim=1)[0, class_idx[i]].item()
                
                # Impact is reduction in probability
                channel_impact = orig_prob - ablated_prob
                channel_importance.append(channel_impact)
            
            results.append(torch.tensor(channel_importance))
            
        return torch.stack(results)
    
    def _compute_layer_importance(self, x, class_idx):
        """Compute importance of each layer using Layer GradCAM"""
        batch_size = x.shape[0]
        layer_importance = {}
        
        # Define layers to analyze
        layers = {
            'conv1': self.conv1,
            'conv2': self.conv2,
            'conv3': self.conv3
        }
        
        for layer_name, layer in layers.items():
            layer_gradcam = LayerGradCam(self.forward_wrapper, layer)
            layer_attrs = []
            
            for i in range(batch_size):
                attr = layer_gradcam.attribute(
                    x[i:i+1], target=class_idx[i].item()
                )
                # Process attribution to create a single importance score per sample
                pooled_attr = torch.mean(attr, dim=1)

                layer_attrs.append(pooled_attr)
                
            layer_importance[layer_name] = torch.cat(layer_attrs).detach()
            
        return layer_importance
    
    def visualize_attributions(self, sample_idx, interpretations, time_axis=None, 
                               channel_names=None, class_names=None):
        """
        Visualize the various interpretation results
        
        Args:
            sample_idx: Index of the sample to visualize
            interpretations: Dictionary returned by interpret() method
            time_axis: Optional array/list with time points for x-axis
            channel_names: Optional list of channel names
            class_names: Optional list of class names
        """
        if not channel_names:
            channel_names = [f'Channel {i}' for i in range(self.input_channels)]
        
        if not class_names:
            class_idx = interpretations['class_idx'][sample_idx].item()
            class_name = f'Class {class_idx}'
        else:
            class_idx = interpretations['class_idx'][sample_idx].item()
            class_name = class_names[class_idx]
            
        # Set up figure
        plt.figure(figsize=(15, 12))
        
        # Original input visualization (top row, first column)
        plt.subplot(3, 3, 1)
        if self.input is not None:
            input_data = self.input[sample_idx].cpu().detach().numpy()
            if time_axis is not None:
                for i in range(input_data.shape[0]):
                    plt.plot(time_axis, input_data[i], label=channel_names[i])
            else:
                for i in range(input_data.shape[0]):
                    plt.plot(input_data[i], label=channel_names[i])
            plt.legend(loc='best')
            plt.title('Input Signal')
            plt.xlabel('Time')
            plt.ylabel('Value')
            
        # GradCAM feature importance (top row, second column)
        if 'feature_importance' in interpretations and interpretations['feature_importance'] is not None:
            plt.subplot(3, 3, 2)
            heatmap = interpretations['feature_importance'][sample_idx].cpu().numpy()
            if time_axis is not None:
                plt.plot(time_axis, heatmap)
            else:
                plt.plot(heatmap)
            plt.title('GradCAM Feature Importance')
            plt.xlabel('Time')
            plt.ylabel('Importance')
            
        # Attention weights (top row, third column)
        if 'attention_weights' in interpretations and interpretations['attention_weights'] is not None:
            plt.subplot(3, 3, 3)
            attention = interpretations['attention_weights'][sample_idx].cpu().numpy()
            
            if time_axis is not None:
                # Need to match attention time axis to input time axis
                # (account for pooling in the network)
                x_points = np.linspace(time_axis[0], time_axis[-1], len(attention))
                plt.plot(x_points, attention)
            else:
                plt.plot(attention)
            plt.title('Attention Weights')
            plt.xlabel('Time')
            plt.ylabel('Attention')
            
        # Channel importance (middle row, first column)
        if 'channel_importance' in interpretations and interpretations['channel_importance'] is not None:
            plt.subplot(3, 3, 4)
            ch_importance = interpretations['channel_importance'][sample_idx].cpu().numpy()
            plt.bar(channel_names, ch_importance)
            plt.title('Channel Importance')
            plt.ylabel('Importance')
            plt.xticks(rotation=45)
            
        # Integrated Gradients (middle row, second column)
        if 'integrated_gradients' in interpretations:
            plt.subplot(3, 3, 5)
            ig_attr = interpretations['integrated_gradients'][sample_idx].cpu().numpy()
            ig_attr_mean = np.mean(ig_attr, axis=0)  # Average across channels for visualization
            
            if time_axis is not None:
                plt.plot(time_axis, ig_attr_mean)
            else:
                plt.plot(ig_attr_mean)
            plt.title('Integrated Gradients')
            plt.xlabel('Time')
            plt.ylabel('Attribution')
            
        # Feature Ablation (middle row, third column)
        if 'feature_ablation' in interpretations:
            plt.subplot(3, 3, 6)
            ablation_scores = interpretations['feature_ablation'][sample_idx].cpu().numpy()
            plt.bar(channel_names, ablation_scores)
            plt.title('Feature Ablation Impact')
            plt.ylabel('Probability Change')
            plt.xticks(rotation=45)
            
        # SHAP values (bottom row, first column)
        if 'gradient_shap' in interpretations:
            plt.subplot(3, 3, 7)
            shap_attr = interpretations['gradient_shap'][sample_idx].cpu().numpy()
            # Visualize average SHAP value over time
            shap_avg = np.mean(shap_attr, axis=0)
            
            if time_axis is not None:
                plt.plot(time_axis, shap_avg)
            else:
                plt.plot(shap_avg)
            plt.title('GradientSHAP Values')
            plt.xlabel('Time')
            plt.ylabel('SHAP Value')
            
        # Occlusion analysis (bottom row, second column)
        if 'occlusion' in interpretations:
            plt.subplot(3, 3, 8)
            occlusion_attr = interpretations['occlusion'][sample_idx].cpu().numpy()
            occlusion_avg = np.mean(occlusion_attr, axis=0)
            
            if time_axis is not None:
                plt.plot(time_axis, occlusion_avg)
            else:
                plt.plot(occlusion_avg)
            plt.title('Occlusion Analysis')
            plt.xlabel('Time')
            plt.ylabel('Attribution')
        
        # Prediction summary (bottom row, third column)
        plt.subplot(3, 3, 9)
        pred_probs = interpretations['prediction'][sample_idx].cpu().numpy()
        classes = list(range(len(pred_probs)))
        if class_names:
            classes = class_names
        plt.bar(classes, pred_probs)
        plt.title(f'Prediction: {class_name}')
        plt.ylabel('Probability')
        plt.ylim([0, 1])
        
        plt.tight_layout()
        return plt.gcf()
    
    def generate_interpretation_report(self, input_data, class_idx=None, 
                                      channel_names=None, class_names=None, 
                                      time_axis=None, methods='all'):
        """
        Generate a comprehensive interpretation report for the given input
        
        Args:
            input_data: Input tensor to analyze
            class_idx: Target class indices (optional)
            channel_names: Names of input channels (optional)
            class_names: Names of output classes (optional)
            time_axis: Time points for x-axis (optional)
            methods: Explainability methods to use
            
        Returns:
            Dictionary containing interpretations and visualization figure
        """
        # Run all interpretation methods
        interpretations = self.interpret(input_data, class_idx, methods=methods)
        
        # Generate visualizations for each sample
        figures = []
        for i in range(input_data.shape[0]):
            fig = self.visualize_attributions(
                i, interpretations, 
                time_axis=time_axis,
                channel_names=channel_names, 
                class_names=class_names
            )
            figures.append(fig)
            plt.close(fig)  # Close to avoid display in notebooks
            
        return {
            'interpretations': interpretations,
            'figures': figures
        }
    


class CNN_LSTM_Classifier_XAI_2(nn.Module):
    def __init__(
        self,
        input_channels=3,
        num_classes=3,
        cnn_channels=(16, 32, 64),
        kernel_sizes=(5, 3, 3),
        pool_type="max",  # or 'avg'
        dropout=0.1,
        lstm_hidden_dim=32,
        lstm_num_layers=1,
        bidirectional=True,
        classifier_hidden_dim=32,
        attention_dim=None,  # None = default: 2 * lstm_hidden_dim
    ):
        super(CNN_LSTM_Classifier_XAI_2, self).__init__()
        
        self.input_channels = input_channels
        self.pool_type = pool_type
        self.dropout_rate = dropout
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.cnn_activations = []
        self.lstm_activations = None
        self.attention_weights = None
        self.gradients = None
        self.last_cnn_output = None
        self.input = None

        # CNN Blocks
        self.conv1 = nn.Conv1d(input_channels, cnn_channels[0], kernel_size=kernel_sizes[0], padding=kernel_sizes[0] // 2)
        self.bn1 = nn.BatchNorm1d(cnn_channels[0])
        
        self.conv2 = nn.Conv1d(cnn_channels[0], cnn_channels[1], kernel_size=kernel_sizes[1], padding=kernel_sizes[1] // 2)
        self.bn2 = nn.BatchNorm1d(cnn_channels[1])
        
        self.conv3 = nn.Conv1d(cnn_channels[1], cnn_channels[2], kernel_size=kernel_sizes[2], padding=kernel_sizes[2] // 2)
        self.bn3 = nn.BatchNorm1d(cnn_channels[2])
        
        self.pool = nn.MaxPool1d(kernel_size=2) if pool_type == "max" else nn.AvgPool1d(kernel_size=2)
        self.dropout = nn.Dropout(dropout)

        # LSTM
        self.lstm = nn.LSTM(
            input_size=cnn_channels[2],
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )

        # Attention
        attention_dim = attention_dim or self.num_directions * lstm_hidden_dim
        self.attention = nn.Linear(self.num_directions * lstm_hidden_dim, 1)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.num_directions * lstm_hidden_dim, classifier_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_dim, num_classes)
        )

    def activations_hook(self, grad):
        self.gradients = grad

    def reset_activation_storage(self):
        self.cnn_activations = []
        self.lstm_activations = None
        self.attention_weights = None
        self.gradients = None
        self.last_cnn_output = None

    def forward(self, x, return_attention=False, track_gradients=False):
        self.reset_activation_storage()
        self.input = x

        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        self.cnn_activations.append(x.detach())
        x = self.dropout(x)

        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        self.cnn_activations.append(x.detach())
        x = self.dropout(x)

        x = F.relu(self.bn3(self.conv3(x)))
        cnn_output = x

        if track_gradients and cnn_output.requires_grad:
            cnn_output.register_hook(self.activations_hook)

        self.last_cnn_output = cnn_output
        self.cnn_activations.append(cnn_output.detach())

        x = self.pool(cnn_output)
        x = self.dropout(x)

        x = x.permute(0, 2, 1)  # (batch, time, features)
        lstm_out, _ = self.lstm(x)
        self.lstm_activations = lstm_out.detach()

        attention_scores = self.attention(lstm_out).squeeze(-1)
        attention_weights = F.softmax(attention_scores, dim=1)
        self.attention_weights = attention_weights.detach()

        context_vector = torch.bmm(attention_weights.unsqueeze(1), lstm_out).squeeze(1)
        out = self.classifier(context_vector)

        if return_attention:
            return out, attention_weights
        return out
    

class CNN_LSTM_Classifier_Tunable(nn.Module):
    def __init__(
        self,
        config: Optional[Dict] = None,
        input_channels: int = 3,
        seq_length: int = None,
        num_classes: int = 3,
        cnn_channels: Tuple[int, ...] = (16, 32, 64),
        kernel_sizes: Tuple[int, ...] = (5, 3, 3),
        pool_type: str = "max",  # or 'avg'
        pool_sizes: Tuple[int, ...] = (2, 2, 2),
        use_batch_norm: bool = True,
        activation: str = "relu",  # "relu", "leaky_relu", "elu", "gelu"
        dropout: float = 0.1,
        cnn_dropout: Optional[float] = None,  # Separate dropout for CNN
        lstm_hidden_dim: int = 32,
        lstm_num_layers: int = 1,
        bidirectional: bool = True,
        lstm_dropout: Optional[float] = None,  # Separate dropout for LSTM
        classifier_hidden_dims: List[int] = [32],  # Multiple hidden layers
        attention_dim: Optional[int] = None,  # None = default: 2 * lstm_hidden_dim
        attention_type: str = "basic",  # "basic", "scaled_dot", "multi_head"
        multi_head_num: int = 4,  # For multi-head attention
        residual_connections: bool = False,
        layer_normalization: bool = False,
        weight_init: str = "default",  # "default", "xavier", "kaiming"
    ):
        """
        Enhanced CNN-LSTM model with attention mechanism designed for tuning flexibility.
        
        Args:
            config: Optional dictionary with all hyperparameters to override other arguments
            input_channels: Number of input channels
            seq_length: Length of input sequence (needed for some operations)
            num_classes: Number of output classes
            cnn_channels: Tuple of CNN output channels for each layer
            kernel_sizes: Tuple of kernel sizes for each CNN layer
            pool_type: Pooling type ("max" or "avg")
            pool_sizes: Pooling sizes for each layer
            use_batch_norm: Whether to use batch normalization
            activation: Activation function type
            dropout: Default dropout rate
            cnn_dropout: CNN-specific dropout (if None, uses dropout)
            lstm_hidden_dim: LSTM hidden dimension
            lstm_num_layers: Number of LSTM layers
            bidirectional: Whether LSTM is bidirectional
            lstm_dropout: LSTM-specific dropout (if None, uses dropout)
            classifier_hidden_dims: List of hidden dimensions for classifier
            attention_dim: Attention dimension
            attention_type: Type of attention mechanism
            multi_head_num: Number of heads for multi-head attention
            residual_connections: Whether to use residual connections
            layer_normalization: Whether to use layer normalization
            weight_init: Weight initialization strategy
        """
        super(CNN_LSTM_Classifier_Tunable, self).__init__()
        
        # Override with config if provided
        if config is not None:
            # Set all attributes from config
            for key, value in config.items():
                if hasattr(self, key):
                    setattr(self, key, value)
                elif key in locals():
                    locals()[key] = value
        
        # Store parameters
        self.input_channels = input_channels
        self.seq_length = seq_length
        self.num_classes = num_classes
        self.cnn_channels = cnn_channels
        self.kernel_sizes = kernel_sizes
        self.pool_type = pool_type
        self.pool_sizes = pool_sizes
        self.use_batch_norm = use_batch_norm
        self.activation_type = activation
        self.dropout_rate = dropout
        self.cnn_dropout_rate = cnn_dropout if cnn_dropout is not None else dropout
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_num_layers = lstm_num_layers
        self.bidirectional = bidirectional
        self.lstm_dropout_rate = lstm_dropout if lstm_dropout is not None else dropout
        self.classifier_hidden_dims = classifier_hidden_dims
        self.residual_connections = residual_connections
        self.layer_normalization = layer_normalization
        self.weight_init = weight_init
        self.attention_type = attention_type
        self.multi_head_num = multi_head_num
        
        # Calculate directions
        self.num_directions = 2 if bidirectional else 1
        
        # Default attention dimension if not provided
        self.attention_dim = attention_dim or self.num_directions * lstm_hidden_dim
        
        # Precalcular dimensiones de secuencia después de las capas CNN
        self.input_seq_length = seq_length
        self.output_seq_length = None
        
        if seq_length is not None:
            # Calcular reducción de secuencia por pooling
            seq_reduction = 1
            current_length = seq_length
            
            for pool_size in self.pool_sizes:
                current_length = (current_length + pool_size - 1) // pool_size  # Ceil division
                seq_reduction *= pool_size
                
            self.output_seq_length = current_length
            
            # Verificar dimensiones válidas
            if self.output_seq_length <= 0:
                raise ValueError(f"La secuencia resultante después del pooling es demasiado corta. "
                            f"Secuencia entrada: {seq_length}, reducción: {seq_reduction}")
        
        # For visualization and explanation
        self.cnn_activations = []
        self.lstm_activations = None
        self.attention_weights = None
        self.gradients = None
        self.last_cnn_output = None
        self.input = None
        
        # Create activation function
        self.activation = self._get_activation()
        
        # Create CNN layers
        self.cnn_blocks = nn.ModuleList()
        in_channels = input_channels
        
        for i, (out_channels, kernel_size, pool_size) in enumerate(zip(cnn_channels, kernel_sizes, pool_sizes)):
            block = nn.ModuleDict()
            block["conv"] = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
            
            if use_batch_norm:
                block["bn"] = nn.BatchNorm1d(out_channels)
                
            if layer_normalization:
                # Usamos LayerNorm correctamente para normalizar sobre la dimensión de características
                block["ln"] = nn.LayerNorm([out_channels])
                
            if pool_type == "max":
                block["pool"] = nn.MaxPool1d(kernel_size=pool_size)
            else:
                block["pool"] = nn.AvgPool1d(kernel_size=pool_size)
                
            block["dropout"] = nn.Dropout(self.cnn_dropout_rate)
            
            # Determinar si este bloque puede usar conexión residual
            # Solo si tienen mismas dimensiones de entrada y salida
            if residual_connections and in_channels == out_channels:
                block["has_residual"] = True
            else:
                block["has_residual"] = False
                
            self.cnn_blocks.append(block)
            in_channels = out_channels
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=cnn_channels[-1],
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=self.lstm_dropout_rate if lstm_num_layers > 1 else 0
        )
        
        # Attention mechanism
        lstm_output_dim = self.num_directions * lstm_hidden_dim
        if attention_type == "basic":
            self.attention = nn.Linear(lstm_output_dim, 1)
        elif attention_type == "scaled_dot":
            self.query = nn.Linear(lstm_output_dim, self.attention_dim)
            self.key = nn.Linear(lstm_output_dim, self.attention_dim)
            self.value = nn.Linear(lstm_output_dim, lstm_output_dim)
        elif attention_type == "multi_head":
            self.mha = nn.MultiheadAttention(
                embed_dim=lstm_output_dim,
                num_heads=multi_head_num,
                batch_first=True
            )
            self.attention_ln = nn.LayerNorm(lstm_output_dim)
        else:
            # Caso por defecto para evitar errores
            self.attention = nn.Linear(lstm_output_dim, 1)
            print(f"ADVERTENCIA: Tipo de atención '{attention_type}' no reconocido. Usando 'basic'.")
        
        # Classifier
        classifier_layers = []
        in_dim = lstm_output_dim
        
        for hidden_dim in classifier_hidden_dims:
            classifier_layers.append(nn.Linear(in_dim, hidden_dim))
            classifier_layers.append(self.activation)
            classifier_layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
            
        classifier_layers.append(nn.Linear(in_dim, num_classes))
        self.classifier = nn.Sequential(*classifier_layers)
        
        # Initialize weights
        self._initialize_weights()


    def _get_activation(self):
        if self.activation_type == "relu":
            return nn.ReLU()
        elif self.activation_type == "leaky_relu":
            return nn.LeakyReLU(0.1)
        elif self.activation_type == "elu":
            return nn.ELU()
        elif self.activation_type == "gelu":
            return nn.GELU()
        else:
            return nn.ReLU()
            
    def _initialize_weights(self):
        if self.weight_init == "xavier":
            for m in self.modules():
                if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        elif self.weight_init == "kaiming":
            for m in self.modules():
                if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        
    def activations_hook(self, grad):
        self.gradients = grad
        
    def reset_activation_storage(self):
        self.cnn_activations = []
        self.lstm_activations = None
        self.attention_weights = None
        self.gradients = None
        self.last_cnn_output = None
        
    def forward(self, x, return_attention=False, track_gradients=False):
        """
        Forward pass del modelo
        
        Args:
            x: Input tensor de forma (batch, channels, seq_length)
            return_attention: Si True, devuelve también weights de atención
            track_gradients: Si True, registra hook para gradientes (para explicabilidad)
            
        Returns:
            Logits de clasificación y opcionalmente pesos de atención
        """
        self.reset_activation_storage()
        self.input = x
        
        # Guardar dimensiones originales para debugging
        batch_size, channels, orig_seq_len = x.shape
        
        # Verificar entrada coherente con la configuración del modelo
        if channels != self.input_channels:
            print(f"ADVERTENCIA: Número de canales de entrada ({channels}) "
                f"difiere del configurado ({self.input_channels})")
        
        # Almacenar dimensiones después de cada bloque para debugging
        dims_after_each_block = []
        
        # Process through CNN blocks
        for i, block in enumerate(self.cnn_blocks):
            # Guardar entrada para posible conexión residual
            residual = x if block["has_residual"] else None
            
            # Forward pass por operaciones del bloque
            x = block["conv"](x)
            
            if "bn" in block:
                x = block["bn"](x)
                
            if "ln" in block:
                # Transponemos correctamente para aplicar layer norm
                x_transposed = x.transpose(1, 2)  # (batch, seq, channels)
                x_normalized = block["ln"](x_transposed)
                x = x_normalized.transpose(1, 2)  # Volver a (batch, channels, seq)
                
            x = self.activation(x)
            
            # Aplicar conexión residual si está disponible para este bloque
            if residual is not None:
                x = x + residual
                
            # Aplicar pooling (con la lógica corregida)
            if not (i == len(self.cnn_blocks) - 1 and self.residual_connections):
                x = block["pool"](x)
                
            x = block["dropout"](x)
            self.cnn_activations.append(x.detach())
            
            # Guardar dimensiones actuales
            dims_after_each_block.append(tuple(x.shape))
        
        cnn_output = x
        
        # Verificar dimensiones finales CNN
        final_seq_len = x.shape[2]
        if self.output_seq_length is not None and final_seq_len != self.output_seq_length:
            print(f"ADVERTENCIA: Longitud secuencia después de CNN ({final_seq_len}) "
                f"difiere de la esperada ({self.output_seq_length})")
        
        if track_gradients and cnn_output.requires_grad:
            cnn_output.register_hook(self.activations_hook)
            
        self.last_cnn_output = cnn_output
        
        # Reshape for LSTM: (batch, channels, seq) -> (batch, seq, channels)
        x = cnn_output.permute(0, 2, 1)
        
        # LSTM
        lstm_out, _ = self.lstm(x)
        self.lstm_activations = lstm_out.detach()
        
        # Declarar vectores que usaremos para todos los tipos de atención
        attention_weights = None
        context_vector = None
        
        # Apply attention mechanism según tipo configurado
        if self.attention_type == "basic":
            attention_scores = self.attention(lstm_out).squeeze(-1)
            attention_weights = F.softmax(attention_scores, dim=1)
            context_vector = torch.bmm(attention_weights.unsqueeze(1), lstm_out).squeeze(1)
            
        elif self.attention_type == "scaled_dot":
            Q = self.query(lstm_out)
            K = self.key(lstm_out)
            V = self.value(lstm_out)
            
            scores = torch.bmm(Q, K.transpose(1, 2)) / np.sqrt(self.attention_dim)
            attention_weights = F.softmax(scores, dim=-1)
            context_vector = torch.bmm(attention_weights, V).mean(dim=1)
            
        elif self.attention_type == "multi_head":
            # MultiheadAttention devuelve (attn_output, attn_output_weights)
            attn_output, attn_output_weights = self.mha(lstm_out, lstm_out, lstm_out)
            attention_weights = attn_output_weights
            
            if self.layer_normalization:
                attn_output = self.attention_ln(attn_output + lstm_out)
            
            context_vector = attn_output.mean(dim=1)
        else:
            # Caso por defecto: atención uniforme
            attention_weights = torch.ones(lstm_out.shape[0], lstm_out.shape[1]).to(lstm_out.device)
            attention_weights = attention_weights / lstm_out.shape[1]  # Normalizar
            context_vector = lstm_out.mean(dim=1)
        
        # Verificar que attention_weights existe
        if attention_weights is None:
            attention_weights = torch.ones(lstm_out.shape[0], lstm_out.shape[1]).to(lstm_out.device)
            attention_weights = attention_weights / lstm_out.shape[1]  # Normalizar
        
        # Guardar pesos de atención para visualización/interpretación
        self.attention_weights = attention_weights.detach()
        
        # Verificar dimensiones del vector de contexto
        if context_vector is None:
            context_vector = lstm_out.mean(dim=1)
        
        # Asegurar dimensionalidad correcta: (batch_size, features)
        if len(context_vector.shape) > 2:
            print(f"ADVERTENCIA: Vector contexto tiene forma inesperada {context_vector.shape}. "
                f"Aplicando mean en dim 1.")
            context_vector = context_vector.mean(dim=1)
        elif len(context_vector.shape) == 1:
            context_vector = context_vector.unsqueeze(0)
        
        # Verificación final
        if len(context_vector.shape) != 2:
            print(f"ERROR: Vector contexto debe ser 2D pero es {context_vector.shape}")
            # Intentar corregir
            if len(context_vector.shape) > 2:
                context_vector = context_vector.reshape(batch_size, -1)
        
        # Classification
        out = self.classifier(context_vector)
        
        if return_attention:
            return out, attention_weights
        return out
    
    def get_config(self):
        """Returns the current configuration as a dictionary"""
        return {
            "input_channels": self.input_channels,
            "seq_length": self.seq_length,
            "num_classes": self.num_classes,
            "cnn_channels": self.cnn_channels,
            "kernel_sizes": self.kernel_sizes,
            "pool_type": self.pool_type,
            "pool_sizes": self.pool_sizes,
            "use_batch_norm": self.use_batch_norm,
            "activation": self.activation_type,
            "dropout": self.dropout_rate,
            "cnn_dropout": self.cnn_dropout_rate,
            "lstm_hidden_dim": self.lstm_hidden_dim,
            "lstm_num_layers": self.lstm_num_layers,
            "bidirectional": self.bidirectional,
            "lstm_dropout": self.lstm_dropout_rate,
            "classifier_hidden_dims": self.classifier_hidden_dims,
            "attention_dim": self.attention_dim,
            "attention_type": self.attention_type,
            "multi_head_num": self.multi_head_num,
            "residual_connections": self.residual_connections,
            "layer_normalization": self.layer_normalization,
            "weight_init": self.weight_init
        }
    
    def count_parameters(self):
        """Count and return the number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_intermediate_outputs(self, x):
        """Get all intermediate activations for a given input"""
        _ = self.forward(x, track_gradients=True)
        return {
            "cnn_activations": self.cnn_activations,
            "lstm_activations": self.lstm_activations,
            "attention_weights": self.attention_weights
        }
    
    def visualize_attention(self, x, return_fig=False):
        """Visualize attention weights for a given input"""
        try:
            import matplotlib.pyplot as plt
            
            _, attention_weights = self.forward(x, return_attention=True)
            
            if attention_weights is None:
                print("No attention weights available")
                return None
                
            batch_size = attention_weights.size(0)
            seq_len = attention_weights.size(1)
            
            fig, axes = plt.subplots(batch_size, 1, figsize=(10, 2*batch_size))
            if batch_size == 1:
                axes = [axes]
                
            for i, ax in enumerate(axes):
                weights = attention_weights[i].cpu().detach().numpy()
                ax.bar(range(seq_len), weights)
                ax.set_title(f"Sample {i+1}")
                ax.set_xlabel("Sequence position")
                ax.set_ylabel("Attention weight")
                
            plt.tight_layout()
            
            if return_fig:
                return fig
            plt.show()
            return None
        except ImportError:
            print("matplotlib is required for visualization")
            return None

    def interpret(self, x, class_idx=None, methods=None):
        """
        Enhanced interpretation method with multiple explainability techniques
        
        Args:
            x: Input data tensor
            class_idx: Target class indices to explain (defaults to predicted class)
            methods: List of methods to use, options: ['gradcam', 'integrated_gradients', 
                     'occlusion', 'shap', 'feature_ablation', 'all']
                      
        Returns:
            Dictionary with various interpretability outputs
        """
        try:
            # Try to import Captum components
            from captum.attr import IntegratedGradients, Occlusion, GradientShap, LayerGradCam
        except ImportError:
            raise ImportError("This method requires the 'captum' package. Install with: pip install captum")
            
        if methods is None:
            methods = ['gradcam', 'attention']  # Default methods
        if 'all' in methods:
            methods = ['gradcam', 'integrated_gradients', 'occlusion', 'shap', 
                      'feature_ablation', 'attention', 'layer_importance']
        
        # Store original training state
        was_training = self.training
        lstm_was_training = self.lstm.training

        # Set model to evaluation mode for interpretability
        self.eval()
        self.lstm.train()  # needed for CuDNN backward compatibility
        
        # Base prediction
        x.requires_grad_()
        self.input = x  # Store input for interpretability methods
        
        logits, attention = self.forward(x, return_attention=True, track_gradients=True)
        pred = torch.softmax(logits, dim=1)

        if class_idx is None:
            class_idx = pred.argmax(dim=1)
        
        # Initialize results dictionary
        results = {
            'prediction': pred.detach(),
            'class_idx': class_idx,
            'attention_weights': self.attention_weights,
        }
        
        # Apply selected interpretability methods
        if 'gradcam' in methods:
            for i in range(x.shape[0]):
                pred[i, class_idx[i]].backward(retain_graph=True if i < x.shape[0]-1 else False)
            
            results['feature_importance'] = self.get_feature_importance()
            results['temporal_channel_importance'] = self.get_temporal_channel_importance()
            results['channel_importance'] = self.get_channel_importance()
            results['cnn_activations'] = self.cnn_activations
            
        # Integrated Gradients
        if 'integrated_gradients' in methods:
            ig = IntegratedGradients(self.forward_wrapper)
            results['integrated_gradients'] = self._compute_integrated_gradients(
                ig, x, class_idx)
        
        # Occlusion analysis
        if 'occlusion' in methods:
            occlusion = Occlusion(self.forward_wrapper)
            results['occlusion'] = self._compute_occlusion(occlusion, x, class_idx)
        
        # SHAP (GradientSHAP implementation)
        if 'shap' in methods:
            gradient_shap = GradientShap(self.forward_wrapper)
            results['gradient_shap'] = self._compute_gradient_shap(gradient_shap, x, class_idx)
        
        # Feature ablation (sensitivity analysis)
        if 'feature_ablation' in methods:
            results['feature_ablation'] = self._feature_ablation_analysis(x, class_idx)
        
        # Layer importance analysis
        if 'layer_importance' in methods:
            results['layer_importance'] = self._compute_layer_importance(x, class_idx)
            
        # Restore original training states
        self.train(was_training)
        self.lstm.train(lstm_was_training)
        
        # Clean up to avoid memory issues
        self.input = None
        torch.cuda.empty_cache()
        
        return results

    def forward_wrapper(self, x):
        """Wrapper for Captum compatibility"""
        return self.forward(x)
    
    def get_feature_importance(self):
        """
        Grad-CAM temporal over the output of the last CNN block.
        Returns tensor (batch, time)
        """
        if self.gradients is None or self.last_cnn_output is None:
            return None

        pooled_gradients = torch.mean(self.gradients, dim=[0, 2])  # (channels,)
        cam = self.last_cnn_output.clone()

        for i in range(cam.shape[1]):
            cam[:, i, :] *= pooled_gradients[i]

        heatmap = torch.mean(cam, dim=1).detach()  # (batch, time)
        
        # Apply ReLU to highlight only positive influences
        heatmap = F.relu(heatmap)
        
        # Normalize heatmap for better visualization
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
            
        return heatmap

    def get_channel_importance(self):
        """
        Channel importance: (batch, channels)
        """
        if self.input is None or self.input.grad is None:
            raise ValueError("Input gradients not available. Call interpret() first.")
        return self.input.grad.abs().mean(dim=2).detach()

    def get_temporal_channel_importance(self):
        """
        Temporal-channel importance: (batch, channels, time)
        """
        if self.input is None or self.input.grad is None:
            raise ValueError("Input gradients not available. Call interpret() first.")
        return self.input.grad.abs().detach()
    
    def _compute_integrated_gradients(self, ig, x, class_idx):
        """Compute integrated gradients attribution"""
        batch_size = x.shape[0]
        attributions = []
        
        for i in range(batch_size):
            baseline = torch.zeros_like(x[i:i+1])
            attr = ig.attribute(
                x[i:i+1], baseline, target=class_idx[i].item(), n_steps=50
            )
            attributions.append(attr)
            
        return torch.cat(attributions).detach()
    
    def _compute_occlusion(self, occlusion_algo, x, class_idx):
        """Compute occlusion-based feature attribution"""
        batch_size = x.shape[0]
        attributions = []
        
        # Define sliding window parameters for temporal data
        window_size = min(5, x.shape[2] // 4)  # Adapt window size to input length
        
        for i in range(batch_size):
            attr = occlusion_algo.attribute(
                x[i:i+1], 
                sliding_window_shapes=(1, window_size),
                target=class_idx[i].item(),
                strides=(1, max(1, window_size // 2))
            )
            attributions.append(attr)
            
        return torch.cat(attributions).detach()
    
    def _compute_gradient_shap(self, shap_algo, x, class_idx):
        """Compute GradientSHAP attributions"""
        batch_size = x.shape[0]
        attributions = []
        
        for i in range(batch_size):
            # Create random baselines (typically 10-50 for good estimates)
            baselines = torch.randn(10, *x[i:i+1].shape[1:]) * 0.001
            
            # Ensure baselines device matches input
            baselines = baselines.to(x.device)
            
            attr = shap_algo.attribute(
                x[i:i+1], baselines=baselines, target=class_idx[i].item()
            )
            attributions.append(attr)
            
        return torch.cat(attributions).detach()
    
    def _feature_ablation_analysis(self, x, class_idx):
        """Analyze model by systematically ablating input features"""
        batch_size = x.shape[0]
        results = []
        
        for i in range(batch_size):
            # Store original prediction
            with torch.no_grad():
                orig_output = self.forward(x[i:i+1])
                orig_prob = torch.softmax(orig_output, dim=1)[0, class_idx[i]].item()
            
            # Test ablation of each channel
            channel_importance = []
            for c in range(self.input_channels):
                # Create ablated input (zero out one channel)
                ablated_input = x[i:i+1].clone()
                ablated_input[:, c, :] = 0
                
                # Get prediction on ablated input
                with torch.no_grad():
                    ablated_output = self.forward(ablated_input)
                    ablated_prob = torch.softmax(ablated_output, dim=1)[0, class_idx[i]].item()
                
                # Impact is reduction in probability
                channel_impact = orig_prob - ablated_prob
                channel_importance.append(channel_impact)
            
            results.append(torch.tensor(channel_importance))
            
        return torch.stack(results)
    
    def _compute_layer_importance(self, x, class_idx):
        """Compute importance of each layer using Layer GradCAM"""
        try:
            from captum.attr import LayerGradCam
        except ImportError:
            raise ImportError("This method requires the 'captum' package.")
            
        batch_size = x.shape[0]
        layer_importance = {}
        
        # Define layers to analyze - adapted for our new ModuleList structure
        layers = {}
        for i, block in enumerate(self.cnn_blocks):
            layers[f'conv{i+1}'] = block['conv']
        
        for layer_name, layer in layers.items():
            layer_gradcam = LayerGradCam(self.forward_wrapper, layer)
            layer_attrs = []
            
            for i in range(batch_size):
                attr = layer_gradcam.attribute(
                    x[i:i+1], target=class_idx[i].item()
                )
                # Process attribution to create a single importance score per sample
                pooled_attr = torch.mean(attr, dim=1)

                layer_attrs.append(pooled_attr)
                
            layer_importance[layer_name] = torch.cat(layer_attrs).detach()
            
        return layer_importance
    
    def visualize_attributions(self, sample_idx, interpretations, time_axis=None, 
                               channel_names=None, class_names=None):
        """
        Visualize the various interpretation results
        
        Args:
            sample_idx: Index of the sample to visualize
            interpretations: Dictionary returned by interpret() method
            time_axis: Optional array/list with time points for x-axis
            channel_names: Optional list of channel names
            class_names: Optional list of class names
        """
        try:
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            raise ImportError("This method requires matplotlib and numpy for visualization")
            
        if not channel_names:
            channel_names = [f'Channel {i}' for i in range(self.input_channels)]
        
        if not class_names:
            class_idx = interpretations['class_idx'][sample_idx].item()
            class_name = f'Class {class_idx}'
        else:
            class_idx = interpretations['class_idx'][sample_idx].item()
            class_name = class_names[class_idx]
            
        # Set up figure
        plt.figure(figsize=(15, 12))
        
        # Original input visualization (top row, first column)
        plt.subplot(3, 3, 1)
        if self.input is not None:
            input_data = self.input[sample_idx].cpu().detach().numpy()
            if time_axis is not None:
                for i in range(input_data.shape[0]):
                    plt.plot(time_axis, input_data[i], label=channel_names[i])
            else:
                for i in range(input_data.shape[0]):
                    plt.plot(input_data[i], label=channel_names[i])
            plt.legend(loc='best')
            plt.title('Input Signal')
            plt.xlabel('Time')
            plt.ylabel('Value')
            
        # GradCAM feature importance (top row, second column)
        if 'feature_importance' in interpretations and interpretations['feature_importance'] is not None:
            plt.subplot(3, 3, 2)
            heatmap = interpretations['feature_importance'][sample_idx].cpu().numpy()
            if time_axis is not None:
                plt.plot(time_axis, heatmap)
            else:
                plt.plot(heatmap)
            plt.title('GradCAM Feature Importance')
            plt.xlabel('Time')
            plt.ylabel('Importance')
            
        # Attention weights (top row, third column)
        if 'attention_weights' in interpretations and interpretations['attention_weights'] is not None:
            plt.subplot(3, 3, 3)
            attention = interpretations['attention_weights'][sample_idx].cpu().numpy()
            
            if time_axis is not None:
                # Need to match attention time axis to input time axis
                # (account for pooling in the network)
                x_points = np.linspace(time_axis[0], time_axis[-1], len(attention))
                plt.plot(x_points, attention)
            else:
                plt.plot(attention)
            plt.title('Attention Weights')
            plt.xlabel('Time')
            plt.ylabel('Attention')
            
        # Channel importance (middle row, first column)
        if 'channel_importance' in interpretations and interpretations['channel_importance'] is not None:
            plt.subplot(3, 3, 4)
            ch_importance = interpretations['channel_importance'][sample_idx].cpu().numpy()
            plt.bar(channel_names, ch_importance)
            plt.title('Channel Importance')
            plt.ylabel('Importance')
            plt.xticks(rotation=45)
            
        # Integrated Gradients (middle row, second column)
        if 'integrated_gradients' in interpretations:
            plt.subplot(3, 3, 5)
            ig_attr = interpretations['integrated_gradients'][sample_idx].cpu().numpy()
            ig_attr_mean = np.mean(ig_attr, axis=0)  # Average across channels for visualization
            
            if time_axis is not None:
                plt.plot(time_axis, ig_attr_mean)
            else:
                plt.plot(ig_attr_mean)
            plt.title('Integrated Gradients')
            plt.xlabel('Time')
            plt.ylabel('Attribution')
            
        # Feature Ablation (middle row, third column)
        if 'feature_ablation' in interpretations:
            plt.subplot(3, 3, 6)
            ablation_scores = interpretations['feature_ablation'][sample_idx].cpu().numpy()
            plt.bar(channel_names, ablation_scores)
            plt.title('Feature Ablation Impact')
            plt.ylabel('Probability Change')
            plt.xticks(rotation=45)
            
        # SHAP values (bottom row, first column)
        if 'gradient_shap' in interpretations:
            plt.subplot(3, 3, 7)
            shap_attr = interpretations['gradient_shap'][sample_idx].cpu().numpy()
            # Visualize average SHAP value over time
            shap_avg = np.mean(shap_attr, axis=0)
            
            if time_axis is not None:
                plt.plot(time_axis, shap_avg)
            else:
                plt.plot(shap_avg)
            plt.title('GradientSHAP Values')
            plt.xlabel('Time')
            plt.ylabel('SHAP Value')
            
        # Occlusion analysis (bottom row, second column)
        if 'occlusion' in interpretations:
            plt.subplot(3, 3, 8)
            occlusion_attr = interpretations['occlusion'][sample_idx].cpu().numpy()
            occlusion_avg = np.mean(occlusion_attr, axis=0)
            
            if time_axis is not None:
                plt.plot(time_axis, occlusion_avg)
            else:
                plt.plot(occlusion_avg)
            plt.title('Occlusion Analysis')
            plt.xlabel('Time')
            plt.ylabel('Attribution')
        
        # Prediction summary (bottom row, third column)
        plt.subplot(3, 3, 9)
        pred_probs = interpretations['prediction'][sample_idx].cpu().numpy()
        classes = list(range(len(pred_probs)))
        if class_names:
            classes = class_names
        plt.bar(classes, pred_probs)
        plt.title(f'Prediction: {class_name}')
        plt.ylabel('Probability')
        plt.ylim([0, 1])
        
        plt.tight_layout()
        return plt.gcf()


# Example of creating a model with custom hyperparameters
def create_model_with_config(**kwargs):
    """Helper function to create a model with specified config"""
    config = {
        "input_channels": 3,
        "num_classes": 3,
        "cnn_channels": (16, 32, 64),
        "kernel_sizes": (5, 3, 3),
        "pool_type": "max",
        "dropout": 0.1,
        "lstm_hidden_dim": 32,
        "lstm_num_layers": 1,
        "bidirectional": True,
        "classifier_hidden_dims": [32],
        "attention_type": "basic",
        "residual_connections": False,
        "layer_normalization": False
    }
    
    # Update config with provided kwargs
    config.update(kwargs)
    
    return CNN_LSTM_Classifier_Tunable(config=config)

