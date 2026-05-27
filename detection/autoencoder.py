"""
detection/autoencoder.py — LSTM Autoencoder Anomaly Detector
=============================================================
Step 4c: Deep learning detector using a sequence-to-sequence LSTM Autoencoder.

HOW IT WORKS:
    1. TRAIN: The autoencoder learns to COMPRESS and then RECONSTRUCT normal sequences.
              It is trained only on (or mostly on) normal data.
    2. DETECT: At inference, anomalous sequences produce HIGH reconstruction error
               because the model has never learned to reconstruct anomalous patterns.

ARCHITECTURE:
    Input (seq_len × n_features)
        ↓  LSTM Encoder (compresses to latent vector)
    Latent Vector (hidden_dim)
        ↓  Repeat Vector (expand back to sequence length)
        ↓  LSTM Decoder (reconstructs the sequence)
    Reconstructed Output (seq_len × n_features)

    Anomaly Score = Mean Squared Error (MSE) between input and reconstruction

BEGINNER NOTE:
    Think of it like teaching someone to memorize "normal" music.
    If you play them an unusual note sequence, they'll make more mistakes
    trying to repeat it back. Those "mistakes" (reconstruction error) = anomaly.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional, Dict, Tuple
from pathlib import Path
import warnings

from config import VITAL_SIGNS, DETECTION


# ─────────────────────────────────────────────────────────────────────────────
# PYTORCH MODEL DEFINITION
# ─────────────────────────────────────────────────────────────────────────────

class LSTMEncoder(nn.Module):
    """Encodes a sequence into a fixed-size latent vector."""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, latent_dim: int):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        _, (hidden, _) = self.lstm(x)
        # Use last layer's hidden state
        latent = self.fc(hidden[-1])
        return latent


class LSTMDecoder(nn.Module):
    """Decodes a latent vector back into a sequence."""

    def __init__(self, latent_dim: int, hidden_dim: int, num_layers: int,
                 output_dim: int, seq_len: int):
        super().__init__()
        self.seq_len = seq_len
        self.fc = nn.Linear(latent_dim, hidden_dim)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0,
        )
        self.output_fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, latent):
        # Expand latent to (batch, seq_len, hidden_dim)
        x = self.fc(latent).unsqueeze(1).repeat(1, self.seq_len, 1)
        lstm_out, _ = self.lstm(x)
        return self.output_fc(lstm_out)


class LSTMAutoencoder(nn.Module):
    """Full LSTM Autoencoder: Encoder → latent → Decoder."""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int,
                 latent_dim: int, seq_len: int):
        super().__init__()
        self.encoder = LSTMEncoder(input_dim, hidden_dim, num_layers, latent_dim)
        self.decoder = LSTMDecoder(latent_dim, hidden_dim, num_layers, input_dim, seq_len)

    def forward(self, x):
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction


# ─────────────────────────────────────────────────────────────────────────────
# DETECTOR WRAPPER
# ─────────────────────────────────────────────────────────────────────────────

class LSTMAutoencoderDetector:
    """
    Anomaly detector wrapping the LSTM Autoencoder with training/detection logic.

    Usage:
        det = LSTMAutoencoderDetector()
        det.fit(normal_df)                 # Train on clean data
        results = det.detect(full_df)      # Detect on full data
        print(results["anomaly_flags"])
    """

    def __init__(self):
        cfg = DETECTION["lstm_autoencoder"]
        self.seq_len = cfg["sequence_length"]
        self.hidden_dim = cfg["hidden_dim"]
        self.num_layers = cfg["num_layers"]
        self.latent_dim = cfg["latent_dim"]
        self.epochs = cfg["epochs"]
        self.batch_size = cfg["batch_size"]
        self.lr = cfg["learning_rate"]
        self.threshold_pct = cfg["reconstruction_threshold_percentile"]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[LSTMAutoencoder] = None
        self.threshold: Optional[float] = None
        self._feature_cols: list = []
        self._input_dim: int = 0

    def fit(self, df: pd.DataFrame, feature_cols: Optional[list] = None,
            verbose: bool = True) -> "LSTMAutoencoderDetector":
        """
        Train the autoencoder on (ideally normal) data.

        Args:
            df: DataFrame with vital sign / feature columns.
            feature_cols: Columns to use. Defaults to vital signs.
            verbose: Print training progress.
        """
        print(f"\n🧠 Training LSTM Autoencoder (device: {self.device})")

        self._feature_cols = feature_cols or [c for c in df.columns if c in VITAL_SIGNS]
        data = df[self._feature_cols].fillna(df[self._feature_cols].median()).values.astype(np.float32)
        self._input_dim = data.shape[1]

        # Build sliding window sequences
        X = self._make_sequences(data)   # (N, seq_len, features)
        dataset = TensorDataset(torch.FloatTensor(X))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Build model
        self.model = LSTMAutoencoder(
            input_dim=self._input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            latent_dim=self.latent_dim,
            seq_len=self.seq_len,
        ).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for (batch,) in loader:
                batch = batch.to(self.device)
                recon = self.model(batch)
                loss = criterion(recon, batch)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(loader)
            if verbose and (epoch + 1) % 10 == 0:
                print(f"   Epoch [{epoch+1:3d}/{self.epochs}]  Loss: {avg_loss:.6f}")

        # Compute reconstruction errors on training data → set threshold
        errors = self._reconstruction_errors(X)
        self.threshold = float(np.percentile(errors, self.threshold_pct))

        print(f"   ✅ Training complete. Anomaly threshold (p{self.threshold_pct}): {self.threshold:.6f}")
        return self

    def detect(self, df: pd.DataFrame, verbose: bool = True) -> Dict:
        """
        Detect anomalies by comparing reconstruction errors to threshold.

        Returns:
            Dict with anomaly_flags, anomaly_scores.
        """
        print("\n🔎 LSTM Autoencoder Detection")
        if self.model is None:
            raise RuntimeError("Call fit() before detect().")

        data = df[self._feature_cols].fillna(df[self._feature_cols].median()).values.astype(np.float32)
        X = self._make_sequences(data)
        errors = self._reconstruction_errors(X)

        # Pad errors to match original DataFrame length
        # (sliding windows reduce the effective length by seq_len - 1)
        pad = np.full(len(df) - len(errors), errors[0])
        errors_full = np.concatenate([pad, errors])

        anomaly_scores = pd.Series(errors_full, index=df.index, name="lstm_score")
        anomaly_flags = pd.Series(errors_full > self.threshold, index=df.index, name="lstm_anomaly")

        if verbose:
            n = anomaly_flags.sum()
            print(f"   Threshold: {self.threshold:.6f}")
            print(f"   Detected:  {n} anomalies ({100*n/len(df):.1f}%)")
            print("   ✅ LSTM detection complete.")

        return {"anomaly_flags": anomaly_flags, "anomaly_scores": anomaly_scores}

    def _make_sequences(self, data: np.ndarray) -> np.ndarray:
        """Create sliding window sequences from the data array."""
        seqs = []
        for i in range(len(data) - self.seq_len + 1):
            seqs.append(data[i: i + self.seq_len])
        return np.array(seqs)

    def _reconstruction_errors(self, X: np.ndarray) -> np.ndarray:
        """Compute per-sequence mean squared reconstruction error."""
        self.model.eval()
        errors = []
        with torch.no_grad():
            for i in range(0, len(X), self.batch_size):
                batch = torch.FloatTensor(X[i: i + self.batch_size]).to(self.device)
                recon = self.model(batch)
                mse = ((batch - recon) ** 2).mean(dim=(1, 2)).cpu().numpy()
                errors.extend(mse.tolist())
        return np.array(errors)

    def save(self, path: str):
        """Save model and threshold to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state": self.model.state_dict() if self.model else None,
            "threshold": self.threshold,
            "feature_cols": self._feature_cols,
            "input_dim": self._input_dim,
            "config": {
                "seq_len": self.seq_len, "hidden_dim": self.hidden_dim,
                "num_layers": self.num_layers, "latent_dim": self.latent_dim,
            },
        }, path)
        print(f"   💾 LSTM Autoencoder saved to {path}")
