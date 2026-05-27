"""
preprocessing/normalizer.py — Signal Normalization Module
==========================================================
Normalizes vital signs so all signals are on the same scale.
This is critical for ML models that are sensitive to feature magnitude.

Three strategies:
  - Z-Score:  mean=0, std=1  (good for Gaussian-distributed signals)
  - MinMax:   scales to [0,1] (good for bounded signals like SpO2)
  - Robust:   uses median/IQR (best for signals with outliers/anomalies)

BEGINNER NOTE:
    Without normalization, a model might pay 100x more attention to
    systolic_bp (range 90–200) than temperature (range 36–40),
    just because of the scale difference. Normalization fixes this.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from typing import Dict, Optional, Tuple
import pickle
from pathlib import Path

from config import VITAL_SIGNS, PREPROCESSING


class SignalNormalizer:
    """
    Normalizes physiological signals using a chosen strategy.

    Usage:
        norm = SignalNormalizer(strategy="robust")
        norm_df = norm.fit_transform(clean_df)
        original_df = norm.inverse_transform(norm_df)   # undo normalization
        norm.save("outputs/models/scaler.pkl")           # save for reuse
    """

    def __init__(self, strategy: Optional[str] = None):
        """
        Args:
            strategy: "zscore" | "minmax" | "robust"
                      Defaults to config value.
        """
        self.strategy = strategy or PREPROCESSING["normalization"]
        self.scalers: Dict[str, object] = {}   # one scaler per vital
        self.vital_cols: list = []
        self._fitted = False

    def fit(self, df: pd.DataFrame) -> "SignalNormalizer":
        """
        Fit scalers on the given data (computes mean/std or min/max etc.)
        
        Args:
            df: DataFrame with vital sign columns.
        Returns:
            self (for chaining: norm.fit(df).transform(df))
        """
        self.vital_cols = [c for c in df.columns if c in VITAL_SIGNS]

        for col in self.vital_cols:
            values = df[col].dropna().values.reshape(-1, 1)
            scaler = self._get_scaler()
            scaler.fit(values)
            self.scalers[col] = scaler

        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply fitted normalization to the DataFrame.
        
        Returns:
            DataFrame with same structure but normalized vital columns.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before transform().")

        out = df.copy()
        for col in self.vital_cols:
            if col in out.columns:
                vals = out[col].values.reshape(-1, 1)
                out[col] = self.scalers[col].transform(vals).flatten()
        return out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(df).transform(df)

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Undo normalization — convert back to original units.
        Useful for displaying results in clinically meaningful scales.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before inverse_transform().")

        out = df.copy()
        for col in self.vital_cols:
            if col in out.columns:
                vals = out[col].values.reshape(-1, 1)
                out[col] = self.scalers[col].inverse_transform(vals).flatten()
        return out

    def save(self, path: str):
        """Save fitted scalers to disk for reuse."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"scalers": self.scalers,
                         "strategy": self.strategy,
                         "vital_cols": self.vital_cols}, f)
        print(f"   💾 Normalizer saved to {path}")

    @classmethod
    def load(cls, path: str) -> "SignalNormalizer":
        """Load a previously saved normalizer."""
        with open(path, "rb") as f:
            state = pickle.load(f)
        obj = cls(strategy=state["strategy"])
        obj.scalers = state["scalers"]
        obj.vital_cols = state["vital_cols"]
        obj._fitted = True
        return obj

    def _get_scaler(self):
        """Return a fresh scikit-learn scaler based on the chosen strategy."""
        if self.strategy == "zscore":
            return StandardScaler()
        elif self.strategy == "minmax":
            return MinMaxScaler(feature_range=(0, 1))
        elif self.strategy == "robust":
            # RobustScaler uses median and IQR — robust to anomalies
            return RobustScaler(quantile_range=(10.0, 90.0))
        else:
            raise ValueError(f"Unknown normalization strategy: {self.strategy}. "
                             f"Choose from 'zscore', 'minmax', 'robust'.")
