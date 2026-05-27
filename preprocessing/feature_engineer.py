"""
preprocessing/feature_engineer.py — Feature Engineering Module
===============================================================
Derives additional informative features from raw vital signs:
  - Rolling statistics (mean, std, min, max over sliding windows)
  - Rate-of-change (first derivative — how fast is the signal changing?)
  - Pulse Pressure (systolic_bp - diastolic_bp)
  - Shock Index (heart_rate / systolic_bp)
  - SpO2-HR product (surrogate for oxygen delivery)
  - Hour-of-day (captures circadian effects)

BEGINNER NOTE:
    These derived features help anomaly detectors and attribution methods
    capture PATTERNS, not just raw values. For example:
    - A heart rate of 90 is normal.
    - A heart rate that INCREASED by 30 bpm in the last 5 minutes is alarming.
    The rate-of-change feature captures this.
"""

import numpy as np
import pandas as pd
from typing import List, Optional

from config import VITAL_SIGNS, PREPROCESSING


class FeatureEngineer:
    """
    Derives clinical and statistical features from physiological signals.

    Usage:
        fe = FeatureEngineer(window=10)
        enriched_df = fe.transform(clean_df)
        print(fe.feature_names_)
    """

    def __init__(self, window: Optional[int] = None):
        """
        Args:
            window: Rolling window size in samples. Defaults to config value.
        """
        self.window = window or PREPROCESSING["rolling_window"]
        self.feature_names_: List[str] = []
        self._original_vital_cols: List[str] = []

    def transform(self, df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
        """
        Add engineered features to the DataFrame.

        Args:
            df: Cleaned DataFrame with vital sign columns.
            verbose: Print feature summary.

        Returns:
            Enriched DataFrame with additional feature columns.
        """
        print("\n⚙️  Step 2b: Feature Engineering")
        out = df.copy()
        self._original_vital_cols = [c for c in df.columns if c in VITAL_SIGNS]
        new_features = []

        # ── Rolling statistics ──────────────────────────────────────
        for col in self._original_vital_cols:
            series = out[col]

            # Rolling mean (smooth trend)
            feat = f"{col}_roll_mean"
            out[feat] = series.rolling(self.window, min_periods=1).mean()
            new_features.append(feat)

            # Rolling std (local volatility — high std → unstable signal)
            feat = f"{col}_roll_std"
            out[feat] = series.rolling(self.window, min_periods=1).std().fillna(0)
            new_features.append(feat)

            # Rolling min/max (range within window)
            feat = f"{col}_roll_min"
            out[feat] = series.rolling(self.window, min_periods=1).min()
            new_features.append(feat)

            feat = f"{col}_roll_max"
            out[feat] = series.rolling(self.window, min_periods=1).max()
            new_features.append(feat)

            # Rate of change (first difference, normalized by rolling std)
            diff = series.diff().fillna(0)
            roll_std = series.rolling(self.window, min_periods=1).std().replace(0, 1)
            feat = f"{col}_roc"
            out[feat] = diff / roll_std
            new_features.append(feat)

        # ── Clinical composite features ─────────────────────────────
        available = set(out.columns)

        # Pulse Pressure = Systolic BP - Diastolic BP
        # Normal: 40 mmHg. Narrow (<25) or wide (>65) = pathology
        if "systolic_bp" in available and "diastolic_bp" in available:
            out["pulse_pressure"] = out["systolic_bp"] - out["diastolic_bp"]
            new_features.append("pulse_pressure")

        # Shock Index = HR / Systolic BP
        # Normal: <0.7. >1.0 = potential shock state
        if "heart_rate" in available and "systolic_bp" in available:
            sbp = out["systolic_bp"].replace(0, np.nan)
            out["shock_index"] = out["heart_rate"] / sbp
            out["shock_index"] = out["shock_index"].fillna(1.0)
            new_features.append("shock_index")

        # Respiratory-SpO2 burden (RR * (100 - SpO2))
        # High when patient is both breathing fast AND desaturating
        if "respiratory_rate" in available and "spo2" in available:
            out["resp_spo2_burden"] = (
                out["respiratory_rate"] * (100 - out["spo2"])
            )
            new_features.append("resp_spo2_burden")

        # Mean Arterial Pressure = DBP + (SBP - DBP)/3
        if "systolic_bp" in available and "diastolic_bp" in available:
            out["map"] = (
                out["diastolic_bp"] + (out["systolic_bp"] - out["diastolic_bp"]) / 3
            )
            new_features.append("map")

        # ── Time features ────────────────────────────────────────────
        if hasattr(out.index, "hour"):
            out["hour_of_day"] = out.index.hour
            out["hour_sin"] = np.sin(2 * np.pi * out.index.hour / 24)
            out["hour_cos"] = np.cos(2 * np.pi * out.index.hour / 24)
            new_features.extend(["hour_of_day", "hour_sin", "hour_cos"])

        self.feature_names_ = new_features

        if verbose:
            print(f"   ✅ Added {len(new_features)} engineered features")
            print(f"   📋 Clinical: pulse_pressure, shock_index, resp_spo2_burden, MAP")
            print(f"   📊 Rolling stats: mean, std, min, max, rate-of-change per vital")

        return out

    def get_feature_columns(self, include_original: bool = True) -> List[str]:
        """Return list of all feature column names."""
        if include_original:
            return self._original_vital_cols + self.feature_names_
        return self.feature_names_
