"""
detection/statistical.py — Statistical Anomaly Detector
========================================================
Step 4a: Detect anomalies using classical statistics on STL residuals.

Two methods:
  1. ROLLING Z-SCORE:
     At each time point, compute how many standard deviations the residual
     is from the local rolling mean. Values beyond ±threshold are anomalies.

     Formula: z(t) = (x(t) - rolling_mean(t)) / rolling_std(t)

  2. IQR METHOD:
     Compute interquartile range. Flag values outside [Q1 - k*IQR, Q3 + k*IQR].
     More robust to heavy-tailed distributions than Z-score.

WHEN TO USE WHICH:
  - Z-score: Good when signal is approximately Gaussian (bell-shaped).
  - IQR: Better when signal has outliers that distort mean/std estimates.

BEGINNER NOTE:
    These methods are transparent and easy to explain to clinicians.
    "This heart rate value is 3.5 standard deviations above normal"
    is a statement clinicians understand.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple

from config import VITAL_SIGNS, DETECTION


class StatisticalDetector:
    """
    Detects anomalies in STL residuals using rolling Z-score and IQR methods.

    Usage:
        det = StatisticalDetector()
        results = det.detect(residuals_df)
        print(results["anomaly_flags"])    # Boolean DataFrame
        print(results["zscore_df"])        # Z-score values
    """

    def __init__(
        self,
        zscore_threshold: Optional[float] = None,
        iqr_multiplier: Optional[float] = None,
        window: int = 30,
    ):
        """
        Args:
            zscore_threshold: Std deviations beyond which → anomaly. Default 3.0.
            iqr_multiplier: IQR multiplier. Default 1.5 (Tukey's rule).
            window: Rolling window size for computing local statistics.
        """
        self.zscore_threshold = zscore_threshold or DETECTION["zscore_threshold"]
        self.iqr_multiplier = iqr_multiplier or DETECTION["iqr_multiplier"]
        self.window = window
        self._zscore_df: Optional[pd.DataFrame] = None
        self._iqr_anomalies: Optional[pd.DataFrame] = None

    def detect(self, residuals: pd.DataFrame, verbose: bool = True) -> Dict:
        """
        Run both Z-score and IQR detection on the residuals DataFrame.

        Args:
            residuals: DataFrame of STL residuals (output of STLDecomposer.get_residuals()).
            verbose: Print detection summary.

        Returns:
            Dict with keys:
              - 'zscore_df'       : rolling Z-scores per vital
              - 'zscore_anomalies': boolean flags from Z-score method
              - 'iqr_anomalies'   : boolean flags from IQR method
              - 'anomaly_flags'   : combined (OR) boolean flags
              - 'anomaly_scores'  : continuous score (max abs z-score)
        """
        print("\n🔍 Step 4a: Statistical Anomaly Detection")
        vital_cols = [c for c in residuals.columns if c in VITAL_SIGNS]

        zscore_df = self._rolling_zscore(residuals[vital_cols])
        iqr_df = self._iqr_detection(residuals[vital_cols])

        # Boolean anomaly flags
        zscore_anomalies = zscore_df.abs() > self.zscore_threshold
        iqr_anomalies = iqr_df

        # Combined: flag if EITHER method flags it
        combined = zscore_anomalies | iqr_anomalies

        # Continuous score: max absolute Z-score across all vitals
        anomaly_scores = zscore_df.abs().max(axis=1)

        self._zscore_df = zscore_df

        if verbose:
            n_total = len(residuals)
            n_anomaly = combined.any(axis=1).sum()
            print(f"   Z-score threshold: ±{self.zscore_threshold}")
            print(f"   IQR multiplier:    {self.iqr_multiplier}x")
            print(f"   Total anomalies:   {n_anomaly} / {n_total} "
                  f"({100 * n_anomaly / n_total:.1f}%)")

            print(f"\n   {'Vital':<22} {'Z-score flags':>15} {'IQR flags':>12}")
            print(f"   {'─'*22} {'─'*15} {'─'*12}")
            for col in vital_cols:
                z_count = zscore_anomalies[col].sum()
                i_count = iqr_anomalies[col].sum() if col in iqr_anomalies else 0
                print(f"   {col:<22} {z_count:>15} {i_count:>12}")
            print("   ✅ Statistical detection complete.")

        return {
            "zscore_df": zscore_df,
            "zscore_anomalies": zscore_anomalies,
            "iqr_anomalies": iqr_anomalies,
            "anomaly_flags": combined,
            "anomaly_scores": anomaly_scores,
        }

    def _rolling_zscore(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute rolling Z-score for each column.

        Rolling Z-score is better than global Z-score because it adapts
        to the LOCAL mean and variance of the signal. This handles
        signals that drift over time (e.g., temperature rising during sepsis).
        """
        zscore_df = pd.DataFrame(index=df.index)
        for col in df.columns:
            series = df[col]
            roll_mean = series.rolling(self.window, min_periods=1).mean()
            roll_std = series.rolling(self.window, min_periods=1).std().replace(0, 1e-6)
            zscore_df[col] = (series - roll_mean) / roll_std
        return zscore_df

    def _iqr_detection(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply Tukey's IQR fences to detect outliers.

        Q1 = 25th percentile, Q3 = 75th percentile
        IQR = Q3 - Q1
        Lower fence = Q1 - multiplier * IQR
        Upper fence = Q3 + multiplier * IQR
        Points outside the fences are flagged.
        """
        anomaly_df = pd.DataFrame(index=df.index)
        for col in df.columns:
            series = df[col]
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - self.iqr_multiplier * IQR
            upper = Q3 + self.iqr_multiplier * IQR
            anomaly_df[col] = (series < lower) | (series > upper)
        return anomaly_df

    def get_zscore_dataframe(self) -> Optional[pd.DataFrame]:
        """Return the last computed Z-score DataFrame."""
        return self._zscore_df
