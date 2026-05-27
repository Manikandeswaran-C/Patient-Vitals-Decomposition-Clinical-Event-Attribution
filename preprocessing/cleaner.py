"""
preprocessing/cleaner.py — Signal Cleaning Module
==================================================
Step 2 of the pipeline: Clean raw physiological signals.

Operations performed:
  1. Missing value detection and imputation
  2. Physiological range validation (clip impossible values)
  3. Noise filtering (moving average / Savitzky-Golay smoothing)
  4. Artifact removal (sudden spikes beyond physical possibility)

BEGINNER EXPLANATION:
    Real ICU monitors drop packets, have sensor disconnections, and
    produce motion artifacts. Before any analysis, we must clean the signal.
    Think of this like washing vegetables before cooking.
"""

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from typing import Optional, Dict, Tuple
import warnings

from config import VITAL_SIGNS, PREPROCESSING


class SignalCleaner:
    """
    Cleans raw physiological time-series data.

    Usage:
        cleaner = SignalCleaner()
        clean_df, report = cleaner.clean(raw_df)
        print(report)
    """

    def __init__(self):
        self.cleaning_report: Dict = {}

    def clean(
        self, df: pd.DataFrame, verbose: bool = True
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Run the full cleaning pipeline on the input DataFrame, and flag artifacts.

        Args:
            df: Raw DataFrame with vital sign columns and DatetimeIndex.
            verbose: Print cleaning statistics.

        Returns:
            (cleaned_df, report): Cleaned DataFrame and a report dict.
        """
        print("\n🧹 Step 2: Signal Cleaning & Artifact Detection")
        df = df.copy()
        vital_cols = [c for c in df.columns if c in VITAL_SIGNS]
        self.cleaning_report = {"n_original": len(df), "vitals": {}}

        # Detect artifacts first while data is still raw
        artifact_df = self.detect_artifacts(df)

        for col in vital_cols:
            col_report = {}
            cfg = VITAL_SIGNS[col]

            # --- 1. Count and record artifacts ---
            col_report["sensor_disconnects"] = int(artifact_df[f"{col}_sensor_disconnect"].sum())
            col_report["motion_artifacts"] = int(artifact_df[f"{col}_motion_artifact"].sum())

            # --- 2. Count initial missing values ---
            n_missing_before = df[col].isna().sum()
            col_report["missing_before"] = int(n_missing_before)

            # --- 3. Treat artifacts and impossible values as NaN to impute ---
            # Anything flagged as sensor disconnect or impossible should be replaced with NaN
            impossible_mask = (
                (df[col] < cfg["critical_low"] * 0.4) |
                (df[col] > cfg["critical_high"] * 1.6)
            )
            n_impossible = impossible_mask.sum()
            col_report["impossible_clipped"] = int(n_impossible)

            bad_data_mask = impossible_mask | artifact_df[f"{col}_sensor_disconnect"] | artifact_df[f"{col}_motion_artifact"]
            df.loc[bad_data_mask, col] = np.nan

            # --- 4. Impute missing values ---
            missing_fraction = df[col].isna().mean()
            if missing_fraction > PREPROCESSING["max_missing_fraction"]:
                warnings.warn(
                    f"Column '{col}' has {100*missing_fraction:.1f}% missing/artifact values. "
                    f"Data quality is low."
                )

            method = PREPROCESSING["interpolation_method"]
            if method == "linear":
                df[col] = df[col].interpolate(method="linear", limit_direction="both")
            elif method == "cubic":
                df[col] = df[col].interpolate(method="cubic", limit_direction="both")
            elif method == "forward_fill":
                df[col] = df[col].ffill().bfill()

            # Fill any remaining NaNs at edges with median
            df[col] = df[col].fillna(df[col].median())

            n_missing_after = df[col].isna().sum()
            col_report["missing_after"] = int(n_missing_after)

            # --- 5. Smooth to remove high-frequency noise ---
            df[col] = self._smooth_signal(df[col].values, col)
            col_report["smoothed"] = True

            self.cleaning_report["vitals"][col] = col_report

        self.cleaning_report["n_cleaned"] = len(df)

        if verbose:
            self._print_report(vital_cols)

        return df, self.cleaning_report

    def _smooth_signal(self, signal: np.ndarray, vital_name: str) -> np.ndarray:
        """
        Apply Savitzky-Golay filter for noise smoothing.
        """
        window_map = {
            "heart_rate": 7,
            "systolic_bp": 7,
            "diastolic_bp": 7,
            "spo2": 5,
            "respiratory_rate": 7,
            "temperature": 11,
            "etco2": 7,
        }
        window = window_map.get(vital_name, 7)

        if len(signal) < window:
            return signal

        try:
            return savgol_filter(signal, window_length=window, polyorder=3)
        except Exception:
            return signal

    def detect_artifacts(
        self, df: pd.DataFrame, threshold_std: float = 4.5
    ) -> pd.DataFrame:
        """
        Flag time points with Sensor Disconnects or Motion Artifacts.

        - Sensor Disconnect: Flatlines at 0 or NaN, or repeating values outside normal ranges.
        - Motion Artifact: Sudden changes (first differences) > threshold * std.
        """
        vital_cols = [c for c in df.columns if c in VITAL_SIGNS]
        artifact_df = pd.DataFrame(index=df.index)

        for col in vital_cols:
            cfg = VITAL_SIGNS[col]
            val = df[col]

            # 1. Sensor Disconnect (flatline to 0, NaN, or constant implausible value)
            is_nan = val.isna()
            is_zero = val == 0.0
            
            # Constant value check: rolling variance over 5 minutes is 0
            rolling_var = val.rolling(window=5, min_periods=1).var()
            is_flatline = (rolling_var == 0.0) & ((val < cfg["normal_low"]) | (val > cfg["normal_high"]))

            artifact_df[f"{col}_sensor_disconnect"] = is_nan | is_zero | is_flatline

            # 2. Motion Artifact (sudden extreme jump in first difference)
            diff = val.diff().abs()
            # Calculate standard deviation using a robust estimator (MAD equivalent)
            median_diff = diff.median()
            mad_diff = (diff - median_diff).abs().median()
            robust_std = 1.4826 * mad_diff if mad_diff > 0 else diff.std()
            
            # If standard deviation is 0 (due to constants), use a small default threshold
            threshold = max(robust_std * threshold_std, 0.05 * (cfg["normal_high"] - cfg["normal_low"]))
            
            # We flag jumps that are physiologically impossible
            artifact_df[f"{col}_motion_artifact"] = diff > threshold

        return artifact_df

    def _print_report(self, vital_cols):
        """Print a human-readable cleaning summary."""
        print(f"   {'Vital':<22} {'Disconnects':<14} {'Motion Noise':<14} {'Impossible Clipped'}")
        print(f"   {'─'*22} {'─'*14} {'─'*14} {'─'*18}")
        for col in vital_cols:
            r = self.cleaning_report["vitals"].get(col, {})
            disconnects = r.get("sensor_disconnects", 0)
            motion = r.get("motion_artifacts", 0)
            impossible = r.get("impossible_clipped", 0)
            print(f"   {col:<22} {disconnects:<14} {motion:<14} {impossible}")
        print(f"   Refilled and smoothed all vitals. {self.cleaning_report['n_cleaned']} samples ready.")
