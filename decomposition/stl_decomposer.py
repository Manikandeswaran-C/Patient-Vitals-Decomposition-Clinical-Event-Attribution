"""
decomposition/stl_decomposer.py — STL Signal Decomposition
===========================================================
Step 3 of the pipeline: Decompose each vital sign into:
  - TREND:    Long-term direction (e.g., slowly rising HR over 6 hours)
  - SEASONAL: Repeating periodic patterns (e.g., circadian rhythm)
  - RESIDUAL: What remains after removing trend + seasonal components
              → This is where anomalies "live"

WHY STL (Seasonal-Trend decomposition using LOESS)?
  - Robust to outliers (LOESS = Locally Weighted Regression)
  - Handles missing values better than classical decomposition
  - Allows the seasonal component to change over time

VARIANCE ANALYSIS:
  We compute what fraction of total signal variance each component explains.
  If residual variance > threshold → signal is "unstable" → flag for closer scrutiny.

BEGINNER ANALOGY:
    Imagine heart rate over 24 hours:
      - Trend = "This patient's HR has been slowly creeping up since midnight"
      - Seasonal = "HR is always lower at 3am and higher at 3pm" (circadian)
      - Residual = "At 2:17am there was a sudden spike — that's the anomaly"
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL
from typing import Dict, Tuple, Optional
import warnings

from config import VITAL_SIGNS, DECOMPOSITION


class STLDecomposer:
    """
    Applies STL decomposition to each vital sign in the dataset.

    Usage:
        decomp = STLDecomposer(period=60)
        results = decomp.decompose(clean_df)
        variance_report = decomp.variance_analysis()
        print(decomp.get_residuals())   # DataFrame of residuals only
    """

    def __init__(self, period: Optional[int] = None):
        """
        Args:
            period: The expected cycle length in samples.
                    For 1-sample/minute data: 60 = hourly period, 1440 = daily.
                    Defaults to config value.
        """
        self.period = period or DECOMPOSITION["stl_period"]
        self._results: Dict[str, object] = {}      # STL result objects per vital
        self._components: Dict[str, pd.DataFrame] = {}  # DataFrames per vital
        self._variance_report: Dict[str, Dict] = {}

    def decompose(self, df: pd.DataFrame, verbose: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Decompose all vital signs using STL.

        Args:
            df: Cleaned DataFrame with vital sign columns and DatetimeIndex.
            verbose: Print progress and results.

        Returns:
            Dict mapping vital name → DataFrame with columns:
            ['observed', 'trend', 'seasonal', 'residual']
        """
        print("\n📊 Step 3: STL Signal Decomposition")
        vital_cols = [c for c in df.columns if c in VITAL_SIGNS]

        for col in vital_cols:
            series = df[col].dropna()

            # STL requires at least 2 * period data points
            if len(series) < 2 * self.period:
                warnings.warn(
                    f"'{col}': Only {len(series)} samples but period={self.period}. "
                    f"Skipping STL — using raw signal as residual."
                )
                comp = pd.DataFrame({
                    "observed": series,
                    "trend": series.rolling(self.period, center=True, min_periods=1).mean(),
                    "seasonal": np.zeros(len(series)),
                    "residual": series - series.rolling(self.period, center=True, min_periods=1).mean().fillna(0),
                }, index=series.index)
                self._components[col] = comp
                continue

            # Run STL decomposition
            # seasonal must be odd, >= 7
            seasonal_window = max(7, self.period | 1)  # Ensure odd

            try:
                stl = STL(
                    series,
                    period=self.period,
                    seasonal=seasonal_window,
                    robust=True,   # Robust LOESS — downweights outliers automatically
                )
                result = stl.fit()
                self._results[col] = result

                comp = pd.DataFrame({
                    "observed":  series.values,
                    "trend":     result.trend,
                    "seasonal":  result.seasonal,
                    "residual":  result.resid,
                }, index=series.index)
                self._components[col] = comp

            except Exception as e:
                warnings.warn(f"STL failed for '{col}': {e}. Using fallback.")
                trend = series.rolling(self.period, center=True, min_periods=1).mean()
                comp = pd.DataFrame({
                    "observed": series.values,
                    "trend": trend.values,
                    "seasonal": np.zeros(len(series)),
                    "residual": (series - trend.fillna(series.mean())).values,
                }, index=series.index)
                self._components[col] = comp

        # Compute variance analysis
        self._variance_report = self._compute_variance_analysis(vital_cols)

        if verbose:
            self._print_variance_report(vital_cols)

        return self._components

    def get_residuals(self) -> pd.DataFrame:
        """
        Extract residuals for all decomposed vitals into a single DataFrame.
        
        The residual is the most important output:
        It's the "unexplained" part after removing trend and seasonal.
        Anomalies manifest as large residual values.
        """
        residuals = {}
        for col, comp in self._components.items():
            residuals[col] = comp["residual"]
        return pd.DataFrame(residuals)

    def get_trend(self) -> pd.DataFrame:
        """Return trend components for all vitals."""
        trends = {}
        for col, comp in self._components.items():
            trends[col] = comp["trend"]
        return pd.DataFrame(trends)

    def get_seasonal(self) -> pd.DataFrame:
        """Return seasonal components for all vitals."""
        seasonal = {}
        for col, comp in self._components.items():
            seasonal[col] = comp["seasonal"]
        return pd.DataFrame(seasonal)

    def variance_analysis(self) -> pd.DataFrame:
        """
        Return a DataFrame showing variance contribution of each component.
        
        CLINICAL MEANING:
            - High residual variance (>30%) → signal is unstable or noisy
            - High trend variance (>50%) → patient is in a changing state
            - High seasonal variance → normal circadian behavior dominates
        """
        rows = []
        for col, report in self._variance_report.items():
            rows.append({
                "vital": col,
                "trend_var_%": report["trend_pct"],
                "seasonal_var_%": report["seasonal_pct"],
                "residual_var_%": report["residual_pct"],
                "total_variance": report["total_var"],
                "unstable_flag": report["unstable"],
            })
        return pd.DataFrame(rows).set_index("vital")

    def _compute_variance_analysis(
        self, vital_cols, residual_threshold: float = 0.30
    ) -> Dict:
        """
        Compute percentage variance of trend, seasonal, residual components.
        Flag vitals where residual explains more than threshold of total variance.
        """
        report = {}
        for col in vital_cols:
            if col not in self._components:
                continue
            comp = self._components[col]
            total_var = comp["observed"].var()
            if total_var == 0:
                total_var = 1e-9  # Avoid division by zero

            trend_var = comp["trend"].var()
            seasonal_var = comp["seasonal"].var()
            residual_var = comp["residual"].var()

            residual_pct = residual_var / total_var * 100

            report[col] = {
                "trend_pct": round(trend_var / total_var * 100, 2),
                "seasonal_pct": round(seasonal_var / total_var * 100, 2),
                "residual_pct": round(residual_pct, 2),
                "total_var": round(total_var, 4),
                "unstable": residual_pct > (residual_threshold * 100),
            }
        return report

    def _print_variance_report(self, vital_cols):
        """Print the variance decomposition table."""
        print(f"\n   {'Vital':<22} {'Trend%':>8} {'Seasonal%':>10} "
              f"{'Residual%':>10} {'Unstable?':>10}")
        print(f"   {'─'*22} {'─'*8} {'─'*10} {'─'*10} {'─'*10}")
        for col in vital_cols:
            r = self._variance_report.get(col, {})
            flag = "⚠️  YES" if r.get("unstable") else "NO"
            print(
                f"   {col:<22} {r.get('trend_pct', 0):>7.1f}% "
                f"{r.get('seasonal_pct', 0):>9.1f}% "
                f"{r.get('residual_pct', 0):>9.1f}%  {flag}"
            )
        print("   ✅ STL decomposition complete.")
