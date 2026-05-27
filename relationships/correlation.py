"""
relationships/correlation.py — Cross-Signal Correlation & Lead-Lag Analysis
============================================================================
Step 6a: Identify relationships between vital signs.

TWO ANALYSES:
  1. CONTEMPORANEOUS CORRELATION:
     Do signals move together at the same time?
     - Pearson: linear correlation (e.g., HR and SBP)
     - Spearman: monotonic correlation (robust to outliers)

  2. LEAD-LAG / CROSS-CORRELATION:
     Does signal A PREDICT signal B a few minutes later?
     Cross-correlation at lag k: corr(X(t), Y(t+k))
     If peak correlation is at lag=5 → X leads Y by 5 minutes.

     CLINICAL EXAMPLE:
       Respiratory rate rising → SpO2 falling 3 minutes later
       (RR leads SpO2 by 3 samples at 1-min intervals)

BEGINNER NOTE:
    "Correlation ≠ Causation" — but lead-lag tells us the DIRECTION
    of the relationship, which is a key step toward causality.
    Granger causality (in causality.py) takes this further.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import correlate
from typing import Dict, Tuple, Optional, List

from config import VITAL_SIGNS, RELATIONSHIPS


class CorrelationAnalyzer:
    """
    Analyzes pairwise correlations and cross-signal lead-lag relationships.

    Usage:
        ca = CorrelationAnalyzer()
        corr_matrix = ca.compute_correlation(df)
        lead_lag_df = ca.compute_lead_lag(residuals_df)
        ca.print_report()
    """

    def __init__(self):
        self._correlation_matrix: Optional[pd.DataFrame] = None
        self._pvalue_matrix: Optional[pd.DataFrame] = None
        self._lead_lag_results: Dict = {}

    def compute_correlation(
        self,
        df: pd.DataFrame,
        method: Optional[str] = None,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Compute pairwise Pearson or Spearman correlation matrix.

        Args:
            df: DataFrame of vital signs (or residuals).
            method: "pearson" or "spearman". Defaults to config.
            verbose: Print significant correlations.

        Returns:
            Correlation matrix DataFrame.
        """
        print("\n🔗 Step 6a: Cross-Signal Correlation Analysis")
        method = method or RELATIONSHIPS["correlation_method"]
        vital_cols = [c for c in df.columns if c in VITAL_SIGNS]

        if len(vital_cols) < 2:
            print("   ⚠️ Need at least 2 vital signs for correlation analysis.")
            return pd.DataFrame()

        data = df[vital_cols].dropna()
        n_vitals = len(vital_cols)

        corr_mat = np.zeros((n_vitals, n_vitals))
        pval_mat = np.zeros((n_vitals, n_vitals))

        for i, col_a in enumerate(vital_cols):
            for j, col_b in enumerate(vital_cols):
                if i == j:
                    corr_mat[i, j] = 1.0
                    pval_mat[i, j] = 0.0
                else:
                    if method == "pearson":
                        r, p = stats.pearsonr(data[col_a], data[col_b])
                    else:
                        r, p = stats.spearmanr(data[col_a], data[col_b])
                    corr_mat[i, j] = r
                    pval_mat[i, j] = p

        self._correlation_matrix = pd.DataFrame(
            corr_mat, index=vital_cols, columns=vital_cols
        )
        self._pvalue_matrix = pd.DataFrame(
            pval_mat, index=vital_cols, columns=vital_cols
        )

        if verbose:
            sig_level = RELATIONSHIPS["significance_level"]
            print(f"\n   Method: {method.capitalize()} | Significance: p < {sig_level}")
            print(f"\n   Significant correlations (|r| > 0.3):")
            print(f"   {'Signal A':<22} {'Signal B':<22} {'r':>8} {'p-value':>12} {'Strength'}")
            print(f"   {'─'*22} {'─'*22} {'─'*8} {'─'*12} {'─'*14}")
            for i, col_a in enumerate(vital_cols):
                for j, col_b in enumerate(vital_cols):
                    if i >= j:
                        continue
                    r = corr_mat[i, j]
                    p = pval_mat[i, j]
                    if abs(r) > 0.3 and p < sig_level:
                        direction = "↑↑" if r > 0 else "↑↓"
                        strength = (
                            "Strong" if abs(r) > 0.7 else
                            "Moderate" if abs(r) > 0.5 else "Weak"
                        )
                        print(f"   {col_a:<22} {col_b:<22} {r:>8.3f} {p:>12.2e}  {direction} {strength}")
            print("   ✅ Correlation analysis complete.")

        return self._correlation_matrix

    def compute_lead_lag(
        self,
        residuals: pd.DataFrame,
        max_lag: Optional[int] = None,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Compute cross-correlation to find lead-lag relationships between residuals.

        The residual signal is used (not raw) because:
        - Trend and seasonal components create spurious correlations
        - Residuals reveal the TRUE dynamic coupling between signals

        Args:
            residuals: DataFrame of STL residuals.
            max_lag: Maximum lag in samples to test.
            verbose: Print results.

        Returns:
            DataFrame with columns: [signal_a, signal_b, peak_lag, peak_corr, direction]
            peak_lag > 0 → signal_a LEADS signal_b by peak_lag samples
            peak_lag < 0 → signal_b LEADS signal_a
        """
        print("\n🔀 Step 6b: Cross-Signal Lead-Lag Analysis")
        max_lag = max_lag or RELATIONSHIPS["cross_correlation"]["max_lag"]
        vital_cols = [c for c in residuals.columns if c in VITAL_SIGNS]
        rows = []

        for i, col_a in enumerate(vital_cols):
            for col_b in vital_cols[i+1:]:
                sig_a = residuals[col_a].fillna(0).values
                sig_b = residuals[col_b].fillna(0).values

                # Normalize signals
                sig_a = (sig_a - sig_a.mean()) / (sig_a.std() + 1e-9)
                sig_b = (sig_b - sig_b.mean()) / (sig_b.std() + 1e-9)

                # Compute full cross-correlation
                xcorr = correlate(sig_a, sig_b, mode="full")
                xcorr /= len(sig_a)  # Normalize by length

                # Lags range from -(N-1) to +(N-1)
                n = len(sig_a)
                lags = np.arange(-(n - 1), n)

                # Restrict to ±max_lag
                mask = np.abs(lags) <= max_lag
                xcorr_restricted = xcorr[mask]
                lags_restricted = lags[mask]

                # Find peak
                peak_idx = np.argmax(np.abs(xcorr_restricted))
                peak_lag = int(lags_restricted[peak_idx])
                peak_corr = float(xcorr_restricted[peak_idx])

                # Determine directionality
                if peak_lag > 0:
                    leader, follower = col_a, col_b
                    lag_abs = peak_lag
                elif peak_lag < 0:
                    leader, follower = col_b, col_a
                    lag_abs = abs(peak_lag)
                else:
                    leader, follower = col_a, col_b
                    lag_abs = 0

                rows.append({
                    "signal_a": col_a,
                    "signal_b": col_b,
                    "peak_lag_samples": peak_lag,
                    "peak_correlation": round(peak_corr, 4),
                    "leader": leader,
                    "follower": follower,
                    "lag_magnitude": lag_abs,
                    "coupling_strength": abs(peak_corr),
                })

                self._lead_lag_results[f"{col_a}___{col_b}"] = {
                    "lags": lags_restricted,
                    "xcorr": xcorr_restricted,
                }

        result_df = pd.DataFrame(rows).sort_values("coupling_strength", ascending=False)

        if verbose and not result_df.empty:
            print(f"   {'Leader':<22} {'Follower':<22} {'Lag':>8} {'Corr':>8}")
            print(f"   {'─'*22} {'─'*22} {'─'*8} {'─'*8}")
            for _, row in result_df.head(10).iterrows():
                print(
                    f"   {row['leader']:<22} {row['follower']:<22} "
                    f"{row['lag_magnitude']:>6} min  {row['peak_correlation']:>8.3f}"
                )
            print("   ✅ Lead-lag analysis complete.")

        return result_df

    def get_correlation_matrix(self) -> Optional[pd.DataFrame]:
        return self._correlation_matrix

    def get_pvalue_matrix(self) -> Optional[pd.DataFrame]:
        return self._pvalue_matrix

    def get_xcorr_data(self, col_a: str, col_b: str) -> Optional[Dict]:
        """Get raw cross-correlation arrays for plotting."""
        key = f"{col_a}___{col_b}"
        return self._lead_lag_results.get(key)
