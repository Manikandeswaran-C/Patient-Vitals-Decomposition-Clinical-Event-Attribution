"""
relationships/causality.py — Granger Causality Analysis
=========================================================
Step 6c: Test if one vital sign PREDICTS another using Granger causality.

WHAT IS GRANGER CAUSALITY?
    X "Granger-causes" Y if knowing the PAST values of X significantly
    improves the prediction of Y beyond using Y's own past alone.

    This is NOT true causality — it's PREDICTIVE causality.
    But in clinical context, it's extremely useful:
    "Rising respiratory rate predicts SpO2 drops 3 minutes later"
    is actionable clinical information, regardless of whether RR
    truly "causes" SpO2 to drop.

HOW IT WORKS:
    1. Fit a VAR (Vector Autoregression) model: Y ~ past(Y)
    2. Fit an extended VAR model: Y ~ past(Y) + past(X)
    3. F-test: Does adding past(X) significantly reduce prediction error?
       If yes → X Granger-causes Y at that lag.

BEGINNER NOTE:
    We test multiple lag values (1 to max_lag).
    The lag at which the F-test is most significant is the
    "predictive horizon" — how many minutes ahead X predicts Y.
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests
from typing import Dict, Optional, List
import warnings

from config import VITAL_SIGNS, RELATIONSHIPS


class GrangerCausalityAnalyzer:
    """
    Tests Granger causality relationships between vital sign residuals.

    Usage:
        gca = GrangerCausalityAnalyzer(max_lag=10)
        results = gca.analyze(residuals_df)
        network = gca.build_causal_network(results)
        gca.print_report(results)
    """

    def __init__(self, max_lag: Optional[int] = None, significance: Optional[float] = None):
        self.max_lag = max_lag or RELATIONSHIPS["granger"]["max_lag"]
        self.significance = significance or RELATIONSHIPS["granger"]["significance_level"]
        self._results: Dict = {}

    def analyze(
        self,
        residuals: pd.DataFrame,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Run Granger causality tests for all ordered pairs of vital signs.

        For each pair (X, Y), tests: "Does X Granger-cause Y?"
        This is asymmetric: X→Y and Y→X are tested separately.

        Args:
            residuals: DataFrame of STL residuals (stationarity preferred).
            verbose: Print significant causal relationships.

        Returns:
            DataFrame with columns:
            [cause, effect, best_lag, min_pvalue, significant, interpretation]
        """
        print("\n🧬 Step 6c: Granger Causality Analysis")
        vital_cols = [c for c in residuals.columns if c in VITAL_SIGNS]

        if len(vital_cols) < 2:
            print("   ⚠️ Need at least 2 vital signs for Granger causality.")
            return pd.DataFrame()

        rows = []

        for cause in vital_cols:
            for effect in vital_cols:
                if cause == effect:
                    continue

                # Granger test requires [effect, cause] column order
                pair_data = residuals[[effect, cause]].dropna()

                if len(pair_data) < self.max_lag * 3:
                    continue

                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        test_result = grangercausalitytests(
                            pair_data,
                            maxlag=self.max_lag,
                            verbose=False,
                        )

                    # Extract best (minimum) p-value across all lags
                    pvalues_by_lag = {}
                    for lag, result in test_result.items():
                        # result[0] contains test results for different test types
                        # 'ssr_ftest' = F-test on sum-of-squared residuals
                        f_test = result[0].get("ssr_ftest")
                        if f_test is not None:
                            pvalues_by_lag[lag] = f_test[1]  # p-value

                    if not pvalues_by_lag:
                        continue

                    best_lag = min(pvalues_by_lag, key=pvalues_by_lag.get)
                    min_pvalue = pvalues_by_lag[best_lag]
                    significant = min_pvalue < self.significance

                    # Interpretation
                    if significant:
                        interpretation = (
                            f"{cause} predicts {effect} with a {best_lag}-sample lag "
                            f"(p={min_pvalue:.4f}). Changes in {cause} precede "
                            f"changes in {effect} by ~{best_lag} minute(s)."
                        )
                    else:
                        interpretation = f"No significant predictive relationship from {cause} to {effect}."

                    rows.append({
                        "cause": cause,
                        "effect": effect,
                        "best_lag": best_lag,
                        "min_pvalue": round(min_pvalue, 6),
                        "significant": significant,
                        "interpretation": interpretation,
                    })

                    self._results[f"{cause}→{effect}"] = {
                        "pvalues_by_lag": pvalues_by_lag,
                        "best_lag": best_lag,
                        "significant": significant,
                    }

                except Exception as e:
                    warnings.warn(f"Granger test failed for {cause}→{effect}: {e}")

        result_df = pd.DataFrame(rows)

        if not result_df.empty:
            result_df = result_df.sort_values("min_pvalue")

        if verbose:
            self._print_report(result_df)

        return result_df

    def build_causal_network(self, results_df: pd.DataFrame) -> Dict:
        """
        Build a directed causal network from significant Granger relationships.

        Returns:
            Dict with:
              - 'edges': list of (cause, effect, lag, pvalue) tuples
              - 'nodes': list of vital sign names
              - 'adjacency': adjacency matrix as DataFrame
        """
        significant = results_df[results_df["significant"]]
        vital_cols = list(set(results_df["cause"].tolist() + results_df["effect"].tolist()))

        # Adjacency matrix
        adj = pd.DataFrame(0.0, index=vital_cols, columns=vital_cols)
        edges = []

        for _, row in significant.iterrows():
            adj.loc[row["cause"], row["effect"]] = 1.0 / row["min_pvalue"]  # weight by significance
            edges.append((row["cause"], row["effect"], row["best_lag"], row["min_pvalue"]))

        return {
            "edges": edges,
            "nodes": vital_cols,
            "adjacency": adj,
        }

    def _print_report(self, result_df: pd.DataFrame):
        """Print significant causal relationships."""
        sig = result_df[result_df["significant"]]
        print(f"\n   Significance level: p < {self.significance}")
        print(f"   Max lag tested: {self.max_lag} samples")
        print(f"   Significant relationships: {len(sig)} / {len(result_df)}")

        if not sig.empty:
            print(f"\n   {'Cause':<22} {'→'} {'Effect':<22} {'Lag':>6} {'p-value':>10}")
            print(f"   {'─'*22}   {'─'*22} {'─'*6} {'─'*10}")
            for _, row in sig.iterrows():
                print(
                    f"   {row['cause']:<22} → {row['effect']:<22} "
                    f"{row['best_lag']:>5}  {row['min_pvalue']:>10.4f}"
                )
        else:
            print("   No significant Granger causal relationships found.")

        print("   ✅ Granger causality analysis complete.")
