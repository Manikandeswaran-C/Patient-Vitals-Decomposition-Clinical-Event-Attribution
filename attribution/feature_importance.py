"""
attribution/feature_importance.py — Global Feature Importance Analyzer
=======================================================================
Computes which vital signs are most important for anomaly detection globally.
Complements SHAP (which is per-anomaly) with dataset-level insights.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from typing import Optional, List, Dict


class FeatureImportanceAnalyzer:
    """
    Computes global feature importance using multiple methods:
      1. Mean absolute SHAP values (from SHAPExplainer output)
      2. Permutation importance via RandomForest surrogate
      3. Variance-based importance (features with high variance near anomalies)

    Usage:
        fia = FeatureImportanceAnalyzer()
        importance_df = fia.compute(feature_df, anomaly_flags, shap_values)
        fia.print_report(importance_df)
    """

    def compute(
        self,
        feature_df: pd.DataFrame,
        anomaly_flags: pd.Series,
        shap_df: Optional[pd.DataFrame] = None,
        feature_cols: Optional[List[str]] = None,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Compute feature importance using all available methods.

        Args:
            feature_df: Full feature DataFrame.
            anomaly_flags: Boolean Series indicating anomalies.
            shap_df: Optional SHAP values DataFrame from SHAPExplainer.
            feature_cols: Columns to analyze. Defaults to numeric columns.
            verbose: Print report.

        Returns:
            DataFrame with columns: [shap_importance, rf_importance, variance_importance, combined_rank]
        """
        print("\n📊 Step 5b: Global Feature Importance Analysis")

        if feature_cols:
            cols = [c for c in feature_cols if c in feature_df.columns]
        else:
            cols = feature_df.select_dtypes(include=[np.number]).columns.tolist()
            cols = [c for c in cols if c not in ("anomaly_label", "hour_of_day")]

        X = feature_df[cols].fillna(feature_df[cols].median())
        y = anomaly_flags.astype(int)

        results = pd.DataFrame(index=cols)

        # ── Method 1: Mean absolute SHAP ──────────────────────────────
        if shap_df is not None and not shap_df.empty:
            shap_cols = [c for c in cols if c in shap_df.columns]
            shap_importance = shap_df[shap_cols].abs().mean()
            shap_importance = shap_importance.reindex(cols).fillna(0)
            results["shap_importance"] = shap_importance
        else:
            results["shap_importance"] = 0.0

        # ── Method 2: Random Forest surrogate importance ───────────────
        rf_importance = self._rf_importance(X, y, cols)
        results["rf_importance"] = rf_importance

        # ── Method 3: Variance ratio near anomalies ────────────────────
        var_importance = self._variance_importance(X, anomaly_flags, cols)
        results["variance_importance"] = var_importance

        # ── Combined rank ──────────────────────────────────────────────
        # Normalize each method to [0, 1] then average
        for col in ["shap_importance", "rf_importance", "variance_importance"]:
            max_val = results[col].max()
            if max_val > 0:
                results[f"{col}_norm"] = results[col] / max_val
            else:
                results[f"{col}_norm"] = 0.0

        norm_cols = [c for c in results.columns if c.endswith("_norm")]
        results["combined_score"] = results[norm_cols].mean(axis=1)
        results["rank"] = results["combined_score"].rank(ascending=False).astype(int)
        results = results.sort_values("combined_score", ascending=False)

        # Drop intermediate norm columns
        results = results.drop(columns=norm_cols)

        if verbose:
            self.print_report(results)

        return results

    def _rf_importance(
        self, X: pd.DataFrame, y: pd.Series, cols: List[str]
    ) -> pd.Series:
        """Train a RandomForest classifier and extract feature importances."""
        try:
            if y.sum() < 5:
                return pd.Series(0.0, index=cols)

            rf = RandomForestClassifier(
                n_estimators=100, random_state=42, n_jobs=-1,
                class_weight="balanced",
            )
            rf.fit(X, y)
            return pd.Series(rf.feature_importances_, index=cols)
        except Exception:
            return pd.Series(0.0, index=cols)

    def _variance_importance(
        self,
        X: pd.DataFrame,
        anomaly_flags: pd.Series,
        cols: List[str],
    ) -> pd.Series:
        """
        Compute variance ratio: variance near anomalies / overall variance.
        High ratio → feature behaves differently near anomalies → important.
        """
        anomaly_idx = anomaly_flags[anomaly_flags].index
        if len(anomaly_idx) == 0:
            return pd.Series(0.0, index=cols)

        overall_var = X.var() + 1e-9
        anomaly_var = X.loc[anomaly_idx].var() + 1e-9
        ratio = anomaly_var / overall_var
        return ratio.reindex(cols).fillna(0)

    def print_report(self, importance_df: pd.DataFrame, top_n: int = 10):
        """Print the top N most important features."""
        top = importance_df.head(top_n)
        print(f"\n   Top {top_n} Most Important Features for Anomaly Attribution:")
        print(f"   {'Rank':<6} {'Feature':<28} {'SHAP':>8} {'RF':>8} {'Var':>8} {'Score':>8}")
        print(f"   {'─'*6} {'─'*28} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")
        for feat, row in top.iterrows():
            print(
                f"   {int(row.get('rank', 0)):<6} {feat:<28} "
                f"{row.get('shap_importance', 0):>8.4f} "
                f"{row.get('rf_importance', 0):>8.4f} "
                f"{row.get('variance_importance', 0):>8.4f} "
                f"{row.get('combined_score', 0):>8.4f}"
            )
        print("   ✅ Feature importance analysis complete.")
