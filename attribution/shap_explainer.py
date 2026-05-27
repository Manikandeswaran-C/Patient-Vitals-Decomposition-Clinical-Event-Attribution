"""
attribution/shap_explainer.py — SHAP-Based Anomaly Attribution
==============================================================
Step 5: Explain WHY each anomaly occurred using SHAP values.

WHAT IS SHAP?
    SHAP = SHapley Additive exPlanations (Lundberg & Lee, 2017)
    Based on cooperative game theory (Shapley values).
    For each anomaly, it answers:
    "How much did each vital sign CONTRIBUTE to the anomaly score?"

    SHAP values sum to the total anomaly score:
      anomaly_score = baseline + SHAP(HR) + SHAP(SpO2) + SHAP(RR) + ...

    Positive SHAP → that vital INCREASED the anomaly score (pushed toward anomaly)
    Negative SHAP → that vital DECREASED the anomaly score (pushed toward normal)

ANOMALY TYPES (based on SHAP attribution):
    1. BASELINE SHIFT:   Trend component has high SHAP → patient state drifting
    2. CIRCADIAN DEVIATION: Seasonal component has high SHAP → abnormal time-of-day pattern
    3. ACUTE IRREGULAR EVENT: Residual has high SHAP → sudden unexplained spike

BEGINNER NOTE:
    SHAP is like asking "who is responsible for this outcome?"
    If HR contributes 60% of the anomaly score → HR is the main culprit.
    This gives clinicians a clear, justified reason to look at HR first.
"""

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import IsolationForest
from typing import Dict, Optional, List, Tuple


class SHAPExplainer:
    """
    Computes SHAP attributions for anomalies detected by Isolation Forest.

    Usage:
        explainer = SHAPExplainer()
        explainer.fit(feature_df, isolation_forest_model)
        shap_df = explainer.explain_anomalies(anomaly_indices)
        attribution_types = explainer.classify_attribution_types(
            stl_components, shap_df
        )
    """

    def __init__(self, background_samples: int = 100):
        self.background_samples = background_samples
        self._explainer = None
        self._shap_values: Optional[np.ndarray] = None
        self._feature_cols: List[str] = []
        self._base_df: Optional[pd.DataFrame] = None

    def fit(
        self,
        feature_df: pd.DataFrame,
        model: IsolationForest,
        feature_cols: Optional[List[str]] = None,
    ) -> "SHAPExplainer":
        """
        Build the SHAP explainer using the trained Isolation Forest.

        Args:
            feature_df: Feature DataFrame used to train the IF model.
            model: Fitted IsolationForest instance.
            feature_cols: Feature columns to use. Defaults to all numeric cols.
        """
        print("\n🔬 Step 5: SHAP Attribution Setup")

        if feature_cols:
            self._feature_cols = [c for c in feature_cols if c in feature_df.columns]
        else:
            self._feature_cols = feature_df.select_dtypes(include=[np.number]).columns.tolist()
            self._feature_cols = [c for c in self._feature_cols
                                  if c not in ("anomaly_label", "hour_of_day")]

        X = feature_df[self._feature_cols].fillna(feature_df[self._feature_cols].median())
        self._base_df = X

        # Sample background data for SHAP (random subset of normal points)
        n_bg = min(self.background_samples, len(X))
        background = shap.sample(X, n_bg, random_state=42)

        # TreeExplainer is fast for tree-based models (Isolation Forest)
        # If IF is not tree-compatible, fall back to KernelExplainer
        try:
            self._explainer = shap.TreeExplainer(model, data=background)
            print("   Using TreeExplainer (fast, exact for Isolation Forest)")
        except Exception:
            print("   Falling back to KernelExplainer (slower, model-agnostic)")
            predict_fn = lambda x: model.decision_function(x)
            self._explainer = shap.KernelExplainer(predict_fn, background)

        print(f"   Features: {', '.join(self._feature_cols[:5])}{'...' if len(self._feature_cols) > 5 else ''}")
        print(f"   Background samples: {n_bg}")
        return self

    def explain_anomalies(
        self, anomaly_mask: pd.Series, max_explain: int = 50
    ) -> pd.DataFrame:
        """
        Compute SHAP values for detected anomaly time points.

        Args:
            anomaly_mask: Boolean Series (True = anomaly).
            max_explain: Maximum number of anomalies to explain (for speed).

        Returns:
            DataFrame of SHAP values: rows = anomaly time points,
            columns = feature names.
        """
        print(f"\n   🧮 Computing SHAP values for anomalies...")
        anomaly_idx = anomaly_mask[anomaly_mask].index

        if len(anomaly_idx) == 0:
            print("   No anomalies to explain.")
            return pd.DataFrame()

        # Limit for computational feasibility
        if len(anomaly_idx) > max_explain:
            anomaly_idx = anomaly_idx[:max_explain]
            print(f"   Limiting to first {max_explain} anomalies for speed.")

        X_anomaly = self._base_df.loc[anomaly_idx]

        try:
            shap_values = self._explainer.shap_values(X_anomaly)
            # For Isolation Forest, shap_values may be a list → take [0]
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
        except Exception as e:
            print(f"   ⚠️ SHAP computation error: {e}. Using approximate method.")
            shap_values = self._approximate_shap(X_anomaly)

        self._shap_values = shap_values
        shap_df = pd.DataFrame(
            shap_values,
            index=anomaly_idx,
            columns=self._feature_cols,
        )

        print(f"   ✅ SHAP values computed for {len(shap_df)} anomalies.")
        return shap_df

    def classify_attribution_types(
        self,
        stl_components: Dict[str, pd.DataFrame],
        shap_df: pd.DataFrame,
        trend_threshold: float = 1.5,
        seasonal_threshold: float = 1.5,
    ) -> pd.DataFrame:
        """
        Classify each anomaly into one of three types based on SHAP + STL components.

        Types:
            - BASELINE_SHIFT:       Sustained drift in trend component (Z-score > threshold)
            - CIRCADIAN_DEVIATION:  Sustained shift in circadian seasonality (Z-score > threshold)
            - ACUTE_EVENT:          Sudden transient spike in residual component

        Args:
            stl_components: Output of STLDecomposer.decompose()
            shap_df: SHAP values DataFrame from explain_anomalies()
            trend_threshold: Minimum Trend component Z-score to classify as baseline shift
            seasonal_threshold: Minimum Seasonal component Z-score to classify as circadian deviation

        Returns:
            DataFrame with anomaly classification, Z-scores, and dominant vital.
        """
        classifications = []

        # Precompute std and mean of trend, seasonal, and residual components for all vitals to scale deviations
        component_stds = {}
        component_means = {}
        for vital, comp in stl_components.items():
            component_stds[vital] = {
                "trend": comp["trend"].std(),
                "seasonal": comp["seasonal"].std(),
                "residual": comp["residual"].std(),
            }
            component_means[vital] = {
                "trend": comp["trend"].mean(),
                "seasonal": comp["seasonal"].mean(),
                "residual": comp["residual"].mean(),
            }

        for ts in shap_df.index:
            row = {"timestamp": ts}

            # Find dominant vital sign (highest absolute SHAP value)
            abs_shap = shap_df.loc[ts].abs()
            # Only consider vital sign columns (not engineered features)
            vital_shap = abs_shap[
                [c for c in abs_shap.index if not any(
                    c.endswith(s) for s in
                    ["_roll_mean", "_roll_std", "_roll_min", "_roll_max", "_roc"]
                )]
            ]
            dominant_vital = vital_shap.idxmax()
            row["dominant_vital"] = dominant_vital
            row["shap_magnitude"] = float(abs_shap.sum())

            # Check trend and seasonal Z-scores at this time point
            trend_z = 0.0
            seasonal_z = 0.0
            residual_z = 0.0

            if dominant_vital in stl_components:
                comp = stl_components[dominant_vital]
                if ts in comp.index:
                    t_val = comp.loc[ts, "trend"]
                    s_val = comp.loc[ts, "seasonal"]
                    r_val = comp.loc[ts, "residual"]
                    
                    t_std = component_stds[dominant_vital]["trend"]
                    s_std = component_stds[dominant_vital]["seasonal"]
                    r_std = component_stds[dominant_vital]["residual"]
                    
                    t_mean = component_means[dominant_vital]["trend"]
                    s_mean = component_means[dominant_vital]["seasonal"]
                    r_mean = component_means[dominant_vital]["residual"]
                    
                    trend_z = abs(t_val - t_mean) / (t_std + 1e-9)
                    seasonal_z = abs(s_val - s_mean) / (s_std + 1e-9)
                    residual_z = abs(r_val - r_mean) / (r_std + 1e-9)

            row["trend_z"] = round(trend_z, 3)
            row["seasonal_z"] = round(seasonal_z, 3)
            row["residual_z"] = round(residual_z, 3)

            # Classify by dominant component deviation
            z_scores = {"trend": trend_z, "seasonal": seasonal_z, "residual": residual_z}
            dominant_comp = max(z_scores, key=z_scores.get)

            if dominant_comp == "trend" and trend_z > trend_threshold:
                row["anomaly_type"] = "BASELINE_SHIFT"
                row["explanation"] = (
                    f"Gradual baseline shift detected in {dominant_vital} (trend deviation Z-score: {trend_z:.1f}). "
                    f"Suggests a developing clinical trend, e.g. onset of fever, sustained hemodynamic drift, or medication response."
                )
            elif dominant_comp == "seasonal" and seasonal_z > seasonal_threshold:
                row["anomaly_type"] = "CIRCADIAN_DEVIATION"
                row["explanation"] = (
                    f"Circadian rhythm deviation detected in {dominant_vital} (diurnal seasonal deviation Z-score: {seasonal_z:.1f}). "
                    f"Suggests disrupted sleep patterns, autonomic dysregulation, or environmental factors (ICU delirium/noise)."
                )
            else:
                row["anomaly_type"] = "ACUTE_EVENT"
                row["explanation"] = (
                    f"Acute irregular event in {dominant_vital} (residual spike Z-score: {residual_z:.1f}). "
                    f"A sudden unexplained cardiorespiratory spike or transient crisis event."
                )

            classifications.append(row)

        result = pd.DataFrame(classifications)
        if not result.empty and "timestamp" in result.columns:
            result = result.set_index("timestamp")

        return result

    def get_mean_shap_importance(self) -> pd.Series:
        """
        Compute mean absolute SHAP value per feature (global importance).
        Higher = that feature contributes more to anomaly detection on average.
        """
        if self._shap_values is None:
            return pd.Series()
        return pd.Series(
            np.abs(self._shap_values).mean(axis=0),
            index=self._feature_cols,
        ).sort_values(ascending=False)

    def _approximate_shap(self, X: pd.DataFrame) -> np.ndarray:
        """Fallback: use feature deviation from background mean as proxy for SHAP."""
        bg_mean = self._base_df.mean()
        diff = (X - bg_mean).values
        return diff
