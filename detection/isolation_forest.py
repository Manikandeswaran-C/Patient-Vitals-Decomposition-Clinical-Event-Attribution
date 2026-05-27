"""
detection/isolation_forest.py — Isolation Forest Anomaly Detector
==================================================================
Step 4b: Multivariate anomaly detection using Isolation Forest.

WHY ISOLATION FOREST?
    Statistical methods (Z-score, IQR) look at ONE vital sign at a time.
    Isolation Forest considers ALL vitals SIMULTANEOUSLY.

    This catches "context anomalies" — cases where each individual vital
    is within normal range, but the COMBINATION is impossible.
    Example: HR=100 (normal) + SpO2=97% (normal) + Temp=39.8°C (barely normal)
             COMBINED → possible early sepsis, flagged by multivariate detection.

HOW IT WORKS:
    The algorithm builds random decision trees that try to "isolate" each point.
    Anomalies are isolated faster (fewer splits needed) because they're in
    sparse regions of the feature space. Anomaly score = average path length.

BEGINNER ANALOGY:
    Imagine a game of 20 Questions. Normal points take many questions to identify
    (they blend in with others). Anomalies take very few questions
    ("Is it the only point that high?" → YES → anomaly found quickly).
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from typing import Optional, Dict
import joblib
from pathlib import Path

from config import VITAL_SIGNS, DETECTION


class IsolationForestDetector:
    """
    Detects multivariate anomalies using Isolation Forest.

    Usage:
        det = IsolationForestDetector()
        results = det.fit_detect(feature_df)
        print(results["anomaly_flags"])   # boolean Series
        print(results["anomaly_scores"])  # continuous scores (lower = more anomalous)
    """

    def __init__(
        self,
        n_estimators: Optional[int] = None,
        contamination: Optional[float] = None,
        random_state: Optional[int] = None,
    ):
        """
        Args:
            n_estimators: Number of isolation trees.
            contamination: Expected fraction of anomalies in the data.
            random_state: For reproducibility.
        """
        cfg = DETECTION["isolation_forest"]
        self.n_estimators = n_estimators or cfg["n_estimators"]
        self.contamination = contamination or cfg["contamination"]
        self.random_state = random_state or cfg["random_state"]

        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
            max_features=cfg["max_features"],
            n_jobs=-1,      # Use all CPU cores
        )
        self._feature_cols: list = []
        self._fitted = False

    def fit_detect(
        self,
        df: pd.DataFrame,
        feature_cols: Optional[list] = None,
        verbose: bool = True,
    ) -> Dict:
        """
        Fit the model and detect anomalies in one step.

        Args:
            df: DataFrame with feature columns (vitals + engineered features).
            feature_cols: Columns to use as features.
                          Defaults to vital sign columns only.
            verbose: Print detection summary.

        Returns:
            Dict with:
              - 'anomaly_flags'  : boolean Series (True = anomaly)
              - 'anomaly_scores' : float Series (lower = more anomalous)
              - 'raw_predictions': +1 (normal) or -1 (anomaly) per sklearn convention
        """
        print("\n🌲 Step 4b: Isolation Forest Detection")

        # Select feature columns
        if feature_cols:
            self._feature_cols = [c for c in feature_cols if c in df.columns]
        else:
            self._feature_cols = [c for c in df.columns if c in VITAL_SIGNS]

        X = df[self._feature_cols].copy()

        # Handle missing values
        X = X.fillna(X.median())

        print(f"   Features used: {', '.join(self._feature_cols)}")
        print(f"   Training on {len(X)} samples with {len(self._feature_cols)} features...")

        # Fit and predict in one step
        raw_preds = self.model.fit_predict(X)

        # sklearn returns: +1 = inlier (normal), -1 = outlier (anomaly)
        anomaly_flags = pd.Series(raw_preds == -1, index=df.index, name="if_anomaly")

        # anomaly_score: decision_function returns negative scores for anomalies
        # We negate it so HIGHER score = MORE anomalous (more intuitive)
        raw_scores = self.model.decision_function(X)
        anomaly_scores = pd.Series(-raw_scores, index=df.index, name="if_score")

        self._fitted = True

        if verbose:
            n_anomaly = anomaly_flags.sum()
            n_total = len(df)
            print(f"   Detected {n_anomaly} anomalies ({100*n_anomaly/n_total:.1f}%)")
            print(f"   Contamination setting: {self.contamination:.1%}")
            print("   ✅ Isolation Forest detection complete.")

        return {
            "anomaly_flags": anomaly_flags,
            "anomaly_scores": anomaly_scores,
            "raw_predictions": pd.Series(raw_preds, index=df.index),
        }

    def score_samples(self, df: pd.DataFrame) -> pd.Series:
        """Score new samples using a fitted model (for online detection)."""
        if not self._fitted:
            raise RuntimeError("Call fit_detect() first.")
        X = df[self._feature_cols].fillna(df[self._feature_cols].median())
        scores = -self.model.decision_function(X)
        return pd.Series(scores, index=df.index)

    def save(self, path: str):
        """Persist the fitted model to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "feature_cols": self._feature_cols}, path)
        print(f"   💾 Isolation Forest model saved to {path}")

    @classmethod
    def load(cls, path: str) -> "IsolationForestDetector":
        """Load a previously saved model."""
        state = joblib.load(path)
        obj = cls()
        obj.model = state["model"]
        obj._feature_cols = state["feature_cols"]
        obj._fitted = True
        return obj
