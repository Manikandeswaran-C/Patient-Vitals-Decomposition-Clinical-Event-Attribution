"""
detection/ensemble.py — Ensemble Anomaly Detector with Cohen's Kappa
=====================================================================
Combines statistical, Isolation Forest, and LSTM predictions using voting.
Also computes Cohen's Kappa to measure AGREEMENT between detectors.

WHY ENSEMBLE?
    No single detector is perfect:
    - Statistical: High false-positives during transitions
    - Isolation Forest: Can miss slow drifts
    - LSTM: Computationally expensive, needs training data
    Combining them reduces both false positives and false negatives.

COHEN'S KAPPA:
    Measures agreement between two detectors beyond chance.
    κ = (P_observed - P_chance) / (1 - P_chance)
    κ > 0.8 = near-perfect agreement
    κ = 0   = agreement is random
    κ < 0   = detectors systematically disagree (interesting!)
"""

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from typing import Dict, List, Optional


class EnsembleDetector:
    """
    Combines multiple anomaly detectors via majority voting.

    Usage:
        ens = EnsembleDetector(min_votes=2)
        results = ens.combine(stat_flags, if_flags, lstm_flags)
        print(results["ensemble_flags"])
        print(results["kappa_matrix"])
    """

    def __init__(self, min_votes: int = 2):
        """
        Args:
            min_votes: Minimum number of detectors that must agree to flag anomaly.
                       With 3 detectors: min_votes=2 → majority vote.
        """
        self.min_votes = min_votes
        self._detector_names: List[str] = []

    def combine(self, verbose: bool = True, **detector_flags) -> Dict:
        """
        Combine boolean anomaly flag Series from multiple detectors.

        Args:
            verbose: Print summary.
            **detector_flags: Named boolean Series, e.g.:
                              statistical=stat_series,
                              isolation_forest=if_series,
                              lstm=lstm_series

        Returns:
            Dict with:
              - 'ensemble_flags'   : final boolean anomaly Series
              - 'vote_counts'      : int Series (how many detectors flagged each point)
              - 'detector_results' : DataFrame with each detector's flags
              - 'kappa_matrix'     : pairwise Cohen's Kappa DataFrame
              - 'agreement_report' : human-readable agreement statistics
        """
        print("\n🗳️  Step 4d: Ensemble & Agreement Analysis")

        self._detector_names = list(detector_flags.keys())
        flags_df = pd.DataFrame(detector_flags)

        # Cast to bool to be safe
        flags_df = flags_df.astype(bool)

        # Vote count per time point
        vote_counts = flags_df.sum(axis=1)

        # Final ensemble decision: anomaly if >= min_votes detectors agree
        ensemble_flags = vote_counts >= self.min_votes

        # Compute pairwise Cohen's Kappa
        kappa_matrix = self._compute_kappa_matrix(flags_df)

        # Agreement report
        agreement_report = self._build_agreement_report(flags_df, kappa_matrix)

        if verbose:
            n_total = len(flags_df)
            n_ensemble = ensemble_flags.sum()
            print(f"\n   Min votes required: {self.min_votes}/{len(self._detector_names)}")
            print(f"   Ensemble anomalies: {n_ensemble} / {n_total} "
                  f"({100*n_ensemble/n_total:.1f}%)")
            print(f"\n   Per-detector breakdown:")
            for col in flags_df.columns:
                cnt = flags_df[col].sum()
                print(f"     {col:<22}: {cnt} ({100*cnt/n_total:.1f}%)")
            print(f"\n   Pairwise Cohen's Kappa:")
            print(kappa_matrix.to_string())
            print("\n   ✅ Ensemble detection complete.")

        return {
            "ensemble_flags": ensemble_flags,
            "vote_counts": vote_counts,
            "detector_results": flags_df,
            "kappa_matrix": kappa_matrix,
            "agreement_report": agreement_report,
        }

    def _compute_kappa_matrix(self, flags_df: pd.DataFrame) -> pd.DataFrame:
        """Compute pairwise Cohen's Kappa for all detector pairs."""
        cols = flags_df.columns.tolist()
        kappa_data = {}
        for col_a in cols:
            kappa_data[col_a] = {}
            for col_b in cols:
                if col_a == col_b:
                    kappa_data[col_a][col_b] = 1.0
                else:
                    a = flags_df[col_a].astype(int).values
                    b = flags_df[col_b].astype(int).values
                    # Only compute if both have at least 2 unique values
                    if len(np.unique(a)) < 2 or len(np.unique(b)) < 2:
                        kappa_data[col_a][col_b] = np.nan
                    else:
                        kappa_data[col_a][col_b] = round(cohen_kappa_score(a, b), 3)
        return pd.DataFrame(kappa_data)

    def _build_agreement_report(
        self, flags_df: pd.DataFrame, kappa_matrix: pd.DataFrame
    ) -> Dict:
        """Build a human-readable agreement summary."""
        report = {"detector_counts": {}, "pairwise_kappa": {}, "interpretation": {}}

        for col in flags_df.columns:
            report["detector_counts"][col] = int(flags_df[col].sum())

        cols = flags_df.columns.tolist()
        for i, a in enumerate(cols):
            for b in cols[i+1:]:
                k = kappa_matrix.loc[a, b]
                pair = f"{a} vs {b}"
                report["pairwise_kappa"][pair] = k
                if np.isnan(k):
                    report["interpretation"][pair] = "Cannot compute (all same class)"
                elif k > 0.8:
                    report["interpretation"][pair] = "Near-perfect agreement"
                elif k > 0.6:
                    report["interpretation"][pair] = "Substantial agreement"
                elif k > 0.4:
                    report["interpretation"][pair] = "Moderate agreement"
                elif k > 0.2:
                    report["interpretation"][pair] = "Fair agreement"
                else:
                    report["interpretation"][pair] = "Slight or no agreement"

        return report
