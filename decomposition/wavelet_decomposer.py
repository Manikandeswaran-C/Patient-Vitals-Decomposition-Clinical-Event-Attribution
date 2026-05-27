"""
decomposition/wavelet_decomposer.py — Wavelet Signal Decomposition
===================================================================
Alternative to STL that decomposes signals into FREQUENCY BANDS
rather than trend/seasonal/residual.

WHY WAVELETS?
    STL works well for regular periodic patterns (daily cycles).
    Wavelets excel at detecting TRANSIENT events — brief bursts
    of abnormal activity at specific frequency scales.

    For example:
    - Level 1 detail = high-frequency noise (Hz range)
    - Level 2 detail = minute-to-minute variation
    - Level 3 detail = 5–15 minute patterns
    - Approximation = slow drift (trend-like)

BEGINNER ANALOGY:
    Wavelets are like a musical equalizer. Instead of just hearing
    the total sound (signal), you can see exactly how much bass,
    mid, and treble (different frequency components) are present
    at every moment in time.
"""

import numpy as np
import pandas as pd
import pywt
from typing import Dict, List, Optional, Tuple
import warnings

from config import DECOMPOSITION, VITAL_SIGNS


class WaveletDecomposer:
    """
    Applies Discrete Wavelet Transform (DWT) to physiological signals.

    Usage:
        wd = WaveletDecomposer(wavelet="db4", levels=4)
        results = wd.decompose(clean_df)
        energy_df = wd.energy_distribution()    # energy per frequency band
        anomaly_scores = wd.detail_anomaly_scores()
    """

    def __init__(
        self,
        wavelet: Optional[str] = None,
        levels: Optional[int] = None,
    ):
        """
        Args:
            wavelet: Wavelet family. "db4" (Daubechies 4) is standard for biomedical.
                     Options: "db4", "db8", "haar", "sym4", "coif2"
            levels: Number of decomposition levels. More levels = more frequency bands.
        """
        self.wavelet = wavelet or DECOMPOSITION["wavelet_family"]
        self.levels = levels or DECOMPOSITION["wavelet_levels"]
        self._coefficients: Dict[str, List[np.ndarray]] = {}
        self._signals: Dict[str, np.ndarray] = {}

    def decompose(self, df: pd.DataFrame, verbose: bool = True) -> Dict:
        """
        Apply multilevel DWT to each vital sign.

        Args:
            df: DataFrame with vital sign columns.
            verbose: Print summary.

        Returns:
            Dict: {vital → {'approximation': arr, 'details': [arr1, arr2, ...]}}
            - approximation = low-frequency / trend component
            - details[0] = finest (highest frequency) detail
            - details[-1] = coarsest (lowest frequency) detail
        """
        print("\n🌊 Wavelet Decomposition (multi-scale frequency analysis)")
        vital_cols = [c for c in df.columns if c in VITAL_SIGNS]
        results = {}

        for col in vital_cols:
            signal = df[col].fillna(df[col].median()).values
            self._signals[col] = signal

            # Maximum usable levels for this signal length
            max_levels = pywt.dwt_max_level(len(signal), self.wavelet)
            n_levels = min(self.levels, max_levels)

            # Perform multilevel DWT
            # coeffs[0] = approximation (cA), coeffs[1:] = details (cD)
            coeffs = pywt.wavedec(signal, self.wavelet, level=n_levels)
            self._coefficients[col] = coeffs

            results[col] = {
                "approximation": coeffs[0],                # Low-frequency trend
                "details": coeffs[1:],                     # High-to-low frequency
                "n_levels": n_levels,
                "wavelet": self.wavelet,
            }

        if verbose:
            print(f"   Wavelet: {self.wavelet}, Levels: {self.levels}")
            print(f"   Decomposed {len(vital_cols)} vital signs")
            print("   ✅ Wavelet decomposition complete.")

        return results

    def reconstruct(self, vital: str, zero_detail_levels: Optional[List[int]] = None) -> np.ndarray:
        """
        Reconstruct the signal, optionally zeroing out specific detail levels.

        This allows filtering specific frequency bands.
        Example: zero_detail_levels=[1,2] removes the highest-frequency noise.

        Args:
            vital: Name of the vital sign.
            zero_detail_levels: List of detail levels (1=finest) to zero out.

        Returns:
            Reconstructed signal array.
        """
        if vital not in self._coefficients:
            raise KeyError(f"Vital '{vital}' not decomposed yet. Call decompose() first.")

        coeffs = list(self._coefficients[vital])  # copy

        if zero_detail_levels:
            for level in zero_detail_levels:
                if 1 <= level <= len(coeffs) - 1:
                    coeffs[level] = np.zeros_like(coeffs[level])

        return pywt.waverec(coeffs, self.wavelet)[:len(self._signals[vital])]

    def energy_distribution(self) -> pd.DataFrame:
        """
        Compute the energy (variance) in each frequency band per vital.

        CLINICAL MEANING:
            Most energy should be in the approximation (slow trend) and
            low-detail levels. If high-detail levels have high energy →
            the signal is noisy or has transient anomalies.

        Returns:
            DataFrame with rows = vitals, columns = frequency bands.
        """
        rows = []
        for col, coeffs in self._coefficients.items():
            energies = [np.sum(c ** 2) for c in coeffs]
            total = sum(energies) or 1
            row = {"vital": col}
            row["approx_%"] = round(energies[0] / total * 100, 2)
            for i, e in enumerate(energies[1:], 1):
                row[f"detail_L{i}_%"] = round(e / total * 100, 2)
            rows.append(row)

        return pd.DataFrame(rows).set_index("vital")

    def detail_anomaly_scores(self, threshold_multiplier: float = 3.0) -> pd.DataFrame:
        """
        Compute per-sample anomaly scores from the finest detail coefficients.

        High-amplitude detail coefficients at the finest level correspond to
        sudden, sharp changes — typical of anomalies or artifacts.

        Args:
            threshold_multiplier: Scores above mean + N*std are flagged.

        Returns:
            DataFrame of anomaly scores per vital (aligned to original index).
        """
        scores = {}
        for col, coeffs in self._coefficients.items():
            # Use the finest detail (coeffs[1] = level 1)
            detail = coeffs[1] if len(coeffs) > 1 else coeffs[0]

            # Reconstruct to original signal length
            # Use only the finest detail band
            dummy = [np.zeros_like(c) for c in coeffs]
            dummy[1] = coeffs[1] if len(coeffs) > 1 else coeffs[0]
            detail_signal = pywt.waverec(dummy, self.wavelet)[:len(self._signals[col])]

            # Anomaly score = absolute value of detail signal
            score = np.abs(detail_signal)
            scores[col] = score

        return pd.DataFrame(scores)
