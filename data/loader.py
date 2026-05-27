"""
data/loader.py — Real Dataset Loader
=====================================
Supports loading from MIMIC-IV, eICU, WESAD, PPG-DaLiA, and CSV files.
All loaders return a standardized DataFrame matching the generator output format.

BEGINNER NOTE:
    MIMIC-IV and eICU require registration at physionet.org.
    If you have access, place the CSV files in data/raw/ and call the loader.
    Otherwise, use the PhysioDataGenerator for synthetic data.
"""

import numpy as np
import pandas as pd
import os
from pathlib import Path
from typing import Optional, List
import warnings

from config import VITAL_SIGNS


# Standard column mapping: maps dataset-specific names → our internal names
MIMIC_COLUMN_MAP = {
    "heart rate": "heart_rate",
    "non-invasive blood pressure systolic": "systolic_bp",
    "non-invasive blood pressure diastolic": "diastolic_bp",
    "spo2": "spo2",
    "respiratory rate": "respiratory_rate",
    "temperature fahrenheit": "temperature_f",  # will convert to °C
    "temperature celsius": "temperature",
}

EICU_COLUMN_MAP = {
    "heartrate": "heart_rate",
    "systemicsystolic": "systolic_bp",
    "systemicdiastolic": "diastolic_bp",
    "sao2": "spo2",
    "respiratoryrate": "respiratory_rate",
    "temperature": "temperature",
}


class PhysioDataLoader:
    """
    Loads physiological time-series from various clinical datasets.

    Usage:
        # From CSV (any format, column mapping required)
        loader = PhysioDataLoader()
        df = loader.from_csv("my_data.csv", column_map={"HR": "heart_rate", ...})

        # From MIMIC-IV (requires physionet.org access)
        df = loader.from_mimic("path/to/mimic_vitals.csv", patient_id="10001")

        # From eICU
        df = loader.from_eicu("path/to/vitalPeriodic.csv", patient_id=141168)
    """

    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)

    # ──────────────────────────────────────────────────────────────────
    # UNIVERSAL CSV LOADER
    # ──────────────────────────────────────────────────────────────────

    def from_csv(
        self,
        filepath: str,
        column_map: Optional[dict] = None,
        timestamp_col: str = "timestamp",
        patient_id_col: Optional[str] = None,
        patient_id: Optional[str] = None,
        resample_freq: str = "1min",
    ) -> pd.DataFrame:
        """
        Load from any CSV file with a flexible column mapping.

        Args:
            filepath: Path to the CSV file.
            column_map: Dict mapping CSV column names → internal vital names.
                        Example: {"HR": "heart_rate", "SpO2": "spo2"}
            timestamp_col: Name of the timestamp column in the CSV.
            patient_id_col: Column containing patient IDs (to filter by).
            patient_id: If patient_id_col is given, filter to this patient.
            resample_freq: Resample to this frequency ("1min", "5min", etc.)

        Returns:
            Standardized DataFrame with DatetimeIndex.
        """
        print(f"📂 Loading data from: {filepath}")
        df = pd.read_csv(filepath)

        # Filter to specific patient if requested
        if patient_id_col and patient_id:
            df = df[df[patient_id_col] == patient_id].copy()
            if df.empty:
                raise ValueError(f"No data found for patient {patient_id}")

        # Parse timestamps
        if timestamp_col in df.columns:
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])
            df = df.set_index(timestamp_col).sort_index()

        # Apply column mapping
        if column_map:
            df = df.rename(columns=column_map)

        # Keep only recognized vital sign columns
        valid_cols = [c for c in df.columns if c in VITAL_SIGNS]
        df = df[valid_cols]

        # Resample to uniform frequency
        df = df.resample(resample_freq).mean()

        # Convert Fahrenheit to Celsius if needed
        if "temperature_f" in df.columns:
            df["temperature"] = (df["temperature_f"] - 32) * 5 / 9
            df = df.drop(columns=["temperature_f"])

        df = self._standardize_output(df, patient_id or "unknown")
        print(f"   ✅ Loaded {len(df)} samples, {len(df.columns)} vitals")
        return df

    # ──────────────────────────────────────────────────────────────────
    # MIMIC-IV LOADER
    # ──────────────────────────────────────────────────────────────────

    def from_mimic(
        self,
        vitals_path: str,
        patient_id: Optional[str] = None,
        resample_freq: str = "1min",
    ) -> pd.DataFrame:
        """
        Load from MIMIC-IV chartevents or vitals summary CSV.

        Expected columns: ['subject_id', 'charttime', 'label', 'valuenum']
        This is the "long format" typical of MIMIC-IV chartevents.

        MIMIC-IV Access: https://physionet.org/content/mimiciv/
        """
        print(f"🏥 Loading MIMIC-IV data from: {vitals_path}")
        df = pd.read_csv(vitals_path, low_memory=False)

        # Filter patient
        if patient_id and "subject_id" in df.columns:
            df = df[df["subject_id"] == int(patient_id)]

        # Pivot from long to wide format
        df["charttime"] = pd.to_datetime(df["charttime"])
        df["label"] = df["label"].str.lower().str.strip()

        pivoted = df.pivot_table(
            index="charttime", columns="label", values="valuenum", aggfunc="mean"
        )

        # Apply MIMIC column mapping
        pivoted = pivoted.rename(columns=MIMIC_COLUMN_MAP)

        # Keep valid vitals
        valid_cols = [c for c in pivoted.columns if c in VITAL_SIGNS]
        pivoted = pivoted[valid_cols].sort_index()

        # Resample to uniform frequency
        pivoted = pivoted.resample(resample_freq).mean()

        df_out = self._standardize_output(pivoted, patient_id or "mimic")
        print(f"   ✅ Loaded {len(df_out)} MIMIC samples")
        return df_out

    # ──────────────────────────────────────────────────────────────────
    # eICU LOADER
    # ──────────────────────────────────────────────────────────────────

    def from_eicu(
        self,
        vital_periodic_path: str,
        patient_id: Optional[int] = None,
        resample_freq: str = "5min",
    ) -> pd.DataFrame:
        """
        Load from eICU vitalPeriodic.csv (the periodic vital signs table).

        eICU Access: https://physionet.org/content/eicu-crd/

        Expected columns: ['patientunitstayid', 'observationoffset', 'heartrate', ...]
        """
        print(f"🏥 Loading eICU data from: {vital_periodic_path}")
        df = pd.read_csv(vital_periodic_path, low_memory=False)

        if patient_id and "patientunitstayid" in df.columns:
            df = df[df["patientunitstayid"] == patient_id]

        df = df.rename(columns=EICU_COLUMN_MAP)

        # Create timestamps from observationoffset (minutes from ICU admission)
        if "observationoffset" in df.columns:
            base_time = pd.Timestamp("2100-01-01")
            df["timestamp"] = base_time + pd.to_timedelta(
                df["observationoffset"], unit="min"
            )
            df = df.set_index("timestamp").sort_index()

        valid_cols = [c for c in df.columns if c in VITAL_SIGNS]
        df = df[valid_cols]
        df = df.resample(resample_freq).mean()

        df_out = self._standardize_output(df, str(patient_id) or "eicu")
        print(f"   ✅ Loaded {len(df_out)} eICU samples")
        return df_out

    # ──────────────────────────────────────────────────────────────────
    # HELPERS
    # ──────────────────────────────────────────────────────────────────

    def _standardize_output(self, df: pd.DataFrame, patient_id: str) -> pd.DataFrame:
        """
        Ensure the output DataFrame matches our standard format:
        - DatetimeIndex
        - Only recognized vital sign columns
        - 'patient_id' column
        - 'anomaly_label' column (0 = unknown for real data)
        """
        df = df.copy()
        valid_cols = [c for c in df.columns if c in VITAL_SIGNS]
        df = df[valid_cols]
        df["patient_id"] = patient_id
        df["anomaly_label"] = 0  # Unknown for real data
        return df

    def list_available_datasets(self) -> List[str]:
        """List CSV files in the data/raw directory."""
        if not self.data_dir.exists():
            return []
        return [str(f) for f in self.data_dir.glob("*.csv")]
