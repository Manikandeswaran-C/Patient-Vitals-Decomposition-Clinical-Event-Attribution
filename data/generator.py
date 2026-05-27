"""
data/generator.py — Synthetic Physiological Data Generator
============================================================
Generates realistic multi-vital ICU time-series with:
  - Circadian (24-hour) rhythms per vital sign
  - Physiological inter-signal correlations (HR↑ when SpO2↓, etc.)
  - Realistic Gaussian measurement noise
  - Injected anomalies: spike, drift, combined crisis events

BEGINNER EXPLANATION:
    We create "fake patient data" that behaves like real ICU monitoring.
    Useful because real ICU data (MIMIC-IV, eICU) requires ethics approval.
    We inject known anomalies so we can verify our pipeline detects them.
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional, List, Dict
import warnings

from config import DATA_GENERATION, VITAL_SIGNS


class PhysioDataGenerator:
    """
    Generates synthetic multivariate physiological time-series data.

    Usage:
        gen = PhysioDataGenerator(patient_id="P001", seed=42)
        df = gen.generate()
        print(df.head())
        # Columns: heart_rate, systolic_bp, diastolic_bp, spo2,
        #          respiratory_rate, temperature, etco2,
        #          anomaly_label, patient_id
    """

    def __init__(
        self,
        patient_id: str = "P001",
        seed: Optional[int] = None,
        duration_hours: Optional[int] = None,
        sampling_interval_seconds: Optional[int] = None,
    ):
        self.patient_id = patient_id
        self.seed = seed if seed is not None else DATA_GENERATION["random_seed"]
        self.rng = np.random.default_rng(self.seed)
        self.duration_hours = duration_hours or DATA_GENERATION["duration_hours"]
        self.sampling_interval = (
            sampling_interval_seconds or DATA_GENERATION["sampling_interval_seconds"]
        )
        self.n_samples = int(self.duration_hours * 3600 / self.sampling_interval)
        self._anomaly_mask = np.zeros(self.n_samples, dtype=int)
        self._anomaly_details: List[Dict] = []

    def generate(self) -> pd.DataFrame:
        """Run the full generation pipeline and return a labeled DataFrame."""
        print(f"\n🏥 Generating synthetic ICU data for patient {self.patient_id}...")
        print(f"   Duration : {self.duration_hours}h  ({self.n_samples} samples)")
        print(f"   Interval : {self.sampling_interval}s per sample")

        timestamps = self._create_timestamps()
        signals = self._generate_base_signals()
        signals = self._add_noise(signals)
        signals = self._inject_anomalies(signals)

        df = pd.DataFrame(signals, index=timestamps)
        df["anomaly_label"] = self._anomaly_mask
        df["patient_id"] = self.patient_id
        df = self._clip_to_physiological_limits(df)

        print(f"   ✅ Generated {len(df)} samples with "
              f"{self._anomaly_mask.sum()} anomaly points "
              f"({100*self._anomaly_mask.mean():.1f}%)")
        return df

    def get_anomaly_details(self) -> pd.DataFrame:
        """Returns a DataFrame describing each injected anomaly event."""
        return pd.DataFrame(self._anomaly_details)

    def _create_timestamps(self) -> pd.DatetimeIndex:
        start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        return pd.date_range(
            start=start,
            periods=self.n_samples,
            freq=f"{self.sampling_interval}s"
        )

    def _circadian_factor(self, phase_shift: float = 0.0) -> np.ndarray:
        """Asymmetric 24-hour circadian factor mimicking body temperature / alert diurnal curves."""
        t = np.linspace(0, 2 * np.pi * self.duration_hours / 24, self.n_samples) + phase_shift
        # A combination of sines creates a rapid morning rise, midday plateau, and night drop
        factor = np.sin(t) + 0.3 * np.sin(2 * t) - 0.1 * np.cos(3 * t)
        # Normalize to range [-1, 1]
        return factor / (factor.max() - factor.min()) * 2

    def _generate_base_signals(self) -> Dict[str, np.ndarray]:
        """
        Generate all vitals with realistic physiology and feedback loops:
        - Asymmetric circadian modulation
        - Shared slow drift (physiological baseline shifts)
        - Physiological Coupling (Baroreflex, Thermogenic HR shift, Hypoxic RR drive)
        """
        n = self.n_samples
        circadian = self._circadian_factor()

        # Shared slow random walk (representing underlying patient state change)
        shared_state = np.cumsum(self.rng.normal(0, 0.006, n))
        shared_state -= shared_state.mean()

        # ── Temperature (36.5–37.5 °C) ─────────────────────────────
        temperature = (
            37.0
            + 0.5 * self._circadian_factor(-np.pi / 3)
            + 0.2 * shared_state
        )

        # ── Heart Rate (60–100 bpm) with Thermogenic effect ────────
        # 10 bpm increase per 1°C temperature increase (clinical rule)
        temp_effect = np.clip(temperature - 37.0, 0, None) * 10.0
        heart_rate = (
            75
            + 8 * circadian
            + 6 * shared_state
            + temp_effect
        )

        # ── Systolic BP (90–140 mmHg) ──────────────────────────────
        systolic_bp = (
            120
            + 10 * circadian
            + 8 * shared_state
            + 0.4 * (heart_rate - 75)   # HR-BP coupling (cardiac output)
        )

        # ── Diastolic BP (60–90 mmHg) ──────────────────────────────
        diastolic_bp = (
            80
            + 6 * self._circadian_factor(0.1)
            + 4 * shared_state
            + 0.25 * (heart_rate - 75)
        )

        # Baroreflex Feedback loop: low BP causes compensatory tachycardia
        bp_drop = np.clip(95 - systolic_bp, 0, None)
        heart_rate += 0.8 * bp_drop

        # ── SpO2 (95–100%) ───────────────────────────────
        spo2 = (
            98.5
            - 0.4 * self._circadian_factor(np.pi)
            - 1.2 * shared_state
        )

        # ── Respiratory Rate (12–20 breaths/min) with Hypoxic Drive ─
        # Low SpO2 causes compensatory hyperventilation
        hypoxic_drive = np.clip(94.5 - spo2, 0, None) * 4.0
        respiratory_rate = (
            15
            + 2 * self._circadian_factor(np.pi)
            + 2.5 * shared_state
            + hypoxic_drive
        )

        # ── End-Tidal CO2 (35–45 mmHg) — ventilation-perfusion relation
        etco2 = (
            40
            - 0.9 * (respiratory_rate - 15)
            + 1.5 * self._circadian_factor(0.4)
        )

        return {
            "heart_rate": heart_rate,
            "systolic_bp": systolic_bp,
            "diastolic_bp": diastolic_bp,
            "spo2": spo2,
            "respiratory_rate": respiratory_rate,
            "temperature": temperature,
            "etco2": etco2,
        }

    def _add_noise(self, signals: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Add realistic Gaussian measurement noise per vital."""
        noise_stds = {
            "heart_rate": 1.5, "systolic_bp": 2.5, "diastolic_bp": 1.5,
            "spo2": 0.25, "respiratory_rate": 0.4, "temperature": 0.04, "etco2": 0.8,
        }
        return {
            vital: signal + self.rng.normal(0, noise_stds.get(vital, 1.0), len(signal))
            for vital, signal in signals.items()
        }

    def _inject_anomalies(self, signals: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Inject realistic clinical anomaly events and measurement artifacts:
          SPIKE        — sudden brief deviation (arrhythmia, clinical event)
          DRIFT        — gradual sustained shift (sepsis, hyperthermia)
          DROPOUT      — sensor disconnect (drop to 0/NaN flatline)
          NOISE_BURST  — motion artifact (wild variance burst)
          COMBINED     — multi-vital crisis (hemodynamic shock)
        """
        anomaly_types = DATA_GENERATION["anomaly_types"]
        n = self.n_samples
        events = []

        # 1. Spikes
        if "spike" in anomaly_types:
            for _ in range(self.rng.integers(2, 4)):
                start = int(self.rng.integers(n // 10, 9 * n // 10))
                duration = int(self.rng.integers(1, 4))
                vital = str(self.rng.choice(list(signals.keys()), 1)[0])
                mag = float(self.rng.uniform(3.5, 6)) * float(self.rng.choice([-1, 1]))
                events.append({"type": "spike", "start": start, "end": start + duration, "vital": vital, "magnitude": mag})

        # 2. Drifts
        if "drift" in anomaly_types:
            for _ in range(self.rng.integers(2, 3)):
                start = int(self.rng.integers(n // 8, 7 * n // 8))
                duration = int(self.rng.integers(25, 50))
                vital = str(self.rng.choice(list(signals.keys()), 1)[0])
                mag = float(self.rng.uniform(2.5, 4.5)) * float(self.rng.choice([-1, 1]))
                events.append({"type": "drift", "start": start, "end": min(start + duration, n - 1), "vital": vital, "magnitude": mag})

        # 3. Sensor Dropouts (Flatline / Disconnect)
        if "dropout" in anomaly_types:
            for _ in range(self.rng.integers(1, 3)):
                start = int(self.rng.integers(n // 8, 7 * n // 8))
                duration = int(self.rng.integers(5, 15))  # 5-15 mins disconnect
                vital = str(self.rng.choice(["spo2", "heart_rate", "etco2"], 1)[0])
                events.append({"type": "dropout", "start": start, "end": min(start + duration, n - 1), "vital": vital})

        # 4. Noise Bursts (Motion Artifact)
        if "noise_burst" in anomaly_types:
            for _ in range(self.rng.integers(1, 3)):
                start = int(self.rng.integers(n // 8, 7 * n // 8))
                duration = int(self.rng.integers(8, 20))  # 8-20 mins movement
                vital = str(self.rng.choice(["heart_rate", "systolic_bp", "respiratory_rate"], 1)[0])
                events.append({"type": "noise_burst", "start": start, "end": min(start + duration, n - 1), "vital": vital})

        # 5. Combined Crises
        if "combined" in anomaly_types:
            for _ in range(self.rng.integers(1, 2)):
                start = int(self.rng.integers(n // 5, 4 * n // 5))
                duration = int(self.rng.integers(15, 30))
                events.append({
                    "type": "combined",
                    "start": start,
                    "end": min(start + duration, n - 1),
                    "vitals": {
                        "heart_rate": 3.8,
                        "spo2": -3.2,
                        "respiratory_rate": 3.5,
                        "systolic_bp": -3.5,
                    }
                })

        augmented = {k: v.copy() for k, v in signals.items()}

        for event in events:
            s, e = int(event["start"]), int(event["end"])
            # Flatline dropouts and noise bursts are marked as artifacts, which we also record
            self._anomaly_mask[s:e] = 1

            if event["type"] == "spike":
                vital = event["vital"]
                std = augmented[vital].std()
                augmented[vital][s:e] += event["magnitude"] * std
                self._anomaly_details.append({
                    "start": s, "end": e, "type": "spike",
                    "vital": vital, "magnitude_std": event["magnitude"]
                })
            elif event["type"] == "drift":
                vital = event["vital"]
                std = augmented[vital].std()
                ramp = np.linspace(0, event["magnitude"] * std, e - s)
                augmented[vital][s:e] += ramp
                self._anomaly_details.append({
                    "start": s, "end": e, "type": "drift",
                    "vital": vital, "magnitude_std": event["magnitude"]
                })
            elif event["type"] == "dropout":
                # Flatline value to zero or NaN (which the cleaner will interpolate/detect)
                vital = event["vital"]
                # 80% chance of flatlining to zero, 20% to NaN
                if self.rng.random() > 0.2:
                    augmented[vital][s:e] = 0.0
                else:
                    augmented[vital][s:e] = np.nan
                self._anomaly_details.append({
                    "start": s, "end": e, "type": "dropout",
                    "vital": vital, "magnitude_std": "sensor_disconnect"
                })
            elif event["type"] == "noise_burst":
                # Wild motion noise added
                vital = event["vital"]
                std = augmented[vital].std()
                noise = self.rng.normal(0, std * 2.5, e - s)
                augmented[vital][s:e] += noise
                self._anomaly_details.append({
                    "start": s, "end": e, "type": "noise_burst",
                    "vital": vital, "magnitude_std": "motion_noise"
                })
            elif event["type"] == "combined":
                for vital, mag in event["vitals"].items():
                    std = augmented[vital].std()
                    augmented[vital][s:e] += mag * std
                self._anomaly_details.append({
                    "start": s, "end": e, "type": "combined",
                    "vital": "+".join(event["vitals"].keys()), "magnitude_std": "multi"
                })

        print(f"   💉 Injected {len(events)} anomaly/artifact events")
        return augmented

    def _clip_to_physiological_limits(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure no vital goes outside physiologically plausible bounds."""
        for col in [c for c in df.columns if c in VITAL_SIGNS]:
            cfg = VITAL_SIGNS[col]
            df[col] = df[col].clip(
                lower=cfg["critical_low"] * 0.8,
                upper=cfg["critical_high"] * 1.1,
            )
        return df
