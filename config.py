"""
config.py — Central Configuration for the PhysioAnomalyPipeline
================================================================
All tunable parameters are defined here so you can easily experiment
without touching the core logic files.

Beginner tip:
    Think of this as the "settings panel" for the entire pipeline.
    Change values here to see how the pipeline behaves differently.
"""

# ─────────────────────────────────────────────────────────────
# VITAL SIGN DEFINITIONS
# Each vital sign has:
#   - unit       : measurement unit
#   - normal_low : lower bound of the clinical normal range
#   - normal_high: upper bound of the clinical normal range
#   - color      : color used in all plots for this signal
#   - description: plain-English description
# ─────────────────────────────────────────────────────────────
VITAL_SIGNS = {
    "heart_rate": {
        "unit": "bpm",
        "normal_low": 60,
        "normal_high": 100,
        "color": "#e74c3c",          # red
        "description": "Heart Rate — number of heartbeats per minute.",
        "critical_low": 40,
        "critical_high": 150,
    },
    "systolic_bp": {
        "unit": "mmHg",
        "normal_low": 90,
        "normal_high": 140,
        "color": "#e67e22",          # orange
        "description": "Systolic Blood Pressure — peak pressure in arteries.",
        "critical_low": 70,
        "critical_high": 200,
    },
    "diastolic_bp": {
        "unit": "mmHg",
        "normal_low": 60,
        "normal_high": 90,
        "color": "#f39c12",          # amber
        "description": "Diastolic Blood Pressure — resting arterial pressure.",
        "critical_low": 40,
        "critical_high": 130,
    },
    "spo2": {
        "unit": "%",
        "normal_low": 95,
        "normal_high": 100,
        "color": "#3498db",          # blue
        "description": "SpO2 — peripheral oxygen saturation.",
        "critical_low": 85,
        "critical_high": 100,
    },
    "respiratory_rate": {
        "unit": "breaths/min",
        "normal_low": 12,
        "normal_high": 20,
        "color": "#2ecc71",          # green
        "description": "Respiratory Rate — breaths per minute.",
        "critical_low": 6,
        "critical_high": 40,
    },
    "temperature": {
        "unit": "°C",
        "normal_low": 36.5,
        "normal_high": 37.5,
        "color": "#9b59b6",          # purple
        "description": "Core Body Temperature — thermoregulation indicator.",
        "critical_low": 35.0,
        "critical_high": 40.0,
    },
    "etco2": {
        "unit": "mmHg",
        "normal_low": 35,
        "normal_high": 45,
        "color": "#1abc9c",          # teal
        "description": "End-Tidal CO2 — exhaled CO2 indicating ventilation.",
        "critical_low": 20,
        "critical_high": 60,
    },
}

# ─────────────────────────────────────────────────────────────
# DATA GENERATION SETTINGS
# ─────────────────────────────────────────────────────────────
DATA_GENERATION = {
    # How many hours of patient monitoring to simulate
    "duration_hours": 24,

    # Sampling interval in seconds (60 = 1 reading per minute)
    "sampling_interval_seconds": 60,

    # Random seed for reproducibility (change to get different patients)
    "random_seed": 42,

    # Fraction of time points that will contain injected anomalies
    "anomaly_fraction": 0.05,

    # Types of anomalies to inject
    # Options: "spike", "drift", "dropout", "noise_burst", "combined"
    "anomaly_types": ["spike", "drift", "dropout", "noise_burst", "combined"],

    # Realistic physiological noise level (standard deviation as fraction of signal)
    "noise_level": 0.02,
}

# ─────────────────────────────────────────────────────────────
# PREPROCESSING SETTINGS
# ─────────────────────────────────────────────────────────────
PREPROCESSING = {
    # Window size for rolling statistics (in samples)
    "rolling_window": 10,

    # Maximum fraction of missing values allowed before dropping a column
    "max_missing_fraction": 0.3,

    # Interpolation method for filling missing values
    # Options: "linear", "cubic", "forward_fill"
    "interpolation_method": "linear",

    # Normalization strategy
    # Options: "zscore", "minmax", "robust"
    "normalization": "robust",
}

# ─────────────────────────────────────────────────────────────
# DECOMPOSITION SETTINGS
# ─────────────────────────────────────────────────────────────
DECOMPOSITION = {
    # STL decomposition period (in samples)
    # For 1-sample-per-minute data: 60 = 1 hour period
    "stl_period": 60,

    # Wavelet family to use for DWT
    # Options: "db4" (Daubechies), "haar", "sym4"
    "wavelet_family": "db4",

    # Number of wavelet decomposition levels
    "wavelet_levels": 4,
}

# ─────────────────────────────────────────────────────────────
# ANOMALY DETECTION SETTINGS
# ─────────────────────────────────────────────────────────────
DETECTION = {
    # --- Statistical Detector ---
    # Number of standard deviations to consider an anomaly
    "zscore_threshold": 3.0,

    # IQR multiplier (standard = 1.5, stricter = 1.0)
    "iqr_multiplier": 1.5,

    # --- Isolation Forest ---
    "isolation_forest": {
        "n_estimators": 200,
        "contamination": 0.05,   # Expected fraction of anomalies
        "random_state": 42,
        "max_features": 1.0,
    },

    # --- LSTM Autoencoder ---
    "lstm_autoencoder": {
        "sequence_length": 30,   # Look-back window (30 minutes)
        "hidden_dim": 64,
        "num_layers": 2,
        "latent_dim": 16,
        "epochs": 50,
        "batch_size": 32,
        "learning_rate": 0.001,
        "reconstruction_threshold_percentile": 95,
    },

    # --- Ensemble ---
    # Minimum number of detectors that must agree to flag an anomaly
    "ensemble_min_votes": 2,
}

# ─────────────────────────────────────────────────────────────
# ATTRIBUTION / EXPLANATION SETTINGS
# ─────────────────────────────────────────────────────────────
ATTRIBUTION = {
    # Number of background samples for SHAP KernelExplainer
    "shap_background_samples": 100,

    # Number of anomaly samples to explain (too many = slow)
    "max_anomalies_to_explain": 50,

    # Feature importance aggregation method
    # Options: "mean_abs", "median_abs"
    "importance_aggregation": "mean_abs",
}

# ─────────────────────────────────────────────────────────────
# RELATIONSHIP ANALYSIS SETTINGS
# ─────────────────────────────────────────────────────────────
RELATIONSHIPS = {
    # Correlation method
    # Options: "pearson" (linear), "spearman" (monotonic), "kendall"
    "correlation_method": "pearson",

    # Significance threshold for p-values
    "significance_level": 0.05,

    # Granger causality settings
    "granger": {
        "max_lag": 10,           # Maximum lag (in samples) to test
        "significance_level": 0.05,
    },

    # Cross-correlation settings
    "cross_correlation": {
        "max_lag": 30,           # Maximum lag to compute
    },
}

# ─────────────────────────────────────────────────────────────
# VISUALIZATION SETTINGS
# ─────────────────────────────────────────────────────────────
VISUALIZATION = {
    "figure_dpi": 150,
    "figure_width": 16,
    "figure_height": 10,
    "save_plots": True,
    "output_dir": "outputs/plots",
    "style": "seaborn-v0_8-darkgrid",

    # Color for anomaly markers on plots
    "anomaly_color": "#ff4757",
    "anomaly_marker": "v",
    "anomaly_marker_size": 100,
}

# ─────────────────────────────────────────────────────────────
# OUTPUT SETTINGS
# ─────────────────────────────────────────────────────────────
OUTPUT = {
    "report_dir": "outputs/reports",
    "data_dir": "outputs/data",
    "model_dir": "outputs/models",
}
