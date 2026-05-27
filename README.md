# 🏥 Interpretable Physiological Time-Series Anomaly Attribution Pipeline

A **complete, beginner-friendly** end-to-end system for ICU and remote patient monitoring data.

## 🎯 Project Objective

This pipeline goes **beyond anomaly detection**. It:
- 📊 **Decomposes** physiological signals into trend, seasonal, and residual components
- 🔍 **Detects** abnormalities using statistical + ML + deep learning methods
- 🧠 **Explains WHY** abnormalities occur using SHAP attribution
- 🔗 **Identifies relationships** between vital signs (correlation + Granger causality)
- 📋 **Generates** clinically understandable insights in plain English

## 🏗️ Architecture

```
PhysioAnomalyPipeline/
├── data/              → Synthetic ICU data generation & loading
├── preprocessing/     → Signal cleaning, normalization, feature engineering
├── decomposition/     → STL + Wavelet signal decomposition
├── detection/         → Statistical, Isolation Forest, LSTM Autoencoder detectors
├── attribution/       → SHAP-based anomaly attribution & feature importance
├── relationships/     → Cross-correlation & Granger causality analysis
├── insights/          → Clinical interpretation & natural language explanations
├── visualization/     → Interactive matplotlib/plotly dashboards
├── pipeline/          → End-to-end orchestration
└── notebooks/         → Beginner-friendly Jupyter tutorial
```

## 🩺 Vital Signs Monitored

| Signal | Normal Range | Clinical Significance |
|--------|-------------|----------------------|
| Heart Rate (HR) | 60–100 bpm | Cardiac function |
| Systolic BP | 90–140 mmHg | Circulatory pressure |
| Diastolic BP | 60–90 mmHg | Vascular resistance |
| SpO2 | 95–100% | Oxygen saturation |
| Respiratory Rate | 12–20 breaths/min | Pulmonary function |
| Temperature | 36.5–37.5 °C | Metabolic/infection state |
| ETCO2 | 35–45 mmHg | Ventilation adequacy |

## 🚀 Quick Start (CLI)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the quickstart runner (with UTF-8 stream support)
python -X utf8 run_pipeline.py

# 3. Or run the main pipeline module directly
python -X utf8 -m pipeline.main_pipeline --patient-id P001

# 4. Open the Jupyter tutorial (recommended for beginners)
jupyter notebook notebooks/tutorial.ipynb
```

The CLI quick start will:
- generate a full 24h synthetic ICU record (`data.generator.PhysioDataGenerator`)
- run the entire pipeline (`pipeline.main_pipeline.PhysioAnomalyPipeline`)
- write plots under `outputs/plots/`
- write a plain-text clinical report under `outputs/reports/{patient_id}_report.txt`

## 🖥️ Interactive Dashboard (Streamlit UI)

For an interactive, clinician-friendly dashboard, launch the Streamlit app:

```bash
streamlit run app.py
```

From the sidebar you can:
- choose between **Synthetic Generator** and **Upload CSV File**
- tune anomaly detector thresholds (Z-score, Isolation Forest contamination, ensemble votes)
- optionally enable the **LSTM Autoencoder** detector (slower, uses PyTorch)

The UI exposes:
- a KPI overview (patient ID, anomaly count, stability badge, primary anomaly driver)
- clinical alert cards and recommendations
- interactive STL decomposition plots
- SHAP-based feature importance and anomaly tables
- coupling / causality visualizations and artifact-cleaning comparisons

## 📥 Using Your Own CSV Data

You can run the pipeline on your own vital-sign CSVs via:

### Option 1 — CLI

```bash
python -X utf8 -m pipeline.main_pipeline \
  --patient-id P001 \
  --data-path path/to/my_vitals.csv \
  --data-source csv
```

Then customize the column mapping inside your own script using `data.loader.PhysioDataLoader.from_csv`, or follow the commented example in `run_pipeline.py`.

### Option 2 — Streamlit UI

1. Start `streamlit run app.py`.
2. In the sidebar, select **Upload CSV File**.
3. Upload your CSV and map each column to the internal vital names using the dropdowns.
4. Click **Run Diagnostics Pipeline**.

Minimum expected columns (after mapping):
- `heart_rate`
- `systolic_bp`
- `diastolic_bp`
- `spo2`
- `respiratory_rate`
- `temperature`
- `etco2` (optional but recommended)

## 📂 Output Directory Structure

Running the pipeline will create an `outputs/` directory with the following structure:
```
outputs/
├── plots/
│   ├── 01_raw_signals.png          # Raw signals with anomaly markers
│   ├── 02_stl_{vital_name}.png     # STL decomposition plots per vital
│   ├── 03_zscore_timeline.png      # Rolling Z-score thresholds
│   ├── 04_shap_importance.png      # SHAP feature attribution
│   ├── 05_correlation_heatmap.png  # Contemporaneous correlations
│   ├── 06_anomaly_timeline.png     # Anomaly types over time
│   ├── 07_variance_decomposition.png# Trend/Seasonal/Residual variance %
│   ├── 08_lead_lag.png             # Cross-correlation lag functions
│   └── 09_ensemble_agreement.png   # Cohen's Kappa detector agreement
└── reports/
    └── {patient_id}_report.txt     # Plain-English clinical explanation report
```

## 📦 Requirements

- Python 3.8+
- See `requirements.txt` for all dependencies

## 🧩 Key Concepts (For Beginners)

### What is Signal Decomposition?
Breaking a signal (e.g., heart rate over time) into:
- **Trend**: Long-term direction (rising? falling?)
- **Seasonal**: Repeating patterns (day/night cycles)
- **Residual**: What's left — the "noise" or unexpected events

### What is Anomaly Detection?
Finding time points where measurements are "unusually" different from expected.

### What is Anomaly Attribution?
Figuring out **which vital signs caused** or **contributed most** to an anomaly.
We use SHAP (SHapley Additive exPlanations) — a game-theory-based method.

### What is Granger Causality?
Testing if one signal helps **predict** another signal. If HR predicts BP changes,
HR may "Granger-cause" BP in this patient's data.

## 📊 Output

The pipeline produces:
1. **Anomaly timeline** — when anomalies occurred
2. **SHAP attribution plots** — which vitals caused each anomaly
3. **Correlation heatmaps** — how vitals relate to each other
4. **Clinical report** — plain English explanation of findings
5. **Decomposition plots** — trend/seasonal/residual for each vital

## 🔬 Methods Used

| Task | Method | Why? |
|------|--------|------|
| Decomposition | STL (Seasonal-Trend-LOESS) | Robust to outliers, handles missing data |
| Decomposition | Discrete Wavelet Transform | Multi-scale frequency analysis |
| Detection | Z-score / IQR | Simple, interpretable baseline |
| Detection | Isolation Forest | Handles multivariate anomalies |
| Detection | LSTM Autoencoder | Captures temporal dependencies |
| Attribution | SHAP TreeExplainer | Exact, model-agnostic explanations |
| Relationships | Pearson/Spearman Correlation | Linear/monotonic relationships |
| Relationships | Granger Causality | Temporal predictive relationships |
| Insights | Rule-based NLG | Clinically validated decision rules |
