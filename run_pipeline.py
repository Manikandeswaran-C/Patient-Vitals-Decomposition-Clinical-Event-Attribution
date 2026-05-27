"""
run_pipeline.py — Quick-start runner for the PhysioAnomalyPipeline
===================================================================
Simply run:  python run_pipeline.py
That's it! Everything else is automatic.
"""

from pipeline.main_pipeline import PhysioAnomalyPipeline

if __name__ == "__main__":
    # ── Option 1: Synthetic data (default, no files needed) ──────────
    pipeline = PhysioAnomalyPipeline(
        patient_id="P001",
        output_dir="outputs",
        use_lstm=False,   # Set True if you have time / GPU
        verbose=True,
    )
    results = pipeline.run(data_source="synthetic")

    # ── Option 2: Load your own CSV ───────────────────────────────────
    # pipeline.run(
    #     data_path="data/raw/my_vitals.csv",
    #     data_source="csv",
    #     column_map={
    #         "HR":   "heart_rate",
    #         "SpO2": "spo2",
    #         "RR":   "respiratory_rate",
    #         "Temp": "temperature",
    #         "SBP":  "systolic_bp",
    #         "DBP":  "diastolic_bp",
    #     }
    # )

    # Access results programmatically
    print("\n📊 Quick Results Summary:")
    print(f"   Total samples         : {len(results['clean_data'])}")
    print(f"   Anomalies detected    : {results['ensemble_flags'].sum()}")
    print(f"   Anomaly types found   : "
          f"{results['anomaly_classifications'].get('anomaly_type', {}).value_counts().to_dict() if not results['anomaly_classifications'].empty and 'anomaly_type' in results['anomaly_classifications'].columns else 'N/A'}")
    print(f"\n   Output plots   → outputs/plots/")
    print(f"   Clinical report → outputs/reports/P001_report.txt")
