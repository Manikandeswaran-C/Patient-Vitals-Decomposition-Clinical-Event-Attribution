"""
pipeline/main_pipeline.py — End-to-End Orchestration
=====================================================
This is the MASTER script that runs the full interpretable anomaly
attribution pipeline from raw data to clinical report.

PIPELINE STEPS:
  1.  Data Generation / Loading
  2.  Signal Cleaning (missing values, artifacts, smoothing)
  3.  Normalization
  4.  Feature Engineering (rolling stats, clinical composites)
  5.  STL Decomposition (trend / seasonal / residual)
  6.  Variance Analysis (which component drives variability?)
  7.  Wavelet Decomposition (multi-scale frequency analysis)
  8a. Statistical Detection (rolling Z-score + IQR on residuals)
  8b. Isolation Forest Detection (multivariate)
  8c. LSTM Autoencoder Detection (temporal)
  8d. Ensemble + Cohen's Kappa Agreement
  9.  SHAP Attribution (which vitals caused each anomaly?)
  10. Feature Importance (global ranking)
  11. Correlation Analysis (inter-signal relationships)
  12. Lead-Lag Analysis (residual cross-correlation)
  13. Granger Causality (predictive relationships)
  14. Clinical Interpretation (rule-based insights + NLG)
  15. Visualization (all plots saved to outputs/plots/)
  16. Report Generation (text report saved to outputs/reports/)
"""

import os
import sys
import time
import warnings
import argparse
from pathlib import Path
from typing import Optional, Dict
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DATA_GENERATION, OUTPUT, DETECTION
from data.generator import PhysioDataGenerator
from data.loader import PhysioDataLoader
from preprocessing.cleaner import SignalCleaner
from preprocessing.normalizer import SignalNormalizer
from preprocessing.feature_engineer import FeatureEngineer
from decomposition.stl_decomposer import STLDecomposer
from decomposition.wavelet_decomposer import WaveletDecomposer
from detection.statistical import StatisticalDetector
from detection.isolation_forest import IsolationForestDetector
from detection.autoencoder import LSTMAutoencoderDetector
from detection.ensemble import EnsembleDetector
from attribution.shap_explainer import SHAPExplainer
from attribution.feature_importance import FeatureImportanceAnalyzer
from relationships.correlation import CorrelationAnalyzer
from relationships.causality import GrangerCausalityAnalyzer
from insights.clinical_interpreter import ClinicalInterpreter
from visualization.dashboard import PhysioDashboard


class PhysioAnomalyPipeline:
    """
    End-to-end interpretable physiological anomaly attribution pipeline.

    Usage (quickstart):
        pipeline = PhysioAnomalyPipeline(patient_id="P001")
        results = pipeline.run()

    Usage (with real data):
        pipeline = PhysioAnomalyPipeline(patient_id="P001")
        results = pipeline.run(data_path="my_data.csv", column_map={...})

    Results dict contains all intermediate outputs for further analysis.
    """

    def __init__(
        self,
        patient_id: str = "P001",
        output_dir: str = "outputs",
        use_lstm: bool = False,       # Set True if you have GPU / time to train
        verbose: bool = True,
    ):
        self.patient_id = patient_id
        self.output_dir = Path(output_dir)
        self.use_lstm = use_lstm
        self.verbose = verbose

        # Create output directories
        for subdir in ["plots", "reports", "data", "models"]:
            (self.output_dir / subdir).mkdir(parents=True, exist_ok=True)

        self._results: Dict = {}

    def run(
        self,
        data_path: Optional[str] = None,
        column_map: Optional[dict] = None,
        data_source: str = "synthetic",  # "synthetic" | "csv" | "mimic" | "eicu"
    ) -> Dict:
        """
        Execute the full pipeline.

        Args:
            data_path: Path to CSV file (if not using synthetic data).
            column_map: Column name mapping for CSV loading.
            data_source: Which data source to use.

        Returns:
            Dict containing all intermediate outputs and final report.
        """
        banner = "=" * 65
        print(f"\n{banner}")
        print("  🏥 PhysioAnomalyPipeline — Interpretable ICU Signal Analysis")
        print(f"  Patient: {self.patient_id}")
        print(banner)
        t_start = time.time()

        # ─── STEP 1: DATA ────────────────────────────────────────────
        df_raw = self._step1_load_data(data_path, column_map, data_source)

        # ─── STEP 2: CLEANING ────────────────────────────────────────
        df_clean, clean_report = self._step2_clean(df_raw)

        # ─── STEP 3: NORMALIZATION ───────────────────────────────────
        normalizer, df_norm = self._step3_normalize(df_clean)

        # ─── STEP 4: FEATURE ENGINEERING ─────────────────────────────
        df_features = self._step4_feature_engineering(df_clean)

        # ─── STEP 5: STL DECOMPOSITION ───────────────────────────────
        stl_components, residuals, variance_df = self._step5_stl(df_clean)

        # ─── STEP 6: WAVELET DECOMPOSITION ───────────────────────────
        wavelet_results = self._step6_wavelet(df_clean)

        # ─── STEP 7: STATISTICAL DETECTION ───────────────────────────
        stat_results = self._step7_statistical_detection(residuals)

        # ─── STEP 8: ISOLATION FOREST ────────────────────────────────
        if_detector, if_results = self._step8_isolation_forest(df_features)

        # ─── STEP 9: LSTM AUTOENCODER ────────────────────────────────
        lstm_flags = self._step9_lstm(df_features) if self.use_lstm else None

        # ─── STEP 10: ENSEMBLE ───────────────────────────────────────
        ensemble_results = self._step10_ensemble(stat_results, if_results, lstm_flags)
        ensemble_flags = ensemble_results["ensemble_flags"]

        # ─── STEP 11: SHAP ATTRIBUTION ───────────────────────────────
        shap_df, classification_df = self._step11_attribution(
            df_features, if_detector, ensemble_flags, stl_components
        )

        # ─── STEP 12: FEATURE IMPORTANCE ─────────────────────────────
        importance_df = self._step12_feature_importance(
            df_features, ensemble_flags, shap_df
        )

        # ─── STEP 13: CORRELATION ────────────────────────────────────
        corr_analyzer, corr_matrix, lead_lag_df = self._step13_correlation(
            df_clean, residuals
        )

        # ─── STEP 14: GRANGER CAUSALITY ──────────────────────────────
        granger_df = self._step14_granger(residuals)

        # ─── STEP 15: CLINICAL INSIGHTS ──────────────────────────────
        report = self._step15_clinical_report(
            df_clean, classification_df, corr_matrix,
            granger_df, importance_df, lead_lag_df, variance_df
        )

        # ─── STEP 16: VISUALIZATION ──────────────────────────────────
        self._step16_visualize(
            df_clean, ensemble_flags, stl_components, stat_results,
            importance_df, corr_matrix, classification_df,
            variance_df, lead_lag_df, corr_analyzer, ensemble_results
        )

        elapsed = time.time() - t_start
        print(f"\n{banner}")
        print(f"  ✅ Pipeline complete in {elapsed:.1f} seconds")
        print(f"  📁 Plots   → {self.output_dir}/plots/")
        print(f"  📋 Report  → {self.output_dir}/reports/")
        print(banner)

        # Collect all results
        self._results = {
            "patient_id": self.patient_id,
            "raw_data": df_raw,
            "clean_data": df_clean,
            "features": df_features,
            "stl_components": stl_components,
            "residuals": residuals,
            "variance_report": variance_df,
            "wavelet_results": wavelet_results,
            "statistical_detection": stat_results,
            "isolation_forest": if_results,
            "ensemble": ensemble_results,
            "ensemble_flags": ensemble_flags,
            "shap_values": shap_df,
            "anomaly_classifications": classification_df,
            "feature_importance": importance_df,
            "correlation_matrix": corr_matrix,
            "lead_lag": lead_lag_df,
            "granger": granger_df,
            "clinical_report": report,
        }
        return self._results

    # ──────────────────────────────────────────────────────────────────
    # PRIVATE STEP METHODS
    # ──────────────────────────────────────────────────────────────────

    def _step1_load_data(self, data_path, column_map, source) -> pd.DataFrame:
        if source == "synthetic" or data_path is None:
            gen = PhysioDataGenerator(patient_id=self.patient_id)
            return gen.generate()
        elif source == "csv":
            loader = PhysioDataLoader()
            return loader.from_csv(data_path, column_map=column_map)
        elif source == "mimic":
            loader = PhysioDataLoader()
            return loader.from_mimic(data_path, patient_id=self.patient_id)
        elif source == "eicu":
            loader = PhysioDataLoader()
            return loader.from_eicu(data_path, patient_id=int(self.patient_id))
        else:
            raise ValueError(f"Unknown data_source: {source}")

    def _step2_clean(self, df):
        cleaner = SignalCleaner()
        df_clean, report = cleaner.clean(df, verbose=self.verbose)
        self.clean_report = report
        return df_clean, report

    def _step3_normalize(self, df):
        normalizer = SignalNormalizer()
        df_norm = normalizer.fit_transform(df)
        normalizer.save(str(self.output_dir / "models" / "normalizer.pkl"))
        return normalizer, df_norm

    def _step4_feature_engineering(self, df):
        fe = FeatureEngineer()
        return fe.transform(df, verbose=self.verbose)

    def _step5_stl(self, df):
        print("\n📊 Step 3: STL Decomposition")
        decomposer = STLDecomposer()
        components = decomposer.decompose(df, verbose=self.verbose)
        residuals = decomposer.get_residuals()
        variance_df = decomposer.variance_analysis()
        return components, residuals, variance_df

    def _step6_wavelet(self, df):
        print("\n🌊 Step 3b: Wavelet Decomposition")
        wd = WaveletDecomposer()
        results = wd.decompose(df, verbose=self.verbose)
        energy = wd.energy_distribution()
        if self.verbose:
            print("\n   Wavelet Energy Distribution:")
            print(energy.to_string())
        return results

    def _step7_statistical_detection(self, residuals):
        det = StatisticalDetector()
        return det.detect(residuals, verbose=self.verbose)

    def _step8_isolation_forest(self, df_features):
        from config import VITAL_SIGNS
        feature_cols = [c for c in df_features.columns if c in VITAL_SIGNS]
        det = IsolationForestDetector()
        results = det.fit_detect(df_features, feature_cols=feature_cols, verbose=self.verbose)
        det.save(str(self.output_dir / "models" / "isolation_forest.pkl"))
        return det, results

    def _step9_lstm(self, df_features):
        from config import VITAL_SIGNS
        feature_cols = [c for c in df_features.columns if c in VITAL_SIGNS]
        det = LSTMAutoencoderDetector()
        det.fit(df_features, feature_cols=feature_cols, verbose=self.verbose)
        results = det.detect(df_features, verbose=self.verbose)
        det.save(str(self.output_dir / "models" / "lstm_autoencoder.pt"))
        return results["anomaly_flags"]

    def _step10_ensemble(self, stat_results, if_results, lstm_flags=None):
        ens = EnsembleDetector(min_votes=DETECTION["ensemble_min_votes"])
        kwargs = {
            "statistical": stat_results["anomaly_flags"].any(axis=1),
            "isolation_forest": if_results["anomaly_flags"],
        }
        if lstm_flags is not None:
            kwargs["lstm"] = lstm_flags
        return ens.combine(verbose=self.verbose, **kwargs)

    def _step11_attribution(self, df_features, if_detector, ensemble_flags, stl_components):
        from config import VITAL_SIGNS
        feature_cols = [c for c in df_features.columns if c in VITAL_SIGNS]

        explainer = SHAPExplainer()
        try:
            explainer.fit(df_features, if_detector.model, feature_cols=feature_cols)
            shap_df = explainer.explain_anomalies(
                ensemble_flags, max_explain=DETECTION.get("max_anomalies", 50)
            )
            classification_df = explainer.classify_attribution_types(stl_components, shap_df)
        except Exception as e:
            print(f"   ⚠️ SHAP attribution error: {e}. Skipping.")
            shap_df = pd.DataFrame()
            classification_df = pd.DataFrame()

        return shap_df, classification_df

    def _step12_feature_importance(self, df_features, ensemble_flags, shap_df):
        from config import VITAL_SIGNS
        feature_cols = [c for c in df_features.columns if c in VITAL_SIGNS]
        fia = FeatureImportanceAnalyzer()
        return fia.compute(
            df_features, ensemble_flags,
            shap_df=shap_df,
            feature_cols=feature_cols,
            verbose=self.verbose,
        )

    def _step13_correlation(self, df_clean, residuals):
        ca = CorrelationAnalyzer()
        corr_matrix = ca.compute_correlation(df_clean, verbose=self.verbose)
        lead_lag_df = ca.compute_lead_lag(residuals, verbose=self.verbose)
        return ca, corr_matrix, lead_lag_df

    def _step14_granger(self, residuals):
        gca = GrangerCausalityAnalyzer()
        return gca.analyze(residuals, verbose=self.verbose)

    def _step15_clinical_report(
        self, df_clean, classification_df, corr_matrix,
        granger_df, importance_df, lead_lag_df, variance_df
    ):
        ci = ClinicalInterpreter()
        report = ci.generate_report(
            df=df_clean,
            anomaly_classifications=classification_df if not classification_df.empty else None,
            correlation_matrix=corr_matrix if not corr_matrix.empty else None,
            granger_results=granger_df if not granger_df.empty else None,
            feature_importance=importance_df if not importance_df.empty else None,
            lead_lag_df=lead_lag_df if not lead_lag_df.empty else None,
            patient_id=self.patient_id,
            variance_report=variance_df if not variance_df.empty else None,
            clean_report=getattr(self, "clean_report", None),
        )
        report_path = str(self.output_dir / "reports" / f"{self.patient_id}_report.txt")
        ci.save_report(report, report_path)

        print(f"\n{'─'*65}")
        print(report["full_text"][:1500])
        print(f"{'─'*65}")
        return report

    def _step16_visualize(
        self, df_clean, ensemble_flags, stl_components, stat_results,
        importance_df, corr_matrix, classification_df,
        variance_df, lead_lag_df, corr_analyzer, ensemble_results
    ):
        print("\n🎨 Step 9: Generating Visualizations")
        dash = PhysioDashboard(output_dir=str(self.output_dir / "plots"))

        try:
            dash.plot_raw_signals(df_clean, anomaly_flags=ensemble_flags)
        except Exception as e:
            print(f"   ⚠️ Raw signal plot failed: {e}")

        try:
            # Plot first 3 vitals for STL (to avoid too many files)
            vitals_to_plot = list(stl_components.keys())[:3]
            for v in vitals_to_plot:
                dash.plot_stl_decomposition({v: stl_components[v]}, vital=v)
        except Exception as e:
            print(f"   ⚠️ STL plot failed: {e}")

        try:
            zscore_df = stat_results.get("zscore_df")
            if zscore_df is not None:
                dash.plot_zscore_timeline(zscore_df)
        except Exception as e:
            print(f"   ⚠️ Z-score plot failed: {e}")

        try:
            if not importance_df.empty:
                shap_imp = importance_df.get("shap_importance", pd.Series())
                if not shap_imp.empty:
                    dash.plot_shap_importance(shap_imp.sort_values(ascending=False))
        except Exception as e:
            print(f"   ⚠️ SHAP importance plot failed: {e}")

        try:
            if not corr_matrix.empty:
                dash.plot_correlation_heatmap(corr_matrix)
        except Exception as e:
            print(f"   ⚠️ Correlation heatmap failed: {e}")

        try:
            if classification_df is not None and not classification_df.empty:
                dash.plot_anomaly_timeline(classification_df, df_clean)
        except Exception as e:
            print(f"   ⚠️ Timeline plot failed: {e}")

        try:
            if not variance_df.empty:
                dash.plot_variance_decomposition(variance_df)
        except Exception as e:
            print(f"   ⚠️ Variance plot failed: {e}")

        try:
            if lead_lag_df is not None and not lead_lag_df.empty:
                dash.plot_lead_lag(lead_lag_df, corr_analyzer)
        except Exception as e:
            print(f"   ⚠️ Lead-lag plot failed: {e}")

        try:
            kappa = ensemble_results.get("kappa_matrix")
            if kappa is not None and not kappa.empty:
                dash.plot_ensemble_agreement(kappa)
        except Exception as e:
            print(f"   ⚠️ Ensemble agreement plot failed: {e}")

        dash.close_all()
        print(f"   ✅ All plots saved to {self.output_dir}/plots/")


# ─────────────────────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Interpretable Physiological Anomaly Attribution Pipeline"
    )
    parser.add_argument("--patient-id", default="P001", help="Patient identifier")
    parser.add_argument("--data-path", default=None, help="Path to CSV data file")
    parser.add_argument("--data-source", default="synthetic",
                        choices=["synthetic", "csv", "mimic", "eicu"],
                        help="Data source type")
    parser.add_argument("--output-dir", default="outputs", help="Output directory")
    parser.add_argument("--use-lstm", action="store_true",
                        help="Include LSTM Autoencoder (slower, requires PyTorch)")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")
    args = parser.parse_args()

    pipeline = PhysioAnomalyPipeline(
        patient_id=args.patient_id,
        output_dir=args.output_dir,
        use_lstm=args.use_lstm,
        verbose=not args.quiet,
    )

    results = pipeline.run(
        data_path=args.data_path,
        data_source=args.data_source,
    )

    print(f"\n✅ Done. Access results via the returned dict or check outputs/")
    return results


if __name__ == "__main__":
    main()
