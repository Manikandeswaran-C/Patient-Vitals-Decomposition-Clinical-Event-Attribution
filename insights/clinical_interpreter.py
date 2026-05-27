"""
insights/clinical_interpreter.py — Clinical Insight Generator
==============================================================
Step 7: Translate all analysis results into plain English clinical insights.

This module generates:
  1. Per-anomaly clinical explanations (what happened and why)
  2. Temporal event timeline (chronological narrative)
  3. Vital sign relationship insights (from correlation + Granger)
  4. Overall patient stability assessment
  5. A structured clinical report (text + JSON)

DESIGN PHILOSOPHY:
    All language is evidence-based and hedged appropriately.
    We use "suggests" / "consistent with" / "may indicate"
    rather than definitive diagnoses — the system supports clinicians,
    it does not replace clinical judgment.
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from config import VITAL_SIGNS


# ─────────────────────────────────────────────────────────────
# CLINICAL RULES (evidence-based thresholds)
# ─────────────────────────────────────────────────────────────

CLINICAL_RULES = {
    "sepsis_triad": {
        "description": "Sepsis-like pattern",
        "conditions": {
            "heart_rate": (">", 90),
            "respiratory_rate": (">", 20),
            "temperature": (">", 38.0),
        },
        "explanation": (
            "Elevated heart rate, respiratory rate, and temperature simultaneously — "
            "consistent with Systemic Inflammatory Response Syndrome (SIRS) / sepsis. "
            "Immediate clinical review recommended."
        ),
        "severity": "HIGH",
    },
    "hypoxemic_distress": {
        "description": "Hypoxemic respiratory distress",
        "conditions": {
            "spo2": ("<", 94),
            "respiratory_rate": (">", 22),
        },
        "explanation": (
            "Low oxygen saturation combined with elevated respiratory rate — "
            "consistent with hypoxemic respiratory distress. "
            "Assess airway, breathing, and oxygen delivery."
        ),
        "severity": "HIGH",
    },
    "hemodynamic_instability": {
        "description": "Hemodynamic instability",
        "conditions": {
            "systolic_bp": ("<", 90),
            "heart_rate": (">", 100),
        },
        "explanation": (
            "Low systolic blood pressure with compensatory tachycardia — "
            "consistent with hemodynamic compromise. "
            "Consider fluid resuscitation and vasopressor assessment."
        ),
        "severity": "CRITICAL",
    },
    "hypertensive_urgency": {
        "description": "Hypertensive urgency",
        "conditions": {
            "systolic_bp": (">", 180),
            "diastolic_bp": (">", 110),
        },
        "explanation": (
            "Severely elevated blood pressure — "
            "consistent with hypertensive urgency or emergency. "
            "Monitor for end-organ damage."
        ),
        "severity": "HIGH",
    },
    "bradycardia": {
        "description": "Significant bradycardia",
        "conditions": {
            "heart_rate": ("<", 50),
        },
        "explanation": (
            "Markedly low heart rate — consider vagal event, beta-blocker toxicity, "
            "hypothyroidism, or conduction system disease."
        ),
        "severity": "MODERATE",
    },
    "hyperthermia": {
        "description": "Hyperthermia / Fever",
        "conditions": {
            "temperature": (">", 38.5),
        },
        "explanation": (
            "Elevated body temperature suggests infection, inflammation, "
            "or drug reaction. Combine with other SIRS criteria for sepsis assessment."
        ),
        "severity": "MODERATE",
    },
    "hypothermia": {
        "description": "Hypothermia",
        "conditions": {
            "temperature": ("<", 36.0),
        },
        "explanation": (
            "Low body temperature — consider environmental exposure, "
            "septic shock (cold phase), or metabolic disturbance."
        ),
        "severity": "MODERATE",
    },
}

SHOCK_INDEX_THRESHOLDS = {
    0.7: ("Normal", "LOW"),
    1.0: ("Mild shock index elevation — monitor closely", "MODERATE"),
    1.4: ("Significant shock index — possible hemorrhagic shock or severe sepsis", "HIGH"),
}


class ClinicalInterpreter:
    """
    Generates clinical insights from all pipeline outputs.

    Usage:
        ci = ClinicalInterpreter()
        report = ci.generate_report(
            df=clean_df,
            anomaly_classifications=classification_df,
            correlation_matrix=corr_df,
            granger_results=granger_df,
            feature_importance=importance_df,
            lead_lag_df=lead_lag_df,
        )
        ci.save_report(report, "outputs/reports/patient_report.txt")
    """

    def generate_report(
        self,
        df: pd.DataFrame,
        anomaly_classifications: Optional[pd.DataFrame] = None,
        correlation_matrix: Optional[pd.DataFrame] = None,
        granger_results: Optional[pd.DataFrame] = None,
        feature_importance: Optional[pd.DataFrame] = None,
        lead_lag_df: Optional[pd.DataFrame] = None,
        patient_id: str = "Unknown",
        variance_report: Optional[pd.DataFrame] = None,
        clean_report: Optional[Dict] = None,
    ) -> Dict:
        """
        Generate a complete clinical interpretation report.

        Returns:
            Dict with sections: summary, anomaly_insights, vital_insights,
            relationship_insights, timeline, recommendations, full_text
        """
        print("\n📋 Step 7: Generating Clinical Interpretability Report")

        report = {
            "patient_id": patient_id,
            "generated_at": datetime.now().isoformat(),
            "monitoring_period": self._get_monitoring_period(df),
            "summary": {},
            "vital_status": {},
            "anomaly_insights": [],
            "relationship_insights": [],
            "timeline": [],
            "recommendations": [],
            "clean_report": clean_report,
        }

        # ── 1. Vital sign status ──────────────────────────────────────
        report["vital_status"] = self._assess_vital_status(df)

        # ── 2. Active clinical rule alerts ───────────────────────────
        alerts = self._check_clinical_rules(df)
        report["active_alerts"] = alerts

        # ── 3. Anomaly classification insights ───────────────────────
        if anomaly_classifications is not None and not anomaly_classifications.empty:
            report["anomaly_insights"] = self._interpret_anomaly_classifications(
                anomaly_classifications
            )

        # ── 4. Relationship insights ──────────────────────────────────
        if correlation_matrix is not None:
            report["relationship_insights"].extend(
                self._interpret_correlations(correlation_matrix)
            )
        if granger_results is not None and not granger_results.empty:
            report["relationship_insights"].extend(
                self._interpret_granger(granger_results)
            )
        if lead_lag_df is not None and not lead_lag_df.empty:
            report["relationship_insights"].extend(
                self._interpret_lead_lag(lead_lag_df)
            )

        # ── 5. Feature importance insights ───────────────────────────
        if feature_importance is not None and not feature_importance.empty:
            top_vital = feature_importance.index[0]
            report["summary"]["primary_driver"] = top_vital
            report["summary"]["primary_driver_note"] = (
                f"{top_vital} shows the highest contribution to anomaly scores — "
                f"this vital sign deserves the closest monitoring attention."
            )

        # ── 6. Variance stability assessment ─────────────────────────
        if variance_report is not None:
            report["stability_assessment"] = self._assess_stability(variance_report)

        # ── 7. Build timeline ─────────────────────────────────────────
        if anomaly_classifications is not None and not anomaly_classifications.empty:
            report["timeline"] = self._build_timeline(anomaly_classifications)

        # ── 8. Recommendations ────────────────────────────────────────
        report["recommendations"] = self._generate_recommendations(
            alerts, report.get("anomaly_insights", [])
        )

        # ── 9. Generate full text report ──────────────────────────────
        report["full_text"] = self._render_text_report(report)

        print(f"   ✅ Clinical report generated: "
              f"{len(report['anomaly_insights'])} anomaly insights, "
              f"{len(alerts)} active alerts")

        return report

    def save_report(self, report: Dict, path: str):
        """Save the text report to a file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(report["full_text"])
        print(f"   💾 Report saved to {path}")

    # ──────────────────────────────────────────────────────────────────
    # PRIVATE METHODS
    # ──────────────────────────────────────────────────────────────────

    def _get_monitoring_period(self, df: pd.DataFrame) -> Dict:
        if not isinstance(df.index, pd.DatetimeIndex):
            return {"start": "unknown", "end": "unknown", "duration_hours": 0}
        duration = (df.index[-1] - df.index[0]).total_seconds() / 3600
        return {
            "start": str(df.index[0]),
            "end": str(df.index[-1]),
            "duration_hours": round(duration, 1),
            "n_samples": len(df),
        }

    def _assess_vital_status(self, df: pd.DataFrame) -> Dict:
        """Classify each vital as normal/abnormal based on recent values."""
        status = {}
        vital_cols = [c for c in df.columns if c in VITAL_SIGNS]
        recent = df[vital_cols].tail(30).mean()  # Last 30 samples

        for col in vital_cols:
            cfg = VITAL_SIGNS[col]
            val = recent.get(col, np.nan)
            if np.isnan(val):
                status[col] = {"value": None, "status": "NO_DATA"}
                continue

            if val < cfg["normal_low"]:
                s = "LOW"
            elif val > cfg["normal_high"]:
                s = "HIGH"
            else:
                s = "NORMAL"

            status[col] = {
                "value": round(float(val), 2),
                "unit": cfg["unit"],
                "status": s,
                "normal_range": f"{cfg['normal_low']}–{cfg['normal_high']} {cfg['unit']}",
            }
        return status

    def _check_clinical_rules(self, df: pd.DataFrame) -> List[Dict]:
        """Check rule-based clinical alerts on recent data."""
        alerts = []
        vital_cols = [c for c in df.columns if c in VITAL_SIGNS]
        recent_means = df[vital_cols].tail(30).mean()

        for rule_name, rule in CLINICAL_RULES.items():
            conditions = rule["conditions"]
            triggered = True

            for vital, (op, threshold) in conditions.items():
                if vital not in recent_means or np.isnan(recent_means[vital]):
                    triggered = False
                    break
                val = recent_means[vital]
                if op == ">" and not (val > threshold):
                    triggered = False
                    break
                if op == "<" and not (val < threshold):
                    triggered = False
                    break

            if triggered:
                alerts.append({
                    "rule": rule_name,
                    "description": rule["description"],
                    "severity": rule["severity"],
                    "explanation": rule["explanation"],
                })

        return alerts

    def _interpret_anomaly_classifications(
        self, classifications: pd.DataFrame
    ) -> List[Dict]:
        """Convert anomaly classification rows into narrative insights."""
        insights = []
        type_counts = classifications["anomaly_type"].value_counts().to_dict() if "anomaly_type" in classifications.columns else {}

        for atype, count in type_counts.items():
            subset = classifications[classifications["anomaly_type"] == atype]
            dominant_vitals = subset["dominant_vital"].value_counts().head(2).index.tolist() if "dominant_vital" in subset.columns else []

            vitals_str = " and ".join(dominant_vitals) if dominant_vitals else "multiple vitals"

            if atype == "BASELINE_SHIFT":
                narrative = (
                    f"Detected {count} baseline shift event(s) primarily in {vitals_str}. "
                    f"This suggests a sustained change in the patient's physiological baseline, "
                    f"potentially indicating disease progression or response to treatment."
                )
            elif atype == "CIRCADIAN_DEVIATION":
                narrative = (
                    f"Detected {count} circadian deviation event(s) in {vitals_str}. "
                    f"The expected day/night physiological cycle is disrupted, "
                    f"which may reflect autonomic dysfunction, sleep disruption, or medication effects."
                )
            else:  # ACUTE_EVENT
                narrative = (
                    f"Detected {count} acute event(s) in {vitals_str}. "
                    f"Sudden transient deviations suggest arrhythmia, pain episodes, "
                    f"procedural artifacts, or acute physiological stress responses."
                )

            insights.append({
                "type": atype,
                "count": count,
                "dominant_vitals": dominant_vitals,
                "narrative": narrative,
            })

        return insights

    def _interpret_correlations(self, corr_matrix: pd.DataFrame) -> List[Dict]:
        """Generate insights from significant correlations."""
        insights = []
        cols = corr_matrix.columns.tolist()

        KNOWN_RELATIONSHIPS = {
            ("heart_rate", "systolic_bp"): "sympathetic co-activation",
            ("heart_rate", "respiratory_rate"): "cardiopulmonary coupling",
            ("respiratory_rate", "spo2"): "ventilation-oxygenation coupling",
            ("spo2", "etco2"): "respiratory gas exchange",
        }

        for i, col_a in enumerate(cols):
            for col_b in cols[i+1:]:
                if col_a not in corr_matrix.index or col_b not in corr_matrix.columns:
                    continue
                r = corr_matrix.loc[col_a, col_b]
                if abs(r) < 0.4:
                    continue

                direction = "positively" if r > 0 else "negatively"
                strength = "strongly" if abs(r) > 0.7 else "moderately"
                pair = tuple(sorted([col_a, col_b]))
                mechanism = KNOWN_RELATIONSHIPS.get(pair, "physiological coupling")

                insights.append({
                    "type": "correlation",
                    "signals": [col_a, col_b],
                    "correlation": round(r, 3),
                    "narrative": (
                        f"{col_a} and {col_b} are {strength} {direction} correlated "
                        f"(r={r:.3f}), consistent with {mechanism}."
                    ),
                })
        return insights

    def _interpret_granger(self, granger_df: pd.DataFrame) -> List[Dict]:
        """Generate insights from Granger causality results."""
        insights = []
        significant = granger_df[granger_df["significant"]] if "significant" in granger_df.columns else pd.DataFrame()

        for _, row in significant.iterrows():
            insights.append({
                "type": "granger_causality",
                "cause": row["cause"],
                "effect": row["effect"],
                "lag": row["best_lag"],
                "narrative": row["interpretation"],
            })
        return insights

    def _interpret_lead_lag(self, lead_lag_df: pd.DataFrame) -> List[Dict]:
        """Generate insights from cross-correlation lead-lag analysis."""
        insights = []
        strong = lead_lag_df[lead_lag_df["coupling_strength"] > 0.3] if "coupling_strength" in lead_lag_df.columns else pd.DataFrame()

        for _, row in strong.head(5).iterrows():
            if row["lag_magnitude"] > 0:
                narrative = (
                    f"{row['leader']} leads {row['follower']} by "
                    f"{row['lag_magnitude']} sample(s) "
                    f"(cross-correlation: {row['peak_correlation']:.3f}). "
                    f"Monitor {row['leader']} as an early warning indicator."
                )
            else:
                narrative = (
                    f"{row['signal_a']} and {row['signal_b']} are synchronously coupled "
                    f"(cross-correlation: {row['peak_correlation']:.3f})."
                )

            insights.append({
                "type": "lead_lag",
                "signals": [row.get("leader", row["signal_a"]), row.get("follower", row["signal_b"])],
                "lag": row["lag_magnitude"],
                "narrative": narrative,
            })
        return insights

    def _assess_stability(self, variance_report: pd.DataFrame) -> Dict:
        """Assess overall signal stability from variance decomposition."""
        if variance_report.empty:
            return {}

        unstable = variance_report[variance_report.get("unstable_flag", pd.Series(False, index=variance_report.index))].index.tolist() if "unstable_flag" in variance_report.columns else []
        avg_residual = variance_report["residual_var_%"].mean() if "residual_var_%" in variance_report.columns else 0

        if avg_residual > 40:
            stability = "UNSTABLE"
            note = "High residual variance across vitals suggests significant unexplained variability."
        elif avg_residual > 25:
            stability = "BORDERLINE"
            note = "Moderate residual variance — some signals show irregular patterns."
        else:
            stability = "STABLE"
            note = "Low residual variance — signals are well-explained by trend and seasonal components."

        return {
            "overall_stability": stability,
            "avg_residual_variance_pct": round(avg_residual, 1),
            "unstable_vitals": unstable,
            "note": note,
        }

    def _build_timeline(self, classifications: pd.DataFrame) -> List[Dict]:
        """
        Build a collapsed, chronological event timeline from anomaly classifications.
        Consecutive anomaly time points of the same type are merged into single event blocks
        to reduce noise and provide duration and peak severity metrics.
        """
        timeline = []
        if "anomaly_type" not in classifications.columns or classifications.empty:
            return timeline

        # Sort classifications chronologically
        sorted_df = classifications.sort_index()

        active_event = None

        for ts, row in sorted_df.iterrows():
            etype = row.get("anomaly_type", "UNKNOWN")
            vital = row.get("dominant_vital", "unknown")
            shap_mag = float(row.get("shap_magnitude", 0))
            
            # Extract Z-scores
            t_z = float(row.get("trend_z", 0))
            s_z = float(row.get("seasonal_z", 0))
            r_z = float(row.get("residual_z", 0))
            max_z = max(t_z, s_z, r_z)

            if active_event is None:
                # Start new event
                active_event = {
                    "start_time": ts,
                    "end_time": ts,
                    "event_type": etype,
                    "dominant_vital": vital,
                    "peak_shap": shap_mag,
                    "peak_z": max_z,
                    "explanation": row.get("explanation", ""),
                    "count": 1
                }
            else:
                # Check if we can merge: same type, same vital, and within 10 minutes
                time_diff_mins = (ts - active_event["end_time"]).total_seconds() / 60.0
                if (etype == active_event["event_type"] and 
                    vital == active_event["dominant_vital"] and 
                    time_diff_mins <= 10.0):
                    # Merge / Extend
                    active_event["end_time"] = ts
                    active_event["peak_shap"] = max(active_event["peak_shap"], shap_mag)
                    active_event["peak_z"] = max(active_event["peak_z"], max_z)
                    active_event["count"] += 1
                else:
                    # Save active and start new
                    timeline.append(active_event)
                    active_event = {
                        "start_time": ts,
                        "end_time": ts,
                        "event_type": etype,
                        "dominant_vital": vital,
                        "peak_shap": shap_mag,
                        "peak_z": max_z,
                        "explanation": row.get("explanation", ""),
                        "count": 1
                    }

        if active_event is not None:
            timeline.append(active_event)

        # Format timeline items for rendering
        formatted_timeline = []
        for i, ev in enumerate(timeline):
            duration_mins = int((ev["end_time"] - ev["start_time"]).total_seconds() / 60.0) + 1
            start_str = ev["start_time"].strftime("%H:%M")
            end_str = ev["end_time"].strftime("%H:%M")
            
            if duration_mins == 1:
                time_span_str = f"at {start_str}"
            else:
                time_span_str = f"from {start_str} to {end_str} ({duration_mins} mins)"

            severity_label = "CRITICAL" if ev["peak_z"] > 4.0 else ("HIGH" if ev["peak_z"] > 2.5 else "MODERATE")

            formatted_timeline.append({
                "timestamp": ev["start_time"],
                "end_timestamp": ev["end_time"],
                "duration_minutes": duration_mins,
                "event_type": ev["event_type"],
                "dominant_vital": ev["dominant_vital"],
                "severity": severity_label,
                "peak_z": ev["peak_z"],
                "time_span_str": time_span_str,
                "description": ev["explanation"]
            })

        return formatted_timeline

    def _generate_recommendations(
        self, alerts: List[Dict], anomaly_insights: List[Dict]
    ) -> List[str]:
        """Generate prioritized clinical recommendations."""
        recs = []
        severities = {a["severity"] for a in alerts}

        if "CRITICAL" in severities:
            recs.append("⚠️  CRITICAL: Immediate bedside clinical assessment required.")
            recs.append("Activate rapid response / code blue protocol if indicated.")
        if "HIGH" in severities:
            recs.append("🔴 HIGH PRIORITY: Notify attending physician within 15 minutes.")

        for alert in alerts:
            recs.append(f"→ Review: {alert['description']} — {alert['explanation'][:100]}...")

        # Anomaly type-specific
        types = [i["type"] for i in anomaly_insights]
        if "BASELINE_SHIFT" in types:
            recs.append("→ Trend-based alert: Review medication changes, fluid balance, and disease progression.")
        if "CIRCADIAN_DEVIATION" in types:
            recs.append("→ Circadian disruption: Review sleep quality, sedation levels, and vasopressor timing.")
        if "ACUTE_EVENT" in types:
            recs.append("→ Acute events detected: Review ECG, check line/sensor placement for artifacts.")

        if not recs:
            recs.append("✅ No critical alerts. Continue routine monitoring per protocol.")

        return recs

    def _render_text_report(self, report: Dict) -> str:
        """Render the full structured report as a readable text document."""
        lines = []
        sep = "=" * 70

        lines.append(sep)
        lines.append(" PHYSIOLOGICAL ANOMALY ATTRIBUTION REPORT")
        lines.append(sep)
        lines.append(f" Patient ID      : {report['patient_id']}")
        lines.append(f" Generated At    : {report['generated_at']}")
        period = report.get("monitoring_period", {})
        lines.append(f" Monitoring Start: {period.get('start', 'N/A')}")
        lines.append(f" Monitoring End  : {period.get('end', 'N/A')}")
        lines.append(f" Duration        : {period.get('duration_hours', 0)} hours")
        lines.append(f" Total Samples   : {period.get('n_samples', 0)}")
        lines.append(sep)

        # Vital status
        lines.append("\n[ CURRENT VITAL SIGN STATUS ]")
        for vital, status in report.get("vital_status", {}).items():
            s = status.get("status", "?")
            v = status.get("value", "N/A")
            unit = status.get("unit", "")
            rng = status.get("normal_range", "")
            flag = "⚠️ " if s != "NORMAL" else "   "
            lines.append(f"  {flag}{vital:<22}: {v} {unit:<15} [{s}]  (Normal: {rng})")

        # Active alerts
        lines.append("\n[ ACTIVE CLINICAL ALERTS ]")
        alerts = report.get("active_alerts", [])
        if alerts:
            for alert in alerts:
                lines.append(f"  [{alert['severity']}] {alert['description']}")
                lines.append(f"    → {alert['explanation']}")
        else:
            lines.append("  No active clinical alerts.")

        # Cleaning & Artifact Summary
        clean_report = report.get("clean_report")
        if clean_report and "vitals" in clean_report:
            lines.append("\n[ SIGNAL CLEANING & ARTIFACT SUMMARY ]")
            lines.append(f"  {'Vital Sign':<22} {'Disconnects':<14} {'Motion Noise':<14} {'Impossible Clipped'}")
            lines.append(f"  {'─'*22} {'─'*14} {'─'*14} {'─'*18}")
            for vital, info in clean_report["vitals"].items():
                disc = info.get("sensor_disconnects", 0)
                mot = info.get("motion_artifacts", 0)
                imp = info.get("impossible_clipped", 0)
                if disc > 0 or mot > 0 or imp > 0:
                    lines.append(f"  {vital:<22} {disc:<14} {mot:<14} {imp}")
            # If no artifacts were detected
            tot_disc = sum(info.get("sensor_disconnects", 0) for info in clean_report["vitals"].values())
            tot_mot = sum(info.get("motion_artifacts", 0) for info in clean_report["vitals"].values())
            if tot_disc == 0 and tot_mot == 0:
                lines.append("  No significant sensor disconnects or motion artifacts detected.")

        # Stability
        stab = report.get("stability_assessment", {})
        if stab:
            lines.append("\n[ SIGNAL STABILITY ASSESSMENT ]")
            lines.append(f"  Overall Stability: {stab.get('overall_stability', 'N/A')}")
            lines.append(f"  Avg. Residual Variance: {stab.get('avg_residual_variance_pct', 0)}%")
            lines.append(f"  Note: {stab.get('note', '')}")
            if stab.get("unstable_vitals"):
                lines.append(f"  Unstable Vitals: {', '.join(stab['unstable_vitals'])}")

        # Anomaly insights
        lines.append("\n[ ANOMALY ATTRIBUTION INSIGHTS ]")
        for insight in report.get("anomaly_insights", []):
            lines.append(f"\n  [{insight['type']}] — {insight['count']} event(s)")
            lines.append(f"    {insight['narrative']}")

        # Relationship insights
        lines.append("\n[ INTER-SIGNAL RELATIONSHIP INSIGHTS ]")
        for rel in report.get("relationship_insights", []):
            lines.append(f"  • {rel['narrative']}")

        # Timeline
        timeline = report.get("timeline", [])
        if timeline:
            lines.append(f"\n[ TEMPORAL EVENT TIMELINE ({len(timeline)} event blocks) ]")
            for i, event in enumerate(timeline[:20], 1):  # Show max 20
                lines.append(
                    f"  {i:3d}. [{event['severity']:<8}] {event['event_type']:<22} in {event['dominant_vital']:<15} "
                    f"{event['time_span_str']}"
                )
                lines.append(f"       → {event['description']}")
            if len(timeline) > 20:
                lines.append(f"  ... and {len(timeline)-20} more event blocks.")

        # Recommendations
        lines.append("\n[ CLINICAL RECOMMENDATIONS ]")
        for rec in report.get("recommendations", []):
            lines.append(f"  {rec}")

        # Summary
        summary = report.get("summary", {})
        if summary.get("primary_driver"):
            lines.append("\n[ KEY FINDING ]")
            lines.append(f"  Primary anomaly driver: {summary['primary_driver']}")
            lines.append(f"  {summary.get('primary_driver_note', '')}")

        lines.append(f"\n{sep}")
        lines.append(" END OF REPORT — For clinical decision support only.")
        lines.append(" Always confirm findings with qualified clinical assessment.")
        lines.append(sep)

        return "\n".join(lines)
