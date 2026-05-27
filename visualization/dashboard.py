"""
visualization/dashboard.py — Interactive Visualization Dashboard
================================================================
Generates all plots for the pipeline:
  1. Raw vital signs with anomaly markers
  2. STL decomposition plots (trend / seasonal / residual)
  3. Z-score timeline with thresholds
  4. SHAP attribution bar plots (per anomaly + global)
  5. Correlation heatmap
  6. Lead-lag cross-correlation plots
  7. Granger causality network diagram
  8. Temporal event timeline chart
  9. Wavelet energy distribution
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import seaborn as sns
from pathlib import Path
from typing import Dict, Optional, List
import warnings

from config import VITAL_SIGNS, VISUALIZATION


class PhysioDashboard:
    """
    Generates and saves all visualization plots.

    Usage:
        dash = PhysioDashboard(output_dir="outputs/plots")
        dash.plot_raw_signals(df, anomaly_flags)
        dash.plot_stl_decomposition(stl_components)
        dash.plot_correlation_heatmap(corr_matrix)
        dash.plot_shap_importance(shap_importance)
        dash.plot_timeline(classification_df)
    """

    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = Path(output_dir or VISUALIZATION["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Apply visual style
        try:
            plt.style.use(VISUALIZATION["style"])
        except Exception:
            plt.style.use("seaborn-v0_8-darkgrid")

        self.dpi = VISUALIZATION["figure_dpi"]
        self.anomaly_color = VISUALIZATION["anomaly_color"]
        self.fig_width = VISUALIZATION["figure_width"]
        self.fig_height = VISUALIZATION["figure_height"]

    # ──────────────────────────────────────────────────────────────────
    # 1. RAW SIGNALS + ANOMALY MARKERS
    # ──────────────────────────────────────────────────────────────────

    def plot_raw_signals(
        self,
        df: pd.DataFrame,
        anomaly_flags: Optional[pd.Series] = None,
        title: str = "Multivariate Physiological Signals with Anomalies",
        save: bool = True,
    ) -> plt.Figure:
        """Plot all vital signs on a shared time axis with anomaly markers."""
        vital_cols = [c for c in df.columns if c in VITAL_SIGNS]
        n = len(vital_cols)

        fig, axes = plt.subplots(n, 1, figsize=(self.fig_width, 2.2 * n), sharex=True)
        if n == 1:
            axes = [axes]

        fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)

        for ax, col in zip(axes, vital_cols):
            cfg = VITAL_SIGNS[col]
            color = cfg.get("color", "#3498db")

            # Plot signal
            ax.plot(df.index, df[col], color=color, linewidth=0.9,
                    alpha=0.85, label=col.replace("_", " ").title())

            # Shade normal range
            ax.axhspan(cfg["normal_low"], cfg["normal_high"],
                       alpha=0.08, color="green", label="Normal range")
            ax.axhline(cfg["normal_low"], color="green", linewidth=0.6, linestyle="--", alpha=0.4)
            ax.axhline(cfg["normal_high"], color="green", linewidth=0.6, linestyle="--", alpha=0.4)

            # Mark anomalies
            if anomaly_flags is not None:
                anom_times = df.index[anomaly_flags.reindex(df.index, fill_value=False)]
                if len(anom_times) > 0:
                    anom_vals = df.loc[anom_times, col]
                    ax.scatter(anom_times, anom_vals,
                               color=self.anomaly_color, s=25, zorder=5,
                               marker="v", alpha=0.8, label="Anomaly")

            ax.set_ylabel(f"{col.replace('_', ' ').title()}\n({cfg['unit']})", fontsize=9)
            ax.legend(loc="upper right", fontsize=7, ncol=3)
            ax.tick_params(axis="x", rotation=30)

        axes[-1].set_xlabel("Time", fontsize=10)
        plt.tight_layout()

        if save:
            path = self.output_dir / "01_raw_signals.png"
            fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
            print(f"   💾 Saved: {path}")

        return fig

    # ──────────────────────────────────────────────────────────────────
    # 2. STL DECOMPOSITION
    # ──────────────────────────────────────────────────────────────────

    def plot_stl_decomposition(
        self,
        stl_components: Dict[str, pd.DataFrame],
        vital: Optional[str] = None,
        save: bool = True,
    ) -> plt.Figure:
        """
        Plot STL decomposition for one (or all) vital sign(s).
        Shows: observed | trend | seasonal | residual
        """
        vitals_to_plot = [vital] if vital else list(stl_components.keys())

        for v in vitals_to_plot:
            if v not in stl_components:
                continue

            comp = stl_components[v]
            cfg = VITAL_SIGNS.get(v, {})
            color = cfg.get("color", "#3498db")

            fig, axes = plt.subplots(4, 1, figsize=(self.fig_width, 10), sharex=True)
            fig.suptitle(
                f"STL Decomposition — {v.replace('_', ' ').title()} "
                f"({cfg.get('unit', '')})",
                fontsize=13, fontweight="bold"
            )

            components = ["observed", "trend", "seasonal", "residual"]
            labels = ["Observed Signal", "Trend Component", "Seasonal Component", "Residual (Anomaly Zone)"]
            colors = [color, "#2ecc71", "#e67e22", self.anomaly_color]

            for ax, comp_name, label, c in zip(axes, components, labels, colors):
                if comp_name in comp.columns:
                    ax.plot(comp.index, comp[comp_name], color=c, linewidth=0.9)
                    if comp_name == "residual":
                        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
                        # Shade large residuals
                        std = comp[comp_name].std()
                        ax.axhspan(-3*std, 3*std, alpha=0.07, color="green")
                        ax.axhline(3*std, color="orange", linestyle=":", linewidth=0.8)
                        ax.axhline(-3*std, color="orange", linestyle=":", linewidth=0.8)
                ax.set_ylabel(label, fontsize=9)
                ax.tick_params(axis="x", rotation=20)

            axes[-1].set_xlabel("Time", fontsize=10)
            plt.tight_layout()

            if save:
                safe_name = v.replace("/", "_")
                path = self.output_dir / f"02_stl_{safe_name}.png"
                fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
                print(f"   💾 Saved: {path}")

        return fig

    # ──────────────────────────────────────────────────────────────────
    # 3. Z-SCORE TIMELINE
    # ──────────────────────────────────────────────────────────────────

    def plot_zscore_timeline(
        self,
        zscore_df: pd.DataFrame,
        threshold: float = 3.0,
        save: bool = True,
    ) -> plt.Figure:
        """Plot rolling Z-scores for all vitals with anomaly threshold lines."""
        vital_cols = [c for c in zscore_df.columns if c in VITAL_SIGNS]
        n = len(vital_cols)

        fig, axes = plt.subplots(n, 1, figsize=(self.fig_width, 2.0 * n), sharex=True)
        if n == 1:
            axes = [axes]
        fig.suptitle("Rolling Z-Score Timeline — Residual Anomaly Detection",
                     fontsize=13, fontweight="bold")

        for ax, col in zip(axes, vital_cols):
            cfg = VITAL_SIGNS.get(col, {})
            color = cfg.get("color", "#3498db")
            z = zscore_df[col]

            ax.plot(z.index, z, color=color, linewidth=0.7, alpha=0.8)
            ax.axhline(threshold, color=self.anomaly_color, linestyle="--",
                       linewidth=1.2, label=f"+{threshold}σ")
            ax.axhline(-threshold, color=self.anomaly_color, linestyle="--",
                       linewidth=1.2, label=f"-{threshold}σ")
            ax.axhline(0, color="gray", linewidth=0.5)

            # Fill anomalous regions
            ax.fill_between(z.index, z, threshold,
                             where=(z > threshold), color=self.anomaly_color, alpha=0.3)
            ax.fill_between(z.index, z, -threshold,
                             where=(z < -threshold), color=self.anomaly_color, alpha=0.3)

            ax.set_ylabel(f"Z-score\n{col.replace('_', ' ').title()}", fontsize=8)
            ax.legend(loc="upper right", fontsize=7)

        axes[-1].set_xlabel("Time", fontsize=10)
        plt.tight_layout()

        if save:
            path = self.output_dir / "03_zscore_timeline.png"
            fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
            print(f"   💾 Saved: {path}")

        return fig

    # ──────────────────────────────────────────────────────────────────
    # 4. SHAP FEATURE IMPORTANCE
    # ──────────────────────────────────────────────────────────────────

    def plot_shap_importance(
        self,
        shap_importance: pd.Series,
        title: str = "Global SHAP Feature Importance (Anomaly Attribution)",
        save: bool = True,
    ) -> plt.Figure:
        """Horizontal bar chart of mean absolute SHAP values."""
        if shap_importance.empty:
            print("   ⚠️ No SHAP data to plot.")
            return plt.figure()

        top_n = min(15, len(shap_importance))
        data = shap_importance.head(top_n).sort_values()

        fig, ax = plt.subplots(figsize=(10, max(4, top_n * 0.5)))

        colors = ["#e74c3c" if i >= top_n - 3 else "#3498db"
                  for i in range(len(data))]

        bars = ax.barh(range(len(data)), data.values, color=colors, alpha=0.85)
        ax.set_yticks(range(len(data)))
        ax.set_yticklabels([d.replace("_", " ").title() for d in data.index], fontsize=9)
        ax.set_xlabel("Mean |SHAP Value| — Contribution to Anomaly Score", fontsize=10)
        ax.set_title(title, fontsize=12, fontweight="bold")

        # Value labels
        for bar, val in zip(bars, data.values):
            ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                    f"{val:.4f}", va="center", fontsize=8)

        red_patch = mpatches.Patch(color="#e74c3c", label="Top 3 contributors")
        blue_patch = mpatches.Patch(color="#3498db", label="Other features")
        ax.legend(handles=[red_patch, blue_patch], fontsize=9)

        plt.tight_layout()
        if save:
            path = self.output_dir / "04_shap_importance.png"
            fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
            print(f"   💾 Saved: {path}")

        return fig

    # ──────────────────────────────────────────────────────────────────
    # 5. CORRELATION HEATMAP
    # ──────────────────────────────────────────────────────────────────

    def plot_correlation_heatmap(
        self,
        corr_matrix: pd.DataFrame,
        title: str = "Inter-Signal Correlation Matrix",
        save: bool = True,
    ) -> plt.Figure:
        """Plot correlation heatmap with significance highlighting."""
        if corr_matrix.empty:
            return plt.figure()

        fig, ax = plt.subplots(figsize=(9, 7))

        labels = [c.replace("_", "\n").title() for c in corr_matrix.columns]

        sns.heatmap(
            corr_matrix,
            ax=ax,
            annot=True,
            fmt=".2f",
            cmap="RdYlGn",
            center=0,
            vmin=-1,
            vmax=1,
            square=True,
            linewidths=0.5,
            xticklabels=labels,
            yticklabels=labels,
            annot_kws={"size": 9},
        )

        ax.set_title(title, fontsize=13, fontweight="bold", pad=15)
        plt.xticks(rotation=30, ha="right", fontsize=8)
        plt.yticks(rotation=0, fontsize=8)
        plt.tight_layout()

        if save:
            path = self.output_dir / "05_correlation_heatmap.png"
            fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
            print(f"   💾 Saved: {path}")

        return fig

    # ──────────────────────────────────────────────────────────────────
    # 6. ANOMALY TIMELINE
    # ──────────────────────────────────────────────────────────────────

    def plot_anomaly_timeline(
        self,
        classifications: pd.DataFrame,
        df: pd.DataFrame,
        save: bool = True,
    ) -> plt.Figure:
        """
        Plot a temporal event timeline showing anomaly types over time.
        Color-coded: BASELINE_SHIFT=blue, CIRCADIAN_DEVIATION=orange, ACUTE_EVENT=red
        """
        if classifications.empty or "anomaly_type" not in classifications.columns:
            print("   ⚠️ No classification data for timeline.")
            return plt.figure()

        type_colors = {
            "BASELINE_SHIFT": "#3498db",
            "CIRCADIAN_DEVIATION": "#e67e22",
            "ACUTE_EVENT": "#e74c3c",
        }

        fig, (ax_timeline, ax_count) = plt.subplots(
            2, 1, figsize=(self.fig_width, 7), sharex=False,
            gridspec_kw={"height_ratios": [3, 1]}
        )
        fig.suptitle("Temporal Anomaly Event Timeline", fontsize=13, fontweight="bold")

        # ── Timeline scatter ──────────────────────────────────────────
        y_positions = {"BASELINE_SHIFT": 3, "CIRCADIAN_DEVIATION": 2, "ACUTE_EVENT": 1}
        for atype, ypos in y_positions.items():
            subset = classifications[classifications["anomaly_type"] == atype]
            if subset.empty:
                continue
            ax_timeline.scatter(
                subset.index, [ypos] * len(subset),
                color=type_colors[atype], s=60, alpha=0.8, zorder=3,
                label=f"{atype.replace('_', ' ')} ({len(subset)})"
            )

        ax_timeline.set_yticks([1, 2, 3])
        ax_timeline.set_yticklabels(["Acute Event", "Circadian Deviation", "Baseline Shift"])
        ax_timeline.set_xlabel("Time")
        ax_timeline.legend(loc="upper right", fontsize=9)
        ax_timeline.grid(axis="x", alpha=0.3)

        # ── Count bar chart ───────────────────────────────────────────
        type_counts = classifications["anomaly_type"].value_counts()
        bar_colors = [type_colors.get(t, "#999") for t in type_counts.index]
        type_counts.plot(kind="bar", ax=ax_count, color=bar_colors, alpha=0.8, edgecolor="white")
        ax_count.set_title("Anomaly Type Distribution", fontsize=10)
        ax_count.set_xlabel("")
        ax_count.set_ylabel("Count")
        ax_count.tick_params(axis="x", rotation=20)

        plt.tight_layout()
        if save:
            path = self.output_dir / "06_anomaly_timeline.png"
            fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
            print(f"   💾 Saved: {path}")

        return fig

    # ──────────────────────────────────────────────────────────────────
    # 7. VARIANCE DECOMPOSITION BAR CHART
    # ──────────────────────────────────────────────────────────────────

    def plot_variance_decomposition(
        self,
        variance_df: pd.DataFrame,
        save: bool = True,
    ) -> plt.Figure:
        """Stacked bar chart showing trend/seasonal/residual variance % per vital."""
        if variance_df.empty:
            return plt.figure()

        plot_cols = [c for c in ["trend_var_%", "seasonal_var_%", "residual_var_%"]
                     if c in variance_df.columns]

        fig, ax = plt.subplots(figsize=(10, 5))

        vitals = variance_df.index.tolist()
        x = np.arange(len(vitals))
        width = 0.5
        bottoms = np.zeros(len(vitals))
        colors = ["#3498db", "#e67e22", "#e74c3c"]
        labels = ["Trend", "Seasonal", "Residual"]

        for col, color, label in zip(plot_cols, colors, labels):
            vals = variance_df[col].fillna(0).values
            ax.bar(x, vals, width, bottom=bottoms, color=color, alpha=0.82,
                   label=label, edgecolor="white")
            bottoms += vals

        ax.set_xticks(x)
        ax.set_xticklabels(
            [v.replace("_", "\n").title() for v in vitals], fontsize=9
        )
        ax.set_ylabel("Variance Contribution (%)", fontsize=10)
        ax.set_title("STL Variance Decomposition per Vital Sign", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)

        # Mark unstable vitals
        if "unstable_flag" in variance_df.columns:
            for i, (v, row) in enumerate(variance_df.iterrows()):
                if row.get("unstable_flag"):
                    ax.text(i, bottoms[i] + 1, "⚠️", ha="center", fontsize=10)

        plt.tight_layout()
        if save:
            path = self.output_dir / "07_variance_decomposition.png"
            fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
            print(f"   💾 Saved: {path}")

        return fig

    # ──────────────────────────────────────────────────────────────────
    # 8. LEAD-LAG CROSS-CORRELATION
    # ──────────────────────────────────────────────────────────────────

    def plot_lead_lag(
        self,
        lead_lag_df: pd.DataFrame,
        correlation_analyzer,
        top_n: int = 4,
        save: bool = True,
    ) -> plt.Figure:
        """Plot cross-correlation functions for the top N coupled signal pairs."""
        if lead_lag_df.empty:
            return plt.figure()

        top = lead_lag_df.head(top_n)
        n_pairs = len(top)
        fig, axes = plt.subplots(1, n_pairs, figsize=(5 * n_pairs, 4))
        if n_pairs == 1:
            axes = [axes]

        fig.suptitle("Cross-Correlation Functions (Lead-Lag Analysis)",
                     fontsize=12, fontweight="bold")

        for ax, (_, row) in zip(axes, top.iterrows()):
            xcorr_data = correlation_analyzer.get_xcorr_data(row["signal_a"], row["signal_b"])
            if xcorr_data is None:
                continue

            lags = xcorr_data["lags"]
            xcorr = xcorr_data["xcorr"]

            ax.bar(lags, xcorr, width=0.8, color="#3498db", alpha=0.7)
            ax.axvline(0, color="gray", linewidth=1, linestyle="--")
            ax.axhline(0, color="gray", linewidth=0.5)

            peak_lag = row["peak_lag_samples"]
            peak_corr = row["peak_correlation"]
            ax.axvline(peak_lag, color=self.anomaly_color, linewidth=1.5,
                       linestyle="-", label=f"Peak @ lag={peak_lag}")

            a_name = row["signal_a"].replace("_", " ").title()
            b_name = row["signal_b"].replace("_", " ").title()
            ax.set_title(f"{a_name}\n↔ {b_name}", fontsize=9)
            ax.set_xlabel("Lag (samples)")
            ax.set_ylabel("Cross-correlation")
            ax.legend(fontsize=8)
            ax.text(0.02, 0.92, f"r={peak_corr:.3f}", transform=ax.transAxes,
                    fontsize=9, color=self.anomaly_color)

        plt.tight_layout()
        if save:
            path = self.output_dir / "08_lead_lag.png"
            fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
            print(f"   💾 Saved: {path}")

        return fig

    # ──────────────────────────────────────────────────────────────────
    # 9. ENSEMBLE AGREEMENT HEATMAP
    # ──────────────────────────────────────────────────────────────────

    def plot_ensemble_agreement(
        self,
        kappa_matrix: pd.DataFrame,
        save: bool = True,
    ) -> plt.Figure:
        """Heatmap showing pairwise Cohen's Kappa between detectors."""
        if kappa_matrix.empty:
            return plt.figure()

        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            kappa_matrix.astype(float),
            ax=ax,
            annot=True,
            fmt=".3f",
            cmap="Blues",
            vmin=0,
            vmax=1,
            square=True,
            linewidths=0.5,
        )
        ax.set_title("Detector Agreement — Cohen's Kappa Matrix",
                     fontsize=12, fontweight="bold")
        plt.tight_layout()

        if save:
            path = self.output_dir / "09_ensemble_agreement.png"
            fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
            print(f"   💾 Saved: {path}")

        return fig

    def close_all(self):
        """Close all open matplotlib figures to free memory."""
        plt.close("all")
