import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import warnings

# Import pipeline components
import config
from data.generator import PhysioDataGenerator
from pipeline.main_pipeline import PhysioAnomalyPipeline
from preprocessing.cleaner import SignalCleaner

# Set page config for premium look
st.set_page_config(
    page_title="PhysioAnomaly AI — Clinical Diagnostics & Attribution Dashboard",
    layout="wide",
    page_icon="🏥",
    initial_sidebar_state="expanded"
)

# Custom Premium Styling
st.markdown("""
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
    
    /* Global styles */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Glow cards for metrics */
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
        text-align: center;
        transition: all 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(0, 150, 255, 0.15);
        border-color: rgba(0, 150, 255, 0.3);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 5px;
    }
    .metric-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #8892b0;
    }
    
    /* Alert badge styles */
    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 50px;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
    }
    .badge-stable { background-color: #27ae60; color: white; }
    .badge-borderline { background-color: #f39c12; color: white; }
    .badge-unstable { background-color: #c0392b; color: white; }
    .badge-critical { background-color: #c0392b; color: white; }
    .badge-high { background-color: #e67e22; color: white; }
    .badge-moderate { background-color: #2980b9; color: white; }
    .badge-normal { background-color: #27ae60; color: white; }
    
    /* Sidebar header */
    .sidebar-header {
        font-size: 1.2rem;
        font-weight: 700;
        margin-bottom: 15px;
        color: #3498db;
    }
</style>
""", unsafe_allow_html=True)

st.title("🏥 PhysioAnomaly AI")
st.subheader("Multivariate Physiological Anomaly Detection & Attribution Platform")

# ─────────────────────────────────────────────────────────────
# SIDEBAR: DATA GENERATION & PIPELINE SETTINGS
# ─────────────────────────────────────────────────────────────
st.sidebar.markdown("<div class='sidebar-header'>🔌 Diagnostics Controller</div>", unsafe_allow_html=True)

# Data source selection
data_source = st.sidebar.selectbox("Data Source", ["Synthetic Generator", "Upload CSV File"])

raw_df = None
patient_id = "P001"

if data_source == "Synthetic Generator":
    with st.sidebar.expander("🛠️ Generator Parameters", expanded=True):
        patient_id = st.text_input("Patient ID", "P001")
        seed = st.number_input("Random Seed", value=42)
        duration_hours = st.slider("Duration (Hours)", 4, 48, 24, step=4)
        noise_level = st.slider("Signal Noise Level (std)", 0.0, 0.1, 0.02, step=0.01)
        anomaly_fraction = st.slider("Anomaly Fraction", 0.0, 0.2, 0.05, step=0.01)
        
        st.markdown("**Injected Anomalies:**")
        inject_types = []
        if st.checkbox("Spikes (Acute Events)", value=True): inject_types.append("spike")
        if st.checkbox("Drifts (Baseline Shifts)", value=True): inject_types.append("drift")
        if st.checkbox("Dropouts (Sensor Disconnects)", value=True): inject_types.append("dropout")
        if st.checkbox("Noise Bursts (Motion Artifacts)", value=True): inject_types.append("noise_burst")
        if st.checkbox("Combined Crises", value=True): inject_types.append("combined")

    # Update global config dynamically
    config.DATA_GENERATION["random_seed"] = seed
    config.DATA_GENERATION["duration_hours"] = duration_hours
    config.DATA_GENERATION["noise_level"] = noise_level
    config.DATA_GENERATION["anomaly_fraction"] = anomaly_fraction
    config.DATA_GENERATION["anomaly_types"] = inject_types

else:
    uploaded_file = st.sidebar.file_uploader("Upload Patient Vital Sign CSV", type=["csv"])
    if uploaded_file is not None:
        raw_df = pd.read_csv(uploaded_file)
        st.sidebar.success("CSV Uploaded successfully!")
        
        # Map columns
        with st.sidebar.expander("🗺️ Column Mapping", expanded=True):
            st.markdown("Map your CSV columns to the clinical vitals:")
            col_map = {}
            for vital in config.VITAL_SIGNS.keys():
                default_choice = 0
                for i, col in enumerate(raw_df.columns):
                    if vital.lower() in col.lower():
                        default_choice = i
                        break
                selected_col = st.selectbox(f"{vital} ({config.VITAL_SIGNS[vital]['unit']})", raw_df.columns, index=default_choice)
                col_map[selected_col] = vital
    else:
        st.sidebar.info("Please upload a CSV file to continue.")

with st.sidebar.expander("🧠 Detector Parameters", expanded=False):
    zscore_thresh = st.slider("Statistical Z-Score Threshold", 1.5, 5.0, 3.0, step=0.5)
    if_contamination = st.slider("Isolation Forest Contamination", 0.01, 0.2, 0.05, step=0.01)
    ensemble_votes = st.slider("Ensemble Vote Threshold", 1, 3, 2)
    use_lstm = st.checkbox("Include LSTM Autoencoder", value=False)
    
    if use_lstm:
        st.warning("⚠️ Training LSTM Autoencoder on the fly will take ~20-30 seconds.")

    # Apply detector settings dynamically
    config.DETECTION["zscore_threshold"] = zscore_thresh
    config.DETECTION["isolation_forest"]["contamination"] = if_contamination
    config.DETECTION["ensemble_min_votes"] = ensemble_votes

# Diagnostics run button
run_pipeline = st.sidebar.button("⚡ Run Diagnostics Pipeline", use_container_width=True)

# ─────────────────────────────────────────────────────────────
# PIPELINE EXECUTION & CACHING
# ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def execute_diagnostics(patient_id, data_source, use_lstm, config_hash, csv_data=None, col_mapping=None):
    pipeline = PhysioAnomalyPipeline(
        patient_id=patient_id,
        output_dir="outputs",
        use_lstm=use_lstm,
        verbose=False
    )
    if data_source == "Synthetic Generator" or csv_data is None:
        results = pipeline.run(data_source="synthetic")
    else:
        # Save temp CSV
        temp_path = "outputs/temp_uploaded.csv"
        csv_data.to_csv(temp_path, index=False)
        results = pipeline.run(
            data_path=temp_path,
            data_source="csv",
            column_map=col_mapping
        )
    return results

# Track configuration changes to invalidate cache
config_hash = (
    config.DATA_GENERATION.get("random_seed"),
    config.DATA_GENERATION.get("duration_hours"),
    config.DATA_GENERATION.get("noise_level"),
    config.DATA_GENERATION.get("anomaly_fraction"),
    str(config.DATA_GENERATION.get("anomaly_types")),
    config.DETECTION.get("zscore_threshold"),
    config.DETECTION.get("isolation_forest", {}).get("contamination"),
    config.DETECTION.get("ensemble_min_votes")
)

# Run default generation if first load
if "pipeline_results" not in st.session_state or run_pipeline:
    if data_source == "Upload CSV File" and raw_df is None:
        st.error("Please upload a CSV vital signs file before running the pipeline.")
    else:
        with st.spinner("🏥 Orchestrating Diagnostics Pipeline... Running cleaning, STL decomposition, wavelets, Isolation Forest, SHAP, and causal inference..."):
            try:
                results = execute_diagnostics(
                    patient_id=patient_id,
                    data_source=data_source,
                    use_lstm=use_lstm,
                    config_hash=config_hash,
                    csv_data=raw_df,
                    col_mapping=col_map if data_source == "Upload CSV File" else None
                )
                st.session_state["pipeline_results"] = results
                st.success("🎉 Pipeline executed successfully!")
            except Exception as e:
                st.error(f"Error executing pipeline: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

# ─────────────────────────────────────────────────────────────
# VIEW RENDER
# ─────────────────────────────────────────────────────────────
if "pipeline_results" in st.session_state:
    results = st.session_state["pipeline_results"]
    clean_df = results["clean_data"]
    raw_df = results["raw_data"]
    ensemble_flags = results["ensemble_flags"]
    anomaly_classifications = results["anomaly_classifications"]
    stability = results["clinical_report"].get("stability_assessment", {})
    vitals_status = results["clinical_report"].get("vital_status", {})
    active_alerts = results["clinical_report"].get("active_alerts", [])
    
    # ── METRIC DASHBOARD OVERVIEW ─────────────────────────────
    # Get stability label & badge
    stab_label = stability.get("overall_stability", "STABLE")
    if stab_label == "STABLE":
        stab_badge = "<span class='badge badge-stable'>Stable</span>"
    elif stab_label == "BORDERLINE":
        stab_badge = "<span class='badge badge-borderline'>Borderline</span>"
    else:
        stab_badge = "<span class='badge badge-unstable'>Unstable</span>"

    # Row 1 KPI cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value'>{results['patient_id']}</div>
            <div class='metric-label'>Patient ID</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        num_anom = ensemble_flags.sum()
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value'>{num_anom}</div>
            <div class='metric-label'>Detected Anomalies (mins)</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value'>{stab_badge}</div>
            <div class='metric-label'>Signal Stability ({stability.get('avg_residual_variance_pct', 0)}% Resid)</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        driver = results["clinical_report"].get("summary", {}).get("primary_driver", "None")
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value' style='color:#e74c3c;'>{driver}</div>
            <div class='metric-label'>Primary Anomaly Driver</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── TABS LAYOUT ──────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📋 Clinical Report & Timeline",
        "📊 STL Decomposition & Signals",
        "🧠 Attribution & SHAP",
        "🔀 Physiological Coupling",
        "🧹 Artifact Correction"
    ])

    # ─────────────────────────────────────────────────────────────
    # TAB 1: CLINICAL REPORT & TIMELINE
    # ─────────────────────────────────────────────────────────────
    with tab1:
        st.header("📋 Clinical Diagnostics & Timeline")
        
        # Clinical Alerts
        col_alert, col_rec = st.columns(2)
        with col_alert:
            st.subheader("🚨 Clinical Rule Alerts")
            if active_alerts:
                for alert in active_alerts:
                    st.markdown(f"""
                    <div style='background-color:rgba(192, 57, 43, 0.15); border-left: 5px solid #c0392b; padding: 15px; border-radius: 4px; margin-bottom: 10px;'>
                        <span class='badge badge-critical'>{alert['severity']}</span> <b>{alert['description']}</b>
                        <div style='margin-top: 5px; font-size: 0.9rem;'>{alert['explanation']}</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("✅ No active rules-based clinical alerts. Patient parameters are within baseline limits.")

        with col_rec:
            st.subheader("💡 Recommendations")
            recs = results["clinical_report"].get("recommendations", [])
            for rec in recs:
                st.markdown(f"**{rec}**")

        st.markdown("<hr>", unsafe_allow_html=True)
        st.subheader("⏱️ Temporal Event Timeline")
        timeline = results["clinical_report"].get("timeline", [])
        
        if timeline:
            # Render a styled list of collapsed events
            for i, ev in enumerate(timeline, 1):
                badge_class = f"badge-{ev['severity'].lower()}"
                st.markdown(f"""
                <div style='background: rgba(255, 255, 255, 0.02); border: 1px solid rgba(255, 255, 255, 0.05); border-radius: 8px; padding: 15px; margin-bottom: 12px;'>
                    <div style='display: flex; justify-content: space-between; align-items: center;'>
                        <div>
                            <span class='badge {badge_class}'>{ev['severity']}</span>
                            <b style='font-size:1.1rem; margin-left: 10px;'>{ev['event_type']} in {ev['dominant_vital']}</b>
                        </div>
                        <div style='color: #8892b0; font-size:0.9rem;'>⏰ <b>{ev['time_span_str']}</b> (Peak Z: {ev['peak_z']:.1f})</div>
                    </div>
                    <div style='margin-top: 8px; color: #a8b2d1; font-size: 0.95rem;'>{ev['description']}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No timeline events found.")

    # ─────────────────────────────────────────────────────────────
    # TAB 2: STL DECOMPOSITION & SIGNALS
    # ─────────────────────────────────────────────────────────────
    with tab2:
        st.header("📊 Interactive STL Decomposition & Residual Analysis")
        st.markdown("""
        Select a vital sign to review its STL components.
        - **Trend**: Long-term changes or baseline shifts.
        - **Seasonal**: Daily circadian oscillations.
        - **Residual**: High-frequency irregular events.
        """)

        vital_cols = [c for c in clean_df.columns if c in config.VITAL_SIGNS]
        selected_vital = st.selectbox("Select Vital Sign to Decompose", vital_cols)

        if selected_vital in results["stl_components"]:
            comp = results["stl_components"][selected_vital]
            
            # Anomaly points for highlight
            anom_points = comp[results["ensemble_flags"]]

            # Plot Observed
            fig_obs = go.Figure()
            fig_obs.add_trace(go.Scatter(x=comp.index, y=comp['observed'], name="Observed", line=dict(color=config.VITAL_SIGNS[selected_vital]["color"])))
            fig_obs.add_trace(go.Scatter(x=anom_points.index, y=anom_points['observed'], mode='markers', name="Anomaly", marker=dict(color='red', size=8)))
            fig_obs.update_layout(title="Observed Signal", margin=dict(t=30, b=10), height=220)

            # Plot Trend
            fig_trend = go.Figure()
            fig_trend.add_trace(go.Scatter(x=comp.index, y=comp['trend'], name="Trend", line=dict(color='#f1c40f')))
            fig_trend.add_trace(go.Scatter(x=anom_points.index, y=anom_points['trend'], mode='markers', name="Anomaly Trend", marker=dict(color='red', size=8)))
            fig_trend.update_layout(title="Trend Component (Baseline Drift)", margin=dict(t=30, b=10), height=200)

            # Plot Seasonal
            fig_sea = go.Figure()
            fig_sea.add_trace(go.Scatter(x=comp.index, y=comp['seasonal'], name="Seasonal", line=dict(color='#2ecc71')))
            fig_sea.update_layout(title="Seasonal Component (Circadian Profile)", margin=dict(t=30, b=10), height=200)

            # Plot Residual
            fig_res = go.Figure()
            fig_res.add_trace(go.Scatter(x=comp.index, y=comp['residual'], name="Residual", line=dict(color='#95a5a6')))
            fig_res.add_trace(go.Scatter(x=anom_points.index, y=anom_points['residual'], mode='markers', name="Anomaly Residual", marker=dict(color='red', size=8)))
            fig_res.update_layout(title="Residual Component (Acute Spikes / Irregularity)", margin=dict(t=30, b=10), height=200)

            st.plotly_chart(fig_obs, use_container_width=True)
            st.plotly_chart(fig_trend, use_container_width=True)
            st.plotly_chart(fig_sea, use_container_width=True)
            st.plotly_chart(fig_res, use_container_width=True)

    # ─────────────────────────────────────────────────────────────
    # TAB 3: ATTRIBUTION & SHAP
    # ─────────────────────────────────────────────────────────────
    with tab3:
        st.header("🧠 SHAP Feature Attribution & Anomaly Classification")
        
        # Plotly plot of raw signals with anomalies
        st.subheader("🔍 Vital Sign Anomaly Highlights")
        vitals_to_plot = st.multiselect("Vitals to plot", vital_cols, default=vital_cols[:3])
        
        if vitals_to_plot:
            fig_vitals = go.Figure()
            for v in vitals_to_plot:
                fig_vitals.add_trace(go.Scatter(
                    x=clean_df.index, y=clean_df[v],
                    name=v,
                    line=dict(color=config.VITAL_SIGNS[v]["color"])
                ))
                # Add anomalies for this vital
                v_anom = clean_df.loc[ensemble_flags & (anomaly_classifications["dominant_vital"] == v)]
                if not v_anom.empty:
                    fig_vitals.add_trace(go.Scatter(
                        x=v_anom.index, y=v_anom[v],
                        mode='markers',
                        name=f"{v} Anomaly",
                        marker=dict(color='red', size=9, symbol='x')
                    ))
            fig_vitals.update_layout(title="Cleaned Patient Signals with Highlighted Anomalies", height=400)
            st.plotly_chart(fig_vitals, use_container_width=True)

        st.markdown("<hr>", unsafe_allow_html=True)

        col_importance, col_table = st.columns([2, 3])

        
        
        with col_importance:
            st.subheader("📊 Global SHAP Feature Importance")
            importance_df = results["feature_importance"]
            if not importance_df.empty:
                imp_plot_df = importance_df.reset_index()

                # Pick the best available importance metric column.
                # `FeatureImportanceAnalyzer` outputs one or more of these columns.
                preferred_cols = [
                    ("combined_score", "Combined importance score"),
                    ("shap_importance", "SHAP importance"),
                    ("rf_importance", "RandomForest importance"),
                    ("variance_importance", "Variance importance"),
                ]
                imp_col, imp_label = next(
                    ((c, lbl) for (c, lbl) in preferred_cols if c in imp_plot_df.columns),
                    (None, None),
                )

                if imp_col is None:
                    st.info(
                        "Feature importance is available, but no plottable importance column was found. "
                        f"Columns present: {list(imp_plot_df.columns)}"
                    )
                else:
                    # Plotly Bar chart
                    fig_imp = px.bar(
                        imp_plot_df.sort_values(imp_col, ascending=True),
                        x=imp_col,
                        y="index",
                        orientation="h",
                        title="Global Feature Importance",
                        labels={"index": "Feature Name", imp_col: imp_label},
                        color=imp_col,
                        color_continuous_scale="Reds",
                    )
                fig_imp.update_layout(yaxis={'categoryorder':'total ascending'}, height=450)
                st.plotly_chart(fig_imp, use_container_width=True)
            else:
                st.info("No anomalies detected; SHAP importance is empty.")

        with col_table:
            st.subheader("🎯 Classifications Table")
            if not anomaly_classifications.empty:
                # Format index and display
                disp_df = anomaly_classifications.copy()
                disp_df.index = disp_df.index.strftime("%H:%M")
                
                # Filter columns for cleaner view
                cols_to_show = ["dominant_vital", "anomaly_type", "trend_z", "seasonal_z", "residual_z"]
                cols_to_show = [c for c in cols_to_show if c in disp_df.columns]
                
                st.dataframe(disp_df[cols_to_show].head(30), use_container_width=True)
                st.caption("Showing first 30 anomalies. Z-scores represent the standard deviations of each component deviation.")
            else:
                st.info("No anomalies detected.")

    # ─────────────────────────────────────────────────────────────
    # TAB 4: PHYSIOLOGICAL COUPLING
    # ─────────────────────────────────────────────────────────────
    with tab4:
        st.header("🔀 Physiological Inter-Signal Coupling & Granger Causality")
        
        col_corr, col_causality = st.columns(2)
        
        with col_corr:
            st.subheader("🔥 Cross-Correlation Heatmap")
            corr_matrix = results["correlation_matrix"]
            if corr_matrix is not None:
                fig_heat = px.imshow(
                    corr_matrix,
                    x=corr_matrix.columns,
                    y=corr_matrix.index,
                    color_continuous_scale="RdBu_r",
                    zmin=-1.0, zmax=1.0,
                    title="Vital Sign Inter-Correlation Matrix"
                )
                st.plotly_chart(fig_heat, use_container_width=True)

        with col_causality:
            st.subheader("🔄 Directed Causality Network (Granger)")
            granger_df = results["granger"]
            if granger_df is not None and not granger_df.empty:
                # Build network nodes
                sig_df = granger_df[granger_df["significant"]] if "significant" in granger_df.columns else pd.DataFrame()
                
                num_nodes = len(vital_cols)
                angles = np.linspace(0, 2*np.pi, num_nodes, endpoint=False)
                pos = {v: (np.cos(a), np.sin(a)) for v, a in zip(vital_cols, angles)}
                
                node_x = []
                node_y = []
                node_text = []
                node_color = []
                for node, (x, y) in pos.items():
                    node_x.append(x)
                    node_y.append(y)
                    node_text.append(node)
                    node_color.append(config.VITAL_SIGNS[node]["color"])
                    
                node_trace = go.Scatter(
                    x=node_x, y=node_y,
                    mode='markers+text',
                    text=node_text,
                    textposition="top center",
                    hoverinfo='text',
                    marker=dict(
                        showscale=False,
                        color=node_color,
                        size=28,
                        line=dict(color='white', width=2)
                    )
                )
                
                edge_traces = []
                for _, row in sig_df.iterrows():
                    cause = row["cause"]
                    effect = row["effect"]
                    if cause in pos and effect in pos:
                        x0, y0 = pos[cause]
                        x1, y1 = pos[effect]
                        
                        # Edge path
                        edge_traces.append(go.Scatter(
                            x=[x0, x1, None],
                            y=[y0, y1, None],
                            mode='lines',
                            line=dict(width=1.5, color='rgba(255,255,255,0.4)'),
                            hoverinfo='none'
                        ))
                        
                        # Arrow tip at midpoint
                        mx, my = (x0 + 2*x1)/3.0, (y0 + 2*y1)/3.0
                        edge_traces.append(go.Scatter(
                            x=[mx], y=[my],
                            mode='markers',
                            marker=dict(symbol='triangle-up', size=8, color='#e74c3c'),
                            hoverinfo='text',
                            text=f"{cause} → {effect} (lag {row['best_lag']}m)"
                        ))
                        
                fig_net = go.Figure(
                    data=edge_traces + [node_trace],
                    layout=go.Layout(
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=20, r=20, t=20),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        height=400,
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)'
                    )
                )
                st.plotly_chart(fig_net, use_container_width=True)
            else:
                st.info("No significant causal relationships found.")

        st.markdown("<hr>", unsafe_allow_html=True)
        st.subheader("⏳ Lead-Lag & Granger Causality Metrics")
        col_lead, col_gran = st.columns(2)
        
        with col_lead:
            st.markdown("**Leader-Follower Pairs (Cross-Correlation):**")
            st.dataframe(results["lead_lag"].head(10), use_container_width=True)
            
        with col_gran:
            st.markdown("**Causal Interactions (Granger Significance):**")
            if granger_df is not None and not granger_df.empty:
                # `GrangerCausalityAnalyzer` outputs `min_pvalue` (not `p-value`).
                sig = granger_df[granger_df["significant"]] if "significant" in granger_df.columns else granger_df
                cols = [c for c in ["cause", "effect", "best_lag", "min_pvalue"] if c in sig.columns]
                st.dataframe(sig[cols].head(10), use_container_width=True)
            else:
                st.info("No Granger relationships.")

    # ─────────────────────────────────────────────────────────────
    # TAB 5: ARTIFACT CORRECTION
    # ─────────────────────────────────────────────────────────────
    with tab5:
        st.header("🧹 Artifact Correction & Preprocessing Quality")
        st.markdown("""
        The preprocessing pipeline explicitly isolates and imputes two clinical artifact types:
        - **Sensor Disconnects**: Drops to 0 or flatline NaNs.
        - **Motion Artifacts**: Extreme physiological jumps.
        """)

        # Clean report table
        clean_report = results["clinical_report"].get("clean_report")
        if clean_report and "vitals" in clean_report:
            cleaning_data = []
            for vital, info in clean_report["vitals"].items():
                cleaning_data.append({
                    "Vital Sign": vital,
                    "Sensor Disconnects (Fixed)": info.get("sensor_disconnects", 0),
                    "Motion Noise (Imputed)": info.get("motion_artifacts", 0),
                    "Impossible Values (Clipped)": info.get("impossible_clipped", 0),
                    "Final Imputation Method": config.PREPROCESSING["interpolation_method"]
                })
            st.table(pd.DataFrame(cleaning_data))

        # Show Before & After raw vs clean comparison
        st.subheader("🔄 Imputation Comparison Plot")
        clean_vital_select = st.selectbox("Select Vital Sign to Compare Imputations", vital_cols)
        
        if clean_vital_select in raw_df.columns:
            fig_compare = go.Figure()
            fig_compare.add_trace(go.Scatter(
                x=raw_df.index, y=raw_df[clean_vital_select],
                name="Raw Signal (with Artifacts)",
                line=dict(color='rgba(231, 76, 60, 0.4)', dash='dot')
            ))
            fig_compare.add_trace(go.Scatter(
                x=clean_df.index, y=clean_df[clean_vital_select],
                name="Cleaned & Smoothed Signal",
                line=dict(color=config.VITAL_SIGNS[clean_vital_select]["color"], width=2)
            ))
            fig_compare.update_layout(
                title=f"Raw vs. Imputed/Smoothed Signal for {clean_vital_select}",
                height=400,
                xaxis_title="Time",
                yaxis_title=config.VITAL_SIGNS[clean_vital_select]["unit"]
            )
            st.plotly_chart(fig_compare, use_container_width=True)
