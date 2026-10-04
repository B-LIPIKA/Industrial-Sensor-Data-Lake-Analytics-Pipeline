import os
import time
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sample_data_generator import generate_industrial_sensor_dataset
from data_pipeline.ingestion import IngestionPipeline
from analytics.duckdb_engine import DuckDBAnalyticsEngine
from query_engine.nl_to_sql import NaturalLanguageQueryEngine

# --- Streamlit Page Config ---
st.set_page_config(
    page_title="Industrial Sensor Data Lake & Analytics",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom Styling (Dark Industrial Glassmorphism Theme) ---
st.markdown("""
<style>
    /* Global Styling */
    .stApp {
        background-color: #0d1117;
        color: #e6edf3;
        font-family: 'DM Sans', system-ui, -apple-system, sans-serif;
    }
    
    /* Header Banner */
    .header-container {
        background: linear-gradient(135deg, #161b22 0%, #21262d 100%);
        padding: 24px;
        border-radius: 12px;
        border: 1px solid #30363d;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    }
    
    .header-title {
        font-size: 28px;
        font-weight: 700;
        color: #58a6ff;
        margin: 0;
    }
    
    .header-subtitle {
        font-size: 14px;
        color: #8b949e;
        margin-top: 4px;
    }

    /* Metric Cards */
    .kpi-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 18px;
        text-align: center;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .kpi-card:hover {
        border-color: #58a6ff;
        transform: translateY(-2px);
    }
    .kpi-value {
        font-size: 26px;
        font-weight: 700;
        color: #f0f6fc;
        font-family: 'JetBrains Mono', monospace;
    }
    .kpi-label {
        font-size: 12px;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 4px;
    }

    /* SQL Code Block */
    .sql-box {
        background-color: #161b22;
        border-left: 4px solid #58a6ff;
        padding: 12px 16px;
        border-radius: 6px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 13px;
        color: #79c0ff;
    }

    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #161b22;
        border-radius: 8px;
        color: #8b949e;
        padding: 8px 16px;
        border: 1px solid #30363d;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f6feb !important;
        color: #ffffff !important;
        border-color: #58a6ff !important;
    }
</style>
""", unsafe_allow_html=True)

# --- Initialize Pipeline & Engines ---
DATA_LAKE_DIR = "data/lake"
SAMPLE_CSV_PATH = "data/sample/industrial_sensor_data.csv"

@st.cache_resource
def get_pipeline():
    return IngestionPipeline(lake_dir=DATA_LAKE_DIR)

@st.cache_resource
def get_analytics_engine():
    return DuckDBAnalyticsEngine(lake_dir=DATA_LAKE_DIR)

pipeline = get_pipeline()
analytics_engine = get_analytics_engine()

# Ensure initial sample dataset exists if data lake is empty
if not os.path.exists(DATA_LAKE_DIR) or not pipeline.storage.list_all_parquet_files():
    if not os.path.exists(SAMPLE_CSV_PATH):
        generate_industrial_sensor_dataset(output_path=SAMPLE_CSV_PATH)
    pipeline.process_file_or_dataframe(SAMPLE_CSV_PATH)
    analytics_engine.refresh_views()

# --- Sidebar ---
st.sidebar.image("https://img.icons8.com/color/96/000000/industrial-scales.png", width=60)
st.sidebar.title("Industrial Data Lake")
st.sidebar.caption("Python · DuckDB · Parquet · PyArrow · LangChain")

st.sidebar.markdown("---")
st.sidebar.subheader("🔌 API Configuration")
api_key_input = st.sidebar.text_input("Gemini API Key (Optional)", type="password", help="Enter Google Gemini API Key for LLM SQL translation. If left empty, built-in smart rule engine is used.")
if api_key_input:
    os.environ["GEMINI_API_KEY"] = api_key_input

nl_engine = NaturalLanguageQueryEngine(analytics_engine, api_key=api_key_input)

st.sidebar.markdown("---")
st.sidebar.subheader("💾 Lake Storage Status")
stats = pipeline.storage.get_storage_stats()
st.sidebar.metric("Parquet Files", stats["file_count"])
st.sidebar.metric("Total Lake Size", f"{stats['total_size_mb']} MB")

# --- Main App Header ---
st.markdown("""
<div class="header-container">
    <div class="header-title">🏭 Industrial Sensor Data Lake & Analytics Engine</div>
    <div class="header-subtitle">Incremental Ingestion Pipeline · Partitioned Parquet Lake · DuckDB SQL Rollups · LangChain NL Query UI</div>
</div>
""", unsafe_allow_html=True)

# Global KPI Bar
kpis = analytics_engine.get_kpi_summary()
c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    st.markdown(f'<div class="kpi-card"><div class="kpi-value">{kpis["total_records"]:,}</div><div class="kpi-label">Total Telemetry Records</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="kpi-card"><div class="kpi-value">{kpis["machines_monitored"]}</div><div class="kpi-label">Active Machines</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown(f'<div class="kpi-card"><div class="kpi-value">{kpis["overall_health_pct"]}%</div><div class="kpi-label">Operational Health</div></div>', unsafe_allow_html=True)
with c4:
    st.markdown(f'<div class="kpi-card"><div class="kpi-value">{kpis["total_anomalies"]:,}</div><div class="kpi-label">Detected Anomalies</div></div>', unsafe_allow_html=True)
with c5:
    st.markdown(f'<div class="kpi-card"><div class="kpi-value">{kpis["date_range"][0]} to {kpis["date_range"][1]}</div><div class="kpi-label">Date Horizon</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# --- Navigation Tabs ---
tab_analytics, tab_ingest, tab_nl_query, tab_sql = st.tabs([
    "📊 Executive Analytics", 
    "📤 Data Ingestion & Quality", 
    "💬 AI Natural Language Query", 
    "⚡ DuckDB SQL Workbench"
])

# ==========================================
# TAB 1: EXECUTIVE ANALYTICS
# ==========================================
with tab_analytics:
    st.subheader("Sensor Telemetry & Rollup Analytics")
    
    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([1, 1, 1])
    with col_ctrl1:
        machine_opt = st.selectbox("Select Machine", ["All Machines", "PUMP-01", "COMPRESSOR-02", "TURBINE-03", "CONVEYOR-04"])
    with col_ctrl2:
        sensor_type_opt = st.selectbox("Select Sensor Type", ["flow_rate", "temperature", "pressure", "vibration", "power_consumption"])
    with col_ctrl3:
        rollup_level = st.radio("Analytics Aggregation Level", ["Raw Telemetry", "Hourly Rollup", "Daily Rollup"], horizontal=True)

    # Build SQL Query based on selections
    where_clause = f"WHERE sensor_type = '{sensor_type_opt}'"
    if machine_opt != "All Machines":
        where_clause += f" AND machine_id = '{machine_opt}'"

    if rollup_level == "Daily Rollup":
        sql = f"SELECT date, machine_id, avg_value, min_value, max_value, anomaly_count FROM daily_sensor_rollups {where_clause} ORDER BY date ASC;"
        df_chart = analytics_engine.execute_sql(sql)
        if not df_chart.empty:
            fig = px.line(
                df_chart, x="date", y="avg_value", color="machine_id",
                title=f"Daily Average {sensor_type_opt.upper()} Trend",
                labels={"avg_value": "Avg Reading", "date": "Date"},
                template="plotly_dark"
            )
            st.plotly_chart(fig, width="stretch")
    elif rollup_level == "Hourly Rollup":
        sql = f"SELECT date, hour, machine_id, avg_value, max_value, anomaly_count FROM hourly_sensor_rollups {where_clause} ORDER BY date ASC, hour ASC LIMIT 500;"
        df_chart = analytics_engine.execute_sql(sql)
        if not df_chart.empty:
            df_chart["datetime_str"] = df_chart["date"] + " " + df_chart["hour"].astype(str).str.zfill(2) + ":00"
            fig = px.line(
                df_chart, x="datetime_str", y="avg_value", color="machine_id",
                title=f"Hourly Average {sensor_type_opt.upper()} Trend",
                labels={"avg_value": "Avg Reading", "datetime_str": "Timestamp"},
                template="plotly_dark"
            )
            st.plotly_chart(fig, width="stretch")
    else:
        sql = f"SELECT timestamp, machine_id, sensor_id, value, status FROM raw_sensor_data {where_clause} ORDER BY timestamp ASC LIMIT 1000;"
        df_chart = analytics_engine.execute_sql(sql)
        if not df_chart.empty:
            fig = px.scatter(
                df_chart, x="timestamp", y="value", color="status", symbol="machine_id",
                color_discrete_map={"OK": "#2da44e", "WARNING": "#d97706", "CRITICAL": "#cf222e"},
                title=f"Raw Sensor Readings: {sensor_type_opt.upper()}",
                template="plotly_dark"
            )
            st.plotly_chart(fig, width="stretch")

    # Machine Health Table
    st.subheader("Machine Health Overview")
    df_health = analytics_engine.execute_sql("SELECT * FROM machine_health_summary ORDER BY health_score_pct DESC;")
    st.dataframe(df_health, width="stretch")

# ==========================================
# TAB 2: INGESTION & DATA QUALITY HUB
# ==========================================
with tab_ingest:
    st.subheader("Incremental Data Ingestion & Quality Audit")
    
    col_u1, col_u2 = st.columns([2, 1])
    
    with col_u1:
        uploaded_file = st.file_uploader("Upload Sensor Dataset (CSV or Parquet)", type=["csv", "parquet"])
        if uploaded_file is not None:
            if st.button("🚀 Ingest Uploaded File into Data Lake"):
                with st.spinner("Executing Data Quality audit and partitioning Parquet storage..."):
                    success, res = pipeline.process_file_or_dataframe(uploaded_file)
                    if success:
                        st.success(f"Successfully ingested {res['processed_records']:,} records into Hive Parquet Data Lake!")
                        analytics_engine.refresh_views()
                        st.json(res["storage_summary"])
                    else:
                        st.error(f"Data Quality Validation Failed: {res['quality_report']}")

    with col_u2:
        st.markdown("#### ⚡ Quick Actions")
        if st.button("🎲 Generate & Ingest 3-Month Dataset"):
            with st.spinner("Generating synthetic industrial telemetry dataset..."):
                df_synth = generate_industrial_sensor_dataset(output_path=SAMPLE_CSV_PATH)
                success, res = pipeline.process_file_or_dataframe(df_synth)
                if success:
                    analytics_engine.refresh_views()
                    st.success(f"Generated & Ingested {len(df_synth):,} records!")
                    st.rerun()

    st.markdown("---")
    st.subheader("🛡️ Data Quality Validation Benchmark")
    df_sample = analytics_engine.execute_sql("SELECT * FROM raw_sensor_data LIMIT 100;")
    if not df_sample.empty:
        from data_pipeline.quality import DataQualityEngine
        passed, q_report, _ = DataQualityEngine.validate_dataset(df_sample)
        
        q1, q2, q3, q4 = st.columns(4)
        q1.metric("Data Quality Score", f"{q_report['quality_score']}%", delta="PASS" if passed else "FAIL")
        q2.metric("Missing Value Cells", sum(q_report['missing_values'].values()))
        q3.metric("Out-of-Range Sensor Values", q_report['out_of_range_count'])
        q4.metric("Duplicate Telemetry Rows", q_report['duplicate_count'])

# ==========================================
# TAB 3: AI NATURAL LANGUAGE QUERY ASSISTANT
# ==========================================
with tab_nl_query:
    st.subheader("💬 Ask Any Question Over The Industrial Sensor Data")
    st.caption("Converts natural language user requests into DuckDB SQL over partitioned Parquet files and visualizes results.")

    # Preset Question Chips
    st.markdown("##### 💡 Example Questions (Click to test):")
    chip_cols = st.columns(3)
    preset_q = None
    with chip_cols[0]:
        if st.button("🌊 Flow rate from Jan to Feb till 10pm"):
            preset_q = "from jan to feb till 10pm what was the flow"
        if st.button("🌡️ Max temp on PUMP-01 in February"):
            preset_q = "What was the max temperature on PUMP-01 in February?"
    with chip_cols[1]:
        if st.button("⚡ Hourly power for TURBINE-03"):
            preset_q = "Show hourly power consumption trend for TURBINE-03"
        if st.button("⚠️ Top 10 vibration anomalies"):
            preset_q = "Find top 10 vibration anomalies across all machines"
    with chip_cols[2]:
        if st.button("🏥 Operational machine health summary"):
            preset_q = "Show overall machine operational health summary"

    user_query = st.text_input("Enter your question about the dataset:", value=preset_q if preset_q else "from jan to feb till 10pm what was the flow")

    if st.button("🔍 Execute Natural Language Query") or preset_q:
        with st.spinner("Translating NL request to validated DuckDB SQL and analyzing dataset..."):
            start_t = time.time()
            result = nl_engine.query(user_query)
            exec_time = round((time.time() - start_t) * 1000, 2)

        if result["success"]:
            st.markdown(f"**Engine Used:** `{'Gemini LLM (LangChain)' if result['used_llm'] else 'DuckDB Hybrid SQL Engine'}` | **Execution Time:** `{exec_time} ms` | **Rows Returned:** `{result['row_count']}`")
            
            # Answer Box
            st.info(f"💡 **Answer Summary:** {result['explanation']}")
            
            # SQL Query
            st.markdown("##### 📜 Generated DuckDB SQL Query")
            st.code(result["sql"], language="sql")
            
            # Visualizations
            df_res = result["result_df"]
            if df_res is not None and not df_res.empty:
                st.markdown("##### 📈 Data Visualization")
                
                chart_type = result["chart_type"]
                cols = df_res.columns.tolist()
                
                if chart_type == "time_series" and len(df_res) > 1:
                    x_col = "date" if "date" in cols else ("timestamp" if "timestamp" in cols else cols[0])
                    y_col = "avg_value" if "avg_value" in cols else ("value" if "value" in cols else cols[1])
                    color_col = "machine_id" if "machine_id" in cols else None
                    
                    fig = px.line(
                        df_res, x=x_col, y=y_col, color=color_col,
                        title=f"Time-Series: {user_query}",
                        template="plotly_dark"
                    )
                    st.plotly_chart(fig, width="stretch")
                elif chart_type == "bar_chart":
                    x_col = "machine_id" if "machine_id" in cols else cols[0]
                    y_col = "health_score_pct" if "health_score_pct" in cols else cols[1]
                    fig = px.bar(df_res, x=x_col, y=y_col, title="Machine Health Comparison", template="plotly_dark")
                    st.plotly_chart(fig, width="stretch")
                elif chart_type == "metric_card":
                    st.metric(label="Calculated Result", value=f"{df_res.iloc[0].get('avg_value', df_res.iloc[0].iloc[0])}")

                # Results Table
                st.markdown("##### 📋 Result Data Table")
                st.dataframe(df_res, width="stretch")
        else:
            st.error(f"Error Processing Query: {result['error']}")

# ==========================================
# TAB 4: DUCKDB SQL WORKBENCH
# ==========================================
with tab_sql:
    st.subheader("⚡ Direct DuckDB SQL Workbench over Partitioned Parquet Lake")
    
    st.markdown("Available Tables & Views: `raw_sensor_data`, `hourly_sensor_rollups`, `daily_sensor_rollups`, `machine_health_summary`")
    
    default_sql = """SELECT 
    date,
    machine_id,
    sensor_type,
    ROUND(AVG(avg_value), 2) as avg_sensor_val,
    SUM(anomaly_count) as total_anomalies
FROM daily_sensor_rollups
WHERE date >= '2026-01-01' AND date <= '2026-02-28'
GROUP BY date, machine_id, sensor_type
ORDER BY date ASC, avg_sensor_val DESC
LIMIT 50;"""

    query_input = st.text_area("DuckDB SQL Query Editor", value=default_sql, height=160)
    
    if st.button("▶ Run SQL Query"):
        try:
            start_t = time.time()
            df_sql = analytics_engine.execute_sql(query_input)
            duration = round((time.time() - start_t) * 1000, 2)
            
            st.success(f"Query executed successfully in {duration} ms. Returned {len(df_sql)} rows.")
            st.dataframe(df_sql, width="stretch")
            
            csv_data = df_sql.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Results as CSV", data=csv_data, file_name="duckdb_query_results.csv", mime="text/csv")
        except Exception as e:
            st.error(f"SQL Execution Error: {str(e)}")
