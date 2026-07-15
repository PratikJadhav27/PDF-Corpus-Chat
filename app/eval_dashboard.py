import streamlit as st
import os
import json
import pandas as pd
import glob
from pathlib import Path

# Title and Description
st.set_page_config(page_title="RAG Eval Dashboard", layout="wide")
st.title("📊 RAG Evaluation Dashboard")
st.markdown("Track the performance of the Multi-PDF ChatApp RAG Pipeline over time.")

# Load data
REPORTS_DIR = Path("evals/reports")

@st.cache_data
def load_eval_reports():
    reports = []
    if not REPORTS_DIR.exists():
        return pd.DataFrame()
    
    for report_file in REPORTS_DIR.glob("*.json"):
        with open(report_file, "r") as f:
            try:
                data = json.load(f)
                
                # Extract summary and metadata
                summary = data.get("summary", {})
                row = {
                    "Run Name": report_file.name,
                    "Generated At": pd.to_datetime(data.get("generated_at")),
                    "Collection": data.get("collection_name", "N/A"),
                    "Mode": data.get("retrieval_mode", "N/A"),
                    "Top K": data.get("top_k", "N/A"),
                    "Total Questions": summary.get("total_questions", 0),
                    "Citation Rate (%)": summary.get("citation_rate", 0),
                    "IDK Rate (%)": summary.get("idk_rate", 0),
                    "Avg Latency (s)": summary.get("avg_latency_sec", 0),
                    "Citation Validity": summary.get("avg_citation_validity_ratio", 0),
                    "Faithfulness": summary.get("avg_faithfulness", 0),
                    "Context Relevance": summary.get("avg_context_relevance", 0),
                    "Completeness": summary.get("avg_answer_completeness", 0),
                    "Composite Score": summary.get("avg_composite_score", 0),
                }
                reports.append(row)
            except Exception as e:
                st.error(f"Failed to parse {report_file.name}: {e}")
                
    if not reports:
        return pd.DataFrame()
        
    df = pd.DataFrame(reports)
    # Sort by generation time
    df = df.sort_values(by="Generated At").reset_index(drop=True)
    return df

df = load_eval_reports()

if df.empty:
    st.info("No evaluation reports found. Run the eval script to generate some data.")
else:
    # 1. High-level metrics for the latest run
    st.subheader("Latest Run Metrics")
    latest_run = df.iloc[-1]
    
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Composite Score", f"{latest_run['Composite Score']:.2f}")
    col2.metric("Faithfulness", f"{latest_run['Faithfulness']:.2f}")
    col3.metric("Citation Validity", f"{latest_run['Citation Validity']:.2f}")
    col4.metric("Context Relevance", f"{latest_run['Context Relevance']:.2f}")
    col5.metric("Avg Latency (s)", f"{latest_run['Avg Latency (s)']:.2f}")
    
    st.divider()
    
    # 2. Trends Plot
    st.subheader("📈 Score Trends Over Time")
    
    # Select metrics to plot
    metrics_to_plot = st.multiselect(
        "Select Metrics",
        ["Composite Score", "Faithfulness", "Citation Validity", "Context Relevance", "Completeness"],
        default=["Composite Score", "Faithfulness", "Citation Validity"]
    )
    
    if metrics_to_plot:
        # Prepare data for plotting
        plot_data = df.set_index("Generated At")[metrics_to_plot]
        st.line_chart(plot_data)
        
    st.divider()
    
    # 3. Raw Data Table
    st.subheader("📋 Evaluation History")
    st.dataframe(
        df.style.format({
            "Citation Rate (%)": "{:.1f}%",
            "IDK Rate (%)": "{:.1f}%",
            "Avg Latency (s)": "{:.2f}",
            "Citation Validity": "{:.2f}",
            "Faithfulness": "{:.2f}",
            "Context Relevance": "{:.2f}",
            "Completeness": "{:.2f}",
            "Composite Score": "{:.2f}",
        }),
        use_container_width=True
    )
