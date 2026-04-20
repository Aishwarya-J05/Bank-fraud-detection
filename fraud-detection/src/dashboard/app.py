# src/dashboard/app.py
"""
Streamlit analyst dashboard for fraud detection.
Connects to the FastAPI backend for predictions.
"""

import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Bank Fraud Detection Engine")
st.caption("Real-time transaction scoring with XGBoost + SHAP explainability")

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("API Status")
    try:
        health = requests.get(f"{API_URL}/health", timeout=3).json()
        st.success(f"✅ API Online")
        st.metric("Threshold", health["threshold"])
    except Exception:
        st.error("❌ API Offline — start the FastAPI server first")
        st.code("python -m src.api.main")
        st.stop()

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["Single Transaction", "Batch Scoring"])

# ── Tab 1: Single Transaction ─────────────────────────────────────────────
with tab1:
    st.subheader("Score a Single Transaction")

    col1, col2 = st.columns(2)

    with col1:
        amount = st.number_input("Amount ($)", min_value=0.01, value=150.0, step=0.01)
        time_val = st.number_input("Time (seconds elapsed)", min_value=0.0, value=1000.0)
        v14 = st.number_input("V14 (strongest fraud signal)", value=0.0, format="%.4f")
        v10 = st.number_input("V10", value=0.0, format="%.4f")
        v17 = st.number_input("V17", value=0.0, format="%.4f")

    with col2:
        v4 = st.number_input("V4", value=0.0, format="%.4f")
        v12 = st.number_input("V12", value=0.0, format="%.4f")
        v11 = st.number_input("V11", value=0.0, format="%.4f")
        v3 = st.number_input("V3", value=0.0, format="%.4f")
        v7 = st.number_input("V7", value=0.0, format="%.4f")

    # Load known fraud transaction for demo
    if st.button("Load Known Fraud Transaction"):
        st.session_state["fraud_demo"] = True
        st.rerun()

    if st.button("Analyze Transaction", type="primary"):
        payload = {
            "Time": time_val, "Amount": amount,
            "V1": 0.0, "V2": 0.0, "V3": v3, "V4": v4,
            "V5": 0.0, "V6": 0.0, "V7": v7, "V8": 0.0,
            "V9": 0.0, "V10": v10, "V11": v11, "V12": v12,
            "V13": 0.0, "V14": v14, "V15": 0.0, "V16": 0.0,
            "V17": v17, "V18": 0.0, "V19": 0.0, "V20": 0.0,
            "V21": 0.0, "V22": 0.0, "V23": 0.0, "V24": 0.0,
            "V25": 0.0, "V26": 0.0, "V27": 0.0, "V28": 0.0
        }

        with st.spinner("Scoring transaction..."):
            response = requests.post(f"{API_URL}/predict", json=payload).json()

        # Result display
        st.divider()
        res_col1, res_col2, res_col3, res_col4 = st.columns(4)

        fraud_prob = response["fraud_probability"]
        is_fraud = response["is_fraud"]

        with res_col1:
            st.metric("Fraud Probability", f"{fraud_prob:.1%}")
        with res_col2:
            if is_fraud:
                st.error(f"🚨 FRAUD DETECTED")
            else:
                st.success(f"✅ LEGITIMATE")
        with res_col3:
            risk_color = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}
            st.metric("Risk Level", f"{risk_color[response['risk_level']]} {response['risk_level']}")
        with res_col4:
            st.metric("Latency", f"{response['inference_latency_ms']:.1f}ms")

        # SHAP explanation
        st.subheader("Risk Factor Explanation (SHAP)")
        st.caption("Why did the model make this decision?")

        factors = response["top_risk_factors"]
        feat_names = list(factors.keys())
        shap_vals = [factors[f]["shap_value"] for f in feat_names]
        colors = ["crimson" if v > 0 else "steelblue" for v in shap_vals]

        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.barh(feat_names, shap_vals, color=colors)
        ax.axvline(x=0, color='black', linewidth=0.8)
        ax.set_xlabel("SHAP Value (positive = increases fraud risk)")
        ax.set_title("Top 5 Features Driving This Prediction")

        for bar, val in zip(bars, shap_vals):
            ax.text(
                val + (0.05 if val >= 0 else -0.05),
                bar.get_y() + bar.get_height() / 2,
                f"{val:+.3f}",
                va='center', ha='left' if val >= 0 else 'right',
                fontsize=9
            )

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Raw response
        with st.expander("Raw API Response"):
            st.json(response)


# ── Tab 2: Batch Scoring ──────────────────────────────────────────────────
with tab2:
    st.subheader("Batch Transaction Scoring")
    st.caption("Upload a CSV file with transaction data to score multiple transactions at once.")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write(f"Loaded {len(df):,} transactions")
        st.dataframe(df.head())

        if st.button("Score All Transactions", type="primary"):
            required_cols = ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)]
            missing = [c for c in required_cols if c not in df.columns]

            if missing:
                st.error(f"Missing columns: {missing}")
            else:
                records = df[required_cols].to_dict(orient="records")

                with st.spinner(f"Scoring {len(records)} transactions..."):
                    response = requests.post(
                        f"{API_URL}/predict/batch",
                        json=records
                    ).json()

                results_df = pd.DataFrame(response["results"])
                results_df = results_df.merge(
                    df.reset_index(drop=True),
                    left_index=True, right_index=True
                )

                # Summary metrics
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Total Transactions", response["total"])
                m2.metric("Flagged as Fraud", response["flagged"])
                m3.metric("Fraud Rate", f"{response['flagged']/response['total']*100:.2f}%")
                m4.metric("Batch Latency", f"{response['latency_ms']:.1f}ms")

                # Results table
                st.subheader("Results")
                display_df = results_df[["index", "fraud_probability", "is_fraud", "risk_level", "Amount"]].copy()
                display_df["fraud_probability"] = display_df["fraud_probability"].apply(lambda x: f"{x:.1%}")

                st.dataframe(
                    display_df.style.apply(
                        lambda row: ["background-color: #ffcccc" if row["is_fraud"] else "" 
                                     for _ in row], axis=1
                    )
                )

                # Download results
                csv = results_df.to_csv(index=False)
                st.download_button(
                    "Download Results CSV",
                    csv,
                    "fraud_predictions.csv",
                    "text/csv"
                )