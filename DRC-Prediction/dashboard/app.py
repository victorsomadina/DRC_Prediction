import os
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from pathlib import Path

API_URL = os.getenv("DASHBOARD_API_URL", "http://localhost:8000")
MODELS_DIR = Path(__file__).resolve().parents[1] / "models"

st.set_page_config(
    page_title="TerraNova | DRC Cost Forecasting",
    layout="wide",
    page_icon="🌊",
)

st.title("Disaster Recovery Cost Forecasting")
st.caption("TerraNova Resilience Analytics Ltd — Predictive Budget Planning Tool")

with st.sidebar:
    st.header("Disaster Scenario")

    incident_type = st.selectbox(
        "Incident Type",
        ["Hurricane", "Flood", "Tornado", "Severe Storm", "Earthquake",
         "Wildfire", "Winter Storm", "Drought", "Other"],
    )
    declaration_type = st.selectbox("Declaration Type", ["DR", "EM", "FM"])
    state = st.selectbox("State", [
        "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA","HI","ID","IL",
        "IN","IA","KS","KY","LA","ME","MD","MA","MI","MN","MS","MO","MT",
        "NE","NV","NH","NJ","NM","NY","NC","ND","OH","OK","OR","PA","RI",
        "SC","SD","TN","TX","UT","VT","VA","WA","WV","WI","WY",
    ])
    duration = st.slider("Incident Duration (days)", 1, 180, 14)
    year = st.number_input("Declaration Year", min_value=2000, max_value=2030, value=2024)
    freq = st.slider("5-Year Rolling Disaster Frequency (state avg)", 0.0, 20.0, 3.0, 0.1)
    project_count = st.number_input("Expected Project Count", min_value=1, value=100)
    avg_project = st.number_input("Avg Project Amount ($)", min_value=0, value=250000, step=10000)
    allocated_budget = st.number_input(
        "Allocated Budget ($)", min_value=0, value=5_000_000, step=100_000
    )
    is_high_cost = 1 if incident_type in {"Hurricane", "Flood", "Tornado", "Severe Storm"} else 0

    predict_btn = st.button("Run Forecast", type="primary", use_container_width=True)

col1, col2, col3 = st.columns(3)

if predict_btn:
    payload = {
        "incidentType": incident_type,
        "declarationType": declaration_type,
        "state": state,
        "incident_duration_days": float(duration),
        "declaration_year": int(year),
        "rolling_5yr_freq": float(freq),
        "project_count": int(project_count),
        "avg_project_amount": float(avg_project),
        "is_high_cost_type": is_high_cost,
    }

    with st.spinner("Querying prediction model..."):
        try:
            resp = requests.post(f"{API_URL}/predict-cost", json=payload, timeout=15)
            resp.raise_for_status()
            result = resp.json()
            predicted = result["predicted_recovery_cost_usd"]
            model_name = result["model_name"]

            budget_gap = predicted - allocated_budget
            gap_pct = (budget_gap / allocated_budget * 100) if allocated_budget > 0 else 0

            with col1:
                st.metric(
                    "Predicted Recovery Cost",
                    f"${predicted:,.0f}",
                    delta=None,
                )
            with col2:
                st.metric(
                    "Allocated Budget",
                    f"${allocated_budget:,.0f}",
                )
            with col3:
                st.metric(
                    "Budget Gap",
                    f"${budget_gap:,.0f}",
                    delta=f"{gap_pct:+.1f}%",
                    delta_color="inverse",
                )

            st.info(f"Model: **{model_name}** | {result['confidence_note']}")

            fig = go.Figure(go.Bar(
                x=["Allocated Budget", "Predicted Recovery Cost"],
                y=[allocated_budget, predicted],
                marker_color=["#2ecc71", "#e74c3c" if predicted > allocated_budget else "#2ecc71"],
                text=[f"${allocated_budget:,.0f}", f"${predicted:,.0f}"],
                textposition="outside",
            ))
            fig.update_layout(
                title="Budget vs Predicted Recovery Cost",
                yaxis_title="USD ($)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=350,
            )
            st.plotly_chart(fig, use_container_width=True)

        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to API. Make sure FastAPI server is running at " + API_URL)
        except Exception as e:
            st.error(f"Prediction failed: {e}")

else:
    with col1:
        st.metric("Predicted Recovery Cost", "—")
    with col2:
        st.metric("Allocated Budget", f"${allocated_budget:,.0f}")
    with col3:
        st.metric("Budget Gap", "—")
    st.info("Configure scenario inputs in the sidebar and click **Run Forecast**.")

st.divider()
st.subheader("Feature Importance (Trained Model)")

model_path = MODELS_DIR / "best_model.pkl"
if model_path.exists():
    artifact = joblib.load(model_path)
    pipeline = artifact["pipeline"]
    model = pipeline.named_steps["model"]

    if hasattr(model, "feature_importances_"):
        preprocessor = pipeline.named_steps["preprocessor"]
        try:
            feature_names = preprocessor.get_feature_names_out()
        except Exception:
            feature_names = [f"feature_{i}" for i in range(len(model.feature_importances_))]

        importance_df = pd.DataFrame({
            "feature": feature_names,
            "importance": model.feature_importances_,
        }).sort_values("importance", ascending=True).tail(15)

        fig2 = go.Figure(go.Bar(
            x=importance_df["importance"],
            y=importance_df["feature"],
            orientation="h",
            marker_color="#3498db",
        ))
        fig2.update_layout(
            title=f"Top Feature Importances — {artifact['model_name']}",
            xaxis_title="Importance Score",
            height=450,
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Feature importance not available for this model type.")
else:
    st.warning("No trained model found. Run the training pipeline first.")
