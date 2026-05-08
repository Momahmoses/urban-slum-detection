"""
Urban Slum Detection & Infrastructure Planning
===============================================
ML classification of informal settlements from satellite imagery proxy features.
Supports city planning in Lagos, Abuja, Ibadan, Port Harcourt, and West Africa.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))
from data_generator import generate_slum_dataset
from model import (
    FEATURE_COLS, METRICS_PATH, MODEL_PATH,
    load_model, predict_cell, save_model, train,
)

st.set_page_config(page_title="Urban Slum Detection | West Africa", page_icon="🏙️", layout="wide")

CONF_COLORS = {"High": "#E74C3C", "Medium": "#F39C12", "Low": "#2ECC71"}


@st.cache_resource(show_spinner="Training slum detection model…")
def get_model_and_data():
    df = generate_slum_dataset()
    if MODEL_PATH.exists() and METRICS_PATH.exists():
        pipeline = load_model()
        with open(METRICS_PATH) as f:
            metrics = json.load(f)
    else:
        pipeline, metrics = train(df)
        save_model(pipeline, metrics)
    probs = pipeline.predict_proba(df[FEATURE_COLS])[:, 1]
    df["informal_probability"] = probs
    df["confidence_tier"] = pd.cut(probs, bins=[0, 0.40, 0.65, 1.0], labels=["Low", "Medium", "High"]).astype(str)
    return pipeline, metrics, df


pipeline, metrics, df = get_model_and_data()

with st.sidebar:
    st.title("🏙️ Slum Detection")
    st.caption("West Africa Urban Intelligence")
    st.divider()
    page = st.radio("Navigation", ["Overview", "Detection Map", "Classify a Grid Cell", "Model Performance"], label_visibility="collapsed")
    st.divider()
    city_filter = st.multiselect("Filter by City", sorted(df["city"].unique()), default=sorted(df["city"].unique()))

df_f = df[df["city"].isin(city_filter)]

if page == "Overview":
    st.title("Urban Slum Detection & Infrastructure Planning")
    st.markdown("Satellite imagery-based classification of informal settlements to support city planners in prioritizing water, sanitation, and road infrastructure.")
    st.divider()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Grid Cells Analyzed", f"{len(df_f):,}")
    c2.metric("Informal Settlements Detected", f"{df_f['is_informal_settlement'].sum():,}", f"{df_f['is_informal_settlement'].mean():.1%} of cells", delta_color="inverse")
    c3.metric("Model ROC-AUC", f"{metrics['roc_auc_test']:.4f}")
    c4.metric("F1 Score (Slum)", f"{metrics['f1_slum']:.4f}")
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        city_stats = df_f.groupby("city")["is_informal_settlement"].agg(["sum", "count"]).reset_index()
        city_stats.columns = ["city", "informal", "total"]
        city_stats["pct"] = (city_stats["informal"] / city_stats["total"] * 100).round(1)
        fig = px.bar(city_stats.sort_values("pct", ascending=False), x="city", y="pct",
                     color="pct", color_continuous_scale="Reds",
                     title="Informal Settlement Rate by City (%)",
                     labels={"pct": "% of Grid Cells"}, height=400)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig2 = px.histogram(df_f, x="informal_probability", color="confidence_tier",
                            color_discrete_map=CONF_COLORS, nbins=40,
                            title="Distribution of Informal Settlement Probabilities",
                            labels={"informal_probability": "P(Informal)"}, height=400)
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        fig3 = px.scatter(df_f.sample(min(2000, len(df_f)), random_state=42),
                          x="building_regularity", y="road_accessibility",
                          color="confidence_tier", color_discrete_map=CONF_COLORS,
                          opacity=0.5, title="Building Regularity vs Road Accessibility",
                          labels={"building_regularity": "Building Regularity", "road_accessibility": "Road Accessibility"},
                          height=380)
        st.plotly_chart(fig3, use_container_width=True)
    with col4:
        fi = metrics["feature_importance"]
        fi_s = sorted(fi.items(), key=lambda x: x[1])
        fig4 = go.Figure(go.Bar(x=[v for _, v in fi_s], y=[k for k, _ in fi_s],
                                orientation="h", marker_color="#E74C3C"))
        fig4.update_layout(title="Feature Importance", height=380, margin={"l": 200})
        st.plotly_chart(fig4, use_container_width=True)

elif page == "Detection Map":
    st.title("Informal Settlement Detection Map")
    sample = df_f.sample(min(1200, len(df_f)), random_state=42)
    fig = px.scatter_mapbox(
        sample, lat="latitude", lon="longitude",
        color="confidence_tier", color_discrete_map=CONF_COLORS,
        size="informal_probability", size_max=12,
        hover_name="city",
        hover_data={"informal_probability": ":.2%", "confidence_tier": True, "latitude": False, "longitude": False},
        mapbox_style="carto-positron", zoom=5,
        center={"lat": 7.0, "lon": 3.5},
        title="Informal Settlement Probability Map", height=560,
    )
    st.plotly_chart(fig, use_container_width=True)

elif page == "Classify a Grid Cell":
    st.title("Classify a Custom Grid Cell")
    col1, col2 = st.columns(2)
    with col1:
        ndvi = st.slider("NDVI (vegetation index)", 0.0, 0.9, 0.25, 0.01)
        built_up_density = st.slider("Built-up Density", 0.0, 1.0, 0.7, 0.01)
        road_accessibility = st.slider("Road Accessibility (0=none, 1=excellent)", 0.0, 1.0, 0.3, 0.01)
        building_regularity = st.slider("Building Regularity (0=chaotic, 1=planned)", 0.0, 1.0, 0.3, 0.01)
        roof_material_score = st.slider("Roof Material Quality (0=makeshift, 1=permanent)", 0.0, 1.0, 0.35, 0.01)
        mean_building_size_m2 = st.slider("Mean Building Size (m²)", 10.0, 400.0, 40.0, 5.0)
    with col2:
        building_spacing_m = st.slider("Avg Building Spacing (m)", 0.5, 30.0, 2.0, 0.5)
        texture_entropy = st.slider("Texture Entropy (satellite)", 0.1, 1.0, 0.65, 0.01)
        brightness_index = st.slider("Brightness Index", 0.1, 0.9, 0.42, 0.01)
        water_access_score = st.slider("Water Access Score", 0.0, 1.0, 0.3, 0.01)
        sanitation_score = st.slider("Sanitation Score", 0.0, 1.0, 0.25, 0.01)
        population_density = st.number_input("Population Density (persons/km²)", 100, 100000, 15000, 500)

    if st.button("Classify Grid Cell", type="primary"):
        result = predict_cell(pipeline, {
            "ndvi": ndvi, "built_up_density": built_up_density,
            "road_accessibility": road_accessibility, "building_regularity": building_regularity,
            "roof_material_score": roof_material_score, "mean_building_size_m2": mean_building_size_m2,
            "building_spacing_m": building_spacing_m, "texture_entropy": texture_entropy,
            "brightness_index": brightness_index, "water_access_score": water_access_score,
            "sanitation_score": sanitation_score, "population_density": population_density,
        })
        st.divider()
        r1, r2, r3 = st.columns(3)
        r1.metric("P(Informal Settlement)", f"{result['informal_probability']:.2%}")
        r2.metric("Classification", "Informal Settlement" if result["is_informal"] else "Formal Area")
        r3.metric("Confidence", result["confidence_tier"])
        if result["is_informal"]:
            st.error("This grid cell is classified as an **informal settlement**. Recommend: infrastructure assessment for water, sanitation, and road access.")
        else:
            st.success("This grid cell appears to be a **formal settlement area**.")

elif page == "Model Performance":
    st.title("Model Performance — Gradient Boosting Classifier")
    st.divider()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Test ROC-AUC", f"{metrics['roc_auc_test']:.4f}")
    m2.metric("CV AUC", f"{metrics['cv_auc_mean']:.4f} ± {metrics['cv_auc_std']:.4f}")
    m3.metric("Accuracy", f"{metrics['accuracy']:.4f}")
    m4.metric("Slum F1-Score", f"{metrics['f1_slum']:.4f}")

    cm = metrics["confusion_matrix"]
    import numpy as np
    fig = go.Figure(go.Heatmap(
        z=np.array(cm), x=["Formal", "Informal"], y=["Formal", "Informal"],
        colorscale="Reds", showscale=False,
        text=[[str(v) for v in row] for row in cm],
        texttemplate="%{text}", textfont={"size": 18},
    ))
    fig.update_layout(title="Confusion Matrix", xaxis_title="Predicted", yaxis_title="Actual", height=350)
    st.plotly_chart(fig, use_container_width=True)
    with st.expander("Full Metrics"):
        st.json(metrics)
