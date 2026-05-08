"""Visualize slum detection results and land cover maps."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import folium
import geopandas as gpd
from shapely.geometry import Point
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import OUTPUTS_DIR, STUDY_CITIES, LAND_COVER_CLASSES

CLASS_COLORS = {
    "formal_residential": "#3498db",
    "informal_settlement": "#e74c3c",
    "commercial": "#f39c12",
    "industrial": "#95a5a6",
    "green_space": "#27ae60",
    "water_body": "#2980b9",
    "bare_land": "#d4ac0d",
}


def plot_sample_patches(patches: np.ndarray, labels: np.ndarray, n: int = 12):
    fig, axes = plt.subplots(2, 6, figsize=(18, 6))
    axes = axes.flatten()
    for i in range(min(n, len(patches))):
        rgb = patches[i, :, :, :3]
        axes[i].imshow(rgb)
        cls = LAND_COVER_CLASSES[labels[i]]
        axes[i].set_title(cls.replace("_", "\n"), fontsize=8)
        for spine in axes[i].spines.values():
            spine.set_edgecolor(CLASS_COLORS.get(cls, "gray"))
            spine.set_linewidth(3)
        axes[i].axis("off")
    plt.suptitle("Sample Satellite Image Patches — Urban Land Cover", fontsize=13, y=1.02)
    plt.tight_layout()
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    plt.savefig(os.path.join(OUTPUTS_DIR, "sample_patches.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Sample patches plot saved.")


def create_city_map(meta: pd.DataFrame, predictions: np.ndarray):
    """Create folium map showing predicted land cover per city."""
    from scipy.spatial import cKDTree
    center = [7.0, 5.0]
    m = folium.Map(location=center, zoom_start=6, tiles="CartoDB positron")

    meta = meta.copy()
    meta["predicted_class"] = [LAND_COVER_CLASSES[p] for p in predictions]

    for city, info in STUDY_CITIES.items():
        city_meta = meta[meta["city"] == city]
        rng = np.random.default_rng(hash(city) % 2**32)
        for _, row in city_meta.iterrows():
            lat = info["lat"] + rng.uniform(-0.3, 0.3)
            lon = info["lon"] + rng.uniform(-0.3, 0.3)
            cls = row["predicted_class"]
            folium.CircleMarker(
                location=[lat, lon], radius=5,
                color=CLASS_COLORS.get(cls, "gray"),
                fill=True, fill_opacity=0.7,
                popup=f"City: {city}<br>Class: {cls.replace('_',' ').title()}",
            ).add_to(m)

        folium.Marker(
            [info["lat"], info["lon"]],
            popup=f"<b>{city}</b>",
            icon=folium.Icon(color="black", icon="home"),
        ).add_to(m)

    legend_html = "<div style='position:fixed;bottom:30px;left:30px;z-index:1000;background:white;padding:12px;border-radius:8px;box-shadow:2px 2px 6px rgba(0,0,0,0.3);font-size:12px;'><b>Land Cover</b><br>"
    for cls, color in CLASS_COLORS.items():
        legend_html += f"<span style='color:{color};'>&#9632;</span> {cls.replace('_',' ').title()}<br>"
    legend_html += "</div>"
    m.get_root().html.add_child(folium.Element(legend_html))

    out = os.path.join(OUTPUTS_DIR, "slum_detection_map.html")
    m.save(out)
    print(f"City map saved → {out}")


def plot_class_distribution(meta: pd.DataFrame, predictions: np.ndarray):
    meta = meta.copy()
    meta["predicted_class"] = [LAND_COVER_CLASSES[p] for p in predictions]
    counts = meta["predicted_class"].value_counts()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = [CLASS_COLORS.get(c, "gray") for c in counts.index]
    axes[0].barh(counts.index, counts.values, color=colors)
    axes[0].set_xlabel("Count")
    axes[0].set_title("Predicted Land Cover Distribution")

    by_city = meta.groupby(["city", "predicted_class"]).size().unstack(fill_value=0)
    if "informal_settlement" in by_city.columns:
        by_city["informal_settlement"].plot(kind="bar", ax=axes[1], color="#e74c3c")
        axes[1].set_title("Informal Settlement Count by City")
        axes[1].set_ylabel("Count")
        axes[1].tick_params(axis="x", rotation=0)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, "class_distribution.png"), dpi=150)
    plt.close()
    print("Distribution chart saved.")
