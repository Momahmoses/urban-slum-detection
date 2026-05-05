import folium
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from folium.plugins import HeatMap


CLASS_COLORS = {"Formal": "#2166ac", "Informal": "#d73027", "Industrial": "#f4a582"}


def plot_slum_map(gdf, output_path):
    center = [gdf["latitude"].mean(), gdf["longitude"].mean()]
    m = folium.Map(location=center, zoom_start=7, tiles="CartoDB positron")

    heat_data = [
        [row["latitude"], row["longitude"], row["informal_probability"]]
        for _, row in gdf.iterrows()
        if row["pred_class"] == "Informal"
    ]
    HeatMap(heat_data, radius=12, blur=8, name="Informal Settlement Density").add_to(m)

    for _, row in gdf.sample(min(500, len(gdf))).iterrows():
        color = CLASS_COLORS.get(row["pred_class"], "#ccc")
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=4, color=color, fill=True, fill_opacity=0.6,
            popup=folium.Popup(
                f"<b>Class:</b> {row['pred_class']}<br>"
                f"<b>City:</b> {row['city']}<br>"
                f"<b>Informal Prob:</b> {row['informal_probability']:.2%}",
                max_width=200
            )
        ).add_to(m)

    folium.LayerControl().add_to(m)
    m.save(output_path)
    print(f"Slum map saved: {output_path}")


def plot_confusion_matrix(cm, output_path):
    fig, ax = plt.subplots(figsize=(8, 6))
    labels = ["Formal", "Informal", "Industrial"]
    sns.heatmap(cm, annot=True, fmt="d", cmap="Reds",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix — Urban Slum Detection")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_class_distribution(df, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    counts = df["pred_class"].value_counts()
    colors = [CLASS_COLORS.get(k, "#ccc") for k in counts.index]
    axes[0].pie(counts.values, labels=counts.index, colors=colors, autopct="%1.1f%%")
    axes[0].set_title("Tile Classification Distribution")

    city_class = df.groupby(["city", "pred_class"]).size().unstack(fill_value=0)
    city_class.plot(kind="bar", stacked=True, color=list(CLASS_COLORS.values()), ax=axes[1])
    axes[1].set_title("Settlement Type by City")
    axes[1].set_xlabel("City")
    axes[1].set_ylabel("Tile Count")
    axes[1].legend(title="Class")
    axes[1].tick_params(axis="x", rotation=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
