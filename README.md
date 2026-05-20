# Urban Slum Detection & Infrastructure Planning

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

High-resolution satellite imagery + ML image classification to map informal settlements in Lagos, Abuja, and Accra — helping city planners prioritise water, sanitation, road, and electricity infrastructure investment.

---

## Problem Statement

70+ million Nigerians live in informal settlements lacking paved roads, piped water, electricity, and sanitation. Without accurate spatial mapping of these settlements, infrastructure budgets are allocated inefficiently. This system automates settlement detection from satellite imagery at scale.

---

## Features

| Feature | Description |
|---------|-------------|
| Tile Classification | Formal / Informal / Industrial labelling per image tile |
| Gradient Boosting Model | Texture, spectral, and structural feature classifier |
| Informal Settlement Heatmap | Folium-based density map across study cities |
| Infrastructure Gap Scoring | Priority investment ranking by population and gap severity |
| Multi-City Support | Lagos, Abuja (Nigeria) and Accra (Ghana) |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Remote Sensing | Sentinel-2, Google Earth Engine |
| Machine Learning | scikit-learn (Gradient Boosting) |
| Geospatial | GeoPandas, Folium, Rasterio |
| Data | pandas, NumPy |
| Visualisation | Matplotlib, Seaborn |

---

## Quick Start

```bash
git clone https://github.com/Momahmoses/urban-slum-detection.git
cd urban-slum-detection
pip install -r requirements.txt
python main.py
```

---

## Data Sources

- Sentinel-2 Level-2A 10m resolution imagery
- WorldPop gridded population data
- OpenStreetMap building footprints
- UN-Habitat informal settlement boundaries

---

## Author

**Momah Moses** — Geospatial AI Engineer & Data Scientist
[GitHub](https://github.com/Momahmoses) · [Portfolio](https://momahmoses-ng-gis-portfolio.hf.space)
