# Urban Slum Detection & Infrastructure Planning

A machine learning system that classifies urban land cover from multi-spectral satellite imagery to detect informal settlements in Lagos, Abuja, Kano, and Accra — enabling city planners to prioritize water, sanitation, and road infrastructure.

## Overview

Processes high-resolution satellite image patches (RGB + NIR) to:
- Classify land cover into 7 categories including informal settlements
- Map slum distribution across target cities
- Quantify informal settlement density per urban area
- Guide infrastructure investment decisions

## Land Cover Classes

| Class | Description |
|-------|-------------|
| Informal Settlement | High-density, irregular housing (slums) |
| Formal Residential | Planned residential areas |
| Commercial | Business districts |
| Industrial | Factories, warehouses |
| Green Space | Parks, vegetation |
| Water Body | Rivers, lakes |
| Bare Land | Undeveloped areas |

## Features

- **Spectral Feature Extraction**: NDVI, NDBI, per-band statistics from 64×64 patches
- **Classification**: Random Forest (production-ready for CNN with TensorFlow)
- **Slum Mapping**: City-level interactive Folium maps
- **Imbalance Handling**: Class-weighted training for rare classes
- **Infrastructure Planning**: Top priority zones for water/sanitation intervention

## Project Structure

```
urban-slum-detection/
├── src/
│   ├── data_ingestion.py   # Synthetic patch generation
│   ├── model.py            # Feature extraction & classification
│   └── visualization.py    # Maps and distribution charts
├── data/sample/
├── models/
├── outputs/
├── config.py
├── main.py
└── requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python main.py
```

## Data Sources (Production)

- Imagery: Maxar/Planet Labs 50cm resolution, Sentinel-2 10m
- Building footprints: Microsoft Building Footprints Africa
- OSM: Road network, amenity points

## Author

**MOMAH MOSES .C.**  
Data Scientist & ML Engineer | [GitHub](https://github.com/Momahmoses)
