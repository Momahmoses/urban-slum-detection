# Urban Slum Detection & Infrastructure Planning

High-resolution satellite imagery + ML image classification to map informal settlements in Lagos, Abuja, and Accra — helping city planners prioritize water, sanitation, road, and electricity infrastructure investment.

## Features
- Tile-based classification: Formal / Informal / Industrial
- Gradient Boosting classifier on texture, spectral, and structural features
- Informal settlement heatmap (Folium)
- Infrastructure priority scoring by settlement area
- Confusion matrix and class distribution charts

## Project Structure
```
urban-slum-detection/
├── src/
│   ├── data_loader.py     # Tile feature extraction and data generation
│   ├── model.py           # GBM classifier, prediction, infrastructure prioritization
│   └── visualize.py       # Maps and charts
├── data/raw/              # VHR satellite tiles, OSM layers
├── models/                # Saved classifier
├── outputs/               # Slum map, reports, charts
├── config.yaml
├── main.py
└── requirements.txt
```

## Setup
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py
```

## Data Sources
| Layer | Source |
|-------|--------|
| Satellite imagery | Maxar / Planet Labs / Sentinel-2 |
| Urban boundaries | OpenStreetMap / GADM |
| Population | WorldPop 100m |
| Road network | OSM via OSMnx |

## Output
- `outputs/slum_detection_map.html` — interactive informal settlement map
- `outputs/slum_report.csv` — per-tile classification results
- `outputs/infrastructure_priorities.csv` — ranked infrastructure needs
- `outputs/confusion_matrix.png` — model accuracy breakdown

## Author
MOMAH MOSES .C.
