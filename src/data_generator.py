"""
Synthetic dataset generator for urban slum detection.
Produces grid-cell level land-use features derived from satellite imagery proxies.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

RANDOM_SEED = 42
N_CELLS = 6000

CITIES = [
    ("Lagos", 6.5244, 3.3792), ("Abuja", 9.0765, 7.3986),
    ("Ibadan", 7.3775, 3.9470), ("Port Harcourt", 4.8156, 7.0498),
    ("Kano", 12.0022, 8.5920), ("Accra", 5.6037, -0.1870),
    ("Kumasi", 6.6885, -1.6244),
]


def generate_slum_dataset(n_cells: int = N_CELLS, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Generate synthetic satellite-imagery-derived grid-cell dataset for slum classification.

    Each record represents a ~250m x 250m grid cell. Features mimic spectral,
    texture, and structural signals from high-resolution satellite imagery.
    """
    rng = np.random.default_rng(seed)
    records = []

    for cell_id in range(n_cells):
        city, lat, lon = CITIES[rng.integers(0, len(CITIES))]
        lat_j = lat + rng.normal(0, 0.15)
        lon_j = lon + rng.normal(0, 0.15)

        is_slum_ground_truth = int(rng.random() < 0.38)

        if is_slum_ground_truth:
            ndvi = float(np.clip(rng.normal(0.12, 0.06), 0, 0.5))
            built_up_density = float(np.clip(rng.normal(0.82, 0.1), 0.4, 1.0))
            road_accessibility = float(np.clip(rng.beta(2, 5), 0.05, 0.6))
            building_regularity = float(np.clip(rng.normal(0.25, 0.1), 0, 0.6))
            roof_material_score = float(np.clip(rng.normal(0.30, 0.12), 0, 0.7))
            mean_building_size_m2 = max(10, float(rng.normal(28, 10)))
            building_spacing_m = max(0.5, float(rng.normal(1.5, 1.0)))
            texture_entropy = float(np.clip(rng.normal(0.72, 0.1), 0.4, 1.0))
            brightness_index = float(np.clip(rng.normal(0.38, 0.08), 0.1, 0.7))
            water_access_score = float(np.clip(rng.beta(2, 6), 0, 0.5))
            sanitation_score = float(np.clip(rng.beta(2, 6), 0, 0.5))
            pop_density = max(200, float(rng.lognormal(9.5, 0.6)))
        else:
            ndvi = float(np.clip(rng.normal(0.42, 0.15), 0.05, 0.9))
            built_up_density = float(np.clip(rng.normal(0.45, 0.2), 0.05, 0.95))
            road_accessibility = float(np.clip(rng.beta(5, 2), 0.3, 1.0))
            building_regularity = float(np.clip(rng.normal(0.72, 0.12), 0.3, 1.0))
            roof_material_score = float(np.clip(rng.normal(0.72, 0.15), 0.3, 1.0))
            mean_building_size_m2 = max(30, float(rng.normal(120, 50)))
            building_spacing_m = max(2.0, float(rng.normal(8, 4)))
            texture_entropy = float(np.clip(rng.normal(0.40, 0.12), 0.1, 0.8))
            brightness_index = float(np.clip(rng.normal(0.60, 0.12), 0.2, 0.9))
            water_access_score = float(np.clip(rng.beta(6, 2), 0.4, 1.0))
            sanitation_score = float(np.clip(rng.beta(6, 2), 0.4, 1.0))
            pop_density = max(50, float(rng.lognormal(7.5, 0.8)))

        records.append({
            "cell_id": cell_id, "city": city,
            "latitude": round(lat_j, 6), "longitude": round(lon_j, 6),
            "ndvi": round(ndvi, 4), "built_up_density": round(built_up_density, 4),
            "road_accessibility": round(road_accessibility, 4),
            "building_regularity": round(building_regularity, 4),
            "roof_material_score": round(roof_material_score, 4),
            "mean_building_size_m2": round(mean_building_size_m2, 1),
            "building_spacing_m": round(building_spacing_m, 2),
            "texture_entropy": round(texture_entropy, 4),
            "brightness_index": round(brightness_index, 4),
            "water_access_score": round(water_access_score, 4),
            "sanitation_score": round(sanitation_score, 4),
            "population_density": round(pop_density, 1),
            "is_informal_settlement": is_slum_ground_truth,
        })

    return pd.DataFrame(records)


def save_dataset(output_dir: str | Path = "data/raw") -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = generate_slum_dataset()
    path = output_dir / "slum_data.csv"
    df.to_csv(path, index=False)
    return path
