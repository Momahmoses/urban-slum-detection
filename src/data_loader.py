import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon
import yaml


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def generate_synthetic_tiles(n_tiles=3000, seed=42):
    """
    Simulate satellite image tile features extracted from high-resolution imagery.
    In production, tiles come from Maxar/Planet/Sentinel imagery via GEE or STAC.
    """
    np.random.seed(seed)
    n = n_tiles

    data = {
        "tile_id": [f"TILE_{i:05d}" for i in range(n)],
        "latitude": np.random.uniform(4.0, 14.0, n),
        "longitude": np.random.uniform(3.0, 14.5, n),
        "city": np.random.choice(["Lagos", "Abuja", "Accra"], n),
        "mean_r": np.random.uniform(50, 220, n),
        "mean_g": np.random.uniform(40, 200, n),
        "mean_b": np.random.uniform(30, 180, n),
        "mean_nir": np.random.uniform(60, 255, n),
        "std_texture": np.random.uniform(5, 80, n),
        "roof_density": np.random.uniform(0.1, 0.95, n),
        "building_regularity": np.random.uniform(0, 1, n),
        "road_distance_m": np.random.exponential(scale=200, size=n),
        "ndvi": np.random.uniform(-0.1, 0.5, n),
        "brightness": np.random.uniform(40, 220, n),
        "entropy": np.random.uniform(0.5, 7.5, n),
        "edge_density": np.random.uniform(0.01, 0.9, n),
    }
    df = pd.DataFrame(data)

    slum_score = (
        0.4 * (1 - df["building_regularity"])
        + 0.3 * df["roof_density"]
        + 0.2 * df["entropy"] / df["entropy"].max()
        + 0.1 * df["edge_density"]
        - 0.2 * (1 - df["road_distance_m"] / df["road_distance_m"].max())
        + np.random.normal(0, 0.08, n)
    )
    df["label"] = pd.cut(slum_score, bins=[-np.inf, 0.3, 0.65, np.inf],
                         labels=[0, 1, 2]).astype(int)
    return df


def to_geodataframe(df):
    return gpd.GeoDataFrame(
        df,
        geometry=[Point(lon, lat) for lon, lat in zip(df["longitude"], df["latitude"])],
        crs="EPSG:4326"
    )
