"""Generate synthetic multi-spectral image patches for urban classification."""

import numpy as np
import pandas as pd
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import SAMPLE_DIR, STUDY_CITIES, IMG_SIZE, N_CHANNELS, LAND_COVER_CLASSES

N_CLASSES = len(LAND_COVER_CLASSES)
CLASS_NAMES = list(LAND_COVER_CLASSES.values())


def generate_patch(label: int, img_size: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a synthetic multi-spectral image patch for a given land cover class."""
    patch = np.zeros((img_size, img_size, N_CHANNELS), dtype=np.float32)

    if label == 1:  # informal_settlement: high density, mixed low texture
        patch[:, :, 0] = rng.uniform(0.25, 0.55, (img_size, img_size))  # R
        patch[:, :, 1] = rng.uniform(0.20, 0.45, (img_size, img_size))  # G
        patch[:, :, 2] = rng.uniform(0.15, 0.40, (img_size, img_size))  # B
        patch[:, :, 3] = rng.uniform(0.10, 0.30, (img_size, img_size))  # NIR (low vegetation)
    elif label == 0:  # formal_residential: structured, more uniform
        patch[:, :, 0] = rng.uniform(0.35, 0.65, (img_size, img_size))
        patch[:, :, 1] = rng.uniform(0.35, 0.65, (img_size, img_size))
        patch[:, :, 2] = rng.uniform(0.35, 0.65, (img_size, img_size))
        patch[:, :, 3] = rng.uniform(0.30, 0.60, (img_size, img_size))
    elif label == 4:  # green_space: high NIR, high G
        patch[:, :, 0] = rng.uniform(0.05, 0.20, (img_size, img_size))
        patch[:, :, 1] = rng.uniform(0.30, 0.65, (img_size, img_size))
        patch[:, :, 2] = rng.uniform(0.05, 0.20, (img_size, img_size))
        patch[:, :, 3] = rng.uniform(0.60, 0.90, (img_size, img_size))
    elif label == 5:  # water: low all, high B
        patch[:, :, 0] = rng.uniform(0.02, 0.12, (img_size, img_size))
        patch[:, :, 1] = rng.uniform(0.10, 0.30, (img_size, img_size))
        patch[:, :, 2] = rng.uniform(0.30, 0.65, (img_size, img_size))
        patch[:, :, 3] = rng.uniform(0.02, 0.10, (img_size, img_size))
    else:  # commercial/industrial/bare
        val = rng.uniform(0.3, 0.7)
        patch[:, :, :] = rng.uniform(val - 0.1, val + 0.1, (img_size, img_size, N_CHANNELS))

    # Add spatial texture (random noise + smoothing)
    noise = rng.normal(0, 0.02, patch.shape).astype(np.float32)
    patch = np.clip(patch + noise, 0, 1)
    return patch


def generate_dataset(n_samples: int = 2000, seed: int = 42):
    rng = np.random.default_rng(seed)
    # Simulate class imbalance (informal settlements ~25%)
    class_probs = [0.25, 0.25, 0.15, 0.10, 0.12, 0.08, 0.05]
    labels = rng.choice(N_CLASSES, size=n_samples, p=class_probs)
    patches = np.array([generate_patch(lbl, IMG_SIZE, rng) for lbl in labels])
    cities = list(STUDY_CITIES.keys())
    city_labels = rng.choice(cities, size=n_samples)

    meta = pd.DataFrame({
        "sample_id": range(n_samples),
        "label": labels,
        "class_name": [CLASS_NAMES[l] for l in labels],
        "city": city_labels,
    })
    os.makedirs(SAMPLE_DIR, exist_ok=True)
    np.save(os.path.join(SAMPLE_DIR, "patches.npy"), patches)
    np.save(os.path.join(SAMPLE_DIR, "labels.npy"), labels)
    meta.to_csv(os.path.join(SAMPLE_DIR, "metadata.csv"), index=False)
    print(f"Generated {n_samples} image patches ({IMG_SIZE}x{IMG_SIZE}x{N_CHANNELS})")
    print(f"Informal settlements: {(labels == 1).sum()} ({(labels == 1).mean():.1%})")
    return patches, labels, meta


def load_dataset():
    patches_path = os.path.join(SAMPLE_DIR, "patches.npy")
    labels_path = os.path.join(SAMPLE_DIR, "labels.npy")
    if os.path.exists(patches_path):
        return np.load(patches_path), np.load(labels_path)
    return generate_dataset()[:2]
