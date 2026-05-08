"""Main pipeline: Urban Slum Detection & Infrastructure Planning."""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from src.data_ingestion import generate_dataset, load_dataset
from src.model import train, evaluate, extract_features
from src.visualization import plot_sample_patches, create_city_map, plot_class_distribution
import pandas as pd
import numpy as np


def main():
    print("=" * 60)
    print("  Urban Slum Detection & Infrastructure Planning")
    print("  Cities: Lagos, Abuja, Kano (Nigeria) + Accra (Ghana)")
    print("=" * 60)

    print("\n[1/5] Generating synthetic satellite image patches...")
    patches, labels, meta = generate_dataset(n_samples=2000)

    print("\n[2/5] Visualizing sample patches...")
    plot_sample_patches(patches, labels)

    print("\n[3/5] Training land cover classifier...")
    clf, scaler, X_test, y_test = train(patches, labels)

    print("\n[4/5] Evaluating model...")
    evaluate(clf, X_test, y_test)

    print("\n[5/5] Generating city-level slum maps...")
    X_all = extract_features(patches)
    X_all_s = scaler.transform(X_all)
    all_preds = clf.predict(X_all_s)
    create_city_map(meta, all_preds)
    plot_class_distribution(meta, all_preds)

    informal_count = (all_preds == 1).sum()
    print(f"\n  Informal settlements detected: {informal_count} patches ({informal_count/len(all_preds):.1%})")
    print("\n✓ Pipeline complete. Outputs saved to ./outputs/")


if __name__ == "__main__":
    main()
