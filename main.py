import os
import numpy as np
from src.data_loader import load_config, generate_synthetic_tiles, to_geodataframe
from src.model import train, predict, save_model, feature_importance, prioritize_infrastructure
from src.visualize import plot_slum_map, plot_confusion_matrix, plot_class_distribution


def main():
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    config = load_config("config.yaml")
    print(f"[1/5] Config loaded — targets: {', '.join(config['targets'])}")

    df = generate_synthetic_tiles(n_tiles=3000)
    print(f"[2/5] {len(df)} image tiles loaded")

    pipeline, metrics, (_, y_test, y_pred) = train(df, config)
    print(f"[3/5] Model trained — Accuracy: {metrics['accuracy']:.4f}")
    print(metrics["classification_report"])
    save_model(pipeline, "models/slum_classifier.pkl")

    result_df = predict(pipeline, df)
    result_df.drop(columns=["geometry"], errors="ignore").to_csv(
        config["output"]["prediction_report"], index=False
    )

    infra_df = prioritize_infrastructure(result_df, config)
    infra_df.to_csv(config["output"]["infrastructure_plan"], index=False)

    informal_count = (result_df["pred_class"] == "Informal").sum()
    print(f"[4/5] {informal_count} informal settlement tiles detected")
    print(f"      Infrastructure plan: {len(infra_df)} priority entries")

    gdf = to_geodataframe(result_df)
    plot_slum_map(gdf, config["output"]["slum_map"])
    plot_confusion_matrix(metrics["confusion_matrix"], config["output"]["confusion_matrix"])
    plot_class_distribution(result_df, "outputs/class_distribution.png")
    print("[5/5] All outputs saved to /outputs/")
    print("\nDone.")


if __name__ == "__main__":
    main()
