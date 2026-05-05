import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib


FEATURES = [
    "mean_r", "mean_g", "mean_b", "mean_nir", "std_texture",
    "roof_density", "building_regularity", "road_distance_m",
    "ndvi", "brightness", "entropy", "edge_density"
]
CLASS_NAMES = ["Formal Residential", "Informal Settlement", "Industrial/Commercial"]


def build_pipeline():
    clf = GradientBoostingClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        subsample=0.8, random_state=42
    )
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


def train(df, config):
    X = df[FEATURES]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config["model"]["test_size"],
        random_state=config["model"]["random_state"],
        stratify=y
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred, target_names=CLASS_NAMES),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
    }
    return pipeline, metrics, (X_test, y_test, y_pred)


def predict(pipeline, df):
    df = df.copy()
    probs = pipeline.predict_proba(df[FEATURES])
    df["pred_label"] = pipeline.predict(df[FEATURES])
    df["pred_class"] = df["pred_label"].map({0: "Formal", 1: "Informal", 2: "Industrial"})
    df["informal_probability"] = probs[:, 1]
    return df


def save_model(pipeline, path):
    joblib.dump(pipeline, path)


def load_model(path):
    return joblib.load(path)


def feature_importance(pipeline):
    clf = pipeline.named_steps["clf"]
    return pd.DataFrame({
        "feature": FEATURES,
        "importance": clf.feature_importances_
    }).sort_values("importance", ascending=False)


def prioritize_infrastructure(df, config):
    informal = df[df["pred_class"] == "Informal"].copy()
    informal["infra_priority_score"] = (
        informal["informal_probability"]
        * np.log1p(informal.get("population_density", 1))
    )
    priorities = []
    for _, row in informal.iterrows():
        for infra in config["infrastructure_priorities"]:
            priorities.append({
                "tile_id": row["tile_id"],
                "latitude": row["latitude"],
                "longitude": row["longitude"],
                "city": row["city"],
                "infrastructure": infra,
                "priority_score": row["infra_priority_score"]
            })
    return pd.DataFrame(priorities).sort_values("priority_score", ascending=False)
