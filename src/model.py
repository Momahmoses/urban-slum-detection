"""
Urban informal settlement detection model.
Binary classifier using satellite imagery proxy features.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

FEATURE_COLS = [
    "ndvi", "built_up_density", "road_accessibility", "building_regularity",
    "roof_material_score", "mean_building_size_m2", "building_spacing_m",
    "texture_entropy", "brightness_index", "water_access_score",
    "sanitation_score", "population_density",
]
TARGET_COL = "is_informal_settlement"
MODEL_PATH = Path("assets/slum_model.pkl")
METRICS_PATH = Path("assets/metrics.json")


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(
            n_estimators=250, max_depth=5, learning_rate=0.07,
            subsample=0.8, min_samples_leaf=15, random_state=42,
        )),
    ])


def train(df: pd.DataFrame) -> tuple[Pipeline, dict]:
    X, y = df[FEATURE_COLS], df[TARGET_COL]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    pipeline = build_pipeline()
    pipeline.fit(X_tr, y_tr)
    y_pred = pipeline.predict(X_te)
    y_prob = pipeline.predict_proba(X_te)[:, 1]
    cv_auc = cross_val_score(pipeline, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=42), scoring="roc_auc")
    report = classification_report(y_te, y_pred, output_dict=True)
    metrics = {
        "roc_auc_test": round(float(roc_auc_score(y_te, y_prob)), 4),
        "cv_auc_mean": round(float(cv_auc.mean()), 4),
        "cv_auc_std": round(float(cv_auc.std()), 4),
        "accuracy": round(float(report["accuracy"]), 4),
        "f1_slum": round(float(report["1"]["f1-score"]), 4),
        "recall_slum": round(float(report["1"]["recall"]), 4),
        "confusion_matrix": confusion_matrix(y_te, y_pred).tolist(),
        "feature_importance": dict(zip(FEATURE_COLS, pipeline.named_steps["clf"].feature_importances_.tolist())),
    }
    return pipeline, metrics


def predict_cell(pipeline: Pipeline, features: dict) -> dict:
    X = pd.DataFrame([features])[FEATURE_COLS]
    prob = float(pipeline.predict_proba(X)[0, 1])
    label = int(pipeline.predict(X)[0])
    tier = "High" if prob >= 0.65 else "Medium" if prob >= 0.40 else "Low"
    return {"informal_probability": round(prob, 4), "is_informal": label, "confidence_tier": tier}


def save_model(pipeline: Pipeline, metrics: dict) -> None:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)


def load_model() -> Pipeline:
    return joblib.load(MODEL_PATH)
