"""CNN model for satellite image land cover classification."""

import numpy as np
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import IMG_SIZE, N_CHANNELS, MODELS_DIR, OUTPUTS_DIR, CNN_PARAMS, LAND_COVER_CLASSES

N_CLASSES = len(LAND_COVER_CLASSES)


def build_cnn():
    """Build CNN using only numpy (no deep learning framework required for demo)."""
    # For production: use TensorFlow/Keras CNN. Here we use a sklearn RF on flattened patches.
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(
        n_estimators=150,
        max_depth=15,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )


def extract_features(patches: np.ndarray) -> np.ndarray:
    """Extract hand-crafted spectral + texture features from image patches."""
    n = patches.shape[0]
    feats = []
    for i in range(n):
        p = patches[i]
        r, g, b, nir = p[:, :, 0], p[:, :, 1], p[:, :, 2], p[:, :, 3]
        ndvi = (nir - r) / (nir + r + 1e-9)
        ndbi = (r - nir) / (r + nir + 1e-9)  # approx built-up index
        feat = [
            r.mean(), r.std(), g.mean(), g.std(), b.mean(), b.std(),
            nir.mean(), nir.std(),
            ndvi.mean(), ndvi.std(), ndvi.max(),
            ndbi.mean(), ndbi.std(),
            p.mean(), p.std(),
            np.percentile(r, 25), np.percentile(r, 75),
            np.percentile(nir, 25), np.percentile(nir, 75),
        ]
        feats.append(feat)
    return np.array(feats, dtype=np.float32)


def train(patches: np.ndarray, labels: np.ndarray):
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    import joblib

    X = extract_features(patches)
    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=0.2, random_state=42, stratify=labels
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = build_cnn()
    clf.fit(X_train_s, y_train)

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(clf, os.path.join(MODELS_DIR, "slum_classifier.pkl"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
    print("Model saved.")
    return clf, scaler, X_test_s, y_test


def evaluate(clf, X_test, y_test):
    from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
    import matplotlib.pyplot as plt

    y_pred = clf.predict(X_test)
    class_names = list(LAND_COVER_CLASSES.values())
    print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=45)
    plt.title("Confusion Matrix — Urban Land Cover Classification")
    plt.tight_layout()
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    plt.savefig(os.path.join(OUTPUTS_DIR, "confusion_matrix.png"), dpi=150)
    plt.close()
