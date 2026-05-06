import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
SAMPLE_DIR = os.path.join(DATA_DIR, "sample")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
MODELS_DIR = os.path.join(BASE_DIR, "models")

STUDY_CITIES = {
    "Lagos": {"lat": 6.5244, "lon": 3.3792},
    "Abuja": {"lat": 9.0765, "lon": 7.3986},
    "Kano": {"lat": 12.0022, "lon": 8.5920},
    "Accra": {"lat": 5.6037, "lon": -0.1870},
}

IMG_SIZE = 64
N_CHANNELS = 4  # R, G, B, NIR
PATCH_SIZE_M = 250

LAND_COVER_CLASSES = {
    0: "formal_residential",
    1: "informal_settlement",
    2: "commercial",
    3: "industrial",
    4: "green_space",
    5: "water_body",
    6: "bare_land",
}

CNN_PARAMS = {
    "epochs": 20,
    "batch_size": 32,
    "learning_rate": 0.001,
    "dropout_rate": 0.4,
}
