from pathlib import Path

import pandas as pd
from ucimlrepo import fetch_ucirepo

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
CSV_PATH = DATA_DIR / "heart_disease.csv"


def fetch_and_save_heart_disease_data(force_reload: bool = False) -> pd.DataFrame:
    """Download the UCI heart disease dataset and save a local CSV copy."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if CSV_PATH.exists() and not force_reload:
        return pd.read_csv(CSV_PATH)

    heart_disease = fetch_ucirepo(id=45)
    features = heart_disease.data.features.copy()
    targets = heart_disease.data.targets.copy()

    dataset = pd.concat([features, targets], axis=1)
    dataset = dataset.rename(columns={"num": "target"})
    dataset.to_csv(CSV_PATH, index=False)

    return dataset


def load_heart_disease_data() -> pd.DataFrame:
    """Load the saved dataset. If it does not exist, fetch and save it first."""
    if not CSV_PATH.exists():
        return fetch_and_save_heart_disease_data()

    return pd.read_csv(CSV_PATH)
