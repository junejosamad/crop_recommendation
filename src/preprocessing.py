from __future__ import annotations

import numpy as np
import pandas as pd

from utils import FEATURE_COLUMNS


def load_crop_dataset(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def clean_crop_dataset(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    for column in FEATURE_COLUMNS:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
        lower = cleaned[column].quantile(0.01)
        upper = cleaned[column].quantile(0.99)
        cleaned[column] = cleaned[column].clip(lower=lower, upper=upper)
        cleaned[column] = cleaned[column].fillna(cleaned[column].median())
    cleaned["label"] = cleaned["label"].astype(str).str.strip().str.lower()
    return cleaned


def add_yield_index(df: pd.DataFrame) -> pd.DataFrame:
    """Create a deterministic crop yield index for the regression module.

    The public crop recommendation dataset contains crop suitability labels but no
    measured yield column. This target combines nutrient balance, rainfall,
    humidity, pH, and temperature suitability into a quantitative yield index.
    """
    engineered = df.copy()
    n_balance = 1 - np.abs(engineered["N"] - 70) / 120
    p_balance = 1 - np.abs(engineered["P"] - 55) / 110
    k_balance = 1 - np.abs(engineered["K"] - 55) / 150
    ph_score = 1 - np.abs(engineered["ph"] - 6.5) / 4
    temp_score = np.exp(-((engineered["temperature"] - 26) ** 2) / 80)
    humidity_score = engineered["humidity"] / 100
    rainfall_score = np.minimum(engineered["rainfall"] / 180, 1.35)

    crop_factor = engineered["label"].map(
        {
            "rice": 1.18,
            "banana": 1.15,
            "grapes": 1.12,
            "cotton": 1.08,
            "coffee": 1.06,
            "maize": 1.04,
            "jute": 1.03,
            "coconut": 1.03,
        }
    ).fillna(1.0)

    base = (
        2.0
        + 1.25 * n_balance
        + 1.05 * p_balance
        + 1.00 * k_balance
        + 1.35 * ph_score
        + 1.55 * temp_score
        + 1.25 * humidity_score
        + 1.45 * rainfall_score
    )
    engineered["yield_index"] = (base * crop_factor).clip(lower=1.0, upper=12.5)
    return engineered


def prepare_dataset(path: str) -> pd.DataFrame:
    raw = load_crop_dataset(path)
    cleaned = clean_crop_dataset(raw)
    return add_yield_index(cleaned)
