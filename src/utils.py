from __future__ import annotations

from pathlib import Path

FEATURE_COLUMNS = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "Crop_recommendation.csv"
MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"


CLUSTER_GUIDANCE = {
    0: "Balanced zone: maintain current fertilizer schedule and monitor pH monthly.",
    1: "Moisture-sensitive zone: improve drainage and avoid over-irrigation.",
    2: "Nutrient-intensive zone: schedule soil testing before additional NPK application.",
    3: "Dry or low-input zone: prioritize mulching, irrigation planning, and organic matter.",
    4: "High-value specialty zone: keep tighter climate and irrigation records.",
}


def ensure_directories() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def yield_confidence_bounds(prediction: float, rmse: float) -> tuple[float, float]:
    """Approximate 95% prediction interval around the yield index."""
    margin = 1.96 * rmse
    return max(0.0, prediction - margin), prediction + margin


def agronomic_cluster_guidance(cluster_id: int) -> str:
    return CLUSTER_GUIDANCE.get(
        int(cluster_id),
        "General zone: validate recommendations with local agronomist observations.",
    )
