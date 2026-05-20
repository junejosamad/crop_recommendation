from __future__ import annotations

import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from preprocessing import prepare_dataset
from utils import DATA_PATH, FEATURE_COLUMNS, MODEL_DIR, RESULTS_DIR, ensure_directories


def _numeric_preprocessor(scale: bool = False) -> ColumnTransformer:
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scaler", StandardScaler()))
    return ColumnTransformer([("num", Pipeline(steps), FEATURE_COLUMNS)])


def _save_feature_importance(model: Pipeline) -> str:
    tree = model.named_steps["classifier"]
    importances = pd.Series(tree.feature_importances_, index=FEATURE_COLUMNS).sort_values()
    fig, ax = plt.subplots(figsize=(8, 5))
    importances.plot(kind="barh", ax=ax, color="#2f6f73")
    ax.set_title("Decision Tree Feature Importance")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    path = RESULTS_DIR / "feature_importance.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return str(path)


def _save_cluster_plot(X_scaled: np.ndarray, labels: np.ndarray) -> str:
    fig, ax = plt.subplots(figsize=(8, 5))
    scatter = ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap="tab10", s=28, alpha=0.8)
    ax.set_title("Soil Profile Clusters")
    ax.set_xlabel("Scaled nitrogen")
    ax.set_ylabel("Scaled phosphorus")
    fig.colorbar(scatter, ax=ax, label="Cluster")
    fig.tight_layout()
    path = RESULTS_DIR / "cluster_scatter.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return str(path)


def _save_residual_plot(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    residuals = y_true - y_pred
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(y_pred, residuals, color="#7a3e65", alpha=0.75, s=32)
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_title("Linear Regression Residual Analysis")
    ax.set_xlabel("Predicted yield index")
    ax.set_ylabel("Residual")
    fig.tight_layout()
    path = RESULTS_DIR / "residual_plot.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return str(path)


def train_all() -> dict:
    ensure_directories()
    df = prepare_dataset(str(DATA_PATH))
    X = df[FEATURE_COLUMNS]

    label_encoder = LabelEncoder()
    y_crop = label_encoder.fit_transform(df["label"])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_crop, test_size=0.2, random_state=42, stratify=y_crop
    )

    crop_model = Pipeline(
        [
            ("preprocess", _numeric_preprocessor(scale=False)),
            ("classifier", DecisionTreeClassifier(max_depth=9, random_state=42)),
        ]
    )
    crop_model.fit(X_train, y_train)
    y_pred = crop_model.predict(X_test)
    crop_metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision_macro": round(float(precision_score(y_test, y_pred, average="macro", zero_division=0)), 4),
        "recall_macro": round(float(recall_score(y_test, y_pred, average="macro", zero_division=0)), 4),
    }

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    knn_cluster_model = Pipeline(
        [
            ("preprocess", _numeric_preprocessor(scale=True)),
            ("clusterer", KNeighborsClassifier(n_neighbors=7, weights="distance")),
        ]
    )
    knn_cluster_model.fit(X, cluster_labels)
    cluster_metrics = {
        "clusters": 5,
        "silhouette_score": round(float(silhouette_score(X_scaled, cluster_labels)), 4),
    }

    y_yield = df["yield_index"]
    Xr_train, Xr_test, yr_train, yr_test = train_test_split(
        X, y_yield, test_size=0.2, random_state=42
    )
    yield_model = Pipeline(
        [
            ("preprocess", _numeric_preprocessor(scale=True)),
            ("regressor", LinearRegression()),
        ]
    )
    yield_model.fit(Xr_train, yr_train)
    yr_pred = yield_model.predict(Xr_test)
    rmse = float(np.sqrt(mean_squared_error(yr_test, yr_pred)))
    yield_metrics = {
        "rmse": round(rmse, 4),
        "mae": round(float(mean_absolute_error(yr_test, yr_pred)), 4),
        "r2": round(float(r2_score(yr_test, yr_pred)), 4),
    }

    plots = {
        "feature_importance": _save_feature_importance(crop_model),
        "cluster_scatter": _save_cluster_plot(X_scaled, cluster_labels),
        "residual_plot": _save_residual_plot(yr_test.to_numpy(), yr_pred),
    }

    bundle = {
        "feature_columns": FEATURE_COLUMNS,
        "crop_model": crop_model,
        "label_encoder": label_encoder,
        "knn_cluster_model": knn_cluster_model,
        "kmeans_reference": kmeans,
        "cluster_scaler": scaler,
        "yield_model": yield_model,
        "yield_rmse": rmse,
        "metrics": {
            "decision_tree_classifier": crop_metrics,
            "knn_soil_segmentation": cluster_metrics,
            "linear_regression_yield": yield_metrics,
        },
        "plots": plots,
    }
    joblib.dump(bundle, MODEL_DIR / "agri_ai_bundle.joblib")

    with (RESULTS_DIR / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(bundle["metrics"], f, indent=2)

    processed_path = DATA_PATH.parent / "processed_crop_data.csv"
    df.to_csv(processed_path, index=False)
    return bundle["metrics"]


if __name__ == "__main__":
    metrics = train_all()
    print(json.dumps(metrics, indent=2))
