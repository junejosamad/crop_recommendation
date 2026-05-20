# Smart Agriculture Decision Support System

This project assembles three classical AI modules into one agricultural decision-support application:

- Decision Tree Classifier for crop recommendation.
- KNN-based soil zone assignment trained from K-Means reference clusters.
- Linear Regression for quantitative crop yield-index prediction.

The system uses the public Crop Recommendation dataset with soil nutrients and climate readings. Because the dataset has crop labels but no measured yield column, this project creates a deterministic `yield_index` target from agronomic suitability factors and documents that engineering choice in `data/metadata.md`.

## Architecture

```text
Raw CSV dataset
    |
    v
src/preprocessing.py
    |-- missing value imputation
    |-- outlier clipping
    |-- feature cleanup
    |-- engineered yield_index
    |
    v
src/models.py
    |-- Decision Tree crop classifier
    |-- KNN soil segmentation model
    |-- Linear Regression yield model
    |-- serialized model bundle
    |-- metrics and plots
    |
    v
src/gui.py
    |-- Tkinter input form
    |-- integrated model inference
    |-- embedded matplotlib visualizations
```

## Repository Layout

```text
repository/
├── data/
│   ├── Crop_recommendation.csv
│   ├── processed_crop_data.csv
│   └── metadata.md
├── src/
│   ├── preprocessing.py
│   ├── models.py
│   ├── gui.py
│   └── utils.py
├── models/
│   └── agri_ai_bundle.joblib
├── results/
│   ├── metrics.json
│   ├── feature_importance.png
│   ├── cluster_scatter.png
│   └── residual_plot.png
├── requirements.txt
├── LICENSE
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Train Models

```bash
python src/models.py
```

This command regenerates:

- `models/agri_ai_bundle.joblib`
- `results/metrics.json`
- `results/feature_importance.png`
- `results/cluster_scatter.png`
- `results/residual_plot.png`
- `data/processed_crop_data.csv`

## Run GUI

```bash
python src/gui.py
```

Enter soil and climate values, then select **Run Inference**. The GUI returns the recommended crop, soil cluster, agronomic guidance, predicted yield index, and an approximate confidence bound.

## Algorithmic Rationale

Decision Trees are interpretable and expose feature importance, making them suitable for crop recommendation where agronomists need to understand which soil or climate variables influenced a decision.

The soil segmentation module first derives homogeneous zones using K-Means reference labels, then trains a KNN classifier to assign new farm readings to those zones. This keeps inference simple while still reporting a clustering-oriented silhouette score.

Linear Regression is used for transparent yield-index prediction. Its residual plot and regression metrics help determine whether the engineered quantitative target is being modeled consistently.

## Quantitative Performance Summary

| Module | Metric | Result |
| --- | --- | ---: |
| Decision Tree Classifier | Accuracy | 0.9250 |
| Decision Tree Classifier | Macro Precision | 0.9142 |
| Decision Tree Classifier | Macro Recall | 0.9250 |
| KNN Soil Segmentation | Silhouette Score | 0.2929 |
| Linear Regression | RMSE | 0.5306 |
| Linear Regression | MAE | 0.4084 |
| Linear Regression | R² | 0.8089 |

## Visualizations

- Feature importance: `results/feature_importance.png`
- Cluster scatter: `results/cluster_scatter.png`
- Residual plot: `results/residual_plot.png`

## Future Work

1. Integrate live IoT soil sensors for real-time nitrogen, phosphorus, potassium, temperature, humidity, and pH acquisition.
2. Fuse satellite imagery and weather forecasts with ensemble learning to improve crop recommendation under changing regional climate conditions.
