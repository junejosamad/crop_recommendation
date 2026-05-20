# Crop Recommendation Dataset Metadata

Source dataset: public Crop Recommendation dataset, commonly mirrored from Kaggle.

Rows: 2,200  
Features: 7 numeric agronomic inputs plus one crop label.

| Column | Description | Role |
| --- | --- | --- |
| N | Nitrogen ratio/content in soil | Input feature |
| P | Phosphorus ratio/content in soil | Input feature |
| K | Potassium ratio/content in soil | Input feature |
| temperature | Temperature in degree Celsius | Input feature |
| humidity | Relative humidity percentage | Input feature |
| ph | Soil pH value | Input feature |
| rainfall | Rainfall in millimeters | Input feature |
| label | Recommended crop type | Classification target |
| yield_index | Engineered quantitative yield suitability score | Regression target |

## Preprocessing Rationale

- Numeric fields are coerced to numeric values.
- Missing numeric values are imputed with the median.
- Outliers are clipped at the 1st and 99th percentiles to reduce the effect of extreme measurements.
- Labels are normalized to lowercase strings.
- The dataset does not contain measured crop yield, so `yield_index` is generated deterministically from nutrient balance, rainfall, humidity, pH suitability, temperature suitability, and crop-specific productivity factors.
