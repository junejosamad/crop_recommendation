# Integration and Deployment of a Multi-Model Agricultural Intelligence System

## Abstract

This report presents a Smart Agriculture Decision Support System that integrates crop recommendation, soil profile segmentation, and crop yield-index prediction into a unified Python application. The system uses a public crop recommendation dataset containing nitrogen, phosphorus, potassium, temperature, humidity, pH, rainfall, and crop labels. A Decision Tree Classifier recommends crop type, a KNN-based segmentation component assigns soil zones, and a Linear Regression model predicts a quantitative yield index. The trained models are serialized with joblib and exposed through a Tkinter graphical interface with embedded matplotlib visualizations.

## Introduction

Precision agriculture increasingly relies on data-driven decision support to optimize crop choice, fertilizer planning, irrigation, and field zoning. Traditional farm advisory workflows often treat crop selection, soil classification, and yield estimation as separate tasks. A unified system can reduce this fragmentation by converting soil and climate measurements into integrated recommendations. Prior research in agricultural machine learning has shown that soil nutrients, rainfall, temperature, humidity, and pH can support crop recommendation and farm planning. This project applies interpretable classical models so the final system remains suitable for educational and resource-constrained deployment contexts.

## Methodology

The dataset was loaded from `data/Crop_recommendation.csv` and processed through `src/preprocessing.py`. Numeric inputs were coerced to numeric types, missing values were imputed with medians, outliers were clipped at the 1st and 99th percentiles, and crop labels were normalized. Since the dataset does not include observed crop yield, a deterministic `yield_index` was engineered from nutrient balance, pH suitability, temperature suitability, humidity, rainfall, and crop-specific productivity factors.

The Decision Tree Classifier was trained using the seven numeric input variables to predict the crop label. This model was selected because its feature importance vector supports transparent explanation. The soil segmentation module derives reference clusters using K-Means and trains a KNN classifier to assign new readings to the nearest homogeneous soil zone. The Linear Regression model predicts the engineered yield index and supports residual analysis. All models are saved inside `models/agri_ai_bundle.joblib`.

The GUI was implemented in Tkinter. It accepts user inputs for N, P, K, temperature, humidity, pH, and rainfall. It then invokes the serialized crop classifier, soil segmentation model, and yield model sequentially. The interface displays the recommended crop, soil zone, agronomic guidance, predicted yield index, and approximate confidence bounds.

## Results and Discussion

The Decision Tree Classifier achieved 0.9250 accuracy, 0.9142 macro precision, and 0.9250 macro recall. These results indicate strong crop-label prediction on the selected dataset. The KNN soil segmentation workflow produced a silhouette score of 0.2929, suggesting moderate cluster separation. The Linear Regression model achieved RMSE 0.5306, MAE 0.4084, and R² 0.8089 against the engineered yield index. The residual plot indicates whether regression errors are centered near zero and whether systematic bias remains.

The feature-importance plot shows which soil and climate variables contribute most to crop recommendations. The cluster scatter plot visualizes soil zone separation using scaled nutrient dimensions. The residual plot supports evaluation of the regression module. A key limitation is that the yield target is engineered rather than measured from field harvest records, so the regression component demonstrates integration and modeling workflow rather than validated agronomic yield forecasting.

## Industrial Application

In a commercial agri-tech setting, this system could support farm managers by connecting soil test reports and climate readings to actionable recommendations. The crop classifier can guide crop selection, the soil segmentation module can support variable-rate fertilizer and irrigation zones, and the yield model can provide a preliminary productivity indicator. With IoT sensors and real yield records, the architecture could be extended into a deployable advisory dashboard for regional agronomists and growers.

## Research Extension

Future work should integrate live IoT sensor networks for continuous soil and microclimate monitoring. This would allow model predictions to update as field conditions change. A second extension is satellite imagery fusion, where vegetation indices, surface temperature, and regional rainfall forecasts could be combined with tabular soil features using ensemble models or deep learning.

## Conclusion

The project demonstrates integration of classification, clustering-oriented segmentation, and regression into a single agricultural intelligence pipeline. It includes modular preprocessing, serialized models, quantitative metrics, visualization artifacts, and a Tkinter interface. The assembled system satisfies the core OEL goal of binding multiple AI components into an executable decision-support application while maintaining clear documentation and reproducible training.
