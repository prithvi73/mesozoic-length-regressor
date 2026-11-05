# Mesozoic Length Regressor 🦖

A Supervised Machine Learning Regression pipeline designed to predict the continuous physical length (in meters) of prehistoric animals based strictly on their taxonomic classifications, diet, and temporal fossil records.

## Overview
This project processes paleontological data, heavily focusing on Feature Engineering and categorical text translation. The pipeline ingests non-numeric biological descriptors and transforms them into a mathematically pure, scaled feature matrix to calculate continuous size estimates without relying on direct skeletal measurements.

## Technical Flow
* **Data Preprocessing:** Cleans raw fossil records by purging arbitrary identifiers and geographic coordinates that hold no physical correlation to the animal's size.
* **Feature Encoding:** Utilizes Pandas `get_dummies()` to apply One-Hot Encoding to categorical text columns (`diet`, `type`, `class`, `family`), transforming biological taxonomy into binary integer matrices.
* **Feature Scaling:** Implements Scikit-learn's `StandardScaler` to normalize the resulting feature matrix distributions.
* **Model Training:** Partitions the matrix into an 80/20 split and trains a `RandomForestRegressor` to map the non-linear relationships between taxonomy and physical dimensions.
* **Evaluation:** Tests inference on hidden data, achieving a Mean Absolute Error (MAE) of ~0.69 meters.

## Tech Stack
* Python
* Pandas
* Scikit-learn