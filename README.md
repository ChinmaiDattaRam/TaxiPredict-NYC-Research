# Predicting NYC Taxi Trip Duration Using Machine Learning and XGBoost

### 🚕 Predict taxi travel times in NYC using state-of-the-art ML models.

This repository contains the complete pipeline for predicting NYC taxi trip durations using the 2024 Yellow Taxi dataset. We implement a variety of regression baselines, a highly-tuned XGBoost model, and an experimental Graph Neural Network (GraphSAGE) to capture zone-to-zone relationships.

## Key Features
- **Exploratory Data Analysis**: Automated cleaning and statistical filtering.
- **Advanced Feature Engineering**: Temporal (cyclic hour encoding), Economic (fare-per-mile), and Kinematic (average speed) features.
- **Model Suite**: 
    - Linear Regression
    - K-Nearest Neighbors (KNN)
    - Decision Tree & Random Forest
    - XGBoost (Tuned Final Model)
    - Graph Neural Network (GraphSAGE implementation using `torch-geometric`)

## Performance
Based on the experiments in the paper, **XGBoost** achieved the best results with an $R^2$ score of approximately **0.90** and an MAE of **1.6 minutes**.

## Dataset
The project uses the **2024 NYC Yellow Taxi Trip Records**.
- **Source**: [NYC TLC Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)
- **File Requirement**: Place the dataset in the root directory and rename it to `ML_AAT.csv`.

## Usage
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`.
3. Add your data as `ML_AAT.csv`.
4. Run the pipeline: `python main.py`.

---
*Developed as part of the Information Science and Engineering research at B.M.S. College of Engineering, Bengaluru.*
