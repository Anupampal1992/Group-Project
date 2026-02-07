# Traffic Accident Severity Prediction — Project Report

## 1) Problem Statement
Traffic departments want to predict accident severity (Low / Moderate / High) based on environmental, road, vehicle, and driver-related factors. This project builds a classification pipeline, documents exploratory data analysis, preprocessing, baseline modeling, evaluation, and insights.

## 2) Dataset Overview
- **Source file:** `Traffic Accident Severity Predictor Dataset.csv`
- **Rows:** 20,000 total
- **Target:** `Accident_Severity` (Low / Moderate / High)
- **Features used:** Weather, Road_Type, Time_of_Day, Traffic_Density, Speed_Limit, Number_of_Vehicles, Driver_Alcohol, Road_Condition, Vehicle_Type, Driver_Age, Driver_Experience, Road_Light_Condition
- **Ignored column:** `Accident` (binary flag; excluded to avoid leakage and because it is not part of the stated feature list)

### Missing Values
There are **42 rows** with missing values across all feature columns and `Accident_Severity`. These rows were removed for modeling to maintain consistent input.

### Class Distribution (after dropping missing rows)
- Low: 11,862
- Moderate: 6,090
- High: 2,006

This indicates a **class imbalance** toward Low severity.

## 3) Exploratory Data Analysis (EDA)
Summary observations from quick inspection:
- **Weather:** Clear and Rainy conditions dominate the dataset.
- **Class imbalance** suggests we should evaluate with macro-averaged metrics (e.g., Macro F1) in addition to accuracy.

## 4) Data Preprocessing
- **Missing values:** rows with missing values removed.
- **Feature types:**
  - **Categorical:** Weather, Road_Type, Time_of_Day, Road_Condition, Vehicle_Type, Road_Light_Condition
  - **Numeric:** Traffic_Density, Speed_Limit, Number_of_Vehicles, Driver_Alcohol, Driver_Age, Driver_Experience
- **Train/test split:** stratified 80/20 split with fixed random seed (42).

## 5) Modeling Approach
A **Naive Bayes baseline** was implemented in pure Python:
- **Numeric features:** Gaussian Naive Bayes (mean/variance per class).
- **Categorical features:** Multinomial Naive Bayes with Laplace smoothing.

This baseline is intentionally simple and can be used as a reference point before more advanced models.

## 6) Model Evaluation
Baseline performance on the test split:
- **Accuracy:** 0.5910
- **Macro F1:** 0.2574

Per-class metrics:
- **Low:** Precision 0.5942, Recall 0.9867, F1 0.7417
- **Moderate:** Precision 0.3585, Recall 0.0159, F1 0.0305
- **High:** Precision 0.0000, Recall 0.0000, F1 0.0000

**Confusion matrix insights:**
- The model overwhelmingly predicts **Low**, which is consistent with the class imbalance.
- High-severity cases are not captured by the baseline model, indicating a need for improved modeling and balancing techniques.

## 7) Insights & Recommendations
- **Class imbalance** strongly affects model performance. Consider:
  - Oversampling (SMOTE), undersampling, or class-weighted loss.
- **Model upgrades:** Random Forests, Gradient Boosted Trees, or XGBoost often perform better on mixed data types.
- **Feature engineering:**
  - Bin numeric variables (e.g., speed limits or driver experience bands).
  - Combine conditions (e.g., `Weather + Road_Condition`) to capture interactions.

## 8) Next Steps
- Introduce a robust feature engineering pipeline with categorical encoding and scaling.
- Experiment with tree-based models + hyperparameter tuning.
- Create visualization dashboards: class distribution, feature-importance plots, confusion matrices.

## 9) Reproducibility
Run the baseline model:
```bash
python traffic_accident_model.py
```

## 10) Team Collaboration Notes
- Clear separation of tasks: EDA, preprocessing, modeling, evaluation, documentation.
- A shared report and presentation outline are provided for final delivery.
