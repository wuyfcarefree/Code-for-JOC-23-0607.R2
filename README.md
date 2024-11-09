# Random Forest Model for Feature Importance and Prediction Analysis on Hourly Data
This repository contains a Python script for training a Random Forest model to analyze feature importance and make predictions on hourly environmental data. The code processes multiple Excel files containing time-series data, trains the model on a subset of the data, and evaluates its predictive accuracy on the test set. The results include RMSE values and feature importance scores, which are saved in an Excel file and visualized with a bar plot.

# Prerequisites
Ensure you have the following libraries installed:
﻿- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `joblib`
- `openpyxl`

# Folder Structure
﻿The main script requires a folder with multiple Excel files, each representing data for a specific hour. Make sure to update the `folder_path` variable with the path to your data folder.

 # Data Format
﻿Each Excel file should include the following columns:
1. `Hour` - Encoded cyclically to represent time of day.
2. `SKT`, `U10`, `V10`, `netLW`, `netSHW`, `TCC` - Features used as predictors.
3. Target column - The target variable used for model training and evaluation.

# Script Overview
﻿1. **Data Loading and Preprocessing**:
- The script reads all Excel files from the specified folder, shuffles each dataset, and applies a cyclic encoding to the `Hour` column.
- Data from all files are concatenated for training and testing.
﻿2. **Model Training**:
- A Random Forest Regressor is trained on the combined data with specified parameters (`n_estimators`, `max_depth`, `max_features`, etc.).
﻿3. **Evaluation**:
- Predictions are made on the test set, and evaluation metrics (RMSE, MAE, R²) are calculated for both training and testing sets.
﻿4. **Feature Importance Analysis**:
- Feature importance values from the Random Forest model are extracted and saved to an Excel file (`results.xlsx`).
- A bar plot is generated to visualize the importance of each feature.
