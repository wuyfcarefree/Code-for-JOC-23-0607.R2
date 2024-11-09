import glob
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib

'''
folder_path: Path to the folder containing hourly data files
'''

# Folder path
folder_path = r"your_work_path"
file_paths = glob.glob(os.path.join(folder_path, '*.xlsx'))
file_names = [os.path.basename(file) for file in file_paths]
count = 1

for file_name in file_names:
    dataset = pd.read_excel(os.path.join(folder_path, file_name))
    dataset['Hour'] = np.sin(2 * np.pi * dataset['Hour'] / 24)  # Encode hour as a cyclic feature
    x_train0 = dataset.iloc[:, [1, 5, 6, 7, 8, 9, 10]].values
    y_train0 = dataset.iloc[:, 4].values

    # Shuffle data
    shuffled_indices = np.random.permutation(x_train0.shape[0])
    x_train0 = x_train0[shuffled_indices]
    y_train0 = y_train0[shuffled_indices]
    x_train1, x_test1, y_train1, y_test1 = train_test_split(x_train0, y_train0, test_size=0.3)

    # Combine data from all files
    if count == 1:
        x_train = x_train1
        y_train = y_train1
        x_test = x_test1
        y_test = y_test1
    else:
        x_train = np.vstack((x_train, x_train1))
        x_test = np.vstack((x_test, x_test1))
        y_train = np.concatenate((y_train, y_train1))
        y_test = np.concatenate((y_test, y_test1))
    count += 1

# Train the Random Forest model
regressor = RandomForestRegressor(n_estimators=100, max_depth=40, max_features=7, random_state=42)
regressor.fit(x_train, y_train)

# Make predictions and evaluate
y_test_pred = regressor.predict(x_test)
y_train_pred = regressor.predict(x_train)
rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))

# Feature importance analysis
x_column_name = ['Hour', 'SKT', 'U10', 'V10', 'netLW', 'netSHW', 'TCC']
random_forest_importance = regressor.feature_importances_

# Save feature importances to Excel
importance_df = pd.DataFrame({
    'Feature': x_column_name,
    'Importance': random_forest_importance
})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Save as Excel file
output_path = r"path\results.xlsx"
importance_df.to_excel(output_path, index=False)

# Plot feature importance
plt.figure()
plt.clf()
importance_plot_x_values = list(range(len(random_forest_importance)))
plt.bar(importance_plot_x_values, random_forest_importance, orientation='vertical')
plt.xticks(importance_plot_x_values, x_column_name)
plt.xlabel('Variable')
plt.ylabel('Importance')
plt.ylim([0, 0.25])
plt.show()
