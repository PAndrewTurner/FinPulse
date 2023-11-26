import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
import joblib
import warnings
import joblib


np.set_printoptions(precision=2)
warnings.filterwarnings("ignore")

# Read the Walmart store sales data from a CSV file
data = pd.read_csv("Walmart_Store_sales.csv")

# Extract features (x) and target variable (y) from the data
x = data.drop(['Weekly_Sales', 'Date'], axis=1)
y = data['Weekly_Sales']

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=28)

# Create and train a RandomForestRegressor model
rf_model = RandomForestRegressor(n_estimators=1000, random_state=28)
rf_model.fit(x_train, y_train)

# Save the trained RandomForestRegressor model to a file using joblib
rf_filename = 'rf_model.joblib'
joblib.dump(rf_model, rf_filename)

print("RF Model Done")

# Create and train an XGBRegressor model
xgb_model = XGBRegressor(n_estimators=1000, random_state=28)
xgb_model.fit(x_train, y_train)

# Save the trained XGBRegressor model to a file using joblib
xgb_filename = 'xgb_model.joblib'
joblib.dump(xgb_model, xgb_filename)

print("XGB Model Done")

gb_model = GradientBoostingRegressor(n_estimators=1000, random_state=28)
gb_model.fit(x_train, y_train)

gb_filename = 'gb_model.joblib'
joblib.dump(gb_model, gb_filename)

# Make predictions on the test set using both models
y_pred_rf = rf_model.predict(x_test)
y_pred_xgb = xgb_model.predict(x_test)
y_pred_gb = gb_model.predict(x_test)


# Calculate and print the Root Mean Squared Error (RMSE) for both models
rmse_rf = float(format(np.sqrt(mean_squared_error(y_test, y_pred_rf)), '.3f'))
rmse_xgb = float(format(np.sqrt(mean_squared_error(y_test, y_pred_xgb)), '.3f'))
rmse_gb = float(format(np.sqrt(mean_squared_error(y_test, y_pred_gb)), '.3f'))
print("\nRandom Forest RMSE: ", rmse_rf)
print("XGBoost RMSE: ", rmse_xgb)
print("Gradient Boost RMSE: ", rmse_gb)

# Random Forest RMSE:  150619.75
# XGBoost RMSE:  142014.462
# Gradient Boost RMSE:  127761.866
