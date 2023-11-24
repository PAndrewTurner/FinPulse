import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split  # for splitting the data
from sklearn.metrics import mean_squared_error  # for calculating the cost function
import matplotlib.pyplot as plt
import pickle as pkl
import warnings

# Set numpy print options for better readability
np.set_printoptions(precision=2)

# Suppress warnings for better readability
warnings.filterwarnings("ignore")

# Read the Walmart store sales data from a CSV file
data = pd.read_csv("Walmart_Store_sales.csv")

# Extract features (x) and target variable (y) from the data
x = data.drop(['Weekly_Sales', 'Date'], axis=1)  # Features
y = data['Weekly_Sales']  # Target

# Display the head of the dataset
data.head()

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=28)

# Create and train a RandomForestRegressor model
model = RandomForestRegressor(n_estimators=1000, random_state=28)
model.fit(x_train, y_train)

# Save the trained model to a file using pickle
filename = 'model.pkl'
pkl.dump(model, open(filename, 'wb'))

# Make predictions on the test set
y_pred = model.predict(x_test)

# Calculate and print the Root Mean Squared Error (RMSE)
rmse = float(format(np.sqrt(mean_squared_error(y_test, y_pred)), '.3f'))
print("\nRMSE: ", rmse)

# Read the test data from a CSV file
test_data = pd.read_csv("test_data.csv")

# Initialize empty lists for storing predictions and dates
vals = []
dates = []

# Extract dates from the test_data and append to the dates list
for i in range(len(test_data)):
    st = str(test_data['Month'].loc[i]) + "/" + str(test_data['Day'].loc[i]) + "/" + str(test_data['Year'].loc[i])
    dates.append(st)

print(dates)

# Define feature columns for the test data
feature_columns = ['Store', 'Month', 'Day', 'Year', 'Holiday _Flag', 'Temprature', 'Fuel_Price', 'CPI', 'Unemployment']

# Make predictions on the test data
for index, row in test_data.iterrows():
    # Extract features from the current row
    features = row[feature_columns].values.reshape(1, -1)

    # Make a prediction using the trained model
    vals.append(model.predict(features))

print(vals)

# Display the predictions along with the corresponding dates
for x in range(len(vals)):
    amnt = str(vals[x]).replace('[', '').replace(']', '')
    print("Date: ", dates[x], "\tPredicted Income: $", amnt)
