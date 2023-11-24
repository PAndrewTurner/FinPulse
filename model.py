import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split # for splitting the data
from sklearn.metrics import mean_squared_error # for calculating the cost function
import matplotlib.pyplot as plt
import pickle as pkl
import warnings
np.set_printoptions(precision=2)
warnings.filterwarnings("ignore")

data = pd.read_csv("Walmart_Store_sales.csv")

x = data.drop(['Weekly_Sales', 'Date'], axis = 1) # Features
y = data['Weekly_Sales'] # Target
data.head()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 28)
model = RandomForestRegressor(n_estimators = 1000, random_state = 28)
model.fit(x_train, y_train)

filename = 'model.pkl'
pkl.dump(model, open(filename, 'wb'))
y_pred = model.predict(x_test)
rmse = float(format(np.sqrt(mean_squared_error(y_test, y_pred)), '.3f'))
print("\nRMSE: ", rmse)

test_data = pd.read_csv("test_data.csv")

vals = []
dates = []

for i in range(len(test_data)):
    st = str(test_data['Month'].loc[i]) + "/" + str(test_data['Day'].loc[i]) + "/" + str(test_data['Year'].loc[i])
    dates.append(st)
print(dates)

feature_columns = ['Store', 'Month', 'Day', 'Year', 'Holiday _Flag', 'Temprature', 'Fuel_Price', 'CPI', 'Unemployment']

for index, row in test_data.iterrows():
    # Extract features from the current row, replace this with your actual feature extraction logic
    # For example, assuming 'feature_columns' is a list of column names used as features
    features = row[feature_columns].values.reshape(1, -1)

    # Make a prediction using the model
    vals.append(model.predict(features))

print(vals)

for x in range(len(vals)):
    amnt = str(vals[x]).replace('[', '').replace(']', '')
    print("Date: ", dates[x], "\tPredicted Income: $", amnt)