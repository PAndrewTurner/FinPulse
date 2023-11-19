# FinPulse

The FinPulse model uses <b>RandomForest regression</b> with the <b>sklearn</b> Python library. The dataset used is from Walmart's sales data from different stores across different regions and includes features such as date, store number, weekly sales, holidays, temprature, fuel price, consumer price index, and the unemployment rate.

The Dataset was split using the train_test_split function within the sklearn.model_selection library and then a Randon Forest model was created. The target variable was selected as the weekly sales, and all others except the full date string were chosen as the features. The full date was parsed into the seperate components for day, month, and year. 

The RandomForest model uses 1000 trees with random_state set at 28, and after fitting the model has a <b>mean square error (MSE)</b> of 150619.75

After the model was fit and tested, it was pickled and saved so other computers may open the model as needed to run the model.

In the second part of the code, we see preliminary explorartory analysis for the purpose of visualizing the dataset.

# Variables for ML RF model

Store: Int<br>
Month: Int<br>
Day: Int<br>
Month: Int<br>
Holiday_Flag: Binary (1 for holiday during the week, 0 for none)<br>
Temprature: Float<br>
Fuel Price: Float<br>
CPI: Float<br>
Unemployment: Float<br>

# Using the Model for Predictions
