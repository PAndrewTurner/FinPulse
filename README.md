![alt text](https://github.com/PAndrewTurner/FinPulse/blob/main/FinPulse%20Logo.jpg?raw=true)

# FinPulse

FinPulse is an autonomous finance platform that uses machine learning and statistical programming to enhance the way business owners manage their finances. It offers bespoke machine learning-trained models to predict and forecast sales and income to help plan their business operations. 

The FinPulse model uses <b>RandomForest regression</b> with the <b>sklearn</b> Python library. The dataset used is from Walmart's sales data from different stores across different regions and includes features such as <b>date, store number, weekly sales, holidays, temprature, fuel price, consumer price index, and the unemployment rate.</b>

The Dataset was split using the <b>train_test_split</b> function within the <b>sklearn.model_selection</b> library and then a Randon Forest model was created. The target variable was selected as the weekly sales, and all others except the full date string were chosen as the features. The full date was parsed into the seperate components for day, month, and year. 

The RandomForest model uses 1000 trees with random_state set at 28, and after fitting the model has a <b>mean square error (MSE)</b> of 150619.75

After the model was fit and tested, it was pickled and saved so other computers may open the model as needed to run the model.

The ML model was created, trained, and fit in the model.py file, while a GUI was created to allow for interativity with the code, housed in the main.py file.

Due to the model being pre-trained, selecting the test_data.csv file selects the dataset to make predictions for the future. Both the dataset and the model must be selected. Once run, a grapgh can then be populated in the window, showing the predictions.

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

To use the model, it must be un-pickled using the Python <b>Pickle</b> library. It can be opened via the following:

<b>loaded_model = pickle.load(open(filename, 'rb'))</b><br>

After loading, the model can be used with the <b>predict()</b> method.<br>

<b> prediction = loaded_model.predict(new_data)</b><br>

This precit method then returns the predicted weekly sale data for the inputted parameters.

# Current Model Limitations

1) Most stores do not post sales data publicly, so pre-trained models for certain industries, store sizes, etc. is not possible. 

2) Dataset is broadly across all stores. The Store number is the strongest feature, with CPI and Unemployment Rate behind it. With filtering for specific stores, the Month becomes the strongest feature, followed by the CPI. 

3) For large companies that have multiple locations, it may benefit to have localized models for specific locations, or those that fit certain criteria/metrics.

4) Due to the nature of initial develpment/concept implementation, a manually created dataset for testing was needed, due to the lack of publicly available data.

# Future Improvements/Plans

1) Models can be individually developed for specific stores, or stores can be categorized based on metrics by the algorithm, allowing for more specific predictions

2) While the model handles seasonality in most situations, outside influences may affect weekly sales data. This can be handled with manually inserted flags whenever these events occur.

3) Data could be further drilled down into specific departments as a subset model to create added value

4) Future implementations would seek to add autonomous finance aspects of indepedntly pulling the data and running the models without manual input/interaction.
