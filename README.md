![alt text](https://github.com/PAndrewTurner/FinPulse/blob/main/FinPulse%20Logo.jpg?raw=true)

# FinPulse

FinPulse is an autonomous finance platform that uses machine learning and statistical programming to enhance the way business owners manage their finances. It offers bespoke machine learning-trained models to predict and forecast sales and income to help plan their business operations. 

## Model Creation

main.py: A graphical user interface (GUI) application built using Tkinter that allows users to load a dataset, load a pre-trained machine learning model, run predictions on the dataset, and visualize the results.

model.py: A script for training and saving machine learning models using the Walmart store sales dataset. The script uses Random Forest, XGBoost, and Gradient Boosting regressors to predict weekly sales.

The Dataset was split using the <b>train_test_split</b> function within the <b>sklearn.model_selection</b> library and then the RandomForest, XGB, and Gradient Booster models are created. The target variable was selected as the weekly sales, and all others except the full date string were chosen as the features. The full date was parsed into the seperate components for day, month, and year. 

After the models are fit and tested, they are exported and saved so other computers may open the models as needed to run the models.

The ML models were created, trained, and fit in the model.py file, while a GUI was created to allow for interativity with the code, housed in the main.py file.

Due to the model being pre-trained, selecting the test_data.csv file selects the dataset to make predictions for the future. Both the dataset and the model must be selected. Once run, a grapgh can then be populated in the window, showing the predictions.

## Additional Notes
The trained models will be saved as rf_model.joblib, xgb_model.joblib, and gb_model.joblib.
The script will print the Root Mean Squared Error (RMSE) for each model on the test set.

# How to Use main.py GUI
Load Dataset: Click the "Load Dataset" button to select a CSV file containing the financial dataset.
Load Model: Click the "Load Model" button to select a pre-trained machine learning model (joblib or pkl format).
Run Model: Click the "Run Model" button to make predictions on the loaded dataset using the selected model.
Display Graph: Click the "Display Graph" button to visualize the predictions on a line plot.

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


# Current Model Limitations

### Data Quality and Features:
The performance of the models heavily depends on the quality and relevance of the features in the dataset. If important features are missing or if the data is noisy, it can adversely impact the model's accuracy.

### Assumption of Stationarity:
The models assume that the underlying patterns in the data remain relatively stable over time. If the financial data exhibits non-stationary behavior, the model may struggle to make accurate predictions.

### Sensitivity to Hyperparameters:
The performance of the models can be sensitive to the choice of hyperparameters. The current hyperparameters used in the scripts might not be optimal for all datasets, and fine-tuning may be required for better results.

### Limited Evaluation Metrics:
The models are evaluated based on the Root Mean Squared Error (RMSE) on the test set. While RMSE provides a measure of prediction accuracy, it might not capture all aspects of model performance, especially in scenarios with imbalanced data or outliers.

### Assumption of Linearity:
Random Forest, XGBoost, and Gradient Boosting models assume a certain level of linearity in the relationships between features and the target variable. If the underlying relationships are highly non-linear, these models may not capture them accurately.

# Future Improvements/Plans


### Feature Engineering
Invest time in thorough feature engineering to identify and create relevant features that better capture the underlying patterns in the financial data.

### Data Preprocessing
Implement robust data preprocessing techniques to handle missing values, outliers, and scale features appropriately. This can enhance the models' ability to generalize to new data.

### Time Series Analysis
If the financial data exhibits time-dependent patterns, consider applying advanced time series analysis techniques or using specialized time series models for more accurate predictions.

### Hyperparameter Tuning
Conduct a comprehensive hyperparameter tuning process using techniques like grid search or randomized search to find the optimal hyperparameters for the models.

### Ensemble Methods
Explore ensemble methods by combining predictions from multiple models. This can help mitigate the weaknesses of individual models and improve overall predictive performance.

### Model Explainability
Enhance model interpretability by using techniques such as SHAP (SHapley Additive exPlanations) values or LIME (Local Interpretable Model-agnostic Explanations) to understand and explain the model's predictions.

### Regularization Techniques
Experiment with regularization techniques to prevent overfitting, especially if the dataset is limited in size.

### Continuous Monitoring and Updating
Implement a system for continuous monitoring of model performance and update the models as new data becomes available. This ensures that the models stay relevant over time.

### User Feedback Integration
If applicable, incorporate user feedback into the model training process to adapt the models to changing business conditions or user requirements.

### Cross-Validation
Implement robust cross-validation strategies to get a better estimate of the model's generalization performance and reduce the risk of overfitting to the training data.
