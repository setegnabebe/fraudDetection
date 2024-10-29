# Import necessary libraries
import shap
import lime
from lime.lime_tabular import LimeTabularExplainer
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load the prepared data
fraud_data = pd.read_csv("./data/Fraud_Data.csv")

# Data preparation function
def prepare_data(data, target_column):
    # Convert date column if it exists
    if 'TransactionDate' in data.columns:
        data['TransactionDate'] = pd.to_datetime(data['TransactionDate'])
        data['year'] = data['TransactionDate'].dt.year
        data['month'] = data['TransactionDate'].dt.month
        data['day'] = data['TransactionDate'].dt.day
        data['hour'] = data['TransactionDate'].dt.hour
        data['day_of_week'] = data['TransactionDate'].dt.dayofweek
        data.drop(columns=['TransactionDate'], inplace=True)

    # Separate features and target
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Encode categorical columns if necessary
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = LabelEncoder().fit_transform(X[col])

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Keep a copy of the unscaled test data for explainability purposes
    X_test_unscaled = X_test.copy()

    # Standardize numeric columns
    scaler = StandardScaler()
    X_train[X_train.select_dtypes(include=[np.number]).columns] = scaler.fit_transform(X_train.select_dtypes(include=[np.number]))
    X_test[X_test.select_dtypes(include=[np.number]).columns] = scaler.transform(X_test.select_dtypes(include=[np.number]))

    return X_train, X_test, X_test_unscaled, y_train, y_test, X.columns

# Prepare data
X_train, X_test, X_test_unscaled, y_train, y_test, feature_names = prepare_data(fraud_data, 'class')

# Train a sample model (Random Forest)
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# SHAP Explainability
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_test_unscaled)

# Verify if shap_values is a list (binary classification)
if isinstance(shap_values, list):
    shap_values = shap_values[1]  # Use class 1 (e.g., "Fraud") SHAP values for explanation

# SHAP Summary Plot (global explanation)
plt.title("SHAP Summary Plot")
shap.summary_plot(shap_values, X_test_unscaled, feature_names=feature_names)
plt.show()

# SHAP Force Plot for a single instance
idx = 0  
plt.title("SHAP Force Plot for a Single Prediction")
shap.force_plot(explainer.expected_value[1], shap_values[idx], X_test_unscaled.iloc[idx], feature_names=feature_names, matplotlib=True)
plt.show()

# SHAP Dependence Plot
plt.title("SHAP Dependence Plot")
shap.dependence_plot(0, shap_values, X_test_unscaled, feature_names=feature_names)
plt.show()

# Initialize LIME explainer
lime_explainer = LimeTabularExplainer(X_train, mode="classification", 
                                      feature_names=feature_names, 
                                      class_names=['Not Fraud', 'Fraud'], 
                                      discretize_continuous=True)

# Explain a single instance in the test set
idx = 1  # Select an instance from the test set
exp = lime_explainer.explain_instance(X_test[idx], rf_model.predict_proba, num_features=5)

# Display LIME explanation results
exp.show_in_notebook(show_table=True)
exp.as_pyplot_figure()
plt.title("LIME Feature Importance Plot for a Single Prediction")
plt.show()
