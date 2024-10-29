# Import necessary libraries
import os  # For directory management
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
import tensorflow as tf
import mlflow
import mlflow.sklearn
import mlflow.keras
import joblib  # For saving models and scalers

# Load Fraud Data
fraud_data = pd.read_csv("./data/Fraud_Data.csv")

# Data Preparation function
def prepare_data(data, target_column):
    # Handle date columns (example: 'TransactionDate')
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

    # Standardize numeric columns
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.select_dtypes(include=[np.number]))
    X_test = scaler.transform(X_test.select_dtypes(include=[np.number]))

    # Ensure the models directory exists
    os.makedirs('models', exist_ok=True)

    # Save the scaler for later use
    joblib.dump(scaler, 'models/scaler.pkl')
    print("Scaler saved to 'models/scaler.pkl'")

    return X_train, X_test, y_train, y_test

# Prepare data
X_train_fd, X_test_fd, y_train_fd, y_test_fd = prepare_data(fraud_data, 'class')

# Initialize MLflow experiment tracking
mlflow.set_experiment("Fraud Detection Models")

# Model training and logging function
def train_and_log_model(model, model_name, X_train, X_test, y_train, y_test):
    with mlflow.start_run(run_name=model_name):
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)
        print(f"{model_name} accuracy: {accuracy}")

        # Log model parameters, metrics, and model
        mlflow.log_param("model_name", model_name)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(model, model_name)

        # Save the trained model for deployment
        joblib.dump(model, f'models/{model_name}_model.pkl')
        print(f"{model_name} model saved to 'models/{model_name}_model.pkl'")

# Model Selection and Training
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "MLP": MLPClassifier(max_iter=500),
}

# Train and log each model
for model_name, model in models.items():
    train_and_log_model(model, model_name, X_train_fd, X_test_fd, y_train_fd, y_test_fd)

# Deep Learning Models (using TensorFlow/Keras)
def train_deep_learning_model(model, model_name, X_train, X_test, y_train, y_test):
    with mlflow.start_run(run_name=model_name):
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.2)

        loss, accuracy = model.evaluate(X_test, y_test)
        print(f"{model_name} accuracy: {accuracy}")

        # Log model parameters, metrics, and model
        mlflow.log_param("model_name", model_name)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.keras.log_model(model, model_name)

        # Save the Keras model for deployment with correct file extension
        model.save(f'models/{model_name}_model.keras')  # Changed to use the .keras extension
        print(f"{model_name} model saved to 'models/{model_name}_model.keras'")

# Define and train Keras models
input_shape = (X_train_fd.shape[1],)
cnn_model = tf.keras.Sequential([
    tf.keras.layers.Reshape((X_train_fd.shape[1], 1), input_shape=input_shape),
    tf.keras.layers.Conv1D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling1D(2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

train_deep_learning_model(cnn_model, "CNN", X_train_fd, X_test_fd, y_train_fd, y_test_fd)
