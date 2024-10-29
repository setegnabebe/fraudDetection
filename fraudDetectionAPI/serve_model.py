from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import mlflow
import os

app = Flask(__name__)

# Load the model paths
model_names = {
    "Logistic Regression": "models/Logistic Regression_model.pkl",
    "Decision Tree": "models/Decision Tree_model.pkl",
    "Random Forest": "models/Random Forest_model.pkl",
    "Gradient Boosting": "models/Gradient Boosting_model.pkl",
    "MLP": "models/MLP_model.pkl",
    "CNN": "models/CNN_model.keras"  # Ensure this path is correct
}

# Load the scaler
try:
    scaler = joblib.load('models/scaler.pkl')
except FileNotFoundError:
    raise Exception("Scaler file not found. Please check the path.")

def preprocess_input(input_data):
    # Preprocess the input data for the model
    # Assuming input_data is a dictionary, convert it to a DataFrame
    input_df = pd.DataFrame([input_data])

    # Standardize the numeric features
    input_scaled = scaler.transform(input_df.select_dtypes(include=[np.number]))
    return input_scaled

@app.route('/predict/<model_name>', methods=['POST'])
def predict(model_name):
    # Check if the model name is valid
    if model_name not in model_names:
        return jsonify({"error": "Model not found"}), 404

    # Get the JSON data from the request
    input_data = request.get_json()

    # Preprocess the input data
    input_scaled = preprocess_input(input_data)

    # Load the model
    try:
        if model_name != "CNN":  # For non-Keras models
            model = joblib.load(model_names[model_name])
            prediction = model.predict(input_scaled)
        else:  # For Keras model
            import tensorflow as tf
            model = tf.keras.models.load_model(model_names[model_name])
            prediction = model.predict(input_scaled)
            prediction = (prediction > 0.5).astype(int).flatten()

        return jsonify({"prediction": int(prediction[0])})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
