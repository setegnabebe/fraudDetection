from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
import os
import tensorflow as tf

app = Flask(__name__)

# Set base path for models, with a default of './models/' if MODEL_BASE_PATH is not provided
model_base_path = os.getenv("MODEL_BASE_PATH", "./models/")
print(f"Model base path set to: {model_base_path}")

# Load the model paths with updated paths
model_names = {
    "Logistic Regression": os.path.join(model_base_path, "Logistic Regression_model.pkl"),
    "Decision Tree": os.path.join(model_base_path, "Decision Tree_model.pkl"),
    "Random Forest": os.path.join(model_base_path, "Random Forest_model.pkl"),
    "Gradient Boosting": os.path.join(model_base_path, "Gradient Boosting_model.pkl"),
    "MLP": os.path.join(model_base_path, "MLP_model.pkl"),
    "CNN": os.path.join(model_base_path, "CNN_model.keras")
}

# Print model paths for verification
print(f"Model paths: {model_names}")

# Load the scaler with updated path
scaler_path = os.path.join(model_base_path, 'scaler.pkl')
try:
    scaler = joblib.load(scaler_path)
    print("Scaler loaded successfully.")
except (FileNotFoundError, KeyError) as e:
    print(f"Error loading scaler. Attempting a reload. Original error: {e}")
    try:
        import pickle
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        print("Scaler loaded successfully using pickle.")
    except Exception as e:
        raise Exception(f"Failed to load scaler after multiple attempts. Error: {e}")

def preprocess_input(input_data):
    # Validate input data
    if not isinstance(input_data, dict):
        raise ValueError("Input data must be a JSON object.")
    
    # Preprocess the input data for the model
    input_df = pd.DataFrame([input_data])
    
    # Standardize the numeric features
    try:
        input_scaled = scaler.transform(input_df.select_dtypes(include=[np.number]))
    except Exception as e:
        raise ValueError(f"Error in preprocessing input data: {e}")
    
    return input_scaled

@app.route('/predict/<model_name>', methods=['POST'])
def predict(model_name):
    # Check if the model name is valid
    if model_name not in model_names:
        return jsonify({"error": "Model not found"}), 404

    # Get the JSON data from the request
    input_data = request.get_json()

    # Preprocess the input data
    try:
        input_scaled = preprocess_input(input_data)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    # Load the model
    model_path = model_names[model_name]
    try:
        if model_name != "CNN":  # For non-Keras models
            model = joblib.load(model_path)
            prediction = model.predict(input_scaled)
        else:  # For Keras model
            import logging
            tf.get_logger().setLevel(logging.ERROR)
            model = tf.keras.models.load_model(model_path)
            prediction = model.predict(input_scaled)
            prediction = (prediction > 0.5).astype(int).flatten()

        return jsonify({"prediction": int(prediction[0])})
    except FileNotFoundError:
        return jsonify({"error": f"Model file not found at {model_path}"}), 404
    except Exception as e:
        return jsonify({"error": f"Error during model prediction: {str(e)}"}), 500

@app.route('/', methods=['GET'])
def index():
    return jsonify({
        "message": "Welcome to the Fraud Detection Model API",
        "available_models": list(model_names.keys()),
        "usage": "Send a POST request to /predict/<model_name> with JSON data"
    })

if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    app.run(host='0.0.0.0', port=5000)
