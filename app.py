from absl import app, logging
import numpy as np
import pickle

from flask import Flask, request, jsonify, abort
import os

# Initialize Flask application
app = Flask(__name__)

# It's better to load models in a function or under if __name__ == '__main__' to avoid issues with unneeded executions during imports


@app.before_first_request
def load_model_data():
    app.logger.info(f"Current directory: {os.getcwd()}")
    app.logger.info(f"Directory contents: {os.listdir('.')}")
    model_path = 'decision_tree_model.pkl'
    encoder_path = 'encoder.sav'
    try:
        global model, encoder
        model = pickle.load(open(model_path, 'rb'))
        encoder = pickle.load(open(encoder_path, 'rb'))
        app.logger.info("Model and encoder loaded successfully.")
    except FileNotFoundError as e:
        app.logger.error(f"File not found. Model path: {model_path}, Encoder path: {encoder_path}")
        abort(500, description="Model loading failed: File not found")
    except Exception as e:
        app.logger.error(f"Failed to load model or encoder: {e}")
        abort(500, description="Model loading failed")

# API endpoint for predictions
@app.route('/predict', methods=['POST'])
def get_detections():
    try:
        data = request.json
        if not data:
            abort(400, description="No data provided")

        required_fields = ['gender', 'age', 'hypertension', 'heart_disease', 'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level']
        if not all(field in data for field in required_fields):
            abort(400, description="Missing data fields")

        # Encoding categorical data
        try:
            encoded_smoking_history = encoder.transform([data['smoking_history']])[0]
            encoded_gender = {'Male': 1, 'Female': 0}.get(data['gender'], 2)
        except ValueError as e:
            app.logger.error(f"Encoding failed: {e}")
            abort(400, description="Invalid data for encoding")

        # Prepare the model input
        model_input = np.array([[encoded_gender, data['age'], data['hypertension'], data['heart_disease'], encoded_smoking_history, data['bmi'], data['HbA1c_level'], data['blood_glucose_level']]])

        # Make prediction
        prediction = model.predict(model_input)
        return jsonify({"prediction": int(prediction[0])})

    except Exception as e:
        app.logger.error(f"Error during prediction: {e}")
        abort(500, description="Prediction failed")

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
