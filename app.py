from absl import app, logging
import numpy as np
import pickle
from flask import Flask, request, jsonify, abort
from sklearn.preprocessing import LabelEncoder

# Initialize Flask application
app = Flask(__name__)

# Load encoder
with open('encoder.sav', 'rb') as file:
    encoder = pickle.load(file)

# Load model
with open('diabetes_model.sav', 'rb') as file:
    model = pickle.load(file)

# API endpoint for predictions
@app.route('/predict', methods=['POST'])
def get_detections():
    try:
        # Validate incoming JSON data
        data = request.json
        if not all(key in data for key in ['gender', 'age', 'hypertension', 'heart_disease', 'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level']):
            abort(400)

        # Encoding categorical variables
        gender_encoded = encoder.transform([data['gender']])[0]
        smoking_history_encoded = encoder.transform([data['smoking_history']])[0]

        # Prepare model input
        model_input = np.array([[gender_encoded, data['age'], data['hypertension'], data['heart_disease'], smoking_history_encoded, data['bmi'], data['HbA1c_level'], data['blood_glucose_level']]])

        # Make prediction
        prediction = model.predict(model_input)

        # Return prediction
        return jsonify({"prediction": int(prediction[0])})

    except Exception as e:
        app.logger.error(f"An error occurred: {str(e)}")
        abort(500)

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
