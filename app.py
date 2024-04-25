from flask import Flask, request, jsonify, abort
import numpy as np
import pickle
import os

# Initialize Flask application
app = Flask(__name__)  # Use double underscores around name

# Load your trained model and encoder
model = pickle.load(open('lr.pkl', 'rb'))
encoder = pickle.load(open('encoder.sav', 'rb'))

# API endpoint for predictions
@app.route('/predict', methods=['POST'])
def get_detections():
    try:
        data = request.json
        if not data:
            abort(400, description="No data provided")

        # Log incoming data
        print(f"Data received: {data}")

        # Extract features from request
        gender = data.get('gender')
        age = data.get('age')
        hypertension = data.get('hypertension')
        heart_disease = data.get('heart_disease')
        smoking_history = data.get('smoking_history')
        bmi = data.get('bmi')
        HbA1c_level = data.get('HbA1c_level')
        blood_glucose_level = data.get('blood_glucose_level')

        # Encode categorical data
        encoded_smoking_history = encoder.transform([smoking_history])[0]
        encoded_gender = {'Male': 1, 'Female': 0}.get(gender, 2)  # Default to 2 for other genders

        # Prepare the model input
        model_input = np.array([[encoded_gender, age, hypertension, heart_disease, encoded_smoking_history, bmi, HbA1c_level, blood_glucose_level]])

        # Make prediction
        prediction = model.predict(model_input)

        # Return the prediction
        return jsonify({"prediction": int(prediction[0])})

    except Exception as e:
        # Log and handle any other errors
        print(f"An error occurred: {e}")
        abort(500, description=str(e))

# Check if the script is run directly and not imported
if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
