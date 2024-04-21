from flask import Flask, request, jsonify, abort
import pickle
import numpy as np

# Initialize Flask application
app = Flask(__name__)

# Load the model from a pickle file
model_file_path = 'diabetes_model.sav'
with open(model_file_path, 'rb') as file:
    model = pickle.load(file)


@app.route('/')
def home():
    return "Welcome to the Diabetes Prediction API"

# API endpoint for predictions
@app.route('/predict', methods=['POST'])
def get_detections():
    try:
        data = request.json
        print(data)

        # Extract features from the request
        features = [
            data['gender'],
            data['age'],
            data['hypertension'],
            data['heart_disease'],
            data['smoking_history'],
            data['bmi'],
            data['HbA1c_level'],
            data['blood_glucose_level']
        ]

        # Perform preprocessing on categorical features
        gender = data['gender']
        smoking_history = data['smoking_history']
        encoded_gender = 1 if gender == "Male" else 0 if gender == "Female" else 2
        
        encoded_smoking_history = 0
        if smoking_history == "never":
            encoded_smoking_history = 1
        elif smoking_history == "ever":
            encoded_smoking_history = 2
        elif smoking_history == "current":
            encoded_smoking_history = 3
        elif smoking_history == "not current":
            encoded_smoking_history = 4
       else:
            encoded_smoking_history = 5

        # Prepare model input
        model_input = np.array([[
            encoded_gender,
            *features[1:],
            encoded_smoking_history
        ]])

        # Make prediction
        prediction = model.predict(model_input)

        # Return prediction as JSON
        return jsonify({"diabetes_prediction": int(prediction[0])})

    except FileNotFoundError:
        abort(404)


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
