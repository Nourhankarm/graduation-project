from absl import app, logging
import numpy as np
import pickle

from flask import Flask, request, jsonify, abort

# Initialize Flask application
app = Flask(__name__)

# Try to load the model and encoder safely
try:
    model = pickle.load(open('decision_tree_model.pkl', 'rb'))
    encoder = pickle.load(open('encoder.sav', 'rb'))
except FileNotFoundError as e:
    logging.error(f"Error loading model or encoder: {e}")
    # Optionally, raise an exception here to stop the app from running if the files are critical
    raise e

# API endpoint for predictions
@app.route('/predict', methods=['POST'])
def get_detections():
    try:
        data = request.json
        # Data processing and prediction logic here

        # Return prediction
        return jsonify({"prediction": int(prediction[0])})

    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        abort(500, description="Error during prediction")

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)

