from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Load your trained model (make sure the path is accessible from your Jupyter Notebook)
#model = pickle.load(open('decision_tree_model.sav', 'rb'))
# Load the model from a pickle file
model_file_path = 'decision_tree_model.sav'
with open(model_file_path, 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return "Welcome to the Diabetes Prediction API"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    print(data["gender"])
    features = [data['gender'], data['age'], data['hypertension'], data['heart_disease'],
                data['smoking_history'], data['bmi'], data['HbA1c_level'], data['blood_glucose_level']]
    prediction = model.predict([features])
    return jsonify({'diabetes_prediction': int(prediction[0])})


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
