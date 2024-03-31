#!/usr/bin/env python
# coding: utf-8

# In[3]:


get_ipython().system('pip install flask --upgrade')


# In[ ]:


from absl import app, logging
import numpy as np
import pickle

from flask import Flask, request, jsonify, abort
import os

# Initialize Flask application
app = Flask(__name__)

# Load the model from a pickle file
model_file_path = 'diabetes_model.sav'
with open(model_file_path, 'rb') as file:
    model = pickle.load(file)

# Correct the syntax error in the path and load the encoder
encoder_file_path = 'encoder.sav'  # Corrected path
with open(encoder_file_path, 'rb') as file:
    encoder = pickle.load(file)

# API endpoint for predictions
@app.route('/predict', methods=['POST'])
def get_detections():
    try:
        data = request.json
        print(data)
        gender = data['gender']
        age = data['age']
        hypertension = data['hypertension']
        smoking_history = data['smoking_history']
        bni = data['bni']
        HbA1c_level = data['HbA1c_level']
        blood_glucose_level = data['blood_glucose_level'] 
        print(f"data received: gender= {data['gender']}, age= {age}, hypertension= {hypertension}, smoking_history= {data['smoking_history']}, bni= {bni}, HbA1c_level= {HbA1c_level}, blood_glucose_level= {blood_glucose_level}")
        #  'gender' and 'smoking_history' are categorical and need to be encoded
        print(encoder.classes_)
        encoded_smoking_history = encoder.transform([smoking_history])[0]
        print(encoded_smoking_history)
        encoded_gender = encoder.transform([gender])[0]
        
        

        # Log received data
        print(f"data received: gender= {data['gender']}, age= {age}, hypertension= {hypertension}, smoking_history= {data['smoking_history']}, bni= {bni}, HbA1c_level= {HbA1c_level}, blood_glucose_level= {blood_glucose_level}")

        # Prepare the model input by replacing 'gender' and 'smoking_history' with their encoded forms
        model_input = np.array([[encoded_gender, age, hypertension, encoded_smoking_history, bni, HbA1c_level, blood_glucose_level]])

        # Make prediction
        prediction = model.predict(model_input)

        # Log and return the prediction
        print(f"prediction: {prediction}")
        return jsonify({"prediction": int(prediction[0])})

    except FileNotFoundError:
        abort(404)


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)



# In[ ]:



