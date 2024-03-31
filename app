{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "09ee2065",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Requirement already satisfied: flask in c:\\users\\dell\\anaconda3\\lib\\site-packages (2.2.2)\n",
      "Collecting flask\n",
      "  Obtaining dependency information for flask from https://files.pythonhosted.org/packages/93/a6/aa98bfe0eb9b8b15d36cdfd03c8ca86a03968a87f27ce224fb4f766acb23/flask-3.0.2-py3-none-any.whl.metadata\n",
      "  Downloading flask-3.0.2-py3-none-any.whl.metadata (3.6 kB)\n",
      "Collecting Werkzeug>=3.0.0 (from flask)\n",
      "  Obtaining dependency information for Werkzeug>=3.0.0 from https://files.pythonhosted.org/packages/c3/fc/254c3e9b5feb89ff5b9076a23218dafbc99c96ac5941e900b71206e6313b/werkzeug-3.0.1-py3-none-any.whl.metadata\n",
      "  Downloading werkzeug-3.0.1-py3-none-any.whl.metadata (4.1 kB)\n",
      "Requirement already satisfied: Jinja2>=3.1.2 in c:\\users\\dell\\anaconda3\\lib\\site-packages (from flask) (3.1.2)\n",
      "Collecting itsdangerous>=2.1.2 (from flask)\n",
      "  Obtaining dependency information for itsdangerous>=2.1.2 from https://files.pythonhosted.org/packages/68/5f/447e04e828f47465eeab35b5d408b7ebaaaee207f48b7136c5a7267a30ae/itsdangerous-2.1.2-py3-none-any.whl.metadata\n",
      "  Downloading itsdangerous-2.1.2-py3-none-any.whl.metadata (2.9 kB)\n",
      "Collecting click>=8.1.3 (from flask)\n",
      "  Obtaining dependency information for click>=8.1.3 from https://files.pythonhosted.org/packages/00/2e/d53fa4befbf2cfa713304affc7ca780ce4fc1fd8710527771b58311a3229/click-8.1.7-py3-none-any.whl.metadata\n",
      "  Downloading click-8.1.7-py3-none-any.whl.metadata (3.0 kB)\n",
      "Collecting blinker>=1.6.2 (from flask)\n",
      "  Obtaining dependency information for blinker>=1.6.2 from https://files.pythonhosted.org/packages/fa/2a/7f3714cbc6356a0efec525ce7a0613d581072ed6eb53eb7b9754f33db807/blinker-1.7.0-py3-none-any.whl.metadata\n",
      "  Downloading blinker-1.7.0-py3-none-any.whl.metadata (1.9 kB)\n",
      "Requirement already satisfied: colorama in c:\\users\\dell\\anaconda3\\lib\\site-packages (from click>=8.1.3->flask) (0.4.6)\n",
      "Requirement already satisfied: MarkupSafe>=2.0 in c:\\users\\dell\\anaconda3\\lib\\site-packages (from Jinja2>=3.1.2->flask) (2.1.1)\n",
      "Downloading flask-3.0.2-py3-none-any.whl (101 kB)\n",
      "   ---------------------------------------- 0.0/101.3 kB ? eta -:--:--\n",
      "   ----------- --------------------------- 30.7/101.3 kB 660.6 kB/s eta 0:00:01\n",
      "   --------------------------- ----------- 71.7/101.3 kB 787.7 kB/s eta 0:00:01\n",
      "   ----------------------------------- --- 92.2/101.3 kB 871.5 kB/s eta 0:00:01\n",
      "   -------------------------------------- 101.3/101.3 kB 583.7 kB/s eta 0:00:00\n",
      "Downloading blinker-1.7.0-py3-none-any.whl (13 kB)\n",
      "Downloading click-8.1.7-py3-none-any.whl (97 kB)\n",
      "   ---------------------------------------- 0.0/97.9 kB ? eta -:--:--\n",
      "   ------------------------- -------------- 61.4/97.9 kB 3.4 MB/s eta 0:00:01\n",
      "   ------------------------------------- -- 92.2/97.9 kB 1.3 MB/s eta 0:00:01\n",
      "   ------------------------------------- -- 92.2/97.9 kB 1.3 MB/s eta 0:00:01\n",
      "   ---------------------------------------- 97.9/97.9 kB 626.4 kB/s eta 0:00:00\n",
      "Downloading itsdangerous-2.1.2-py3-none-any.whl (15 kB)\n",
      "Downloading werkzeug-3.0.1-py3-none-any.whl (226 kB)\n",
      "   ---------------------------------------- 0.0/226.7 kB ? eta -:--:--\n",
      "   --------------------- ------------------ 122.9/226.7 kB 2.4 MB/s eta 0:00:01\n",
      "   ---------------------------------------  225.3/226.7 kB 2.3 MB/s eta 0:00:01\n",
      "   ---------------------------------------  225.3/226.7 kB 2.3 MB/s eta 0:00:01\n",
      "   ---------------------------------------- 226.7/226.7 kB 1.4 MB/s eta 0:00:00\n",
      "Installing collected packages: Werkzeug, itsdangerous, click, blinker, flask\n",
      "  Attempting uninstall: Werkzeug\n",
      "    Found existing installation: Werkzeug 2.2.3\n",
      "    Uninstalling Werkzeug-2.2.3:\n",
      "      Successfully uninstalled Werkzeug-2.2.3\n",
      "  Attempting uninstall: itsdangerous\n",
      "    Found existing installation: itsdangerous 2.0.1\n",
      "    Uninstalling itsdangerous-2.0.1:\n",
      "      Successfully uninstalled itsdangerous-2.0.1\n",
      "  Attempting uninstall: click\n",
      "    Found existing installation: click 8.0.4\n",
      "    Uninstalling click-8.0.4:\n",
      "      Successfully uninstalled click-8.0.4\n",
      "  Attempting uninstall: flask\n",
      "    Found existing installation: Flask 2.2.2\n",
      "    Uninstalling Flask-2.2.2:\n",
      "      Successfully uninstalled Flask-2.2.2\n",
      "Successfully installed Werkzeug-3.0.1 blinker-1.7.0 click-8.1.7 flask-3.0.2 itsdangerous-2.1.2\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.\n",
      "python-lsp-black 1.2.1 requires black>=22.3.0, but you have black 0.0 which is incompatible.\n"
     ]
    }
   ],
   "source": [
    "!pip install flask --upgrade"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f13cac2a",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      " * Serving Flask app '__main__'\n",
      " * Debug mode: off\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\DELL\\anaconda3\\Lib\\site-packages\\sklearn\\base.py:347: InconsistentVersionWarning: Trying to unpickle estimator SVC from version 1.0.2 when using version 1.3.0. This might lead to breaking code or invalid results. Use at your own risk. For more info please refer to:\n",
      "https://scikit-learn.org/stable/model_persistence.html#security-maintainability-limitations\n",
      "  warnings.warn(\n",
      "C:\\Users\\DELL\\anaconda3\\Lib\\site-packages\\sklearn\\base.py:347: InconsistentVersionWarning: Trying to unpickle estimator LabelEncoder from version 1.2.2 when using version 1.3.0. This might lead to breaking code or invalid results. Use at your own risk. For more info please refer to:\n",
      "https://scikit-learn.org/stable/model_persistence.html#security-maintainability-limitations\n",
      "  warnings.warn(\n",
      "WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.\n",
      " * Running on all addresses (0.0.0.0)\n",
      " * Running on http://127.0.0.1:5000\n",
      " * Running on http://192.168.1.11:5000\n",
      "Press CTRL+C to quit\n",
      "127.0.0.1 - - [31/Mar/2024 04:12:13] \"POST /predict HTTP/1.1\" 400 -\n",
      "127.0.0.1 - - [31/Mar/2024 04:14:01] \"POST /predict HTTP/1.1\" 400 -\n",
      "[2024-03-31 04:15:03,963] ERROR in app: Exception on /predict [POST]\n",
      "Traceback (most recent call last):\n",
      "  File \"C:\\Users\\DELL\\anaconda3\\Lib\\site-packages\\sklearn\\utils\\_encode.py\", line 225, in _encode\n",
      "    return _map_to_integer(values, uniques)\n",
      "           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n",
      "  File \"C:\\Users\\DELL\\anaconda3\\Lib\\site-packages\\sklearn\\utils\\_encode.py\", line 165, in _map_to_integer\n",
      "    return np.array([table[v] for v in values])\n",
      "                    ^^^^^^^^^^^^^^^^^^^^^^^^^^\n",
      "  File \"C:\\Users\\DELL\\anaconda3\\Lib\\site-packages\\sklearn\\utils\\_encode.py\", line 165, in <listcomp>\n",
      "    return np.array([table[v] for v in values])\n",
      "                     ~~~~~^^^\n",
      "  File \"C:\\Users\\DELL\\anaconda3\\Lib\\site-packages\\sklearn\\utils\\_encode.py\", line 159, in __missing__\n",
      "    raise KeyError(key)\n",
      "KeyError: 'Male'\n",
      "\n",
      "During handling of the above exception, another exception occurred:\n",
      "\n",
      "Traceback (most recent call last):\n",
      "  File \"C:\\Users\\DELL\\anaconda3\\Lib\\site-packages\\flask\\app.py\", line 1463, in wsgi_app\n",
      "    response = self.full_dispatch_request()\n",
      "               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n",
      "  File \"C:\\Users\\DELL\\anaconda3\\Lib\\site-packages\\flask\\app.py\", line 872, in full_dispatch_request\n",
      "    rv = self.handle_user_exception(e)\n",
      "         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n",
      "  File \"C:\\Users\\DELL\\anaconda3\\Lib\\site-packages\\flask\\app.py\", line 870, in full_dispatch_request\n",
      "    rv = self.dispatch_request()\n",
      "         ^^^^^^^^^^^^^^^^^^^^^^^\n",
      "  File \"C:\\Users\\DELL\\anaconda3\\Lib\\site-packages\\flask\\app.py\", line 855, in dispatch_request\n",
      "    return self.ensure_sync(self.view_functions[rule.endpoint])(**view_args)  # type: ignore[no-any-return]\n",
      "           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n",
      "  File \"C:\\Users\\DELL\\AppData\\Local\\Temp\\ipykernel_9224\\3653755914.py\", line 39, in get_detections\n",
      "    encoded_gender = encoder.transform([gender])[0]\n",
      "                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^\n",
      "  File \"C:\\Users\\DELL\\anaconda3\\Lib\\site-packages\\sklearn\\utils\\_set_output.py\", line 140, in wrapped\n",
      "    data_to_wrap = f(self, X, *args, **kwargs)\n",
      "                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^\n",
      "  File \"C:\\Users\\DELL\\anaconda3\\Lib\\site-packages\\sklearn\\preprocessing\\_label.py\", line 137, in transform\n",
      "    return _encode(y, uniques=self.classes_)\n",
      "           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n",
      "  File \"C:\\Users\\DELL\\anaconda3\\Lib\\site-packages\\sklearn\\utils\\_encode.py\", line 227, in _encode\n",
      "    raise ValueError(f\"y contains previously unseen labels: {str(e)}\")\n",
      "ValueError: y contains previously unseen labels: 'Male'\n",
      "127.0.0.1 - - [31/Mar/2024 04:15:03] \"POST /predict HTTP/1.1\" 500 -\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "{'gender': 'Male', 'age': 45, 'hypertension': 1, 'smoking_history': 'never', 'bni': 26.5, 'HbA1c_level': 6.2, 'blood_glucose_level': 150}\n",
      "data received: gender= Male, age= 45, hypertension= 1, smoking_history= never, bni= 26.5, HbA1c_level= 6.2, blood_glucose_level= 150\n",
      "['No Info' 'current' 'ever' 'former' 'never' 'not current']\n",
      "4\n"
     ]
    }
   ],
   "source": [
    "from absl import app, logging\n",
    "import numpy as np\n",
    "import pickle\n",
    "\n",
    "from flask import Flask, request, jsonify, abort\n",
    "import os\n",
    "\n",
    "# Initialize Flask application\n",
    "app = Flask(__name__)\n",
    "\n",
    "# Load the model from a pickle file\n",
    "model_file_path = 'diabetes_model.sav'\n",
    "with open(model_file_path, 'rb') as file:\n",
    "    model = pickle.load(file)\n",
    "\n",
    "# Correct the syntax error in the path and load the encoder\n",
    "encoder_file_path = 'encoder.sav'  # Corrected path\n",
    "with open(encoder_file_path, 'rb') as file:\n",
    "    encoder = pickle.load(file)\n",
    "\n",
    "# API endpoint for predictions\n",
    "@app.route('/predict', methods=['POST'])\n",
    "def get_detections():\n",
    "    try:\n",
    "        data = request.json\n",
    "        print(data)\n",
    "        gender = data['gender']\n",
    "        age = data['age']\n",
    "        hypertension = data['hypertension']\n",
    "        smoking_history = data['smoking_history']\n",
    "        bni = data['bni']\n",
    "        HbA1c_level = data['HbA1c_level']\n",
    "        blood_glucose_level = data['blood_glucose_level'] \n",
    "        print(f\"data received: gender= {data['gender']}, age= {age}, hypertension= {hypertension}, smoking_history= {data['smoking_history']}, bni= {bni}, HbA1c_level= {HbA1c_level}, blood_glucose_level= {blood_glucose_level}\")\n",
    "        #  'gender' and 'smoking_history' are categorical and need to be encoded\n",
    "        print(encoder.classes_)\n",
    "        encoded_smoking_history = encoder.transform([smoking_history])[0]\n",
    "        print(encoded_smoking_history)\n",
    "        encoded_gender = encoder.transform([gender])[0]\n",
    "        \n",
    "        \n",
    "\n",
    "        # Log received data\n",
    "        print(f\"data received: gender= {data['gender']}, age= {age}, hypertension= {hypertension}, smoking_history= {data['smoking_history']}, bni= {bni}, HbA1c_level= {HbA1c_level}, blood_glucose_level= {blood_glucose_level}\")\n",
    "\n",
    "        # Prepare the model input by replacing 'gender' and 'smoking_history' with their encoded forms\n",
    "        model_input = np.array([[encoded_gender, age, hypertension, encoded_smoking_history, bni, HbA1c_level, blood_glucose_level]])\n",
    "\n",
    "        # Make prediction\n",
    "        prediction = model.predict(model_input)\n",
    "\n",
    "        # Log and return the prediction\n",
    "        print(f\"prediction: {prediction}\")\n",
    "        return jsonify({\"prediction\": int(prediction[0])})\n",
    "\n",
    "    except FileNotFoundError:\n",
    "        abort(404)\n",
    "\n",
    "\n",
    "if __name__ == '__main__':\n",
    "    app.run(debug=False, host='0.0.0.0', port=5000)\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "517475f9",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}