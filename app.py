import logging
from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import pickle
from datetime import datetime
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.DEBUG)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppresses INFO and WARNING messages

app = Flask(__name__)

# Load the trained model and scaler
try:
    model = load_model('final_marks_predictor_model.h5')
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    logging.info("Model and scaler loaded successfully")
except Exception as e:
    logging.error(f"Error loading model or scaler: {e}")

def predict_new_input(model, scaler, age, year1_marks, year2_marks, studytime, failures):
    try:
        new_input = pd.DataFrame({
            'age': [age],
            'year1_marks': [year1_marks],
            'year2_marks': [year2_marks],
            'studytime': [studytime],
            'failures': [failures]
        })
        new_input_scaled = scaler.transform(new_input)
        predicted_marks = model.predict(new_input_scaled)
        logging.info(f"Prediction successful: {predicted_marks[0][0]}")
        return predicted_marks[0][0]
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        logging.info(f"Received data: {request.json}")

        data = request.json
        name = data['name']
        age = int(data['age'])
        year1_marks = float(data['year1_marks'])
        year2_marks = float(data['year2_marks'])
        studytime = float(data['study_time'])
        failures = int(data['failures'])

        prediction = predict_new_input(model, scaler, age, year1_marks, year2_marks, studytime, failures)

        if prediction is None:
            logging.error("Prediction returned None.")
            return jsonify({'error': 'Prediction failed due to internal error.'}), 500
        
        rounded_prediction = round(float(prediction), 2)
        logging.info(f"Prediction result: {rounded_prediction}")

        return jsonify({'prediction': rounded_prediction})
    
    except KeyError as ke:
        logging.error(f"KeyError: {ke}")
        return jsonify({'error': f'Missing required field: {ke}'}), 400
    except ValueError as ve:
        logging.error(f"ValueError: {ve}")
        return jsonify({'error': 'Invalid input. Please ensure all fields contain correct values.'}), 400
    except Exception as e:
        logging.error(f"Error during form submission: {e}", exc_info=True)
        return jsonify({'error': 'Internal server error.'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)