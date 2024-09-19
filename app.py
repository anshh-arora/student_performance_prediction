import os
import logging
from logging.handlers import RotatingFileHandler
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import pickle
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
handler = RotatingFileHandler('app.log', maxBytes=10000, backupCount=1)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

app = Flask(__name__)
CORS(app)
app.logger.addHandler(handler)

# Load the trained model and scaler
try:
    model = load_model('final_marks_predictor_model.h5')
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    app.logger.info("Model and scaler loaded successfully")
except Exception as e:
    app.logger.error(f"Error loading model or scaler: {e}")

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
        app.logger.info(f"Prediction successful: {predicted_marks[0][0]}")
        return predicted_marks[0][0]
    except Exception as e:
        app.logger.error(f"Error during prediction: {e}")
        return None

@app.route('/')
def index():
    app.logger.info("Index page requested")
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        app.logger.info(f"Received data: {request.json}")
        data = request.json
        name = data['name']
        age = int(data['age'])
        year1_marks = float(data['year1_marks'])
        year2_marks = float(data['year2_marks'])
        studytime = float(data['study_time'])
        failures = int(data['failures'])

        prediction = predict_new_input(model, scaler, age, year1_marks, year2_marks, studytime, failures)

        if prediction is None:
            app.logger.error("Prediction returned None.")
            return jsonify({'error': 'Prediction failed due to internal error.'}), 500
        
        rounded_prediction = round(float(prediction), 2)
        app.logger.info(f"Prediction result: {rounded_prediction}")
        return jsonify({'prediction': rounded_prediction})
    
    except KeyError as ke:
        app.logger.error(f"KeyError: {ke}")
        return jsonify({'error': f'Missing required field: {ke}'}), 400
    except ValueError as ve:
        app.logger.error(f"ValueError: {ve}")
        return jsonify({'error': 'Invalid input. Please ensure all fields contain correct values.'}), 400
    except Exception as e:
        app.logger.error(f"Error during form submission: {e}", exc_info=True)
        return jsonify({'error': 'Internal server error.'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)



    