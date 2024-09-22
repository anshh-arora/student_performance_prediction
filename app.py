from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import logging
from logging.handlers import RotatingFileHandler
import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import pickle
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
handler = RotatingFileHandler('app.log', maxBytes=10000, backupCount=1)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

logger.info("Starting application...")

app = Flask(__name__)
CORS(app)

# Load the trained model and scaler
model = None
scaler = None

def load_model_and_scaler():
    global model, scaler
    try:
        model = load_model('final_marks_predictor_model.h5')
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        logger.info("Model and scaler loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model or scaler: {e}")

# Load model and scaler when the application starts
load_model_and_scaler()

def predict_new_input(age, year1_marks, year2_marks, studytime, failures):
    try:
        new_input = np.array([[age, year1_marks, year2_marks, studytime, failures]])
        new_input_scaled = scaler.transform(new_input)
        predicted_marks = model.predict(new_input_scaled, verbose=0)
        logger.info(f"Prediction successful: {predicted_marks[0][0]}")
        return predicted_marks[0][0]
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        logger.info(f"Received data: {request.json}")
        data = request.json
        if not data:
            return jsonify({'error': 'No JSON data received'}), 400

        name = data.get('name')
        age = int(data.get('age', 0))
        year1_marks = float(data.get('year1_marks', 0))
        year2_marks = float(data.get('year2_marks', 0))
        studytime = float(data.get('study_time', 0))
        failures = int(data.get('failures', 0))

        prediction = predict_new_input(age, year1_marks, year2_marks, studytime, failures)

        if prediction is None:
            logger.error("Prediction returned None.")
            return jsonify({'error': 'Prediction failed due to internal error.'}), 500
        
        rounded_prediction = round(float(prediction), 2)
        logger.info(f"Prediction result: {rounded_prediction}")
        return jsonify({'prediction': rounded_prediction})
    
    except KeyError as ke:
        logger.error(f"KeyError: {ke}")
        return jsonify({'error': f'Missing required field: {ke}'}), 400
    except ValueError as ve:
        logger.error(f"ValueError: {ve}")
        return jsonify({'error': 'Invalid input. Please ensure all fields contain correct values.'}), 400
    except Exception as e:
        logger.error(f"Error during form submission: {e}", exc_info=True)
        return jsonify({'error': 'Internal server error.'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    logger.info(f"Flask app starting on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)

logger.info("Application shutdown")