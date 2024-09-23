import os
import warnings
import logging
from logging.handlers import RotatingFileHandler
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import pickle
from dotenv import load_dotenv

# Suppress TensorFlow and other warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

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
        logger.info("Starting prediction...")
        logger.debug(f"Input values - Age: {age}, Year1 Marks: {year1_marks}, Year2 Marks: {year2_marks}, Study Time: {studytime}, Failures: {failures}")

        feature_names = ['age', 'year1_marks', 'year2_marks', 'studytime', 'failures']
        new_input_df = pd.DataFrame([[age, year1_marks, year2_marks, studytime, failures]], columns=feature_names)

        if model is None or scaler is None:
            logger.error("Model or scaler is not loaded.")
            return None
        
        new_input_scaled = scaler.transform(new_input_df)
        logger.debug(f"Scaled input: {new_input_scaled}")

        predicted_marks = model.predict(new_input_scaled, verbose=0)
        logger.info(f"Prediction successful: {predicted_marks[0][0]}")
        return predicted_marks[0][0]
    except Exception as e:
        logger.error(f"Error during prediction: {e}", exc_info=True)
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

        age = int(data.get('age', 0))
        year1_marks = float(data.get('year1_marks', 0))
        year2_marks = float(data.get('year2_marks', 0))
        studytime = float(data.get('studytime', 0))
        failures = int(data.get('failures', 0))

        prediction = predict_new_input(age, year1_marks, year2_marks, studytime, failures)

        if prediction is None:
            logger.error("Prediction returned None.")
            return jsonify({'error': 'Prediction failed due to internal error.'}), 500
        
        rounded_prediction = round(float(prediction), 2)
        logger.info(f"Prediction result: {rounded_prediction}")
        return jsonify({'prediction': rounded_prediction})

    except Exception as e:
        logger.error(f"Error during prediction: {e}", exc_info=True)
        return jsonify({'error': 'Internal server error.'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"Flask app starting on port {port}")
    app.run(host='0.0.0.0', port=port, debug=True)