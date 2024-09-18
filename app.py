from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import pickle
from datetime import datetime
import os

app = Flask(__name__)

# Load the trained model and scaler
model = load_model('final_marks_predictor_model.h5')
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

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
        return predicted_marks[0][0]
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Log the incoming data
        print("Received data:", request.json)

        # Retrieve JSON data
        data = request.json
        name = data['name']
        age = int(data['age'])
        year1_marks = float(data['year1_marks'])
        year2_marks = float(data['year2_marks'])
        studytime = float(data['study_time'])
        failures = int(data['failures'])

        # Predict final marks
        prediction = predict_new_input(model, scaler, age, year1_marks, year2_marks, studytime, failures)

        if prediction is None:
            print("Prediction returned None.")
            return jsonify({'error': 'Prediction failed due to internal error.'}), 500
        
        # Round the prediction to 2 decimal places
        rounded_prediction = round(float(prediction), 2)
        
        # Log the prediction result
        print(f"Prediction result: {rounded_prediction}")

        # Return the result as a JSON response
        response = jsonify({'prediction': rounded_prediction})
        response.headers['Content-Type'] = 'application/json'
        return response

    except KeyError as ke:
        print(f"KeyError: {ke}")
        return jsonify({'error': f'Missing required field: {ke}'}), 400
    except ValueError as ve:
        print(f"ValueError: {ve}")
        return jsonify({'error': 'Invalid input. Please ensure all fields contain correct values.'}), 400
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': 'Internal server error.'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)