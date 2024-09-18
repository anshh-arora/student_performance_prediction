from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import pickle
import mysql.connector
from mysql.connector import Error
from datetime import datetime
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# Load the trained model and scaler
model = load_model('final_marks_predictor_model.h5')
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Database configuration from environment variables
db_config = {
    'host': os.getenv('DB_HOST'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'database': os.getenv('DB_NAME')
}

def get_db_connection():
    try:
        connection = mysql.connector.connect(**db_config)
        if connection.is_connected():
            return connection
    except Error as e:
        print(f"Error while connecting to MySQL: {e}")
        return None

def init_db():
    conn = get_db_connection()
    if conn is not None:
        try:
            cursor = conn.cursor()
            cursor.execute('''CREATE TABLE IF NOT EXISTS student_prediction_data
                            (id INT AUTO_INCREMENT PRIMARY KEY,
                             name VARCHAR(255),
                             age INT,
                             year1_marks FLOAT,
                             year2_marks FLOAT,
                             study_time FLOAT,
                             failures INT,
                             predicted_score FLOAT,
                             timestamp DATETIME)''')
            conn.commit()
        except Error as e:
            print(f"Error creating table: {e}")
        finally:
            if conn.is_connected():
                cursor.close()
                conn.close()

init_db()

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
        # Log the incoming form data
        print("Received form data:", request.form)

        # Retrieve form data
        name = request.form['name']
        age = int(request.form['age'])
        year1_marks = float(request.form['year1_marks'])
        year2_marks = float(request.form['year2_marks'])
        studytime = float(request.form['study_time'])
        failures = int(request.form['failures'])

        # Predict final marks
        prediction = predict_new_input(model, scaler, age, year1_marks, year2_marks, studytime, failures)

        if prediction is None:
            print("Prediction returned None.")
            return jsonify({'error': 'Prediction failed due to internal error.'}), 500
        
        # Round the prediction to 2 decimal places
        rounded_prediction = round(float(prediction), 2)
        
        # Log the prediction result
        print(f"Prediction result: {rounded_prediction}")

        # Store data in the database
        conn = get_db_connection()
        if conn is not None:
            try:
                cursor = conn.cursor()
                cursor.execute('''INSERT INTO student_prediction_data 
                                (name, age, year1_marks, year2_marks, study_time, failures, predicted_score, timestamp)
                                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)''',
                               (name, age, year1_marks, year2_marks, studytime, failures, rounded_prediction, datetime.now()))
                conn.commit()
            except Error as e:
                print(f"Error inserting data: {e}")
                return jsonify({'error': 'Database error occurred.'}), 500
            finally:
                if conn.is_connected():
                    cursor.close()
                    conn.close()

        # Return the result as a JSON response
        return jsonify({'prediction': rounded_prediction})
    
    except KeyError as ke:
        print(f"KeyError: {ke}")
        return jsonify({'error': f'Missing required field: {ke}'}), 400
    except ValueError as ve:
        print(f"ValueError: {ve}")
        return jsonify({'error': 'Invalid input. Please ensure all fields contain correct values.'}), 400
    except Exception as e:
        print(f"Error during form submission: {e}")
        return jsonify({'error': 'Internal server error.'}), 500

if __name__ == '__main__':
    app.run(debug=True)
