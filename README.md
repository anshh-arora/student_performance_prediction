# Final Marks Predictor

![Final Marks Predictor](https://via.placeholder.com/800x400.png?text=Final+Marks+Predictor+Demo)

## Project Overview

The Final Marks Predictor is an innovative machine learning project designed to forecast student performance. By leveraging advanced techniques such as Multilayer Perceptron (MLP) and Artificial Neural Networks (ANN), this project aims to provide accurate predictions of students' final marks based on various input features.

### Key Features

- Utilizes MLP and ANN for precise predictions
- Implements data preprocessing and feature engineering
- Provides a user-friendly web interface for easy interaction
- Offers real-time predictions through a Flask-based web application

## Table of Contents

- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Workflow](#model-workflow)
- [Files Description](#files-description)
- [Benefits](#benefits)
- [Demo Video](#demo-video)
- [How to Contribute](#how-to-contribute)
- [Contact](#contact)

## Project Structure

```
final-marks-predictor/
│
├── static/                 # Static files for the web app
├── templates/              # HTML templates for the web app
├── .env                    # Environment variables
├── .gitignore              # Git ignore file
├── app.log                 # Application logs
├── app.py                  # Main Flask application
├── final_marks_predictor_model.h5  # Trained model file
├── final_model.ipynb       # Jupyter notebook for model development
├── model.py                # Python script for model creation and training
├── modified_student_data.csv  # Processed dataset
├── Procfile                # Heroku deployment file
├── README.md               # Project documentation (this file)
├── requirements.txt        # Python dependencies
└── scaler.pkl              # Saved scaler for data transformation
```

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/final-marks-predictor.git
   ```

2. Navigate to the project directory:
   ```
   cd final-marks-predictor
   ```

3. Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

4. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

5. Set up environment variables:
   ```
   cp .env.example .env
   # Edit .env file with your configurations
   ```

## Usage

1. Run the Flask application:
   ```
   python app.py
   ```

2. Open your web browser and navigate to `http://127.0.0.1:5000/`

3. Input the required features in the web interface to get predictions.

## Model Workflow

1. Data Loading and Exploratory Data Analysis (EDA)
2. Data Preprocessing and Feature Engineering
3. Train-Test Split
4. Model Selection (MLP/ANN)
5. Model Training and Hyperparameter Tuning
6. Model Evaluation and Comparison
7. Export of Best Performing Model

## Files Description

- `app.py`: The main Flask application that serves the web interface and handles predictions.
- `final_model.ipynb`: Jupyter notebook containing the entire model development process, from data analysis to model evaluation.
- `model.py`: Python script for creating, training, and exporting the prediction model.
- `modified_student_data.csv`: The cleaned and processed dataset used for training the model.
- `final_marks_predictor_model.h5`: The saved, trained neural network model.
- `scaler.pkl`: Pickle file containing the fitted scaler for feature normalization.

## Benefits

1. **Accurate Predictions**: Leverages advanced ML techniques for high-precision forecasting of student performance.
2. **Early Intervention**: Enables educators to identify at-risk students early and provide timely support.
3. **Data-Driven Decision Making**: Provides valuable insights for educational policy and curriculum development.
4. **User-Friendly Interface**: Offers an intuitive web application for easy access to predictions.
5. **Scalability**: Can be easily adapted to different educational contexts and datasets.

## Demo Video

[![Final Marks Predictor Demo](https://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg)](https://www.youtube.com/watch?v=YOUTUBE_VIDEO_ID_HERE)

Click the image above to watch a demonstration of the Final Marks Predictor in action.

## How to Contribute

We welcome contributions to improve the Final Marks Predictor! Here's how you can contribute:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/AmazingFeature`)
3. Make your changes
4. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
5. Push to the branch (`git push origin feature/AmazingFeature`)
6. Open a Pull Request

Please ensure your code adheres to our coding standards and include tests for new features.

## Contact

Your Name - [@your_twitter](https://twitter.com/your_twitter) - email@example.com

Project Link: [https://github.com/yourusername/final-marks-predictor](https://github.com/yourusername/final-marks-predictor)

---

Thank you for your interest in the Final Marks Predictor project. We look forward to your contributions and feedback!
