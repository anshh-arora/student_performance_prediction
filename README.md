# Final Marks Predictor

This project predicts student marks using machine learning techniques, including Multilayer Perceptron (MLP) and Artificial Neural Networks (ANN). The project includes data preprocessing, model training, evaluation, and deployment through a Flask web app. The dataset `modified_data.csv` is processed in `final_model.ipynb`, where data cleaning, train-test split, and model comparison are performed. The trained model is saved as `final_marks_predictor.h5` and used in the Flask app (`app.py`) to predict marks via a web interface.

## Project Structure
- **app.py**: Main Flask application.
- **requirements.txt**: Python dependencies.
- **final_marks_predictor.h5**: Trained model.
- **scaler.pkl**: Scaler for data transformation.
- **final_model.ipynb**: Jupyter notebook for model training.
- **model.py**: Python script for model creation, training, and export.
- **modified_data.csv**: Cleaned dataset.

## Setup Instructions
1. Clone the repo: `git clone https://github.com/yourusername/final-marks-predictor.git`
2. Create and activate a virtual environment.
3. Install dependencies: `pip install -r requirements.txt`
4. Run the app: `python app.py` and access it at `http://127.0.0.1:5000/`

## Model Workflow
- Data loading and EDA.
- Train-test split.
- MLP/ANN model training.
- Model comparison and export.

## Usage
- Input features through the web interface and get predicted marks based on the trained model.

## Future Enhancements
- Incorporate more complex models and additional features for better accuracy.
