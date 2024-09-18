import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from matplotlib import pyplot as plt
import seaborn as sns

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Load the dataset
df = pd.read_csv("modified_student_data.csv")

# Display the first few rows to check the data
df.head()

# Function to remove outliers using IQR method
def remove_outliers_iqr(df):
    """
    Removes outliers from a dataframe using the IQR method, applied only to numerical columns.
    
    Args:
        df: Pandas DataFrame with numerical columns.
    
    Returns:
        DataFrame with outliers removed.
    """
    # Select only numerical columns
    df_numeric = df.select_dtypes(include=[float, int])

    # Calculate Q1 (25th percentile) and Q3 (75th percentile) for each numerical column
    Q1 = df_numeric.quantile(0.25)
    Q3 = df_numeric.quantile(0.75)
    
    # Calculate the Interquartile Range (IQR)
    IQR = Q3 - Q1
    
    # Define the bounds for outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Filter the data and keep only rows that are within the bounds for numeric columns
    df_clean = df[~((df_numeric < lower_bound) | (df_numeric > upper_bound)).any(axis=1)]
    
    # Print number of outliers removed
    num_outliers = df.shape[0] - df_clean.shape[0]
    print(f"Removed {num_outliers} outliers")
    
    return df_clean

# Remove outliers from the dataset
df_clean = remove_outliers_iqr(df)


# EDA (Expolratory data analysis)

# Plot the distribution of final grades (G3)
plt.figure(figsize=(10, 6))
sns.histplot(df['G3'], bins=10, kde=True, color='skyblue')
plt.title('Distribution of Final Grades (G3)')
plt.xlabel('Final Grade (G3)')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()


# Plot study time vs final grades (G3)
plt.figure(figsize=(10, 6))
sns.barplot(data=df, x='studytime', y='G3', hue='sex', palette='coolwarm')
plt.title('Study Time vs Final Grades (G3)')
plt.xlabel('Study Time')
plt.ylabel('Final Grade (G3)')
plt.legend(title='Sex')
plt.grid(True)
plt.show()


# Plot final grades (G3) by family size (famsize)
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='famsize', y='G3', palette='pastel')
plt.title('Final Grades (G3) by Family Size')
plt.xlabel('Family Size')
plt.ylabel('Final Grade (G3)')
plt.grid(True)
plt.show()

# Plot final grades (G3) by mother's education (Medu) and father's education (Fedu)
plt.figure(figsize=(12, 6))
sns.scatterplot(data=df, x='Medu', y='G3', hue='Fedu', palette='viridis', style='sex')
plt.title('Final Grades (G3) by Mother\'s Education (Medu) and Father\'s Education (Fedu)')
plt.xlabel('Mother\'s Education (Medu)')
plt.ylabel('Final Grade (G3)')
plt.legend(title='Father\'s Education (Fedu)')
plt.grid(True)
plt.show()



# Encode categorical features
label_encoders = {}
categorical_columns = df_clean.select_dtypes(include=['object']).columns

for col in categorical_columns:
    le = LabelEncoder()
    df_clean[col] = le.fit_transform(df_clean[col])
    label_encoders[col] = le

# Check the data types and ensure no missing values remain
print(df_clean.info())

# Rename G1, G2, G3 to meaningful names
df_clean = df_clean.rename(columns={'G1': 'year1_marks', 'G2': 'year2_marks', 'G3': 'final_marks'})

# Define the features (columns to use for prediction)
X = df_clean[['age', 'year1_marks', 'year2_marks', 'studytime', 'failures']]

# Define the target (final year marks)
y = df_clean['final_marks']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the StandardScaler
scaler = StandardScaler()

# Fit on training data and transform both training and test data
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Check the shapes of scaled data
print(f"X_train_scaled shape: {X_train_scaled.shape}, X_test_scaled shape: {X_test_scaled.shape}")


####### MLP Neural network
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

# Build the MLP model using scikit-learn
mlp = MLPRegressor(hidden_layer_sizes=(100,),  # Number of neurons in each hidden layer
                   solver='adam',              # Adam optimizer for gradient descent
                   learning_rate_init=0.01,    # Initial learning rate
                   max_iter=100,              # Maximum number of training iterations
                   random_state=42)

# Train the model
mlp.fit(X_train_scaled, y_train)

# Predict the target values for the test set
y_pred_mlp = mlp.predict(X_test_scaled)

# Calculate performance metrics
mse_mlp = mean_squared_error(y_test, y_pred_mlp)
rmse_mlp = np.sqrt(mse_mlp)
r2_mlp = r2_score(y_test, y_pred_mlp)

print(f"MLP - Mean Squared Error: {mse_mlp}")
print(f"MLP - Root Mean Squared Error: {rmse_mlp}")
print(f"MLP - R^2 Score: {r2_mlp}")

# Plot loss curve during training (MLPRegressor has a loss_curve_ attribute)
plt.figure(figsize=(12, 6))
plt.plot(mlp.loss_curve_)
plt.title('MLP Training Loss Curve')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.grid(True)
plt.show()

# Scatter plot of actual vs predicted values (MLP)
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_mlp, color='blue', label='Predicted vs Actual')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.title('MLP - Predicted vs Actual Values')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.grid(True)
plt.legend()
plt.show()



####### ANN
# Ensure reproducibility at the TensorFlow level
tf.keras.utils.set_random_seed(42)

# Define the model (Artificial Neural Network)
ann = tf.keras.models.Sequential([
    Dense(40, activation='relu', input_shape=(X_train_scaled.shape[1],)),  # Input layer
    Dropout(0.2),  # Dropout for regularization
    Dense(100, activation='relu'),  # Hidden layer with 100 units
    Dropout(0.2),
    Dense(60, activation='relu'),  # Another hidden layer
    Dropout(0.2),
    Dense(1)  # Output layer with 1 unit for regression
])

# Create Adam optimizer with a learning rate of 0.01
optimizer = Adam(learning_rate=0.01)

# Compile the model using Mean Squared Error as the loss function
ann.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])

# Train the model
history = ann.fit(X_train_scaled, y_train, batch_size=32, epochs=500, verbose=1, validation_split=0.2)

# Predict the target values for the test set
y_pred_ann = ann.predict(X_test_scaled)

# Calculate MSE, RMSE, MAE, and R-squared
mse_ann = mean_squared_error(y_test, y_pred_ann)
rmse_ann = np.sqrt(mse_ann)
mae_ann = mean_absolute_error(y_test, y_pred_ann)
r2_ann = r2_score(y_test, y_pred_ann)

# Print performance metrics
print(f"ANN - Mean Squared Error: {mse_ann}")
print(f"ANN - Root Mean Squared Error: {rmse_ann}")
print(f"ANN - Mean Absolute Error: {mae_ann}")
print(f"ANN - R-squared: {r2_ann}")

# Plot the training history
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

## comparision
"""MLP - Mean Squared Error: 2.288575149650712
MLP - Root Mean Squared Error: 1.5128037379814714
MLP - R^2 Score: 0.7013122175623938
"""
# Scores for MLP
mlp_mse = 2.2885
mlp_rmse = 1.5128
mlp_r2 = 0.7013

"""
ANN - Mean Squared Error: 1.0139725169793248
ANN - Root Mean Squared Error: 1.0069620236033356
ANN - Mean Absolute Error: 0.8039977654166843
ANN - R-squared: 0.867663859499892

"""
# Scores for ANN
ann_mse = 1.0139
ann_rmse = 1.0069
ann_mae = 0.8039
ann_r2 = 0.8676

# Create subplots
plt.figure(figsize=(14, 6))

# MSE comparison
plt.subplot(1, 3, 1)
plt.bar(['MLP', 'ANN'], [mlp_mse, ann_mse], color=['lightblue', 'lightgreen'])
plt.title('Mean Squared Error (MSE)')
plt.ylabel('MSE')
plt.grid(True)

# RMSE comparison

plt.subplot(1, 3, 2)
plt.bar(['MLP', 'ANN'], [mlp_rmse, ann_rmse], color=['lightblue', 'lightgreen'])
plt.title('Root Mean Squared Error (RMSE)')
plt.ylabel('RMSE')
plt.grid(True)

# R² comparison
plt.subplot(1, 3, 3)
plt.bar(['MLP', 'ANN'], [mlp_r2, ann_r2], color=['lightblue', 'lightgreen'])
plt.title('R² Score')
plt.ylabel('R² Score')
plt.ylim(0, 1)
plt.grid(True)

# Show the plot
plt.suptitle('Comparison of MLP vs ANN Model Performance')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()


# Define a function to test predictions
# Define a function to test predictions
def test_predictions(model, X_test, y_test):
    """
    This function takes the trained model, test data (X_test), and true target values (y_test),
    then predicts the target values and shows a comparison between the actual and predicted values.

    Args:
        model: Trained model used for prediction.
        X_test: Test data features.
        y_test: Actual target values (true final year marks).

    Returns:
        A DataFrame showing the actual vs predicted values and error metrics.
    """
    # Predict the target values for the test set
    y_pred = model.predict(X_test)
    
    # Flatten y_pred to match dimensions with y_test
    y_pred = y_pred.flatten()

    # Create a DataFrame to compare actual and predicted values
    results_df = pd.DataFrame({
        'Actual Marks': y_test,
        'Predicted Marks': y_pred
    })

    # Calculate the error metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Print error metrics
    print(f"Mean Squared Error: {mse}")
    print(f"Root Mean Squared Error: {rmse}")
    print(f"Mean Absolute Error: {mae}")
    print(f"R-squared: {r2}")

    # Show the first 10 predictions vs actual values
    print("\nComparison of Actual vs Predicted Values:")
    print(results_df.head(10))

    return results_df

# Call the function to test predictions
results = test_predictions(ann, X_test_scaled, y_test)


## prediction on new data
def predict_new_input(model, scaler, age, year1_marks, year2_marks, studytime, failures):
    """
    Function to take new input data and predict final marks using the trained model.
    
    Args:
        model: Trained Keras model used for prediction.
        scaler: Fitted StandardScaler used for scaling input data.
        age: Age of the student.
        year1_marks: Marks in year 1 (G1).
        year2_marks: Marks in year 2 (G2).
        studytime: Time spent studying.
        failures: Number of failures.
    
    Returns:
        Predicted final marks.
    """
    # Create a DataFrame for the new input (to match the structure of the original input)
    new_input = pd.DataFrame({
        'age': [age],
        'year1_marks': [year1_marks],
        'year2_marks': [year2_marks],
        'studytime': [studytime],
        'failures': [failures]
    })

    # Scale the new input data using the fitted scaler
    new_input_scaled = scaler.transform(new_input)

    # Predict final marks using the trained model
    predicted_marks = model.predict(new_input_scaled)
    
    # Return the predicted final marks
    return predicted_marks[0][0]  # Since it's a single prediction, return just the value

# Example: Predict for a student with the following details:
age = 18
year1_marks = 100
year2_marks = 100
studytime = 100
failures = 0

# Call the prediction function with trained model and input data
predicted_final_marks = predict_new_input(ann, scaler, age, year1_marks, year2_marks, studytime, failures)

# Print the predicted final marks
print(f"Predicted final marks: {predicted_final_marks:.2f}")


#### Export the model 
# Save the trained model to an HDF5 file (this saves the architecture, weights, and optimizer state)
ann.save("final_marks_predictor_model.h5")

# Save the scaler as well (using pickle since it is necessary for preprocessing)
import pickle
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
