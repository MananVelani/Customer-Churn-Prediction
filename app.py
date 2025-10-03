# app.py

import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle

# Initialize the flask app
app = Flask(__name__)

# Load the pickle file
with open('model.pkl', 'rb') as file:
    model_data = pickle.load(file)

# Extract the objects from the dictionary
model = model_data['model']
encoder = model_data['encoder']
scaler = model_data['scaler']
categorical_cols = model_data['categorical_cols']
numerical_cols = model_data['numerical_cols']

# Define the home page route
@app.route('/')
def home():
    return render_template('index.html')

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Get the form data as a dictionary
    form_data = request.form.to_dict()

    # Convert SeniorCitizen to integer
    form_data['SeniorCitizen'] = int(form_data['SeniorCitizen'])
    
    # Convert numerical columns from string to float
    for col in numerical_cols:
        if col != 'SeniorCitizen': # SeniorCitizen is already an int
            form_data[col] = float(form_data[col])

    # Create a pandas DataFrame from the form data
    # The [0] is important to create a DataFrame with a single row
    input_df = pd.DataFrame([form_data])

    # Separate categorical and numerical data from the input DataFrame
    input_cat_df = input_df[categorical_cols]
    input_num_df = input_df[numerical_cols]

    # 1. One-Hot Encode the categorical features
    # Use the loaded encoder to transform the data
    input_cat_encoded = encoder.transform(input_cat_df)

    # 2. Combine encoded categorical features with numerical features
    # The order must be the same as during training
    processed_input = np.concatenate([input_cat_encoded, input_num_df.values], axis=1)

    # 3. Scale the combined features
    # Use the loaded scaler to transform the data
    scaled_input = scaler.transform(processed_input)

    # 4. Make a prediction
    prediction = model.predict(scaled_input)
    prediction_proba = model.predict_proba(scaled_input)

    # Convert prediction to a user-friendly message
    churn_probability = prediction_proba[0][1] * 100  # Probability of 'Yes'
    if prediction[0] == 1:
        output = "Yes, this customer is likely to churn."
    else:
        output = "No, this customer is not likely to churn."

    # Render the result on the home page
    return render_template('index.html', 
                           prediction_text=f'{output}',
                           probability_text=f'Probability of Churn: {churn_probability:.2f}%')

# Run the app
if __name__ == "__main__":
    app.run(debug=True)