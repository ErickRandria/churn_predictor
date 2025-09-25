from flask import Flask, request, render_template
import joblib
import pandas as pd
import numpy as np
import os
import logging

# Initialize the Flask application
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the trained model pipeline
try:
    if os.path.exists('model.pkl'):
        model_pipeline = joblib.load('model.pkl')
        logger.info("Model pipeline loaded successfully.")
    else:
        raise FileNotFoundError("model.pkl not found. Please ensure the model file exists.")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    model_pipeline = None

# Define the feature columns based on the Telco Customer Churn dataset
EXPECTED_FEATURES = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
    'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
    'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
    'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
    'MonthlyCharges', 'TotalCharges'
]

def preprocess_input_data(form_data):
    """
    Preprocess the input data to match the training data format
    """
    try:
        # Create a DataFrame from the form data
        data = pd.DataFrame([form_data])
        
        # Handle missing features by setting default values
        for feature in EXPECTED_FEATURES:
            if feature not in data.columns:
                if feature in ['tenure', 'MonthlyCharges', 'TotalCharges']:
                    data[feature] = 0
                elif feature == 'SeniorCitizen':
                    data[feature] = 0
                else:
                    data[feature] = 'No'  # Default for categorical features
        
        # Select only the expected features in the correct order
        data = data[EXPECTED_FEATURES]
        
        # Convert numerical columns
        numerical_columns = ['tenure', 'MonthlyCharges', 'TotalCharges']
        for col in numerical_columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')
            # Replace NaN values with 0 for numerical columns
            data[col] = data[col].fillna(0)
        
        # Convert SeniorCitizen to numeric (it's usually 0 or 1 in the dataset)
        if 'SeniorCitizen' in data.columns:
            data['SeniorCitizen'] = pd.to_numeric(data['SeniorCitizen'], errors='coerce')
            data['SeniorCitizen'] = data['SeniorCitizen'].fillna(0)
        
        # Handle TotalCharges special case (sometimes it's a string with spaces)
        if 'TotalCharges' in data.columns:
            data['TotalCharges'] = data['TotalCharges'].astype(str).str.strip()
            data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
            data['TotalCharges'] = data['TotalCharges'].fillna(0)
        
        logger.info(f"Preprocessed data shape: {data.shape}")
        logger.info(f"Data types: {data.dtypes.to_dict()}")
        
        return data
        
    except Exception as e:
        logger.error(f"Error in preprocessing: {str(e)}")
        raise

# Define the home page route
@app.route('/')
def home():
    return render_template('index.html')

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model_pipeline is None:
            return render_template('index.html', 
                                 prediction_text="Error: Model not loaded. Please check server logs.",
                                 error=True)
        
        # Get the form data
        form_data = request.form.to_dict()
        logger.info(f"Received form data: {form_data}")
        
        # Validate that we have some data
        if not form_data:
            return render_template('index.html', 
                                 prediction_text="Error: No data received from form.",
                                 error=True)
        
        # Preprocess the input data
        data = preprocess_input_data(form_data)
        
        # Make prediction
        prediction = model_pipeline.predict(data)
        prediction_proba = model_pipeline.predict_proba(data)
        
        logger.info(f"Prediction: {prediction[0]}")
        logger.info(f"Prediction probabilities: {prediction_proba[0]}")
        
        # Interpret the prediction
        if prediction[0] == 1:
            result = f"⚠️ Customer is Likely to Churn (Confidence: {prediction_proba[0][1]:.2%})"
            risk_level = "high"
        else:
            result = f"✅ Customer is Unlikely to Churn (Confidence: {prediction_proba[0][0]:.2%})"
            risk_level = "low"
        
        # Additional insights based on probability
        churn_probability = prediction_proba[0][1]
        if churn_probability > 0.8:
            insight = "Very High Risk - Immediate attention required"
        elif churn_probability > 0.6:
            insight = "High Risk - Consider retention strategies"
        elif churn_probability > 0.4:
            insight = "Medium Risk - Monitor customer behavior"
        else:
            insight = "Low Risk - Customer appears satisfied"
        
        return render_template('index.html', 
                             prediction_text=result,
                             insight=insight,
                             risk_level=risk_level,
                             churn_probability=f"{churn_probability:.1%}")
        
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        return render_template('index.html', 
                             prediction_text=f"Error making prediction: {str(e)}",
                             error=True)

# Health check route
@app.route('/health')
def health():
    status = {
        'model_loaded': model_pipeline is not None,
        'status': 'healthy' if model_pipeline is not None else 'unhealthy'
    }
    return status

# Run the app
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)