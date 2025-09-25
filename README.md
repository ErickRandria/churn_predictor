# Telco Customer Churn Prediction

A machine learning project that predicts customer churn for a telecommunications company using XGBoost classifier with a complete Flask web application for real-time predictions.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Features](#features)
- [Model Performance](#model-performance)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Technical Details](#technical-details)
- [API Endpoints](#api-endpoints)
- [Contributing](#contributing)
- [License](#license)

## Overview

Customer churn prediction is crucial for telecommunications companies to identify customers at risk of leaving and implement retention strategies. This project provides:

- **Machine Learning Model**: XGBoost classifier with comprehensive preprocessing pipeline
- **Web Application**: Flask-based interface for real-time churn predictions
- **Data Analysis**: Comprehensive exploratory data analysis and correlation studies
- **Production Ready**: Complete deployment-ready application with error handling

## Dataset

The project uses the **Telco Customer Churn** dataset from Kaggle, which contains:
- **7,043 customers** with 21 features
- **Customer demographics**: Gender, age, partner status
- **Service details**: Phone service, internet service, streaming options
- **Account information**: Contract type, payment method, charges
- **Target variable**: Churn (Yes/No)

**Dataset Source**: [Kaggle - Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

## Features

### Customer Demographics
- Gender, Senior Citizen status
- Partner and Dependents information

### Services
- Phone Service and Multiple Lines
- Internet Service (DSL, Fiber optic, None)
- Online Security, Backup, Device Protection
- Tech Support, Streaming TV/Movies

### Account Information
- Contract type (Month-to-month, One year, Two year)
- Payment method and billing preferences
- Monthly charges and Total charges
- Tenure (months as customer)

## Model Performance

The XGBoost model achieves the following performance metrics:

- **Accuracy**: 77.29%
- **Precision**: 58% (Churn class)
- **Recall**: 51% (Churn class)
- **F1-Score**: 54% (Churn class)

### Classification Report
```
              precision    recall  f1-score   support
           0       0.83      0.87      0.85      1035
           1       0.58      0.51      0.54       374
    accuracy                           0.77      1409
   macro avg       0.71      0.69      0.70      1409
weighted avg       0.76      0.77      0.77      1409
```

## Installation

### Prerequisites
- Python 3.7+
- pip package manager

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/telco-churn-prediction.git
   cd telco-churn-prediction
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the dataset**
   - Download the Telco Customer Churn dataset from Kaggle
   - Place `WA_Fn-UseC_-Telco-Customer-Churn.csv` in the project root

5. **Train the model** (if needed)
   ```bash
   jupyter notebook telco_churn_analysis.ipynb
   ```

## Usage

### Running the Web Application

1. **Start the Flask server**
   ```bash
   python app.py
   ```

2. **Access the application**
   - Open your browser and navigate to `http://localhost:5000`
   - Fill in the customer information form
   - Click "Predict Churn Risk" to get the prediction

### API Usage

The application provides a REST API endpoint:

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "gender=Male&SeniorCitizen=0&Partner=Yes&..."
```

## Project Structure

```
telco-churn-prediction/
├── app.py                          # Flask application
├── model.pkl                       # Trained model pipeline
├── telco_churn_analysis.ipynb      # Jupyter notebook with full analysis
├── templates/
│   └── index.html                  # Web interface template
├── requirements.txt                # Python dependencies
├── WA_Fn-UseC_-Telco-Customer-Churn.csv  # Dataset
└── README.md                       # Project documentation
```

## Technical Details

### Data Preprocessing Pipeline
- **Numerical Features**: Standardized using StandardScaler
- **Categorical Features**: One-hot encoded with unknown value handling
- **Missing Values**: TotalCharges imputed with 0 for new customers
- **Target Encoding**: Binary encoding for Churn (Yes=1, No=0)

### Model Architecture
- **Algorithm**: XGBoost Classifier
- **Pipeline**: Scikit-learn Pipeline with preprocessing and classification
- **Cross-validation**: Stratified train-test split (80/20)
- **Feature Engineering**: Comprehensive preprocessing of mixed data types

### Key Features of the Web Application
- **Responsive Design**: Modern, mobile-friendly interface
- **Form Validation**: Client-side and server-side validation
- **Smart Interactions**: Dynamic form fields based on service selections
- **Error Handling**: Comprehensive error management and logging
- **Health Check**: Application monitoring endpoint

## API Endpoints

### Main Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main application interface |
| `/predict` | POST | Churn prediction endpoint |
| `/health` | GET | Application health check |

### Request Format

The `/predict` endpoint accepts form data with all 19 customer features:

```json
{
  "gender": "Male",
  "SeniorCitizen": "0",
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": "12",
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "DSL",
  "OnlineSecurity": "Yes",
  "OnlineBackup": "No",
  "DeviceProtection": "No",
  "TechSupport": "Yes",
  "StreamingTV": "No",
  "StreamingMovies": "No",
  "Contract": "One year",
  "PaperlessBilling": "No",
  "PaymentMethod": "Mailed check",
  "MonthlyCharges": "65.50",
  "TotalCharges": "786.00"
}
```

### Response Format

```json
{
  "prediction": "Customer is Unlikely to Churn (Confidence: 78.5%)",
  "risk_level": "low",
  "churn_probability": "21.5%",
  "insight": "Low Risk - Customer appears satisfied"
}
```

## Model Insights

### Key Predictive Features
Based on correlation analysis, the most important features for churn prediction are:

1. **Contract Type**: Month-to-month contracts show higher churn rates
2. **Tenure**: Longer-tenured customers are less likely to churn
3. **Monthly Charges**: Higher charges correlate with increased churn risk
4. **Payment Method**: Electronic check payments show higher churn rates
5. **Internet Service**: Fiber optic customers show different churn patterns

### Business Insights
- **New Customer Focus**: Customers with short tenure require attention
- **Contract Strategy**: Encouraging longer-term contracts reduces churn
- **Service Quality**: Internet service quality impacts customer satisfaction
- **Pricing Strategy**: Monthly charges significantly influence churn decisions

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 style guidelines
- Add unit tests for new features
- Update documentation for API changes
- Ensure all tests pass before submitting PR

## Future Enhancements

- [ ] Model interpretability with SHAP values
- [ ] A/B testing framework for model versions
- [ ] Real-time model monitoring and drift detection
- [ ] Integration with customer databases
- [ ] Advanced ensemble methods
- [ ] Feature importance visualization

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Dataset**: IBM Sample Data Sets
- **Kaggle**: For hosting the dataset
- **Scikit-learn**: For machine learning utilities
- **XGBoost**: For the gradient boosting framework
- **Flask**: For the web application framework

## Contact

For questions or suggestions, please open an issue on GitHub or contact [email_address](mailto:hajjaerick@gmail.com).

---
