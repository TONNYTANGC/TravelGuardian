from flask import Flask, request, jsonify
from dotenv import load_dotenv
import joblib
import numpy as np
import pandas as pd
import warnings
import os
from flask_cors import CORS

# Ignore warnings
warnings.filterwarnings('ignore')
# Load environment variables from .env file
load_dotenv() 

app = Flask(__name__)
CORS(app)

app.config['DEBUG'] = os.environ.get('FLASK_DEBUG')

# Function to load saved models and scalers
def load_model_and_scaler(model_path, scaler_path):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

# Load all models and scalers
heart_model, heart_scaler = load_model_and_scaler('calibrated_heart_svm_model.pkl', 'heart_scaler.pkl')
hypertension_model, hypertension_scaler = load_model_and_scaler('hypertension_model.pkl', 'hypertension_scaler.pkl')  # RandomForest
diabetes_model, diabetes_scaler = load_model_and_scaler('diabetes_knn_model.pkl', 'diabetes_scaler.pkl')
asthma_model, asthma_scaler = load_model_and_scaler('asthma_knn_model.pkl', 'asthma_scaler.pkl')
airquality_model, airquality_scaler = load_model_and_scaler('airquality_knn_model.pkl', 'airquality_scaler.pkl')

# Load feature names
heart_features = joblib.load('heart_feature_names.pkl')
hypertension_features = joblib.load('hypertension_feature_names.pkl')
diabetes_features = joblib.load('diabetes_feature_names.pkl')
asthma_features = joblib.load('asthma_feature_names.pkl')
airquality_features = joblib.load('airquality_feature_names.pkl')

# Function to align features
def align_features(input_data, expected_features):
    aligned_data = pd.DataFrame(columns=expected_features)
    input_df = pd.DataFrame(input_data, columns=expected_features)
    for feature in expected_features:
        if feature in input_df.columns:
            aligned_data[feature] = input_df[feature]
        else:
            aligned_data[feature] = 0  # Fill missing features with default value
    return aligned_data

# Function to determine asthma risk addition based on AQI
def estimate_asthma_risk(aqi):
    if aqi < 50:
        return 0
    elif 50 <= aqi < 75:
        return 5
    elif 75 <= aqi < 100:
        return 10

# Function to predict travel risk
def predict_travel_risk(heart_data, hypertension_data, diabetes_data, asthma_data, airquality_data, mobility_data,other_disease_severities=None):
    heart_data_aligned = align_features(heart_data, heart_features)
    hypertension_data_aligned = align_features(hypertension_data, hypertension_features)
    diabetes_data_aligned = align_features(diabetes_data, diabetes_features)
    asthma_data_aligned = align_features(asthma_data, asthma_features)
    airquality_data_aligned = align_features(airquality_data, airquality_features)

    heart_data_scaled = heart_scaler.transform(heart_data_aligned.values)
    hypertension_data_scaled = hypertension_scaler.transform(hypertension_data_aligned.values)
    diabetes_data_scaled = diabetes_scaler.transform(diabetes_data_aligned.values)
    asthma_data_scaled = asthma_scaler.transform(asthma_data_aligned.values)
    airquality_data_scaled = airquality_scaler.transform(airquality_data_aligned.values)

    heart_probs = heart_model.predict_proba(heart_data_scaled)[:, 1][0] * 100
    hypertension_probs = hypertension_model.predict_proba(hypertension_data_scaled)[:, 1][0] * 100
    diabetes_probs = diabetes_model.predict_proba(diabetes_data_scaled)[:, 1][0] * 100
    asthma_probs = asthma_model.predict_proba(asthma_data_scaled)[:, 1][0] * 100
    airquality_probs = airquality_model.predict(airquality_data_scaled)[0]

    asthma_risk_addition = estimate_asthma_risk(airquality_probs)
    asthma_probs += asthma_probs * (asthma_risk_addition / 100)

    mobility_severity = (mobility_data) * 100  

    heart_weight = 0.2
    hypertension_weight = 0.2
    diabetes_weight = 0.2
    asthma_weight = 0.2
    mobility_weight = 0.1

    if other_disease_severities is None:
        other_disease_severities = []

    other_disease_weight = 0.1 / max(1, len(other_disease_severities))
    other_disease_risk = sum(severity * other_disease_weight for severity in other_disease_severities)

    combined_risk = (heart_weight * heart_probs +
                     hypertension_weight * hypertension_probs +
                     diabetes_weight * diabetes_probs +
                     asthma_weight * asthma_probs +
                     mobility_weight * mobility_severity +  # Example mobility severity
                     other_disease_risk)

    return combined_risk

# Flask route to accept POST requests for travel risk prediction
@app.route('/predict_travel_risk', methods=['POST'])
def predict():
    try:
        data = request.json

        heart_data = np.array([data['heart_data']])
        hypertension_data = np.array([data['hypertension_data']])
        diabetes_data = np.array([data['diabetes_data']])
        asthma_data = np.array([data['asthma_data']])
        airquality_data = np.array([data['airquality_data']])
        other_disease_severities = data.get('other_disease_severities', [])
        mobility_data = np.array([data['mobility_data']])

        risk_score = predict_travel_risk(heart_data, hypertension_data, diabetes_data, asthma_data, airquality_data, other_disease_severities, mobility_data)

        return jsonify({'risk_score': risk_score}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
