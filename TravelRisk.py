import joblib
import numpy as np
import pandas as pd
import warnings
# Ignore warnings
warnings.filterwarnings('ignore')

# Function to load saved models and scalers
def load_model_and_scaler(model_path, scaler_path):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

# Load all your saved models and scalers
heart_model, heart_scaler = load_model_and_scaler('calibrated_heart_svm_model.pkl', 'heart_scaler.pkl')
hypertension_model, hypertension_scaler = load_model_and_scaler('hypertension_model.pkl', 'hypertension_scaler.pkl') #random forest 
diabetes_model, diabetes_scaler = load_model_and_scaler('diabetes_knn_model.pkl','diabetes_scaler.pkl')
asthma_model, asthma_scaler = load_model_and_scaler('asthma_knn_model.pkl','asthma_scaler.pkl')
airquality_model, airquality_scaler = load_model_and_scaler('airquality_knn_model.pkl','airquality_scaler.pkl')

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
            aligned_data[feature] = 0  # Fill missing features with default value, e.g., 0
    return aligned_data

# Function to determine asthma risk addition based on AQI
def estimate_asthma_risk(aqi):
    if aqi < 50:
        return 0
    elif 50 <= aqi < 75:
        return 5
    elif 75 <= aqi < 100:
        return 10
     
# Function to predict travel risk with custom weights and environmental factors modifying specific disease risks
def predict_travel_risk(heart_data, hypertension_data, diabetes_data, asthma_data, airquality_data, other_disease_severities=None):
    # Align features
    heart_data_aligned = align_features(heart_data, heart_features)
    hypertension_data_aligned = align_features(hypertension_data, hypertension_features)
    diabetes_data_aligned = align_features(diabetes_data, diabetes_features)
    asthma_data_aligned = align_features(asthma_data, asthma_features)
    airquality_data_aligned = align_features(airquality_data, airquality_features)

    # Preprocess each input data using respective scalers
    heart_data_scaled = heart_scaler.transform(heart_data_aligned)
    hypertension_data_scaled = hypertension_scaler.transform(hypertension_data_aligned)
    diabetes_data_scaled = diabetes_scaler.transform(diabetes_data_aligned)
    asthma_data_scaled = asthma_scaler.transform(asthma_data_aligned)
    airquality_data_scaled = airquality_scaler.transform(airquality_data_aligned) 

    # Predict severity levels using each model
    heart_probs = heart_model.predict_proba(heart_data_scaled)[:, 1][0] *100
    print(heart_probs)
    hypertension_probs = hypertension_model.predict_proba(hypertension_data_scaled)[:, 1][0] * 100 
    print(hypertension_probs)
    diabetes_probs = diabetes_model.predict_proba(diabetes_data_scaled)[:, 1][0] * 100 
    print(diabetes_probs)
    asthma_probs = asthma_model.predict_proba(asthma_data_scaled)[:, 1][0] *100 
    print(asthma_probs)
    airquality_probs = airquality_model.predict(airquality_data_scaled)
    print(airquality_probs[0])
    
    # Update asthma risk based on air quality
    aqi = airquality_probs[0]
    asthma_risk_addition = estimate_asthma_risk(aqi)
    asthma_probs = asthma_probs + (asthma_probs * (asthma_risk_addition / 100))

    # Define weights for each component
    heart_weight = 0.2
    hypertension_weight = 0.2
    diabetes_weight = 0.2
    asthma_weight = 0.2
    mobility_weight = 0.1

    # Default other_disease_severities to an empty list if not provided
    if other_disease_severities is None:
        other_disease_severities = []
    
    # Define weights for other diseases
    other_disease_weight = 0.1 / max(1, len(other_disease_severities))  # Distribute 0.1 weight among all other diseases

    # Calculate the total risk from other diseases
    other_disease_risk = sum(severity * other_disease_weight for severity in other_disease_severities)
    
    # Combine predictions with weights
    combined_risk = (heart_weight * heart_probs + 
                     hypertension_weight * hypertension_probs + 
                     diabetes_weight * diabetes_probs + 
                     asthma_weight * asthma_probs + 
                     mobility_weight * mobility_severity + 
                     other_disease_risk)
    
    return combined_risk

# Example usage:
heart_data = np.array([[58, 0, 1, 165, 1, 1, 2, 2, 3]]) 
hypertension_data = np.array([[61,0,0,225,95,28.58,65]]) 
diabetes_data = np.array([[0, 50, 4.7, 46, 4.9, 4.2, 0.9, 2.4, 1.4, 0.5, 24]])
asthma_data = np.array([[1, 0, 5, 1, 6.58, 1, 2.19, 0, 1]])
airquality_data = np.array([[2, 4, 5, 4, 2, 2, 4, 3, 2]])  

# Assume other diseases severities on a scale from 0 to 1
other_disease_severities = [0.7, 0.5, 0.3]  # Replace with actual severities of other diseases
mobility_severity = (1 / 20) * 100 
# Converting the data into DataFrames with proper column names
heart_data_df = pd.DataFrame(heart_data, columns=heart_features)
hypertension_data_df = pd.DataFrame(hypertension_data, columns=hypertension_features)
diabetes_data_df = pd.DataFrame(diabetes_data, columns=diabetes_features)
asthma_data_df = pd.DataFrame(asthma_data, columns=asthma_features)
airquality_data_df = pd.DataFrame(airquality_data, columns=airquality_features)

predicted_risk = predict_travel_risk(heart_data_df, hypertension_data_df, diabetes_data_df, asthma_data_df, airquality_data_df, other_disease_severities)
print(f"Predicted Travel Risk: {predicted_risk}")
