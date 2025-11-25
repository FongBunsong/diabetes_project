# api.py
from flask import Flask, request, jsonify
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the model
with open('diabetes_model.pkl', 'rb') as f:
    model_data = pickle.load(f)

model = model_data['model']
scaler = model_data['scaler']
label_encoders = model_data['label_encoders']
feature_names = model_data['feature_names']

# Define encoding mappings based on your training data
gender_mapping = {'Female': 0, 'Male': 1}
smoking_mapping = {'never': 0, 'No Info': 1, 'current': 2, 'former': 3, 'ever': 4, 'not current': 5}

@app.route('/')
def home():
    return "Diabetes Prediction API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data
        data = request.get_json()
        
        # Create input array in correct order
        input_data = []
        for feature in feature_names:
            if feature in ['gender', 'smoking_history']:
                # Handle categorical features
                if feature == 'gender':
                    input_data.append(gender_mapping.get(data.get(feature), 1))
                elif feature == 'smoking_history':
                    input_data.append(smoking_mapping.get(data.get(feature), 1))
            else:
                # Handle numerical features
                input_data.append(float(data.get(feature, 0)))
        
        # Convert to DataFrame and scale
        input_df = pd.DataFrame([input_data], columns=feature_names)
        input_scaled = scaler.transform(input_df)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]
        
        # Return result
        result = {
            'diabetes_prediction': int(prediction),
            'probability': float(probability),
            'message': 'Diabetic' if prediction == 1 else 'Not Diabetic'
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)