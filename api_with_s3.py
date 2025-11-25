# api_with_s3.py
from flask import Flask, request, jsonify
import pickle
import pandas as pd
import numpy as np
import boto3
import json
from datetime import datetime
import uuid

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

# S3 Configuration - REPLACE WITH YOUR BUCKET NAME
S3_BUCKET_NAME = 'diabetes-predictions-bucket'  # Change this to your actual bucket name
s3_client = boto3.client('s3', region_name='us-east-1')  # Change region if needed

def save_to_s3(input_data, prediction_result):
    """Save input data and prediction result to S3 bucket"""
    try:
        # Create unique filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        file_id = str(uuid.uuid4())[:8]
        filename = f"predictions/{timestamp}_{file_id}.json"
        
        # Prepare data to save
        s3_data = {
            'prediction_id': file_id,
            'timestamp': datetime.now().isoformat(),
            'input_data': input_data,
            'prediction_result': prediction_result
        }
        
        # Save to S3
        s3_client.put_object(
            Bucket=S3_BUCKET_NAME,
            Key=filename,
            Body=json.dumps(s3_data, indent=2),
            ContentType='application/json'
        )
        
        print(f"Successfully saved to S3: {filename}")
        return filename
        
    except Exception as e:
        print(f"Error saving to S3: {str(e)}")
        return None

@app.route('/')
def home():
    return "Diabetes Prediction API with S3 Storage is running! Use POST /predict with JSON data."

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return jsonify({
            'message': 'Send a POST request with JSON data to get predictions.',
            'required_features': feature_names,
            's3_bucket': S3_BUCKET_NAME
        })

    try:
        # Get JSON data
        data = request.get_json(force=True)

        # Check for missing features and warn
        missing_features = [f for f in feature_names if f not in data]
        if missing_features:
            return jsonify({'error': f'Missing features: {missing_features}'}), 400

        # Prepare input data
        input_data = []
        for feature in feature_names:
            value = data.get(feature)
            if feature == 'gender':
                input_data.append(gender_mapping.get(value, 1))
            elif feature == 'smoking_history':
                input_data.append(smoking_mapping.get(value, 1))
            else:
                try:
                    input_data.append(float(value))
                except (TypeError, ValueError):
                    return jsonify({'error': f'Invalid value for feature {feature}'}), 400

        # Convert to DataFrame and scale
        input_df = pd.DataFrame([input_data], columns=feature_names)
        input_scaled = scaler.transform(input_df)

        # Make prediction
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]

        # Prepare result
        result = {
            'diabetes_prediction': int(prediction),
            'probability': float(probability),
            'message': 'Diabetic' if prediction == 1 else 'Not Diabetic',
            'timestamp': datetime.now().isoformat()
        }

        # Save to S3
        s3_filename = save_to_s3(data, result)
        
        if s3_filename:
            result['s3_location'] = f"s3://{S3_BUCKET_NAME}/{s3_filename}"
            result['s3_status'] = 'success'
        else:
            result['s3_status'] = 'failed'
            result['s3_error'] = 'Could not save to S3 bucket'

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    try:
        # Test S3 connection
        s3_client.head_bucket(Bucket=S3_BUCKET_NAME)
        s3_status = 'connected'
    except Exception as e:
        s3_status = f'error: {str(e)}'
    
    return jsonify({
        'status': 'healthy',
        'model_loaded': True,
        's3_bucket': S3_STATUS,
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)