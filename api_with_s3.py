import boto3
import json
from flask import Flask, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# Load your model
with open('diabetes_model.pkl', 'rb') as f:
    model_data = pickle.load(f)
model = model_data['model']
scaler = model_data['scaler']

# S3 setup
s3 = boto3.client('s3')
BUCKET_NAME = 'diabetes-predictions-fongbunsong'

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    # Convert input to DataFrame
    df = pd.DataFrame([data])
    
    # Apply scaler if needed
    # X = scaler.transform(df)  # Uncomment if your model needs scaling
    
    # Make prediction
    prediction = model.predict(df)[0]
    
    # Prepare data to store in S3
    s3_data = {
        'input': data,
        'prediction': int(prediction)
    }
    
    # Create a unique filename
    filename = f"prediction_{data.get('age', 'NA')}_{data.get('gender', 'NA')}.json"
    
    # Upload to S3
    s3.put_object(
        Bucket=BUCKET_NAME,
        Key=filename,
        Body=json.dumps(s3_data)
    )
    
    return jsonify({'prediction': int(prediction)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
