# create_pickle.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import pickle

print("=== TASK 5: CREATING PICKLE FILE ===")

# Load data
df = pd.read_csv('diabetes_data.csv')
print(f"Original data shape: {df.shape}")

# Create a clean copy without warnings
df_clean = df.copy()

# Remove duplicates
df_clean = df_clean.drop_duplicates()
print(f"After removing duplicates: {df_clean.shape}")

# Handle missing values - using .loc to avoid warnings
for col in df_clean.columns:
    if df_clean[col].isnull().sum() > 0:
        if df_clean[col].dtype == 'object':
            # For categorical columns, use most frequent value
            most_frequent = df_clean[col].mode()[0]
            df_clean.loc[df_clean[col].isnull(), col] = most_frequent
        else:
            # For numerical columns, use median
            median_val = df_clean[col].median()
            df_clean.loc[df_clean[col].isnull(), col] = median_val

# Encode categorical variables - using .loc to avoid warnings
label_encoders = {}
for col in df_clean.select_dtypes(include=['object']).columns:
    if col != 'diabetes':
        le = LabelEncoder()
        # Use .loc to avoid SettingWithCopyWarning
        df_clean.loc[:, col] = le.fit_transform(df_clean[col].astype(str))
        label_encoders[col] = le
        print(f"Encoded column: {col}")

# Prepare data
X = df_clean.drop('diabetes', axis=1)
y = df_clean['diabetes']

print(f"Features shape: {X.shape}")
print(f"Target distribution:\n{y.value_counts()}")

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_scaled, y)

# Save as pickle
model_data = {
    'model': model,
    'scaler': scaler,
    'label_encoders': label_encoders,
    'feature_names': list(X.columns)
}

with open('diabetes_model.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print("SUCCESS: Pickle file created: diabetes_model.pkl")
print("Feature names saved:", list(X.columns))
print("Label encoders created for:", list(label_encoders.keys()))
print("Task 5 completed successfully")