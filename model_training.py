# model_training.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

print("=== TASK 1: MODEL TRAINING AND EVALUATION ===")

# Step 1: Load data
print("Loading data...")
df = pd.read_csv('diabetes_data.csv')
print(f"Original data shape: {df.shape}")

# Step 2: Data cleaning
print("Cleaning data...")
df_clean = df.drop_duplicates()
print(f"After removing duplicates: {df_clean.shape}")

# Step 3: Handle missing values
print("Handling missing values...")
for col in df_clean.columns:
    if df_clean[col].isnull().sum() > 0:
        if df_clean[col].dtype == 'object':
            df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
        else:
            df_clean[col].fillna(df_clean[col].median(), inplace=True)

# Step 4: Encode categorical variables
print("Encoding categorical variables...")
label_encoders = {}
for col in df_clean.select_dtypes(include=['object']).columns:
    if col != 'diabetes':
        le = LabelEncoder()
        df_clean[col] = le.fit_transform(df_clean[col].astype(str))
        label_encoders[col] = le

# Step 5: Prepare features and target
X = df_clean.drop('diabetes', axis=1)
y = df_clean['diabetes']

# Step 6: Remove outliers
print("Removing outliers...")
for col in X.select_dtypes(include=[np.number]).columns:
    Q1 = X[col].quantile(0.25)
    Q3 = X[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    X[col] = np.clip(X[col], lower, upper)

# Step 7: Scale features
print("Scaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 8: Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 9: Define models
models = {
    'Random Forest': RandomForestClassifier(),
    'Logistic Regression': LogisticRegression(),
    'SVM': SVC(),
    'Decision Tree': DecisionTreeClassifier(),
    'K-Neighbors': KNeighborsClassifier()
}

# Step 10: Train and evaluate models
print("Training models...")
results = []

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    results.append({
        'Model': name,
        'Accuracy': round(accuracy, 4),
        'Precision': round(precision, 4),
        'Recall': round(recall, 4),
        'F1-Score': round(f1, 4)
    })

# Step 11: Display results
results_df = pd.DataFrame(results)
print("\n" + "="*60)
print("PERFORMANCE RESULTS TABLE")
print("="*60)
print(results_df.to_string(index=False))
print("="*60)

# Step 12: Find best model
best_model_idx = results_df['F1-Score'].idxmax()
best_model_name = results_df.loc[best_model_idx, 'Model']
print(f"\nBEST MODEL: {best_model_name}")
print("Task 1 completed successfully!")