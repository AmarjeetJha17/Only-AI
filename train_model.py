"""
Train and save model using joblib dump/load
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# Load data
print("[*] Loading data...")
df = pd.read_csv('cleaned_students_dropout01.csv')
print(f"    Dataset shape: {df.shape}")

# Define columns
categorical_cols = ['School_Type', 'Location', 'Infrastructure', 'Teaching_Staff', 'Gender', 'Caste', 'Socioeconomic_Status']
feature_order = ['School_Type', 'Location', 'Infrastructure', 'Teaching_Staff', 'Gender', 'Caste', 'Age', 'Standard', 'Socioeconomic_Status']

# Build encoders
print("[*] Building encoders...")
encoders = {}
for col in categorical_cols:
    enc = LabelEncoder()
    enc.fit(df[col])
    encoders[col] = enc
    print(f"    {col}: {len(enc.classes_)} unique values")

# Prepare features
print("[*] Preparing features...")
X = df[feature_order].copy()
y = df['Dropout_Status']

# Encode categorical columns
X_encoded = X.copy()
for col in categorical_cols:
    X_encoded[col] = encoders[col].transform(X[col])

print(f"    Features shape: {X_encoded.shape}")
print(f"    Target classes: {y.unique()}")

# Train model
print("[*] Training RandomForest model...")
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_encoded, y)
print(f"    Model trained successfully")

# Create artifacts dictionary
artifacts = {
    'model': model,
    'encoders': encoders,
    'categorical_cols': categorical_cols,
    'feature_order': feature_order,
}

# Save using joblib.dump
model_path = 'dropout_artifacts.joblib'
print(f"[*] Saving model to {model_path}...")
joblib.dump(artifacts, model_path)
file_size = os.path.getsize(model_path) / (1024 * 1024)  # Convert to MB
print(f"    Saved successfully ({file_size:.2f} MB)")

# Load using joblib.load (verify it works)
print(f"[*] Loading model from {model_path}...")
loaded_artifacts = joblib.load(model_path)
print(f"    Keys in loaded artifacts: {list(loaded_artifacts.keys())}")

# Quick test
print("\n[*] Testing with sample prediction...")
test_input = {
    'School_Type': 'Government',
    'Location': 'Urban',
    'Infrastructure': 'Good',
    'Teaching_Staff': 'Adequate',
    'Gender': 'Male',
    'Caste': 'General',
    'Age': 14,
    'Standard': 8,
    'Socioeconomic_Status': 'Middle Income',
}

# Encode input
encoded_test = {}
for key, value in test_input.items():
    if key in loaded_artifacts['encoders']:
        encoded_test[key] = loaded_artifacts['encoders'][key].transform([value])[0]
    else:
        encoded_test[key] = value

# Create feature array
X_test = np.array([encoded_test[col] for col in feature_order]).reshape(1, -1)

# Predict
pred = loaded_artifacts['model'].predict(X_test)[0]
probs = loaded_artifacts['model'].predict_proba(X_test)[0]
prob_dict = {cls: f"{p*100:.1f}%" for cls, p in zip(loaded_artifacts['model'].classes_, probs)}

print(f"    Input: {test_input}")
print(f"    Prediction: {pred}")
print(f"    Probabilities: {prob_dict}")

print("\n[+] Training complete! Ready to use with Streamlit app.")
