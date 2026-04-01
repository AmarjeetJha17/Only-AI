from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import pickle
import os
from typing import Optional

app = FastAPI(title="Student Dropout Prediction API", version="1.0.0")

# Load data
df = pd.read_csv('cleaned_students_dropout01.csv')

# Initialize encoders
encoders = {}
categorical_cols = ['School_Type', 'Location', 'Infrastructure', 'Teaching_Staff', 'Gender', 'Caste', 'Socioeconomic_Status']

for col in categorical_cols:
    encoders[col] = LabelEncoder()
    encoders[col].fit(df[col])

# Train model if not exists
model_path = 'dropout_model.pkl'
if not os.path.exists(model_path):
    X = df.drop('Dropout_Status', axis=1)
    y = df['Dropout_Status']

    # Encode categorical variables
    X_encoded = X.copy()
    for col in categorical_cols:
        X_encoded[col] = encoders[col].transform(X[col])

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_encoded, y)

    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
else:
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

@app.get("/")
def read_root():
    return {"message": "Student Dropout Prediction API", "version": "1.0.0"}

@app.get("/data")
def get_all_data(skip: int = Query(0, ge=0), limit: int = Query(10, ge=1, le=1000)):
    """Get all student data with pagination"""
    total = len(df)
    data = df.iloc[skip:skip+limit].to_dict(orient='records')
    return {"total": total, "skip": skip, "limit": limit, "data": data}

@app.get("/data/dropout/{status}")
def get_by_status(status: str, skip: int = Query(0, ge=0), limit: int = Query(10, ge=1, le=1000)):
    """Filter data by dropout status"""
    filtered = df[df['Dropout_Status'] == status]
    total = len(filtered)
    data = filtered.iloc[skip:skip+limit].to_dict(orient='records')
    return {"status": status, "total": total, "skip": skip, "limit": limit, "data": data}

@app.get("/stats")
def get_statistics():
    """Get overall statistics"""
    return {
        "total_students": len(df),
        "dropout_count": len(df[df['Dropout_Status'] == 'Dropout']),
        "enrolled_count": len(df[df['Dropout_Status'] == 'Enrolled']),
        "dropout_rate": round(len(df[df['Dropout_Status'] == 'Dropout']) / len(df) * 100, 2),
        "age_stats": {
            "min": int(df['Age'].min()),
            "max": int(df['Age'].max()),
            "mean": round(float(df['Age'].mean()), 2)
        },
        "by_school_type": df.groupby('School_Type')['Dropout_Status'].value_counts().to_dict()
    }

@app.get("/stats/location")
def get_location_stats():
    """Get statistics by location"""
    stats = df.groupby('Location')['Dropout_Status'].value_counts().unstack(fill_value=0).to_dict()
    return {"location_stats": stats}

@app.post("/predict")
def predict_dropout(
    School_Type: str,
    Location: str,
    Infrastructure: str,
    Teaching_Staff: str,
    Gender: str,
    Caste: str,
    Age: int,
    Standard: int,
    Socioeconomic_Status: str
):
    """Predict dropout status for a student"""
    try:
        input_data = {
            'School_Type': School_Type,
            'Location': Location,
            'Infrastructure': Infrastructure,
            'Teaching_Staff': Teaching_Staff,
            'Gender': Gender,
            'Caste': Caste,
            'Age': Age,
            'Standard': Standard,
            'Socioeconomic_Status': Socioeconomic_Status
        }

        # Encode input
        encoded_input = {}
        for col, val in input_data.items():
            if col in categorical_cols:
                encoded_input[col] = encoders[col].transform([val])[0]
            else:
                encoded_input[col] = val

        # Create feature array
        feature_order = ['School_Type', 'Location', 'Infrastructure', 'Teaching_Staff', 'Gender', 'Caste', 'Age', 'Standard', 'Socioeconomic_Status']
        X = np.array([encoded_input[col] for col in feature_order]).reshape(1, -1)

        # Predict
        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0]

        return {
            "output": prediction
        }
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

@app.get("/columns")
def get_columns():
    """Get available columns and their values"""
    return {
        "columns": df.columns.tolist(),
        "categorical_values": {
            col: df[col].unique().tolist() for col in categorical_cols
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
