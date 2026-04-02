# Student Dropout Prediction - Setup & Usage

## 📦 Prerequisites
Install dependencies:
```bash
pip install -r requirements.txt
```

## 🔧 Workflow

### 1️⃣ Train & Save Model (Joblib Dump)
```bash
python train_model.py
```
This will:
- Load training data
- Build label encoders for categorical columns
- Train a RandomForest classifier
- **Save everything using `joblib.dump()`** → `dropout_artifacts.joblib`

### 2️⃣ Use Model for Predictions

#### Option A: Local Streamlit App (Joblib Load)
```bash
streamlit run streamlit_app.py
```
- Toggle **OFF** "Use FastAPI backend"
- The app will **load the model using `joblib.load()`**
- Make predictions locally without API

#### Option B: FastAPI Backend + Streamlit
Terminal 1 - Start API:
```bash
python main.py
uvicorn main:app --reload
```

Terminal 2 - Start Streamlit:
```bash
streamlit run streamlit_app.py
```
- Toggle **ON** "Use FastAPI backend"
- Makes API calls to FastAPI server

## 🎯 Key Files

| File | Purpose |
|------|---------|
| `train_model.py` | Train model and save using **joblib.dump()** |
| `dropout_artifacts.joblib` | Saved model/encoders (binary file) |
| `streamlit_app.py` | UI with **joblib.load()** for local predictions |
| `main.py` | FastAPI backend with model training |
| `cleaned_students_dropout01.csv` | Training data |

## 📚 Joblib Dump/Load Details

### Saving (Dump)
```python
import joblib

artifacts = {
    'model': trained_model,
    'encoders': encoders_dict,
    'feature_order': ['col1', 'col2', ...]
}
joblib.dump(artifacts, 'path/to/file.joblib')
```

### Loading (Load)
```python
import joblib

artifacts = joblib.load('path/to/file.joblib')
model = artifacts['model']
encoders = artifacts['encoders']
```

## 🚀 Quick Start
```bash
# 1. Train and save
python train_model.py

# 2. Run Streamlit app
streamlit run streamlit_app.py

# 3. Toggle "Use FastAPI backend" OFF for local predictions
```
