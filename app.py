"""
Student Dropout Prediction - Streamlit App
Loads model and encoders from joblib. Creates them if they don't exist.
"""
import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import warnings
import os

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="🎓 Student Dropout Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🎓 Student Dropout Predictor")
st.markdown("---")

# ============================================================================
# Load or Create Model & Encoders
# ============================================================================

@st.cache_resource
def load_or_create_model():
    """Load existing model or create a new one from data"""
    model_path = "dropout_model.joblib"

    # Try to load existing model
    if os.path.exists(model_path):
        try:
            artifacts = joblib.load(model_path)
            st.sidebar.success("✅ Model loaded successfully")
            return artifacts
        except Exception as e:
            st.sidebar.warning(f"⚠️ Could not load model: {e}")

    # Create new model
    st.sidebar.info("📊 Creating new model from data...")

    df = pd.read_csv('cleaned_students_dropout01.csv')

    categorical_cols = ['School_Type', 'Location', 'Infrastructure', 'Teaching_Staff', 'Gender', 'Caste', 'Socioeconomic_Status']
    feature_order = ['School_Type', 'Location', 'Infrastructure', 'Teaching_Staff', 'Gender', 'Caste', 'Age', 'Standard', 'Socioeconomic_Status']

    # Build encoders
    encoders = {}
    for col in categorical_cols:
        enc = LabelEncoder()
        enc.fit(df[col])
        encoders[col] = enc

    # Prepare features
    X = df[feature_order].copy()
    y = df['Dropout_Status']

    X_encoded = X.copy()
    for col in categorical_cols:
        X_encoded[col] = encoders[col].transform(X[col])

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_encoded, y)

    # Save artifacts
    artifacts = {
        'model': model,
        'encoders': encoders,
        'categorical_cols': categorical_cols,
        'feature_order': feature_order,
    }

    joblib.dump(artifacts, model_path)
    st.sidebar.success(f"✅ Model saved to {model_path}")

    return artifacts

# Load artifacts
artifacts = load_or_create_model()
model = artifacts['model']
encoders = artifacts['encoders']
feature_order = artifacts['feature_order']
categorical_cols = artifacts['categorical_cols']

# ============================================================================
# Sidebar - Info
# ============================================================================
with st.sidebar:
    st.markdown("### 📋 Model Information")
    st.write(f"**Model Type:** Random Forest")
    st.write(f"**Classes:** {', '.join(model.classes_)}")
    st.write(f"**Features:** {len(feature_order)}")
    st.markdown("---")

# ============================================================================
# Main Prediction Form
# ============================================================================
st.markdown("### Enter Student Information")

col1, col2 = st.columns(2)

with col1:
    school_type = st.selectbox(
        "🏫 School Type",
        options=sorted(encoders['School_Type'].classes_)
    )
    location = st.selectbox(
        "📍 Location",
        options=sorted(encoders['Location'].classes_)
    )
    infrastructure = st.selectbox(
        "🏢 Infrastructure",
        options=sorted(encoders['Infrastructure'].classes_)
    )
    teaching_staff = st.selectbox(
        "👨‍🏫 Teaching Staff",
        options=sorted(encoders['Teaching_Staff'].classes_)
    )
    age = st.slider("👤 Age", min_value=3, max_value=25, value=14)

with col2:
    gender = st.selectbox(
        "👥 Gender",
        options=sorted(encoders['Gender'].classes_)
    )
    caste = st.selectbox(
        "📊 Caste",
        options=sorted(encoders['Caste'].classes_)
    )
    socio = st.selectbox(
        "💰 Socioeconomic Status",
        options=sorted(encoders['Socioeconomic_Status'].classes_)
    )
    standard = st.slider("📚 Standard (Grade)", min_value=1, max_value=12, value=8)

# ============================================================================
# Prediction Logic
# ============================================================================

if st.button("🔮 Predict Dropout Risk", use_container_width=True, type="primary"):
    # Prepare input
    user_input = {
        'School_Type': school_type,
        'Location': location,
        'Infrastructure': infrastructure,
        'Teaching_Staff': teaching_staff,
        'Gender': gender,
        'Caste': caste,
        'Age': age,
        'Standard': standard,
        'Socioeconomic_Status': socio,
    }

    # Encode categorical features
    encoded_input = {}
    for key, value in user_input.items():
        if key in encoders:
            encoded_input[key] = encoders[key].transform([value])[0]
        else:
            encoded_input[key] = value

    # Create feature array
    X_test = np.array([encoded_input[col] for col in feature_order]).reshape(1, -1)

    # Make prediction
    prediction = model.predict(X_test)[0]
    probabilities = model.predict_proba(X_test)[0]
    prob_dict = {cls: float(p) for cls, p in zip(model.classes_, probabilities)}

    # Display results
    st.markdown("---")
    st.markdown("### 📊 Prediction Results")

    col1, col2, col3 = st.columns(3)

    with col1:
        if prediction == "Dropout":
            st.error(f"### 🚨 {prediction}")
            st.caption("High Risk")
        else:
            st.success(f"### ✅ {prediction}")
            st.caption("Low Risk")

    with col2:
        max_prob = max(probabilities)
        st.metric("Confidence", f"{max_prob*100:.1f}%")

    with col3:
        st.metric("Prediction Class", prediction)

    # Probability distribution
    st.markdown("### 📈 Class Probabilities")
    prob_chart_data = pd.DataFrame({
        'Class': list(prob_dict.keys()),
        'Probability': list(prob_dict.values())
    })
    st.bar_chart(prob_chart_data.set_index('Class'))

    # Input summary
    st.markdown("### 📝 Input Summary")
    input_df = pd.DataFrame(user_input, index=[0]).T
    input_df.columns = ['Value']
    st.dataframe(input_df, use_container_width=True)

# ============================================================================
# Footer
# ============================================================================
st.markdown("---")
st.caption("💡 This model predicts student dropout risk based on school, demographic, and socioeconomic factors.")
