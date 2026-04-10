# Only-AI 🚀

**AI-Powered Personalized Education to Combat the Silent Academic Dropout Crisis**

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)

## 🎯 Overview

**Only-AI** is an intelligent **EdTech** solution built for **Hackathon 2026** that tackles India's **"Silent Academic Dropout Crisis"**.

It uses machine learning to predict student dropout risk in real time and provides a **personalized adaptive learning layer** that continuously maps a student’s understanding, identifies exact knowledge gaps, and dynamically rebuilds their learning path — making quality, personalized education accessible to every student regardless of location or income.

## 🔥 The Problem

India produces over 15 million graduates annually, yet a large proportion enter higher education without mastering foundational concepts. Traditional classrooms follow a rigid, one-size-fits-all approach:

- Teachers manage 60–80 students and cannot monitor individual comprehension in real time.
- Gaps in understanding (e.g., calculus, organic chemistry) go undetected until exams.
- Personalized support (coaching, tutors) is expensive and inaccessible in tier-2/3 cities.
- Online content is abundant but completely passive and non-adaptive.

**Result**: Students lose confidence, disengage, and drop out — **not because they lack intelligence, but because the system never adapts to them**.

## 💡 Our Solution

Only-AI builds an **intelligent personalized learning layer** that:

- Predicts dropout risk using student data
- Identifies specific knowledge gaps
- Delivers real-time personalized interventions
- Adapts learning paths dynamically

## ✨ Key Features

- **Dropout Risk Prediction** using Random Forest Classifier
- **Interactive Streamlit Dashboard** with:
  - Home page with key metrics
  - Full Exploratory Data Analysis (EDA)
  - Power BI-style visualizations
  - Real-time **Predict** page with SHAP explainability
- **FastAPI Backend** for model serving
- **Comprehensive data pipeline** (cleaning → EDA → modeling)
- **Professional Power BI dashboard** (`powerbi_project.pbix`)

## 🛠 Tech Stack

- **Language**: Python
- **Machine Learning**: scikit-learn (Random Forest)
- **Frontend**: Streamlit
- **Backend**: FastAPI + Uvicorn
- **Data Handling**: Pandas, NumPy
- **Visualization**: Plotly, Matplotlib, Power BI
- **Model Persistence**: Joblib
- **Interpretability**: SHAP

## 📁 Project Structure

```bash
Only-AI/
├── streamlit_app.py              # Main Streamlit frontend
├── main.py                       # FastAPI backend
├── app.py
├── train_model.py                # Model training script
├── prediction_model.ipynb        # Model development & evaluation
├── data_cleaning.ipynb
├── EDA+data_visualisation.ipynb
├── students_dropout.csv          # Raw dataset
├── cleaned_students_dropout01.csv
├── requirements.txt
├── powerbi_project.pbix          # Power BI dashboard
├── SETUP.md
└── dropout_artifacts.joblib      # Trained model (generated after training)
```
1. Clone the Repository
```bash
git clone https://github.com/AmarjeetJha17/Only-AI.git
cd Only-AI
```
# 📊 Dataset

students_dropout.csv → Raw student demographic and performance data
cleaned_students_dropout01.csv → Preprocessed data used for modeling

# 📈 Visualizations

Detailed EDA → EDA+data_visualisation.ipynb
Interactive Power BI dashboard → powerbi_project.pbix

# 🔮 Future Enhancements

Real-time concept-level knowledge gap detection
Integration with LMS platforms (Moodle, Google Classroom, etc.)
AI-powered learning resource recommender
Mobile app support
Multi-language support (Hindi + regional languages)

# 🤝 Contributing
Contributions, issues, and feature requests are welcome!
Feel free to fork the repository and submit a pull request.
# 📄 License
This project is open-source and available under the MIT License.
# 🙏 Acknowledgements
Built with ❤️ for Hackathon 2026 with the vision of making quality, personalized education truly accessible to every student in India.
