import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib

st.set_page_config(page_title="Student Dropout Live Demo", page_icon="🎓", layout="wide")

# ====================== LOAD DATA & MODEL ======================
@st.cache_data
def load_data():
    return pd.read_csv("cleaned_students_dropout01.csv")

df = load_data()

@st.cache_resource
def load_model():
    return joblib.load("dropout_model.pkl")   # ← from your prediction_model.ipynb

model = load_model()

# ====================== SIDEBAR NAVIGATION ======================
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["🏠 Home", "📊 Full EDA", "📊 Power BI Dashboard", "🔮 Predict"],
    label_visibility="collapsed"
)

st.sidebar.success("✅ Model loaded from prediction_model.ipynb")

# ====================== HOME ======================
if page == "🏠 Home":
    st.title("🎓 Student Dropout Prediction - Live Demo")
    st.markdown("**Real-time prediction + Full EDA from your notebook**")
    
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Total Students", f"{len(df):,}")
    with c2: st.metric("Total Dropouts", (df['Dropout_Status']=='Dropout').sum())
    with c3: st.metric("Dropout Rate", f"{(df['Dropout_Status']=='Dropout').mean()*100:.2f}%")
    with c4: st.metric("Avg Age", f"{df['Age'].mean():.2f}")

    st.subheader("Sample Data")
    st.dataframe(df.head(10), use_container_width=True)

# ====================== FULL EDA ======================
elif page == "📊 Full EDA":
    st.title("📊 Full Exploratory Data Analysis")
    tab1, tab2, tab3, tab4 = st.tabs(["Distributions", "Categorical", "vs Target", "Correlation"])
    
    with tab1:
        st.subheader("Age & Standard Distribution")
        fig = make_subplots(rows=2, cols=2)
        fig.add_trace(go.Histogram(x=df['Age'], marker_color='#1f77b4'), row=1, col=1)
        fig.add_trace(go.Box(y=df['Age'], marker_color='#1f77b4'), row=1, col=2)
        fig.add_trace(go.Histogram(x=df['Standard'], marker_color='#ff7f0e'), row=2, col=1)
        fig.add_trace(go.Box(y=df['Standard'], marker_color='#ff7f0e'), row=2, col=2)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Categorical Feature Counts")
        cat_cols = ['School_Type', 'Location', 'Gender', 'Caste', 'Socioeconomic_Status']
        for col in cat_cols:
            fig = px.bar(df[col].value_counts().reset_index(), x=col, y='count', title=col)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Dropout by Category")
        col = st.selectbox("Select Category", ['School_Type', 'Location', 'Gender', 'Caste'])
        fig = px.bar(df.groupby(col)['Dropout_Status'].value_counts(normalize=True).unstack()*100,
                     barmode='group', title=f"Dropout Rate by {col}")
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("Correlation Heatmap")
        df_enc = df.copy()
        df_enc['Dropout_Encoded'] = df_enc['Dropout_Status'].map({'Dropout':1, 'Enrolled':0})
        corr = df_enc[['Age','Standard','Dropout_Encoded']].corr().round(2)
        fig = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns,
                                        colorscale='RdYlBu_r', text=corr.values, texttemplate="%{text}"))
        st.plotly_chart(fig, use_container_width=True)

# ====================== POWER BI DASHBOARD ======================
elif page == "📊 Power BI Dashboard":
    st.title("SILENT ACADEMIC DROPOUT CRISIS")
    
    # KPI Cards
    k1, k2, k3, k4 = st.columns(4)
    with k1: st.metric("Total Students", f"{len(df):,}", "10K")
    with k2: st.metric("Total Dropouts", (df['Dropout_Status']=='Dropout').sum(), "5996")
    with k3: st.metric("Dropout %", f"{(df['Dropout_Status']=='Dropout').mean()*100:.2f}%", "58.80")
    with k4: st.metric("Avg Age", f"{df['Age'].mean():.2f}", "12.78")
    
    # Dashboard Charts
    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("Dropout % by Infrastructure")
        fig1 = px.pie(df, names='Infrastructure', values=df.groupby('Infrastructure')['Dropout_Status'].apply(lambda x: (x=='Dropout').mean()*100), hole=0.4)
        st.plotly_chart(fig1, use_container_width=True)
    with c2:
        st.subheader("Dropout % by Gender")
        fig2 = px.bar(df.groupby('Gender')['Dropout_Status'].value_counts(normalize=True).unstack()*100, barmode='group')
        st.plotly_chart(fig2, use_container_width=True)
    with c3:
        st.subheader("Dropout % by Teaching Staff")
        fig3 = px.pie(df, names='Teaching_Staff', values=df.groupby('Teaching_Staff')['Dropout_Status'].apply(lambda x: (x=='Dropout').mean()*100))
        st.plotly_chart(fig3, use_container_width=True)
    
    c4, c5, c6 = st.columns(3)
    with c4:
        st.subheader("Dropout % by School Type")
        fig4 = px.bar(df.groupby('School_Type')['Dropout_Status'].value_counts(normalize=True).unstack()*100, barmode='group')
        st.plotly_chart(fig4, use_container_width=True)
    with c5:
        st.subheader("Dropout % by Location")
        fig5 = px.line(df.groupby('Location')['Dropout_Status'].value_counts(normalize=True).unstack()*100)
        st.plotly_chart(fig5, use_container_width=True)
    with c6:
        st.subheader("Dropout % by Socioeconomic Status")
        fig6 = px.bar(df.groupby('Socioeconomic_Status')['Dropout_Status'].value_counts(normalize=True).unstack()*100, barmode='group')
        st.plotly_chart(fig6, use_container_width=True)

# ====================== PREDICT ======================
elif page == "🔮 Predict":
    st.title("🔮 Real-time Dropout Prediction")
    
    col1, col2 = st.columns(2)
    with col1:
        school_type = st.selectbox("School Type", df['School_Type'].unique())
        location = st.selectbox("Location", df['Location'].unique())
        infrastructure = st.selectbox("Infrastructure", df['Infrastructure'].unique())
        teaching_staff = st.selectbox("Teaching Staff", df['Teaching_Staff'].unique())
        gender = st.selectbox("Gender", df['Gender'].unique())
    with col2:
        caste = st.selectbox("Caste", df['Caste'].unique())
        age = st.slider("Age", 10, 16, 14)
        standard = st.slider("Standard", 5, 12, 9)
        ses = st.selectbox("Socioeconomic Status", df['Socioeconomic_Status'].unique())
    
    if st.button("🚀 Predict Now", type="primary"):
        input_df = pd.DataFrame([{
            'School_Type': school_type, 'Location': location, 'Infrastructure': infrastructure,
            'Teaching_Staff': teaching_staff, 'Gender': gender, 'Caste': caste,
            'Age': age, 'Standard': standard, 'Socioeconomic_Status': ses
        }])
        
        pred = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0][1]
        
        st.success("✅ Prediction Complete")
        colA, colB = st.columns(2)
        with colA:
            st.metric("Predicted Status", "🚨 Dropout" if pred == 1 else "✅ Enrolled")
        with colB:
            st.metric("Dropout Probability", f"{proba*100:.1f}%")

st.caption("Streamlit App • Full EDA • Power BI Dashboard • Trained Model")