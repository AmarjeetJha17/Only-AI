import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib

st.set_page_config(page_title="Student Dropout Live Demo", page_icon="🎓", layout="wide")

st.title("🎓 Student Dropout Prediction - Live Demo")
st.markdown("**Real-time prediction + Full EDA from your notebook**")

# ====================== LOAD DATA ======================
@st.cache_data
def load_data():
    return pd.read_csv("cleaned_students_dropout01.csv")

df = load_data()

# ====================== LOAD MODEL ======================
@st.cache_resource
def load_model():
    return joblib.load("dropout_model.pkl")   # ← your saved model from prediction_model.ipynb

model = load_model()

# ====================== SIDEBAR ======================
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "📊 Full EDA", "🔮 Predict"])

# ====================== HOME ======================
if page == "🏠 Home":
    st.header("Overview")
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Total Students", len(df))
    with c2: st.metric("Dropout Rate", f"{(df['Dropout_Status']=='Dropout').mean()*100:.1f}%")
    with c3: st.metric("Enrolled", (df['Dropout_Status']=='Enrolled').sum())
    with c4: st.metric("Dropouts", (df['Dropout_Status']=='Dropout').sum())
    
    st.subheader("Sample Data")
    st.dataframe(df.head(10), use_container_width=True)

# ====================== FULL EDA (ALL PLOTS) ======================
elif page == "📊 Full EDA":
    st.header("Full Exploratory Data Analysis")
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Univariate Distributions",
        "📊 Categorical Frequencies",
        "📉 Numerical vs Target",
        "📊 Categorical vs Target",
        "🔥 Correlation & Insights"
    ])
    
    # Tab 1: Univariate
    with tab1:
        st.subheader("Age & Standard Distribution")
        fig_num = make_subplots(rows=2, cols=2, subplot_titles=("Age Histogram", "Age Boxplot", "Standard Histogram", "Standard Boxplot"))
        fig_num.add_trace(go.Histogram(x=df['Age'], marker_color='#1f77b4'), row=1, col=1)
        fig_num.add_trace(go.Box(y=df['Age'], marker_color='#1f77b4'), row=1, col=2)
        fig_num.add_trace(go.Histogram(x=df['Standard'], marker_color='#ff7f0e'), row=2, col=1)
        fig_num.add_trace(go.Box(y=df['Standard'], marker_color='#ff7f0e'), row=2, col=2)
        st.plotly_chart(fig_num, use_container_width=True)
    
    # Tab 2: Categorical Frequencies
    with tab2:
        st.subheader("Categorical Feature Counts")
        cat_cols = ['School_Type', 'Location', 'Gender', 'Caste', 'Socioeconomic_Status', 'Dropout_Status']
        fig_cat = make_subplots(rows=3, cols=2, subplot_titles=[f"{col}" for col in cat_cols])
        for i, col in enumerate(cat_cols):
            counts = df[col].value_counts().reset_index()
            fig_cat.add_trace(go.Bar(x=counts[col], y=counts['count'], name=col), row=i//2+1, col=i%2+1)
        st.plotly_chart(fig_cat, use_container_width=True)
        
        st.subheader("Infrastructure & Teaching Staff")
        fig_extra = make_subplots(rows=1, cols=2)
        fig_extra.add_trace(go.Bar(x=df['Infrastructure'].value_counts().index, y=df['Infrastructure'].value_counts().values), row=1, col=1)
        fig_extra.add_trace(go.Bar(x=df['Teaching_Staff'].value_counts().index, y=df['Teaching_Staff'].value_counts().values), row=1, col=2)
        st.plotly_chart(fig_extra, use_container_width=True)
    
    # Tab 3: Numerical vs Target
    with tab3:
        st.subheader("Age & Standard by Dropout Status")
        fig_box = make_subplots(rows=1, cols=2)
        fig_box.add_trace(go.Box(x=df['Dropout_Status'], y=df['Age'], name='Age'), row=1, col=1)
        fig_box.add_trace(go.Box(x=df['Dropout_Status'], y=df['Standard'], name='Standard'), row=1, col=2)
        st.plotly_chart(fig_box, use_container_width=True)
        
        fig_violin = make_subplots(rows=1, cols=2)
        fig_violin.add_trace(go.Violin(x=df['Dropout_Status'], y=df['Age'], name='Age', box_visible=True), row=1, col=1)
        fig_violin.add_trace(go.Violin(x=df['Dropout_Status'], y=df['Standard'], name='Standard', box_visible=True), row=1, col=2)
        st.plotly_chart(fig_violin, use_container_width=True)
    
    # Tab 4: Categorical vs Target
    with tab4:
        st.subheader("Categorical Features vs Dropout Status")
        cat_vs_target = ['School_Type', 'Location', 'Infrastructure', 'Teaching_Staff', 'Gender', 'Caste']
        for col in cat_vs_target:
            fig = px.bar(df.groupby([col, 'Dropout_Status']).size().reset_index(name='count'),
                         x=col, y='count', color='Dropout_Status', barmode='group')
            st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Socioeconomic Status vs Dropout")
        fig_ses = px.bar(df.groupby(['Socioeconomic_Status', 'Dropout_Status']).size().reset_index(name='count'),
                         x='Socioeconomic_Status', y='count', color='Dropout_Status')
        st.plotly_chart(fig_ses, use_container_width=True)
    
    # Tab 5: Correlation & Key Insights
    with tab5:
        st.subheader("Correlation Heatmap")
        df_encoded = df.copy()
        df_encoded['Dropout_Encoded'] = df_encoded['Dropout_Status'].map({'Dropout':1, 'Enrolled':0})
        corr = df_encoded[['Age','Standard','Dropout_Encoded']].corr().round(2)
        fig_heat = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns,
                                             colorscale='RdYlBu_r', text=corr.values, texttemplate="%{text}"))
        st.plotly_chart(fig_heat, use_container_width=True)
        
        st.subheader("Key Dropout Rates")
        for col in ['School_Type', 'Socioeconomic_Status', 'Gender', 'Caste']:
            st.write(f"**{col}**")
            st.dataframe(df.groupby(col)['Dropout_Status'].value_counts(normalize=True).unstack() * 100)

# ====================== PREDICT ======================
elif page == "🔮 Predict Dropout":
    st.header("🔮 Real-time Dropout Prediction")
    
    col1, col2 = st.columns(2)
    with col1:
        school_type = st.selectbox("School Type", df['School_Type'].unique())
        location = st.selectbox("Location", df['Location'].unique())
        infrastructure = st.selectbox("Infrastructure", df['Infrastructure'].unique())
        teaching_staff = st.selectbox("Teaching Staff", df['Teaching_Staff'].unique())
        gender = st.selectbox("Gender", df['Gender'].unique())
    with col2:
        caste = st.selectbox("Caste", df['Caste'].unique())
        age = st.slider("Age", int(df['Age'].min()), int(df['Age'].max()), 14)
        standard = st.slider("Standard", int(df['Standard'].min()), int(df['Standard'].max()), 9)
        ses = st.selectbox("Socioeconomic Status", df['Socioeconomic_Status'].unique())
    
    if st.button("🚀 Predict Now", type="primary"):
        input_df = pd.DataFrame([{
            'School_Type': school_type, 'Location': location, 'Infrastructure': infrastructure,
            'Teaching_Staff': teaching_staff, 'Gender': gender, 'Caste': caste,
            'Age': age, 'Standard': standard, 'Socioeconomic_Status': ses
        }])
        
        pred = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0][1]
        
        st.success("Prediction Complete!")
        colA, colB = st.columns(2)
        with colA:
            st.metric("Predicted Status", "🚨 Dropout" if pred == 1 else "✅ Enrolled")
        with colB:
            st.metric("Dropout Probability", f"{proba*100:.1f}%")

st.sidebar.success("✅ All EDA plots added")
st.sidebar.caption("Streamlit App • Full EDA • Trained Model")