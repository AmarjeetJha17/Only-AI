import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import shap
import numpy as np

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

    # Feature Importance Section
    st.subheader("🎯 Top Features Influencing Dropout")
    feature_importance = {
        'Feature': [
            'Socioeconomic Status',
            'School Type (Government)',
            'School Type (Private)',
            'Standard',
            'Infrastructure'
        ],
        'Importance': [0.1764, 0.1117, 0.0774, -0.0183, -0.0177],
        'Type': [
            'ordinal__Socioeconomic_Status',
            'onehot__School_Type_Government',
            'onehot__School_Type_Private',
            'num__Standard',
            'ordinal__Infrastructure'
        ]
    }
    
    imp_df = pd.DataFrame(feature_importance)
    imp_df['Impact'] = imp_df['Importance'].apply(
        lambda x: '🔴 Increases Dropout Risk' if x > 0 else '🟢 Reduces Dropout Risk'
    )
    imp_df['Abs_Importance'] = imp_df['Importance'].abs()
    imp_df = imp_df.sort_values('Abs_Importance', ascending=False)
    
    col_a, col_b = st.columns([2, 1])
    with col_a:
        fig_imp = px.bar(
            imp_df, 
            x='Importance', 
            y='Feature', 
            orientation='h',
            color='Importance',
            color_continuous_scale=['green', 'yellow', 'red'],
            color_continuous_midpoint=0,
            title='Feature Importance (Coefficients)',
            labels={'Importance': 'Coefficient Value', 'Feature': 'Feature Name'}
        )
        fig_imp.update_layout(yaxis={'categoryorder': 'total ascending'}, height=300)
        st.plotly_chart(fig_imp, use_container_width=True)
    
    with col_b:
        st.dataframe(
            imp_df[['Feature', 'Importance', 'Impact']].round(4),
            use_container_width=True,
            hide_index=True
        )

    st.subheader("Sample Data")
    st.dataframe(df.head(10), use_container_width=True)

# ====================== FULL EDA ======================
elif page == "📊 Full EDA":
    st.title("📊 Full Exploratory Data Analysis")
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Distributions", "Categorical", "vs Target", "Correlation", "Feature Importance"])
    
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
    
    with tab5:
        st.subheader("🎯 Model Feature Importance")
        st.markdown("**These are the most influential features in predicting student dropout:**")
        
        feature_importance = {
            'Feature': [
                'Socioeconomic Status',
                'School Type (Government)',
                'School Type (Private)',
                'Standard',
                'Infrastructure'
            ],
            'Coefficient': [0.1764, 0.1117, 0.0774, -0.0183, -0.0177],
            'Encoded_Name': [
                'ordinal__Socioeconomic_Status',
                'onehot__School_Type_Government',
                'onehot__School_Type_Private',
                'num__Standard',
                'ordinal__Infrastructure'
            ]
        }
        
        imp_df = pd.DataFrame(feature_importance)
        imp_df['Impact'] = imp_df['Coefficient'].apply(
            lambda x: '🔴 Increases Dropout Risk' if x > 0 else '🟢 Reduces Dropout Risk'
        )
        imp_df['Abs_Coefficient'] = imp_df['Coefficient'].abs()
        imp_df_sorted = imp_df.sort_values('Abs_Coefficient', ascending=False)
        
        # Bar chart
        fig_imp = px.bar(
            imp_df_sorted, 
            x='Coefficient', 
            y='Feature', 
            orientation='h',
            color='Coefficient',
            color_continuous_scale=['green', 'yellow', 'red'],
            color_continuous_midpoint=0,
            title='Feature Coefficients from Trained Model',
            labels={'Coefficient': 'Coefficient Value', 'Feature': 'Feature Name'},
            text='Coefficient'
        )
        fig_imp.update_traces(texttemplate='%{text:.4f}', textposition='outside')
        fig_imp.update_layout(yaxis={'categoryorder': 'total ascending'}, height=400)
        st.plotly_chart(fig_imp, use_container_width=True)
        
        # Data table
        st.dataframe(
            imp_df_sorted[['Feature', 'Coefficient', 'Impact', 'Encoded_Name']].reset_index(drop=True),
            use_container_width=True
        )
        
        st.info("""
        **Interpretation:**
        - **Positive coefficients** (red bars): Higher values increase dropout probability
        - **Negative coefficients** (green bars): Higher values decrease dropout probability
        - **Socioeconomic Status** has the strongest influence on dropout predictions
        """)
        

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
        proba_values = model.predict_proba(input_df)[0]
        class_probability = dict(zip(model.classes_, proba_values))
        dropout_probability = class_probability.get('Dropout', max(proba_values))
        
        st.success("✅ Prediction Complete")
        colA, colB = st.columns(2)
        with colA:
            st.metric("Predicted Status", f"🚨 {pred}" if pred == 'Dropout' else f"✅ {pred}")
        with colB:
            st.metric("Dropout Probability", f"{dropout_probability*100:.1f}%")
        
        # SHAP Feature Importance
        st.subheader("📊 Top Contributing Factors")
        try:
            # Get preprocessor and classifier from pipeline
            preprocessor = model.named_steps['preprocessor']
            classifier = model.named_steps['classifier']
            
            # Transform input and get feature names
            X_transformed = preprocessor.transform(input_df)
            feature_names = preprocessor.get_feature_names_out()
            
            # Calculate SHAP values
            explainer = shap.TreeExplainer(classifier)
            shap_values = explainer.shap_values(X_transformed)
            
            # Handle different SHAP output formats
            if isinstance(shap_values, list):
                vals = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
            else:
                if shap_values.ndim == 3:
                    class_idx = 1 if shap_values.shape[2] > 1 else 0
                    vals = shap_values[0, :, class_idx]
                else:
                    vals = shap_values[0]
            
            # Create contribution dict and sort
            contrib = dict(zip(feature_names, vals))
            sorted_contrib = sorted(contrib.items(), key=lambda x: abs(float(x[1])), reverse=True)[:5]
            
            # Display as a styled table
            contrib_df = pd.DataFrame(sorted_contrib, columns=['Feature', 'Contribution'])
            contrib_df['Impact'] = contrib_df['Contribution'].apply(
                lambda x: '🔴 Increases Risk' if x > 0 else '🟢 Decreases Risk'
            )
            contrib_df['Contribution'] = contrib_df['Contribution'].round(4)
            st.dataframe(contrib_df, use_container_width=True, hide_index=True)
            
            # Bar chart visualization
            fig = px.bar(
                contrib_df, x='Contribution', y='Feature', orientation='h',
                color='Contribution', color_continuous_scale=['green', 'red'],
                title='Feature Contribution to Dropout Prediction'
            )
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not compute feature importance: {e}")

st.caption("Streamlit App • Full EDA • Power BI Dashboard • Trained Model")