# ==========================================
# STREAMLIT GUI FOR TELCO CHURN PREDICTION
# ==========================================
# to run: streamlit run churn_gui_streamlit.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Telco Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
    }
    .churn-high {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
    }
    .churn-low {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
        color: #000000;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# LOAD MODEL
# ==========================================
@st.cache_resource
def load_model():
    """Load the trained model and feature columns"""
    try:
        model = joblib.load("model.pkl")
        feature_columns = joblib.load("columns.pkl")
        return model, feature_columns
    except Exception as e:
        st.error(f"❌ Failed to load model: {str(e)}")
        st.info("Please run 'train_model.py' first to generate model files")
        return None, None

# ==========================================
# PREPROCESSING FUNCTIONS
# ==========================================
def preprocess_data(df):
    """Preprocess data for prediction"""
    df = df.copy()
    
    # Convert SeniorCitizen
    if 'SeniorCitizen' in df.columns:
        if df['SeniorCitizen'].dtype == 'object':
            df['SeniorCitizen'] = df['SeniorCitizen'].map({'Yes': 1, 'No': 0})
        df['SeniorCitizen'] = pd.to_numeric(df['SeniorCitizen'], errors='coerce')
    
    # Convert TotalCharges to numeric
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df['TotalCharges'].fillna(df['MonthlyCharges'] * df['tenure'], inplace=True)
    
    # Create additional features
    if 'TotalCharges' in df.columns and 'tenure' in df.columns:
        df['AvgMonthlySpend'] = df['TotalCharges'] / (df['tenure'] + 1)
    
    services = ['PhoneService', 'OnlineSecurity', 'OnlineBackup', 
               'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    
    available_services = [s for s in services if s in df.columns]
    if available_services:
        df['ServicesCount'] = (df[available_services] == 'Yes').sum(axis=1)
    
    if 'MonthlyCharges' in df.columns:
        threshold = df['MonthlyCharges'].median()
        df['HighValue'] = (df['MonthlyCharges'] > threshold).astype(int)
    
    if 'Contract' in df.columns:
        df['IsLongContract'] = df['Contract'].isin(['One year', 'Two year']).astype(int)
        if 'tenure' in df.columns:
            df['RiskScore'] = (
                (df['Contract'] == 'Month-to-month').astype(int) +
                (df['MonthlyCharges'] > df['MonthlyCharges'].median()).astype(int) +
                (df['tenure'] < 12).astype(int)
            )
    
    return df

# ==========================================
# SIDEBAR - Navigation
# ==========================================
st.sidebar.image("https://img.icons8.com/color/96/000000/artificial-intelligence.png", width=80)
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["🏠 Home", "🔍 Single Prediction", "📁 Batch Prediction", "📈 Analytics", "ℹ️ Model Info"]
)

# Load model
model, feature_columns = load_model()

# ==========================================
# HOME PAGE
# ==========================================
if page == "🏠 Home":
    st.markdown("""
    <div class="main-header">
        <h1>📊 Telco Customer Churn Prediction System</h1>
        <p>AI-Powered Customer Retention Intelligence</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("🎯 Model Accuracy", "74%", "↑ 5%")
    with col2:
        st.metric("🔍 Churn Recall", "79.1%", "↑ 29.4%")
    with col3:
        st.metric("⚡ Optimal Threshold", "0.35", "Optimized")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🚀 Key Features")
        st.markdown("""
        - **Real-time Churn Prediction**
        - **Batch Processing for Large Datasets**
        - **Interactive Analytics Dashboard**
        - **Risk Score Calculation**
        - **Automated Recommendations**
        """)
    
    with col2:
        st.subheader("📊 Business Impact")
        st.markdown("""
        - **79.1%** of churners detected proactively
        - **Reduce customer churn by up to 40%**
        - **Increase customer lifetime value**
        - **Optimize retention campaigns**
        """)
    
    st.markdown("---")
    
    # Quick stats
    st.subheader("📈 Quick Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">📊 Total Customers<br><h2>7,043</h2></div>', 
                   unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card">⚠️ Churned<br><h2>1,869</h2><span>26.5%</span></div>', 
                   unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card">✅ Retained<br><h2>5,174</h2><span>73.5%</span></div>', 
                   unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-card">💵 Avg Monthly Charge<br><h2>$64.76</h2></div>', 
                   unsafe_allow_html=True)

# ==========================================
# SINGLE PREDICTION PAGE
# ==========================================
elif page == "🔍 Single Prediction":
    st.title("🔍 Single Customer Prediction")
    st.markdown("Enter customer details below to predict churn risk")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Personal Information")
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Partner", ["No", "Yes"])
        dependents = st.selectbox("Dependents", ["No", "Yes"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        
        st.subheader("💳 Payment Information")
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
        payment_method = st.selectbox("Payment Method", 
                                     ["Electronic check", "Mailed check", 
                                      "Bank transfer (automatic)", "Credit card (automatic)"])
    
    with col2:
        st.subheader("📞 Services")
        phone_service = st.selectbox("Phone Service", ["No", "Yes"])
        
        if phone_service == "Yes":
            multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes"])
        else:
            multiple_lines = "No phone service"
        
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        
        if internet_service != "No":
            online_security = st.selectbox("Online Security", ["No", "Yes"])
            online_backup = st.selectbox("Online Backup", ["No", "Yes"])
            device_protection = st.selectbox("Device Protection", ["No", "Yes"])
            tech_support = st.selectbox("Tech Support", ["No", "Yes"])
            streaming_tv = st.selectbox("Streaming TV", ["No", "Yes"])
            streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes"])
        else:
            online_security = online_backup = device_protection = tech_support = streaming_tv = streaming_movies = "No internet service"
        
        st.subheader("💰 Charges")
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=70.0)
        total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=500.0)
    
    # Collect all inputs
    input_data = {
        'gender': gender,
        'SeniorCitizen': senior_citizen,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
    }
    
    # Predict button
    if st.button("🎯 Predict Churn Risk", type="primary", use_container_width=True):
        if model is not None:
            # Preprocess
            df_input = pd.DataFrame([input_data])
            df_processed = preprocess_data(df_input)
            
            # Encode
            df_encoded = pd.get_dummies(df_processed, drop_first=True)
            
            # Align columns
            for col in feature_columns:
                if col not in df_encoded.columns:
                    df_encoded[col] = 0
            
            df_encoded = df_encoded[feature_columns]
            
            # Predict
            probability = model.predict_proba(df_encoded)[0][1]
            prediction = "Churn" if probability >= 0.35 else "No Churn"
            risk_level = "High" if probability >= 0.6 else "Medium" if probability >= 0.35 else "Low"
            
            # Display result
            if prediction == "Churn":
                st.markdown(f"""
                <div class="prediction-box churn-high">
                    <h2>⚠️ HIGH CHURN RISK DETECTED!</h2>
                    <h3>Churn Probability: {probability:.1%}</h3>
                    <p>Risk Level: {risk_level}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Recommendations
                with st.expander("📌 Recommendations", expanded=True):
                    st.markdown("""
                    - 🎁 **Offer special discounts** or loyalty rewards
                    - 📞 **Proactive customer support** outreach
                    - 💳 **Suggest longer-term contracts** with benefits
                    - 📱 **Promote additional services** to increase engagement
                    - ⭐ **Request feedback** and address pain points
                    """)
            else:
                st.markdown(f"""
                <div class="prediction-box churn-low">
                    <h2>✅ LOW CHURN RISK</h2>
                    <h3>Churn Probability: {probability:.1%}</h3>
                    <p>Customer is likely to stay</p>
                </div>
                """, unsafe_allow_html=True)
                
                with st.expander("📌 Recommendations", expanded=True):
                    st.markdown("""
                    - 👍 **Continue excellent service** delivery
                    - 🎯 **Consider upsell opportunities**
                    - 📧 **Send satisfaction surveys**
                    - 🏆 **Recognize as loyal customer**
                    """)
            
            # Gauge chart
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = probability * 100,
                title = {'text': "Churn Risk Score"},
                domain = {'x': [0, 1], 'y': [0, 1]},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkred" if probability >= 0.35 else "darkgreen"},
                    'steps': [
                        {'range': [0, 35], 'color': "lightgreen"},
                        {'range': [35, 60], 'color': "yellow"},
                        {'range': [60, 100], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 35
                    }
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

# ==========================================
# BATCH PREDICTION PAGE
# ==========================================
elif page == "📁 Batch Prediction":
    st.title("📁 Batch Prediction")
    st.markdown("Upload a CSV file with multiple customers to predict churn")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success(f"✅ File loaded successfully! {len(df)} customers found")
        
        # Show preview
        with st.expander("📊 Data Preview"):
            st.dataframe(df.head(10))
        
        if st.button("🚀 Run Predictions", type="primary", use_container_width=True):
            if model is not None:
                with st.spinner("Processing predictions..."):
                    # Process data
                    df_processed = preprocess_data(df)
                    df_encoded = pd.get_dummies(df_processed, drop_first=True)
                    
                    # Align columns
                    for col in feature_columns:
                        if col not in df_encoded.columns:
                            df_encoded[col] = 0
                    
                    df_encoded = df_encoded[feature_columns]
                    
                    # Predict
                    probabilities = model.predict_proba(df_encoded)[:, 1]
                    predictions = ["Churn" if p >= 0.35 else "No Churn" for p in probabilities]
                    
                    # Add results
                    df['Churn_Prediction'] = predictions
                    df['Churn_Probability'] = probabilities
                    df['Risk_Level'] = pd.cut(probabilities, 
                                              bins=[0, 0.35, 0.6, 1], 
                                              labels=['Low', 'Medium', 'High'])
                    
                    # Summary statistics
                    st.subheader("📊 Prediction Summary")
                    col1, col2, col3 = st.columns(3)
                    
                    churn_count = (df['Churn_Prediction'] == 'Churn').sum()
                    with col1:
                        st.metric("⚠️ High Risk Customers", churn_count, 
                                 f"{churn_count/len(df)*100:.1f}%")
                    with col2:
                        st.metric("✅ Low Risk Customers", len(df) - churn_count,
                                 f"{(len(df)-churn_count)/len(df)*100:.1f}%")
                    with col3:
                        st.metric("📊 Total Processed", len(df))
                    
                    # Show results
                    st.subheader("📋 Detailed Results")
                    st.dataframe(df)
                    
                    # Download button
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv,
                        file_name=f"churn_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                    
                    # Visualization
                    st.subheader("📈 Risk Distribution")
                    fig = px.pie(df, names='Risk_Level', title='Customer Risk Level Distribution',
                                color='Risk_Level', color_discrete_map={'Low':'green','Medium':'yellow','High':'red'})
                    st.plotly_chart(fig, use_container_width=True)

# ==========================================
# ANALYTICS PAGE
# ==========================================
elif page == "📈 Analytics":
    st.title("📈 Analytics Dashboard")
    
    # Load sample data for visualization
    @st.cache_data
    def load_sample_data():
        return pd.read_csv("Telco-Customer-Churn.csv")
    
    try:
        df = load_sample_data()
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Overall Churn Rate", f"{df['Churn'].mean()*100:.1f}%")
        with col2:
            st.metric("Avg Monthly Charges", f"${df['MonthlyCharges'].mean():.2f}")
        with col3:
            st.metric("Avg Tenure", f"{df['tenure'].mean():.1f} months")
        with col4:
            st.metric("Total Customers", len(df))
        
        # Churn by Contract Type
        st.subheader("📊 Churn Rate by Contract Type")
        contract_churn = df.groupby('Contract')['Churn'].mean().reset_index()
        fig = px.bar(contract_churn, x='Contract', y='Churn', 
                    title='Churn Rate by Contract Type',
                    color='Churn', color_continuous_scale='RdYlGn_r',
                    text=contract_churn['Churn'].apply(lambda x: f'{x*100:.1f}%'))
        fig.update_traces(textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Churn by Payment Method
            st.subheader("💳 Churn by Payment Method")
            payment_churn = df.groupby('PaymentMethod')['Churn'].mean().reset_index()
            fig = px.bar(payment_churn, x='PaymentMethod', y='Churn',
                        title='Churn Rate by Payment Method',
                        color='Churn', color_continuous_scale='RdYlGn_r')
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Churn by Internet Service
            st.subheader("🌐 Churn by Internet Service")
            internet_churn = df.groupby('InternetService')['Churn'].mean().reset_index()
            fig = px.pie(internet_churn, values='Churn', names='InternetService',
                        title='Churn Distribution by Internet Service')
            st.plotly_chart(fig, use_container_width=True)
        
        # Monthly Charges Distribution
        st.subheader("💰 Monthly Charges Distribution by Churn")
        fig = px.box(df, x='Churn', y='MonthlyCharges', 
                    title='Monthly Charges Distribution',
                    labels={'Churn': 'Churn Status', 'MonthlyCharges': 'Monthly Charges ($)'},
                    color='Churn', color_discrete_map={0:'green',1:'red'})
        st.plotly_chart(fig, use_container_width=True)
        
        # Tenure Analysis
        st.subheader("📅 Customer Tenure Analysis")
        df['TenureGroup'] = pd.cut(df['tenure'], bins=[0,12,24,48,72], 
                                   labels=['<1 year', '1-2 years', '2-4 years', '4+ years'])
        tenure_churn = df.groupby('TenureGroup')['Churn'].mean().reset_index()
        fig = px.line(tenure_churn, x='TenureGroup', y='Churn',
                     title='Churn Rate by Tenure',
                     markers=True)
        fig.update_traces(line_color='red', line_width=3)
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Could not load data: {str(e)}")
        st.info("Make sure 'Telco-Customer-Churn.csv' is in the same directory")

# ==========================================
# MODEL INFO PAGE
# ==========================================
elif page == "ℹ️ Model Info":
    st.title("ℹ️ Model Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Model Performance")
        st.markdown("""
        | Metric | Value |
        |--------|--------|
        | **Accuracy** | 74% |
        | **Churn Recall** | 79.1% |
        | **Churn Precision** | 51% |
        | **F1-Score (Churn)** | 0.62 |
        | **Optimal Threshold** | 0.35 |
        """)
        
        st.subheader("🛠️ Techniques Used")
        st.markdown("""
        - ✅ SMOTETomek (Handling Imbalance)
        - ✅ Random Forest Classifier
        - ✅ Hyperparameter Tuning
        - ✅ Threshold Optimization
        - ✅ Feature Engineering
        """)
    
    with col2:
        st.subheader("📊 Feature Importance")
        if model is not None:
            importance_df = pd.DataFrame({
                'Feature': feature_columns[:10],
                'Importance': model.feature_importances_[:10]
            }).sort_values('Importance', ascending=True)
            
            fig = px.bar(importance_df, x='Importance', y='Feature',
                        orientation='h', title='Top 10 Features',
                        color='Importance', color_continuous_scale='Viridis')
            st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("💡 Business Recommendations")
    st.info("""
    - **Target Month-to-month contract customers** with retention offers
    - **Proactively engage customers** with high monthly charges
    - **Focus retention efforts** on customers with less than 12 months tenure
    - **Offer bundled services** to increase switching costs
    - **Implement early warning system** using this model
    """)

st.sidebar.markdown("---")
st.sidebar.info("""
**Developed with ❤️**\n
Machine Learning Model for Customer Churn Prediction\n
© 2024 All Rights Reserved
""")