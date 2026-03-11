import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set page config
st.set_page_config(page_title="Crop Disease Prediction", layout="wide")
st.title("🌾 Crop Disease Prediction System")

# Load and prepare data
@st.cache_data
def load_model_and_preprocessors():
    # Load the saved model
    with open("dt_best_model.pkl", "rb") as file:
        model = pickle.load(file)
    
    # Load the scaler
    with open("scaler.pkl", "rb") as file:
        scaler = pickle.load(file)
    
    # Load the label encoder
    with open("label_encoder.pkl", "rb") as file:
        label_encoder = pickle.load(file)
    
    # Load the feature columns
    with open("feature_columns.pkl", "rb") as file:
        feature_columns = pickle.load(file)
    
    # Load the dataset for analysis
    df = pd.read_csv("crop_disease_environment_large_dataset_3000.csv")
    
    return model, scaler, label_encoder, feature_columns, df

# Check if model files exist
if os.path.exists("dt_best_model.pkl") and os.path.exists("scaler.pkl") and os.path.exists("label_encoder.pkl") and os.path.exists("feature_columns.pkl"):
    model, scaler, label_encoder, feature_columns, df = load_model_and_preprocessors()
else:
    st.error("Model files not found. Please run main.py first to train the model.")
    st.stop()

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Page:", ["🔮 Predict Disease", "📊 Dataset Analysis", "📈 Model Performance"])

if page == "🔮 Predict Disease":
    st.header("Make Predictions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        crop = st.selectbox("Crop Type:", ["Pepper", "Tomato", "Potato"])
        temperature = st.slider("Temperature (°C):", min_value=15, max_value=35, value=25)
    
    with col2:
        soil_type = st.selectbox("Soil Type:", ["Clay", "Loamy", "Sandy"])
        humidity = st.slider("Humidity (%):", min_value=40, max_value=100, value=70)
    
    with col3:
        soil_ph = st.slider("Soil pH:", min_value=4.5, max_value=8.0, value=6.5, step=0.1)
        # rainfall range based on typical data
        rainfall = st.slider("Rainfall (mm):", min_value=10, max_value=500, value=100)
    
    if st.button("🔍 Predict Disease", use_container_width=True):
        # Create input dataframe with all columns in correct order
        input_data = pd.DataFrame({
            'temperature': [temperature],
            'humidity': [humidity],
            'soil_ph': [soil_ph],
            'rainfall': [rainfall],
            'crop_Pepper': [1 if crop == 'Pepper' else 0],
            'crop_Potato': [1 if crop == 'Potato' else 0],
            'crop_Tomato': [1 if crop == 'Tomato' else 0],
            'soil_type_Clay': [1 if soil_type == 'Clay' else 0],
            'soil_type_Loamy': [1 if soil_type == 'Loamy' else 0],
            'soil_type_Sandy': [1 if soil_type == 'Sandy' else 0],
        })
        
        # Ensure columns are in the same order as during training
        input_data = input_data[feature_columns]
        
        # Scale input
        input_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        # Check if the model has predict_proba
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(input_scaled)[0]
            
            # Display results
            st.success(f"## Predicted Disease: **{label_encoder.classes_[prediction]}**")
            
            st.subheader("Prediction Confidence:")
            confidence_df = pd.DataFrame({
                'Disease': label_encoder.classes_,
                'Probability': probabilities
            }).sort_values('Probability', ascending=False)
            
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.barh(confidence_df['Disease'], confidence_df['Probability'], color='steelblue')
            ax.set_xlabel('Confidence Score')
            ax.set_title('Prediction Probabilities')
            st.pyplot(fig)
        else:
            st.success(f"## Predicted Disease: **{label_encoder.classes_[prediction]}**")

elif page == "📊 Dataset Analysis":
    st.header("Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Samples", df.shape[0])
    col2.metric("Features", df.shape[1])
    col3.metric("Missing Values", df.isnull().sum().sum())
    col4.metric("Duplicate Rows", df.duplicated().sum())
    
    st.subheader("Dataset Head:")
    st.dataframe(df.head(10))
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Disease Distribution")
        fig, ax = plt.subplots()
        df['disease'].value_counts().plot(kind='barh', ax=ax, color='steelblue')
        ax.set_xlabel('Count')
        st.pyplot(fig)
    
    with col2:
        st.subheader("Crop Distribution")
        fig, ax = plt.subplots()
        df['crop'].value_counts().plot(kind='pie', ax=ax, autopct='%1.1f%%')
        st.pyplot(fig)
    
    st.subheader("Feature Statistics:")
    st.dataframe(df.describe())

elif page == "📈 Model Performance":
    st.header("Model Evaluation")
    
    # Get predictions on entire dataset
    x = df.drop('disease', axis=1)
    y = df['disease']
    x_encoded = pd.get_dummies(x, columns=["crop", "soil_type"])
    
    # Reindex to ensure same columns as training
    x_encoded = x_encoded.reindex(columns=feature_columns, fill_value=0)
    
    y_encoded = label_encoder.transform(y)
    x_scaled = scaler.transform(x_encoded)
    y_pred = model.predict(x_scaled)
    
    accuracy = accuracy_score(y_encoded, y_pred)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Overall Accuracy", f"{accuracy:.2%}")
    col2.metric("Total Samples", x.shape[0])
    col3.metric("Number of Classes", len(label_encoder.classes_))
    
    st.subheader("Classification Report:")
    report = classification_report(y_encoded, y_pred, target_names=label_encoder.classes_, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df)
    
    if hasattr(model, 'feature_importances_'):
        st.subheader("Feature Importance:")
        feature_imp = pd.DataFrame({
            'Feature': feature_columns,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False).head(10)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(feature_imp['Feature'], feature_imp['Importance'], color='coral')
        ax.set_xlabel('Importance')
        ax.set_title('Top 10 Most Important Features')
        st.pyplot(fig)

st.sidebar.markdown("---")
st.sidebar.info("🌾 Crop Disease Prediction using ML Model")
