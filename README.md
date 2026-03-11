🌾 Crop Disease Prediction System
📌 Project Overview

This project is a Machine Learning-based Crop Disease Prediction System that predicts possible crop diseases based on environmental conditions such as temperature, humidity, soil pH, rainfall, crop type, and soil type.

The system uses a Decision Tree Machine Learning model and provides an interactive Streamlit web interface for predictions, dataset analysis, and model performance visualization.

🚀 Features

🌱 Predict crop diseases using environmental data

📊 Dataset analysis and visualization

📈 Model performance evaluation

🖥️ Interactive Streamlit web interface

🤖 Machine Learning model training and prediction

🧠 Machine Learning Model

The project uses:

Decision Tree Classifier

StandardScaler for feature scaling

LabelEncoder for encoding disease labels

One-hot encoding for categorical features

📂 Project Structure
Crop-Disease-Prediction
│
├── main.py                         # Model training script
├── streamlit_app.py                # Streamlit web application
├── crop_disease_environment_large_dataset_3000.csv
├── dt_best_model.pkl               # Trained ML model
├── scaler.pkl                      # Feature scaler
├── label_encoder.pkl               # Label encoder
├── feature_columns.pkl             # Feature column structure
└── README.md
⚙️ Installation

Clone the repository

git clone https://github.com/your-username/crop-disease-prediction.git
cd crop-disease-prediction

Install required libraries

pip install pandas numpy scikit-learn matplotlib seaborn streamlit
▶️ How to Run the Project
Step 1: Train the Model

Run the training script

python main.py

This will generate:

dt_best_model.pkl

scaler.pkl

label_encoder.pkl

feature_columns.pkl

Step 2: Run the Streamlit App
streamlit run streamlit_app.py

Then open the browser at:

http://localhost:8501
📊 Dataset Features

The dataset contains the following features:

Crop Type

Soil Type

Temperature

Humidity

Soil pH

Rainfall

Disease (Target Variable)

📈 Application Pages
🔮 Predict Disease

Allows users to input environmental conditions and predict crop disease.

📊 Dataset Analysis

Shows dataset statistics, distributions, and summaries.

📈 Model Performance

Displays accuracy, classification report, and feature importance.

🛠️ Technologies Used

Python

Scikit-learn

Pandas

NumPy

Matplotlib

Seaborn

Streamlit

🎯 Future Improvements

Use advanced ML models like Random Forest / XGBoost

Add image-based disease detection

Deploy the application on Streamlit Cloud or AWS

Expand dataset with more crops and diseases

