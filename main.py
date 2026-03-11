import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os

# Load dataset
dataset_path = "crop_disease_environment_large_dataset_3000.csv"
if not os.path.exists(dataset_path):
    print(f"Error: {dataset_path} not found.")
    exit(1)

df = pd.read_csv(dataset_path)

print("Dataset Info:")
print(f"Shape: {df.shape}")
print(f"\nFirst few rows:")
print(df.head())
print(f"\nMissing values: {df.isnull().sum().sum()}")
print(f"Duplicates: {df.duplicated().sum()}")

# Separate features and target
x = df.drop('disease', axis=1)
y = df['disease']

# Encode categorical features (one-hot encoding)
x_encoded = pd.get_dummies(x, columns=["crop", "soil_type"])

# Encode target variable (label encoding)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(
    x_encoded, y_encoded, test_size=0.2, random_state=42
)

print(f"\nTraining set size: {x_train.shape}")
print(f"Test set size: {x_test.shape}")

# Scale features
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Build and train Decision Tree model
model = DecisionTreeClassifier(random_state=42)
model.fit(x_train_scaled, y_train)

# Make predictions
y_pred = model.predict(x_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"\n==================================================")
print(f"Model Accuracy: {accuracy:.4f}")
print(f"==================================================\n")

print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Save the model and preprocessors
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

with open("feature_columns.pkl", "wb") as f:
    pickle.dump(list(x_encoded.columns), f)

with open("dt_best_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("\nModel and preprocessors saved successfully.")
