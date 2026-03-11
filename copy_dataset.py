import shutil
import os

source = r"f:\OneDrive\Desktop\ml.project_crop\crop_disease_environment_large_dataset_3000.csv"
dest = r"c:\Users\konda\Downloads\crop_predict\crop_disease_environment_large_dataset_3000.csv"

try:
    shutil.copy2(source, dest)
    print("Dataset copied successfully.")
except Exception as e:
    print(f"Error copying dataset: {e}")
