"""
Download and prepare Heart Disease (Cleveland) dataset.
Saves CSV to data/heart_disease.csv
"""
import os
import pandas as pd

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

def download_and_prepare(out_path="data/heart_disease.csv"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cols = ["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal","target"]
    try:
        df = pd.read_csv(URL, header=None, names=cols, na_values='?')
        df['target'] = df['target'].apply(lambda x: 1 if x>0 else 0)
        df.to_csv(out_path, index=False)
        print(f"Saved dataset to {out_path}")
    except Exception as e:
        print("Failed to download dataset. Error:", e)
        print("If download fails, please download manually from UCI and save as data/heart_disease.csv")

if __name__ == "__main__":
    download_and_prepare()
