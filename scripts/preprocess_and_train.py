"""
Preprocess data, train RandomForest, save cleaned csv and model.
Run: python scripts/preprocess_and_train.py
"""
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import joblib

# try to download data if not present
if not os.path.exists("data/heart_disease.csv"):
    try:
        from data.download_data import download_and_prepare
        download_and_prepare("data/heart_disease.csv")
    except Exception as e:
        print("Could not download data automatically. Please put heart_disease.csv into data/ and rerun.")
        raise e

df = pd.read_csv("data/heart_disease.csv")
print("Loaded data shape:", df.shape)

# simple preprocessing
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
num_cols = [c for c in num_cols if c != "target"]
imputer = SimpleImputer(strategy="median")
df[num_cols] = imputer.fit_transform(df[num_cols])

scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

os.makedirs("data", exist_ok=True)
df.to_csv("data/heart_disease_cleaned.csv", index=False)
print("Saved cleaned data to data/heart_disease_cleaned.csv")

# train model
X = df.drop(columns=["target"])
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

os.makedirs("models", exist_ok=True)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
joblib.dump(clf, "models/final_model.pkl")
print("Saved model to models/final_model.pkl")

# evaluate
preds = clf.predict(X_test)
probs = clf.predict_proba(X_test)[:,1] if hasattr(clf, "predict_proba") else None
print("Classification report:")
print(classification_report(y_test, preds))
if probs is not None:
    try:
        auc = roc_auc_score(y_test, probs)
        print("AUC:", auc)
    except:
        pass
