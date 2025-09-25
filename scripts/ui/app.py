import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Heart Disease Predictor", layout="centered")
st.title("Heart Disease Risk Predictor")

st.write("Enter patient data and click Predict. If model is missing, run training script first.")

features = {
    "age": (29, 77, 54),
    "sex": (0,1,0),
    "cp": (0,3,1),
    "trestbps": (94,200,130),
    "chol": (126,564,246),
    "fbs": (0,1,0),
    "restecg": (0,2,1),
    "thalach": (71,202,150),
    "exang": (0,1,0),
    "oldpeak": (0.0,6.2,1.0),
    "slope": (0,2,1),
    "ca": (0,4,0),
    "thal": (0,3,1)
}

user = {}
for k,(mn,mx,default) in features.items():
    if isinstance(default, int):
        user[k] = st.number_input(k, min_value=mn, max_value=mx, value=int(default))
    else:
        user[k] = st.number_input(k, min_value=float(mn), max_value=float(mx), value=float(default))

if st.button("Predict"):
    X = pd.DataFrame([user])
    model_path = "models/final_model.pkl"
    if not os.path.exists(model_path):
        st.error("Model not found. Run the training script: python scripts/preprocess_and_train.py")
    else:
        model = joblib.load(model_path)
        pred = model.predict(X)[0]
        prob = model.predict_proba(X)[0][1] if hasattr(model, "predict_proba") else None
        st.write("Prediction (1 = disease, 0 = no disease):", int(pred))
        if prob is not None:
            st.write(f"Predicted probability of disease: {prob:.3f}")
