# Heart Disease ML Pipeline (UCI) - Graduation Project

This repository contains an end-to-end ML pipeline for the Heart Disease UCI dataset.

How to run locally:
1. Create venv and install:
   Windows:
     python -m venv venv
     venv\Scripts\activate
     pip install -r requirements.txt
   Linux/Mac:
     python3 -m venv venv
     source venv/bin/activate
     pip install -r requirements.txt

2. Optional: download dataset:
   python data/download_data.py

3. Train model:
   python scripts/preprocess_and_train.py

4. Run Streamlit UI:
   streamlit run ui/app.py

Note: After training a file `models/final_model.pkl` will be created.
