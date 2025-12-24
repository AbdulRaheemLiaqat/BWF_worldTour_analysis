import streamlit as st
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

MODEL_FILE = "random_forest_badminton.pkl"

if os.path.exists(MODEL_FILE):
    model = joblib.load(MODEL_FILE)
else:
    X_dummy = pd.DataFrame(np.random.randint(0,5,size=(100,6)))
    y_dummy = np.random.randint(0,2,size=100)
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_dummy, y_dummy)

st.title("Badminton Match Winner Prediction - Random Forest")

num_features = model.n_features_in_
features = []
for i in range(num_features):
    features.append(st.number_input(f"Feature {i+1}", value=0))

if st.button("Predict"):
    prediction = model.predict([features])[0]
    if prediction == 1:
        st.success("Predicted Winner: Player 1")
    else:
        st.error("Predicted Winner: Player 2")

st.markdown("<p style='text-align:center;'>Created by Abdul Raheem Liaqat</p>", unsafe_allow_html=True)
