import streamlit as st
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
st.set_page_config(page_title="BWF World Tour Analysis", layout="wide")
BASE_DIR = Path(__file__).parent
def load_pkl(name):
    with open(BASE_DIR / name, "rb") as f:
        return pickle.load(f)
model = load_pkl("random_forest_badminton.pkl")
st.title("BWF World Tour Analysis")
st.subheader("Prediction")
x1 = st.number_input("Feature 1")
x2 = st.number_input("Feature 2")
if st.button("Predict"):
    pred = model.predict(np.array([[x1, x2]]))
    st.success(pred)
st.markdown("<p style='text-align:center;'>Created by Abdul Raheem Liaqat</p>", unsafe_allow_html=True)




