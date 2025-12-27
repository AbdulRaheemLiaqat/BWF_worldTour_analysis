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
bwf_df = load_pkl("bwf_df.pkl")
summary_stats = load_pkl("summary_stats.pkl")
model = load_pkl("model.pkl")
st.title("BWF World Tour Analysis")
st.dataframe(bwf_df.head())
st.sidebar.header("Filters")
if "country" in bwf_df.columns:
    countries = st.sidebar.multiselect(
        "Country",
        bwf_df["country"].unique(),
        bwf_df["country"].unique()
    )
    df = bwf_df[bwf_df["country"].isin(countries)]
else:
    df = bwf_df
if "player" in df.columns and "points" in df.columns:
    top_players = (
        df.groupby("player")["points"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )
    st.bar_chart(top_players)
if isinstance(summary_stats, dict):
    for k, v in summary_stats.items():
        st.subheader(k)
        if isinstance(v, pd.DataFrame):
            st.dataframe(v)
        else:
            st.write(v)
st.subheader("Prediction")
x1 = st.number_input("Feature 1")
x2 = st.number_input("Feature 2")
if st.button("Predict"):
    pred = model.predict(np.array([[x1, x2]]))
    st.success(pred)
st.markdown("<p style='text-align:center;'>Created by Abdul Raheem Liaqat</p>", unsafe_allow_html=True)



