import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("matches.csv")
features = [c for c in ["tournament","round","event","discipline","player1","player2"] if c in df.columns]
df = df[features + ["winner"]].dropna()

df["label"] = (df["winner"] == df["player1"]).astype(int)

le = LabelEncoder()
for col in features:
    df[col] = le.fit_transform(df[col])

X = df[features]
y = df["label"]

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

st.title("Badminton Match Winner Prediction - Random Forest")

input_data = []
for col in features:
    val = st.number_input(f"{col}", min_value=0, max_value=int(df[col].max()), value=0)
    input_data.append(val)

if st.button("Predict"):
    pred = model.predict([input_data])[0]
    if pred == 1:
        st.success("Predicted Winner: Player 1")
    else:
        st.error("Predicted Winner: Player 2")

st.markdown("<p style='text-align:center;'>Created by Abdul Raheem Liaqat</p>", unsafe_allow_html=True)
