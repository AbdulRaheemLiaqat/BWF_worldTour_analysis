import streamlit as st
import joblib

model = joblib.load("random_forest_badminton.pkl")

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


