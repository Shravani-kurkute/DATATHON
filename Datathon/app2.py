import streamlit as st
import joblib
import pandas as pd

# Load the model, encoders, and scaler
@st.cache_resource
def load_model():
    return joblib.load("retail_model.pkl")

@st.cache_resource
def load_encoders():
    return joblib.load("label_encoders.pkl")

@st.cache_resource
def load_scaler():
    return joblib.load("scaler.pkl")

model = load_model()
label_encoders = load_encoders()
scaler = load_scaler()

st.title("Retail Prediction App")
st.write("Upload a CSV file to get predictions.")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Preview of uploaded data:", df.head())

    # Ensure categorical columns are properly encoded
    for col in label_encoders:
        if col in df.columns:
            df[col] = label_encoders[col].transform(df[col])

    # Ensure numerical columns are properly scaled
    df_scaled = scaler.transform(df)

    if st.button("Predict"):
        predictions = model.predict(df_scaled)
        df["Predictions"] = predictions
        st.write("Predictions:", df)
        st.download_button("Download Predictions", df.to_csv(index=False), "predictions.csv")
