# src/app.py
import streamlit as st
from data_loader import load_data
from eda import perform_eda
from model import train_and_save_model
import pandas as pd

st.title("Student Performance Analyzer")

# Step 1: Load Data
df = load_data()

# Step 2: Display EDA
st.header("Exploratory Data Analysis")
perform_eda(df)

# Step 3: Predict
st.header("Predict Student Performance")

# Train model once (or you can save/load the model instead)
model = train_and_save_model(df)

st.write("Enter student details below:")

# Create input fields only for features used in the model
model_features = ['math score', 'reading score', 'writing score'] 
feature_values = {}

for col in model_features:
    feature_values[col] = st.number_input(
        f"{col}", 
        float(df[col].min()), 
        float(df[col].max()), 
        float(df[col].mean())
    )

if st.button("Predict"):
    input_df = pd.DataFrame([feature_values])
    # Make sure avg_score is not included
    if 'avg_score' in input_df.columns:
        input_df = input_df.drop(columns=['avg_score'])
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted average score: {prediction:.2f}")
