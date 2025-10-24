# src/model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib
import os

def train_and_save_model(df):
    # ✅ Select features (X) and target (y)
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    if 'avg_score' not in df.columns:
        df['avg_score'] = df[numeric_cols].mean(axis=1)

    feature_cols = [c for c in numeric_cols if c != 'avg_score']
    X = df[feature_cols]
    y = df['avg_score']

    # ✅ Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # ✅ Train a Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("✅ Model trained successfully!")

    # ✅ Ensure models/ directory exists
    os.makedirs("src/models", exist_ok=True)

    # ✅ Save model
    joblib.dump(model, "src/models/student_performance_model.pkl")
    print("✅ Model saved at 'src/models/student_performance_model.pkl'")

    return model


def load_model():
    """Load model from file"""
    if not os.path.exists("src/models/student_performance_model.pkl"):
        raise FileNotFoundError("Model file not found. Please train the model first using train_and_save_model().")

    model = joblib.load("src/models/student_performance_model.pkl")
    print("✅ Model loaded successfully!")
    return model
