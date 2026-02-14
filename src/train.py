import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Lasso
import os

def train_model():
    df = pd.read_csv("data/training_data.csv")

    X = df[['feature_2', 'feature_13', 'feature_9', 'feature_11', 'feature_18']]
    y = df['target']

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('poly', PolynomialFeatures(degree=3, include_bias=False)),
        ('model', Lasso(alpha=0.03, max_iter=20000))
    ])

    pipeline.fit(X, y)
    os.makedirs("models", exist_ok=True)
    joblib.dump(pipeline, "models/model.pkl")

    print("Model saved successfully.")
    return "Model saved on models/model.pkl"