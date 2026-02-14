from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from src.train import train_model
from src.predict import predict

app = FastAPI()
model = joblib.load("models/model.pkl")
class Features(BaseModel):
    feature_2: float
    feature_13: float
    feature_9: float
    feature_11: float
    feature_18: float

@app.post("/predict")
def _predict(features: Features):
    input_data = [(
        features.feature_2,
        features.feature_13,
        features.feature_9,
        features.feature_11,
        features.feature_18
    )]

    prediction = model.predict(input_data)
    return {"prediction": float(prediction[0])}

@app.post("/train_model")
def train():
    result = train_model()
    return {"result":result}

@app.post("/predict_entiredata")
def predict_entiredata():
        result = predict()
        return {"result":result}