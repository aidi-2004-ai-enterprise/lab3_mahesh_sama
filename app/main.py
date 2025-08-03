#Author:Mahesh Sama
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from enum import Enum
import pandas as pd
import xgboost as xgb
import os
import joblib
import logging

logging.basicConfig(level=logging.INFO)

# Enumerators for API input validation:
class Island(str, Enum):
    Torgersen = "Torgersen"
    Biscoe = "Biscoe"
    Dream = "Dream"

class Sex(str, Enum):
    male = "male"
    female = "female"

class PenguinFeatures(BaseModel):
    bill_length_mm: float
    bill_depth_mm: float
    flipper_length_mm: float
    body_mass_g: float
    sex: Sex
    island: Island

# Load model, encoder, and columns
MODEL_PATH = os.path.join(os.path.dirname(__file__), "data", "model.json")
ENCODER_PATH = os.path.join(os.path.dirname(__file__), "data", "label_encoder.pkl")
COLUMNS_PATH = os.path.join(os.path.dirname(__file__), "data", "columns.pkl")

# Use xgboost method to load the model
model = xgb.XGBClassifier()
model.load_model(MODEL_PATH)

# Use joblib to load the pickled objects
label_encoder = joblib.load(ENCODER_PATH)
columns = joblib.load(COLUMNS_PATH)

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Penguin Predictor Active!"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(features: PenguinFeatures):
    logging.info(f"Received input: {features}")
    input_df = pd.DataFrame([features.dict()])

    # formatting
    input_df["sex"] = input_df["sex"].str.lower()
    input_df["island"] = input_df["island"].str.capitalize()

    # encoding
    input_df = pd.get_dummies(input_df, columns=["sex", "island"])

    # Align with trained model columns
    input_df = input_df.reindex(columns=columns, fill_value=0)

    try:
        prediction = model.predict(input_df)[0]
        species = label_encoder.inverse_transform([int(prediction)])[0]
        logging.info(f"Prediction successful: {species}")
        return {"prediction": species}
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        raise HTTPException(status_code=400, detail="Prediction failed. Check input values.")