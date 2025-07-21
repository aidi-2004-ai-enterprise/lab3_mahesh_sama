from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ValidationError
from enum import Enum
import pandas as pd
import joblib
import logging
from typing import Dict
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

app = FastAPI()

class Island(str, Enum):
    Torgersen = "Torgersen"
    Biscoe = "Biscoe"
    Dream = "Dream"

class Sex(str, Enum):
    Male = "male"
    Female = "female"

class PenguinFeatures(BaseModel):
    bill_length_mm: float
    bill_depth_mm: float
    flipper_length_mm: float
    body_mass_g: float
    sex: Sex
    island: Island

def load_model(path: str) -> Dict:
    """Load the trained model and label encoder."""
    if not os.path.exists(path):
        logger.error(f"Model file not found at {path}")
        raise FileNotFoundError(f"Model file not found at {path}")
    logger.info(f"Loading model from {path}")
    return joblib.load(path)

# Load model and label encoder at startup
model_data = load_model("app/data/model.json")
model = model_data["model"]
label_encoder = model_data["label_encoder"]

def preprocess_input(data: PenguinFeatures) -> pd.DataFrame:
    """Preprocess input data with consistent one-hot encoding."""
    # Convert input to DataFrame
    input_dict = data.dict()
    df = pd.DataFrame([input_dict])
    
    # One-hot encode categorical features with explicit prefixes
    df = pd.get_dummies(df, columns=["sex", "island"], prefix={"sex": "sex", "island": "island"}, dtype=int)
    
    # Ensure all expected columns are present in the correct order
    expected_columns = [
        "bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g",
        "sex_female", "sex_male", "island_biscoe", "island_dream", "island_torgersen"
    ]
    for col in expected_columns:
        if col not in df.columns:
            df[col] = 0
    
    # Reorder columns to match training
    df = df[expected_columns]
    return df

@app.post("/predict")
async def predict(data: PenguinFeatures) -> Dict:
    """Predict penguin species from input features."""
    try:
        logger.info("Received prediction request")
        
        # Validate numeric inputs to prevent unrealistic values
        if any(val <= 0 for val in [
            data.bill_length_mm, data.bill_depth_mm, data.flipper_length_mm, data.body_mass_g
        ]):
            logger.debug("Invalid input: Numeric features must be positive")
            raise ValueError("Numeric features must be positive")
        
        # Preprocess input
        X = preprocess_input(data)
        
        # Make prediction
        prediction = model.predict(X)[0]
        species = label_encoder.inverse_transform([prediction])[0]
        
        logger.info(f"Prediction successful: {species}")
        return {"species": species}
    
    except ValidationError as e:
        # Return HTTP 400 for Pydantic validation errors
        error_msg = f"Invalid input: {str(e)}"
        logger.debug(error_msg)
        raise HTTPException(status_code=400, detail=error_msg)
    except Exception as e:
        logger.debug(f"Invalid input: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Invalid input: {str(e)}")

@app.get("/")
async def root() -> Dict:
    """Root endpoint for health check."""
    return {"message": "Penguins Classification API"}