# lab3_mahesh_sama
# Lab 3: Penguins Classification with XGBoost and FastAPI

## Setup Instructions
# Penguins Classification API
# Video link for Demo
https://github.com/aidi-2004-ai-enterprise/lab3_mahesh_sama/blob/main/lab3.mp4
## Project Description
This project implements a FastAPI application to predict penguin species (Adelie, Gentoo, or Chinstrap) using an XGBoost model trained on the Seaborn penguins dataset. The API accepts input features (`bill_length_mm`, `bill_depth_mm`, `flipper_length_mm`, `body_mass_g`, `sex`, `island`) and returns the predicted species. The application includes:

- **Model Training**: `train.py` preprocesses the dataset, trains an XGBoost model, reports weighted F1-scores for training and test sets, and saves the model to `app/data/model.json`.
- **API**: `app/main.py` defines a `/predict` endpoint that validates inputs, performs consistent one-hot encoding, and returns predictions. Invalid inputs (e.g., negative numeric values, invalid `sex` or `island`) return HTTP 400 errors with clear messages.
- **Logging**: Successful predictions are logged at `INFO` level, and errors are logged at `DEBUG` level.
- **Validation**: Ensures `sex` is `male` or `female` and `island` is `Torgersen`, `Biscoe`, or `Dream`.

## Setup Instructions
To set up and run the project locally, follow these steps:

1. **Install `uv`**:
   ```bash
   pip install uv
uv venv
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate
# Install dependencie
uv pip install fastapi uvicorn pandas scikit-learn xgboost seaborn joblib
# Train the model
python train.py
# Run the FastAPI server
uvicorn app.main:app --reload
# Test the API
EX:
for valid input 200
{
    "bill_length_mm": 39.1,
    "bill_depth_mm": 18.7,
    "flipper_length_mm": 181.0,
    "body_mass_g": 3750.0,
    "sex": "male",
    "island": "Torgersen"
}
for invalid input 400
{
    "bill_length_mm": 39.1,
    "bill_depth_mm": 18.7,
    "flipper_length_mm": 181.0,
    "body_mass_g": 3750.0,
    "sex": "invalid",
    "island": "Torgersen"
}
