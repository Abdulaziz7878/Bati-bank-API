import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# Load the trained machine learning model
model_filename = "./RandomForest_model_2024-10-07-14-17-17-752.pkl"

model = joblib.load(model_filename)

# Create a FastAPI instance
app = FastAPI()

# Define the input data model
class InputData(BaseModel):
    Recency_WoE: float
    Frequency_WoE: float
    Monetary_WoE: float
    Stability_WoE: float

@app.get("/")
async def root():
    return {
        "message": "To use this API to pridict a credit score make a post request to /predict with a body format of",
        format:{
        "Recency_WoE": float,
        "Frequency_WoE": float,
        "Monetary_WoE": float,
        "Stability_WoE": float
    }}

# Define the prediction endpoint
@app.post("/predict")
async def predict(data: InputData):
    # Prepare the input data for prediction
    input_data = pd.DataFrame({
        "Recency_WoE": [data.Recency_WoE],
        "Frequency_WoE": [data.Frequency_WoE],
        "Monetary_WoE": [data.Monetary_WoE],
        "Stability_WoE": [data.Stability_WoE]
    })
    
    prediction = model.predict(input_data)
    return {"prediction": int(prediction[0])}  # Return the prediction

# To run the app, use the command: uvicorn app:app --reload
# Replace 'app' with the name of your Python file (without .py)