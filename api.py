from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np

# Load the model
try:
    with open("model.pkl", "rb") as file:
        model = pickle.load(file)
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Initialize FastAPI app
app = FastAPI()

# Define the input schema
class ModelInput(BaseModel):
    feature1: float
    feature2: float
    feature3: float
    # Add all required features here

# Define the root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the ML Model API!"}

# Define the prediction endpoint
@app.post("/predict")
def predict(input_data: ModelInput):
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")

    # Convert input data into a NumPy array for the model
    input_array = np.array([[input_data.feature1, input_data.feature2, input_data.feature3]])

    try:
        # Get the prediction
        prediction = model.predict(input_array)
        return {"prediction": prediction.tolist()}  # Convert to list if it's a NumPy array
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")