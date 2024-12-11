from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import dill
import pandas as pd
import numpy as np

# Load the saved user model data
try:
    with open('api_ey.pkl', 'rb') as file:
        user_model_data = dill.load(file)

    # Extract the components from the loaded data
    model = user_model_data['model']
    scaler = user_model_data['scaler']
    label_encoders = user_model_data['label_encoders']
    feature_columns = user_model_data['feature_columns']
    loaded_models = user_model_data['loaded_models']
    loaded_label_encoders = user_model_data['loaded_label_encoders']
    predict_with_model_loaded = user_model_data['predict_with_model_loaded']
    calculate_impact = user_model_data['calculate_impact']
    provide_recommendations = user_model_data['provide_recommendations']
    severity_percentages = user_model_data['severity_percentages']
    impact_hypertension = user_model_data['impact_hypertension']
    impact_diabetes = user_model_data['impact_diabetes']
    impact_heart_disease = user_model_data['impact_heart_disease']
    impact_cancer = user_model_data['impact_cancer']
except Exception as e:
    print(f"Error loading model data: {e}")
    user_model_data = None

# Initialize FastAPI app
app = FastAPI()

# Define the input schema
class ModelInput(BaseModel):
    Age: int
    BMI: float
    Gender: str
    Smoking_Frequency: int
    Alcohol_Consumption_Frequency: int
    Physical_Activity_Level: int
    Sleep_Duration: float
    Disease_Management_Hypertension: int
    Disease_Management_Diabetes: int
    Disease_Management_Heart_Disease: int
    Disease_Management_Cancer: int

# Define the root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the ML Model API!"}

# Define the prediction endpoint
@app.post("/predict")
def predict(input_data: ModelInput):
    if not user_model_data:
        raise HTTPException(status_code=500, detail="Model data not loaded")

    # Convert input data to a DataFrame
    input_dict = input_data.dict()
    input_df = pd.DataFrame([input_dict])

    # Apply label encoding to categorical columns
    for column, encoder in label_encoders.items():
        if column in input_df:
            input_df[column] = encoder.transform(input_df[column])

    # Scale the input data using the scaler
    try:
        scaled_input = scaler.transform(input_df[feature_columns])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error scaling input data: {e}")

    # Use the primary model to make predictions
    try:
        predicted_data = model.predict(scaled_input)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {e}")

    # Use loaded models for disease severity predictions
    try:
        target_columns = ['Disease_Severity_Hypertension', 'Disease_Severity_Diabetes',
                          'Disease_Severity_Heart_Disease', 'Disease_Severity_Cancer']
        severity_results = predict_with_model_loaded(input_dict, loaded_models, loaded_label_encoders, target_columns)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during severity prediction: {e}")

    # Combine results for output
    try:
        input_selected = input_df[['BMI', 'Alcohol_Consumption_Frequency', 'Smoking_Frequency']]
        predicted_data_df = pd.DataFrame(predicted_data, columns=['Predicted_Age'])
        severity_df = pd.DataFrame(severity_results)
        combined_results = pd.concat([input_selected, predicted_data_df, severity_df], axis=1)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error combining results: {e}")

    # Return combined results as JSON
    return combined_results.to_dict(orient="records")
