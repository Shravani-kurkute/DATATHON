from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator
import joblib
import pandas as pd
import numpy as np

# Load the saved model, encoders, and scaler
try:
    model = joblib.load("retail_model2.pkl")
    label_encoders = joblib.load("label_encoders2.pkl")
    scaler = joblib.load("scaler2.pkl")
except Exception as e:
    raise RuntimeError(f"Error loading model files: {e}")

# Initialize FastAPI app
app = FastAPI()

# Define input structure
class CustomerData(BaseModel):
    Age: int
    Gender: str
    Item_Purchased: str
    Category: str
    Purchase_Amount_USD: float
    Size: str
    Color: str
    Season: str
    Review_Rating: float
    Subscription_Status: str
    Payment_Method: str
    Shipping_Type: str
    Discount_Applied: str
    Promo_Code_Used: str
    Previous_Purchases: int
    Preferred_Payment_Method: str
    Frequency_of_Purchases: str

    # Validators
    @validator('Age', 'Previous_Purchases')
    def must_be_positive(cls, v):
        if v < 0:
            raise ValueError(f"{cls.__name__} must be a positive integer")
        return v

    @validator('Purchase_Amount_USD')
    def purchase_amount_positive(cls, v):
        if v < 0:
            raise ValueError("Purchase Amount must be positive")
        return v

    @validator('Review_Rating')
    def rating_range(cls, v):
        if not (1 <= v <= 5):
            raise ValueError("Review Rating must be between 1 and 5")
        return v

# Home endpoint
@app.get("/")
def home():
    return {"message": "Retail Outlet Prediction API is running!"}

# Prediction endpoint
@app.post("/predict/")
def predict(customer: CustomerData):
    try:
        # Convert input to DataFrame
        input_df = pd.DataFrame([customer.dict()])

        # Encode categorical variables safely
        for col, encoder in label_encoders.items():
            if col in input_df.columns:
                input_df[col] = input_df[col].map(lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1)

        # Ensure numerical data is scaled properly
        input_scaled = scaler.transform(input_df)

        # Make prediction
        prediction = model.predict(input_scaled)

        # Decode predicted label
        predicted_outlet = label_encoders["Location"].inverse_transform(prediction)

        return {"Predicted Retail Outlet": predicted_outlet[0]}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
