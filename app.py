import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware



# Create the FastAPI app
app = FastAPI()

origins = ['*']

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Stock Price Prediction API"}

# Define the prediction endpoint
@app.post("/api/predict_stock")
async def predict_stock_data(data:dict):
    try:
        with open("predict_talha.pkl", "rb") as model_file:
            model = joblib.load(model_file)

        Open_stock=data.get('Open_stock')
        
        Low_stock= data.get('Low_stock')
        High_stock=data.get('High_stock')
        Volume_stock=data.get('Volume_stock')
        Year=data.get('Year')
        Month=data.get('Month')
        Day=data.get('Day')

        inputs={
            'Open_stock' :Open_stock,
            'Low_stock': Low_stock,
            'High_stock':High_stock,
            'Volume_stock':Volume_stock,
            'Year':Year,
            'Month':Month,
            'Day':Day,
        }

        
        # # # Create a DataFrame from the input data
        input_data = pd.DataFrame([inputs])

        # # # Make predictions using the trained model
        predictions = model.predict(input_data)

        return {"prediction": predictions[0][0]}
        

    except Exception as e:
        return {"error": str(e)}


