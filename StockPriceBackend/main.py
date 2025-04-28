from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Optional, List
import pandas as pd
import io
import os
import uvicorn
from pydantic import BaseModel

from model import StockPredictionModel
from utils import add_technical_indicators, validate_csv, extract_market_regime

app = FastAPI(title="Stock Price Prediction API")

# Set up CORS to allow requests from the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with the actual frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create a models directory if it doesn't exist
os.makedirs("models", exist_ok=True)
os.makedirs("uploads", exist_ok=True)

@app.get("/")
async def root():
    return {"message": "Welcome to Stock Price Prediction API"}

@app.post("/upload-csv/")
async def upload_csv(file: UploadFile = File(...)):
    """Upload a CSV file with stock data"""
    # Check if the file is a CSV
    if not file.filename.lower().endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be a CSV")
    
    # Read the content of the file
    contents = await file.read()
    
    try:
        # Parse the CSV
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Validate that it has the required columns
        if not validate_csv(df):
            raise HTTPException(
                status_code=400, 
                detail="CSV must contain the following columns: Date, Open, High, Low, Close, Volume"
            )
        
        # Save the file for future use
        file_path = f"uploads/{file.filename}"
        with open(file_path, "wb") as f:
            f.write(contents)
        
        # Return basic info about the uploaded data
        return {
            "filename": file.filename,
            "rows": len(df),
            "columns": df.columns.tolist(),
            "first_date": df['Date'].iloc[0] if 'Date' in df.columns else None,
            "last_date": df['Date'].iloc[-1] if 'Date' in df.columns else None,
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing the CSV: {str(e)}")

@app.post("/predict/")
async def predict(
    filename: str = Form(...),
    add_indicators: bool = Form(True),
    epochs: int = Form(10),
    batch_size: int = Form(256)
):
    """Train a model and make predictions on uploaded data"""
    try:
        # Check if the file exists
        file_path = f"uploads/{filename}"
        print(f"[PREDICT] Processing file: {file_path}")
        if not os.path.exists(file_path):
            print(f"[PREDICT] File not found: {file_path}")
            raise HTTPException(status_code=404, detail=f"File {filename} not found")
        
        # Load the data
        print(f"[PREDICT] Loading data from file")
        df = pd.read_csv(file_path)
        print(f"[PREDICT] Data loaded successfully, shape: {df.shape}")
        
        # Add technical indicators if requested
        if add_indicators:
            print(f"[PREDICT] Adding technical indicators")
            df = add_technical_indicators(df)
            print(f"[PREDICT] Technical indicators added, new shape: {df.shape}")
        
        # Initialize and train the model
        print(f"[PREDICT] Initializing model")
        model = StockPredictionModel(model_version="dynamic")
        
        print(f"[PREDICT] Starting model training with epochs={epochs}, batch_size={batch_size}")
        try:
            training_results = model.train(
                df, 
                with_indicators=add_indicators,
                epochs=epochs,
                batch_size=batch_size
            )
            print(f"[PREDICT] Model training completed successfully")
        except Exception as train_error:
            print(f"[PREDICT] Error during model training: {str(train_error)}")
            import traceback
            print(f"[PREDICT] Traceback: {traceback.format_exc()}")
            raise Exception(f"Model training failed: {str(train_error)}")
        
        # Return the results
        print(f"[PREDICT] Returning results")
        return {
            "message": "Model trained successfully",
            "filename": filename,
            "with_indicators": add_indicators,
            "loss": training_results['loss'],
            "metrics": training_results['metrics'],
            "plot": training_results['plot']
        }
    
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        print(f"[PREDICT] ERROR: {str(e)}")
        print(f"[PREDICT] Traceback: {error_traceback}")
        raise HTTPException(status_code=500, detail=f"Error training model: {str(e)}\n{error_traceback}")

@app.post("/forecast/")
async def forecast(
    filename: str = Form(...),
    add_indicators: bool = Form(True)
):
    """Make predictions with a pre-trained model"""
    try:
        # Check if the file exists
        file_path = f"uploads/{filename}"
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"File {filename} not found")
        
        # Load the data
        df = pd.read_csv(file_path)
        
        # Add technical indicators if requested
        if add_indicators:
            df = add_technical_indicators(df)
        
        # Initialize model and make predictions
        model = StockPredictionModel()
        prediction_results = model.predict(df, with_indicators=add_indicators)
        
        # Return the results
        return {
            "message": "Prediction completed successfully",
            "filename": filename,
            "with_indicators": add_indicators,
            "metrics": prediction_results['metrics'],
            "plot": prediction_results['plot'],
            "next_day_prediction": prediction_results['next_day_prediction']
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error making prediction: {str(e)}")

@app.post("/update-model/")
async def update_model(
    filename: str = Form(...),
    learning_rate: float = Form(0.0005),
    epochs: int = Form(5)
):
    """Update an existing model with new data to improve predictions"""
    try:
        # Check if the file exists
        file_path = f"uploads/{filename}"
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"File {filename} not found")
        
        # Load the data
        df = pd.read_csv(file_path)
        
        # Add technical indicators for better features
        df = add_technical_indicators(df)
        
        # Check if there's a trained model to update
        try:
            model = StockPredictionModel()
            model.predict(df.head(1), with_indicators=True)  # Just to test if model loads correctly
        except Exception as e:
            raise HTTPException(
                status_code=400, 
                detail="No existing model found to update. Please train a model first."
            )
        
        # Update the model with new data
        update_results = model.update_model(df, learning_rate=learning_rate, epochs=epochs)
        
        # Return results
        return {
            "message": "Model successfully updated",
            "filename": filename,
            "update_number": update_results["update_number"],
            "final_loss": update_results["update_history"]["loss"][-1]
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating model: {str(e)}")

@app.post("/ensemble-predict/")
async def ensemble_predict(
    filename: str = Form(...),
    add_indicators: bool = Form(True),
    num_samples: int = Form(10)
):
    """Make ensemble predictions with uncertainty estimates using Monte Carlo Dropout"""
    try:
        # Check if the file exists
        file_path = f"uploads/{filename}"
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"File {filename} not found")
        
        # Load the data
        df = pd.read_csv(file_path)
        
        # Add technical indicators if requested
        if add_indicators:
            df = add_technical_indicators(df)
        
        # Initialize model and make ensemble predictions
        model = StockPredictionModel()
        prediction_results = model.predict(df, with_indicators=add_indicators, ensemble=True, num_samples=num_samples)
        
        # Return the results
        return {
            "message": "Ensemble prediction completed successfully",
            "filename": filename,
            "with_indicators": add_indicators,
            "metrics": prediction_results['metrics'],
            "plot": prediction_results['plot'],
            "next_day_prediction": prediction_results['next_day_prediction'],
            "uncertainty": prediction_results.get('uncertainty', {}),
            "attention_weights": prediction_results.get('attention_weights')
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error making ensemble prediction: {str(e)}")

@app.get("/model-info/")
async def get_model_info():
    """Get information about the currently loaded model"""
    try:
        # Check if there are any model files
        model_directory = "models"
        if not os.path.exists(model_directory):
            return {"message": "No models found"}
        
        model_files = [f for f in os.listdir(model_directory) if f.endswith('.pt')]
        if not model_files:
            return {"message": "No trained models available"}
        
        # Get info about the default model
        default_model_path = os.path.join(model_directory, "stock_prediction_model.pt")
        if (os.path.exists(default_model_path)):
            import torch
            checkpoint = torch.load(default_model_path, map_location="cpu")
            
            model_info = {
                "model_version": checkpoint.get("model_version", "unknown"),
                "timestamp": checkpoint.get("timestamp", "unknown"),
                "metrics": checkpoint.get("metrics", {}),
                "with_indicators": checkpoint.get("with_indicators", True)
            }
        else:
            model_info = {"message": "Default model exists but metadata couldn't be extracted"}
        
        # Get list of all model versions
        model_versions = []
        for model_file in model_files:
            if model_file != "stock_prediction_model.pt":
                model_versions.append({
                    "filename": model_file,
                    "path": os.path.join(model_directory, model_file),
                    "size_kb": round(os.path.getsize(os.path.join(model_directory, model_file)) / 1024, 2)
                })
        
        return {
            "default_model": model_info,
            "model_versions": model_versions,
            "total_models": len(model_files)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving model info: {str(e)}")

@app.get("/market-regime/")
async def get_market_regime(filename: str):
    """Analyze the market regime for the given stock data"""
    try:
        # Check if the file exists
        file_path = f"uploads/{filename}"
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"File {filename} not found")
        
        # Load the data
        df = pd.read_csv(file_path)
        
        # Extract market regime
        regime = extract_market_regime(df)
        
        # Return the results
        return {
            "filename": filename,
            "market_regime": regime,
            "data_points": len(df),
            "date_range": f"{df['Date'].iloc[0]} to {df['Date'].iloc[-1]}" if 'Date' in df.columns else "Unknown"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing market regime: {str(e)}")

@app.get("/files/")
async def list_files():
    """List all uploaded CSV files"""
    try:
        files = []
        for filename in os.listdir("uploads"):
            if filename.endswith(".csv"):
                files.append(filename)
        return {"files": files}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing files: {str(e)}")

if __name__ == "__main__":
    import sys
    import os
    # Get the absolute path of the backend directory
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    # Add the backend directory to sys.path
    sys.path.insert(0, backend_dir)
    # Run the server
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)