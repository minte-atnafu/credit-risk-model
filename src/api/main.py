from fastapi import FastAPI, HTTPException
from pydantic import ValidationError
from src.api.pydantic_models import PredictionRequest, PredictionResponse
import mlflow
import mlflow.sklearn
import pandas as pd
import logging
import os
import joblib

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set MLflow tracking URI
os.environ["MLFLOW_TRACKING_URI"] = "http://127.0.0.1:5000"

app = FastAPI(title="Credit Risk Prediction API")

# Load the model from MLflow
MODEL_NAME = "CreditRiskModel"
MODEL_VERSION = "1"  # Update if re-run train_model.py
FALLBACK_RUN_ID = "21a42d776a424cd3ba3cb9c8971ab587"  # Update if new run ID
LOCAL_MODEL_PATH = "models/local_credit_risk_model.pkl"  # Local fallback path

try:
    logger.info(f"Attempting to load model {MODEL_NAME} version {MODEL_VERSION}")
    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{MODEL_VERSION}")
    logger.info(f"Successfully loaded model {MODEL_NAME} version {MODEL_VERSION}")
except Exception as e:
    logger.error(f"Failed to load model from registry: {str(e)}")
    logger.info(f"Attempting fallback to run ID {FALLBACK_RUN_ID}")
    try:
        model = mlflow.sklearn.load_model(f"runs:/{FALLBACK_RUN_ID}/randomforest")
        logger.info(f"Successfully loaded model from run {FALLBACK_RUN_ID}")
    except Exception as e2:
        logger.error(f"Failed to load fallback model: {str(e2)}")
        logger.info(f"Attempting to load local model from {LOCAL_MODEL_PATH}")
        try:
            model = joblib.load(LOCAL_MODEL_PATH)
            logger.info(f"Successfully loaded local model from {LOCAL_MODEL_PATH}")
        except Exception as e3:
            logger.error(f"Failed to load local model: {str(e3)}")
            from mlflow.tracking import MlflowClient
            client = MlflowClient()
            try:
                models = client.search_registered_models()
                logger.info(f"Available models: {[m.name for m in models]}")
                for m in models:
                    logger.info(f"Model {m.name}: versions={[v.version for v in m.latest_versions]}")
            except Exception as e4:
                logger.error(f"Failed to access registry: {str(e4)}")
            raise RuntimeError(f"Failed to load model: {str(e)} or fallback: {str(e2)} or local: {str(e3)}")

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Predict credit risk probability for a customer.
    Expects input data matching the 51 features used in training.
    """
    try:
        # Convert Pydantic model to DataFrame
        input_dict = request.dict()
        input_df = pd.DataFrame([input_dict])
        
        # Predict
        prediction_proba = model.predict_proba(input_df)[:, 1][0]
        prediction = int(prediction_proba >= 0.5)
        
        logger.info(f"Prediction made: is_high_risk={prediction}, probability={prediction_proba}")
        
        return PredictionResponse(
            is_high_risk=prediction,
            risk_probability=prediction_proba
        )
    except ValidationError as ve:
        logger.error(f"Input validation error: {str(ve)}")
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint for health check."""
    return {"message": "Credit Risk Prediction API is running"}