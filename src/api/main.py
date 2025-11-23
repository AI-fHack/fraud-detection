from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from src.api.models import Transaction, PredictionResponse
from src.api.predict import get_model, is_model_loaded
from src.ml.preprocess import preprocess
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Fraud Detection API",
    description="API для детекции мошеннических транзакций в мобильном интернет-банкинге",
    version="1.0.0"
)


@app.get("/")
def root():
    """Root endpoint with API information."""
    return {
        "message": "Fraud Detection API",
        "status": "running",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "model_status": "/model/status",
            "predict": "/predict",
            "docs": "/docs"
        }
    }


@app.get("/health")
def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "fraud-detection-api"
    }


@app.get("/model/status")
def model_status():
    """Check if model is loaded and ready."""
    try:
        is_loaded = is_model_loaded()
        return {
            "model_loaded": is_loaded,
            "status": "ready" if is_loaded else "not_loaded",
            "message": "Model is ready for predictions" if is_loaded else "Model file not found. Please train and save the model first."
        }
    except Exception as e:
        logger.error(f"Error checking model status: {str(e)}")
        return {
            "model_loaded": False,
            "status": "error",
            "message": str(e)
        }


@app.post("/predict", response_model=PredictionResponse)
def predict(data: Transaction):
    """
    Predict if a transaction is fraudulent.
    
    Args:
        data: Transaction data with amount, user_id, device_type, etc.
        
    Returns:
        dict: Prediction result with fraud_probability and is_fraud flag
        
    Raises:
        HTTPException: If model is not loaded or prediction fails
    """
    try:
        # Check if model is loaded
        if not is_model_loaded():
            try:
                model = get_model()
            except FileNotFoundError as e:
                logger.error(f"Model not found: {str(e)}")
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Model is not available. Please ensure the model has been trained and saved."
                )
            except Exception as e:
                logger.error(f"Failed to load model: {str(e)}")
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Model is not available. Please ensure the model has been trained and saved."
                )
        
        # Get model (should be loaded at this point)
        try:
            model = get_model()
        except (FileNotFoundError, Exception) as e:
            logger.error(f"Model not available: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model is not available. Please ensure the model has been trained and saved."
            )
        
        # Preprocess input data
        try:
            features = preprocess(data)
        except Exception as e:
            logger.error(f"Preprocessing error: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to preprocess data: {str(e)}"
            )
        
        # Make prediction
        try:
            proba = model.predict_proba([features])[0][1]
            is_fraud = proba > 0.5
            
            # Определение уровня уверенности
            if proba > 0.8 or proba < 0.2:
                confidence = "high"
            elif proba > 0.6 or proba < 0.4:
                confidence = "medium"
            else:
                confidence = "low"
            
            return PredictionResponse(
                fraud_probability=float(proba),
                is_fraud=bool(is_fraud),
                transaction_id=data.transaction_id,
                status="fraud" if is_fraud else "legitimate",
                confidence=confidence
            )
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Prediction failed: {str(e)}"
            )
            
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Unexpected error in predict endpoint: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unhandled exceptions."""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error", "error": str(exc)}
    )
