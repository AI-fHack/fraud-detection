from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from src.api.models import Transaction, PredictionResponse, ExplainResponse, FeatureImportance
from src.api.predict import get_model, get_threshold, get_selected_features, is_model_loaded
from src.ml.preprocess import preprocess
import logging
import numpy as np
import pandas as pd

try:
    import shap
except ImportError:
    shap = None

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
            "explain": "/explain",
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
        
        try:
            model = get_model()
        except (FileNotFoundError, Exception) as e:
            logger.error(f"Model not available: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model is not available. Please ensure the model has been trained and saved."
            )
        
        try:
            features = preprocess(data)
        except Exception as e:
            logger.error(f"Preprocessing error: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to preprocess data: {str(e)}"
            )
        
        try:
            selected_features = get_selected_features()
            
            if selected_features and len(selected_features) > 0:
                features_df = pd.DataFrame([features], columns=selected_features[:len(features)])
                proba = model.predict_proba(features_df)[0][1]
            else:
                proba = model.predict_proba([features])[0][1]
            
            threshold = get_threshold()
            is_fraud = proba >= threshold
            
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
        raise
    except Exception as e:
        logger.error(f"Unexpected error in predict endpoint: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


_shap_explainer = None


def get_shap_explainer():
    """Get or create SHAP explainer."""
    global _shap_explainer
    
    if shap is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="SHAP library is not installed. Please install it: pip install shap"
        )
    
    if _shap_explainer is not None:
        return _shap_explainer
    
    try:
        model = get_model()
        _shap_explainer = shap.TreeExplainer(model)
        return _shap_explainer
    except Exception as e:
        logger.error(f"Failed to create SHAP explainer: {str(e)}")
        raise


@app.post("/explain", response_model=ExplainResponse)
def explain(data: Transaction):
    """
    Explain prediction using SHAP values.
    
    Args:
        data: Transaction data
        
    Returns:
        dict: Prediction with SHAP explanation and top factors
    """
    try:
        if not is_model_loaded():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model is not available."
            )
        
        model = get_model()
        threshold = get_threshold()
        selected_features = get_selected_features()
        
        features = preprocess(data)
        
        if selected_features and len(selected_features) > 0:
            features_df = pd.DataFrame([features], columns=selected_features[:len(features)])
            proba = model.predict_proba(features_df)[0][1]
        else:
            features_array = np.array([features])
            proba = model.predict_proba(features_array)[0][1]
        is_fraud = proba >= threshold
        
        try:
            explainer = get_shap_explainer()
            if selected_features and len(selected_features) > 0:
                features_df = pd.DataFrame([features], columns=selected_features[:len(features)])
                shap_values = explainer.shap_values(features_df)
            else:
                features_array = np.array([features])
                shap_values = explainer.shap_values(features_array)
            
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            
            if isinstance(explainer.expected_value, (list, np.ndarray)):
                base_value = float(explainer.expected_value[1] if len(explainer.expected_value) > 1 else explainer.expected_value[0])
            else:
                base_value = float(explainer.expected_value)
            
            feature_names = selected_features if selected_features and len(selected_features) >= len(features) else [f"feature_{i}" for i in range(len(features))]
            
            shap_dict = {}
            shap_array = shap_values[0] if len(shap_values.shape) > 1 else shap_values
            for i, (name, value) in enumerate(zip(feature_names[:len(shap_array)], shap_array)):
                shap_dict[name] = float(value)
            
            sorted_features = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)
            top_factors = [
                FeatureImportance(
                    feature_name=name,
                    shap_value=value,
                    impact="increases_fraud" if value > 0 else "decreases_fraud"
                )
                for name, value in sorted_features[:10]
            ]
            
        except Exception as e:
            logger.error(f"SHAP explanation error: {str(e)}")
            top_factors = []
            base_value = 0.0
        
        return ExplainResponse(
            fraud_probability=float(proba),
            is_fraud=bool(is_fraud),
            base_value=base_value,
            top_factors=top_factors,
            transaction_id=data.transaction_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in explain endpoint: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Explanation failed: {str(e)}"
        )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unhandled exceptions."""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error", "error": str(exc)}
    )
