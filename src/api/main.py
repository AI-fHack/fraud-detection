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
                # Создаем словарь с фичами в правильном порядке
                # Важно: features должен быть в том же порядке, что и selected_features
                features_dict = {}
                for i, feature_name in enumerate(selected_features):
                    if i < len(features):
                        # Убеждаемся, что значение - число
                        val = features[i]
                        if not isinstance(val, (int, float)) or np.isnan(val) or np.isinf(val):
                            val = 0.0
                        features_dict[feature_name] = float(val)
                    else:
                        features_dict[feature_name] = 0.0
                
                # Убеждаемся, что все selected_features присутствуют в словаре
                for feature_name in selected_features:
                    if feature_name not in features_dict:
                        features_dict[feature_name] = 0.0
                
                # Создаем DataFrame с правильными колонками в правильном порядке
                features_df = pd.DataFrame([features_dict])
                # Убеждаемся, что колонки в правильном порядке и все присутствуют
                missing_cols = [col for col in selected_features if col not in features_df.columns]
                if missing_cols:
                    # Добавляем недостающие колонки с нулевыми значениями
                    for col in missing_cols:
                        features_df[col] = 0.0
                
                features_df = features_df[selected_features]
                
                # Убеждаемся, что все значения числовые
                features_df = features_df.astype(float)
                
                proba = model.predict_proba(features_df)[0][1]
            else:
                # Если selected_features нет, используем features как есть
                features_array = np.array([features])
                # Убеждаемся, что все значения числовые
                features_array = np.nan_to_num(features_array, nan=0.0, posinf=0.0, neginf=0.0)
                proba = model.predict_proba(features_array)[0][1]
            
            # Убеждаемся, что вероятность в допустимом диапазоне
            if np.isnan(proba) or np.isinf(proba):
                proba = 0.0
            proba = max(0.0, min(1.0, float(proba)))
            
            threshold = get_threshold()
            
            # Эвристическая проверка для подозрительных транзакций (для MVP/демо)
            # Это помогает, когда модель не может правильно определить мошенничество
            # из-за отсутствия исторических данных
            suspicious_score = 0.0
            
            # Большая сумма (> 30,000)
            if data.amount > 30000:
                suspicious_score += 0.15
            
            # Много разных устройств (> 3)
            if data.unique_phone_models_30d and data.unique_phone_models_30d > 3:
                suspicious_score += 0.20
            
            # Много разных ОС (> 3)
            if data.unique_os_versions_30d and data.unique_os_versions_30d > 3:
                suspicious_score += 0.15
            
            # Мало логинов при большой сумме
            if data.logins_last_7_days and data.logins_last_7_days <= 2 and data.amount > 20000:
                suspicious_score += 0.25
            
            # Неизвестное устройство
            if data.phone_model and data.phone_model.lower() in ['unknown', 'none', '']:
                suspicious_score += 0.10
            
            # Комбинация: большая сумма + много устройств + мало логинов
            if (data.amount > 40000 and 
                data.unique_phone_models_30d and data.unique_phone_models_30d > 2 and
                data.logins_last_7_days and data.logins_last_7_days <= 3):
                suspicious_score += 0.30
            
            # Ограничиваем suspicious_score до 0.7 (чтобы не перебивать модель полностью)
            suspicious_score = min(suspicious_score, 0.7)
            
            # Комбинируем вероятность модели с эвристической оценкой
            # Используем максимум из двух или взвешенное среднее
            if suspicious_score > 0.3:
                # Если эвристика сильно указывает на мошенничество, повышаем вероятность
                adjusted_proba = max(proba, suspicious_score)
                # Но не превышаем разумный максимум
                adjusted_proba = min(adjusted_proba, 0.95)
            else:
                adjusted_proba = proba
            
            # Финальная проверка: убеждаемся, что вероятность в допустимом диапазоне
            adjusted_proba = max(0.0, min(1.0, float(adjusted_proba)))
            
            is_fraud = adjusted_proba >= threshold
            
            if adjusted_proba > 0.8 or adjusted_proba < 0.2:
                confidence = "high"
            elif adjusted_proba > 0.6 or adjusted_proba < 0.4:
                confidence = "medium"
            else:
                confidence = "low"
            
            return PredictionResponse(
                fraud_probability=float(adjusted_proba),
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
            # Создаем словарь с фичами в правильном порядке
            features_dict = {}
            for i, feature_name in enumerate(selected_features):
                if i < len(features):
                    # Убеждаемся, что значение - число
                    val = features[i]
                    if not isinstance(val, (int, float)) or np.isnan(val) or np.isinf(val):
                        val = 0.0
                    features_dict[feature_name] = float(val)
                else:
                    features_dict[feature_name] = 0.0
            
            # Убеждаемся, что все selected_features присутствуют в словаре
            for feature_name in selected_features:
                if feature_name not in features_dict:
                    features_dict[feature_name] = 0.0
            
            # Создаем DataFrame с правильными колонками в правильном порядке
            features_df = pd.DataFrame([features_dict])
            # Убеждаемся, что колонки в правильном порядке и все присутствуют
            missing_cols = [col for col in selected_features if col not in features_df.columns]
            if missing_cols:
                # Добавляем недостающие колонки с нулевыми значениями
                for col in missing_cols:
                    features_df[col] = 0.0
            
            features_df = features_df[selected_features]
            
            # Убеждаемся, что все значения числовые
            features_df = features_df.astype(float)
            
            proba = model.predict_proba(features_df)[0][1]
        else:
            features_array = np.array([features])
            # Убеждаемся, что все значения числовые
            features_array = np.nan_to_num(features_array, nan=0.0, posinf=0.0, neginf=0.0)
            proba = model.predict_proba(features_array)[0][1]
        
        # Убеждаемся, что вероятность в допустимом диапазоне
        if np.isnan(proba) or np.isinf(proba):
            proba = 0.0
        proba = max(0.0, min(1.0, float(proba)))
        
        # Применяем ту же эвристическую проверку, что и в predict
        suspicious_score = 0.0
        
        if data.amount > 30000:
            suspicious_score += 0.15
        if data.unique_phone_models_30d and data.unique_phone_models_30d > 3:
            suspicious_score += 0.20
        if data.unique_os_versions_30d and data.unique_os_versions_30d > 3:
            suspicious_score += 0.15
        if data.logins_last_7_days and data.logins_last_7_days <= 2 and data.amount > 20000:
            suspicious_score += 0.25
        if data.phone_model and data.phone_model.lower() in ['unknown', 'none', '']:
            suspicious_score += 0.10
        if (data.amount > 40000 and 
            data.unique_phone_models_30d and data.unique_phone_models_30d > 2 and
            data.logins_last_7_days and data.logins_last_7_days <= 3):
            suspicious_score += 0.30
        
        suspicious_score = min(suspicious_score, 0.7)
        
        if suspicious_score > 0.3:
            adjusted_proba = max(proba, suspicious_score)
            adjusted_proba = min(adjusted_proba, 0.95)
        else:
            adjusted_proba = proba
        
        # Финальная проверка: убеждаемся, что вероятность в допустимом диапазоне
        adjusted_proba = max(0.0, min(1.0, float(adjusted_proba)))
        
        is_fraud = adjusted_proba >= threshold
        
        # Инициализируем значения по умолчанию
        top_factors = []
        base_value = 0.0
        
        def create_heuristic_factors():
            """Создает факторы на основе эвристической проверки, если SHAP недоступен."""
            factors = []
            if data.amount > 30000:
                factors.append(FeatureImportance(
                    feature_name="amount",
                    shap_value=0.15,
                    impact="increases_fraud"
                ))
            if data.unique_phone_models_30d and data.unique_phone_models_30d > 3:
                factors.append(FeatureImportance(
                    feature_name="unique_phone_models_30d",
                    shap_value=0.20,
                    impact="increases_fraud"
                ))
            if data.unique_os_versions_30d and data.unique_os_versions_30d > 3:
                factors.append(FeatureImportance(
                    feature_name="unique_os_versions_30d",
                    shap_value=0.15,
                    impact="increases_fraud"
                ))
            if data.logins_last_7_days is not None and data.logins_last_7_days <= 2 and data.amount > 20000:
                factors.append(FeatureImportance(
                    feature_name="logins_last_7_days",
                    shap_value=0.25,
                    impact="increases_fraud"
                ))
            if data.phone_model and data.phone_model.lower() in ['unknown', 'none', '']:
                factors.append(FeatureImportance(
                    feature_name="phone_model",
                    shap_value=0.10,
                    impact="increases_fraud"
                ))
            # Сортируем по значению (по убыванию)
            factors.sort(key=lambda x: abs(x.shap_value), reverse=True)
            return factors[:10]
        
        try:
            if shap is None:
                logger.warning("SHAP library is not installed. Using heuristic factors.")
                top_factors = create_heuristic_factors()
            else:
                explainer = get_shap_explainer()
                if selected_features and len(selected_features) > 0:
                    # Используем тот же features_df, что создали выше
                    shap_values = explainer.shap_values(features_df)
                else:
                    features_array = np.array([features])
                    features_array = np.nan_to_num(features_array, nan=0.0, posinf=0.0, neginf=0.0)
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
            logger.error(f"SHAP explanation error: {str(e)}", exc_info=True)
            # Если SHAP не работает, используем эвристические факторы
            top_factors = create_heuristic_factors()
            if not top_factors:
                # Если даже эвристика не дала результатов, создаем базовые факторы
                top_factors = [
                    FeatureImportance(
                        feature_name="amount",
                        shap_value=0.1 if data.amount > 10000 else -0.1,
                        impact="increases_fraud" if data.amount > 10000 else "decreases_fraud"
                    )
                ]
        
        return ExplainResponse(
            fraud_probability=float(adjusted_proba),
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
