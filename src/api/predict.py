import joblib
from pathlib import Path
from typing import Optional, Dict, Any

_model_package: Optional[Dict[str, Any]] = None
_model: Optional[object] = None
_selected_features: Optional[list] = None
_threshold: Optional[float] = None


def get_model_package():
    """Load the trained model package from file."""
    global _model_package
    
    if _model_package is not None:
        return _model_package
    
    project_root = Path(__file__).parent.parent.parent
    model_path = project_root / "model" / "model.pkl"
    
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found at {model_path}. "
            "Please ensure the model has been trained and saved."
        )
    
    try:
        _model_package = joblib.load(model_path)
        return _model_package
    except Exception as e:
        raise Exception(f"Failed to load model: {str(e)}")


def get_model():
    """Get the trained model."""
    global _model
    
    if _model is not None:
        return _model
    
    package = get_model_package()
    
    if isinstance(package, dict):
        _model = package.get('model', package)
    else:
        _model = package
    
    return _model


def get_selected_features():
    """Get the list of selected features."""
    global _selected_features
    
    if _selected_features is not None:
        return _selected_features
    
    package = get_model_package()
    
    if isinstance(package, dict):
        _selected_features = package.get('selected_features', [])
    else:
        _selected_features = []
    
    return _selected_features


def get_threshold():
    """Get the optimal threshold."""
    global _threshold
    
    if _threshold is not None:
        return _threshold
    
    package = get_model_package()
    
    if isinstance(package, dict):
        _threshold = package.get('threshold', 0.5)
    else:
        _threshold = 0.5
    
    return _threshold


def is_model_loaded() -> bool:
    """Check if model is loaded."""
    try:
        get_model()
        return True
    except (FileNotFoundError, Exception):
        return False
