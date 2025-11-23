import joblib
from pathlib import Path
from typing import Optional

# Global variable to store the model
_model: Optional[object] = None


def get_model():
    """
    Load the trained model from file safely.
    
    Returns:
        Model object or None if model file doesn't exist
        
    Raises:
        FileNotFoundError: If model file doesn't exist
        Exception: If model loading fails
    """
    global _model
    
    if _model is not None:
        return _model
    
    # Get the project root directory (parent of src)
    project_root = Path(__file__).parent.parent.parent
    model_path = project_root / "src" / "ml" / "model.pkl"
    
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found at {model_path}. "
            "Please ensure the model has been trained and saved."
        )
    
    try:
        _model = joblib.load(model_path)
        return _model
    except Exception as e:
        raise Exception(f"Failed to load model: {str(e)}")


def is_model_loaded() -> bool:
    """Check if model is loaded."""
    global _model
    return _model is not None


# Try to load model at module import, but don't fail if it doesn't exist
try:
    model = get_model()
except (FileNotFoundError, Exception):
    model = None
