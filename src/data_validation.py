"""
Функция валидации входных данных для fraud detection модели.
"""
import pandas as pd
import numpy as np

def validate_input_data(df, required_features=None, target_col='target'):
    """Валидация входных данных для fraud detection модели."""
    errors = []
    warnings = []
    
    if not isinstance(df, pd.DataFrame):
        errors.append("Входные данные должны быть pandas.DataFrame")
        return {'is_valid': False, 'errors': errors, 'warnings': warnings}
    
    if df.empty:
        errors.append("DataFrame пуст")
        return {'is_valid': False, 'errors': errors, 'warnings': warnings}
    
    if required_features is not None:
        missing_features = [f for f in required_features if f not in df.columns]
        if missing_features:
            errors.append(f"Отсутствуют обязательные признаки: {missing_features[:10]}")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    non_numeric_cols = [col for col in df.columns 
                       if col not in numeric_cols and col not in [target_col, 'user_id']]
    if non_numeric_cols:
        errors.append(f"Найдены нечисловые колонки: {non_numeric_cols[:5]}")
    
    missing_counts = df.isnull().sum()
    cols_with_missing = missing_counts[missing_counts > 0]
    if len(cols_with_missing) > 0:
        warnings.append(f"Найдены пропущенные значения в {len(cols_with_missing)} колонках")
    
    inf_counts = np.isinf(df.select_dtypes(include=[np.number])).sum()
    cols_with_inf = inf_counts[inf_counts > 0]
    if len(cols_with_inf) > 0:
        errors.append(f"Найдены бесконечные значения в {len(cols_with_inf)} колонках")
    
    is_valid = len(errors) == 0
    
    return {
        'is_valid': is_valid,
        'errors': errors,
        'warnings': warnings,
        'shape': df.shape
    }
