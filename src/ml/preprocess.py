"""
Модуль предобработки данных для ML модели.
Преобразует данные транзакции в вектор признаков для модели.
"""
from typing import List, Optional
import numpy as np


def preprocess(data) -> List[float]:
    """
    Preprocess transaction data into features for model prediction.
    
    Args:
        data: Transaction object with transaction details
        
    Returns:
        list: Feature vector for the model
        
    Note:
        Эта функция будет обновлена после того, как ML-инженер
        предоставит требования к фичам модели.
    """
    features = []
    
    # 1. Основные признаки транзакции
    features.append(float(data.amount))
    features.append(float(data.user_id))
    
    # 2. Признаки даты/времени (если доступны)
    # Можно извлечь: день недели, час, день месяца и т.д.
    if data.transaction_datetime:
        # Заглушка: можно парсить дату и извлекать фичи
        # features.append(extract_hour(data.transaction_datetime))
        # features.append(extract_day_of_week(data.transaction_datetime))
        pass
    
    # 3. Поведенческие признаки устройства
    if data.device_type:
        # Кодирование типа устройства (one-hot или label encoding)
        device_encoding = encode_device_type(data.device_type)
        features.extend(device_encoding)
    else:
        # Если не указано, используем значения по умолчанию
        features.extend([0.0, 0.0, 0.0])  # Заглушка для 3 типов устройств
    
    # 4. Признаки модели телефона и ОС
    if data.phone_model:
        phone_model_hash = hash_device_model(data.phone_model)
        features.append(phone_model_hash)
    else:
        features.append(0.0)
    
    if data.os_version:
        os_hash = hash_os_version(data.os_version)
        features.append(os_hash)
    else:
        features.append(0.0)
    
    # 5. Поведенческие метрики
    features.append(float(data.unique_os_versions_30d) if data.unique_os_versions_30d is not None else 0.0)
    features.append(float(data.unique_phone_models_30d) if data.unique_phone_models_30d is not None else 0.0)
    features.append(float(data.logins_last_7_days) if data.logins_last_7_days is not None else 0.0)
    features.append(float(data.logins_last_30_days) if data.logins_last_30_days is not None else 0.0)
    features.append(float(data.avg_logins_per_day_7d) if data.avg_logins_per_day_7d is not None else 0.0)
    features.append(float(data.avg_logins_per_day_30d) if data.avg_logins_per_day_30d is not None else 0.0)
    
    # 6. Дополнительные вычисляемые признаки
    # Отношение логинов за 7 дней к 30 дням
    if data.logins_last_30_days and data.logins_last_30_days > 0:
        login_ratio = (data.logins_last_7_days or 0) / data.logins_last_30_days
        features.append(float(login_ratio))
    else:
        features.append(0.0)
    
    # Нормализация суммы (можно использовать логарифм для больших сумм)
    features.append(np.log1p(data.amount))  # log(1 + amount)
    
    # Признак частоты смены устройств
    if data.unique_phone_models_30d and data.unique_phone_models_30d > 0:
        device_change_freq = float(data.unique_phone_models_30d) / 30.0
        features.append(device_change_freq)
    else:
        features.append(0.0)
    
    return features


def encode_device_type(device_type: str) -> List[float]:
    """
    Кодирование типа устройства.
    
    Args:
        device_type: Тип устройства (mobile, desktop, tablet)
        
    Returns:
        List[float]: One-hot encoding или label encoding
    """
    device_mapping = {
        'mobile': [1.0, 0.0, 0.0],
        'desktop': [0.0, 1.0, 0.0],
        'tablet': [0.0, 0.0, 1.0],
    }
    return device_mapping.get(device_type.lower(), [0.0, 0.0, 0.0])


def hash_device_model(phone_model: str) -> float:
    """
    Хеширование модели телефона в числовое значение.
    
    Args:
        phone_model: Модель телефона
        
    Returns:
        float: Нормализованное хеш-значение
    """
    # Простое хеширование (можно улучшить)
    hash_value = hash(phone_model) % 10000
    return float(hash_value) / 10000.0  # Нормализация к [0, 1]


def hash_os_version(os_version: str) -> float:
    """
    Хеширование версии ОС в числовое значение.
    
    Args:
        os_version: Версия ОС
        
    Returns:
        float: Нормализованное хеш-значение
    """
    # Простое хеширование (можно улучшить)
    hash_value = hash(os_version) % 10000
    return float(hash_value) / 10000.0  # Нормализация к [0, 1]


def extract_hour(datetime_str: str) -> float:
    """Извлечение часа из строки даты/времени."""
    try:
        # Парсинг формата '2025-01-05 16:32:02.000'
        hour = int(datetime_str.split()[1].split(':')[0])
        return float(hour) / 24.0  # Нормализация к [0, 1]
    except:
        return 0.0


def extract_day_of_week(datetime_str: str) -> float:
    """Извлечение дня недели из строки даты/времени."""
    try:
        from datetime import datetime
        dt = datetime.strptime(datetime_str.split()[0], '%Y-%m-%d')
        day_of_week = dt.weekday()  # 0 = Monday, 6 = Sunday
        return float(day_of_week) / 7.0  # Нормализация к [0, 1]
    except:
        return 0.0
