"""
Модуль предобработки данных для ML модели.
Преобразует данные транзакции в вектор признаков для модели.
"""
from typing import List
import numpy as np
from datetime import datetime
from pathlib import Path


def load_feature_names() -> List[str]:
    """Загружает список финальных фичей из файла."""
    project_root = Path(__file__).parent.parent.parent
    features_file = project_root / "model" / "final_features_list.txt"
    
    if not features_file.exists():
        return []
    
    features = []
    with open(features_file, 'r', encoding='utf-8') as f:
        for line in f.readlines()[4:]:
            if line.strip().startswith('-'):
                feature_name = line.strip()[2:].strip()
                features.append(feature_name)
    
    return features


def preprocess(data) -> List[float]:
    """
    Preprocess transaction data into features for model prediction.
    
    Args:
        data: Transaction object with transaction details
        
    Returns:
        list: Feature vector with 83 features in the correct order
    """
    selected_features = load_feature_names()
    
    if not selected_features:
        selected_features = [
            'time_until_next_tx', 'user_min_amount_total', 'user_tx_count_total',
            'amount', 'time_since_last_tx', 'user_max_amount_total',
            'amount_diff_from_user_avg', 'day_of_month', 'amount_zscore_user',
            'user_avg_amount_total', 'month_cos', 'amount_ratio_to_user_avg',
            'user_std_amount_total', 'month_sin', 'day_of_week', 'hour_cos',
            'month', 'tx_mean_amount_24h', 'hour', 'tx_mean_amount_1h'
        ]
    
    features_dict = {}
    
    features_dict['amount'] = float(data.amount)
    features_dict['user_id'] = float(data.user_id)
    
    hour = 12.0
    day_of_week = 0.0
    day_of_month = 1.0
    month = 6.0
    
    if data.transaction_datetime:
        try:
            dt_str = data.transaction_datetime.split()[0]
            dt = datetime.strptime(dt_str, '%Y-%m-%d')
            if len(data.transaction_datetime.split()) > 1:
                time_part = data.transaction_datetime.split()[1]
                hour = float(time_part.split(':')[0])
            day_of_week = float(dt.weekday())
            day_of_month = float(dt.day)
            month = float(dt.month)
        except:
            pass
    
    features_dict['hour'] = hour
    features_dict['day_of_week'] = day_of_week
    features_dict['day_of_month'] = day_of_month
    features_dict['month'] = month
    features_dict['is_weekend'] = 1.0 if day_of_week >= 5 else 0.0
    features_dict['is_night'] = 1.0 if 22 <= hour <= 23 or 0 <= hour <= 6 else 0.0
    features_dict['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    features_dict['hour_cos'] = np.cos(2 * np.pi * hour / 24)
    features_dict['day_of_week_sin'] = np.sin(2 * np.pi * day_of_week / 7)
    features_dict['day_of_week_cos'] = np.cos(2 * np.pi * day_of_week / 7)
    features_dict['month_sin'] = np.sin(2 * np.pi * month / 12)
    features_dict['month_cos'] = np.cos(2 * np.pi * month / 12)
    
    features_dict['количество_разных_версий_ос__os_ver__за_последние_30_дней_до_transdate___сколько_разных_ос_версий_использовал_клиент'] = float(data.unique_os_versions_30d) if data.unique_os_versions_30d is not None else 0.0
    features_dict['количество_разных_моделей_телефона__phone_model__за_последние_30_дней___насколько_часто_клиент__менял_устройство__по_логам'] = float(data.unique_phone_models_30d) if data.unique_phone_models_30d is not None else 0.0
    features_dict['количество_уникальных_логин_сессий__минутных_тайм_слотов__за_последние_7_дней_до_transdate'] = float(data.logins_last_7_days) if data.logins_last_7_days is not None else 0.0
    features_dict['количество_уникальных_логин_сессий_за_последние_30_дней_до_transdate'] = float(data.logins_last_30_days) if data.logins_last_30_days is not None else 0.0
    features_dict['среднее_число_логинов_в_день_за_последние_7_дней__logins_last_7_days___7'] = float(data.avg_logins_per_day_7d) if data.avg_logins_per_day_7d is not None else 0.0
    features_dict['среднее_число_логинов_в_день_за_последние_30_дней__logins_last_30_days___30'] = float(data.avg_logins_per_day_30d) if data.avg_logins_per_day_30d is not None else 0.0
    
    if data.phone_model:
        features_dict['модель_телефона_из_самой_последней_сессии__по_времени__перед_transdate_encoded'] = float(hash(data.phone_model) % 10000) / 10000.0
    else:
        features_dict['модель_телефона_из_самой_последней_сессии__по_времени__перед_transdate_encoded'] = 0.0
    
    if data.os_version:
        features_dict['версия_ос_из_самой_последней_сессии_перед_transdate_encoded'] = float(hash(data.os_version) % 10000) / 10000.0
    else:
        features_dict['версия_ос_из_самой_последней_сессии_перед_transdate_encoded'] = 0.0
    
    if data.destination:
        features_dict['recipient_freq_encoding'] = float(hash(data.destination) % 10000) / 10000.0
    else:
        features_dict['recipient_freq_encoding'] = 0.0
    
    feature_vector = []
    for feature_name in selected_features:
        if feature_name in features_dict:
            feature_vector.append(features_dict[feature_name])
        else:
            feature_vector.append(0.0)
    
    if len(feature_vector) < 83:
        feature_vector.extend([0.0] * (83 - len(feature_vector)))
    elif len(feature_vector) > 83:
        feature_vector = feature_vector[:83]
    
    return feature_vector
