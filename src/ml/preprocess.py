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
    # Вычисляем среднее число логинов, если не передано
    avg_logins_7d = float(data.avg_logins_per_day_7d) if data.avg_logins_per_day_7d is not None else 0.0
    avg_logins_30d = float(data.avg_logins_per_day_30d) if data.avg_logins_per_day_30d is not None else 0.0
    
    if avg_logins_7d == 0.0 and data.logins_last_7_days is not None:
        avg_logins_7d = float(data.logins_last_7_days) / 7.0
    if avg_logins_30d == 0.0 and data.logins_last_30_days is not None:
        avg_logins_30d = float(data.logins_last_30_days) / 30.0
    
    features_dict['среднее_число_логинов_в_день_за_последние_7_дней__logins_last_7_days___7'] = avg_logins_7d
    features_dict['среднее_число_логинов_в_день_за_последние_30_дней__logins_last_30_days___30'] = avg_logins_30d
    
    # Вычисляем относительное изменение частоты логинов
    if avg_logins_30d > 0:
        freq_change = (avg_logins_7d - avg_logins_30d) / avg_logins_30d
    else:
        freq_change = 0.0
    features_dict['относительное_изменение_частоты_логинов_за_7_дней_к_средней_частоте_за_30_дней_freq7d_freq30d_freq30d_freq__7d____freq__30d____freq__30d_freq7d_freq30d_freq30d___показывает__стал_клиент_заходить_чаще_или_реже_недавно'] = freq_change
    
    # Доля логинов за 7 дней от логинов за 30 дней
    logins_7d = float(data.logins_last_7_days) if data.logins_last_7_days is not None else 0.0
    logins_30d = float(data.logins_last_30_days) if data.logins_last_30_days is not None else 0.0
    if logins_30d > 0:
        login_ratio = logins_7d / logins_30d
    else:
        login_ratio = 0.0
    features_dict['доля_логинов_за_7_дней_от_логинов_за_30_дней'] = login_ratio
    
    # Базовые значения для user статистик
    # Для демо: предполагаем, что средняя транзакция пользователя ~5000
    # Если текущая сумма сильно отличается, это может быть подозрительно
    amount = float(data.amount)
    base_avg_amount = 5000.0  # Предполагаемое среднее для обычного пользователя
    
    # Если сумма большая (>20000), это может быть подозрительно
    if amount > 20000:
        user_avg = base_avg_amount
        user_std = base_avg_amount * 0.5
    else:
        # Для небольших сумм используем текущую сумму как среднее
        user_avg = amount if amount > 0 else base_avg_amount
        user_std = user_avg * 0.3
    
    features_dict['user_avg_amount_total'] = user_avg
    features_dict['user_max_amount_total'] = user_avg * 2.0
    features_dict['user_min_amount_total'] = user_avg * 0.1
    features_dict['user_std_amount_total'] = user_std
    features_dict['user_tx_count_total'] = 10.0  # Базовое значение
    
    # Вычисляем производные фичи от amount (показывают аномалии)
    features_dict['amount_diff_from_user_avg'] = amount - user_avg
    if user_avg > 0:
        features_dict['amount_ratio_to_user_avg'] = amount / user_avg
    else:
        features_dict['amount_ratio_to_user_avg'] = 1.0
    
    # Z-score: насколько текущая сумма отклоняется от среднего
    if user_std > 0:
        features_dict['amount_zscore_user'] = (amount - user_avg) / user_std
    else:
        features_dict['amount_zscore_user'] = 0.0
    
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
    
    # Дополнительные фичи для определения подозрительных паттернов
    # Если много разных устройств или ОС - это подозрительно
    unique_os = float(data.unique_os_versions_30d) if data.unique_os_versions_30d is not None else 0.0
    unique_phones = float(data.unique_phone_models_30d) if data.unique_phone_models_30d is not None else 0.0
    
    # device_changed - флаг смены устройства (подозрительно, если > 2)
    features_dict['device_changed'] = 1.0 if unique_phones > 2 else 0.0
    features_dict['os_changed'] = 1.0 if unique_os > 2 else 0.0
    
    # user_unique_devices - количество уникальных устройств
    features_dict['user_unique_devices'] = unique_phones
    
    # Если мало логинов за неделю при большом количестве устройств - подозрительно
    if logins_7d > 0:
        devices_per_login = unique_phones / logins_7d if logins_7d > 0 else 0.0
    else:
        devices_per_login = unique_phones if unique_phones > 0 else 0.0
    
    # Базовые значения для временных фичей (требуют исторических данных, но для демо используем дефолты)
    features_dict['time_since_last_tx'] = 3600.0  # 1 час назад (базовое значение)
    features_dict['time_until_next_tx'] = 3600.0  # 1 час вперед
    
    # Количество транзакций за последние периоды (базовые значения)
    features_dict['tx_count_1h'] = 1.0
    features_dict['tx_count_12h'] = 2.0
    features_dict['tx_count_24h'] = 3.0
    
    # Средние суммы за периоды
    features_dict['tx_mean_amount_1h'] = amount
    features_dict['tx_mean_amount_12h'] = amount * 0.9
    features_dict['tx_mean_amount_24h'] = amount * 0.8
    
    # Если сумма большая и логинов мало - это подозрительно
    if amount > 20000 and logins_7d < 3:
        features_dict['tx_mean_amount_24h'] = amount * 1.5  # Увеличиваем для подозрительных случаев
    
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
