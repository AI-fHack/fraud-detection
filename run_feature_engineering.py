
import sys
import io
# Устанавливаем UTF-8 для вывода в Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib

warnings.filterwarnings('ignore')

print("Все библиотеки загружены успешно!")


INPUT_PATH = 'data/processed/transactions_with_features.csv'
OUTPUT_PATH = 'data/processed/transactions_with_features_final.csv'
PIPELINE_PATH = 'model/preprocessing_pipeline.pkl'



df = pd.read_csv(INPUT_PATH, low_memory=False)
print(f"Данные загружены: {df.shape}")
print(f"Колонок: {len(df.columns)}")
print(f"Строк: {len(df)}")

# Проверка целевой переменной
if 'target' in df.columns:
    print(f"\nРаспределение классов:")
    print(df['target'].value_counts())
    print(f"Дисбаланс: {df['target'].value_counts()[0] / df['target'].value_counts()[1]:.2f}:1")
else:
    raise ValueError("Колонка 'target' не найдена!")



# Находим категориальные признаки
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
print(f"\nНайдено категориальных колонок: {len(categorical_cols)}")
if len(categorical_cols) > 0:
    print(f"Категориальные колонки: {categorical_cols}")

# Обработка категориальных признаков
label_encoders = {}
frequency_encodings = {}

for col in categorical_cols:
    if col == 'target' or col == 'user_id':
        continue
    
    print(f"\n--- Обработка колонки: {col} ---")
    
    # Проверяем количество уникальных значений
    unique_count = df[col].nunique()
    print(f"  Уникальных значений: {unique_count}")
    
    # Если много уникальных значений (>50) - используем frequency encoding
    # Если мало - можно использовать label encoding
    if unique_count > 50:
        # Frequency Encoding
        freq_map = df[col].value_counts() / len(df)
        df[f'{col}_freq_encoding'] = df[col].map(freq_map).fillna(0)
        frequency_encodings[col] = freq_map
        print(f"  Создан frequency encoding: {col}_freq_encoding")
        df.drop(columns=[col], inplace=True)
    elif unique_count > 2:
        # Label Encoding
        le = LabelEncoder()
        df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str).fillna('UNKNOWN'))
        label_encoders[col] = le
        print(f"  Создан label encoding: {col}_encoded")
        df.drop(columns=[col], inplace=True)
    else:
        # Бинарные признаки
        df[col] = df[col].astype(str).fillna('UNKNOWN')
        print(f"  Бинарный признак оставлен как есть")

print(f"\nОбработка категориальных признаков завершена!")
print(f"Создано label encoders: {len(label_encoders)}")
print(f"Создано frequency encodings: {len(frequency_encodings)}")
print(f"Текущий размер данных: {df.shape}")



numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if 'target' in numeric_cols:
    numeric_cols.remove('target')
if 'user_id' in numeric_cols:
    numeric_cols.remove('user_id')

print(f"\nПроверяем {len(numeric_cols)} числовых признаков на выбросы...")

outliers_info = {}
for col in numeric_cols[:20]:  # Первые 20 для скорости
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    if IQR > 0:
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outlier_count = len(outliers)
        if outlier_count > 0:
            outliers_info[col] = {
                'count': outlier_count,
                'percentage': (outlier_count / len(df)) * 100
            }

print(f"\n Обнаружено выбросов в {len(outliers_info)} колонках")
if len(outliers_info) > 0:
    sorted_outliers = sorted(outliers_info.items(), key=lambda x: x[1]['count'], reverse=True)[:5]
    print("Топ-5 колонок с выбросами:")
    for col, info in sorted_outliers:
        print(f"  {col}: {info['count']} выбросов ({info['percentage']:.2f}%)")

print("\nВАЖНО: Для fraud detection выбросы могут быть признаками мошенничества")
print("Выбросы НЕ удаляются, но информация сохранена для анализа")
print("\nПроверка выбросов завершена")



numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
exclude_cols = ['target', 'user_id']
exclude_patterns = ['_sin', '_cos', 'is_', '_encoded']

cols_to_scale = []
for col in numeric_cols:
    if col in exclude_cols:
        continue
    if any(pattern in col for pattern in exclude_patterns):
        continue
    unique_vals = df[col].dropna().unique()
    if len(unique_vals) <= 2 and set(unique_vals).issubset({0, 1, 0.0, 1.0}):
        continue
    cols_to_scale.append(col)

print(f"\nНайдено {len(cols_to_scale)} признаков для масштабирования")

# Используем MinMaxScaler
scaler = MinMaxScaler()
df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

print(f" Масштабировано {len(cols_to_scale)} признаков с помощью MinMaxScaler")



df.fillna(0, inplace=True)

if 'user_id' in df.columns:
    user_ids = df['user_id'].copy()
    df_for_model = df.drop(columns=['user_id'], errors='ignore')
else:
    df_for_model = df.copy()

print(f"\n Финальная структура данных:")
print(f"  Размер: {df_for_model.shape}")
print(f"  Колонок: {len(df_for_model.columns)}")
print(f"  Строк: {len(df_for_model)}")

if 'target' not in df_for_model.columns:
    raise ValueError(" Колонка 'target' отсутствует!")

output_dir = os.path.dirname(OUTPUT_PATH)
if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)

df_for_model.to_csv(OUTPUT_PATH, index=False)
print(f"\n Финальный датасет сохранен: {OUTPUT_PATH}")




class FraudDetectionPreprocessor:
    """Preprocessing pipeline для fraud detection модели."""
    
    def __init__(self, label_encoders=None, frequency_encodings=None, scaler=None, feature_columns=None):
        self.label_encoders = label_encoders or {}
        self.frequency_encodings = frequency_encodings or {}
        self.scaler = scaler
        self.feature_columns = feature_columns
        
    def fit(self, df):
        if self.feature_columns is None:
            self.feature_columns = [col for col in df.columns if col not in ['target', 'user_id']]
        return self
    
    def transform(self, df):
        df_processed = df.copy()
        
        for col, le in self.label_encoders.items():
            if col in df_processed.columns:
                df_processed[f'{col}_encoded'] = le.transform(
                    df_processed[col].astype(str).fillna('UNKNOWN')
                )
                df_processed.drop(columns=[col], inplace=True, errors='ignore')
        
        for col, freq_map in self.frequency_encodings.items():
            if col in df_processed.columns:
                df_processed[f'{col}_freq_encoding'] = df_processed[col].map(freq_map).fillna(0)
                df_processed.drop(columns=[col], inplace=True, errors='ignore')
        
        df_processed.fillna(0, inplace=True)
        
        if self.scaler is not None and self.feature_columns is not None:
            available_cols = [col for col in self.feature_columns if col in df_processed.columns]
            if available_cols:
                df_processed[available_cols] = self.scaler.transform(df_processed[available_cols])
        
        if self.feature_columns is not None:
            keep_cols = self.feature_columns.copy()
            if 'target' in df_processed.columns:
                keep_cols.append('target')
            if 'user_id' in df_processed.columns:
                keep_cols.append('user_id')
            df_processed = df_processed[[col for col in keep_cols if col in df_processed.columns]]
        
        return df_processed
    
    def fit_transform(self, df):
        return self.fit(df).transform(df)

preprocessor = FraudDetectionPreprocessor(
    label_encoders=label_encoders,
    frequency_encodings=frequency_encodings,
    scaler=scaler,
    feature_columns=[col for col in df_for_model.columns if col != 'target']
)

os.makedirs(os.path.dirname(PIPELINE_PATH), exist_ok=True)
joblib.dump(preprocessor, PIPELINE_PATH)
print(f"\n Preprocessing pipeline сохранен: {PIPELINE_PATH}")



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

validation_result = validate_input_data(df_for_model, 
                                        required_features=[col for col in df_for_model.columns if col != 'target'])

print(f"\nРезультат валидации:")
print(f"  Валидно: {validation_result['is_valid']}")
print(f"  Ошибок: {len(validation_result['errors'])}")
print(f"  Предупреждений: {len(validation_result['warnings'])}")

# Сохраняем функцию валидации
validation_file_path = 'src/data_validation.py'
os.makedirs(os.path.dirname(validation_file_path), exist_ok=True)
with open(validation_file_path, 'w', encoding='utf-8') as f:
    f.write('''"""
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
''')

print(f"\n Функция валидации сохранена: {validation_file_path}")




final_features = [col for col in df_for_model.columns if col != 'target']

print(f"\n ФИНАЛЬНЫЙ НАБОР ПРИЗНАКОВ:")
print(f"  Всего признаков: {len(final_features)}")
print(f"  Размер данных: {df_for_model.shape}")

# Сохраняем список финальных признаков
features_list_path = 'model/final_features_list.txt'
with open(features_list_path, 'w', encoding='utf-8') as f:
    f.write("ФИНАЛЬНЫЙ НАБОР ПРИЗНАКОВ ДЛЯ ML МОДЕЛИ\n")
    f.write("="*80 + "\n\n")
    f.write(f"Всего признаков: {len(final_features)}\n\n")
    for feat in final_features:
        f.write(f"  - {feat}\n")

joblib.dump(final_features, 'model/final_features.pkl')
print(f"\n Список признаков сохранен: {features_list_path}")
print(f" Список признаков сохранен (pickle): model/final_features.pkl")




