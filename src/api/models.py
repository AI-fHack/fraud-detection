from pydantic import BaseModel, Field, validator
from typing import Optional
from datetime import datetime


class Transaction(BaseModel):
    """
    Модель транзакции для детекции мошенничества.
    Основана на структуре данных из transactions.csv и behavior_patterns.csv
    """
    # Основные поля транзакции (обязательные)
    amount: float = Field(
        ...,
        description="Сумма транзакции",
        gt=0,
        example=1000.0
    )
    user_id: int = Field(
        ...,
        description="Уникальный идентификатор клиента (client_id)",
        alias="client_id",
        example=2937833270
    )
    
    # Поля транзакции (опциональные, но рекомендуемые)
    transaction_id: Optional[str] = Field(
        None,
        description="Уникальный идентификатор транзакции (docno)",
        alias="docno",
        example="5343"
    )
    transaction_date: Optional[str] = Field(
        None,
        description="Дата транзакции (transdate)",
        alias="transdate",
        example="2025-01-05 00:00:00.000"
    )
    transaction_datetime: Optional[str] = Field(
        None,
        description="Дата и время транзакции (transdatetime)",
        alias="transdatetime",
        example="2025-01-05 16:32:02.000"
    )
    destination: Optional[str] = Field(
        None,
        description="Зашифрованный идентификатор получателя (direction)",
        alias="direction",
        example="8406e407421ec28bd5f445793ef64fd1"
    )
    
    # Поведенческие данные (опциональные, для улучшения точности)
    device_type: Optional[str] = Field(
        None,
        description="Тип устройства (mobile, desktop, tablet)",
        example="mobile"
    )
    phone_model: Optional[str] = Field(
        None,
        description="Модель телефона из последней сессии",
        example="iPhone16"
    )
    os_version: Optional[str] = Field(
        None,
        description="Версия ОС из последней сессии",
        example="iOS/18.5"
    )
    
    # Поведенческие метрики (опциональные)
    unique_os_versions_30d: Optional[int] = Field(
        None,
        description="Количество разных версий ОС за последние 30 дней",
        ge=0,
        example=1
    )
    unique_phone_models_30d: Optional[int] = Field(
        None,
        description="Количество разных моделей телефона за последние 30 дней",
        ge=0,
        example=1
    )
    logins_last_7_days: Optional[int] = Field(
        None,
        description="Количество уникальных логин-сессий за последние 7 дней",
        ge=0,
        example=2
    )
    logins_last_30_days: Optional[int] = Field(
        None,
        description="Количество уникальных логин-сессий за последние 30 дней",
        ge=0,
        example=2
    )
    avg_logins_per_day_7d: Optional[float] = Field(
        None,
        description="Среднее число логинов в день за последние 7 дней",
        ge=0,
        example=0.2857
    )
    avg_logins_per_day_30d: Optional[float] = Field(
        None,
        description="Среднее число логинов в день за последние 30 дней",
        ge=0,
        example=0.0667
    )
    
    @validator('amount')
    def validate_amount(cls, v):
        """Валидация суммы транзакции."""
        if v <= 0:
            raise ValueError('Amount must be greater than 0')
        if v > 100000000:  # Максимальная сумма (можно настроить)
            raise ValueError('Amount is too large')
        return v
    
    @validator('user_id')
    def validate_user_id(cls, v):
        """Валидация ID пользователя."""
        if v <= 0:
            raise ValueError('User ID must be greater than 0')
        return v
    
    @validator('transaction_date', 'transaction_datetime')
    def validate_date_format(cls, v):
        """Валидация формата даты (опционально)."""
        if v is None:
            return v
        # Можно добавить более строгую валидацию формата даты
        return v
    
    @validator('logins_last_7_days', 'logins_last_30_days')
    def validate_logins_consistency(cls, v, values):
        """Валидация согласованности данных о логинах."""
        if v is not None and 'logins_last_30_days' in values:
            if values.get('logins_last_30_days') is not None:
                if v > values['logins_last_30_days']:
                    raise ValueError('logins_last_7_days cannot be greater than logins_last_30_days')
        return v
    
    class Config:
        populate_by_name = True  # Позволяет использовать как user_id, так и client_id
        schema_extra = {
            "example": {
                "amount": 1000.0,
                "client_id": 2937833270,
                "transaction_id": "5343",
                "transaction_date": "2025-01-05 00:00:00.000",
                "transaction_datetime": "2025-01-05 16:32:02.000",
                "destination": "8406e407421ec28bd5f445793ef64fd1",
                "device_type": "mobile",
                "phone_model": "iPhone16",
                "os_version": "iOS/18.5",
                "unique_os_versions_30d": 1,
                "unique_phone_models_30d": 1,
                "logins_last_7_days": 2,
                "logins_last_30_days": 2,
                "avg_logins_per_day_7d": 0.2857,
                "avg_logins_per_day_30d": 0.0667
            }
        }


class PredictionResponse(BaseModel):
    """Модель ответа для предсказания."""
    fraud_probability: float = Field(
        ...,
        description="Вероятность мошенничества (0-1)",
        ge=0,
        le=1,
        example=0.23
    )
    is_fraud: bool = Field(
        ...,
        description="Флаг мошенничества (True если вероятность > 0.5)",
        example=False
    )
    transaction_id: Optional[str] = Field(
        None,
        description="ID транзакции",
        example="5343"
    )
    status: str = Field(
        ...,
        description="Статус транзакции: 'fraud' или 'legitimate'",
        example="legitimate"
    )
    confidence: Optional[str] = Field(
        None,
        description="Уровень уверенности: 'high', 'medium', 'low'",
        example="high"
    )
