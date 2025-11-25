# fraud-detection AI: Система выявления мошеннических транзакций

**Разработано командой: V - UP**

---

## О проекте

В последние годы доля онлайн-мошенничества значительно выросла. Злоумышленники используют социальную инженерию, фишинг и поддельные сервисы для получения доступа к средствам клиентов. Традиционные антифрод-системы не всегда успевают адаптироваться под новые схемы, особенно когда речь идёт о небольших, но частых переводах через мобильный интернет-банкинг.

**Fraud-detection AI** — это ML-сервис на основе градиентного бустинга (LightGBM), который анализирует транзакции в реальном времени и предсказывает вероятность мошенничества. Система предоставляет не только оценку риска, но и объяснение решения через SHAP-анализ, что помогает аналитикам быстро понять причину подозрения.

---

## Цель проекта

Разработать ML-модель, которая на основе исторических данных о транзакциях и пользовательском поведении блокирует мошеннические переводы и определяет соответствующую вероятность мошенничества. Система должна быть легко интегрируемой в существующие банковские процессы через REST API.

---

## Бизнес-ценность

**Снижение финансовых потерь за счёт раннего обнаружения подозрительных переводов.**

**Адаптивность к новым типам угроз.**

**Снижение доли ложных срабатываний (False Positives) / необоснованных блокировок, сохраняя комфорт клиентов и предотвращая негативный опыт.**

**Автоматизация антифрод-процессов, сокращая время реакции на инциденты и снижая нагрузки с направления по противодействию мошенничества.**

---

## Технический стек

* **Python 3.13+** - Основной язык разработки
* **FastAPI** - Backend & REST API фреймворк
* **LightGBM** - ML модель для градиентного бустинга
* **SHAP** - Библиотека для интерпретации решений модели
* **Docker & docker-compose** - Контейнеризация и оркестрация
* **Pandas, NumPy** - Обработка данных
* **Scikit-learn** - Предобработка и метрики
* **Optuna** - Оптимизация гиперпараметров

---

## Модель

### Характеристики модели

- **Тип:** LightGBM (Gradient Boosting)
- **Версия:** 1.0
- **Количество признаков:** 83
- **Оптимальный порог классификации:** 0.550

### Метрики производительности

- **ROC-AUC:** 0.9765
- **F2-score:** 0.5917 (главная метрика, фокус на Recall)
- **Recall:** 0.6061
- **Precision:** 0.5405
- **F1-score:** 0.5714

Подробная документация модели доступна в [`model/MODEL_DOCUMENTATION.md`](model/MODEL_DOCUMENTATION.md)

---

## Установка и настройка

### Предварительные требования

- Python 3.13 или выше
- pip (менеджер пакетов Python)
- Docker и Docker Compose (опционально, для контейнеризации)

### 1. Клонирование репозитория

```bash
git clone https://github.com/AI-fHack/fraud-detection.git
cd fraud-detection
```

### 2. Создание виртуального окружения

Виртуальное окружение уже создано в папке `venv/`. Если нужно создать заново:

```bash
python -m venv venv
```

### 3. Активация виртуального окружения

**PowerShell:**
```powershell
.\activate.ps1
# или
.\venv\Scripts\Activate.ps1
```

**CMD:**
```cmd
activate.bat
# или
venv\Scripts\activate.bat
```

**Linux/macOS:**
```bash
source venv/bin/activate
```

После активации вы увидите `(venv)` в начале командной строки.

### 4. Установка зависимостей

```bash
pip install -r requirements.txt
```

---

## Запуск приложения

### Вариант 1: Используя скрипт run.py

```bash
python run.py
```

### Вариант 2: Используя uvicorn напрямую

```bash
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

### Вариант 3: Используя Docker

```bash
docker-compose up --build
```

После запуска API будет доступен по адресу:
- **API**: http://localhost:8000
- **Интерактивная документация (Swagger)**: http://localhost:8000/docs
- **Альтернативная документация (ReDoc)**: http://localhost:8000/redoc

---

## API Endpoints

### Health Check

#### `GET /`
Информация об API и доступных эндпоинтах.

**Ответ:**
```json
{
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
```

#### `GET /health`
Проверка работоспособности сервиса.

**Ответ:**
```json
{
  "status": "healthy",
  "service": "fraud-detection-api"
}
```

#### `GET /model/status`
Проверка статуса модели (загружена/не загружена).

**Ответ:**
```json
{
  "model_loaded": true,
  "status": "ready",
  "message": "Model is ready for predictions"
}
```

### Предсказание мошенничества

#### `POST /predict`
Предсказать, является ли транзакция мошеннической.

**Запрос:**
```json
{
  "amount": 1000.0,
  "client_id": 2937833270,
  "transaction_id": "5343",
  "transaction_date": "2025-01-05 00:00:00.000",
  "transaction_datetime": "2025-01-05 16:32:02.000",
  "direction": "8406e407421ec28bd5f445793ef64fd1",
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
```

**Минимальный запрос (обязательные поля):**
```json
{
  "amount": 1000.0,
  "client_id": 2937833270
}
```

**Ответ:**
```json
{
  "fraud_probability": 0.23,
  "is_fraud": false,
  "transaction_id": "5343",
  "status": "legitimate",
  "confidence": "high"
}
```

#### `POST /explain`
Получить объяснение предсказания с использованием SHAP значений.

**Запрос:** (аналогичен `/predict`)

**Ответ:**
```json
{
  "fraud_probability": 0.23,
  "is_fraud": false,
  "base_value": 0.15,
  "top_factors": [
    {
      "feature_name": "amount_zscore_user",
      "shap_value": 0.12,
      "impact": "increases_fraud"
    },
    {
      "feature_name": "time_since_last_tx",
      "shap_value": -0.08,
      "impact": "decreases_fraud"
    }
  ],
  "transaction_id": "5343"
}
```

---

## Структура проекта

```
fraud-detection/
├── src/
│   ├── api/
│   │   ├── main.py          # FastAPI приложение и эндпоинты
│   │   ├── models.py        # Pydantic модели для валидации
│   │   └── predict.py       # Загрузка модели и предсказания
│   ├── ml/
│   │   └── preprocess.py    # Предобработка данных для модели
│   └── data_validation.py   # Валидация входных данных
├── data/
│   ├── transactions.csv              # Исходные данные транзакций
│   ├── behavior_patterns.csv        # Поведенческие паттерны клиентов
│   └── processed/
│       ├── transactions_with_features.csv      # Обработанные данные
│       └── transactions_with_features_final.csv
├── model/
│   ├── final_model.pkl              # Обученная модель
│   ├── preprocessing_pipeline.pkl   # Пайплайн предобработки
│   ├── selected_features.pkl        # Выбранные признаки
│   ├── threshold.pkl                 # Оптимальный порог
│   ├── MODEL_DOCUMENTATION.md        # Документация модели
│   └── *.png                         # Графики метрик и важности признаков
├── notebooks/
│   ├── 01_data_overview.ipynb        # EDA анализ
│   ├── 01_full_data_pipeline.ipynb   # Полный пайплайн обработки
│   ├── 02_feature_engineering.ipynb  # Инженерия признаков
│   ├── 03_final_model.ipynb          # Обучение финальной модели
│   └── eda.ipynb                     # Дополнительный EDA
├── docker-compose.yml                # Docker Compose конфигурация
├── Dockerfile                        # Docker образ
├── requirements.txt                  # Python зависимости
├── run.py                           # Скрипт запуска API
├── train_model.py                   # Скрипт обучения модели
├── run_feature_engineering.py       # Скрипт генерации признаков
├── test_api.py                      # Тесты API
├── FEATURES_DOCUMENTATION.md         # Документация по признакам
├── Fraud_Detection_API.postman_collection.json  # Postman коллекция
└── README.md                         # Этот файл
```

---

## Тестирование

Для тестирования API можно использовать:

1. **Встроенные тесты:**
```bash
python test_api.py
```

2. **Postman коллекция:**
Импортируйте `Fraud_Detection_API.postman_collection.json` в Postman для удобного тестирования всех эндпоинтов.

3. **Swagger UI:**
Откройте http://localhost:8000/docs в браузере для интерактивного тестирования API.

4. **cURL пример:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "amount": 1000.0,
    "client_id": 2937833270,
    "device_type": "mobile"
  }'
```

---

## Дополнительная документация

- **[Документация модели](model/MODEL_DOCUMENTATION.md)** - Подробное описание модели, метрик и использования
- **[Документация признаков](FEATURES_DOCUMENTATION.md)** - Описание всех признаков, используемых в модели

---

## Обучение модели

Если вы хотите переобучить модель на новых данных:

```bash
python train_model.py
```

Этот скрипт:
1. Загружает данные из `data/processed/transactions_with_features_final.csv`
2. Обучает модель LightGBM
3. Оптимизирует порог классификации
4. Сохраняет модель и метрики в папку `model/`

---

## Разработка

Проект разрабатывается командой в рамках хакатона. Каждый участник работает в своем бранче:

- `api-integration` - Backend/Integration (Раяна)
- `data-analysis` - Data Engineering (Мадина)
- `ml-dev` - ML модели и обучение (Нурамир)
- `main` - Основная ветка разработки

### Деактивация виртуального окружения

```bash
deactivate
```

---

## Команда

**V - UP** - Команда разработчиков системы детекции мошенничества

- **Раяна** - Backend/API разработка
- **Мадина** - Data Engineering / Feature Engineering
- **Нурамир** - ML модели и обучение
- **Тамила** - Документация и презентация

---

**Версия:** 1.0.0  
**Последнее обновление:** 2025
