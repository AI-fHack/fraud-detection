import requests
import json
from typing import Dict, Any

BASE_URL = "http://localhost:8000"


def print_response(title: str, response: requests.Response):
    """Печатает красивый вывод ответа."""
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    print(f"Status Code: {response.status_code}")
    try:
        print(json.dumps(response.json(), indent=2, ensure_ascii=False))
    except:
        print(response.text)


def test_root():
    """Тест корневого endpoint."""
    response = requests.get(f"{BASE_URL}/")
    print_response("GET / - Root endpoint", response)
    return response.status_code == 200


def test_health():
    """Тест health check endpoint."""
    response = requests.get(f"{BASE_URL}/health")
    print_response("GET /health - Health check", response)
    return response.status_code == 200


def test_model_status():
    """Тест статуса модели."""
    response = requests.get(f"{BASE_URL}/model/status")
    print_response("GET /model/status - Model status", response)
    return response.status_code == 200


def test_predict_valid():
    """Тест предсказания с валидными данными."""
    data = {
        "amount": 1000.0,
        "user_id": 12345,
        "device_type": "mobile"
    }
    response = requests.post(f"{BASE_URL}/predict", json=data)
    print_response("POST /predict - Valid data", response)
    return response.status_code in [200, 503]


def test_predict_invalid_missing_fields():
    """Тест предсказания с отсутствующими полями."""
    data = {
        "amount": 1000.0
    }
    response = requests.post(f"{BASE_URL}/predict", json=data)
    print_response("POST /predict - Missing required fields", response)
    return response.status_code == 422


def test_predict_invalid_types():
    """Тест предсказания с неверными типами данных."""
    data = {
        "amount": "not_a_number",
        "user_id": 12345,
        "device_type": "mobile"
    }
    response = requests.post(f"{BASE_URL}/predict", json=data)
    print_response("POST /predict - Invalid data types", response)
    return response.status_code == 422


def main():
    """Запуск всех тестов."""
    print("ТЕСТИРОВАНИЕ Fraud Detection API")
    
    
    tests = [
        ("Root endpoint", test_root),
        ("Health check", test_health),
        ("Model status", test_model_status),
        ("Predict (valid)", test_predict_valid),
        ("Predict (missing fields)", test_predict_invalid_missing_fields),
        ("Predict (invalid types)", test_predict_invalid_types),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
            status = "[PASS]" if result else "[FAIL]"
            print(f"\n{status}: {name}")
        except Exception as e:
            results.append((name, False))
            print(f"\n[ERROR]: {name} - {str(e)}")
    
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    print(f"Пройдено: {passed}/{total}")
    for name, result in results:
        status = "[OK]" if result else "[FAIL]"
        print(f"  {status} {name}")
    
    if passed == total:
        print("\nВсе тесты пройдены!")
    else:
        print(f"\n{total - passed} тест(ов) не пройдено")


if __name__ == "__main__":
    main()

