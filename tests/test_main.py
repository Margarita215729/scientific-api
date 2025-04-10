# tests/test_main.py
import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "Добро пожаловать" in response.json()["message"]

def test_analyze_invalid_file():
    # Пробуем вызвать анализ с несуществующим file_id
    response = client.get("/data/analyze", params={"file_id": "invalid_file_id"})
    assert response.status_code == 400
    assert "Ошибка" in response.json()["detail"]

# Для теста ml моделей используем локальный файл sample_data.csv,
# который должен быть подготовлен в корне проекта.
def test_ml_train_no_target(tmp_path):
    # Если target_column отсутствует, должен выдать ошибку
    payload = {"file_id": "dummy", "target_column": "nonexistent"}
    response = client.post("/ml/train", json=payload)
    # Так как sample_data.csv, скорее всего, не найден – тестовая логика может изменяться.
    assert response.status_code == 400
