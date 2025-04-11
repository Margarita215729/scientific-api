# tests/test_api.py
import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_root():
    """Проверка, что корневой endpoint возвращает успешный ответ."""
    response = client.get("/")
    assert response.status_code == 200, f"Ошибка при обращении к корневому endpoint: {response.text}"
    print("Корневой endpoint работает корректно.")

@pytest.mark.parametrize("endpoint,query_param", [
    ("/datasets/arxiv", "quantum computing"),
    ("/datasets/academic", "quantum computing"),
    ("/datasets/adsabs", "black holes"),
    ("/datasets/cern", "particle physics"),
    ("/datasets/google", "climate change"),
    ("/datasets/nasa", "earth"),
    ("/datasets/openml", ""),
    ("/datasets/biorxiv", "genomics")
])
def test_dataset_endpoints(endpoint, query_param):
    """
    Проверка каждого endpoint'а для получения датасетов/публикаций.
    Отправляем запрос с заданным query и max_results=5.
    """
    response = client.get(endpoint, params={"query": query_param, "max_results": 5})
    assert response.status_code == 200, f"Ошибка на {endpoint} с query='{query_param}': {response.text}"
    
    json_data = response.json()
    assert "results" in json_data, f"Некорректный формат ответа на {endpoint}: {json_data}"
    
    # Выводим в терминал число полученных результатов
    results_count = len(json_data.get("results", []))
    print(f"{endpoint}: найдено {results_count} результатов для запроса '{query_param}'.")

def test_data_analysis_invalid_file():
    """
    Тестирование endpoint'а /data/analyze.
    Ожидаем ошибку (400), так как передаем невалидный file_id.
    """
    response = client.get("/data/analyze", params={"file_id": "invalid_file_id"})
    assert response.status_code == 400, f"Ожидалась ошибка для недействительного file_id: {response.text}"
    print(f"/data/analyze: корректно возвращает ошибку для недействительного file_id: {response.text}")

def test_ml_train_invalid():
    """
    Тестирование endpoint'а /ml/train.
    Поскольку sample_data.csv отсутствует или target_column неверный, ожидается ошибка.
    """
    payload = {"file_id": "dummy", "target_column": "nonexistent"}
    response = client.post("/ml/train", json=payload)
    assert response.status_code == 400, f"Ожидалась ошибка при отсутствии файла или неверном target_column: {response.text}"
    print(f"/ml/train: корректно возвращает ошибку при некорректных входных данных: {response.text}")

def test_file_manager_list():
    """
    Тестирование endpoint'а /files/list для просмотра файлов текущей директории.
    """
    response = client.get("/files/list", params={"directory": "."})
    assert response.status_code == 200, f"Ошибка в /files/list: {response.text}"
    json_data = response.json()
    assert isinstance(json_data, dict), f"Ожидался словарь, получено: {type(json_data)}"
    print(f"/files/list: получена структура файлов: {json_data}")

# Дополнительно можно добавить тесты для чтения, разбиения, создания и редактирования файлов через /files endpoints,
# а также другие тесты для дополнительных функций API.