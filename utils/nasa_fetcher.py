"""
utils/nasa_fetcher.py

Реализует поиск датасетов на NASA Open Data.
Используем публичный endpoint: https://data.nasa.gov/api/views.json
Фильтруем результаты по вхождению запроса в заголовок или описание.
"""

import requests
import json

def fetch_from_nasa(query: str, max_results: int = 10) -> dict:
    """
    Получает список датасетов с NASA Open Data и фильтрует их по запросу.
    
    :param query: Строка запроса.
    :param max_results: Максимальное число результатов.
    :return: Словарь с ключами "source" и "results" (отфильтрованные датасеты).
    """
    base_url = "https://data.nasa.gov/api/views.json"
    response = requests.get(base_url)
    if response.status_code != 200:
        raise Exception(f"Ошибка запроса NASA Open Data: {response.status_code}")
    data = response.json()
    results = []
    query_lower = query.lower()
    count = 0
    for record in data:
        title = record.get("name", "")
        description = record.get("description", "")
        if query_lower in title.lower() or query_lower in description.lower():
            results.append({
                "id": record.get("id"),
                "title": title,
                "description": description,
                "url": record.get("link", "")
            })
            count += 1
            if count >= max_results:
                break
    return {"source": "NASA Open Data", "results": results}