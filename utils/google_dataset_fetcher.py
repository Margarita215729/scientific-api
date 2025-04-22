"""
utils/google_dataset_fetcher.py

Утилита для поиска датасетов через Google Dataset Search с использованием SerpAPI.
Документация SerpAPI: https://serpapi.com/
"""

import requests
import json
from config import SERPAPI_KEY

def fetch_from_google_dataset_search(query: str, max_results: int = 10) -> dict:
    """
    Выполняет поиск датасетов через Google Dataset Search с использованием SerpAPI.
    
    :param query: Строка запроса.
    :param max_results: Максимальное число результатов.
    :return: Словарь с ключами "source" и "results".
    """
    if not SERPAPI_KEY:
        raise Exception("Не задана переменная SERPAPI_KEY для доступа к SerpAPI")
    
    base_url = "https://serpapi.com/search.json"
    params = {
        "engine": "google_dataset",
        "q": query,
        "api_key": SERPAPI_KEY,
        "num": max_results
    }
    response = requests.get(base_url, params=params)
    if response.status_code != 200:
        raise Exception(f"Ошибка запроса Google Dataset Search: {response.status_code} {response.text}")
    data = response.json()
    
    # В ответе SerpAPI может быть поле "organizations" или "results", здесь обрабатываем поле results
    results = data.get("results", [])
    formatted = []
    for item in results:
        formatted.append({
            "title": item.get("title"),
            "snippet": item.get("snippet", ""),
            "link": item.get("link")
        })
    return {"source": "Google Dataset Search (via SerpAPI)", "results": formatted}
