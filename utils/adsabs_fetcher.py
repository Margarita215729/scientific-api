"""
utils/adsabs_fetcher.py

Утилита для поиска публикаций в ADSabs (Harvard ADS) через их API.

Для работы с ADSabs используется:
    - URL: https://api.adsabs.harvard.edu/v1/search/query
    - Заголовок авторизации с Bearer токеном (предоставленный ниже).

Документация ADSabs API:
https://api.adsabs.harvard.edu/
"""

import requests
import json

# Константы для доступа к ADSabs API. 
# В реальном проекте рекомендуется хранить токен в переменной окружения.
ADSABS_TOKEN = "zahkzrUHgH9h9LuUgIfYUYVkJ0f86fHLdOzDYjko"
ADSABS_URL = "https://api.adsabs.harvard.edu/v1/search/query"

def fetch_from_adsabs(query: str, max_results: int = 10) -> dict:
    """
    Выполняет поиск публикаций в ADSabs (Harvard ADS) по заданному запросу.
    
    Параметры:
      - query: Строка запроса (например, "exoplanets" или "dark matter").
      - max_results: Максимальное число результатов (по умолчанию 10).
    
    Возвращает:
      - Словарь с ключами "source" (ADSabs Harvard) и "results" – списком публикаций.
    
    Пример поля публикации:
      {
          "title": ["Title of the paper"],
          "author": ["Author 1", "Author 2"],
          "year": "2023",
          "doi": "10.xxxx/xxxxxx"
      }
    """
    headers = {"Authorization": f"Bearer {ADSABS_TOKEN}"}
    
    # Параметры запроса: q – поисковый запрос, rows – число результатов, fl – поля для извлечения
    params = {
        "q": query,
        "rows": max_results,
        "fl": "title,author,year,doi"
    }
    
    response = requests.get(ADSABS_URL, headers=headers, params=params)
    if response.status_code != 200:
        raise Exception(f"Ошибка запроса ADSabs: {response.status_code} {response.text}")
    
    data = response.json()
    # Извлекаем документы из ответа
    docs = data.get("response", {}).get("docs", [])
    return {"source": "ADSabs Harvard", "results": docs}
