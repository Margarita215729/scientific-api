"""
utils/adsabs_fetcher.py

Утилита для поиска публикаций в ADSabs (Harvard ADS) через их API.

Для работы с ADSabs используется:
    - URL: https://api.adsabs.harvard.edu/v1/search/query
    - Заголовок авторизации с Bearer токеном из переменной окружения.

Документация ADSabs API:
https://api.adsabs.harvard.edu/
"""

import os
import requests
import json

# Получаем токен из переменной окружения
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
    adsabs_token = os.environ.get("ADSABS_TOKEN")
    if not adsabs_token:
        raise Exception("Не указан токен ADSABS_TOKEN в переменных окружения")
        
    headers = {"Authorization": f"Bearer {adsabs_token}"}
    
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
