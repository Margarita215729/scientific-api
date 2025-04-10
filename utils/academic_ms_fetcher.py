"""
utils/academic_ms_fetcher.py

Получение данных о научных публикациях.
В данной реализации используется OpenAlex API, который является открытой альтернативой ранее доступному Microsoft Academic.
Документация OpenAlex: https://docs.openalex.org/api
"""

import requests
import json

def fetch_from_academic_ms(query: str, max_results: int = 10) -> dict:
    """
    Ищет публикации по заданному запросу через OpenAlex API.
    
    :param query: Строка запроса (например, "quantum computing").
    :param max_results: Максимальное число результатов.
    :return: Словарь с ключами "source" (OpenAlex) и "results" (список публикаций).
    """
    base_url = "https://api.openalex.org/works"
    params = {
        "search": query,
        "per-page": max_results
    }
    response = requests.get(base_url, params=params)
    if response.status_code != 200:
        raise Exception(f"Ошибка запроса OpenAlex: {response.status_code} {response.text}")
    data = response.json()
    results = []
    for work in data.get("results", []):
        results.append({
            "id": work.get("id"),
            "title": work.get("display_name"),
            "publication_year": work.get("publication_year"),
            "doi": work.get("doi"),
            "url": work.get("primary_location", {}).get("source", {}).get("homepage_url"),
            "abstract": work.get("abstract_inverted_index", None)  # Опционально, требует дополнительной обработки
        })
    return {"source": "OpenAlex", "results": results}
