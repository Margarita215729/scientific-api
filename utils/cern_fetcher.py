"""
utils/cern_fetcher.py

Реализует запросы к CERN Open Data через их открытый API.
Документация: https://opendata.cern.ch/api/records/1.0/search/
"""

import requests
import json

def fetch_from_cern(query: str, max_results: int = 10) -> dict:
    """
    Ищет датасеты в CERN Open Data по запросу.
    
    :param query: Строка запроса.
    :param max_results: Максимальное число результатов.
    :return: Словарь с ключами "source" и "results" (список найденных записей).
    """
    base_url = "https://opendata.cern.ch/api/records/1.0/search/"
    params = {
        "q": query,
        "rows": max_results
    }
    response = requests.get(base_url, params=params)
    if response.status_code != 200:
        raise Exception(f"Ошибка запроса CERN Open Data: {response.status_code} {response.text}")
    data = response.json()
    records = data.get("records", [])
    results = []
    for rec in records:
        fields = rec.get("fields", {})
        results.append({
            "title": fields.get("title"),
            "description": fields.get("description", ""),
            "id": rec.get("recordid"),
            "url": f"https://opendata.cern.ch/record/{rec.get('recordid')}"
        })
    return {"source": "CERN Open Data", "results": results}
