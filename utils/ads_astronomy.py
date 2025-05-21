"""
utils/ads_astronomy.py

Модуль для интеграции API ADSabs (Harvard Astrophysics Data System) для работы
с астрономическими объектами в нашей системе.

Функции в этом модуле позволяют:
1. Искать публикации по координатам объектов (RA, DEC)
2. Находить публикации, связанные с каталогами (SDSS, Euclid, DESI, DES)
3. Получать цитирования и ссылки для ключевых публикаций
4. Строить сети цитирования для исследований по крупномасштабным структурам

Требуется токен доступа к ADS API, который должен быть установлен 
в переменной окружения ADSABS_TOKEN.
"""

import os
import requests
import json
import time
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple

# Базовый URL ADS API
ADS_API_URL = "https://api.adsabs.harvard.edu/v1"

# Пути API
SEARCH_URL = f"{ADS_API_URL}/search/query"
EXPORT_URL = f"{ADS_API_URL}/export/bibtex"
METRICS_URL = f"{ADS_API_URL}/metrics"

def get_ads_token() -> str:
    """Получает токен ADS API из переменной окружения."""
    token = os.environ.get("ADSABS_TOKEN")
    if not token:
        raise ValueError("Не указан токен ADSABS_TOKEN в переменных окружения. Получите токен на https://ui.adsabs.harvard.edu/user/settings/token")
    return token

def search_by_coordinates(ra: float, dec: float, radius: float = 0.1) -> List[Dict[str, Any]]:
    """
    Поиск публикаций по небесным координатам с заданным радиусом.
    
    Параметры:
        ra (float): Прямое восхождение в градусах
        dec (float): Склонение в градусах
        radius (float): Радиус поиска в градусах (по умолчанию 0.1°)
        
    Возвращает:
        List[Dict]: Список публикаций, найденных по этим координатам
    """
    token = get_ads_token()
    headers = {"Authorization": f"Bearer {token}"}
    
    # Создаем запрос по позиции объекта
    # Используем синтаксис ADS для поиска по координатам
    query = f"pos(circle {ra} {dec} {radius})"
    
    params = {
        "q": query,
        "fl": "title,author,year,doi,bibcode,abstract,citation_count,read_count",
        "rows": 100,
        "sort": "citation_count desc"
    }
    
    response = requests.get(SEARCH_URL, headers=headers, params=params)
    if response.status_code != 200:
        raise Exception(f"Ошибка запроса ADS API: {response.status_code} {response.text}")
    
    data = response.json()
    docs = data.get("response", {}).get("docs", [])
    return docs

def search_by_object(object_name: str) -> List[Dict[str, Any]]:
    """
    Поиск публикаций по названию объекта (например, "M31" или "NGC 5128").
    
    Параметры:
        object_name (str): Название астрономического объекта
        
    Возвращает:
        List[Dict]: Список публикаций, связанных с этим объектом
    """
    token = get_ads_token()
    headers = {"Authorization": f"Bearer {token}"}
    
    # Используем поле object для поиска по имени объекта
    query = f"object:\"{object_name}\""
    
    params = {
        "q": query,
        "fl": "title,author,year,doi,bibcode,abstract,citation_count,read_count,object",
        "rows": 100,
        "sort": "citation_count desc"
    }
    
    response = requests.get(SEARCH_URL, headers=headers, params=params)
    if response.status_code != 200:
        raise Exception(f"Ошибка запроса ADS API: {response.status_code} {response.text}")
    
    data = response.json()
    docs = data.get("response", {}).get("docs", [])
    return docs

def search_by_catalog(catalog_name: str, facet: bool = True) -> Dict[str, Any]:
    """
    Поиск публикаций, связанных с определенным астрономическим каталогом.
    
    Параметры:
        catalog_name (str): Название каталога (SDSS, Euclid, DESI, DES)
        facet (bool): Включить группировку по ключевым словам
        
    Возвращает:
        Dict: Данные о публикациях и статистика по каталогу
    """
    token = get_ads_token()
    headers = {"Authorization": f"Bearer {token}"}
    
    # Формируем запрос в зависимости от каталога
    query_terms = {
        "SDSS": "\"Sloan Digital Sky Survey\" OR SDSS",
        "Euclid": "\"Euclid mission\" OR \"Euclid telescope\" OR \"Euclid survey\"",
        "DESI": "\"Dark Energy Spectroscopic Instrument\" OR DESI",
        "DES": "\"Dark Energy Survey\" OR DES"
    }
    
    query = query_terms.get(catalog_name, catalog_name)
    
    params = {
        "q": query,
        "fl": "title,author,year,doi,bibcode,abstract,citation_count,read_count,keyword",
        "rows": 200,
        "sort": "citation_count desc"
    }
    
    # Добавляем фасеты для анализа ключевых слов, если запрошено
    if facet:
        params["facet"] = "true"
        params["facet.field"] = "keyword"
        params["facet.limit"] = 30
    
    response = requests.get(SEARCH_URL, headers=headers, params=params)
    if response.status_code != 200:
        raise Exception(f"Ошибка запроса ADS API: {response.status_code} {response.text}")
    
    data = response.json()
    docs = data.get("response", {}).get("docs", [])
    
    result = {
        "catalog": catalog_name,
        "publications": docs,
        "total_found": data.get("response", {}).get("numFound", 0)
    }
    
    # Добавляем информацию по фасетам, если они были запрошены
    if facet and "facet_counts" in data:
        facet_fields = data.get("facet_counts", {}).get("facet_fields", {})
        keywords = facet_fields.get("keyword", [])
        
        # Преобразуем список [ключ, значение, ключ, значение...] в словарь {ключ: значение}
        keyword_dict = {}
        for i in range(0, len(keywords), 2):
            if i+1 < len(keywords):
                keyword_dict[keywords[i]] = keywords[i+1]
        
        result["keyword_stats"] = keyword_dict
    
    return result

def get_citations_for_paper(bibcode: str) -> List[Dict[str, Any]]:
    """
    Получает список публикаций, цитирующих заданную публикацию.
    
    Параметры:
        bibcode (str): Bibcode публикации в формате ADS
        
    Возвращает:
        List[Dict]: Список цитирующих публикаций
    """
    token = get_ads_token()
    headers = {"Authorization": f"Bearer {token}"}
    
    # Используем оператор цитирования
    query = f"citations(bibcode:{bibcode})"
    
    params = {
        "q": query,
        "fl": "title,author,year,doi,bibcode,abstract,citation_count",
        "rows": 200,
        "sort": "date desc"
    }
    
    response = requests.get(SEARCH_URL, headers=headers, params=params)
    if response.status_code != 200:
        raise Exception(f"Ошибка запроса ADS API: {response.status_code} {response.text}")
    
    data = response.json()
    docs = data.get("response", {}).get("docs", [])
    return docs

def get_references_for_paper(bibcode: str) -> List[Dict[str, Any]]:
    """
    Получает список публикаций, на которые ссылается заданная публикация.
    
    Параметры:
        bibcode (str): Bibcode публикации в формате ADS
        
    Возвращает:
        List[Dict]: Список публикаций, на которые ссылается искомая
    """
    token = get_ads_token()
    headers = {"Authorization": f"Bearer {token}"}
    
    # Используем оператор references
    query = f"references(bibcode:{bibcode})"
    
    params = {
        "q": query,
        "fl": "title,author,year,doi,bibcode,abstract,citation_count",
        "rows": 200,
        "sort": "citation_count desc"
    }
    
    response = requests.get(SEARCH_URL, headers=headers, params=params)
    if response.status_code != 200:
        raise Exception(f"Ошибка запроса ADS API: {response.status_code} {response.text}")
    
    data = response.json()
    docs = data.get("response", {}).get("docs", [])
    return docs

def search_large_scale_structure(keywords: List[str] = None, start_year: int = 2010) -> Dict[str, Any]:
    """
    Выполняет поиск публикаций, связанных с крупномасштабной структурой Вселенной.
    
    Параметры:
        keywords (List[str]): Дополнительные ключевые слова для поиска
        start_year (int): Начальный год публикаций (по умолчанию 2010)
        
    Возвращает:
        Dict: Результаты поиска с группировкой по годам и ключевым словам
    """
    token = get_ads_token()
    headers = {"Authorization": f"Bearer {token}"}
    
    # Базовый поисковый запрос для крупномасштабных структур
    base_query = "\"large scale structure\" OR \"cosmic web\" OR \"galaxy clusters\" OR \"filaments\" OR \"cosmic voids\""
    
    # Если есть дополнительные ключевые слова, добавляем их
    if keywords and len(keywords) > 0:
        additional_terms = " OR ".join([f"\"{kw}\"" for kw in keywords])
        query = f"({base_query}) AND ({additional_terms})"
    else:
        query = base_query
    
    # Добавляем фильтр по году
    query = f"{query} year:{start_year}-"
    
    params = {
        "q": query,
        "fl": "title,author,year,doi,bibcode,abstract,citation_count,read_count,keyword",
        "rows": 300,
        "sort": "citation_count desc",
        "facet": "true",
        "facet.field": ["year", "keyword"],
        "facet.limit": 50
    }
    
    response = requests.get(SEARCH_URL, headers=headers, params=params)
    if response.status_code != 200:
        raise Exception(f"Ошибка запроса ADS API: {response.status_code} {response.text}")
    
    data = response.json()
    docs = data.get("response", {}).get("docs", [])
    
    # Обрабатываем фасеты для статистики
    facet_fields = data.get("facet_counts", {}).get("facet_fields", {})
    
    # Преобразуем фасеты в словари
    year_stats = {}
    keyword_stats = {}
    
    if "year" in facet_fields:
        years = facet_fields["year"]
        for i in range(0, len(years), 2):
            if i+1 < len(years):
                year_stats[years[i]] = years[i+1]
    
    if "keyword" in facet_fields:
        keywords = facet_fields["keyword"]
        for i in range(0, len(keywords), 2):
            if i+1 < len(keywords):
                keyword_stats[keywords[i]] = keywords[i+1]
    
    # Формируем результат
    result = {
        "query": query,
        "total_found": data.get("response", {}).get("numFound", 0),
        "publications": docs,
        "year_stats": year_stats,
        "keyword_stats": keyword_stats
    }
    
    return result

def get_bibtex(bibcodes: List[str]) -> str:
    """
    Получает данные BibTeX для списка публикаций.
    
    Параметры:
        bibcodes (List[str]): Список bibcode публикаций
        
    Возвращает:
        str: Данные в формате BibTeX
    """
    token = get_ads_token()
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    data = {"bibcode": bibcodes}
    
    response = requests.post(EXPORT_URL, headers=headers, json=data)
    if response.status_code != 200:
        raise Exception(f"Ошибка запроса ADS API: {response.status_code} {response.text}")
    
    result = response.json()
    return result.get("export", "") 