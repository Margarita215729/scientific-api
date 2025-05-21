"""
Скрипт для тестирования API астрономических каталогов.
"""

import os
import json
import requests
from pprint import pprint

# Базовый URL API
BASE_URL = "http://localhost:8000"

def test_root_endpoint():
    """Тестирование корневого эндпоинта API."""
    print("\n=== Тестирование корневого эндпоинта ===")
    response = requests.get(f"{BASE_URL}/")
    print(f"Статус: {response.status_code}")
    if response.status_code == 200:
        print(response.json())
    else:
        print(f"Ошибка: {response.text}")

def test_astro_status():
    """Тестирование статуса астрономических каталогов."""
    print("\n=== Тестирование статуса астрономических каталогов ===")
    response = requests.get(f"{BASE_URL}/astro/status")
    print(f"Статус: {response.status_code}")
    if response.status_code == 200:
        status_data = response.json()
        print(f"Статус каталогов: {status_data['status']}")
        for catalog in status_data.get('catalogs', []):
            avail = "✅" if catalog.get("available", False) else "❌"
            name = catalog.get("name", "Неизвестный каталог")
            print(f"  {avail} {name}")
    else:
        print(f"Ошибка: {response.text}")

def test_astro_galaxies(limit=5, source=None, min_z=None, max_z=None):
    """Тестирование получения галактик."""
    print(f"\n=== Тестирование получения галактик (limit={limit}) ===")
    
    params = {"limit": limit}
    if source:
        params["source"] = source
    if min_z is not None:
        params["min_z"] = min_z
    if max_z is not None:
        params["max_z"] = max_z
    
    response = requests.get(f"{BASE_URL}/astro/galaxies", params=params)
    print(f"Статус: {response.status_code}")
    if response.status_code == 200:
        galaxies_data = response.json()
        print(f"Получено галактик: {galaxies_data['count']}")
        print("Первые несколько галактик:")
        for i, galaxy in enumerate(galaxies_data.get('galaxies', [])[:3]):
            print(f"  {i+1}. RA={galaxy.get('RA')}, DEC={galaxy.get('DEC')}, z={galaxy.get('redshift')}, источник={galaxy.get('source')}")
    else:
        print(f"Ошибка: {response.text}")

def test_astro_statistics(catalogs=None):
    """Тестирование получения статистики."""
    print("\n=== Тестирование получения статистики ===")
    
    params = {}
    if catalogs:
        params["catalogs"] = catalogs
    
    response = requests.get(f"{BASE_URL}/astro/statistics", params=params)
    print(f"Статус: {response.status_code}")
    if response.status_code == 200:
        stats_data = response.json()
        print(f"Всего галактик: {stats_data.get('total_galaxies')}")
        print(f"Источники: {stats_data.get('sources')}")
        print("Диапазон красных смещений:")
        z_stats = stats_data.get('redshift', {})
        print(f"  Min: {z_stats.get('min')}, Max: {z_stats.get('max')}, Среднее: {z_stats.get('mean')}")
    else:
        print(f"Ошибка: {response.text}")

def main():
    """Основная функция для тестирования всех эндпоинтов API."""
    test_root_endpoint()
    test_astro_status()
    test_astro_galaxies(limit=5)
    test_astro_statistics()
    
    print("\n=== Тестирование фильтрации по источнику ===")
    test_astro_galaxies(limit=5, source="SDSS")
    
    print("\n=== Тестирование фильтрации по красному смещению ===")
    test_astro_galaxies(limit=5, min_z=0.5, max_z=0.8)

if __name__ == "__main__":
    main() 