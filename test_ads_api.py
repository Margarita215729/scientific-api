"""
Скрипт для тестирования API интеграции с ADS (Astrophysics Data System).
"""

import os
import json
import requests
from pprint import pprint

# Базовый URL API
BASE_URL = "http://localhost:8000"

def test_ads_status():
    """Тестирование доступности ADS API."""
    print("\n=== Проверка статуса ADS API ===")
    # Проверяем, установлена ли переменная окружения ADSABS_TOKEN
    token = os.environ.get("ADSABS_TOKEN")
    if not token:
        print("⚠️ ВНИМАНИЕ: Переменная окружения ADSABS_TOKEN не установлена.")
        print("Установите токен доступа ADS API для продолжения:")
        print("export ADSABS_TOKEN=your_token_here")
        print("Или в Windows:")
        print("set ADSABS_TOKEN=your_token_here")
        return False
    print("✅ Токен ADS API найден в переменных окружения")
    return True

def test_search_by_coordinates():
    """Тестирование поиска публикаций по координатам."""
    print("\n=== Тестирование поиска по координатам ===")
    
    # Координаты центра Туманности Андромеды (M31)
    params = {
        "ra": 10.6847,
        "dec": 41.2687,
        "radius": 0.2
    }
    
    response = requests.get(f"{BASE_URL}/ads/search-by-coordinates", params=params)
    print(f"Статус: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Найдено публикаций: {data.get('count', 0)}")
        if data.get('count', 0) > 0:
            print("Первые 3 публикации:")
            for i, pub in enumerate(data.get('publications', [])[:3]):
                print(f"  {i+1}. {pub.get('title', ['Без названия'])[0]} ({pub.get('year', 'Н/Д')})")
    else:
        print(f"Ошибка: {response.text}")

def test_search_by_object():
    """Тестирование поиска публикаций по названию объекта."""
    print("\n=== Тестирование поиска по названию объекта ===")
    
    # Поиск по имени галактики M31
    params = {
        "object_name": "M31"
    }
    
    response = requests.get(f"{BASE_URL}/ads/search-by-object", params=params)
    print(f"Статус: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Найдено публикаций: {data.get('count', 0)}")
        if data.get('count', 0) > 0:
            print("Первые 3 публикации:")
            for i, pub in enumerate(data.get('publications', [])[:3]):
                print(f"  {i+1}. {pub.get('title', ['Без названия'])[0]} ({pub.get('year', 'Н/Д')})")
    else:
        print(f"Ошибка: {response.text}")

def test_search_by_catalog():
    """Тестирование поиска публикаций, связанных с каталогом."""
    print("\n=== Тестирование поиска по каталогу ===")
    
    catalogs = ["SDSS", "Euclid", "DESI", "DES"]
    
    for catalog in catalogs:
        print(f"\nКаталог: {catalog}")
        params = {
            "catalog": catalog,
            "include_stats": True
        }
        
        response = requests.get(f"{BASE_URL}/ads/search-by-catalog", params=params)
        print(f"Статус: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Всего публикаций: {data.get('total_found', 0)}")
            print(f"Получено публикаций: {len(data.get('publications', []))}")
            
            # Если есть статистика по ключевым словам, показываем топ-5
            if "keyword_stats" in data:
                print("Топ-5 ключевых слов:")
                top_keywords = sorted(data["keyword_stats"].items(), key=lambda x: x[1], reverse=True)[:5]
                for keyword, count in top_keywords:
                    print(f"  {keyword}: {count}")
        else:
            print(f"Ошибка: {response.text}")

def test_large_scale_structure():
    """Тестирование поиска публикаций о крупномасштабных структурах."""
    print("\n=== Тестирование поиска по крупномасштабным структурам ===")
    
    params = {
        "additional_keywords": ["filaments", "voids"],
        "start_year": 2018
    }
    
    response = requests.get(f"{BASE_URL}/ads/large-scale-structure", params=params)
    print(f"Статус: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Запрос: {data.get('query', '')}")
        print(f"Всего найдено: {data.get('total_found', 0)}")
        
        # Статистика по годам
        if "year_stats" in data:
            print("\nСтатистика по годам:")
            years = sorted(data["year_stats"].items(), key=lambda x: x[0], reverse=True)
            for year, count in years[:5]:
                print(f"  {year}: {count}")
        
        # Вывод первых 3 публикаций
        if data.get('publications', []):
            print("\nПервые 3 публикации:")
            for i, pub in enumerate(data.get('publications', [])[:3]):
                print(f"  {i+1}. {pub.get('title', ['Без названия'])[0]} ({pub.get('year', 'Н/Д')})")
                print(f"     Цитирования: {pub.get('citation_count', 0)}")
    else:
        print(f"Ошибка: {response.text}")

def main():
    """Основная функция для тестирования всех эндпоинтов ADS API."""
    if not test_ads_status():
        return
    
    test_search_by_coordinates()
    test_search_by_object()
    test_search_by_catalog()
    test_large_scale_structure()

if __name__ == "__main__":
    main() 