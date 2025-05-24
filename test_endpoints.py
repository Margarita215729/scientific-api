#!/usr/bin/env python3
"""
Простой скрипт для тестирования всех API эндпоинтов
"""

import requests
import json

BASE_URL = "http://localhost:8000"

def test_endpoint(url, description):
    """Тестирует один эндпоинт"""
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            print(f"✅ {description}: OK")
            return True
        else:
            print(f"❌ {description}: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ {description}: Error - {e}")
        return False

def main():
    """Основная функция тестирования"""
    print("🧪 Тестирование API эндпоинтов...\n")
    
    tests = [
        (f"{BASE_URL}/ping", "Ping endpoint"),
        (f"{BASE_URL}/status", "Status endpoint"),
        (f"{BASE_URL}/api", "API info endpoint"),
        (f"{BASE_URL}/astro/status", "Astro status"),
        (f"{BASE_URL}/astro/statistics", "Astro statistics"),
        (f"{BASE_URL}/astro/galaxies?limit=3", "Galaxy data"),
        (f"{BASE_URL}/ads/search-by-coordinates?ra=150.1&dec=2.2", "ADS search by coordinates"),
        (f"{BASE_URL}/ads/search-by-object?object_name=M31", "ADS search by object"),
        (f"{BASE_URL}/ads/search-by-catalog?catalog=SDSS", "ADS search by catalog"),
        (f"{BASE_URL}/ads/large-scale-structure", "ADS large-scale structure"),
    ]
    
    passed = 0
    total = len(tests)
    
    for url, description in tests:
        if test_endpoint(url, description):
            passed += 1
    
    print(f"\n📊 Результаты: {passed}/{total} тестов прошли успешно")
    
    if passed == total:
        print("🎉 Все эндпоинты работают корректно!")
    else:
        print("⚠️  Некоторые эндпоинты требуют внимания")

if __name__ == "__main__":
    main() 