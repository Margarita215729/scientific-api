# config.py
import os
import warnings
from dotenv import load_dotenv

# Загружаем переменные окружения из .env (при локальном запуске)
load_dotenv()

# Получаем переменные окружения с значениями по умолчанию
GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET", "")
GOOGLE_REFRESH_TOKEN = os.environ.get("GOOGLE_REFRESH_TOKEN", "")

# Проверка наличия критически важных переменных
if not all([GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, GOOGLE_REFRESH_TOKEN]):
    warnings.warn("Не установлены одна или несколько переменных окружения: GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, GOOGLE_REFRESH_TOKEN. Некоторые функции API могут быть недоступны.")

# Добавляем переменную для ADSabs API токена
ADSABS_TOKEN = os.environ.get("ADSABS_TOKEN", "")
if not ADSABS_TOKEN:
    warnings.warn("Не установлена переменная ADSABS_TOKEN. Поиск в ADSabs не будет работать.")

# Добавляем переменную для SerpAPI
SERPAPI_KEY = os.environ.get("SERPAPI_KEY", "")
if not SERPAPI_KEY:
    warnings.warn("Не установлена переменная SERPAPI_KEY. Поиск в Google Dataset Search не будет работать.")
