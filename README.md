# Scientific API - Astronomical Data Analysis

Веб-приложение для анализа крупномасштабных структур Вселенной с доступом к реальным астрономическим каталогам.

## 🌟 Возможности

- **Доступ к реальным астрономическим каталогам**: SDSS DR17, Euclid Q1, DESI DR1, DES Y6
- **Интерактивная визуализация**: 3D-распределение галактик, анализ красного смещения
- **Поиск научной литературы**: Интеграция с NASA ADS API
- **Обработка данных**: Автоматическая очистка, нормализация и удаление дубликатов
- **Современный веб-интерфейс**: Адаптивный дизайн с Bootstrap

## 🚀 Деплой на Vercel

### Быстрый деплой

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/yourusername/scientific-api)

### Ручной деплой

1. **Установите Vercel CLI**:
   ```bash
   npm i -g vercel
   ```

2. **Клонируйте репозиторий**:
   ```bash
   git clone https://github.com/yourusername/scientific-api.git
   cd scientific-api
   ```

3. **Деплой**:
   ```bash
   vercel --prod
   ```

### Переменные окружения

Настройте следующие переменные в Vercel Dashboard:

```env
ADS_API_TOKEN=your_ads_token_here
ENVIRONMENT=production
PYTHONPATH=.
```

## 🛠 Локальная разработка

### Требования

- Python 3.9+
- pip

### Установка

1. **Клонируйте репозиторий**:
   ```bash
   git clone https://github.com/yourusername/scientific-api.git
   cd scientific-api
   ```

2. **Создайте виртуальное окружение**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # или
   venv\Scripts\activate  # Windows
   ```

3. **Установите зависимости**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Запустите приложение**:
   ```bash
   python main.py
   ```

Приложение будет доступно по адресу: http://localhost:8000

## 📊 Структура проекта

```
scientific-api/
├── api/                    # API модули
│   ├── astro_catalog_api.py   # Астрономические каталоги
│   ├── ads_api.py             # Поиск литературы
│   └── heavy_api.py           # Тяжелые вычисления
├── ui/                     # Веб-интерфейс
│   ├── index.html             # Главная страница
│   ├── ads.html               # Поиск литературы
│   ├── script.js              # JavaScript
│   └── styles.css             # Стили
├── utils/                  # Утилиты
│   ├── astronomy_catalogs_real.py  # Работа с каталогами
│   ├── data_preprocessor.py        # Предобработка данных
│   └── ads_astronomy_real.py       # ADS API клиент
├── main.py                 # Главный файл приложения
├── vercel.json            # Конфигурация Vercel
└── requirements.txt       # Зависимости Python
```

## 🔧 API Endpoints

### Астрономические данные
- `GET /api/astro/status` - Статус каталогов
- `GET /api/astro/statistics` - Статистика по данным
- `GET /api/astro/galaxies` - Данные галактик с фильтрацией
- `POST /api/astro/download` - Загрузка каталогов

### Научная литература
- `GET /api/ads/search` - Поиск публикаций
- `GET /api/ads/search-by-coordinates` - Поиск по координатам
- `GET /api/ads/search-by-catalog` - Поиск по каталогу

### Документация
- `/api/docs` - Swagger UI
- `/api/redoc` - ReDoc документация

## 📈 Особенности обработки данных

### Удаление дубликатов
- **По уникальному ID**: Удаление точных дубликатов по object_id
- **По координатам**: Удаление объектов в радиусе 1 угловой секунды
- **Приоритет**: Сохранение первого вхождения

### Поддерживаемые каталоги
- **SDSS DR17**: Спектроскопический каталог (SPECOBJID)
- **Euclid Q1**: MER Final каталог (OBJECT_ID)
- **DESI DR1**: ELG clustering каталог (TARGETID)
- **DES Y6**: Gold каталог (COADD_OBJECT_ID)

### Очистка данных
- Удаление строк с критически важными пропущенными значениями (RA, DEC, redshift)
- Валидация координат и красного смещения
- Фильтрация выбросов по статистическим критериям

## 🌐 Используемые источники данных

- **[SDSS](https://www.sdss.org/)**: Sloan Digital Sky Survey
- **[Euclid](https://www.euclid-ec.org/)**: European Space Agency mission
- **[DESI](https://www.desi.lbl.gov/)**: Dark Energy Spectroscopic Instrument
- **[DES](https://www.darkenergysurvey.org/)**: Dark Energy Survey
- **[NASA ADS](https://ui.adsabs.harvard.edu/)**: Astrophysics Data System

## 📝 Лицензия

MIT License - см. файл [LICENSE](LICENSE)

## 🤝 Вклад в проект

1. Fork репозитория
2. Создайте feature branch (`git checkout -b feature/amazing-feature`)
3. Commit изменения (`git commit -m 'Add amazing feature'`)
4. Push в branch (`git push origin feature/amazing-feature`)
5. Откройте Pull Request

## 📞 Поддержка

Если у вас есть вопросы или предложения, создайте [Issue](https://github.com/yourusername/scientific-api/issues) в репозитории.