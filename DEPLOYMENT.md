# Инструкции по деплою на Vercel

## 🚀 Быстрый старт

### 1. Подготовка репозитория

1. **Убедитесь, что все файлы на месте**:
   ```bash
   ls -la
   # Должны быть: main.py, vercel.json, requirements.txt, api/, ui/, utils/
   ```

2. **Проверьте структуру проекта**:
   ```
   scientific-api/
   ├── main.py                 # ✅ Главный файл для Vercel
   ├── vercel.json            # ✅ Конфигурация Vercel
   ├── requirements.txt       # ✅ Зависимости Python
   ├── api/                   # ✅ API модули
   ├── ui/                    # ✅ Веб-интерфейс
   └── utils/                 # ✅ Утилиты
   ```

### 2. Деплой через GitHub

1. **Загрузите код в GitHub**:
   ```bash
   git add .
   git commit -m "Prepare for Vercel deployment"
   git push origin main
   ```

2. **Перейдите на [vercel.com](https://vercel.com)**

3. **Нажмите "New Project"**

4. **Импортируйте ваш GitHub репозиторий**

5. **Настройте переменные окружения** (опционально):
   - `ADS_API_TOKEN` - для поиска научной литературы
   - `ENVIRONMENT=production`

6. **Нажмите "Deploy"**

### 3. Деплой через CLI

1. **Установите Vercel CLI**:
   ```bash
   npm i -g vercel
   ```

2. **Войдите в аккаунт**:
   ```bash
   vercel login
   ```

3. **Деплой**:
   ```bash
   vercel --prod
   ```

## ⚙️ Конфигурация

### vercel.json объяснение

```json
{
  "version": 2,
  "builds": [
    {
      "src": "main.py",           // Главный Python файл
      "use": "@vercel/python"     // Python runtime
    },
    {
      "src": "ui/**",             // Статические файлы UI
      "use": "@vercel/static"
    }
  ],
  "routes": [
    {
      "src": "/static/(.*)",      // Статические файлы
      "dest": "/ui/$1"
    },
    {
      "src": "/ads",              // ADS страница
      "dest": "/ui/ads.html"
    },
    {
      "src": "/",                 // Главная страница
      "dest": "/ui/index.html"
    },
    {
      "src": "/api/(.*)",         // API эндпоинты
      "dest": "/main.py"
    }
  ]
}
```

### Переменные окружения

В Vercel Dashboard → Settings → Environment Variables:

| Переменная | Значение | Описание |
|------------|----------|----------|
| `ADS_API_TOKEN` | `your_token` | NASA ADS API токен |
| `ENVIRONMENT` | `production` | Режим работы |
| `PYTHONPATH` | `.` | Python path |

## 🔧 Локальное тестирование

### Тестирование перед деплоем

1. **Запустите локально**:
   ```bash
   python main.py
   ```

2. **Проверьте эндпоинты**:
   ```bash
   # Health check
   curl http://localhost:8000/api/health
   
   # UI
   curl http://localhost:8000/
   
   # API docs
   curl http://localhost:8000/api/docs
   ```

### Симуляция Vercel окружения

1. **Установите vercel dev**:
   ```bash
   vercel dev
   ```

2. **Тестируйте на localhost:3000**

## 🐛 Troubleshooting

### Частые проблемы

1. **"Module not found" ошибки**:
   - Проверьте `requirements.txt`
   - Убедитесь, что все импорты корректны
   - Проверьте `PYTHONPATH` в `vercel.json`

2. **Статические файлы не загружаются**:
   - Проверьте routes в `vercel.json`
   - Убедитесь, что файлы в папке `ui/`

3. **API не отвечает**:
   - Проверьте логи в Vercel Dashboard
   - Убедитесь, что `main.py` экспортирует `app`

4. **Timeout ошибки**:
   - Увеличьте `maxDuration` в `vercel.json`
   - Оптимизируйте тяжелые операции

### Проверка логов

1. **В Vercel Dashboard**:
   - Functions → View Function Logs

2. **Через CLI**:
   ```bash
   vercel logs [deployment-url]
   ```

## 📊 Мониторинг

### Метрики производительности

1. **Vercel Analytics** - автоматически включен
2. **Function Duration** - следите за таймаутами
3. **Error Rate** - мониторинг ошибок

### Health Checks

Настройте мониторинг эндпоинта:
```
https://your-app.vercel.app/api/health
```

## 🔄 Обновления

### Автоматические деплои

При push в main ветку GitHub:
1. Vercel автоматически создаст новый деплой
2. Проведет тесты
3. Обновит production если все ОК

### Ручные деплои

```bash
# Деплой текущей ветки
vercel

# Деплой в production
vercel --prod

# Деплой конкретного коммита
vercel --prod --force
```

## 🔒 Безопасность

### Рекомендации

1. **Не коммитьте секреты** в код
2. **Используйте Environment Variables** для токенов
3. **Ограничьте CORS** если нужно
4. **Мониторьте использование API**

### Настройка CORS

В `main.py` уже настроен CORS:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Измените для production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## 📈 Оптимизация

### Производительность

1. **Кэширование**:
   - Используйте Vercel Edge Cache
   - Кэшируйте статические данные

2. **Размер функций**:
   - Минимизируйте зависимости
   - Используйте lazy imports

3. **Cold starts**:
   - Оптимизируйте время инициализации
   - Используйте Vercel Edge Functions для критичных эндпоинтов

### Масштабирование

1. **Vercel автоматически масштабирует** функции
2. **Лимиты**:
   - 10 секунд timeout для Hobby плана
   - 30 секунд для Pro плана
3. **Для тяжелых вычислений** рассмотрите внешние сервисы

## 🎯 Production Checklist

- [ ] Все тесты проходят локально
- [ ] `vercel.json` настроен корректно
- [ ] Environment variables настроены
- [ ] Статические файлы доступны
- [ ] API эндпоинты отвечают
- [ ] CORS настроен правильно
- [ ] Логирование работает
- [ ] Health check эндпоинт доступен
- [ ] Документация API доступна
- [ ] Мониторинг настроен 