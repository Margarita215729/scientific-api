# Настройка GitHub Secrets для Docker Hub

## Проблема
GitHub Actions не может авторизоваться в Docker Hub из-за неправильных учетных данных.

## Решение

### 1. Создание Access Token в Docker Hub

1. Войдите в [Docker Hub](https://hub.docker.com/)
2. Перейдите в **Account Settings** → **Security**
3. Нажмите **New Access Token**
4. Введите название токена (например, "GitHub Actions")
5. Выберите **Read & Write** права
6. Скопируйте созданный токен

### 2. Настройка GitHub Secrets

У вас есть два варианта:

#### Вариант 1: Использование аккаунта gretk (основной)

1. Перейдите в ваш GitHub репозиторий
2. Нажмите **Settings** → **Secrets and variables** → **Actions**
3. Нажмите **New repository secret**
4. Добавьте следующие секреты:

##### DOCKER_USERNAME
- **Name**: `DOCKER_USERNAME`
- **Value**: `gretk`

##### DOCKER_PASSWORD
- **Name**: `DOCKER_PASSWORD`
- **Value**: [access token для аккаунта gretk]

#### Вариант 2: Использование аккаунта cutypie (альтернативный)

1. Перейдите в ваш GitHub репозиторий
2. Нажмите **Settings** → **Secrets and variables** → **Actions**
3. Нажмите **New repository secret**
4. Добавьте следующие секреты:

##### DOCKER_USERNAME_CUTYPIE
- **Name**: `DOCKER_USERNAME_CUTYPIE`
- **Value**: `cutypie`

##### DOCKER_PASSWORD_CUTYPIE
- **Name**: `DOCKER_PASSWORD_CUTYPIE`
- **Value**: [ваш Docker Hub access token для аккаунта cutypie]

**Примечание**: Если используете Вариант 2, нужно будет обновить `deploy_production_final.sh` для использования образа `cutypie/scientific-api-app-image`

### 3. Проверка настройки

После добавления секретов:
1. Сделайте push в main ветку
2. GitHub Actions автоматически запустит сборку Docker образа
3. Проверьте логи в Actions tab

### 4. Альтернативное решение

Если у вас нет доступа к аккаунту `gretk`, можно:

1. Создать новый репозиторий в Docker Hub под вашим аккаунтом
2. Обновить `IMAGE_NAME` в `.github/workflows/docker-build.yml`
3. Обновить `deploy_production_final.sh` с новым именем образа

### 5. Команды для проверки

```bash
# Проверить текущий Docker login
docker login

# Проверить доступ к репозиторию
docker pull gretk/scientific-api-app-image:latest

# Тест пуша (если у вас есть права)
docker tag local-image gretk/scientific-api-app-image:test
docker push gretk/scientific-api-app-image:test
```
