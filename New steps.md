Что нужно изменить
A. Нормализация репозитория (обязательно, шаг 1)
Добавить в .gitignore минимум:
.venv/
__pycache__/, .pytest_cache/, .ruff_cache/, .mypy_cache/
.ipynb_checkpoints/
.DS_Store
Удалить из Git-индекса всё, что не должно быть в репо (но оставить локально):
.venv/
LogFiles/
app-logs.zip
любые сгенерированные outputs/reports/data (если случайно попали)
Команды (внутри devcontainer):
git rm -r --cached .venv LogFiles app-logs.zip || true
git rm -r --cached data outputs reports || true
git add .gitignore
git commit -m "chore: remove local artifacts (.venv/logs) and harden .gitignore"
​
Acceptance: git status чистый, .venv физически остаётся локально, но Git её больше не трекает.
B. Убрать Git LFS-блокер навсегда (обязательно, шаг 2)
В .devcontainer/devcontainer.json добавить postCreateCommand, который гарантирует наличие LFS после каждого rebuild:
sudo apt-get update && sudo apt-get install -y git-lfs && git lfs install
​
Если postCreateCommand уже есть — расширить, а не дублировать.
Acceptance: после rebuild git lfs version работает, git push не падает на “git-lfs not found”.
C. Принять единый “истинный” стек (обязательно, шаг 3)
scientific_api/ = ядро исследования (ingest → graph build → features → модели → метрики).
app/ = FastAPI-обёртка для запуска экспериментов/сервиса.
api/ = превратить в thin wrapper, но не держать как третий самостоятельный слой.
ml/ = перенести внутрь scientific_api/ml/,* (без дубликатов функций/форматов).
Acceptance: есть один канонический путь данных и один канонический код-путь построения графов/признаков.
D. Восстановить корректное пакетирование (обязательно, шаг 4)
Добавить pyproject.toml и сделать так, чтобы:
pip install -e . работал в devcontainer
import scientific_api работал одинаково из app/, notebooks/, tests/
Acceptance: в чистом контейнере после rebuild команда python -c "import scientific_api; print(scientific_api.__file__)" проходит.