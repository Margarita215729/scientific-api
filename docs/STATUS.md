## Diagnostics — 2026-01-07

### Commands and outputs

**python --version**

```
Python 3.12.12
```

**git rev-parse HEAD**

```
97805313597807d5ae9773667efedae3ebd5709e
```

**ls -la**

```
total 1488
drwxrwxrwx+  25 vscode root     4096 Jan  6 22:46  .
drwxr-xrwx+   5 vscode root     4096 Nov 13 00:31  ..
drwxrwxrwx+   2 vscode vscode   4096 Nov 13 01:09  .devcontainer
-rw-rw-rw-    1 vscode vscode   3293 Dec 19 01:50  .env.example
drwxrwxrwx+   8 vscode root     4096 Jan  6 23:47  .git
drwxrwxrwx+   4 vscode root     4096 Nov 13 00:30  .github
-rw-rw-rw-    1 vscode root     2788 Jan  6 22:46  .gitignore
drwxr-xr-x+   3 vscode vscode   4096 Jan  3 03:44  .pytest_cache
drwxrwxrwx+   5 vscode vscode   4096 Jan  3 03:44  .venv
-rw-rw-rw-    1 vscode root      600 Nov 13 00:30  .vercelignore
drwxrwxrwx+   2 vscode vscode   4096 Dec 19 01:24  .vscode
-rw-rw-rw-    1 vscode root     4152 Nov 13 00:30  CONTAINER_DEPLOYMENT_STATUS.md
-rw-rw-rw-    1 vscode root     3284 Nov 13 00:30  DEPLOYMENT_SUCCESS.md
-rw-rw-rw-    1 vscode vscode  51428 Jan  6 22:52  DEV_LOG.md
-rw-rw-rw-    1 vscode root     1679 Dec 19 01:50  Dockerfile
-rw-rw-rw-    1 vscode root     5021 Nov 13 00:30  ENVIRONMENT_STATUS_REPORT.md
-rw-rw-rw-    1 vscode root    11278 Nov 13 00:30  FEATURES.md
-rw-rw-rw-    1 vscode root     3582 Nov 13 00:30  GITHUB_SECRETS_SETUP.md
-rw-rw-rw-    1 vscode vscode   5391 Jan  3 03:38  INTEGRATION_GUIDE.md
drwxrwxrwx+   3 vscode root     4096 Nov 13 00:30  LogFiles
-rw-rw-rw-    1 vscode vscode  12912 Dec 30 01:04  PIPELINE_STRUCTURE.md
-rw-rw-rw-    1 vscode root     2492 Nov 13 00:30  PRODUCTION_ANALYSIS.md
-rw-rw-rw-    1 vscode root     8036 Nov 13 00:30  PRODUCTION_READINESS_FINAL.md
-rw-rw-rw-    1 vscode root     8987 Nov 13 00:30  PROJECT_COMPLETION_REPORT.md
-rw-rw-rw-    1 vscode root    11166 Nov 13 00:30  PROJECT_STATUS.md
-rw-rw-rw-    1 vscode root     8008 Nov 13 00:30  README.md
-rw-rw-rw-    1 vscode root     8672 Nov 13 00:30  README_ENGLISH.md
-rw-rw-rw-    1 vscode root     9500 Nov 13 00:30  ROADMAP.md
-rw-rw-rw-    1 vscode vscode  22000 Jan  3 21:25  TECHNICAL_PLAN.md
drwxrwxrwx+   2 vscode vscode   4096 Jan  3 03:44  __pycache__
drwxrwxrwx+   3 vscode root     4096 Jan  3 03:44  api
drwxrwxrwx+   6 vscode vscode   4096 Dec 19 01:47  app
-rw-rw-rw-    1 vscode root     2974 Nov 13 00:30  app-logs.zip
-rw-rw-rw-    1 vscode root     4022 Nov 13 00:30  azure-deployment-with-db.json
-rw-rw-rw-    1 vscode root     5962 Nov 13 00:30  azure-webapp-bicep.bicep
-rwxrwxrwx    1 vscode root     2706 Nov 13 00:30  build_local_docker.sh
-rwxrwxrwx    1 vscode root     8629 Nov 13 00:30  build_project.sh
-rw-rw-rw-    1 vscode root      598 Nov 13 00:30  chapter1.docx
-rw-rw-rw-    1 vscode root     9007 Nov 13 00:30  chapter1.md
-rw-rw-rw-    1 vscode root     1213 Nov 13 00:30  chapter2.docx
-rw-rw-rw-    1 vscode root    16366 Nov 13 00:30  chapter2.md
-rw-rw-rw-    1 vscode root     4768 Nov 13 00:30  check_env.py
drwxrwxrwx+   3 vscode vscode   4096 Jan  6 22:45  configs
drwxrwxrwx+   5 vscode vscode   4096 Dec 19 01:47  data
drwxrwxrwx+   2 vscode root     4096 Nov 13 00:30  database
-rwxrwxrwx    1 vscode root     3666 Nov 13 00:30  deploy_azure.sh
-rwxrwxrwx    1 vscode root     9259 Nov 13 00:30  deploy_azure_bicep.sh
-rwxrwxrwx    1 vscode root     8893 Nov 13 00:30  deploy_final_production.sh
-rwxrwxrwx    1 vscode root     6862 Nov 13 00:30  deploy_production.sh
-rwxrwxrwx    1 vscode root     6768 Nov 13 00:30  deploy_production_final.sh
-rwxrwxrwx    1 vscode root     5338 Nov 13 00:30  deploy_with_database.sh
-rw-rw-rw-    1 vscode root     3187 Dec 19 01:50  docker-compose.yml
-rw-rw-rw-    1 vscode root     1086 Nov 13 00:30  init_database.py
-rw-rw-rw-    1 vscode root      677 Nov 13 00:30  load_env.py
-rw-rw-rw-    1 vscode root     7068 Nov 13 00:30  main.py
-rw-rw-rw-    1 vscode root    11670 Nov 13 00:30  main_azure_with_db.py
-rwxrwxrwx    1 vscode root     8454 Nov 13 00:30  manage_data_pipeline.sh
drwxrwxrwx+   8 vscode vscode   4096 Dec 19 01:48  ml
drwxrwxrwx+ 552 vscode vscode  20480 Nov 13 00:32  node_modules
drwxrwxrwx+   3 vscode vscode   4096 Jan  3 21:08  notebooks
drwxrwxrwx+   5 vscode vscode   4096 Jan  6 22:45  outputs
-rw-rw-rw-    1 vscode vscode 784870 Nov 13 00:32  package-lock.json
-rw-rw-rw-    1 vscode root      319 Nov 13 00:30  package.json
-rw-rw-rw-    1 vscode root      449 Nov 13 00:30  pytest.ini
drwxrwxrwx+   4 vscode vscode   4096 Jan  3 21:04  reports
-rw-rw-rw-    1 vscode root      310 Nov 13 00:30  requirements-vercel.txt
-rw-rw-rw-    1 vscode root     1097 Jan  6 22:52  requirements.txt
-rw-rw-rw-    1 vscode root      290 Nov 13 00:30  requirements_azure.txt
-rw-rw-rw-    1 vscode root      122 Nov 13 00:30  requirements_test.txt
-rw-rw-rw-    1 vscode root     8695 Nov 13 00:30  run_tests.py
-rwxrwxrwx    1 vscode root     1062 Nov 13 00:30  run_tests.sh
-rw-rw-rw-    1 vscode vscode   3047 Dec 19 01:29  scientific-api.http
drwxrwxrwx+   5 vscode vscode   4096 Jan  6 22:46  scientific_api
-rw-rw-rw-    1 vscode root    98304 Nov 13 00:30  scientific_api.db
drwxrwxrwx+   2 vscode vscode   4096 Dec 19 01:47  scripts
-rwxrwxrwx    1 vscode root    10610 Nov 13 00:30  setup_production.py
-rw-rw-rw-    1 vscode root     1206 Nov 13 00:30  start_dev_server.py
-rw-rw-rw-    1 vscode root      668 Nov 13 00:30  start_server.py
-rw-rw-rw-    1 vscode root    14360 Nov 13 00:30  test_production_apis.py
drwxrwxrwx+   3 vscode root     4096 Jan  3 03:44  tests
drwxrwxrwx+   2 vscode root     4096 Nov 13 00:30  ui
-rwxrwxrwx    1 vscode root     2245 Nov 13 00:30  update_azure_app.sh
-rwxrwxrwx    1 vscode root     3996 Nov 13 00:30  update_azure_container.sh
-rwxrwxrwx    1 vscode root     1382 Nov 13 00:30  update_docker_image.sh
drwxrwxrwx+   2 vscode root     4096 Nov 13 00:30  utils
-rw-rw-rw-    1 vscode vscode  17690 Jan  6 22:42 'Инструкция для AI-модели в VS Code.md'
-rw-rw-rw-    1 vscode root     2756 Nov 13 00:30  ЭТАП_ПРОЕКТА.md
```

**find . -maxdepth 3 -type d**

```
[truncated: command produced very long node_modules listing]
```

**git status -sb**

```
## main...origin/main
```

**cat .devcontainer/devcontainer.json**

```json
// For format details, see https://aka.ms/devcontainer.json. For config options, see the
// README at: https://github.com/devcontainers/templates/tree/main/src/alpine
{
        "name": "Scientific API Debian",
        "image": "mcr.microsoft.com/devcontainers/python:3.11-bookworm",
        "customizations": {
                "vscode": {
                        "extensions": [
                                "openai.chatgpt",
                                "ms-python.python",
                                "ms-python.vscode-pylance",
                                "charliermarsh.ruff"
                        ],
                        "settings": {
                                "python.defaultInterpreterPath": "${containerWorkspaceFolder}/.venv/bin/python",
                                "python.languageServer": "Pylance",
                                "python.analysis.typeCheckingMode": "basic"
                        }
                }
        },
        "postCreateCommand": "apt-get update && apt-get install -y git-lfs build-essential libpq-dev && git lfs install && python -m venv .venv && . .venv/bin/activate && pip install -U pip && pip install -e . && pip install numpy pandas scipy scikit-learn matplotlib pyarrow pyyaml httpx astropy tqdm networkx && pip install -r requirements.txt",
        "forwardPorts": [
                8000
        ],
        "remoteUser": "vscode"
}
```

### Notes
- After modifying devcontainer, user must run: Dev Containers: Rebuild Container.