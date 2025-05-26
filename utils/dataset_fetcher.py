import os
import json
import requests
import feedparser
import pandas as pd
from io import BytesIO

# Для работы с PDF
try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

# Для работы с DOCX
try:
    from docx import Document
except ImportError:
    Document = None


# =========================
# Функции для получения данных из ArXiv
# =========================
def fetch_from_arxiv(query: str, max_results: int = 10) -> dict:
    """
    Выполняет поиск публикаций на ArXiv по заданному запросу с использованием ArXiv API.
    
    :param query: Строка запроса.
    :param max_results: Максимальное количество результатов.
    :return: Словарь с ключами "source" (ArXiv) и "results" (список публикаций с заголовками, аннотациями, PDF-ссылками).
    """
    base_url = "http://export.arxiv.org/api/query"
    params = {
        "search_query": query,
        "start": 0,
        "max_results": max_results
    }
    response = requests.get(base_url, params=params)
    if response.status_code != 200:
        raise Exception(f"Ошибка запроса ArXiv: {response.status_code}")
    
    feed = feedparser.parse(response.text)
    results = []
    for entry in feed.entries:
        result = {
            "title": entry.get("title", "").strip(),
            "summary": entry.get("summary", "").strip(),
            "pdf_url": next((link.href for link in entry.links if link.rel == "alternate" and link.type == "application/pdf"), None),
            "published": entry.get("published", "")
        }
        results.append(result)
    return {"source": "arXiv", "results": results}


# =========================
# Функции для получения данных из GitHub
# =========================
def fetch_from_github(query: str, max_results: int = 10, github_token: str = None) -> dict:
    """
    Выполняет поиск репозиториев на GitHub по заданному запросу.
    
    :param query: Строка запроса.
    :param max_results: Максимальное количество результатов.
    :param github_token: Персональный GitHub токен (опционально).
    :return: Словарь с ключами "source" (GitHub) и "results" (список репозиториев).
    """
    base_url = "https://api.github.com/search/repositories"
    headers = {}
    if github_token:
        headers["Authorization"] = f"token {github_token}"
    params = {
        "q": query,
        "per_page": max_results
    }
    response = requests.get(base_url, headers=headers, params=params)
    if response.status_code != 200:
        raise Exception(f"Ошибка запроса GitHub: {response.status_code} {response.text}")
    
    data = response.json()
    results = []
    for item in data.get("items", []):
        result = {
            "name": item.get("full_name"),
            "description": item.get("description"),
            "html_url": item.get("html_url"),
            "language": item.get("language")
        }
        results.append(result)
    return {"source": "GitHub", "results": results}


# =========================
# Функции для получения данных из OpenML
# =========================
def fetch_from_openml(query: str = "", max_results: int = 10) -> dict:
    """
    Выполняет поиск датасетов на OpenML по заданному запросу.
    
    :param query: Строка запроса.
    :param max_results: Максимальное количество результатов.
    :return: Словарь с данными датасетов.
    """
    base_url = "https://www.openml.org/api/v1/json/data/list"
    params = {"limit": max_results}
    if query:
        params["name"] = query
    response = requests.get(base_url, params=params)
    if response.status_code != 200:
        raise Exception(f"Ошибка запроса OpenML: {response.status_code}")
    data = response.json()
    results = data.get("data", {}).get("dataset", [])
    formatted = []
    for dataset in results:
        formatted.append({
            "id": dataset.get("did"),
            "name": dataset.get("name"),
            "description": dataset.get("description", ""),
            "url": f"https://www.openml.org/d/{dataset.get('did')}"
        })
    return {"source": "OpenML", "results": formatted}



# =========================
# Функции для получения данных из PLOS Search API
# =========================
def fetch_from_plos(query: str, max_results: int = 10) -> dict:
    """
    Выполняет поиск публикаций на PLOS (Public Library of Science) через их API.
    
    :param query: Запрос для поиска публикаций.
    :param max_results: Максимальное количество результатов.
    :return: Словарь с данными публикаций.
    """
    base_url = "http://api.plos.org/search"
    params = {
        "q": query,
        "rows": max_results,
        "wt": "json"
    }
    response = requests.get(base_url, params=params)
    if response.status_code != 200:
        raise Exception(f"Ошибка запроса PLOS: {response.status_code}")
    
    data = response.json()
    docs = data.get("response", {}).get("docs", [])
    results = []
    for doc in docs:
        results.append({
            "title": doc.get("title_display"),
            "abstract": doc.get("abstract"),
            "publication_date": doc.get("publication_date"),
            "journal": doc.get("journal"),
            "url": doc.get("id")
        })
    return {"source": "PLOS", "results": results}


# =========================
# Функции для получения данных из BioRxiv через RSS
# =========================
def fetch_from_biorxiv(query: str, max_results: int = 10) -> dict:
    """
    Выполняет поиск публикаций на BioRxiv используя RSS-каналы.
    
    :param query: Запрос (слово или фраза, по которой фильтровать результаты).
    :param max_results: Максимальное количество результатов.
    :return: Словарь с данными публикаций из BioRxiv.
    """
    # Используем общий RSS-канал для BioRxiv (можно расширить: фильтровать по категориям)
    rss_url = "http://connect.biorxiv.org/biorxiv_xml.php?subject=all"
    feed = feedparser.parse(rss_url)
    results = []
    count = 0
    for entry in feed.entries:
        # Фильтрация по query, если оно содержится в заголовке или описании
        if query.lower() in entry.get("title", "").lower() or query.lower() in entry.get("summary", "").lower():
            results.append({
                "title": entry.get("title", "").strip(),
                "summary": entry.get("summary", "").strip(),
                "link": entry.get("link"),
                "published": entry.get("published", "")
            })
            count += 1
            if count >= max_results:
                break
    return {"source": "BioRxiv", "results": results}
 

# =========================
# Универсальная функция для анализа содержимого файлов
# =========================
def read_dataset(file_content: bytes, file_extension: str) -> dict:
    """
    Анализирует содержимое файла (переданного в виде байтов) в зависимости от его расширения.
    Поддерживаемые форматы: .pdf, .txt, .docx, .csv, .json
    
    :param file_content: Содержимое файла в байтах.
    :param file_extension: Расширение файла (например, .pdf, .txt, .docx, .csv, .json).
    :return: Словарь с извлечённой информацией из файла.
    """
    ext = file_extension.lower()
    if ext == ".txt":
        return {"text": file_content.decode("utf-8", errors="ignore")}
    elif ext == ".json":
        try:
            return {"json": json.loads(file_content.decode("utf-8"))}
        except Exception as e:
            raise Exception(f"Ошибка парсинга JSON: {str(e)}")
    elif ext == ".csv":
        try:
            from io import StringIO
            df = pd.read_csv(StringIO(file_content.decode("utf-8", errors="ignore")))
            return {"csv": df.to_dict(orient="list"), "description": df.describe(include="all").to_dict()}
        except Exception as e:
            raise Exception(f"Ошибка парсинга CSV: {str(e)}")
    elif ext == ".docx":
        if Document is None:
            raise ImportError("Для работы с DOCX установите пакет python-docx")
        temp_path = "temp.docx"
        with open(temp_path, "wb") as f:
            f.write(file_content)
        doc = Document(temp_path)
        os.remove(temp_path)
        text = "\n".join([para.text for para in doc.paragraphs])
        return {"docx": text}
    elif ext == ".pdf":
        if PyPDF2 is None:
            raise ImportError("Для работы с PDF установите пакет PyPDF2")
        reader = PyPDF2.PdfReader(BytesIO(file_content))
        text = ""
        for page in reader.pages:
            text += page.extract_text() if page.extract_text() else ""
        return {"pdf": text}
    else:
        raise Exception(f"Формат файла {ext} не поддерживается.")
