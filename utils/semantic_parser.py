import requests

def fetch_semantic_fulltext(keywords, limit):
    """
    Ищет статьи по ключевым словам в Semantic Scholar и возвращает первую найденную
    """
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": keywords,
        "limit": limit,
        "fields": "title,abstract,sections"
    }

    response = requests.get(url, params=params)
    if response.status_code != 200:
        return {"error": f"Semantic Scholar error: {response.status_code}"}

    papers = response.json().get("data", [])

    if not papers:
        return {"error": "No papers found"}

    paper = papers[0]

    return {
        "title": paper.get("title", ""),
        "abstract": paper.get("abstract", ""),
        "sections": paper.get("sections", [])  # может быть пустым
    }
