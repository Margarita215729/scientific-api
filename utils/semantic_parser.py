import requests

def fetch_semantic_fulltext(keywords, limit):
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": keywords,
        "limit": limit,
        "fields": "title,abstract,sections"
    }
    res = requests.get(url, params=params)
    papers = res.json().get("data", [])

    if not papers:
        return {"error": "No papers found"}

    paper = papers[0]
    return {
        "title": paper.get("title", ""),
        "abstract": paper.get("abstract", ""),
        "sections": paper.get("sections", [])
    }
