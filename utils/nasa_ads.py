import requests

def fetch_nasa_abstracts(query):
    headers = {"Authorization": "Bearer zahkzrUHgH9h9LuUgIfYUYVkJ0f86fHLdOzDYjko"}
    url = "https://api.adsabs.harvard.edu/v1/search/query"
    params = {
        "q": query,
        "fl": "title,abstract,author",
        "rows": 5
    }
    res = requests.get(url, headers=headers, params=params)
    return res.json()
