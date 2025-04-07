import requests

def fetch_openml_data(tag, format):
    url = f"https://www.openml.org/api/v1/json/data/list/tag/{tag}"
    res = requests.get(url)
    if res.status_code != 200:
        return {"error": "No datasets found"}
    return res.json()
