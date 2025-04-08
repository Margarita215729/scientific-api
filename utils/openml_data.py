import requests

def fetch_openml_data(tag=None, did=None, format="json"):
    """
    Получить список датасетов по тегу или скачать CSV по dataset ID (did)
    """
    if did:
        # Скачиваем конкретный датасет по ID
        meta_url = f"https://www.openml.org/api/v1/json/data/{did}"
        meta_response = requests.get(meta_url)

        if meta_response.status_code != 200:
            return {"error": f"Failed to get dataset metadata: {meta_response.status_code}"}

        # Получаем URL для скачивания
        dataset = meta_response.json().get("data_set_description", {})
        download_url = dataset.get("url")

        if not download_url:
            return {"error": "No download URL found in dataset metadata"}

        file_response = requests.get(download_url)
        if file_response.status_code != 200:
            return {"error": f"Failed to download dataset: {file_response.status_code}"}

        if format == "csv":
            return {
                "filename": dataset.get("name", f"dataset_{did}.csv"),
                "csv": file_response.text
            }

        return {
            "metadata": dataset,
            "preview": file_response.text[:1000]
        }

    elif tag:
        # Поиск по тегу
        url = f"https://www.openml.org/api/v1/json/data/list/tag/{tag}"
        response = requests.get(url)

        if response.status_code != 200:
            return {"error": f"OpenML error: {response.status_code}"}

        return {
            "data": response.json().get("data", {}).get("dataset", [])
        }

    else:
        return {"error": "Specify either 'tag' or 'did'"}
