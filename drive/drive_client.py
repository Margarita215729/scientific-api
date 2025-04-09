import requests
import os
import json
from .refresh_token_flow import refresh_access_token

# Google Drive API base
GOOGLE_API_BASE = "https://www.googleapis.com/drive/v3"
GOOGLE_UPLOAD_BASE = "https://www.googleapis.com/upload/drive/v3"

def get_headers():
    token = os.getenv("GOOGLE_DRIVE_TOKEN")

    if not token:
        token = refresh_access_token()

    return {
        "Authorization": f"Bearer {token}"
    }

def upload_to_drive(filename, mime_type, content):
    """
    Загрузка файла в Google Drive через multipart upload
    """
    headers = get_headers()

    metadata = {
        "name": filename,
        "mimeType": mime_type
    }

    files = {
        'metadata': ('metadata.json', json.dumps(metadata), 'application/json'),
        'file': (filename, content, mime_type)
    }

    url = f"{GOOGLE_UPLOAD_BASE}/files?uploadType=multipart"
    response = requests.post(url, headers=headers, files=files)

    try:
        return response.json()
    except Exception:
        return {
            "error": "Upload failed",
            "status_code": response.status_code,
            "raw": response.text
        }

def list_files(limit=5):
    """
    Получить список файлов из Google Drive
    """
    headers = get_headers()
    params = {
        "pageSize": limit,
        "fields": "files(id, name, mimeType, modifiedTime)"
    }

    url = f"{GOOGLE_API_BASE}/files"
    response = requests.get(url, headers=headers, params=params)
    return response.json()

def download_file(file_id):
    """
    Скачать содержимое файла по ID
    """
    headers = get_headers()
    download_url = f"{GOOGLE_API_BASE}/files/{file_id}?alt=media"
    response = requests.get(download_url, headers=headers)

    if response.status_code == 200:
        return {"content": response.text}
    else:
        return {
            "error": "Failed to fetch file",
            "status_code": response.status_code,
            "details": response.text
        }
