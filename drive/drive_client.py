import requests
import os
import json

GOOGLE_API_BASE = "https://www.googleapis.com/drive/v3"
UPLOAD_URL = "https://www.googleapis.com/upload/drive/v3/files?uploadType=multipart"

def get_headers():
    token = os.getenv("GOOGLE_DRIVE_TOKEN")
    return {
        "Authorization": f"Bearer {token}"
    }

def upload_to_drive(filename, mime_type, content):
    metadata = {
        "name": filename,
        "mimeType": mime_type
    }

    files = {
        'metadata': ('metadata', json.dumps(metadata), 'application/json'),
        'file': (filename, content, mime_type)
    }

    response = requests.post(UPLOAD_URL, headers=get_headers(), files=files)
    return response.json()

def list_files(limit=5):
    params = {
        "pageSize": limit,
        "fields": "files(id, name, mimeType, modifiedTime)"
    }
    res = requests.get(f"{GOOGLE_API_BASE}/files", headers=get_headers(), params=params)
    return res.json()

def download_file(file_id):
    download_url = f"https://www.googleapis.com/drive/v3/files/{file_id}?alt=media"
    res = requests.get(download_url, headers=get_headers())
    if res.status_code == 200:
        return {"content": res.text}
    else:
        return {"error": "Unable to fetch file", "status": res.status_code}
