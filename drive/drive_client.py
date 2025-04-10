import os
import requests
from .refresh_token_flow import refresh_access_token
UNIVERSE_FOLDER_NAME = "Universe"

def get_headers():
    token = os.getenv("GOOGLE_DRIVE_TOKEN")
    if not token:
        token = refresh_access_token()
    return {"Authorization": f"Bearer {token}"}

def get_universe_folder_id():
    headers = get_headers()
    query = f"name='{UNIVERSE_FOLDER_NAME}' and mimeType='application/vnd.google-apps.folder'"
    url = "https://www.googleapis.com/drive/v3/files"
    params = {
        "q": query,
        "fields": "files(id, name)"
    }
    response = requests.get(url, headers=headers, params=params)
    files = response.json().get("files", [])
    if not files:
        raise Exception(f"Папка '{UNIVERSE_FOLDER_NAME}' не найдена на Google Drive.")
    return files[0]["id"]

def list_files_in_universe(limit=20):
    folder_id = get_universe_folder_id()
    return list_files_in_folder(folder_id=folder_id, limit=limit)


def upload_to_universe(filename, mime_type, content):
    headers = get_headers()
    folder_id = get_universe_folder_id()

    metadata = {
        "name": filename,
        "parents": [folder_id]
    }

    files = {
        "metadata": ('metadata', str(metadata), 'application/json'),
        "file": (filename, content, mime_type)
    }

    url = "https://www.googleapis.com/upload/drive/v3/files?uploadType=multipart"
    response = requests.post(url, headers=headers, files=files)
    return response.json()


def list_folders(limit=20, q=None):
    try:
        headers = get_headers()
        query = "mimeType='application/vnd.google-apps.folder'"
        if q:
            query += f" and name contains '{q}'"

        params = {
            "q": query,
            "pageSize": limit,
            "fields": "files(id, name, mimeType, modifiedTime)"
        }

        url = "https://www.googleapis.com/drive/v3/files"
        response = requests.get(url, headers=headers, params=params)
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def list_files(limit=10, q=None):
    try:
        headers = get_headers()
        params = {
            "pageSize": limit,
            "fields": "files(id, name, mimeType, modifiedTime)"
        }
        if q:
            params["q"] = f"name contains '{q}'"
        url = "https://www.googleapis.com/drive/v3/files"
        response = requests.get(url, headers=headers, params=params)
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def get_file_content(file_id):
    try:
        headers = get_headers()
        url = f"https://www.googleapis.com/drive/v3/files/{file_id}?alt=media"
        response = requests.get(url, headers=headers)
        return {"content": response.text}
    except Exception as e:
        return {"error": str(e)}

def upload_to_drive(filename, mime_type, content):
    try:
        headers = get_headers()
        metadata = {
            "name": filename
        }
        files = {
            "metadata": ('metadata', str(metadata), 'application/json'),
            "file": (filename, content, mime_type)
        }
        url = "https://www.googleapis.com/upload/drive/v3/files?uploadType=multipart"
        response = requests.post(url, headers=headers, files=files)
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def auth_status():
    try:
        headers = get_headers()
        url = "https://www.googleapis.com/drive/v3/about?fields=user"
        response = requests.get(url, headers=headers)
        return {
            "connected": response.status_code == 200,
            "user": response.json().get("user", {}),
            "status_code": response.status_code
        }
    except Exception as e:
        return {"connected": False, "error": str(e)}
