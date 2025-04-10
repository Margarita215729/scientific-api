import os
import json
import requests
from .refresh_token_flow import refresh_access_token

UNIVERSE_FOLDER_ID = "1-07YePaYLOiGqyIq0MszRLnCjd8aRkSt"


def get_universe_folder_id():
    return UNIVERSE_FOLDER_ID


def get_headers():
    token = os.getenv("GOOGLE_DRIVE_TOKEN")
    if not token:
        token = refresh_access_token()
    return {"Authorization": f"Bearer {token}"}


def list_files_in_universe(limit=20):
    folder_id = get_universe_folder_id()
    return list_files_in_folder(folder_id=folder_id, limit=limit)


def build_drive_tree(folder_id, depth=0, max_depth=10):
    if depth > max_depth:
        return [{"name": "MAX_DEPTH_REACHED", "type": "notice"}]

    headers = get_headers()
    url = "https://www.googleapis.com/drive/v3/files"
    query = f"'{folder_id}' in parents"
    params = {
        "q": query,
        "fields": "files(id, name, mimeType)",
        "pageSize": 1000
    }

    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
    except Exception as e:
        return [{"name": f"ERROR: {str(e)}", "type": "error"}]

    items = response.json().get("files", [])
    tree = []

    for item in items:
        node = {
            "id": item["id"],
            "name": item["name"],
            "type": "folder" if item["mimeType"] == "application/vnd.google-apps.folder" else "file"
        }
        if node["type"] == "folder":
            node["children"] = build_drive_tree(item["id"], depth=depth + 1, max_depth=max_depth)
        tree.append(node)

    return tree


def get_universe_tree():
    universe_id = get_universe_folder_id()
    return {
        "name": "Universe",
        "id": universe_id,
        "type": "folder",
        "children": build_drive_tree(universe_id)
    }


def upload_to_universe(filename, mime_type, content):
    headers = get_headers()
    folder_id = get_universe_folder_id()

    metadata = {
        "name": filename,
        "parents": [folder_id]
    }

    files = {
        "metadata": ('metadata', json.dumps(metadata), 'application/json'),
        "file": (filename, content, mime_type)
    }

    url = "https://www.googleapis.com/upload/drive/v3/files?uploadType=multipart"
    response = requests.post(url, headers=headers, files=files)
    response.raise_for_status()
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
        response.raise_for_status()
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
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}


def list_files_in_folder(folder_id, limit=20):
    try:
        headers = get_headers()
        query = f"'{folder_id}' in parents"
        params = {
            "q": query,
            "pageSize": limit,
            "fields": "files(id, name, mimeType, modifiedTime)"
        }

        url = "https://www.googleapis.com/drive/v3/files"
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}


def get_file_content(file_id):
    try:
        headers = get_headers()
        url = f"https://www.googleapis.com/drive/v3/files/{file_id}?alt=media"
        response = requests.get(url, headers=headers)
        response.raise_for_status()
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
            "metadata": ('metadata', json.dumps(metadata), 'application/json'),
            "file": (filename, content, mime_type)
        }

        url = "https://www.googleapis.com/upload/drive/v3/files?uploadType=multipart"
        response = requests.post(url, headers=headers, files=files)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}


def auth_status():
    try:
        headers = get_headers()
        url = "https://www.googleapis.com/drive/v3/about?fields=user"
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return {
            "connected": True,
            "user": response.json().get("user", {}),
            "status_code": response.status_code
        }
    except Exception as e:
        return {"connected": False, "error": str(e)}
