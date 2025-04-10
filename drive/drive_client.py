import os
import json
import requests
from .refresh_token_flow import refresh_access_token

UNIVERSE_FOLDER_ID = "1-07YePaYLOiGqyIq0MszRLnCjd8aRkSt"

def get_headers():
    token = os.getenv("GOOGLE_DRIVE_TOKEN")
    if not token:
        token = refresh_access_token()
    return {"Authorization": f"Bearer {token}"}

def build_drive_tree(folder_id, depth=0, max_depth=10):
    if depth > max_depth:
        return [{"name": "MAX_DEPTH_REACHED", "type": "notice"}]

    headers = get_headers()
    url = "https://www.googleapis.com/drive/v3/files"
    params = {
        "q": f"'{folder_id}' in parents",
        "fields": "files(id, name, mimeType)",
        "pageSize": 1000
    }

    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        items = response.json().get("files", [])
    except Exception as e:
        return [{"name": f"ERROR: {str(e)}", "type": "error"}]

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

def get_universe_tree(max_depth=10):
    return {
        "id": UNIVERSE_FOLDER_ID,
        "name": "Universe",
        "type": "folder",
        "children": build_drive_tree(UNIVERSE_FOLDER_ID, depth=0, max_depth=max_depth)
    }

def generate_universe_snapshot():
    print("[📦] Генерация snapshot дерева Universe...")
    tree = get_universe_tree(max_depth=10)
    os.makedirs("data", exist_ok=True)
    with open("data/universe_tree_snapshot.json", "w") as f:
        json.dump(tree, f, indent=2)
    print("[✓] Snapshot сохранён: data/universe_tree_snapshot.json")

def load_universe_tree_snapshot():
    try:
        with open("data/universe_tree_snapshot.json", "r") as f:
            return json.load(f)
    except Exception as e:
        return {"error": f"Snapshot not found or invalid: {str(e)}"}

def get_file_content(file_id):
    try:
        headers = get_headers()
        url = f"https://www.googleapis.com/drive/v3/files/{file_id}?alt=media"
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return {"content": response.text}
    except Exception as e:
        return {"error": str(e)}

def upload_to_drive(filename, mime_type, content, parent_id=None):
    try:
        headers = get_headers()
        metadata = {"name": filename}
        if parent_id:
            metadata["parents"] = [parent_id]

        files = {
            "metadata": ('metadata', json.dumps(metadata), 'application/json'),
            "file": (filename, content, mime_type)
        }

        url = "https://www.googleapis.com/upload/drive/v3/files?uploadType=multipart"
        response = requests.post(url, headers=headers, files=files)
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def update_file(file_id, new_content, mime_type="text/plain"):
    try:
        headers = get_headers()
        url = f"https://www.googleapis.com/upload/drive/v3/files/{file_id}?uploadType=media"
        response = requests.patch(url, data=new_content.encode(), headers={
            "Authorization": headers["Authorization"],
            "Content-Type": mime_type
        })
        response.raise_for_status()
        return {"status": "updated", "file_id": file_id}
    except Exception as e:
        return {"error": f"Failed to update file: {str(e)}"}
