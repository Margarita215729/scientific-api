import requests
import os

def upload_to_drive(filename, mime_type, content):
    token = os.getenv("GOOGLE_DRIVE_TOKEN")
    headers = {
        "Authorization": f"Bearer {token}"
    }
    metadata = {
        "name": filename,
        "mimeType": mime_type
    }
    files = {
        "data": ("metadata", str(metadata), "application/json"),
        "file": (filename, content, mime_type)
    }
    upload_url = "https://www.googleapis.com/upload/drive/v3/files?uploadType=multipart"
    response = requests.post(upload_url, headers=headers, files=files)
    return response.json()
