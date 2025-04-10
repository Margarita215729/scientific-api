import os
import requests

def refresh_access_token():
    """
    Обновляет access_token через refresh_token, используя Playground OAuth client
    """

    refresh_token = os.getenv("GOOGLE_REFRESH_TOKEN")
    client_id = os.getenv("GOOGLE_CLIENT_ID")
    client_secret = os.getenv("GOOGLE_CLIENT_SECRET")

    if not all([refresh_token, client_id, client_secret]):
        raise Exception("Missing environment variables: GOOGLE_REFRESH_TOKEN, GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET")

    url = "https://oauth2.googleapis.com/token"
    payload = {
        "client_id": client_id,
        "client_secret": client_secret,
        "refresh_token": refresh_token,
        "grant_type": "refresh_token"
    }

    response = requests.post(url, data=payload)

    if response.status_code != 200:
        raise Exception(f"Failed to refresh token: {response.status_code}\n{response.text}")

    access_token = response.json().get("access_token")
    if not access_token:
        raise Exception("No access token returned from Google.")

    # Автоматически обновляем переменную окружения
    os.environ["GOOGLE_DRIVE_TOKEN"] = access_token

    print("[✓] Access token refreshed via OAuth Playground")

    return access_token
