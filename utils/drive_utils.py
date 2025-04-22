# utils/drive_utils.py
from io import BytesIO
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from googleapiclient.errors import HttpError
from config import GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, GOOGLE_REFRESH_TOKEN

def get_drive_service():
    """
    Создает и возвращает сервис Google Drive API.
    Выбрасывает исключение при проблемах с аутентификацией.
    """
    try:
        credentials = Credentials(
            token=None,
            refresh_token=GOOGLE_REFRESH_TOKEN,
            token_uri="https://oauth2.googleapis.com/token",
            client_id=GOOGLE_CLIENT_ID,
            client_secret=GOOGLE_CLIENT_SECRET,
        )
        service = build("drive", "v3", credentials=credentials)
        return service
    except Exception as e:
        raise Exception(f"Ошибка при создании сервиса Google Drive: {str(e)}")

def download_file_from_drive(file_id: str) -> BytesIO:
    """
    Скачивает файл из Google Drive по его ID.
    Возвращает объект BytesIO с содержимым файла.
    Выбрасывает исключение при ошибках доступа или скачивания.
    """
    try:
        service = get_drive_service()
        request = service.files().get_media(fileId=file_id)
        file_stream = BytesIO()
        downloader = MediaIoBaseDownload(file_stream, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
        file_stream.seek(0)
        return file_stream
    except HttpError as e:
        raise Exception(f"Ошибка при скачивании файла с Google Drive: {str(e)}")
    except Exception as e:
        raise Exception(f"Неизвестная ошибка при скачивании файла: {str(e)}")
