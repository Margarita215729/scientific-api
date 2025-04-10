# api/file_manager_api.py
from fastapi import APIRouter, HTTPException, UploadFile, File
from utils import file_manager
import os

router = APIRouter()

@router.get("/list")
async def list_files(directory: str = "."):
    """
    Просмотр файлов и вложенных структур в указанной директории.
    По умолчанию используется текущая рабочая директория.
    """
    try:
        tree = file_manager.list_files(directory)
        return tree
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/read")
async def read_file(file_path: str):
    """
    Чтение содержимого файла по заданному пути.
    """
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Файл не найден")
    try:
        content = file_manager.read_file(file_path)
        return {"file_path": file_path, "content": content}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/split")
async def split_file(file_path: str, chunk_size: int = 1000):
    """
    Разбивает файл на части указанного размера.
    """
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Файл не найден")
    try:
        parts = file_manager.split_file(file_path, chunk_size)
        return {"file_path": file_path, "chunks": parts}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/create")
async def create_file(file_path: str, file_type: str = "txt", content: str = ""):
    """
    Создание нового файла с указанным содержимым.
    """
    try:
        file_manager.create_file(file_path, content, file_type)
        return {"message": f"Файл {file_path} успешно создан"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/edit")
async def edit_file(file_path: str, new_content: str):
    """
    Редактирование файла: перезапись содержимого.
    """
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Файл не найден")
    try:
        file_manager.edit_file(file_path, new_content)
        return {"message": f"Файл {file_path} успешно обновлен"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Загружает файл и сохраняет в рабочую директорию.
    """
    try:
        file_location = f"uploads/{file.filename}"
        os.makedirs("uploads", exist_ok=True)
        with open(file_location, "wb") as f:
            f.write(await file.read())
        return {"message": f"Файл успешно загружен по пути: {file_location}"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
