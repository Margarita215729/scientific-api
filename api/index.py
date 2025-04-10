from fastapi import FastAPI, Query, Body
from fastapi.responses import JSONResponse
from drive.drive_client import *
from utils.arxiv_parser import fetch_arxiv_fulltext
from utils.semantic_parser import fetch_semantic_fulltext
from utils.openml_data import fetch_openml_data
from utils.graph_visualizer import generate_graph_image
import threading

app = FastAPI()


# 👇 Автообновление snapshot каждые 40 сек
def schedule_snapshot_update(interval=40):
    def update_loop():
        generate_universe_snapshot()
        threading.Timer(interval, update_loop).start()

    update_loop()

# ⏱ Запускаем таймер при старте FastAPI
@app.on_event("startup")
def startup_event():
    print("🟢 FastAPI запущен. Включаем обновление snapshot...")
    schedule_snapshot_update()

@app.get("/api/drive/universe/tree/static")
def universe_drive_tree_static():
    return load_universe_tree_snapshot()

@app.get("/api/drive/universe/tree")
def universe_drive_tree(max_depth: int = 3):
    return get_universe_tree(max_depth=max_depth)

@app.get("/api/drive/universe/files")
def get_files_from_universe(limit: int = 10):
    return list_files_in_universe(limit=limit)

@app.post("/api/drive/universe/upload")
def upload_to_universe_endpoint(payload: dict = Body(...)):
    return upload_to_universe(
        filename=payload["filename"],
        mime_type=payload.get("mime_type", "text/plain"),
        content=payload["content"]
    )

@app.get("/api/drive/folders")
def get_drive_folders(limit: int = 20, search: str = None):
    return list_folders(limit=limit, q=search)

@app.get("/api")
def root():
    return {"message": "Scientific Assistant API is live"}


@app.get("/api/drive/list")
def list_drive_files(limit: int = 10, search: str = None):
    return list_files(limit=limit, q=search)

@app.get("/api/drive/file")
def read_drive_file(file_id: str = Query(...)):
    return get_file_content(file_id)

@app.post("/api/drive/upload")
def upload_drive_file(payload: dict = Body(...)):
    filename = payload["filename"]
    mime_type = payload.get("mime_type", "text/plain")
    content = payload["content"]
    return upload_to_drive(filename, mime_type, content)

@app.get("/api/drive/status")
def drive_auth_status():
    return auth_status()

@app.get("/api/arxiv/fulltext")
def get_arxiv_fulltext(keywords: str = Query(...), max_results: int = 1):
    return fetch_arxiv_fulltext(keywords, max_results)

@app.get("/api/semantic/fulltext")
def get_semantic_fulltext(keywords: str = Query(...), limit: int = 1):
    return fetch_semantic_fulltext(keywords, limit)

@app.get("/api/openml/data")
def get_openml_data(tag: str = Query(...), format: str = Query("csv")):
    return fetch_openml_data(tag, format)

@app.post("/api/graph/visualize")
def graph_visualize(payload: dict = Body(...)):
    return generate_graph_image(payload)
