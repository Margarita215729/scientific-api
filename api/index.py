from fastapi import FastAPI, Query, Body
from fastapi.responses import JSONResponse
from drive.drive_client import (
    get_universe_tree,
    load_universe_tree_snapshot,
    generate_universe_snapshot,
    get_file_content,
    upload_to_drive,
    update_file
)
from utils.arxiv_parser import fetch_arxiv_fulltext
from utils.semantic_parser import fetch_semantic_fulltext
from utils.openml_data import fetch_openml_data
from utils.graph_visualizer import generate_graph_image
import threading

app = FastAPI()

# === UNIVERSE TREE ENDPOINTS ===

@app.get("/api/drive/universe/tree")
def universe_drive_tree(max_depth: int = 5):
    return get_universe_tree(max_depth=max_depth)

@app.get("/api/drive/universe/tree/static")
def universe_drive_tree_static():
    return load_universe_tree_snapshot()

# === AUTO SNAPSHOT SCHEDULER ===

@app.on_event("startup")
def schedule_snapshot_loop():
    def update():
        generate_universe_snapshot()
        threading.Timer(40, update).start()
    update()

# === FILE MANAGEMENT ===

@app.get("/api/drive/file")
def download_drive_file(file_id: str = Query(...)):
    return get_file_content(file_id)

@app.post("/api/drive/upload")
def upload_drive_file(payload: dict = Body(...)):
    return upload_to_drive(
        filename=payload["filename"],
        mime_type=payload.get("mime_type", "text/plain"),
        content=payload["content"],
        parent_id=payload.get("parent_id")
    )

@app.post("/api/drive/file/edit")
def edit_drive_file(payload: dict = Body(...)):
    return update_file(
        file_id=payload["file_id"],
        new_content=payload["new_content"],
        mime_type=payload.get("mime_type", "text/plain")
    )

# === OPENML ===

@app.get("/api/openml/data")
def get_openml_data(tag: str = Query(None), did: str = Query(None), format: str = Query("json")):
    return fetch_openml_data(tag=tag, did=did, format=format)

# === ARXIV ===

@app.get("/api/arxiv/fulltext")
def get_arxiv_fulltext(keywords: str = Query(...), max_results: int = 1):
    return fetch_arxiv_fulltext(keywords, max_results)

# === SEMANTIC SCHOLAR ===

@app.get("/api/semantic/fulltext")
def get_semantic_fulltext(keywords: str = Query(...), limit: int = 1):
    return fetch_semantic_fulltext(keywords, limit)

# === GRAPH VISUALIZER ===

@app.post("/api/graph/visualize")
def graph_visualize(payload: dict = Body(...)):
    return generate_graph_image(payload)

# === HEALTH CHECK ===

@app.get("/api")
def root():
    return {"message": "Scientific Assistant API is live"}
