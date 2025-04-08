from fastapi import FastAPI, Query, Body
from fastapi.responses import JSONResponse
from drive.drive_client import upload_to_drive, list_files, download_file
from utils.arxiv_parser import fetch_arxiv_fulltext
from utils.semantic_parser import fetch_semantic_fulltext
from utils.openml_data import fetch_openml_data
from utils.graph_visualizer import generate_graph_image

app = FastAPI()

@app.post("/api/drive/upload")
def upload_file(payload: dict = Body(...)):
    return upload_to_drive(
        filename=payload.get("filename"),
        mime_type=payload.get("mime_type"),
        content=payload.get("content")
    )

@app.get("/api/drive/list")
def list_drive_files(limit: int = 5):
    return list_files(limit)

@app.get("/api/drive/file")
def get_drive_file(file_id: str = Query(...)):
    return download_file(file_id)

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
