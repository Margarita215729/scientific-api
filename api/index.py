from fastapi import FastAPI, Query, Body
from drive.drive_client import (
    get_universe_tree,
    get_file_content
)
from utils.graph_visualizer import generate_graph_image

app = FastAPI()

@app.post("/api/drive/universe/analyze")
def analyze_universe_folder(payload: dict = Body(default={})):
    max_depth = payload.get("max_depth", 5)
    tree = get_universe_tree(max_depth=max_depth)
    extracted = []

    def collect_texts(nodes):
        for node in nodes:
            if node["type"] == "file" and node["name"].endswith((".txt", ".csv", ".json")):
                content = get_file_content(node["id"]).get("content", "")
                extracted.append({
                    "name": node["name"],
                    "content": content
                })
            elif node["type"] == "folder":
                collect_texts(node.get("children", []))

    collect_texts(tree.get("children", []))
    return {
        "summary": f"{len(extracted)} файлов прочитано",
        "files": extracted
    }

@app.post("/api/graph/visualize")
def graph_visualize(payload: dict = Body(...)):
    return generate_graph_image(payload)

@app.get("/api")
def root():
    return {"message": "Scientific GPT API — ready"}
