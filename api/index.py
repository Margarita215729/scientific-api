# api/index.py
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello from Scientific API"}

@app.get("/ping")
async def ping():
    return {"status": "ok", "message": "API is up and running"}
