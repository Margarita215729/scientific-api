# main.py
import uvicorn
from fastapi import FastAPI
from api import data_analysis, ml_models

app = FastAPI(
    title="Scientific API",
    description="Научное API для анализа данных, обучения моделей и генерации отчетов.",
    version="1.0.0"
)

# Регистрируем маршруты из модулей
app.include_router(data_analysis.router, prefix="/data", tags=["Data Analysis"])
app.include_router(ml_models.router, prefix="/ml", tags=["ML Models"])

@app.get("/")
async def root():
    return {"message": "Добро пожаловать в Scientific API!"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

from api import file_manager_api

app.include_router(file_manager_api.router, prefix="/files", tags=["File Manager"])
