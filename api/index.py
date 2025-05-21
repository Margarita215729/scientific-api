# api/index.py
import uvicorn
from fastapi import FastAPI
from api import data_analysis, ml_models, file_manager_api, dataset_api, astro_catalog_api, ads_api

app = FastAPI(
    title="Scientific API",
    description="Научное API для анализа данных, обучения моделей и генерации отчетов.",
    version="1.0.0"
)

app.include_router(data_analysis.router, prefix="/data", tags=["Data Analysis"])
app.include_router(ml_models.router, prefix="/ml", tags=["ML Models"])
app.include_router(file_manager_api.router, prefix="/files", tags=["File Manager"])
app.include_router(dataset_api.router, prefix="/datasets", tags=["Datasets"])
app.include_router(astro_catalog_api.router, prefix="/astro", tags=["Astronomy Catalogs"])
app.include_router(ads_api.router, prefix="/ads", tags=["ADS Literature"])

@app.get("/")
async def root():
    return {"message": "Добро пожаловать в Scientific API!"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

# Для совместимости с Vercel
asgi_app = app
