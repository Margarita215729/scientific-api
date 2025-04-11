# api/dataset_api.py
"""
Этот модуль служит самостоятельной серверлесс‑функцией для работы с датасетами.
Он использует утилиты из папки utils для получения публикаций и датасетов из разных источников.
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse

# Импорт функций-утилит из соответствующих модулей в utils
from utils.dataset_fetcher import fetch_from_arxiv, fetch_from_openml, fetch_from_biorxiv
from utils.academic_ms_fetcher import fetch_from_academic_ms
from utils.adsabs_fetcher import fetch_from_adsabs
from utils.cern_fetcher import fetch_from_cern
from utils.google_dataset_fetcher import fetch_from_google_dataset_search
from utils.nasa_fetcher import fetch_from_nasa

# Создаём собственное FastAPI-приложение для данной функции
app = FastAPI(
    title="Dataset API",
    description="Серверлесс‑функция для получения данных из открытых источников: ArXiv, OpenAlex, ADSabs, CERN, Google Dataset Search, NASA, OpenML, BioRxiv.",
    version="1.0.0"
)

@app.get("/arxiv", summary="Поиск публикаций в ArXiv")
async def get_arxiv(
    query: str = Query(..., description="Строка запроса для публикаций в ArXiv"),
    max_results: int = Query(10, ge=1, le=50, description="Количество результатов")
):
    try:
        result = fetch_from_arxiv(query, max_results)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/academic", summary="Поиск публикаций через OpenAlex")
async def get_academic(
    query: str = Query(..., description="Строка запроса для публикаций через OpenAlex"),
    max_results: int = Query(10, ge=1, le=50, description="Количество результатов")
):
    try:
        result = fetch_from_academic_ms(query, max_results)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/adsabs", summary="Поиск публикаций в ADSabs (Harvard ADS)")
async def get_adsabs(
    query: str = Query(..., description="Строка запроса для публикаций в ADSabs"),
    max_results: int = Query(10, ge=1, le=50, description="Количество результатов")
):
    try:
        result = fetch_from_adsabs(query, max_results)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/cern", summary="Поиск датасетов в CERN Open Data")
async def get_cern(
    query: str = Query(..., description="Строка запроса для датасетов в CERN Open Data"),
    max_results: int = Query(10, ge=1, le=50, description="Количество результатов")
):
    try:
        result = fetch_from_cern(query, max_results)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/google", summary="Поиск датасетов в Google Dataset Search (через SerpAPI)")
async def get_google_dataset(
    query: str = Query(..., description="Строка запроса для Google Dataset Search"),
    max_results: int = Query(10, ge=1, le=50, description="Количество результатов")
):
    try:
        result = fetch_from_google_dataset_search(query, max_results)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/nasa", summary="Поиск датасетов в NASA Open Data")
async def get_nasa(
    query: str = Query(..., description="Строка запроса для NASA Open Data"),
    max_results: int = Query(10, ge=1, le=50, description="Количество результатов")
):
    try:
        result = fetch_from_nasa(query, max_results)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/openml", summary="Поиск датасетов на OpenML")
async def get_openml(
    query: str = Query("", description="Строка запроса для OpenML"),
    max_results: int = Query(10, ge=1, le=50, description="Количество результатов")
):
    try:
        result = fetch_from_openml(query, max_results)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/biorxiv", summary="Поиск публикаций на BioRxiv через RSS")
async def get_biorxiv(
    query: str = Query(..., description="Строка запроса для BioRxiv"),
    max_results: int = Query(10, ge=1, le=50, description="Количество результатов")
):
    try:
        result = fetch_from_biorxiv(query, max_results)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Если Vercel требует наличия обработчика main, можно добавить:
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)