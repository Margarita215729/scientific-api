"""
API-модуль для работы с астрономическими каталогами.
Предоставляет эндпоинты для загрузки, обработки и получения данных из астрономических каталогов.
"""

from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
import os
import json
from typing import Optional, List, Dict, Any

# Импортируем функции из модуля astronomy_catalogs
from utils.astronomy_catalogs_real import (
    AstronomicalDataProcessor,
    get_catalog_info,
    get_comprehensive_statistics,
    fetch_filtered_galaxies,
    DATA_DIR, 
    OUTPUT_DIR
)

router = APIRouter()

# Фоновая задача для загрузки всех каталогов
running_jobs = {}

async def background_download_all_catalogs(job_id: str):
    """Фоновая задача для загрузки всех каталогов."""
    try:
        running_jobs[job_id] = {"status": "running", "progress": 0}
        
        # Создаем процессор данных
        processor = AstronomicalDataProcessor()
        catalogs = []
        
        try:
            sdss_path = await processor.download_sdss_data()
            catalogs.append({"name": "SDSS DR17", "path": sdss_path, "status": "success"})
            running_jobs[job_id]["progress"] = 25
        except Exception as e:
            catalogs.append({"name": "SDSS DR17", "status": "error", "error": str(e)})
        
        try:
            euclid_path = await processor.download_euclid_data()
            catalogs.append({"name": "Euclid Q1", "path": euclid_path, "status": "success"})
            running_jobs[job_id]["progress"] = 50
        except Exception as e:
            catalogs.append({"name": "Euclid Q1", "status": "error", "error": str(e)})
        
        try:
            desi_path = await processor.download_desi_data()
            catalogs.append({"name": "DESI DR1", "path": desi_path, "status": "success"})
            running_jobs[job_id]["progress"] = 75
        except Exception as e:
            catalogs.append({"name": "DESI DR1", "status": "error", "error": str(e)})
        
        try:
            des_path = await processor.download_des_data()
            catalogs.append({"name": "DES Y6", "path": des_path, "status": "success"})
            running_jobs[job_id]["progress"] = 90
        except Exception as e:
            catalogs.append({"name": "DES Y6", "status": "error", "error": str(e)})
        
        # Объединяем данные в единый набор
        try:
            catalog_paths = [cat["path"] for cat in catalogs if cat["status"] == "success"]
            merged_path = await processor.merge_catalogs(catalog_paths)
            catalogs.append({"name": "Merged Dataset", "path": merged_path, "status": "success"})
        except Exception as e:
            catalogs.append({"name": "Merged Dataset", "status": "error", "error": str(e)})
        
        # Завершаем задачу
        running_jobs[job_id] = {
            "status": "completed", 
            "progress": 100,
            "catalogs": catalogs
        }
    except Exception as e:
        running_jobs[job_id] = {
            "status": "failed",
            "error": str(e)
        }

@router.post("/download", summary="Запустить загрузку всех астрономических каталогов в фоновом режиме")
async def start_download(background_tasks: BackgroundTasks):
    """
    Запускает фоновую загрузку всех поддерживаемых астрономических каталогов:
    - SDSS DR17 spectroscopic catalog
    - Euclid Q1 MER Final catalog
    - DESI DR1 (2025) ELG clustering catalog
    - DES Year 6 (DES DR2/Y6 Gold) catalog
    
    Возвращает ID задачи, по которому можно отслеживать прогресс загрузки.
    """
    import uuid
    import asyncio
    job_id = str(uuid.uuid4())
    
    # Запускаем асинхронную задачу
    asyncio.create_task(background_download_all_catalogs(job_id))
    
    return {
        "job_id": job_id,
        "status": "started",
        "message": "Загрузка астрономических каталогов запущена"
    }

@router.get("/download/{job_id}", summary="Получить статус загрузки каталогов")
async def check_download_status(job_id: str):
    """
    Проверяет статус фоновой задачи загрузки каталогов по ID задачи.
    """
    if job_id not in running_jobs:
        raise HTTPException(status_code=404, detail=f"Задача с ID {job_id} не найдена")
    
    return running_jobs[job_id]

@router.get("/status", summary="Проверить статус доступных каталогов")
async def check_catalogs_status():
    """
    Проверяет, какие каталоги уже загружены и доступны для использования.
    """
    if not os.path.exists(OUTPUT_DIR):
        return {"status": "empty", "message": "Каталоги не загружены"}
    
    catalogs = await get_catalog_info()
    return {
        "status": "ok",
        "catalogs": catalogs,
        "data_directory": OUTPUT_DIR
    }

@router.get("/galaxies", summary="Получить подмножество данных галактик с фильтрацией")
async def get_galaxies(
    source: Optional[str] = Query(None, description="Источник данных (SDSS, Euclid, DESI, DES)"),
    limit: int = Query(1000, ge=1, le=100000, description="Максимальное количество возвращаемых строк"),
    min_z: Optional[float] = Query(None, description="Минимальное красное смещение"),
    max_z: Optional[float] = Query(None, description="Максимальное красное смещение"),
    min_ra: Optional[float] = Query(None, description="Минимальное прямое восхождение (RA)"),
    max_ra: Optional[float] = Query(None, description="Максимальное прямое восхождение (RA)"),
    min_dec: Optional[float] = Query(None, description="Минимальное склонение (DEC)"),
    max_dec: Optional[float] = Query(None, description="Максимальное склонение (DEC)"),
    format: str = Query("json", description="Формат ответа (json или csv)")
):
    """
    Возвращает подмножество данных галактик с применением различных фильтров.
    
    - **source**: Ограничение по источнику данных (SDSS, Euclid, DESI, DES)
    - **limit**: Максимальное количество возвращаемых строк
    - **min_z/max_z**: Фильтрация по красному смещению
    - **min_ra/max_ra**: Фильтрация по прямому восхождению
    - **min_dec/max_dec**: Фильтрация по склонению
    - **format**: Формат ответа (json или csv)
    """
    # Подготавливаем фильтры
    filters = {
        "source": source,
        "min_z": min_z,
        "max_z": max_z,
        "min_ra": min_ra,
        "max_ra": max_ra,
        "min_dec": min_dec,
        "max_dec": max_dec,
        "limit": limit
    }
    
    try:
        # Получаем отфильтрованные данные
        result = await fetch_filtered_galaxies(filters, include_ml_features=False)
        galaxies = result["galaxies"]
        
        # Возвращаем в запрошенном формате
        if format.lower() == "csv":
            df = pd.DataFrame(galaxies)
            csv_data = df.to_csv(index=False)
            return JSONResponse(
                content={"data": csv_data, "rows": len(galaxies)},
                headers={"Content-Disposition": "attachment; filename=galaxies.csv"}
            )
        
        # JSON формат по умолчанию
        return {
            "count": len(galaxies),
            "source": source or "all",
            "galaxies": galaxies,
            "processing_time": result.get("processing_time", 0)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при получении данных: {str(e)}")

@router.get("/statistics", summary="Получить статистику по каталогам галактик")
async def get_catalog_statistics(
    catalogs: List[str] = Query(None, description="Список каталогов для анализа (sdss, euclid, desi, des, all)")
):
    """
    Возвращает статистические данные о загруженных каталогах галактик.
    
    - **catalogs**: Список каталогов для анализа (sdss, euclid, desi, des, all)
    Если не указан, используются все доступные каталоги.
    """
    try:
        # Получаем комплексную статистику
        stats = await get_comprehensive_statistics()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при получении статистики: {str(e)}") 