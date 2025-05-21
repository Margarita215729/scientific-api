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
from utils.astronomy_catalogs import (
    get_all_catalogs, 
    get_sdss_data, 
    get_euclid_data, 
    get_desi_data, 
    get_des_data,
    get_euclid_by_regions,
    merge_all_data,
    convert_to_cartesian,
    fetch_galaxy_subset,
    DATA_DIR, 
    OUTPUT_DIR
)

router = APIRouter()

# Фоновая задача для загрузки всех каталогов
running_jobs = {}

def background_download_all_catalogs(job_id: str):
    """Фоновая задача для загрузки всех каталогов."""
    try:
        running_jobs[job_id] = {"status": "running", "progress": 0}
        
        # Последовательно загружаем каталоги
        catalogs = []
        
        try:
            sdss_path = get_sdss_data()
            catalogs.append({"name": "SDSS DR17", "path": sdss_path, "status": "success"})
            running_jobs[job_id]["progress"] = 25
        except Exception as e:
            catalogs.append({"name": "SDSS DR17", "status": "error", "error": str(e)})
        
        try:
            euclid_path = get_euclid_data()
            catalogs.append({"name": "Euclid Q1", "path": euclid_path, "status": "success"})
            running_jobs[job_id]["progress"] = 50
        except Exception as e:
            catalogs.append({"name": "Euclid Q1", "status": "error", "error": str(e)})
        
        try:
            desi_path = get_desi_data()
            catalogs.append({"name": "DESI DR1", "path": desi_path, "status": "success"})
            running_jobs[job_id]["progress"] = 75
        except Exception as e:
            catalogs.append({"name": "DESI DR1", "status": "error", "error": str(e)})
        
        try:
            des_path = get_des_data()
            catalogs.append({"name": "DES Y6", "path": des_path, "status": "success"})
            running_jobs[job_id]["progress"] = 90
        except Exception as e:
            catalogs.append({"name": "DES Y6", "status": "error", "error": str(e)})
        
        # Объединяем данные в единый набор
        try:
            merged_path = merge_all_data()
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
    job_id = str(uuid.uuid4())
    background_tasks.add_task(background_download_all_catalogs, job_id)
    
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
    
    catalogs = []
    catalog_files = {
        "sdss.csv": {"name": "SDSS DR17", "description": "Spectroscopic catalog"},
        "euclid.csv": {"name": "Euclid Q1", "description": "MER Final catalog"},
        "desi.csv": {"name": "DESI DR1", "description": "ELG clustering catalog"},
        "des.csv": {"name": "DES Y6", "description": "Gold catalog"},
        "merged_galaxies.csv": {"name": "Merged Dataset", "description": "Объединенный набор данных"}
    }
    
    for filename, info in catalog_files.items():
        filepath = os.path.join(OUTPUT_DIR, filename)
        if os.path.exists(filepath):
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            
            # Подсчет количества строк (чтение только первой строки для определения формата)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    header = f.readline().strip()
                    
                # Быстрый подсчет строк
                with open(filepath, 'r', encoding='utf-8') as f:
                    row_count = sum(1 for _ in f) - 1  # Вычитаем заголовок
            except Exception as e:
                row_count = "Ошибка подсчета"
                
            catalogs.append({
                "name": info["name"],
                "description": info["description"],
                "filename": filename,
                "size_mb": round(size_mb, 2),
                "rows": row_count,
                "available": True
            })
        else:
            catalogs.append({
                "name": info["name"],
                "description": info["description"],
                "available": False
            })
    
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
    merged_path = os.path.join(OUTPUT_DIR, "merged_galaxies.csv")
    
    # Проверяем, существует ли объединенный файл
    if not os.path.exists(merged_path):
        try:
            # Пробуем создать объединенный файл, если его нет
            _ = merge_all_data()
        except Exception as e:
            raise HTTPException(
                status_code=404, 
                detail=f"Данные галактик не найдены. Сначала запустите загрузку каталогов через /astro/download: {str(e)}"
            )
    
    try:
        # Читаем данные из файла
        df = pd.read_csv(merged_path)
        
        # Применяем фильтры
        if source:
            df = df[df["source"] == source]
        
        if min_z is not None:
            df = df[df["redshift"] >= min_z]
        
        if max_z is not None:
            df = df[df["redshift"] <= max_z]
        
        if min_ra is not None:
            df = df[df["RA"] >= min_ra]
        
        if max_ra is not None:
            df = df[df["RA"] <= max_ra]
        
        if min_dec is not None:
            df = df[df["DEC"] >= min_dec]
        
        if max_dec is not None:
            df = df[df["DEC"] <= max_dec]
        
        # Ограничиваем количество строк
        if limit is not None and len(df) > limit:
            df = df.sample(limit) if limit < len(df) else df
        
        # Возвращаем в запрошенном формате
        if format.lower() == "csv":
            csv_data = df.to_csv(index=False)
            return JSONResponse(
                content={"data": csv_data, "rows": len(df)},
                headers={"Content-Disposition": "attachment; filename=galaxies.csv"}
            )
        
        # JSON формат по умолчанию
        # Обрабатываем NaN значения
        df = df.replace({np.nan: None})
        
        return {
            "count": len(df),
            "source": source or "all",
            "galaxies": df.to_dict(orient="records")
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
    merged_path = os.path.join(OUTPUT_DIR, "merged_galaxies.csv")
    
    # Проверяем, существует ли объединенный файл
    if not os.path.exists(merged_path):
        raise HTTPException(
            status_code=404, 
            detail="Данные галактик не найдены. Сначала запустите загрузку каталогов через /astro/download"
        )
    
    try:
        # Читаем данные из файла
        df = pd.read_csv(merged_path)
        
        # Фильтруем по указанным каталогам
        if catalogs and "all" not in catalogs:
            valid_sources = []
            map_names = {"sdss": "SDSS", "euclid": "Euclid", "desi": "DESI", "des": "DES"}
            for c in catalogs:
                if c.lower() in map_names:
                    valid_sources.append(map_names[c.lower()])
            
            if valid_sources:
                df = df[df["source"].isin(valid_sources)]
        
        if len(df) == 0:
            return {"message": "Нет данных, соответствующих запросу"}
        
        # Общая статистика
        total_galaxies = len(df)
        sources_count = df["source"].value_counts().to_dict()
        
        # Статистика по z (красному смещению)
        z_stats = {
            "min": float(df["redshift"].min()) if not pd.isna(df["redshift"].min()) else None,
            "max": float(df["redshift"].max()) if not pd.isna(df["redshift"].max()) else None,
            "mean": float(df["redshift"].mean()) if not pd.isna(df["redshift"].mean()) else None,
            "median": float(df["redshift"].median()) if not pd.isna(df["redshift"].median()) else None,
            "available_percent": float((df["redshift"].notna().sum() / total_galaxies) * 100)
        }
        
        # Статистика по координатам
        coord_stats = {
            "ra": {
                "min": float(df["RA"].min()),
                "max": float(df["RA"].max()),
                "coverage_degrees": float(df["RA"].max() - df["RA"].min())
            },
            "dec": {
                "min": float(df["DEC"].min()),
                "max": float(df["DEC"].max()),
                "coverage_degrees": float(df["DEC"].max() - df["DEC"].min())
            }
        }
        
        # Статистика по 3D-координатам
        xyz_stats = {
            "x": {
                "min": float(df["X"].min()) if not pd.isna(df["X"].min()) else None,
                "max": float(df["X"].max()) if not pd.isna(df["X"].max()) else None,
                "range_mpc": float(df["X"].max() - df["X"].min()) if not pd.isna(df["X"].min()) else None
            },
            "y": {
                "min": float(df["Y"].min()) if not pd.isna(df["Y"].min()) else None,
                "max": float(df["Y"].max()) if not pd.isna(df["Y"].max()) else None,
                "range_mpc": float(df["Y"].max() - df["Y"].min()) if not pd.isna(df["Y"].min()) else None
            },
            "z": {
                "min": float(df["Z"].min()) if not pd.isna(df["Z"].min()) else None,
                "max": float(df["Z"].max()) if not pd.isna(df["Z"].max()) else None,
                "range_mpc": float(df["Z"].max() - df["Z"].min()) if not pd.isna(df["Z"].min()) else None
            },
            "distance_mpc": {
                "min": float(df["distance_mpc"].min()) if not pd.isna(df["distance_mpc"].min()) else None,
                "max": float(df["distance_mpc"].max()) if not pd.isna(df["distance_mpc"].max()) else None,
                "mean": float(df["distance_mpc"].mean()) if not pd.isna(df["distance_mpc"].mean()) else None
            }
        }
        
        return {
            "total_galaxies": total_galaxies,
            "sources": sources_count,
            "redshift": z_stats,
            "celestial_coordinates": coord_stats,
            "cartesian_coordinates": xyz_stats
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при получении статистики: {str(e)}") 