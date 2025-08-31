"""
Heavy compute service API endpoints for real astronomical data processing.
This service handles data processing, machine learning, and resource-intensive operations.
Designed for Microsoft Container Instances with 12 CPU cores and 20GB RAM.
Now integrates with the database for data sourcing and status checking.
"""

from fastapi import FastAPI, APIRouter, HTTPException, BackgroundTasks, Query
from fastapi.responses import JSONResponse, FileResponse
import os
import logging
from typing import Dict, Any, List, Optional
import asyncio
from datetime import datetime
import json

# Configure logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database and Preprocessor
try:
    from database.config import db # Используем глобальный экземпляр db
    DB_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Database module not available: {e}")
    db = None
    DB_AVAILABLE = False

try:
    from utils.data_preprocessor import AstronomicalDataPreprocessor # Для запуска пайплайна
    PREPROCESSOR_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Data preprocessor not available: {e}")
    AstronomicalDataPreprocessor = None
    PREPROCESSOR_AVAILABLE = False

try:
    from utils.data_processing import DataProcessor, process_astronomical_data_entrypoint # Для кастомной обработки
    PROCESSING_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Data processing module not available: {e}")
    DataProcessor = None
    process_astronomical_data_entrypoint = None
    PROCESSING_AVAILABLE = False

# HTTP client (если нужен для проксирования на другие тяжелые сервисы, пока не используется)
# try:
#     import httpx
#     HTTPX_AVAILABLE = True
# except ImportError:
#     httpx = None
#     HTTPX_AVAILABLE = False

# Heavy compute libraries (для локальной обработки, если потребуется)
try:
    import pandas as pd
    import numpy as np
    HEAVY_LIBS_AVAILABLE = True
except ImportError:
    pd = None
    np = None
    HEAVY_LIBS_AVAILABLE = False

# Logger already configured above

# Data directories (могут быть менее релевантны, если все в БД)
# DATA_DIR = "galaxy_data"
# PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

# HEAVY_COMPUTE_URL и USE_AZURE_API могут быть не нужны, если этот сервис сам выполняет задачи
# HEAVY_COMPUTE_URL = os.getenv("HEAVY_COMPUTE_URL", "").strip()
# USE_AZURE_API = bool(HEAVY_COMPUTE_URL) and not HEAVY_LIBS_AVAILABLE 
# logger.info(f"Heavy API initialized - HEAVY_LIBS_AVAILABLE: {HEAVY_LIBS_AVAILABLE}, USE_AZURE_API: {USE_AZURE_API}")

router = APIRouter()

# Global storage for background tasks status (простое внутрипроцессное хранилище)
# В продакшене лучше использовать Redis или БД для этого
background_tasks_status: Dict[str, Dict[str, Any]] = {}

async def get_current_data_status_from_db() -> Dict[str, Any]:
    """Получает актуальный статус данных из БД."""
    if not DB_AVAILABLE:
        return {"status": "database_not_available", "message": "Database dependencies not loaded", "total_objects": -1}

    try:
        # Check database connection based on the type
        if db.db_type == "cosmosdb_mongo" and db.mongo_client:
            db_connected = True
        elif db.db_type in ["sqlite", "postgresql"] and db.sql_connection:
            db_connected = True
        else:
            db_connected = False
            
        if not db_connected: # Проверка, что соединение было инициализировано
             logger.warning("Attempting to get data status from DB, but DB connection is not active. Trying to connect.")
             await db.connect() # Попытка подключиться, если еще не подключены
             if not (db.mongo_client or db.sql_connection):
                  raise ConnectionError("Database connection could not be established.")

        # Получаем общую статистику, где может быть информация о количестве объектов
        stats = await db.get_statistics() 
        # Пример: ищем метрику 'total_objects_astronomical' или подобную
        total_objects = stats.get('total_objects_astronomical', {}).get('value', 0)
        if not total_objects: # Если такой метрики нет, посчитаем напрямую (может быть медленно)
            objects_sample = await db.get_astronomical_objects(limit=1) # Проверка на наличие хоть каких-то данных
            if objects_sample: # Если есть хоть один, то статус "обработано", но количество неизвестно точно без подсчета
                 # Для точного количества нужен был бы SELECT COUNT(*)
                 # Пока для простоты, если есть хоть один объект, считаем что данные есть
                 total_objects = "partial_data_present" # Не число, а строка-статус
            else:
                total_objects = 0

        # Собираем информацию по каталогам (какие есть в БД)
        # Это может потребовать дополнительного запроса, если мы хотим знать количество по каждому catalog_source
        # Примерный запрос: SELECT catalog_source, COUNT(*) FROM astronomical_objects GROUP BY catalog_source
        # Для простоты, пока не будем делать детальную разбивку по каталогам здесь
        processed_catalogs_info = []
        # Если в stats есть информация по каждому каталогу, используем ее
        # Например, stats.get('SDSS_object_count', {}).get('value', 0)
        # Сейчас просто заглушка
        # for src in ["SDSS", "DESI", "DES", "Euclid"]:
        #     count = stats.get(f"{src}_object_count", {}).get('value')
        #     if count is not None:
        #         processed_catalogs_info.append({"name": src, "objects": count, "status": "processed"})

        if total_objects == 0:
            return {"status": "not_available", "message": "No astronomical data found in the database.", "total_objects": 0}
        
        # Если total_objects это строка-статус, то выводим как есть
        is_processed = isinstance(total_objects, (int, float)) and total_objects > 0 or isinstance(total_objects, str)

        return {
            "status": "completed" if is_processed else "processing_or_empty",
            "message": "Data status from database.",
            "total_objects_in_db": total_objects, # Может быть число или строка-статус
            "catalogs_summary_from_db": processed_catalogs_info, # Пока пустой, нужно доработать
            "last_checked_db_at": datetime.utcnow().isoformat() + "Z"
        }
    except Exception as e:
        logger.error(f"Error getting data status from DB: {e}", exc_info=True)
        return {"status": "error_checking_db", "message": str(e), "total_objects": -1}

@router.get("/ping", tags=["Heavy System"])
async def ping_heavy():
    """Health check for the heavy compute service itself."""
    db_status_info = await get_current_data_status_from_db() if DB_AVAILABLE else {"status": "database_not_available", "message": "Database dependencies not loaded"}
    return {
        "status": "ok" if all([HEAVY_LIBS_AVAILABLE, DB_AVAILABLE, PREPROCESSOR_AVAILABLE, PROCESSING_AVAILABLE]) else "partial",
        "message": "Heavy compute service is operational." if all([HEAVY_LIBS_AVAILABLE, DB_AVAILABLE, PREPROCESSOR_AVAILABLE, PROCESSING_AVAILABLE]) else "Some dependencies not available - running in limited mode",
        "service_type": "heavy-compute-integrated-db",
        "version": "2.1.0",
        "dependencies_status": {
            "heavy_libs_available": HEAVY_LIBS_AVAILABLE,
            "database_available": DB_AVAILABLE,
            "preprocessor_available": PREPROCESSOR_AVAILABLE,
            "processing_available": PROCESSING_AVAILABLE
        },
        "database_integration_status": db_status_info
    }

@router.post("/astro/trigger-preprocessing", tags=["Heavy Astronomical Data"])
async def trigger_full_preprocessing(background_tasks: BackgroundTasks):
    """Запускает полный пайплайн предварительной обработки всех астрономических каталогов."""
    if not all([PREPROCESSOR_AVAILABLE, DB_AVAILABLE]):
        raise HTTPException(
            status_code=503,
            detail="Preprocessing not available - missing dependencies (data_preprocessor or database)"
        )

    import uuid
    task_id = str(uuid.uuid4())

    # Запускаем AstronomicalDataPreprocessor.preprocess_all_catalogs в фоновой задаче
    background_tasks.add_task(run_astronomical_data_pipeline, task_id)

    background_tasks_status[task_id] = {
        "status": "started",
        "task_type": "full_catalog_preprocessing",
        "progress": 0,
        "message": "Initializing full astronomical data preprocessing pipeline...",
        "started_at": datetime.utcnow().isoformat(),
        "details": "This task will download, process, and store data for all configured catalogs into the database."
    }

    logger.info(f"Task {task_id} (full_catalog_preprocessing) started.")
    return {
        "task_id": task_id,
        "status": "started",
        "message": "Full astronomical data preprocessing pipeline initiated."
    }

async def run_astronomical_data_pipeline(task_id: str):
    """Обертка для запуска AstronomicalDataPreprocessor в фоновой задаче."""
    try:
        background_tasks_status[task_id]["message"] = "Data Preprocessor pipeline starting..."
        background_tasks_status[task_id]["progress"] = 5
        
        preprocessor = AstronomicalDataPreprocessor()
        # preprocess_all_catalogs уже подключается к БД и инициализирует ее, если нужно
        results = await preprocessor.preprocess_all_catalogs()
        
        background_tasks_status[task_id].update({
            "status": "completed" if results.get("status", "").startswith("completed") else "failed",
            "progress": 100,
            "message": f"Pipeline finished: {results.get('status')}. Total objects processed: {results.get('total_objects_processed_all_catalogs', 0)}",
            "result_summary": results,
            "completed_at": datetime.utcnow().isoformat()
        })
        logger.info(f"Task {task_id} (full_catalog_preprocessing) completed with status: {results.get('status')}")
    except Exception as e:
        logger.error(f"Error in background task {task_id} (full_catalog_preprocessing): {e}", exc_info=True)
        background_tasks_status[task_id].update({
            "status": "failed",
            "progress": -1, # Ошибка на каком-то этапе
            "message": f"Pipeline failed with error: {str(e)}",
            "error_details": str(e),
            "completed_at": datetime.utcnow().isoformat()
        })

@router.get("/task-status/{task_id}", tags=["Heavy System"])
async def get_task_status(task_id: str):
    """Получить статус фоновой задачи по её ID."""
    if task_id not in background_tasks_status:
        raise HTTPException(status_code=404, detail=f"Task with ID '{task_id}' not found.")
    return background_tasks_status[task_id]

@router.get("/astro/status", tags=["Heavy Astronomical Data"])
async def get_heavy_astronomical_status():
    """Получить актуальный статус доступности данных из базы данных."""
    status_info = await get_current_data_status_from_db()
    if status_info["status"] == "error_checking_db":
        raise HTTPException(status_code=503, detail=f"Error checking database status: {status_info['message']}")
    return status_info


@router.get("/astro/galaxies", tags=["Heavy Astronomical Data"])
async def get_galaxies_data_from_db(
    source: Optional[str] = Query(None, description="Source catalog (SDSS, Euclid, DESI, DES, etc.)"),
    limit: int = Query(100, ge=1, le=10000, description="Maximum number of rows"),
    min_z: Optional[float] = Query(None, description="Minimum redshift"),
    max_z: Optional[float] = Query(None, description="Maximum redshift"),
    min_ra: Optional[float] = Query(None, description="Minimum RA (degrees)"),
    max_ra: Optional[float] = Query(None, description="Maximum RA (degrees)"),
    min_dec: Optional[float] = Query(None, description="Minimum DEC (degrees)"),
    max_dec: Optional[float] = Query(None, description="Maximum DEC (degrees)")
):
    """Получить отфильтрованные данные галактик напрямую из базы данных."""
    if not DB_AVAILABLE:
        raise HTTPException(status_code=503, detail="Database dependencies not available")

    try:
        if not (db.mongo_client or db.sql_connection):
             raise HTTPException(status_code=503, detail="Database not connected. Run preprocessing pipeline or check connection.")

        # Формируем фильтры для db.get_astronomical_objects
        # Этот метод должен быть способен принимать эти фильтры
        # Пока db.get_astronomical_objects принимает только limit, object_type, catalog_source
        # Для более сложной фильтрации его нужно будет доработать или использовать execute_query
        
        # Упрощенная выборка: используем catalog_source, если он есть, и limit.
        # Остальные фильтры (ra, dec, z) потребуют доработки get_astronomical_objects или прямого SQL/Cosmos запроса.
        logger.info(f"Fetching galaxies from DB. Source: {source}, Limit: {limit}, RA: {min_ra}-{max_ra}, DEC: {min_dec}-{max_dec}, Z: {min_z}-{max_z}")

        # Собираем SQL и параметры для SQL баз, или параметры для CosmosDB
        sql_query_parts = ["SELECT * FROM astronomical_objects"]
        conditions = []
        params_list = []

        if source:
            conditions.append(f"catalog_source = { '?' if db.db_type == 'sqlite' else '$%d' % (len(params_list)+1) }")
            params_list.append(source)
        if min_z is not None:
            conditions.append(f"redshift >= { '?' if db.db_type == 'sqlite' else '$%d' % (len(params_list)+1) }")
            params_list.append(min_z)
        if max_z is not None:
            conditions.append(f"redshift <= { '?' if db.db_type == 'sqlite' else '$%d' % (len(params_list)+1) }")
            params_list.append(max_z)
        if min_ra is not None:
            conditions.append(f"ra >= { '?' if db.db_type == 'sqlite' else '$%d' % (len(params_list)+1) }")
            params_list.append(min_ra)
        if max_ra is not None:
            conditions.append(f"ra <= { '?' if db.db_type == 'sqlite' else '$%d' % (len(params_list)+1) }")
            params_list.append(max_ra)
        if min_dec is not None:
            conditions.append(f"dec >= { '?' if db.db_type == 'sqlite' else '$%d' % (len(params_list)+1) }")
            params_list.append(min_dec)
        if max_dec is not None:
            conditions.append(f"dec <= { '?' if db.db_type == 'sqlite' else '$%d' % (len(params_list)+1) }")
            params_list.append(max_dec)

        if conditions:
            sql_query_parts.append("WHERE " + " AND ".join(conditions))
        
        sql_query_parts.append(f"LIMIT { '?' if db.db_type == 'sqlite' else '$%d' % (len(params_list)+1) }")
        params_list.append(limit)

        final_query = " ".join(sql_query_parts)
        
        if db.db_type == "cosmosdb":
            # Для CosmosDB нужна другая логика построения запроса и параметров
            # Используем существующий get_astronomical_objects, если он подходит
            # или строим кастомный запрос.
            # Пока что для CosmosDB будем использовать только source и limit через get_astronomical_objects
            # для более сложной фильтрации, get_astronomical_objects нужно доработать.
            logger.warning("CosmosDB query for galaxies currently only supports 'source' and 'limit'. Other filters ignored for CosmosDB in this simplified version.")
            galaxies = await db.get_astronomical_objects(limit=limit, catalog_source=source)
        else: # SQL
            galaxies = await db.execute_query(final_query, tuple(params_list))

        if not galaxies:
            return {"status": "no_data", "count": 0, "galaxies": [], "filters_applied": {"source": source, "limit": limit, "min_z": min_z, "max_z": max_z, "min_ra":min_ra, "max_ra":max_ra, "min_dec":min_dec, "max_dec":max_dec}}
        
        # Преобразование в float там, где это числа, если они пришли как строки из БД (особенно SQLite)
        # Также заменяем NaN/Infinity на None для JSON-совместимости
        processed_galaxies = []
        for gal in galaxies:
            processed_gal = dict(gal) # Копируем, чтобы не изменять исходный объект Row
            for key, value in processed_gal.items():
                if HEAVY_LIBS_AVAILABLE and isinstance(value, (float, int)) and (np.isnan(value) or np.isinf(value)):
                    processed_gal[key] = None
                # Опционально: преобразование числовых строк в числа, если это не делает драйвер БД
                # elif isinstance(value, str):
                # try:
                # processed_gal[key] = float(value)
                # except (ValueError, TypeError):
                # pass # Оставляем как есть, если не конвертируется
            processed_galaxies.append(processed_gal)

        return {
            "status": "ok",
            "count": len(processed_galaxies),
            "galaxies": processed_galaxies,
            "filters_applied": {"source": source, "limit": limit, "min_z": min_z, "max_z": max_z, "min_ra":min_ra, "max_ra":max_ra, "min_dec":min_dec, "max_dec":max_dec}
        }
        
    except HTTPException: # Перехватываем HTTP исключения, чтобы не логировать их как 500
        raise
    except Exception as e:
        logger.error(f"Error getting galaxy data from DB: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error loading galaxy data from database: {str(e)}")

@router.get("/astro/statistics", tags=["Heavy Astronomical Data"])
async def get_heavy_astronomical_statistics():
    """Получить агрегированную статистику по всем астрономическим данным в БД."""
    try:
        if not (db.mongo_client or db.sql_connection):
             raise HTTPException(status_code=503, detail="Database not connected. Run preprocessing pipeline or check connection.")

        stats = await db.get_statistics()
        # Дополнительно можно посчитать некоторые статистики на лету, если их нет в system_statistics
        # Например, общее количество объектов:
        # count_res = await db.execute_query("SELECT COUNT(*) as total_obj FROM astronomical_objects")
        # if count_res and 'total_obj' in count_res[0]:
        #     stats['total_objects_in_table'] = {'value': count_res[0]['total_obj'], 'unit': 'count'}

        if not stats:
             return {"status": "no_statistics", "message": "No statistics found in database. Possible run preprocessing or calculation tasks."}

        return {
            "status": "ok",
            "statistics_from_db": stats,
            "retrieved_at": datetime.utcnow().isoformat() + "Z"
        }
    except Exception as e:
        logger.error(f"Error getting statistics from DB: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error loading statistics from database: {str(e)}")


@router.post("/data/custom-process", tags=["Heavy Data Processing"])
async def process_custom_astronomical_data_endpoint(
    background_tasks: BackgroundTasks,
    config: Dict[str, Any] # Конфигурация передается в теле POST запроса
):
    """Запускает кастомную обработку астрономических данных (clean, normalize, feature_engineering)."""
    if not all([PROCESSING_AVAILABLE, DB_AVAILABLE]):
        raise HTTPException(
            status_code=503,
            detail="Data processing not available - missing dependencies (data_processing or database)"
        )

    import uuid
    task_id = str(uuid.uuid4())

    # Валидация config может быть добавлена здесь с использованием Pydantic модели
    processing_type = config.get("processing_type")
    if not processing_type or processing_type not in ["clean", "normalize", "feature_engineering"]:
        raise HTTPException(status_code=400, detail="Invalid 'processing_type'. Must be one of: clean, normalize, feature_engineering.")

    background_tasks.add_task(run_custom_data_processing, task_id, config)
    
    background_tasks_status[task_id] = {
        "status": "started",
        "task_type": f"custom_data_processing_{processing_type}",
        "progress": 0,
        "message": f"Initializing custom data processing: {processing_type}...",
        "config": config,
        "started_at": datetime.utcnow().isoformat()
    }
    logger.info(f"Task {task_id} (custom_data_processing_{processing_type}) started with config: {config}")
    return {
        "task_id": task_id,
        "status": "started",
        "message": f"Custom data processing ('{processing_type}') initiated.",
        "config": config
    }

async def run_custom_data_processing(task_id: str, config: dict):
    """Обертка для запуска utils.data_processing.process_astronomical_data_entrypoint."""
    try:
        processing_type = config.get("processing_type", "unknown")
        background_tasks_status[task_id]["message"] = f"Custom data processing ('{processing_type}') starting..."
        background_tasks_status[task_id]["progress"] = 10
        
        # Убедимся, что БД подключена
        if not (db.mongo_client or db.sql_connection):
            await db.connect()

        # process_astronomical_data_entrypoint из utils.data_processing
        result_payload = await process_astronomical_data_entrypoint(config)
        
        background_tasks_status[task_id].update({
            "status": "completed" if result_payload.get("status") == "success" else "failed",
            "progress": 100,
            "message": f"Custom processing ('{processing_type}') finished: {result_payload.get('status')}.",
            "result_payload": result_payload, # Содержит отчет и путь к превью файла
            "completed_at": datetime.utcnow().isoformat()
        })
        logger.info(f"Task {task_id} (custom_data_processing_{processing_type}) completed with status: {result_payload.get('status')}")
    except Exception as e:
        logger.error(f"Error in background task {task_id} (custom_data_processing): {e}", exc_info=True)
        background_tasks_status[task_id].update({
            "status": "failed",
            "progress": -1,
            "message": f"Custom processing failed with error: {str(e)}",
            "error_details": str(e),
            "completed_at": datetime.utcnow().isoformat()
        })


# TODO: Добавить эндпоинты для ML, если они будут выполняться этим сервисом.
# Например, /ml/prepare-dataset, /ml/train-model и т.д.
# Эти эндпоинты должны также использовать фоновые задачи для длительных операций.

# Закомментированные эндпоинты ниже, т.к. они относятся к более старой структуре 
# или требуют значительной переработки для интеграции с БД.

# @router.get("/astro/full/galaxies") ... и другие /astro/full/* эндпоинты
# Если HEAVY_COMPUTE_URL не используется, эти эндпоинты должны быть реализованы здесь, 
# чтобы читать данные из БД, а не проксировать.
# Сейчас они заменены на /astro/galaxies, который читает из БД.

# @router.get("/datasets/list")
# @router.get("/files/status")
# @router.get("/ml/models")
# @router.get("/analysis/quick")
# Эти эндпоинты, если актуальны, должны быть переписаны для работы с текущей архитектурой (БД, фоновые задачи).


# Конец файла. app.include_router(router) будет в main_azure_with_db.py 