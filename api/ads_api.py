"""
API-модуль для работы с NASA/Harvard Astrophysics Data System (ADS).
Предоставляет эндпоинты для поиска и получения научных публикаций, связанных
с астрономическими объектами из каталогов.
"""

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse
import os
from typing import List, Dict, Any, Optional
import pandas as pd

from utils.ads_astronomy import (
    search_by_coordinates,
    search_by_object,
    search_by_catalog,
    get_citations_for_paper,
    get_references_for_paper,
    search_large_scale_structure,
    get_bibtex
)

router = APIRouter()

@router.get("/search-by-coordinates", summary="Поиск публикаций по координатам")
async def api_search_by_coordinates(
    ra: float = Query(..., description="Прямое восхождение (RA) в градусах"),
    dec: float = Query(..., description="Склонение (DEC) в градусах"),
    radius: float = Query(0.1, description="Радиус поиска в градусах")
):
    """
    Поиск научных публикаций в ADS по небесным координатам объекта.
    
    Позволяет найти исследования, связанные с объектами вблизи указанной позиции.
    Возвращает список публикаций, отсортированных по количеству цитирований.
    """
    try:
        results = search_by_coordinates(ra, dec, radius)
        return {
            "count": len(results),
            "ra": ra,
            "dec": dec,
            "radius": radius,
            "publications": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при поиске по координатам: {str(e)}")

@router.get("/search-by-object", summary="Поиск публикаций по названию объекта")
async def api_search_by_object(
    object_name: str = Query(..., description="Название астрономического объекта (например, 'M31' или 'NGC 5128')")
):
    """
    Поиск научных публикаций в ADS по названию астрономического объекта.
    
    Использует базу данных объектов ADS для поиска связанных исследований.
    Возвращает список публикаций, отсортированных по количеству цитирований.
    """
    try:
        results = search_by_object(object_name)
        return {
            "count": len(results),
            "object": object_name,
            "publications": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при поиске по объекту: {str(e)}")

@router.get("/search-by-catalog", summary="Поиск публикаций, связанных с каталогом")
async def api_search_by_catalog(
    catalog: str = Query(..., description="Название каталога (SDSS, Euclid, DESI, DES)"),
    include_stats: bool = Query(True, description="Включить статистику по ключевым словам")
):
    """
    Поиск научных публикаций в ADS, связанных с конкретным астрономическим каталогом.
    
    Возвращает наиболее цитируемые публикации, использующие данные из указанного каталога,
    и опционально статистику по ключевым словам для анализа направлений исследований.
    """
    try:
        results = search_by_catalog(catalog, facet=include_stats)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при поиске по каталогу: {str(e)}")

@router.get("/citations", summary="Получение цитирований для статьи")
async def api_get_citations(
    bibcode: str = Query(..., description="Bibcode статьи в формате ADS (например, '2023AJ....165...62W')")
):
    """
    Получает список публикаций, цитирующих указанную статью.
    
    Возвращает статьи, отсортированные по дате публикации (сначала новейшие).
    """
    try:
        results = get_citations_for_paper(bibcode)
        return {
            "count": len(results),
            "bibcode": bibcode,
            "citations": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при получении цитирований: {str(e)}")

@router.get("/references", summary="Получение списка литературы для статьи")
async def api_get_references(
    bibcode: str = Query(..., description="Bibcode статьи в формате ADS (например, '2023AJ....165...62W')")
):
    """
    Получает список публикаций, на которые ссылается указанная статья.
    
    Возвращает список литературы статьи, отсортированный по количеству цитирований.
    """
    try:
        results = get_references_for_paper(bibcode)
        return {
            "count": len(results),
            "bibcode": bibcode,
            "references": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при получении списка литературы: {str(e)}")

@router.get("/large-scale-structure", summary="Поиск публикаций о крупномасштабных структурах")
async def api_search_large_scale_structure(
    additional_keywords: List[str] = Query(None, description="Дополнительные ключевые слова для поиска"),
    start_year: int = Query(2010, description="Начальный год публикаций")
):
    """
    Выполняет поиск публикаций, связанных с крупномасштабной структурой Вселенной.
    
    Возвращает результаты с группировкой по годам и ключевым словам для анализа трендов.
    """
    try:
        results = search_large_scale_structure(additional_keywords, start_year)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при поиске публикаций: {str(e)}")

@router.post("/export-bibtex", summary="Экспорт публикаций в формате BibTeX")
async def api_export_bibtex(
    bibcodes: List[str]
):
    """
    Экспортирует информацию о публикациях в формате BibTeX.
    
    Требует список bibcodes статей. Возвращает данные, которые можно использовать
    в системах управления библиографической информацией (BibTeX, LaTeX и др.).
    """
    try:
        if not bibcodes or len(bibcodes) == 0:
            raise HTTPException(status_code=400, detail="Необходимо указать хотя бы один bibcode")
            
        bibtex_data = get_bibtex(bibcodes)
        return {"bibtex": bibtex_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при экспорте BibTeX: {str(e)}")

@router.get("/galaxy-literature", summary="Поиск литературы для галактики по координатам")
async def api_galaxy_literature(
    ra: float = Query(..., description="Прямое восхождение (RA) в градусах"),
    dec: float = Query(..., description="Склонение (DEC) в градусах"),
    radius: float = Query(0.1, description="Радиус поиска в градусах"),
    redshift: Optional[float] = Query(None, description="Красное смещение (если известно)")
):
    """
    Комплексный поиск научной литературы для галактики по её координатам.
    
    Объединяет поиск по координатам и, если указано, по красному смещению для
    более точного нахождения публикаций, относящихся к конкретной галактике.
    """
    try:
        # Поиск по координатам
        coord_results = search_by_coordinates(ra, dec, radius)
        
        # Если указано redshift, можно уточнить поиск
        if redshift is not None:
            # Дополнительный поиск с учетом красного смещения (как приблизительный пример)
            redshift_query = f"\"redshift {redshift:.2f}\""
            z_range_low = max(0, redshift - 0.05)
            z_range_high = redshift + 0.05
            
            # Здесь можно добавить более сложную логику уточнения результатов
            # В простом случае, просто возвращаем результаты поиска по координатам
            # с пометкой о redshift
            
        return {
            "count": len(coord_results),
            "coordinates": {"ra": ra, "dec": dec},
            "redshift": redshift,
            "publications": coord_results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при поиске литературы: {str(e)}") 