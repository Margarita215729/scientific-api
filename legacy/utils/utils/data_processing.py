"""
Data processing module for custom astronomical data cleaning and normalization.
Handles various input formats and provides comprehensive data processing pipelines.
"""

import os
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import json
from pathlib import Path

# Добавляем импорт для работы с БД
from database.config import db 

# Configure logging
logger = logging.getLogger(__name__)

# Constants
PROCESSING_OUTPUT_DIR = Path("galaxy_data/processed") # Используем Path
SUPPORTED_FORMATS = ["csv", "fits", "txt", "json", "parquet"]

class DataProcessor:
    """Main class for astronomical data processing."""
    
    def __init__(self):
        PROCESSING_OUTPUT_DIR.mkdir(parents=True, exist_ok=True) # Используем Path
        
    async def process_astronomical_data(self, config: Dict) -> Dict[str, Any]:
        """Main processing function. Теперь асинхронная для работы с БД."""
        processing_type = config.get("processing_type", "clean")
        input_format = config.get("input_format", "csv") # Этот параметр может стать менее релевантным
        output_format = config.get("output_format", "csv") # Аналогично
        
        logger.info(f"Processing data: {processing_type}, {input_format} -> {output_format}")
        
        if processing_type == "clean":
            return await self._clean_data(config) # await
        elif processing_type == "normalize":
            return await self._normalize_data(config) # await
        elif processing_type == "feature_engineering":
            return await self._engineer_features(config) # await
        else:
            logger.error(f"Unknown processing type: {processing_type}")
            raise ValueError(f"Unknown processing type: {processing_type}")
    
    async def _clean_data(self, config: Dict) -> Dict[str, Any]:
        """Clean astronomical data. Теперь асинхронная."""
        logger.info("Cleaning astronomical data...")
        
        # Load existing data from database
        df = await self._load_available_data_from_db()
        
        if df.empty:
            logger.warning("No data available from database for cleaning.")
            # Опционально: можно попробовать загрузить из файлов как fallback
            # df = self._load_available_data_from_files() 
            # if df.empty:
            #     raise Exception("No data available for processing from any source")
            return { "status": "no_data", "message": "No data in DB to clean."}

        
        original_count = len(df)
        
        # Remove duplicates (по основным полям, если object_id еще нет или он не уникален до чистки)
        # df = df.drop_duplicates(subset=['ra', 'dec', 'catalog_source']) # Пример
        df = df.drop_duplicates() # Простой вариант, если данные уже имеют некий ID
        logger.info(f"Removed {original_count - len(df)} duplicate rows")
        
        # Clean coordinate columns (примеры остаются те же)
        if "ra" in df.columns:
            df = df[(df["ra"].notna()) & (df["ra"] >= 0) & (df["ra"] <= 360)]
            if not df.empty:
                ra_q1, ra_q3 = df["ra"].quantile([0.01, 0.99])
                df = df[(df["ra"] >= ra_q1) & (df["ra"] <= ra_q3)]
        
        if "dec" in df.columns:
            df = df[(df["dec"].notna()) & (df["dec"] >= -90) & (df["dec"] <= 90)]
            if not df.empty:
                dec_q1, dec_q3 = df["dec"].quantile([0.01, 0.99])
                df = df[(df["dec"] >= dec_q1) & (df["dec"] <= dec_q3)]
        
        # Clean redshift values
        if "redshift" in df.columns:
            df = df[(df["redshift"].notna()) & (df["redshift"] >= 0) & (df["redshift"] <= 10)]
            if not df.empty:
                z_max = df["redshift"].quantile(0.995)
                df = df[df["redshift"] <= z_max]
        
        # Clean magnitude columns
        magnitude_cols = [col for col in df.columns if "mag_" in col.lower() or "magnitude" in col.lower()]
        for mag_col in magnitude_cols:
            if mag_col in df.columns:
                df = df[df[mag_col].notna()]
                df = df[(df[mag_col] >= 5) & (df[mag_col] <= 35)] # Примерный диапазон
                if not df.empty:
                    mag_q1, mag_q3 = df[mag_col].quantile([0.005, 0.995])
                    df = df[(df[mag_col] >= mag_q1) & (df[mag_col] <= mag_q3)]
        
        # Remove rows with too many missing values
        missing_threshold = 0.5  
        df = df.dropna(thresh=int(len(df.columns) * missing_threshold))
        
        # Clean 3D coordinates if present (X, Y, Z)
        coord_3d = ["X", "Y", "Z"] # Используем стандартные имена из schema.sql
        for coord_col_name in coord_3d:
            if coord_col_name in df.columns:
                df = df[df[coord_col_name].notna() & np.isfinite(df[coord_col_name])]
                if not df.empty:
                    coord_q1, coord_q3 = df[coord_col_name].quantile([0.005, 0.995])
                    df = df[(df[coord_col_name] >= coord_q1) & (df[coord_col_name] <= coord_q3)]
        
        # Сохранение очищенных данных обратно в БД (или как новую версию)
        # Это потребует решения: обновлять существующие записи или создавать новые "cleaned" записи.
        # Для примера, будем считать, что мы обновляем существующие записи, 
        # удаляя те, которые не прошли чистку, и обновляя те, что прошли.
        # Это сложная логика, здесь для простоты просто вернем DataFrame.
        # В реальном приложении: определить, какие ID были удалены, какие обновлены.

        # Сохранение в CSV пока оставим для отладки / промежуточных шагов
        output_filename = PROCESSING_OUTPUT_DIR / "cleaned_data.csv"
        df.to_csv(output_filename, index=False)
        logger.info(f"Cleaned data saved to {output_filename} for review. Rows: {len(df)}")
        
        # Обновление данных в БД (примерная логика)
        # Нужно будет сопоставить df с исходными данными по ID и обновить/удалить
        # Для Cosmos DB: собрать батч операций upsert/delete.
        # Для SQL: выполнить UPDATE и DELETE запросы.
        # Это выходит за рамки простого изменения, поэтому пока не реализуем полностью.
        logger.warning("Database update logic for cleaned data is not fully implemented here. Returning cleaned DataFrame.")

        cleaning_report = {
            "original_rows": original_count,
            "cleaned_rows": len(df),
            "removed_rows": original_count - len(df),
            "removal_percentage": ((original_count - len(df)) / original_count) * 100 if original_count > 0 else 0,
            "columns_processed": list(df.columns),
            "output_file_preview": str(output_filename) # Указываем, что это превью
        }
        
        logger.info(f"Data cleaning complete: {len(df)} rows remaining ({cleaning_report['removal_percentage']:.1f}% removed)")
        
        return {
            "status": "success",
            "processing_type": "clean",
            "report": cleaning_report,
            "cleaned_dataframe_preview_path": str(output_filename) # Возвращаем путь к файлу для отладки
            # "cleaned_data": df # Можно вернуть сам DataFrame, если он небольшой
        }
    
    async def _normalize_data(self, config: Dict) -> Dict[str, Any]:
        """Normalize astronomical data. Теперь асинхронная."""
        logger.info("Normalizing astronomical data...")
        
        # Загрузка данных: сначала пробуем из файла cleaned_data.csv (если предыдущий шаг его создал)
        # В идеале, этот шаг тоже должен брать данные из БД (например, с пометкой is_cleaned=True)
        cleaned_path = PROCESSING_OUTPUT_DIR / "cleaned_data.csv"
        if cleaned_path.exists():
            df = pd.read_csv(cleaned_path)
            logger.info(f"Loaded cleaned data from {cleaned_path} for normalization.")
        else:
            df = await self._load_available_data_from_db(cleaned_only=True) # Пример: флаг для загрузки очищенных
            if df.empty:
                 logger.warning("No cleaned data available from database for normalization. Trying all data.")
                 df = await self._load_available_data_from_db()
                 if df.empty:
                    logger.error("No data available for normalization from any source.")
                    return { "status": "no_data", "message": "No data in DB to normalize."}
        
        if df.empty:
            raise Exception("No data available for normalization")
        
        original_df_columns = list(df.columns)
        
        # Логика нормализации остается схожей, но данные из БД
        if "ra" in df.columns:
            df["ra_normalized"] = df["ra"] / 360.0
        
        if "dec" in df.columns:
            df["dec_normalized"] = (df["dec"] + 90.0) / 180.0
        
        if "redshift" in df.columns and pd.api.types.is_numeric_dtype(df["redshift"]):
            # Log normalization for redshift (better for ML)
            df["redshift_log"] = np.log10(df["redshift"].astype(float) + 0.001) # Убедимся что float
            min_z, max_z = df["redshift"].min(), df["redshift"].max()
            if max_z > min_z:
                df["redshift_normalized"] = (df["redshift"] - min_z) / (max_z - min_z)
            else:
                df["redshift_normalized"] = 0.5 # Если все значения одинаковы
        
        magnitude_cols = [col for col in df.columns if ("mag_" in col.lower() or "magnitude" in col.lower()) and not any(norm_suffix in col for norm_suffix in ['_zscore', '_minmax'])]
        for mag_col in magnitude_cols:
            if mag_col in df.columns and pd.api.types.is_numeric_dtype(df[mag_col]):
                # Z-score normalization
                mean_mag, std_mag = df[mag_col].mean(), df[mag_col].std()
                if std_mag > 0:
                    df[f"{mag_col}_zscore"] = (df[mag_col] - mean_mag) / std_mag
                else:
                    df[f"{mag_col}_zscore"] = 0
                
                # Min-max normalization
                min_mag, max_mag = df[mag_col].min(), df[mag_col].max()
                if max_mag > min_mag:
                    df[f"{mag_col}_minmax"] = (df[mag_col] - min_mag) / (max_mag - min_mag)
                else:
                    df[f"{mag_col}_minmax"] = 0.5
        
        # Нормализация 3D координат (X, Y, Z)
        coord_3d = ["X", "Y", "Z"]
        for coord_col_name in coord_3d:
            if coord_col_name in df.columns and pd.api.types.is_numeric_dtype(df[coord_col_name]):
                mean_coord = df[coord_col_name].mean()
                df[f"{coord_col_name}_centered"] = df[coord_col_name] - mean_coord
                abs_max = df[f"{coord_col_name}_centered"].abs().max()
                if abs_max > 0:
                    df[f"{coord_col_name}_normalized"] = df[f"{coord_col_name}_centered"] / abs_max
                else:
                    df[f"{coord_col_name}_normalized"] = 0
                
        if all(c in df.columns for c in ["X", "Y", "Z"]):
            df["distance_3d"] = np.sqrt(df["X"]**2 + df["Y"]**2 + df["Z"]**2)
            df["distance_3d_log"] = np.log10(df["distance_3d"] + 1) # +1 для избежания log(0)

        if "ra" in df.columns and "dec" in df.columns:
            ra_rad = np.radians(df["ra"].astype(float))
            dec_rad = np.radians(df["dec"].astype(float))
            df["ra_sin"] = np.sin(ra_rad)
            df["ra_cos"] = np.cos(ra_rad)
            df["dec_sin"] = np.sin(dec_rad)
            df["dec_cos"] = np.cos(dec_rad)
        
        output_filename = PROCESSING_OUTPUT_DIR / "normalized_data.csv"
        df.to_csv(output_filename, index=False)
        logger.info(f"Normalized data saved to {output_filename} for review. Columns: {len(df.columns)}")
        
        # Логика обновления БД (аналогично _clean_data)
        logger.warning("Database update logic for normalized data is not fully implemented.")

        normalization_report = {
            "original_columns_count": len(original_df_columns),
            "normalized_columns_count": len(df.columns),
            "added_features_count": len(df.columns) - len(original_df_columns),
            "normalization_methods_used": ["z-score", "min-max", "log", "trigonometric", "centering"],
            "output_file_preview": str(output_filename)
        }
        
        logger.info(f"Data normalization complete: {len(df.columns)} total columns")
        
        return {
            "status": "success",
            "processing_type": "normalize",
            "report": normalization_report,
            "normalized_dataframe_preview_path": str(output_filename)
        }
    
    async def _engineer_features(self, config: Dict) -> Dict[str, Any]:
        """Engineer advanced features. Теперь асинхронная."""
        logger.info("Engineering advanced features...")

        normalized_path = PROCESSING_OUTPUT_DIR / "normalized_data.csv"
        cleaned_path = PROCESSING_OUTPUT_DIR / "cleaned_data.csv"
        
        if normalized_path.exists():
            df = pd.read_csv(normalized_path)
            logger.info(f"Loaded normalized data from {normalized_path} for feature engineering.")
        elif cleaned_path.exists():
            df = pd.read_csv(cleaned_path)
            logger.info(f"Loaded cleaned data from {cleaned_path} for feature engineering.")
        else:
            df = await self._load_available_data_from_db(normalized_only=True) # Загрузка нормализованных из БД
            if df.empty:
                df = await self._load_available_data_from_db(cleaned_only=True) # или очищенных
                if df.empty:
                    df = await self._load_available_data_from_db() # или всех
                    if df.empty:
                        logger.error("No data available for feature engineering from any source.")
                        return { "status": "no_data", "message": "No data in DB to engineer features."}

        if df.empty:
            raise Exception("No data available for feature engineering")
        
        original_columns_count = len(df.columns)
        
        # Логика генерации признаков (остается схожей)
        magnitude_cols = [col for col in df.columns if ("mag_" in col.lower() or "magnitude" in col.lower()) and not any(norm_suffix in col for norm_suffix in ['_zscore', '_minmax', '_normalized', '_log'])]
        if len(magnitude_cols) >= 2:
            for i, mag1_col_name in enumerate(magnitude_cols):
                for mag2_col_name in magnitude_cols[i+1:]:
                    # Убедимся, что колонки числовые перед вычитанием
                    if pd.api.types.is_numeric_dtype(df[mag1_col_name]) and pd.api.types.is_numeric_dtype(df[mag2_col_name]):
                        color_name = f"color_{mag1_col_name.split('_')[-1]}_{mag2_col_name.split('_')[-1]}"
                        df[color_name] = df[mag1_col_name] - df[mag2_col_name]
        
        if "redshift" in df.columns and pd.api.types.is_numeric_dtype(df["redshift"]):
            df["redshift_bin_coarse"] = pd.cut(df["redshift"], 
                                             bins=[0, 0.1, 0.5, 1.0, 2.0, 10.0],
                                             labels=["nearby", "intermediate", "distant", "high_z", "extreme"],
                                             right=False) # Добавляем right=False для корректных интервалов
            
            df["redshift_bin_fine"] = pd.cut(df["redshift"], bins=20, labels=False, right=False)
            
            c_light_kms = 299792.458  
            df["velocity_recession_kms"] = df["redshift"] * c_light_kms
        
        if all(col in df.columns for col in ["ra", "dec"]):
            if pd.api.types.is_numeric_dtype(df["dec"]):
                 df["galactic_latitude_approx"] = 90 - np.abs(df["dec"]) 
            
            if pd.api.types.is_numeric_dtype(df["ra"]) and pd.api.types.is_numeric_dtype(df["dec"]):
                df["sky_region_ra_bin"] = pd.cut(df["ra"], bins=36, labels=False, right=False)
                df["sky_region_dec_bin"] = pd.cut(df["dec"], bins=18, labels=False, right=False)
                df["sky_region_id"] = df["sky_region_ra_bin"] * 18 + df["sky_region_dec_bin"]

        if "distance_3d" in df.columns or all(col in df.columns for col in ["X", "Y", "Z"]):
            if "distance_3d" not in df.columns:
                df["distance_3d"] = np.sqrt(df["X"]**2 + df["Y"]**2 + df["Z"]**2)
            
            if pd.api.types.is_numeric_dtype(df["distance_3d"]):
                df["volume_shell_bin"] = pd.cut(df["distance_3d"], bins=20, labels=False, right=False)
            
            if "redshift" in df.columns and pd.api.types.is_numeric_dtype(df["redshift"]):
                # Angular diameter distance (simplified, for Euclid-like data)
                # Ensure redshift is not -1 to avoid division by zero if (1+z) is part of a formula
                # df["angular_diameter_distance_mpc"] = df["distance_3d"] / (1 + df["redshift"].clip(lower=0)) # clip to avoid z < -1
                # Более простой вариант, если distance_3d уже космологически корректно:
                df["angular_diameter_distance_mpc"] = df["distance_3d"]

        # Статистики по группам (если есть 'catalog_source')
        if "catalog_source" in df.columns:
            for col_stat in ["redshift", "ra", "dec"] + magnitude_cols:
                if col_stat in df.columns and pd.api.types.is_numeric_dtype(df[col_stat]):
                    try:
                        source_stats = df.groupby("catalog_source")[col_stat].agg(["mean", "std", "count"])
                        source_stats.columns = [f"{col_stat}_src_{stat}" for stat in ["mean", "std", "count"]]
                        df = df.merge(source_stats, left_on="catalog_source", right_index=True, how="left")
                    except Exception as e_stat:
                        logger.warning(f"Could not compute source statistics for {col_stat}: {e_stat}")
        
        # Interaction features (пример)
        if "redshift" in df.columns and magnitude_cols and pd.api.types.is_numeric_dtype(df["redshift"]):
            for mag_col in magnitude_cols[:2]: # Ограничимся первыми двумя для примера
                if mag_col in df.columns and pd.api.types.is_numeric_dtype(df[mag_col]):
                    df[f"{mag_col}_div_redshift"] = df[mag_col] / (df["redshift"].astype(float) + 0.001)
                    df[f"{mag_col}_mul_redshift"] = df[mag_col] * df["redshift"].astype(float)

        # Grid-based density (упрощенно)
        if all(col in df.columns for col in ["X", "Y", "Z"]):
            if all(pd.api.types.is_numeric_dtype(df[c]) for c in ["X", "Y", "Z"]):
                try:
                    x_bins = pd.cut(df["X"], bins=10, labels=False, right=False)
                    y_bins = pd.cut(df["Y"], bins=10, labels=False, right=False)
                    z_bins = pd.cut(df["Z"], bins=10, labels=False, right=False)
                    df["grid_cell_id"] = x_bins * 100 + y_bins * 10 + z_bins
                    grid_counts = df["grid_cell_id"].value_counts()
                    df["local_density_grid_approx"] = df["grid_cell_id"].map(grid_counts)
                except Exception as e_grid:
                    logger.warning(f"Could not compute grid-based density: {e_grid}")
        
        temp_cols_to_drop = [col for col in df.columns if col.endswith("_bin")]
        df = df.drop(columns=temp_cols_to_drop, errors="ignore")
        
        output_filename = PROCESSING_OUTPUT_DIR / "feature_engineered_data.csv"
        df.to_csv(output_filename, index=False)
        logger.info(f"Feature engineered data saved to {output_filename} for review. Final columns: {len(df.columns)}")
        
        # Логика обновления БД
        logger.warning("Database update logic for feature engineered data is not fully implemented.")

        feature_report = {
            "original_columns_count": original_columns_count,
            "final_columns_count": len(df.columns),
            "engineered_features_count": len(df.columns) - original_columns_count,
            "feature_categories_counts": {
                "color_indices": len([col for col in df.columns if col.startswith("color_")]),
                "redshift_features": len([col for col in df.columns if "redshift_" in col]),
                "distance_features": len([col for col in df.columns if "distance_" in col or "volume_" in col]),
                "angular_features": len([col for col in df.columns if "_sin" in col or "_cos" in col or "sky_region_" in col]),
                "source_statistical_features": len([col for col in df.columns if "_src_" in col]),
                "interaction_features": len([col for col in df.columns if "_div_" in col or "_mul_" in col]),
                "density_features": len([col for col in df.columns if "density_" in col])
            },
            "output_file_preview": str(output_filename)
        }
        
        logger.info(f"Feature engineering complete: {feature_report['engineered_features_count']} new features created.")
        
        return {
            "status": "success",
            "processing_type": "feature_engineering",
            "report": feature_report,
            "engineered_dataframe_preview_path": str(output_filename)
        }
    
    async def _load_available_data_from_db(self, cleaned_only=False, normalized_only=False, limit=100000) -> pd.DataFrame:
        """Load astronomical data from the database (astronomical_objects table)."""
        logger.info(f"Loading data from database... Cleaned only: {cleaned_only}, Normalized only: {normalized_only}, Limit: {limit}")
        
        # TODO: Добавить фильтры для cleaned_only и normalized_only, если в БД будут соответствующие флаги или таблицы
        # Примерный запрос, который нужно будет адаптировать:
        # query = "SELECT * FROM astronomical_objects" 
        # params = []
        # if cleaned_only:
        #     query += " WHERE is_cleaned = TRUE" # Предполагая наличие такого поля
        # if limit > 0:
        #     query += f" LIMIT ?"
        #     params.append(limit)

        try:
            # Загружаем все объекты, фильтрацию по cleaned/normalized пока опускаем для простоты
            # В реальном приложении здесь будет более сложный запрос или загрузка из разных таблиц/представлений
            objects_data = await db.get_astronomical_objects(limit=limit) 
            if objects_data:
                df = pd.DataFrame(objects_data)
                # Преобразование типов данных, если необходимо (например, строки в числа)
                for col in ['ra', 'dec', 'redshift', 'magnitude', 'X', 'Y', 'Z']:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                logger.info(f"Loaded {len(df)} objects from database.")
                return df
            else:
                logger.warning("No objects found in the database.")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error loading data from database: {e}", exc_info=True)
            return pd.DataFrame()

    # Этот метод может быть устаревшим, если данные всегда берутся из БД
    # def _load_available_data_from_files(self) -> pd.DataFrame:
    #     """Load any available astronomical data from local CSV files (fallback)."""
    #     file_patterns = [
    #         "merged_real_galaxies.csv",
    #         "sdss_real.csv",
    #         "desi_real.csv", 
    #         "des_real.csv",
    #         "euclid_real.csv",
    #         "merged_galaxies.csv" 
    #     ]
        
    #     for pattern in file_patterns:
    #         file_path = PROCESSING_OUTPUT_DIR / pattern
    #         if file_path.exists():
    #             logger.info(f"Loading data from file: {file_path}")
    #             try:
    #                 return pd.read_csv(file_path)
    #             except Exception as e:
    #                 logger.error(f"Error reading file {file_path}: {e}")
        
    #     logger.warning("No astronomical data files found in processed directory for fallback loading.")
    #     return pd.DataFrame()

# Main function for background tasks (может быть вынесена или использоваться как точка входа для Celery/RQ)
async def process_astronomical_data_entrypoint(config: Dict) -> Dict[str, Any]: # Переименована во избежание конфликта
    """Main function for processing astronomical data (background task). Асинхронная."""
    try:
        processor = DataProcessor()
        result = await processor.process_astronomical_data(config) # await
        
        return {
            "status": "success",
            "result": result,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error processing astronomical data: {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        } 