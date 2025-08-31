"""
Data preprocessor for astronomical catalogs.
Automatically downloads, cleans and normalizes astronomical datasets.
Integrates with the database for storing processed data.
"""

import os
import pandas as pd
import numpy as np
import logging
import asyncio
import ssl
from typing import Dict, List, Optional, Any
from pathlib import Path
import json
import sys
from datetime import datetime

# Database integration
from database.config import db # Импортируем наш глобальный объект db

# HTTP client
try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    aiohttp = None
    AIOHTTP_AVAILABLE = False

# FITS file handling
try:
    from astropy.io import fits
    from astropy.table import Table
    ASTROPY_AVAILABLE = True
except ImportError:
    fits = None
    Table = None
    ASTROPY_AVAILABLE = False

# HDF5 file handling (если потребуется)
# try:
#     import h5py
#     H5PY_AVAILABLE = True
# except ImportError:
#     h5py = None
#     H5PY_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AstronomicalDataPreprocessor:
    """Preprocessor for astronomical catalogs from SDSS, DESI, DES, and Euclid."""
    
    def __init__(self, data_dir: str = "galaxy_data"):
        self.data_dir = Path(data_dir)
        self.cache_dir = self.data_dir / "cache"      # Для скачанных сырых файлов
        # self.processed_dir = self.data_dir / "processed" # Эта директория может быть не нужна, если все идет в БД
        
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        # self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        self.catalogs = {
            "SDSS": {
                "url": "https://dr18.sdss.org/sas/dr18/spectro/boss/redux/v6_0_4/spAll-v6_0_4.fits",
                "columns": {
                    "ra": ["PLUG_RA", "RA", "RAJ2000", "ALPHA_J2000"],
                    "dec": ["PLUG_DEC", "DEC", "DECJ2000", "DEJ2000", "DELTA_J2000"],
                    "z": ["Z", "REDSHIFT", "PHOTOZ", "Z_MEAN", "Z_SPEC", "ZPHOT"],
                    # Добавим object_id, если он есть в FITS, или будем генерировать
                    "object_id": ["SPECOBJID", "PLATEID_FIBERID", "THING_ID", "SDSS_ID"], 
                    "mag_g": ["MAG", "PSFMAG_G", "FIBERMAG_G", "MAG_G"],
                    "mag_r": ["MAG", "PSFMAG_R", "FIBERMAG_R", "MAG_R"],
                    "mag_i": ["MAG", "PSFMAG_I", "FIBERMAG_I", "MAG_I"]
                },
                # "processed_name": "sdss_processed.csv", # Не нужно, если пишем в БД
                "sample_size": 50000,
                "db_catalog_source_name": "SDSS" # Имя для поля catalog_source в БД
            },
            "DESI": {
                "url": "https://data.desi.lbl.gov/public/edr/spectro/redux/fuji/zcatalog/zall-pix-fuji.fits",
                "columns": {
                    "ra": ["TARGET_RA", "RA", "RAJ2000"],
                    "dec": ["TARGET_DEC", "DEC", "DECJ2000"],
                    "z": ["Z", "REDSHIFT", "Z_SPEC"],
                    "object_id": ["TARGETID", "DESI_TARGETID"],
                    "mag_g": ["FLUX_G", "MAG_G", "PSFMAG_G"],
                    "mag_r": ["FLUX_R", "MAG_R", "PSFMAG_R"],
                    "mag_z": ["FLUX_Z", "MAG_Z", "PSFMAG_Z"]
                },
                "sample_size": 30000,
                "db_catalog_source_name": "DESI"
            },
            "DES": {
                # Пример URL для DES (может быть Parquet или другой формат)
                "url": "https://desdr-server.ncsa.illinois.edu/despublic/y6a2_files/y6_gold/Y6_GOLD_2_2-519-0000.parquet",
                "file_type": "parquet", # Указываем тип файла, если не FITS
                "hdf5_key": "df", # Ключ датасета в HDF5 файле
                "columns": {
                    "ra": ["ALPHAWIN_J2000", "RA", "RAJ2000", "RA_DEG"],
                    "dec": ["DELTAWIN_J2000", "DEC", "DECJ2000", "DEC_DEG"],
                    "z": ["DNF_ZMEAN_SOF", "Z_PHOT", "PHOTOZ", "Z_MEAN", "REDSHIFT", "Z"],
                    "object_id": ["COADD_OBJECT_ID", "ID"],
                    "mag_g": ["WAVG_MAG_PSF_G", "MAG_AUTO_G", "MAG_G", "G Kron Mag"],
                    "mag_r": ["WAVG_MAG_PSF_R", "MAG_AUTO_R", "MAG_R", "R Kron Mag"],
                    "mag_i": ["WAVG_MAG_PSF_I", "MAG_AUTO_I", "MAG_I", "I Kron Mag"]
                },
                "sample_size": 40000,
                "db_catalog_source_name": "DES"
            },
            "Euclid": {
                "url": "https://irsa.ipac.caltech.edu/ibe/data/euclid/q1/catalogs/MER_FINAL_CATALOG/102018211/EUC_MER_FINAL-CAT_TILE102018211-CC66F6_20241018T214045.289017Z_00.00.fits",
                "columns": {
                    "ra": ["RIGHT_ASCENSION", "RA", "RAJ2000", "ALPHA_J2000"],
                    "dec": ["DECLINATION", "DEC", "DECJ2000", "DELTA_J2000"],
                    "z": ["Z_PHOT", "PHOTOZ", "Z_B", "REDSHIFT"],
                    "object_id": ["OBJECT_ID", "ID", "SOURCE_ID"],
                    "mag_vis": ["FLUX_VIS_1FWHM_APER", "MAG_VIS", "VIS_MAG"],
                    "mag_y": ["FLUX_Y_1FWHM_APER", "MAG_Y", "Y_MAG"],
                    "mag_j": ["FLUX_J_1FWHM_APER", "MAG_J", "J_MAG"],
                    "mag_h": ["FLUX_H_1FWHM_APER", "MAG_H", "H_MAG"]
                },
                "sample_size": 30000,
                "db_catalog_source_name": "Euclid"
            }
        }
    
    def _find_column_name(self, available_columns: List[str], possible_names: List[str]) -> Optional[str]:
        """Find the first matching column name from a list of possibilities (case-insensitive)."""
        available_upper = [str(col).upper() for col in available_columns] # Убедимся, что все строки
        for name in possible_names:
            if str(name).upper() in available_upper:
                idx = available_upper.index(str(name).upper())
                return available_columns[idx]
        return None
    
    async def download_catalog(self, catalog_name_key: str, catalog_info: Dict) -> Optional[Path]:
        """Download a catalog file to cache if not already present."""
        url = catalog_info.get("url")
        if not url:
            logger.warning(f"No URL provided for {catalog_name_key}")
            return None
            
        file_type = catalog_info.get("file_type", "fits") # По умолчанию FITS
        filename = f"{catalog_name_key.lower()}_raw.{file_type}"
        file_path = self.cache_dir / filename
        
        if file_path.exists() and file_path.stat().st_size > 0: # Проверка на размер файла > 0
            logger.info(f"Using cached {catalog_name_key} data: {file_path}")
            return file_path
        
        logger.info(f"Downloading {catalog_name_key} from {url} to {file_path}")
        
        if not AIOHTTP_AVAILABLE:
            logger.error("aiohttp library is not available. Cannot download files.")
            return None

        try:
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            
            # Увеличиваем таймаут
            timeout = aiohttp.ClientTimeout(total=600) # 10 минут общий таймаут
            connector = aiohttp.TCPConnector(ssl=ssl_context, limit_per_host=5) # Ограничиваем кол-во одновременных соединений
            
            async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
                async with session.get(url, allow_redirects=True) as response:
                    if response.status == 200:
                        with open(file_path, 'wb') as f:
                            while True:
                                chunk = await response.content.read(8192) # Читаем чанками
                                if not chunk:
                                    break
                                f.write(chunk)
                        logger.info(f"Downloaded {catalog_name_key} to {file_path}")
                        return file_path
                    else:
                        logger.error(f"Failed to download {catalog_name_key}: HTTP {response.status} - {await response.text()}")
                        if file_path.exists(): file_path.unlink() # Удаляем неполный файл
                        return None
        except asyncio.TimeoutError:
             logger.error(f"Timeout error downloading {catalog_name_key} from {url}")
             if file_path.exists(): file_path.unlink()
             return None
        except Exception as e:
            logger.error(f"Error downloading {catalog_name_key} from {url}: {e}", exc_info=True)
            if file_path.exists() and file_path.is_file(): # Проверка что это файл перед удалением
                 file_path.unlink()
            return None

    def _read_data_from_file(self, file_path: Path, catalog_info: Dict) -> Optional[pd.DataFrame]:
        """Reads data from various file formats into a pandas DataFrame."""
        file_type = catalog_info.get("file_type", "fits")
        logger.info(f"Reading {file_type} file: {file_path}")

        if file_type == "fits":
            if not ASTROPY_AVAILABLE:
                logger.error("astropy is required to read FITS files but not available.")
                return None
            try:
                with fits.open(file_path, memmap=True) as hdul:
                    # Пытаемся найти HDU с данными таблицы (обычно 1-й или 2-й)
                    data_hdu = None
                    for i, hdu in enumerate(hdul):
                        if hasattr(hdu, 'data') and hdu.data is not None and isinstance(hdu.data, (np.ndarray, np.void)) and hdu.is_image == False:
                            # Проверяем, есть ли у данных атрибут 'shape' (для табличных данных)
                            if hasattr(hdu.data, 'shape') and len(hdu.data.shape) > 0 and hdu.data.shape[0] > 0:
                                logger.info(f"Found data in HDU {i} with {hdu.data.shape[0]} rows.")
                                data_hdu = hdu
                                break
                    if data_hdu is None:
                        logger.error(f"No suitable data HDU found in FITS file: {file_path}")
                        return None
                    
                    table = Table(data_hdu.data)
                    # Фильтруем многомерные колонки, как раньше
                    simple_columns = [col_name for col_name in table.colnames if len(table[col_name].shape) <= 1]
                    df = table[simple_columns].to_pandas()
                    logger.info(f"Successfully read FITS file {file_path}, {len(df)} rows.")
                    return df
            except Exception as e:
                logger.error(f"Error reading FITS file {file_path}: {e}", exc_info=True)
                return None
        
        elif file_type == "hdf5":
            # if not H5PY_AVAILABLE: # Если h5py будет использоваться
            #     logger.error("h5py is required to read HDF5 files but not available.")
            #     return None
            try:
                key = catalog_info.get("hdf5_key", "data") # Ключ по умолчанию или из конфига
                df = pd.read_hdf(file_path, key=key)
                logger.info(f"Successfully read HDF5 file {file_path}, key '{key}', {len(df)} rows.")
                return df
            except Exception as e:
                logger.error(f"Error reading HDF5 file {file_path} with key '{key}': {e}", exc_info=True)
                return None
        
        elif file_type == "parquet":
            try:
                df = pd.read_parquet(file_path)
                logger.info(f"Successfully read Parquet file {file_path}, {len(df)} rows.")
                return df
            except Exception as e:
                logger.error(f"Error reading Parquet file {file_path}: {e}", exc_info=True)
                return None
        
        elif file_type == "csv":
            try:
                df = pd.read_csv(file_path)
                logger.info(f"Successfully read CSV file {file_path}, {len(df)} rows.")
                return df
            except Exception as e:
                logger.error(f"Error reading CSV file {file_path}: {e}", exc_info=True)
                return None
        else:
            logger.error(f"Unsupported file type '{file_type}' for {file_path}")
            return None

    def _normalize_columns(self, df: pd.DataFrame, catalog_name_key: str, catalog_info: Dict) -> Optional[pd.DataFrame]:
        """Normalize column names using the mapping and select only mapped columns."""
        available_columns = list(df.columns)
        logger.debug(f"Available columns in {catalog_name_key} before normalization: {available_columns}")
        
        column_mapping = {}
        # Стандартные колонки, которые мы ожидаем в базе данных (из schema.sql)
        # object_id, name, object_type, ra, dec, redshift, magnitude, catalog_source, data_release
        # Плюс X, Y, Z для координат
        # Магнитуды могут быть разными: mag_g, mag_r, и т.д.
        standard_db_columns = {
            'object_id': catalog_info["columns"].get('object_id', ['OBJECT_ID', 'ID']), # Обязательно
            'ra': catalog_info["columns"].get('ra', ['RA']),
            'dec': catalog_info["columns"].get('dec', ['DEC']),
            'redshift': catalog_info["columns"].get('z', ['Z', 'REDSHIFT']), # 'z' более общий для разных каталогов
            # Магнитуды обрабатываются отдельно, т.к. их много и они специфичны
        }

        # Добавляем специфичные для каталога колонки магнитуд
        for standard_mag_name, possible_mag_names in catalog_info["columns"].items():
            if standard_mag_name.startswith("mag_"):
                standard_db_columns[standard_mag_name] = possible_mag_names
        
        final_df_columns = {}
        missing_vital_columns = False

        for target_col, possible_source_cols in standard_db_columns.items():
            found_col = self._find_column_name(available_columns, possible_source_cols)
            if found_col:
                final_df_columns[target_col] = df[found_col]
                logger.debug(f"Mapped source column '{found_col}' to target '{target_col}' for {catalog_name_key}")
            elif target_col in ['object_id', 'ra', 'dec']: # Если обязательные колонки не найдены
                logger.error(f"Vital column '{target_col}' (tried: {possible_source_cols}) not found in {catalog_name_key}. Cannot proceed.")
                missing_vital_columns = True
            else:
                logger.warning(f"Optional column '{target_col}' (tried: {possible_source_cols}) not found in {catalog_name_key}. Will be filled with NaN.")
                final_df_columns[target_col] = pd.Series([np.nan] * len(df)) # Заполняем NaN если не найдено
        
        if missing_vital_columns:
            return None
        
        df_normalized = pd.DataFrame(final_df_columns)
        
        # Добавляем catalog_source
        df_normalized['catalog_source'] = catalog_info.get("db_catalog_source_name", catalog_name_key.upper())
        
        # Генерация object_id если он все еще отсутствует (например, не был в source_columns)
        if 'object_id' not in df_normalized.columns or df_normalized['object_id'].isnull().all():
            logger.warning(f"Column 'object_id' is missing or all null after mapping for {catalog_name_key}. Generating new IDs.")
            # Создаем ID на основе catalog_source и индекса строки
            df_normalized['object_id'] = df_normalized['catalog_source'] + "_" + df_normalized.index.astype(str)
        
        logger.info(f"Normalized columns for {catalog_name_key}: {list(df_normalized.columns)}")
        return df_normalized

    def _clean_dataframe_for_db(self, df: pd.DataFrame) -> pd.DataFrame:
        """Basic cleaning: ensure ra, dec, redshift are numeric and within reasonable bounds."""
        logger.debug(f"Initial rows before cleaning: {len(df)}")
        
        # Преобразование в числовые типы, ошибки -> NaN
        for col in ['ra', 'dec', 'redshift'] + [c for c in df.columns if c.startswith("mag_")]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Удаление строк с NaN в критичных колонках (ra, dec, object_id)
        df.dropna(subset=['ra', 'dec', 'object_id'], inplace=True)
        logger.debug(f"Rows after dropping NaN in ra/dec/object_id: {len(df)}")

        # Фильтрация по диапазонам
        if 'ra' in df.columns: df = df[(df['ra'] >= 0) & (df['ra'] <= 360)]
        if 'dec' in df.columns: df = df[(df['dec'] >= -90) & (df['dec'] <= 90)]
        if 'redshift' in df.columns: df = df[(df['redshift'] >= 0) & (df['redshift'] <= 10)] # Примерный диапазон

        logger.debug(f"Rows after range filtering: {len(df)}")
        return df
    
    def _add_cartesian_coordinates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Cartesian X, Y, Z coordinates if ra, dec, redshift are present."""
        if all(col in df.columns for col in ['ra', 'dec', 'redshift']):
            if df['redshift'].isnull().all() or df['ra'].isnull().all() or df['dec'].isnull().all():
                logger.warning("Cannot compute Cartesian coordinates due to missing ra/dec/redshift values.")
                df['X'] = np.nan
                df['Y'] = np.nan
                df['Z'] = np.nan
                return df

            # Упрощенный расчет (без космологической модели, просто для примера)
            # В реальном приложении использовать astropy.coordinates или подобное
            # Используем redshift как прокси для расстояния (очень грубо)
            # Для более точного расчета нужны космологические параметры
            distance = df['redshift'] * 1000 # Примерная шкала, не физическая!
            
            ra_rad = np.radians(df['ra'])
            dec_rad = np.radians(df['dec'])
            
            df['X'] = distance * np.cos(dec_rad) * np.cos(ra_rad)
            df['Y'] = distance * np.cos(dec_rad) * np.sin(ra_rad)
            df['Z'] = distance * np.sin(dec_rad)
            logger.info("Added simplified Cartesian X, Y, Z coordinates.")
        else:
            logger.info("Skipping Cartesian coordinates (ra, dec, or redshift missing).")
            df['X'] = np.nan
            df['Y'] = np.nan
            df['Z'] = np.nan
        return df

    async def _store_dataframe_to_db(self, df: pd.DataFrame, catalog_name_key: str):
        """Store DataFrame rows into the astronomical_objects table in the database."""
        if df.empty:
            logger.info(f"DataFrame for {catalog_name_key} is empty. Nothing to store.")
            return 0

        logger.info(f"Storing {len(df)} objects from {catalog_name_key} into database...")
        
        # Преобразуем DataFrame в список словарей для вставки
        # Убедимся, что NaN заменены на None для совместимости с JSON/DB
        df_to_store = df.replace({np.nan: None})
        records = df_to_store.to_dict('records')
        
        inserted_count = 0
        updated_count = 0 # Для CosmosDB upsert это не так просто отследить без доп. логики

        for record in records:
            # Убедимся, что все поля, которых нет в schema.sql, удалены или обработаны
            # Это важно для SQL баз. Для CosmosDB можно хранить доп. поля.
            # Для SQL, нужно привести к полям таблицы astronomical_objects:
            # id, object_id, name, object_type, ra, dec, redshift, magnitude (общее поле?), 
            # catalog_source, data_release, created_at, updated_at, X, Y, Z
            
            db_record = {}
            db_record['object_id'] = record.get('object_id')
            db_record['catalog_source'] = record.get('catalog_source', catalog_name_key.upper())
            
            # Основные астрономические поля
            for col in ['ra', 'dec', 'redshift', 'X', 'Y', 'Z']:
                 db_record[col] = record.get(col)
            
            # Обработка магнитуд: можно сохранить их как JSON в поле 'magnitudes_json' 
            # или выбрать одну 'основную' магнитуду для поля 'magnitude'
            magnitudes_data = {k: v for k, v in record.items() if k.startswith('mag_') and v is not None}
            if magnitudes_data:
                db_record['magnitudes_json'] = json.dumps(magnitudes_data)
                # Выберем первую доступную магнитуду для основного поля 'magnitude' (если оно есть в схеме)
                # Это упрощение. В идеале, решить, как отображать разные фильтры.
                first_mag_value = next(iter(magnitudes_data.values()), None)
                if 'magnitude' in db.cosmos_containers_config.get('astronomical_objects', {}): # Проверка на существование поля
                    db_record['magnitude'] = first_mag_value 
                elif 'mag_g' in magnitudes_data : # fallback
                     db_record['magnitude'] = magnitudes_data['mag_g']

            # Дополнительные поля (name, object_type, data_release - если есть в record)
            db_record['name'] = record.get('name') # Если есть колонка name
            db_record['object_type'] = record.get('object_type') # Если есть
            db_record['data_release'] = record.get('data_release') # Если есть
            
            # Временные метки
            current_ts_iso = datetime.utcnow().isoformat() + "Z"
            db_record['created_at'] = record.get('created_at', current_ts_iso)
            db_record['updated_at'] = current_ts_iso

            # Удаляем ключи со значениями None, чтобы они не перезаписывали существующие данные в CosmosDB при частичном обновлении
            # и чтобы не было проблем с SQL NOT NULL ограничениями (если они есть и не обрабатываются ON CONFLICT)
            # db_record_cleaned = {k: v for k, v in db_record.items() if v is not None} # Осторожно с этим, может удалить нужные None
            db_record_cleaned = db_record # Пока оставляем все поля

            if not db_record_cleaned.get('object_id'):
                logger.warning(f"Skipping record due to missing object_id: {db_record_cleaned}")
                continue

            if db.db_type == "cosmosdb":
                # Для CosmosDB, object_id может быть использован как 'id' документа
                db_record_cleaned['id'] = str(db_record_cleaned['object_id']) # Убедимся, что ID - строка
                # Ключ партиции для astronomical_objects - catalog_source
                # Он уже должен быть в db_record_cleaned['catalog_source']
                
                # Проверяем, что ключ партиции существует
                if not db_record_cleaned.get('catalog_source'):
                    logger.error(f"Missing partition key 'catalog_source' for object_id {db_record_cleaned['id']}. Skipping upsert.")
                    continue

                response = await db.upsert_item("astronomical_objects", db_record_cleaned)
                if response: # upsert_item возвращает сам документ в случае успеха
                    inserted_count +=1 # В CosmosDB upsert может быть и insert и update
            else: # SQL (SQLite, PostgreSQL)
                # Для SQL, нужно сформировать INSERT ... ON CONFLICT DO UPDATE ... запрос
                # Это более сложная логика, здесь пока упрощенный вариант - просто INSERT OR IGNORE
                # или отдельный SELECT + INSERT/UPDATE.
                # Пример для SQLite (допуская, что object_id уникален):
                cols = ", ".join(db_record_cleaned.keys())
                placeholders = ", ".join(["?"] * len(db_record_cleaned))
                sql_upsert = f"INSERT OR IGNORE INTO astronomical_objects ({cols}) VALUES ({placeholders})" 
                # Для PostgreSQL синтаксис будет другой: INSERT ... ON CONFLICT (object_id) DO UPDATE SET ...
                
                # Проверим, есть ли такой объект
                existing = await db.execute_query("SELECT id FROM astronomical_objects WHERE object_id = ?", (db_record_cleaned['object_id'],))
                if existing:
                    # Логика UPDATE для SQL
                    set_clause = ", ".join([f"{k} = ?" for k in db_record_cleaned.keys() if k != 'object_id'])
                    sql_update = f"UPDATE astronomical_objects SET {set_clause} WHERE object_id = ?"
                    update_params = [v for k, v in db_record_cleaned.items() if k != 'object_id'] + [db_record_cleaned['object_id']]
                    update_result = await db.execute_query(sql_update, tuple(update_params), is_update_or_delete=True)
                    if update_result and update_result[0].get('affected_rows', 0) > 0:
                         updated_count += 1
                else:
                    # Логика INSERT для SQL
                    insert_params = tuple(db_record_cleaned.values())
                    insert_result = await db.execute_query(sql_upsert, insert_params, is_update_or_delete=True)
                    if insert_result and insert_result[0].get('affected_rows', 0) > 0:
                        inserted_count += 1

        logger.info(f"Finished storing data for {catalog_name_key}. Inserted: {inserted_count}, Updated (SQL only): {updated_count}")
        return inserted_count + updated_count

    async def process_single_catalog(self, catalog_name_key: str, catalog_info: Dict) -> Dict[str, Any]:
        """Download, process, and store a single catalog into the database."""
        logger.info(f"Processing catalog: {catalog_name_key}...")
        
        # 1. Скачивание (если нужно)
        raw_file_path = await self.download_catalog(catalog_name_key, catalog_info)
        if not raw_file_path:
            logger.error(f"Failed to download raw data for {catalog_name_key}. Skipping.")
            return {"status": "download_failed", "catalog": catalog_name_key, "objects_processed": 0}

        # 2. Чтение данных из файла
        df_raw = self._read_data_from_file(raw_file_path, catalog_info)
        if df_raw is None or df_raw.empty:
            logger.error(f"Failed to read data from file {raw_file_path} for {catalog_name_key}. Skipping.")
            return {"status": "read_failed", "catalog": catalog_name_key, "objects_processed": 0}
        
        # 2.1 Сэмплирование, если данных слишком много
        sample_size = catalog_info.get("sample_size")
        if sample_size and len(df_raw) > sample_size:
            logger.info(f"Sampling {sample_size} objects from {len(df_raw)} for {catalog_name_key}.")
            df_raw = df_raw.sample(n=sample_size, random_state=42).reset_index(drop=True)

        # 3. Нормализация колонок (приведение к стандартным именам БД)
        df_normalized = self._normalize_columns(df_raw, catalog_name_key, catalog_info)
        if df_normalized is None or df_normalized.empty:
            logger.error(f"Failed to normalize columns for {catalog_name_key}. Skipping.")
            return {"status": "normalization_failed", "catalog": catalog_name_key, "objects_processed": 0}
        
        # 4. Базовая очистка данных (типы, диапазоны)
        df_cleaned = self._clean_dataframe_for_db(df_normalized.copy()) # copy() чтобы не менять df_normalized
        if df_cleaned.empty:
            logger.warning(f"No data left after cleaning for {catalog_name_key}. Skipping storage.")
            return {"status": "cleaning_failed_empty", "catalog": catalog_name_key, "objects_processed": 0}
            
        # 5. Добавление декартовых координат (если возможно)
        df_final_for_db = self._add_cartesian_coordinates(df_cleaned)

        # 6. Сохранение в базу данных
        try:
            objects_stored_count = await self._store_dataframe_to_db(df_final_for_db, catalog_name_key)
            logger.info(f"Successfully processed and stored {objects_stored_count} objects for {catalog_name_key}.")
            return {"status": "success", "catalog": catalog_name_key, "objects_processed": objects_stored_count}
        except Exception as e_store:
            logger.error(f"Failed to store data for {catalog_name_key} into database: {e_store}", exc_info=True)
            return {"status": "storage_failed", "catalog": catalog_name_key, "error": str(e_store), "objects_processed": 0}
    
    async def preprocess_all_catalogs(self) -> Dict[str, Any]:
        """Preprocess all configured catalogs and store them in the database."""
        logger.info("Starting preprocessing of all astronomical catalogs into database...")
        
        # Убедимся, что соединение с БД установлено
        if not ((db.db_type == 'cosmosdb' and db.cosmos_client) or 
                  (db.db_type != 'cosmosdb' and db.sql_connection)):
            await db.connect()
        # Также убедимся, что сама БД (схема/контейнеры) инициализирована
        await db.init_database()
        
        results_summary = {
            "status": "pending",
            "catalogs_summary": [],
            "total_objects_processed_all_catalogs": 0,
            "started_at": datetime.now().isoformat(),
            "completed_at": None
        }
        
        tasks = []
        for catalog_key, catalog_config in self.catalogs.items():
            tasks.append(self.process_single_catalog(catalog_key, catalog_config))
        
        # Запускаем обработку всех каталогов параллельно
        individual_catalog_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        total_processed_across_all = 0
        for result in individual_catalog_results:
            if isinstance(result, Exception):
                # Здесь нужно решить, как обрабатывать ошибку одного из каталогов
                # Можно добавить информацию об ошибке в results_summary
                logger.error(f"Error processing a catalog: {result}", exc_info=True)
                # Добавляем информацию об ошибке в сводку
                results_summary["catalogs_summary"].append({
                    "catalog": "Unknown (from exception)", 
                    "status": "exception_during_gather", 
                    "error": str(result), 
                    "objects_processed": 0
                })
            elif isinstance(result, dict): # Ожидаемый результат от process_single_catalog
                results_summary["catalogs_summary"].append(result)
                if result.get("status") == "success":
                    total_processed_across_all += result.get("objects_processed", 0)
            else:
                 logger.error(f"Unexpected result type from process_single_catalog: {type(result)} - {result}")
                 results_summary["catalogs_summary"].append({"catalog": "Unknown", "status": "unexpected_result_type", "objects_processed": 0})

        results_summary["total_objects_processed_all_catalogs"] = total_processed_across_all
        results_summary["completed_at"] = datetime.now().isoformat()
        results_summary["status"] = "completed_with_possible_errors" if any(cat.get("status") != "success" for cat in results_summary["catalogs_summary"]) else "completed_successfully"
        
        logger.info(f"Preprocessing of all catalogs completed. Summary: {json.dumps(results_summary, indent=2)}")
        
        # Опционально: создание объединенного датасета из БД (если нужно для каких-то целей)
        # await self.create_merged_dataset_from_db() 
        
        return results_summary

    # async def create_merged_dataset_from_db(self, limit_per_source=10000) -> Optional[Path]:
    #     """(Optional) Create a merged CSV dataset from all data in the database for analysis/export."""
    #     logger.info("Creating merged dataset from database...")
    #     all_objects = []
    #     for catalog_key in self.catalogs.keys():
    #         source_name = self.catalogs[catalog_key].get("db_catalog_source_name", catalog_key.upper())
    #         logger.info(f"Fetching objects for {source_name} from DB...")
    #         # Здесь нужно быть осторожным с limit. Если каталоги большие, это может занять много памяти.
    #         # db.get_astronomical_objects должен поддерживать фильтрацию по catalog_source
    #         objects = await db.get_astronomical_objects(limit=limit_per_source, catalog_source=source_name)
    #         if objects:
    #             all_objects.extend(objects)
    #             logger.info(f"Fetched {len(objects)} from {source_name}.")
        
    #     if not all_objects:
    #         logger.warning("No objects found in the database to create a merged dataset.")
    #         return None
        
    #     merged_df = pd.DataFrame(all_objects)
        
    #     # Удаление дубликатов по object_id (если он глобально уникален)
    #     # или по комбинации catalog_source + object_id_within_catalog
    #     if 'object_id' in merged_df.columns:
    #         initial_count = len(merged_df)
    #         merged_df.drop_duplicates(subset=['object_id'], keep='first', inplace=True)
    #         logger.info(f"Removed {initial_count - len(merged_df)} duplicates by object_id from merged DB data.")
        
    #     # Путь для сохранения объединенного файла (если нужно)
    #     # merged_output_dir = self.data_dir / "processed_from_db"
    #     # merged_output_dir.mkdir(exist_ok=True)
    #     # merged_file_path = merged_output_dir / "merged_catalog_from_db.csv"
        
    #     # df.to_csv(merged_file_path, index=False)
    #     # logger.info(f"Saved merged dataset from DB to {merged_file_path} ({len(merged_df)} objects)")
    #     # return merged_file_path
    #     logger.info(f"Merged dataset from DB created in memory with {len(merged_df)} objects. Not saving to file by default.")
    #     return merged_df # Возвращаем DataFrame

async def main_preprocess(): # Переименована во избежание конфликта с другими main
    """Main function to run the preprocessing for all catalogs."""
    logger.info("--- Starting Astronomical Data Preprocessing --- ")
    preprocessor = AstronomicalDataPreprocessor()
    
    # Инициализируем соединение с БД перед началом всех операций
    # Это можно сделать один раз здесь, а не в каждом методе preprocessor'а
    await db.connect() 
    await db.init_database() # Убедимся, что таблицы/контейнеры созданы
    
    results = await preprocessor.preprocess_all_catalogs()
    
    # Закрываем соединение с БД
    await db.disconnect()

    print("\n" + "="*50)
    print("PREPROCESSING RESULTS SUMMARY")
    print("="*50)
    print(f"Overall Status: {results.get('status')}")
    print(f"Total Objects Processed (all catalogs): {results.get('total_objects_processed_all_catalogs', 0):,}")
    print(f"Started At: {results.get('started_at')}")
    print(f"Completed At: {results.get('completed_at')}")
    
    print("\nIndividual Catalog Results:")
    for catalog_res in results.get("catalogs_summary", []):
        name = catalog_res.get('catalog', 'Unknown')
        status = catalog_res.get('status', 'error')
        processed_count = catalog_res.get('objects_processed', 0)
        if status == 'success':
            print(f"  {name}: ✅ {processed_count:,} objects processed successfully")
        else:
            error_msg = catalog_res.get('error', 'N/A')
            print(f"  {name}: ❌ {status} (Processed: {processed_count:,}) - Error: {error_msg}")
    
    # if results.get('merged_dataset_path'):
    #     print(f"\nMerged dataset (from DB) saved to: {results['merged_dataset_path']}")

if __name__ == "__main__":
    # Для запуска этого скрипта отдельно
    # Убедитесь, что asyncio event loop корректно управляется
    # (например, python -m asyncio utils.data_preprocessor если это модуль)
    # или просто `python utils/data_preprocessor.py` если запускается как скрипт
    if sys.version_info >= (3, 7):
        asyncio.run(main_preprocess())
    else: # Для Python < 3.7
        loop = asyncio.get_event_loop()
        try:
            loop.run_until_complete(main_preprocess())
        finally:
            loop.close() 