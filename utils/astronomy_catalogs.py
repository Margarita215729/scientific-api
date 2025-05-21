"""
utils/astronomy_catalogs.py

Модуль для работы с астрономическими каталогами, необходимыми для проекта обнаружения
крупномасштабных структур Вселенной:
- SDSS DR17 spectroscopic catalog
- Euclid Q1 MER Final catalog
- DESI DR1 (2025) ELG clustering catalog
- DES Year 6 (DES DR2/Y6 Gold) catalog

Модуль предоставляет функции для:
1. Загрузки данных из каждого каталога
2. Нормализации координат и форматов данных
3. Преобразования сферических координат в декартовы для построения 3D-графа
4. Объединения данных в единый набор
"""

import os
import numpy as np
import pandas as pd
import requests
from io import StringIO
import logging
from astropy.io import fits, ascii
from astropy.table import Table
from astropy.cosmology import Planck15 as cosmo
import astropy.units as u
from astropy.coordinates import SkyCoord
from astroquery.ipac.irsa import Irsa
import warnings

# Отключаем предупреждения для более чистого вывода
warnings.filterwarnings('ignore', category=UserWarning, append=True)

# Настройка логирования
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('AstronomyCatalogs')

# Параметры для загрузки данных
DATA_DIR = "galaxy_data"
OUTPUT_DIR = os.path.join(DATA_DIR, "processed")

# URL-адреса для каталогов
SDSS_URL = "https://data.sdss.org/sas/dr17/sdss/spectro/redux/specObj-dr17.fits"
EUCLID_URL = "https://irsa.ipac.caltech.edu/data/Euclid/public/early/Q1/mer/euclid_q1_mer_ppsavcat_v1.0.fits"
DESI_URL = "https://data.desi.lbl.gov/public/dr1/survey/catalogs/dr1/LSS/iron/LSScats/v1.2/ELG_LOPnotqso_NGC_clustering.dat.fits"
JWST_URL = "https://web.corral.tacc.utexas.edu/ceersdata/DR06/MIRI/miri_catalog.dat"
DES_URL = "http://desdr-server.ncsa.illinois.edu/despublic/Y6_GOLD_v2.0.fits"

# Ограничения на количество строк для выборки 
# (чтобы не загружать полностью многогигабайтные файлы)
MAX_ROWS_EUCLID = None    # None = загрузить все
MAX_ROWS_DESI = None
MAX_ROWS_SDSS = None
MAX_ROWS_JWST = None
MAX_ROWS_DES = 1000000  # по умолчанию ограничим DES Y6 ~1e6 строк (для тестов)

CHUNK_SIZE = 1000000  # размер чанка для построчной загрузки больших таблиц

# Константы для астрономических вычислений
SPEED_OF_LIGHT = 299792.458  # км/с
H0 = 67.74  # Постоянная Хаббла, км/с/Мпк
OM0 = 0.3089  # Плотность материи

def initialize_directories():
    """Создает необходимые директории для работы с данными."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    return True

def find_column_name(col_names, options):
    """
    Найти имя колонки в списке col_names, соответствующее одному из вариантов в options.
    Возвращает реальное имя колонки из col_names или None, если не найдено.
    """
    col_names_upper = [name.upper() for name in col_names]
    for opt in options:
        opt_up = opt.upper()
        if opt_up in col_names_upper:
            idx = col_names_upper.index(opt_up)
            return col_names[idx]
    return None

def get_euclid_data():
    """Загрузить Euclid Q1 MER Final catalog из файла FITS и сконвертировать в CSV."""
    initialize_directories()
    output_path = os.path.join(OUTPUT_DIR, "euclid.csv")
    if os.path.exists(output_path):
        logger.info("Euclid: CSV уже есть — пропускаем.")
        return output_path

    logger.info("Euclid: поиск HDU и чтение таблицы...")
    # пытаемся найти в FITS бинарную таблицу с данными
    tbl = None
    for ext in [1, 2, 3]:
        try:
            candidate = Table.read(EUCLID_URL, hdu=ext)
            if len(candidate) > 10:
                tbl = candidate
                logger.info(f"  → HDU={ext}, строк={len(tbl)}")
                break
        except Exception as e:
            logger.warning(f"Ошибка при чтении HDU={ext}: {str(e)}")
            continue
    if tbl is None:
        raise RuntimeError("Euclid: не удалось найти HDU с данными")

    # Определяем колонки RA, DEC, REDSHIFT
    cols = tbl.colnames
    ra_col = find_column_name(cols, ["RA", "RAJ2000", "ALPHA_J2000", "ra_deg", "ra", "Ra", "ra(deg)"])
    dec_col = find_column_name(cols, ["DEC", "DECJ2000", "DEJ2000", "delta_j2000", "dec_deg", "dec", "Dec", "dec(deg)"])
    
    if not ra_col or not dec_col:
        raise RuntimeError(f"Euclid: не найдены RA/DEC в {cols}")

    # Красное смещение в MER Final обычно отсутствует — заполняем NaN
    z_col = find_column_name(cols, ["REDSHIFT", "Z", "Z_SPEC", "Z_SPEC_PHOT", "PHOTOZ", "z_phot"])
    ra_data = np.array(tbl[ra_col])
    dec_data = np.array(tbl[dec_col])
    z_data = np.array(tbl[z_col]) if z_col else np.full(len(tbl), np.nan)

    # Собираем DataFrame и обрезаем, если надо
    df = pd.DataFrame({"RA": ra_data, "DEC": dec_data, "redshift": z_data, "source": "Euclid"})
    if MAX_ROWS_EUCLID is not None and len(df) > MAX_ROWS_EUCLID:
        df = df.iloc[:MAX_ROWS_EUCLID]

    # Сохраняем
    df.to_csv(output_path, index=False)
    logger.info(f"Euclid: сохранено {len(df)} объектов → {output_path}")
    return output_path

def get_sdss_data():
    """Загрузить SDSS DR17 spectro (FITS) и сконвертировать в CSV."""
    initialize_directories()
    output_path = os.path.join(OUTPUT_DIR, "sdss.csv")
    if os.path.exists(output_path):
        logger.info("SDSS: CSV уже есть — пропускаем.")
        return output_path

    logger.info("SDSS: поиск HDU и чтение таблицы...")
    tbl = None
    for ext in [1, 2, 3]:
        try:
            candidate = Table.read(SDSS_URL, hdu=ext)
            if len(candidate) > 10:
                tbl = candidate
                logger.info(f"SDSS: выбрана HDU={ext}, строк={len(tbl)}")
                break
        except Exception as e:
            logger.warning(f"Ошибка при чтении HDU={ext}: {str(e)}")
            continue
    if tbl is None:
        raise RuntimeError("SDSS: не удалось найти подходящий HDU с данными")

    cols = tbl.colnames
    ra_col = find_column_name(cols, ["RA", "RAJ2000", "ALPHA_J2000", "ra_deg"])
    dec_col = find_column_name(cols, ["DEC", "DECJ2000", "DEJ2000", "delta_j2000", "dec_deg"])
    z_col = find_column_name(cols, ["Z", "REDSHIFT", "Z_NOQSO", "Z_SPEC", "PHOTOZ", "redshift"])

    data = {
        "RA": np.array(tbl[ra_col]),
        "DEC": np.array(tbl[dec_col]),
        "redshift": np.array(tbl[z_col]) if z_col else np.full(len(tbl), np.nan),
        "source": "SDSS"
    }
    df = pd.DataFrame(data)
    if MAX_ROWS_SDSS and len(df) > MAX_ROWS_SDSS:
        df = df.iloc[:MAX_ROWS_SDSS]

    df.to_csv(output_path, index=False)
    logger.info(f"SDSS: сохранено {len(df)} объектов → {output_path}")
    return output_path

def get_desi_data():
    """Загрузить DESI DR1 ELG clustering catalog и сконвертировать в CSV."""
    initialize_directories()
    output_path = os.path.join(OUTPUT_DIR, "desi.csv")
    if os.path.exists(output_path):
        logger.info("Файл DESI (desi.csv) уже существует, пропуск загрузки.")
        return output_path
    
    logger.info("Скачивание данных DESI DR1 ELG clustering...")
    hdul = fits.open(DESI_URL, memmap=True)
    data = hdul[1].data
    
    # Имена колонок
    ra_col = find_column_name(hdul[1].columns.names, ["RA", "RAJ2000", "ALPHA_J2000"])
    dec_col = find_column_name(hdul[1].columns.names, ["DEC", "DECJ2000", "DEJ2000", "DELTA_J2000"])
    z_col = find_column_name(hdul[1].columns.names, ["Z", "REDSHIFT", "Z_SPEC", "ZSPEC", "Z_PHOT", "ZPHOT", "PHOTOZ", "Z_MEAN"])
    
    ra_data = data[ra_col]
    dec_data = data[dec_col]
    if z_col:
        z_data = data[z_col]
    else:
        z_data = np.full(len(ra_data), np.nan)
        
    # Обработка масок
    if hasattr(ra_data, 'mask'):
        ra_data = ra_data.filled(np.nan)
    if hasattr(dec_data, 'mask'):
        dec_data = dec_data.filled(np.nan)
    if hasattr(z_data, 'mask'):
        z_data = z_data.filled(np.nan)
        
    df = pd.DataFrame({
        "RA": ra_data, 
        "DEC": dec_data, 
        "redshift": z_data,
        "source": "DESI"
    })
    
    if MAX_ROWS_DESI is not None and len(df) > MAX_ROWS_DESI:
        df = df.iloc[:MAX_ROWS_DESI]

    df.to_csv(output_path, index=False)
    hdul.close()
    logger.info(f"DESI: сохранено объектов: {len(df)}")
    return output_path

def get_des_data():
    """Загрузить каталог DES Y6 GOLD и сохранить в CSV (построчная обработка из-за большого объема)."""
    initialize_directories()
    output_path = os.path.join(OUTPUT_DIR, "des.csv")
    if os.path.exists(output_path):
        logger.info("Файл DES (des.csv) уже существует, пропуск загрузки.")
        return output_path
    
    logger.info("Скачивание данных DES Y6 GOLD (может занять время)...")
    hdul = fits.open(DES_URL, memmap=True)
    data_hdu = hdul[1]
    
    # Определяем имена колонок
    ra_col = find_column_name(data_hdu.columns.names, ["RA", "RAJ2000", "ALPHA_J2000"])
    dec_col = find_column_name(data_hdu.columns.names, ["DEC", "DECJ2000", "DEJ2000", "DELTA_J2000"])
    z_col = find_column_name(data_hdu.columns.names, ["Z", "REDSHIFT", "PHOTOZ", "Z_MEAN", "Z_SPEC", "ZPHOT"])
    
    # Получаем общее число строк и применяем ограничение MAX_ROWS_DES
    total_rows = data_hdu.header.get('NAXIS2', None)
    if total_rows is None:
        total_rows = len(data_hdu.data)  # на случай, если NAXIS2 недоступен
    if MAX_ROWS_DES is not None and total_rows > MAX_ROWS_DES:
        target_rows = MAX_ROWS_DES
    else:
        target_rows = total_rows
        
    # Открываем выходной файл и пишем заголовок
    with open(output_path, 'w', encoding='utf-8') as f_out:
        f_out.write("RA,DEC,redshift,source\n")
        
    # Читаем и сохраняем данные чанками, чтобы не загружать все в память сразу
    rows_processed = 0
    for start in range(0, target_rows, CHUNK_SIZE):
        stop = min(target_rows, start + CHUNK_SIZE)
        data_chunk = data_hdu.data[start:stop]  # извлекаем срез данных
        ra_data = data_chunk[ra_col]
        dec_data = data_chunk[dec_col]
        if z_col:
            z_data = data_chunk[z_col]
        else:
            z_data = np.full(len(ra_data), np.nan)
            
        # Обрабатываем маски при наличии
        if hasattr(ra_data, 'mask'):
            ra_data = ra_data.filled(np.nan)
        if hasattr(dec_data, 'mask'):
            dec_data = dec_data.filled(np.nan)
        if hasattr(z_data, 'mask'):
            z_data = z_data.filled(np.nan)
            
        # Формируем строки для добавления в CSV
        for i in range(len(ra_data)):
            if np.isfinite(ra_data[i]) and np.isfinite(dec_data[i]):
                z_val = z_data[i] if np.isfinite(z_data[i]) else "NA"
                f_out.write(f"{ra_data[i]},{dec_data[i]},{z_val},DES\n")
                rows_processed += 1
                
        logger.info(f"DES: обработано {rows_processed} из ~{target_rows} объектов...")
        
    hdul.close()
    logger.info(f"DES: сохранено объектов: {rows_processed}")
    return output_path

def get_euclid_by_regions():
    """Загрузить данные Euclid Q1 через TAP-запрос к IRSA, используя конкретные области."""
    initialize_directories()
    output_path = os.path.join(OUTPUT_DIR, "euclid_regions.csv")
    if os.path.exists(output_path):
        logger.info("Файл Euclid regions (euclid_regions.csv) уже существует, пропуск загрузки.")
        return output_path
    
    # Список полей: (название, RA, Dec, радиус)
    regions = [
        ("EDFS", 34.5, -4.5, 0.5),
        ("Euclid Deep Field North", 34.0, 0.0, 0.5),
        ("CEERS", 150.0, 2.2, 0.5),
        ("Fornax", 83.8, -5.4, 0.5)
    ]
    
    # Инициализируем результирующий DataFrame
    df_combined = pd.DataFrame(columns=['RA', 'DEC', 'redshift', 'source', 'region'])
    
    total_count = 0
    for name, ra, dec, radius in regions:
        # Создаем объект SkyCoord для центра области
        coord = SkyCoord(ra, dec, unit='deg')
        try:
            # Выполняем конусный запрос к каталогу euclid_q1_mer_catalogue
            table = Irsa.query_region(coord, catalog="euclid_q1_mer_catalogue",
                                     spatial="Cone", radius=radius * u.deg)
        except Exception as e:
            logger.error(f"Ошибка запроса для области {name}: {e}")
            continue  # пропускаем эту область в случае ошибки
    
        if table is None or len(table) == 0:
            logger.warning(f"{name}: Нет данных в указанной области (0 результатов).")
            continue
            
        # Находим нужные колонки
        ra_col = find_column_name(table.colnames, ["RA", "RAJ2000", "ra", "ALPHA_J2000"])
        dec_col = find_column_name(table.colnames, ["DEC", "DECJ2000", "dec", "DEJ2000"])
        z_col = find_column_name(table.colnames, ["Z", "REDSHIFT", "z", "Z_SPEC", "Z_PHOT"])
        
        # Конвертируем Table в DataFrame
        region_df = pd.DataFrame({
            'RA': table[ra_col].data if ra_col else [],
            'DEC': table[dec_col].data if dec_col else [],
            'redshift': table[z_col].data if z_col else np.full(len(table), np.nan),
            'source': "Euclid",
            'region': name
        })
        
        # Добавляем в общий DataFrame
        df_combined = pd.concat([df_combined, region_df])
        total_count += len(region_df)
        logger.info(f"Область {name}: добавлено {len(region_df)} объектов")
        
    # Сохраняем результат
    if len(df_combined) > 0:
        df_combined.to_csv(output_path, index=False)
        logger.info(f"Euclid (регионы): сохранено {total_count} объектов")
    else:
        logger.warning("Euclid (регионы): не удалось получить данные ни для одного региона")
    
    return output_path

def convert_to_cartesian(df):
    """
    Преобразовать сферические координаты (RA, DEC, redshift) в декартовы (X, Y, Z).
    
    Args:
        df: DataFrame с колонками RA, DEC, redshift
        
    Returns:
        DataFrame с добавленными колонками X, Y, Z
    """
    # Проверка наличия необходимых колонок
    required_cols = ["RA", "DEC", "redshift"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"В DataFrame отсутствует колонка {col}")
    
    # Конвертируем в радианы
    ra_rad = np.radians(df["RA"].values)
    dec_rad = np.radians(df["DEC"].values)
    
    # Получаем z и вычисляем расстояние
    z_vals = df["redshift"].values
    distances = np.full(len(z_vals), np.nan)
    
    # Маска для галактик с известным z
    mask = np.isfinite(z_vals)
    if mask.any():
        # Расчет сопутствующего расстояния для галактик с известным z
        distances[mask] = cosmo.comoving_distance(z_vals[mask]).to(u.Mpc).value
    
    # Для галактик без z берем среднее расстояние (если есть хоть одно известное)
    if mask.any():
        distances[~mask] = np.median(distances[mask])
    
    # Расчет декартовых координат
    x = distances * np.cos(dec_rad) * np.cos(ra_rad)
    y = distances * np.cos(dec_rad) * np.sin(ra_rad)
    z = distances * np.sin(dec_rad)
    
    # Добавляем колонки X, Y, Z в DataFrame
    df_out = df.copy()
    df_out["X"] = x
    df_out["Y"] = y
    df_out["Z"] = z
    df_out["distance_mpc"] = distances
    
    return df_out

def merge_all_data(output_filename="merged_galaxies.csv"):
    """
    Объединить все каталоги в единый набор и преобразовать координаты в декартовы.
    
    Args:
        output_filename: Имя выходного файла
        
    Returns:
        Путь к созданному файлу объединенных данных
    """
    initialize_directories()
    merged_path = os.path.join(OUTPUT_DIR, output_filename)
    if os.path.exists(merged_path):
        logger.info(f"Файл объединенных данных {output_filename} уже существует.")
        return merged_path
    
    # Пути ко всем CSV-файлам
    data_files = [
        (os.path.join(OUTPUT_DIR, "euclid.csv"), "Euclid"),
        (os.path.join(OUTPUT_DIR, "desi.csv"), "DESI"),
        (os.path.join(OUTPUT_DIR, "sdss.csv"), "SDSS"),
        (os.path.join(OUTPUT_DIR, "des.csv"), "DES")
    ]
    
    # Проверяем существование файлов
    existing_files = []
    for file_path, source in data_files:
        if os.path.exists(file_path):
            existing_files.append((file_path, source))
        else:
            logger.warning(f"Файл {file_path} не найден и будет пропущен при объединении.")
    
    if not existing_files:
        raise FileNotFoundError("Нет доступных файлов для объединения.")
    
    # Создаем шапку файла
    with open(merged_path, 'w', encoding='utf-8') as f_out:
        f_out.write("RA,DEC,redshift,source,X,Y,Z,distance_mpc\n")
    
    # Обрабатываем файлы
    total_count = 0
    for file_path, source in existing_files:
        logger.info(f"Обработка {source} из {file_path}...")
        
        # Читаем по чанкам для экономии памяти
        for chunk in pd.read_csv(file_path, chunksize=CHUNK_SIZE):
            # Добавляем колонку source, если ее нет
            if "source" not in chunk.columns:
                chunk["source"] = source
                
            # Преобразуем координаты
            chunk_with_xyz = convert_to_cartesian(chunk)
            
            # Записываем в файл
            chunk_with_xyz.to_csv(merged_path, mode='a', header=False, index=False)
            total_count += len(chunk)
            
    logger.info(f"Объединение завершено. Общее число объектов: {total_count}")
    return merged_path

def get_all_catalogs():
    """
    Загрузить все каталоги, обработать их и создать объединенный набор данных.
    Возвращает словарь с путями к файлам данных.
    """
    initialize_directories()
    
    result = {
        "directory": OUTPUT_DIR,
        "catalogs": {}
    }
    
    # Загрузка данных из разных источников
    try:
        sdss_path = get_sdss_data()
        result["catalogs"]["sdss"] = sdss_path
    except Exception as e:
        logger.error(f"Ошибка при получении данных SDSS: {str(e)}")
    
    try:
        euclid_path = get_euclid_data()
        result["catalogs"]["euclid"] = euclid_path
    except Exception as e:
        logger.error(f"Ошибка при получении данных Euclid: {str(e)}")
    
    try:
        desi_path = get_desi_data()
        result["catalogs"]["desi"] = desi_path
    except Exception as e:
        logger.error(f"Ошибка при получении данных DESI: {str(e)}")
    
    try:
        des_path = get_des_data()
        result["catalogs"]["des"] = des_path
    except Exception as e:
        logger.error(f"Ошибка при получении данных DES: {str(e)}")
    
    # Объединение всех данных
    try:
        merged_path = merge_all_data()
        result["merged"] = merged_path
    except Exception as e:
        logger.error(f"Ошибка при объединении данных: {str(e)}")
    
    return result

def fetch_galaxy_subset(source=None, max_rows=1000, min_redshift=None, max_redshift=None):
    """
    Получить подмножество данных галактик с указанными фильтрами.
    
    Args:
        source: Источник данных (SDSS, Euclid, DESI, DES). Если None, все источники.
        max_rows: Максимальное количество возвращаемых строк.
        min_redshift: Минимальное красное смещение для фильтрации.
        max_redshift: Максимальное красное смещение для фильтрации.
        
    Returns:
        DataFrame с данными галактик.
    """
    merged_path = os.path.join(OUTPUT_DIR, "merged_galaxies.csv")
    
    # Если объединенного файла нет, создаем его
    if not os.path.exists(merged_path):
        try:
            merged_path = merge_all_data()
        except Exception as e:
            raise RuntimeError(f"Не удалось создать объединенный файл: {str(e)}")
    
    # Читаем данные с фильтрацией
    df = pd.read_csv(merged_path)
    
    # Применяем фильтры
    if source:
        df = df[df["source"] == source]
    
    if min_redshift is not None:
        df = df[df["redshift"] >= min_redshift]
    
    if max_redshift is not None:
        df = df[df["redshift"] <= max_redshift]
    
    # Ограничиваем количество строк
    if max_rows is not None and len(df) > max_rows:
        df = df.sample(max_rows) if max_rows < len(df) else df
    
    return df 