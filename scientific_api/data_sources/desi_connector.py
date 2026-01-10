"""
DESI Data Release 1 (DR1) Data Connector

Модуль для загрузки данных из DESI DR1 (Dark Energy Spectroscopic Instrument).
DESI DR1 содержит более 18 миллионов спектров галактик, квазаров и звёзд
с наблюдений May 2021 — June 2022.

Источники данных:
- SPARCL API (NOIRLab): https://astrosparcl.datalab.noirlab.edu/
- Прямой доступ: https://data.desi.lbl.gov/public/dr1/
- CosmoHub: https://cosmohub.pic.es/ (LSS catalogs)

Лицензия: CC BY 4.0 (Creative Commons Attribution 4.0 International)
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Опционально: для работы с FITS файлами
try:
    from astropy.io import fits
    from astropy.table import Table

    ASTROPY_AVAILABLE = True
except ImportError:
    ASTROPY_AVAILABLE = False

# Опционально: для SPARCL API
try:
    from sparcl.client import SparclClient

    SPARCL_AVAILABLE = True
except ImportError:
    SPARCL_AVAILABLE = False

logger = logging.getLogger(__name__)


# DESI DR1 конфигурация
DESI_DR1_BASE_URL = "https://data.desi.lbl.gov/public/dr1"
DESI_LSS_CATALOGS_URL = f"{DESI_DR1_BASE_URL}/survey/catalogs/dr1/LSS/iron/LSScats/v1.5"

# Типы трейсеров DESI
DESI_TRACERS = {
    "BGS": "Bright Galaxy Survey (z < 0.6)",
    "LRG": "Luminous Red Galaxies (0.4 < z < 1.1)",
    "ELG": "Emission Line Galaxies (0.8 < z < 1.6)",
    "QSO": "Quasars (0.8 < z < 3.5)",
}


class DESIDataLoader:
    """
    Загрузчик данных DESI DR1.

    Поддерживает:
    - Загрузку LSS каталогов (Large-Scale Structure)
    - Доступ через SPARCL API
    - Локальные FITS файлы
    """

    def __init__(self, cache_dir: Optional[str] = None, use_sparcl: bool = True):
        """
        Инициализация загрузчика.

        Args:
            cache_dir: Директория для кэширования данных
            use_sparcl: Использовать SPARCL API если доступен
        """
        self.cache_dir = Path(cache_dir) if cache_dir else Path("data/raw/desi")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.use_sparcl = use_sparcl and SPARCL_AVAILABLE
        self._sparcl_client = None

        logger.info(f"DESI DataLoader initialized. Cache: {self.cache_dir}")
        logger.info(
            f"SPARCL available: {SPARCL_AVAILABLE}, Astropy: {ASTROPY_AVAILABLE}"
        )

    @property
    def sparcl_client(self) -> Optional["SparclClient"]:
        """Lazy-инициализация SPARCL клиента."""
        if self._sparcl_client is None and SPARCL_AVAILABLE:
            try:
                self._sparcl_client = SparclClient()
                logger.info("SPARCL client initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize SPARCL: {e}")
        return self._sparcl_client

    def load_lss_catalog(
        self,
        tracer: str = "LRG",
        region: str = "NGC",  # NGC или SGC
        max_rows: Optional[int] = None,
        z_min: float = 0.0,
        z_max: float = 2.0,
    ) -> pd.DataFrame:
        """
        Загрузка LSS (Large-Scale Structure) каталога DESI.

        Args:
            tracer: Тип трейсера (BGS, LRG, ELG, QSO)
            region: Регион неба (NGC - North, SGC - South)
            max_rows: Максимальное число строк
            z_min: Минимальное красное смещение
            z_max: Максимальное красное смещение

        Returns:
            DataFrame с колонками: ra, dec, z, weight, tracer
        """
        cache_file = (
            self.cache_dir
            / f"desi_lss_{tracer}_{region}_z{z_min:.1f}-{z_max:.1f}.parquet"
        )

        if cache_file.exists():
            logger.info(f"Loading from cache: {cache_file}")
            df = pd.read_parquet(cache_file)
            if max_rows and len(df) > max_rows:
                df = df.sample(n=max_rows, random_state=42)
            return df

        # Попытка загрузки через SPARCL или локальные файлы
        df = self._load_lss_via_sparcl(tracer, region, z_min, z_max)

        if df is None or len(df) == 0:
            logger.warning("SPARCL load failed, trying direct download...")
            df = self._download_lss_sample(tracer, region, z_min, z_max)

        if df is not None and len(df) > 0:
            # Кэширование
            df.to_parquet(cache_file)
            logger.info(f"Cached {len(df)} objects to {cache_file}")

            if max_rows and len(df) > max_rows:
                df = df.sample(n=max_rows, random_state=42)

        return df

    def _load_lss_via_sparcl(
        self, tracer: str, region: str, z_min: float, z_max: float
    ) -> Optional[pd.DataFrame]:
        """Загрузка через SPARCL API."""
        if not self.sparcl_client:
            return None

        try:
            # SPARCL поддерживает поиск по RA/Dec и спектральному типу
            # Это пример — реальный запрос зависит от API версии
            logger.info(f"Querying SPARCL for {tracer} in {region}...")

            # Определяем область поиска по региону
            if region == "NGC":
                ra_range = (100, 280)  # Северная галактическая шапка
            else:
                ra_range = (300, 60)  # Южная галактическая шапка

            # Пример запроса (API может отличаться)
            constraints = {
                "spectype": ["GALAXY"] if tracer != "QSO" else ["QSO"],
                "redshift": (z_min, z_max),
            }

            # Поиск через find
            results = self.sparcl_client.find(constraints=constraints, limit=50000)

            if results and len(results) > 0:
                df = pd.DataFrame(
                    {
                        "ra": [r.ra for r in results],
                        "dec": [r.dec for r in results],
                        "z": [r.redshift for r in results],
                        "tracer": tracer,
                        "source": "DESI_DR1",
                    }
                )
                logger.info(f"SPARCL returned {len(df)} objects")
                return df

        except Exception as e:
            logger.warning(f"SPARCL query failed: {e}")

        return None

    def _download_lss_sample(
        self, tracer: str, region: str, z_min: float, z_max: float
    ) -> Optional[pd.DataFrame]:
        """
        Загрузка образца LSS каталога через прямое скачивание.

        Использует публичные файлы с data.desi.lbl.gov
        """
        if not ASTROPY_AVAILABLE:
            logger.error("Astropy required for FITS file loading")
            return None

        try:
            import urllib.request

            # Формируем URL для clustering каталога
            filename = f"{tracer}_{region}_clustering.dat.fits"
            url = f"{DESI_LSS_CATALOGS_URL}/{filename}"

            local_file = self.cache_dir / filename

            if not local_file.exists():
                logger.info(f"Downloading {url}...")
                # Это может занять время для больших файлов
                urllib.request.urlretrieve(url, local_file)
                logger.info(f"Downloaded to {local_file}")

            # Чтение FITS файла
            with fits.open(local_file) as hdul:
                data = Table(hdul[1].data).to_pandas()

            # Стандартизация колонок
            df = pd.DataFrame(
                {
                    "ra": data["RA"].values,
                    "dec": data["DEC"].values,
                    "z": data["Z"].values,
                    "weight": data.get("WEIGHT", np.ones(len(data))),
                    "tracer": tracer,
                    "source": "DESI_DR1",
                }
            )

            # Фильтрация по z
            df = df[(df["z"] >= z_min) & (df["z"] <= z_max)]

            logger.info(f"Loaded {len(df)} {tracer} objects from DESI DR1")
            return df

        except Exception as e:
            logger.error(f"Failed to download DESI data: {e}")
            return None

    def load_combined_sample(
        self,
        tracers: List[str] = ["LRG", "ELG"],
        max_per_tracer: int = 10000,
        z_min: float = 0.4,
        z_max: float = 1.6,
    ) -> pd.DataFrame:
        """
        Загрузка комбинированной выборки из нескольких трейсеров.

        Args:
            tracers: Список типов трейсеров
            max_per_tracer: Максимум объектов на трейсер
            z_min: Минимальное красное смещение
            z_max: Максимальное красное смещение

        Returns:
            Объединённый DataFrame
        """
        dfs = []

        for tracer in tracers:
            for region in ["NGC", "SGC"]:
                df = self.load_lss_catalog(
                    tracer=tracer,
                    region=region,
                    max_rows=max_per_tracer // 2,
                    z_min=z_min,
                    z_max=z_max,
                )
                if df is not None and len(df) > 0:
                    dfs.append(df)

        if dfs:
            combined = pd.concat(dfs, ignore_index=True)
            logger.info(f"Combined sample: {len(combined)} objects from {tracers}")
            return combined

        return pd.DataFrame()

    def get_data_summary(self) -> Dict[str, Any]:
        """Получение сводки по доступным данным DESI."""
        return {
            "data_release": "DESI DR1",
            "release_date": "2025-03",
            "total_spectra": 18_659_804,
            "galaxies": 13_049_402,
            "quasars": 1_553_713,
            "stars": 4_056_689,
            "spectral_coverage": "360-982.4 nm",
            "spectral_resolution": "2000-5500",
            "photometric_bands": ["g", "r", "z", "W1", "W2", "W3", "W4"],
            "coverage_area_deg2": {"dark": 9528, "bright": 9739, "backup": 2726},
            "tracers": DESI_TRACERS,
            "license": "CC BY 4.0",
            "citation": "DESI Collaboration et al. (2025), arXiv:2503.14745",
        }


def load_desi_data(
    tracer: str = "LRG", max_rows: int = 5000, z_range: Tuple[float, float] = (0.4, 1.1)
) -> pd.DataFrame:
    """
    Упрощённая функция загрузки данных DESI.

    Args:
        tracer: Тип объектов (LRG, ELG, BGS, QSO)
        max_rows: Максимальное число объектов
        z_range: Диапазон красных смещений (min, max)

    Returns:
        DataFrame с координатами и redshift
    """
    loader = DESIDataLoader()
    return loader.load_lss_catalog(
        tracer=tracer, max_rows=max_rows, z_min=z_range[0], z_max=z_range[1]
    )


if __name__ == "__main__":
    # Тестовый запуск
    logging.basicConfig(level=logging.INFO)

    loader = DESIDataLoader()

    print("=== DESI DR1 Data Summary ===")
    summary = loader.get_data_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")

    print("\n=== Attempting to load LRG sample ===")
    df = loader.load_lss_catalog(tracer="LRG", max_rows=1000, z_min=0.4, z_max=1.1)

    if df is not None and len(df) > 0:
        print(f"Loaded {len(df)} LRG objects")
        print(f"RA range: {df['ra'].min():.2f} — {df['ra'].max():.2f}")
        print(f"Dec range: {df['dec'].min():.2f} — {df['dec'].max():.2f}")
        print(f"Redshift range: {df['z'].min():.4f} — {df['z'].max():.4f}")
    else:
        print("No data loaded (this is expected if running without network access)")
