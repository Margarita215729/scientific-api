"""Загрузка и чтение DESI DR1 clustering FITS."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import httpx
import pandas as pd
from astropy.io import fits

logger = logging.getLogger(__name__)

DESI_DR1_CLUSTERING_BASE = "https://data.desi.lbl.gov/public/dr1/survey/catalogs/dr1/LSS/iron/LSScats/v1.5/clustering/"
FILENAME_TEMPLATE = "{TARGET}_{PHOTSYS}_clustering.dat.fits"


def download_desi_dr1_clustering(target: str, photsys: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = FILENAME_TEMPLATE.format(TARGET=target, PHOTSYS=photsys)
    url = f"{DESI_DR1_CLUSTERING_BASE}{filename}"
    out_path = out_dir / filename
    if out_path.exists():
        logger.info("Файл DESI уже загружен: %s", out_path)
        return out_path

    logger.info("Скачивание DESI DR1: %s", url)
    with httpx.Client(timeout=60.0, follow_redirects=True) as client:
        resp = client.get(url)
        resp.raise_for_status()
        out_path.write_bytes(resp.content)
    return out_path


def load_desi_clustering_points(fits_path: Path) -> pd.DataFrame:
    if not fits_path.exists():
        raise FileNotFoundError(f"Файл не найден: {fits_path}")
    with fits.open(fits_path, memmap=True) as hdul:
        data_hdu: Optional[fits.BinTableHDU] = None
        for hdu in hdul:
            if (
                hasattr(hdu, "data")
                and hdu.data is not None
                and len(getattr(hdu.data, "shape", [])) > 0
            ):
                data_hdu = hdu
                break
        if data_hdu is None:
            raise ValueError(f"Не удалось найти табличный HDU в {fits_path}")
        df = pd.DataFrame.from_records(data_hdu.data)
    if df.empty:
        raise ValueError(f"Пустой DataFrame после чтения {fits_path}")
    return df
