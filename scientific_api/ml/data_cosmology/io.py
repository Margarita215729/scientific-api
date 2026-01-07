"""
I/O functions for loading cosmological data from SDSS DR17 and DESI DR2.

This module provides functions to load galaxy catalog data from local files.
"""

from pathlib import Path
from typing import Optional, Union

import pandas as pd

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


def load_sdss_catalog(
    file_path: Optional[Union[str, Path]] = None,
    catalog_type: str = "galaxy",
) -> pd.DataFrame:
    """
    Load SDSS DR17 galaxy catalog from local file.

    Args:
        file_path: Path to the catalog file (CSV or Parquet).
                  If None, uses default path from DATA_ROOT/raw/cosmology/sdss_dr17.parquet
        catalog_type: Type of catalog ('galaxy', 'quasar', 'star').

    Returns:
        DataFrame containing the catalog with columns:
            - ra: Right Ascension (degrees)
            - dec: Declination (degrees)
            - redshift: Redshift (z)
            - Additional columns depend on the catalog

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If file format is not supported.
    """
    settings = get_settings()

    if file_path is None:
        file_path = settings.DATA_ROOT / "raw" / "cosmology" / "sdss_dr17.parquet"
    else:
        file_path = Path(file_path)

    if not file_path.exists():
        logger.error(f"SDSS catalog file not found: {file_path}")
        raise FileNotFoundError(f"SDSS catalog file not found: {file_path}")

    logger.info(f"Loading SDSS DR17 {catalog_type} catalog from {file_path}")

    # Determine file format and load
    if file_path.suffix == ".parquet":
        df = pd.read_parquet(file_path)
    elif file_path.suffix == ".csv":
        df = pd.read_csv(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")

    logger.info(f"Loaded {len(df)} objects from SDSS DR17 catalog")
    return df


def load_desi_catalog(
    file_path: Optional[Union[str, Path]] = None,
    catalog_type: str = "galaxy",
) -> pd.DataFrame:
    """
    Load DESI DR2 galaxy catalog from local file.

    Args:
        file_path: Path to the catalog file (CSV or Parquet).
                  If None, uses default path from DATA_ROOT/raw/cosmology/desi_dr2.parquet
        catalog_type: Type of catalog ('galaxy', 'quasar', 'LRG', 'ELG', 'QSO').

    Returns:
        DataFrame containing the catalog with columns:
            - ra: Right Ascension (degrees)
            - dec: Declination (degrees)
            - redshift: Redshift (z)
            - Additional columns depend on the catalog

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If file format is not supported.
    """
    settings = get_settings()

    if file_path is None:
        file_path = settings.DATA_ROOT / "raw" / "cosmology" / "desi_dr2.parquet"
    else:
        file_path = Path(file_path)

    if not file_path.exists():
        logger.error(f"DESI catalog file not found: {file_path}")
        raise FileNotFoundError(f"DESI catalog file not found: {file_path}")

    logger.info(f"Loading DESI DR2 {catalog_type} catalog from {file_path}")

    # Determine file format and load
    if file_path.suffix == ".parquet":
        df = pd.read_parquet(file_path)
    elif file_path.suffix == ".csv":
        df = pd.read_csv(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")

    logger.info(f"Loaded {len(df)} objects from DESI DR2 catalog")
    return df


def load_catalog_sample(
    source: str,
    file_path: Optional[Union[str, Path]] = None,
    sample_size: Optional[int] = None,
    random_state: Optional[int] = None,
) -> pd.DataFrame:
    """
    Load a sample from a cosmological catalog.

    Args:
        source: Data source ('sdss' or 'desi').
        file_path: Optional path to the catalog file.
        sample_size: If provided, return a random sample of this size.
        random_state: Random seed for reproducible sampling.

    Returns:
        DataFrame with sampled catalog data.

    Raises:
        ValueError: If source is not recognized.
    """
    settings = get_settings()
    if random_state is None:
        random_state = settings.ML_RANDOM_SEED

    source = source.lower()

    if source == "sdss":
        df = load_sdss_catalog(file_path)
    elif source == "desi":
        df = load_desi_catalog(file_path)
    else:
        raise ValueError(f"Unknown source: {source}. Use 'sdss' or 'desi'.")

    if sample_size is not None and sample_size < len(df):
        logger.info(f"Sampling {sample_size} objects from {len(df)} total")
        df = df.sample(n=sample_size, random_state=random_state)

    return df


def save_processed_catalog(
    df: pd.DataFrame,
    output_path: Union[str, Path],
    format: str = "parquet",
) -> None:
    """
    Save processed catalog to disk.

    Args:
        df: DataFrame to save.
        output_path: Path where to save the file.
        format: Output format ('parquet' or 'csv').

    Raises:
        ValueError: If format is not supported.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving {len(df)} objects to {output_path}")

    if format == "parquet":
        df.to_parquet(output_path, index=False)
    elif format == "csv":
        df.to_csv(output_path, index=False)
    else:
        raise ValueError(f"Unsupported format: {format}")

    logger.info(f"Successfully saved catalog to {output_path}")
