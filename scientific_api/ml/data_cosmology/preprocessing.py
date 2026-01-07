"""
Preprocessing functions for cosmological data.

This module provides functions to filter galaxy catalogs and compute
3D Cartesian coordinates from observational data (RA, Dec, redshift).
"""

from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy.constants import speed_of_light

from app.core.logging import get_logger

logger = get_logger(__name__)

# Speed of light in km/s
C_KM_S = speed_of_light / 1000  # ~299792.458 km/s


def filter_by_coordinates(
    df: pd.DataFrame,
    ra_min: Optional[float] = None,
    ra_max: Optional[float] = None,
    dec_min: Optional[float] = None,
    dec_max: Optional[float] = None,
) -> pd.DataFrame:
    """
    Filter catalog by RA and Dec ranges.

    Args:
        df: Input catalog DataFrame.
        ra_min: Minimum Right Ascension (degrees).
        ra_max: Maximum Right Ascension (degrees).
        dec_min: Minimum Declination (degrees).
        dec_max: Maximum Declination (degrees).

    Returns:
        Filtered DataFrame.
    """
    mask = pd.Series([True] * len(df), index=df.index)

    if ra_min is not None:
        mask &= df["ra"] >= ra_min
    if ra_max is not None:
        mask &= df["ra"] <= ra_max
    if dec_min is not None:
        mask &= df["dec"] >= dec_min
    if dec_max is not None:
        mask &= df["dec"] <= dec_max

    filtered = df[mask].copy()
    logger.info(
        f"Filtered by coordinates: {len(df)} -> {len(filtered)} objects "
        f"(RA: [{ra_min}, {ra_max}], Dec: [{dec_min}, {dec_max}])"
    )
    return filtered


def filter_by_redshift(
    df: pd.DataFrame,
    z_min: Optional[float] = None,
    z_max: Optional[float] = None,
) -> pd.DataFrame:
    """
    Filter catalog by redshift range.

    Args:
        df: Input catalog DataFrame.
        z_min: Minimum redshift.
        z_max: Maximum redshift.

    Returns:
        Filtered DataFrame.
    """
    mask = pd.Series([True] * len(df), index=df.index)

    if z_min is not None:
        mask &= df["redshift"] >= z_min
    if z_max is not None:
        mask &= df["redshift"] <= z_max

    filtered = df[mask].copy()
    logger.info(
        f"Filtered by redshift: {len(df)} -> {len(filtered)} objects "
        f"(z: [{z_min}, {z_max}])"
    )
    return filtered


def compute_comoving_distance(
    redshift: np.ndarray,
    H0: float = 70.0,
    Omega_m: float = 0.3,
    Omega_lambda: float = 0.7,
) -> np.ndarray:
    """
    Compute comoving distance from redshift using simplified cosmology.

    This is a simplified approximation. For precise calculations, use astropy.

    Args:
        redshift: Array of redshift values.
        H0: Hubble constant in km/s/Mpc (default: 70).
        Omega_m: Matter density parameter (default: 0.3).
        Omega_lambda: Dark energy density parameter (default: 0.7).

    Returns:
        Comoving distance in Mpc.
    """
    # Simplified approximation: D_c ≈ c * z / H0 for small z
    # For more accurate computation, integrate over E(z)
    z = np.asarray(redshift)

    # Hubble parameter at z: H(z) = H0 * sqrt(Omega_m * (1+z)^3 + Omega_lambda)
    # Comoving distance: D_c = (c/H0) * integral_0^z dz' / E(z')
    # where E(z) = H(z)/H0

    # For simplicity, use linear approximation for low z
    # For production, replace with proper integration or astropy
    distance_mpc = C_KM_S * z / H0

    logger.debug(
        f"Computed comoving distance for {len(z)} objects "
        f"(z range: [{z.min():.4f}, {z.max():.4f}])"
    )
    return distance_mpc


def ra_dec_z_to_cartesian(
    ra: np.ndarray,
    dec: np.ndarray,
    redshift: np.ndarray,
    H0: float = 70.0,
    Omega_m: float = 0.3,
    Omega_lambda: float = 0.7,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert RA, Dec, redshift to 3D Cartesian coordinates.

    Args:
        ra: Right Ascension in degrees.
        dec: Declination in degrees.
        redshift: Redshift values.
        H0: Hubble constant in km/s/Mpc.
        Omega_m: Matter density parameter.
        Omega_lambda: Dark energy density parameter.

    Returns:
        Tuple of (x, y, z) arrays in Mpc.
    """
    # Convert angles to radians
    ra_rad = np.deg2rad(ra)
    dec_rad = np.deg2rad(dec)

    # Compute comoving distance
    distance = compute_comoving_distance(redshift, H0, Omega_m, Omega_lambda)

    # Convert spherical to Cartesian
    x = distance * np.cos(dec_rad) * np.cos(ra_rad)
    y = distance * np.cos(dec_rad) * np.sin(ra_rad)
    z = distance * np.sin(dec_rad)

    logger.info(
        f"Converted {len(ra)} objects to Cartesian coordinates "
        f"(x: [{x.min():.2f}, {x.max():.2f}] Mpc, "
        f"y: [{y.min():.2f}, {y.max():.2f}] Mpc, "
        f"z: [{z.min():.2f}, {z.max():.2f}] Mpc)"
    )

    return x, y, z


def add_cartesian_coordinates(
    df: pd.DataFrame,
    H0: float = 70.0,
    Omega_m: float = 0.3,
    Omega_lambda: float = 0.7,
) -> pd.DataFrame:
    """
    Add Cartesian coordinate columns to catalog DataFrame.

    Args:
        df: Catalog DataFrame with 'ra', 'dec', 'redshift' columns.
        H0: Hubble constant in km/s/Mpc.
        Omega_m: Matter density parameter.
        Omega_lambda: Dark energy density parameter.

    Returns:
        DataFrame with added 'x', 'y', 'z' columns in Mpc.
    """
    df = df.copy()

    x, y, z = ra_dec_z_to_cartesian(
        df["ra"].values,
        df["dec"].values,
        df["redshift"].values,
        H0,
        Omega_m,
        Omega_lambda,
    )

    df["x"] = x
    df["y"] = y
    df["z"] = z

    logger.info(f"Added Cartesian coordinates to {len(df)} objects")
    return df


def preprocess_catalog(
    df: pd.DataFrame,
    ra_range: Optional[Tuple[float, float]] = None,
    dec_range: Optional[Tuple[float, float]] = None,
    z_range: Optional[Tuple[float, float]] = None,
    add_coords: bool = True,
    H0: float = 70.0,
    Omega_m: float = 0.3,
    Omega_lambda: float = 0.7,
) -> pd.DataFrame:
    """
    Apply full preprocessing pipeline to catalog.

    Args:
        df: Input catalog DataFrame.
        ra_range: Tuple of (ra_min, ra_max) in degrees.
        dec_range: Tuple of (dec_min, dec_max) in degrees.
        z_range: Tuple of (z_min, z_max).
        add_coords: Whether to add Cartesian coordinates.
        H0: Hubble constant in km/s/Mpc.
        Omega_m: Matter density parameter.
        Omega_lambda: Dark energy density parameter.

    Returns:
        Preprocessed DataFrame.
    """
    logger.info(f"Starting preprocessing of {len(df)} objects")

    # Apply filters
    if ra_range is not None:
        df = filter_by_coordinates(df, ra_min=ra_range[0], ra_max=ra_range[1])

    if dec_range is not None:
        df = filter_by_coordinates(df, dec_min=dec_range[0], dec_max=dec_range[1])

    if z_range is not None:
        df = filter_by_redshift(df, z_min=z_range[0], z_max=z_range[1])

    # Add Cartesian coordinates
    if add_coords:
        df = add_cartesian_coordinates(df, H0, Omega_m, Omega_lambda)

    logger.info(f"Preprocessing complete: {len(df)} objects remaining")
    return df
