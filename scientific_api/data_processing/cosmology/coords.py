"""Преобразования в комовинговые координаты для космологических точек."""

from __future__ import annotations

import numpy as np
from astropy import units as u
from astropy.cosmology import Planck18


def ra_dec_z_to_xyz(
    ra_deg: np.ndarray, dec_deg: np.ndarray, z: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Преобразует массивы RA/DEC/Z в декартовы координаты (Мпк).

    Используется космология Planck18, расчёт векторизованный.
    """

    ra_rad = np.deg2rad(ra_deg)
    dec_rad = np.deg2rad(dec_deg)

    # Комовинговое расстояние в Мпк
    chi = Planck18.comoving_distance(z * 1.0).to(u.Mpc).value

    x = chi * np.cos(dec_rad) * np.cos(ra_rad)
    y = chi * np.cos(dec_rad) * np.sin(ra_rad)
    z_axis = chi * np.sin(dec_rad)
    return x, y, z_axis
