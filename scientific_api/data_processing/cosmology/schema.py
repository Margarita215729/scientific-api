"""Единая схема нормализованных космологических точек."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

REQUIRED_COLUMNS: List[str] = [
    "source",
    "sample",
    "obj_id",
    "ra_deg",
    "dec_deg",
    "z",
    "x_mpc",
    "y_mpc",
    "z_mpc",
    "weight",
]


@dataclass
class DatasetManifest:
    preset: str
    n_sdss_raw: int
    n_desi_raw: int
    n_sdss_after_cuts: int
    n_desi_after_cuts: int
    n_matched: int
    sdss_path: str
    desi_path: str
    sdss_matched_path: str
    desi_matched_path: str
    note: str = ""
