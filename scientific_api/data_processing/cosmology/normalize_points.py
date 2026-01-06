"""Нормализация сырых данных SDSS/DESI в единую схему с комовинговыми координатами."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from scientific_api.data_processing.cosmology.column_map import (
    ColumnResolutionError,
    resolve_desi_columns,
)
from scientific_api.data_processing.cosmology.coords import ra_dec_z_to_xyz
from scientific_api.data_processing.cosmology.schema import (
    REQUIRED_COLUMNS,
    DatasetManifest,
)
from scientific_api.data_sources.cosmology.desi_dr1_lss import (
    load_desi_clustering_points,
)


def _apply_basic_filters(df: pd.DataFrame, preset: Dict) -> pd.DataFrame:
    region = preset["region"]
    z_cfg = preset["redshift"]
    df = df[(df["ra_deg"].between(region["ra_min"], region["ra_max"]))]
    df = df[(df["dec_deg"].between(region["dec_min"], region["dec_max"]))]
    df = df[(df["z"].between(z_cfg["z_min"], z_cfg["z_max"]))]
    df = df.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["obj_id", "ra_deg", "dec_deg", "z", "weight"]
    )
    df = df[(df["z"] > 0) & (df["weight"] > 0)]
    return df


def _add_xyz(df: pd.DataFrame) -> pd.DataFrame:
    x, y, z_axis = ra_dec_z_to_xyz(
        df["ra_deg"].to_numpy(), df["dec_deg"].to_numpy(), df["z"].to_numpy()
    )
    df["x_mpc"] = x
    df["y_mpc"] = y
    df["z_mpc"] = z_axis
    return df


def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Отсутствуют обязательные колонки после нормализации: {missing}"
        )
    return df


def normalize_sdss_csv_to_parquet(
    raw_csv_path: Path, preset: Dict, preset_name: str
) -> Path:
    df = pd.read_csv(raw_csv_path)
    required = ["obj_id", "ra_deg", "dec_deg", "z"]
    for col in required:
        if col not in df.columns:
            raise ValueError(
                f"В CSV {raw_csv_path} нет колонки {col}. Доступные: {df.columns.tolist()[:30]}"
            )

    df = df.rename(columns={"mag_r": "mag_r_raw"})
    df["source"] = "sdss_dr17"
    df["sample"] = f"{preset['sdss_dr17']['class']}_{preset_name}"
    df["weight"] = 1.0
    df = _apply_basic_filters(df, preset)
    df = _add_xyz(df)
    df = _ensure_columns(df)

    out_dir = Path("data/processed/cosmology") / preset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "sdss_dr17.parquet"
    df.to_parquet(out_path, index=False)
    return out_path


def normalize_desi_fits_to_parquet(
    fits_paths: Tuple[Path, ...],
    preset: Dict,
    preset_name: str,
    mapping_out_path: Path | None = None,
) -> Path:
    frames = []
    mapping_records = []
    for fp in fits_paths:
        df_raw = load_desi_clustering_points(fp)
        try:
            mapping = resolve_desi_columns(df_raw.columns)
        except ColumnResolutionError as err:
            raise ColumnResolutionError(err.missing, df_raw.columns, path=str(fp))
        df = df_raw.rename(columns=mapping)
        df["source"] = "desi_dr1"
        df["sample"] = f"{fp.stem}_{preset_name}"
        mapping_records.append({"file": str(fp), "mapping": mapping})
        frames.append(df)

    if not frames:
        raise ValueError("Не загружено ни одного FITS файла DESI")

    if mapping_out_path:
        mapping_out_path.parent.mkdir(parents=True, exist_ok=True)
        with mapping_out_path.open("w", encoding="utf-8") as f:
            json.dump(mapping_records, f, indent=2, ensure_ascii=False)

    df = pd.concat(frames, ignore_index=True)
    df = _apply_basic_filters(df, preset)
    df = _add_xyz(df)
    df = _ensure_columns(df)

    out_dir = Path("data/processed/cosmology") / preset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "desi_dr1.parquet"
    df.to_parquet(out_path, index=False)
    return out_path


def downsample_matched(
    sdss_path: Path, desi_path: Path, preset_name: str, seed: int = 42
) -> Tuple[Path, Path, int]:
    rng = np.random.default_rng(seed)
    sdss = pd.read_parquet(sdss_path)
    desi = pd.read_parquet(desi_path)
    n = min(len(sdss), len(desi))
    if n == 0:
        raise ValueError("После фильтров нет данных для выравнивания SDSS и DESI")
    sdss_idx = rng.choice(len(sdss), size=n, replace=False)
    desi_idx = rng.choice(len(desi), size=n, replace=False)
    sdss_match = sdss.iloc[sdss_idx].reset_index(drop=True)
    desi_match = desi.iloc[desi_idx].reset_index(drop=True)

    out_dir = Path("data/processed/cosmology") / preset_name
    sdss_out = out_dir / "sdss_dr17__matched.parquet"
    desi_out = out_dir / "desi_dr1__matched.parquet"
    sdss_match.to_parquet(sdss_out, index=False)
    desi_match.to_parquet(desi_out, index=False)
    return sdss_out, desi_out, n


def write_manifest(manifest: DatasetManifest) -> Path:
    manifest_dir = Path("data/processed/cosmology/manifests")
    manifest_dir.mkdir(parents=True, exist_ok=True)
    path = manifest_dir / f"{manifest.preset}.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(manifest.__dict__, f, indent=2, ensure_ascii=False)
    return path
