"""Единая точка входа для загрузки SDSS/DESI и нормализации."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import yaml

from scientific_api.data_processing.cosmology.normalize_points import (
    downsample_matched,
    normalize_desi_fits_to_parquet,
    normalize_sdss_csv_to_parquet,
    write_manifest,
)
from scientific_api.data_processing.cosmology.schema import DatasetManifest
from scientific_api.data_sources.cosmology.desi_dr1_lss import (
    download_desi_dr1_clustering,
    load_desi_clustering_points,
)
from scientific_api.data_sources.cosmology.sdss_dr17_sql import (
    fetch_sdss_dr17_points_rect_tiled,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "cosmology_presets.yaml"


def _load_config() -> Dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _git_hash() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT)
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def run_ingest(preset_name: str) -> Tuple[str, DatasetManifest]:
    cfg = _load_config()
    if preset_name not in cfg.get("presets", {}):
        raise ValueError(f"Пресет {preset_name} не найден в {CONFIG_PATH}")
    preset = cfg["presets"][preset_name]

    # SDSS
    sdss_raw_dir = ROOT / "data" / "raw" / "sdss" / "dr17" / preset_name
    sdss_raw_path = fetch_sdss_dr17_points_rect_tiled(preset, preset_name, sdss_raw_dir)
    n_sdss_raw = len(pd.read_csv(sdss_raw_path))

    # DESI
    desi_raw_dir = ROOT / "data" / "raw" / "desi" / "dr1" / preset_name
    desi_paths = []
    n_desi_raw_total = 0
    for entry in preset["desi_dr1"]["files"]:
        p = download_desi_dr1_clustering(
            entry["target"], entry["photsys"], desi_raw_dir
        )
        desi_paths.append(p)
        n_desi_raw_total += len(load_desi_clustering_points(p))

    mapping_out = (
        ROOT / "outputs" / "ingestion" / preset_name / "desi_column_mapping.json"
    )

    sdss_parquet = normalize_sdss_csv_to_parquet(sdss_raw_path, preset, preset_name)
    desi_parquet = normalize_desi_fits_to_parquet(
        tuple(desi_paths), preset, preset_name, mapping_out_path=mapping_out
    )

    sdss_match, desi_match, n_matched = downsample_matched(
        sdss_parquet, desi_parquet, preset_name, seed=preset["limits"]["seed"]
    )

    manifest = DatasetManifest(
        preset=preset_name,
        n_sdss_raw=n_sdss_raw,
        n_desi_raw=n_desi_raw_total,
        n_sdss_after_cuts=len(pd.read_parquet(sdss_parquet)),
        n_desi_after_cuts=len(pd.read_parquet(desi_parquet)),
        n_matched=n_matched,
        sdss_path=str(sdss_parquet),
        desi_path=str(desi_parquet),
        sdss_matched_path=str(sdss_match),
        desi_matched_path=str(desi_match),
        note="matched по минимальному количеству точек",
    )
    manifest_path = write_manifest(manifest)

    run_meta = {
        "preset": preset_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_hash(),
        "sdss_raw_csv": str(sdss_raw_path),
        "desi_fits": [str(p) for p in desi_paths],
        "sdss_parquet": str(sdss_parquet),
        "desi_parquet": str(desi_parquet),
        "sdss_matched": str(sdss_match),
        "desi_matched": str(desi_match),
        "manifest": str(manifest_path),
    }
    out_meta_dir = ROOT / "outputs" / "ingestion" / preset_name
    out_meta_dir.mkdir(parents=True, exist_ok=True)
    run_meta_path = out_meta_dir / "run_meta.json"
    with run_meta_path.open("w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2, ensure_ascii=False)

    summary = (
        f"preset={preset_name} sdss_after={manifest.n_sdss_after_cuts} "
        f"desi_after={manifest.n_desi_after_cuts} matched={n_matched}"
    )
    print(summary)
    return summary, manifest


def run(preset_name: str) -> Dict:
    """Public callable for notebooks and scripts.

    Returns a dict with paths and counts to be used downstream.
    """

    summary, manifest = run_ingest(preset_name)
    return {
        "preset": preset_name,
        "summary": summary,
        "manifest": manifest.model_dump() if hasattr(manifest, "model_dump") else manifest.__dict__,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Космологический ingestion SDSS/DESI")
    parser.add_argument("--preset", required=True, choices=["low_z", "high_z"])
    args = parser.parse_args()
    run_ingest(args.preset)


if __name__ == "__main__":
    main()
