"""Тилованный запрос SDSS DR17 через SkyServer SqlSearch."""

from __future__ import annotations

import io
import json
import logging
import math
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import httpx
import pandas as pd

logger = logging.getLogger(__name__)

SDSS_SQL_URL = "https://skyserver.sdss.org/dr17/SkyServerWS/SearchTools/SqlSearch"

SKYSERVER_SQL_URL = "https://skyserver.sdss.org/dr17/SkyServerWS/SearchTools/SqlSearch"

REQUIRED_COLUMNS = ["obj_id", "ra_deg",
                    "dec_deg", "z", "zwarning", "class", "mag_r"]


def _git_hash() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=Path(__file__).resolve().parent.parent.parent,
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def _run_query(sql: str, *, max_retries: int = 6) -> pd.DataFrame:
    """
    Robust SDSS SkyServer SQL query.
    Uses GET cmd=...&format=csv which is the most compatible access pattern.
    """
    last_exc: Exception | None = None

    headers = {
        "User-Agent": "scientific-api/1.0 (thesis ingestion; contact: github.com/Margarita215729)"
    }

    with httpx.Client(timeout=60.0, headers=headers, follow_redirects=True) as client:
        for attempt in range(1, max_retries + 1):
            try:
                r = client.get(SKYSERVER_SQL_URL, params={
                               "cmd": sql, "format": "csv"})
                r.raise_for_status()

                # SkyServer CSV sometimes starts with a comment line
                text = r.text.lstrip("\ufeff")
                if text.startswith("#"):
                    # drop first comment line
                    text = "\n".join(text.splitlines()[1:])

                df = pd.read_csv(io.StringIO(text))
                return df

            except Exception as exc:
                last_exc = exc
                wait = min(2 ** attempt, 60)
                logger.warning("SDSS query failed (attempt %d/%d): %s; sleeping %ds",
                               attempt, max_retries, exc, wait)
                time.sleep(wait)

    raise RuntimeError(f"SDSS DR17 query exhausted retries: {last_exc}")


def _build_sql(tile: Tuple[float, float], preset: Dict) -> str:
    ra_min, ra_max = tile
    dec_min = preset["region"]["dec_min"]
    dec_max = preset["region"]["dec_max"]
    z_min = preset["redshift"]["z_min"]
    z_max = preset["redshift"]["z_max"]
    sclass = preset["sdss_dr17"]["class"]
    zwarns = ", ".join(str(z) for z in preset["sdss_dr17"]["zwarning_allowed"])
    limit = preset["sdss_dr17"]["per_query_limit"]
    sql = f"""
SELECT TOP {limit}
  p.objid        AS obj_id,
  p.ra           AS ra_deg,ёъ
  p.dec          AS dec_deg,
  s.z            AS z,
  s.zWarning     AS zwarning,
  s.class        AS class,
  p.psfMag_r     AS mag_r
FROM PhotoObj AS p
JOIN SpecObj  AS s ON s.bestObjID = p.objID
WHERE
  p.ra  BETWEEN {ra_min} AND {ra_max}
  AND p.dec BETWEEN {dec_min} AND {dec_max}
  AND s.z BETWEEN {z_min} AND {z_max}
  AND s.class = '{sclass}'
  AND s.zWarning IN ({zwarns})
ORDER BY p.objid
"""
    return sql


def fetch_sdss_dr17_points_rect_tiled(
    preset: Dict, preset_name: str, out_raw_dir: Path
) -> Path:
    out_raw_dir.mkdir(parents=True, exist_ok=True)
    target = preset["limits"]["target_points_per_source"]
    ra_span = preset["region"]["ra_max"] - preset["region"]["ra_min"]
    bin_deg = preset["sdss_dr17"]["ra_bin_deg"]
    n_bins = max(1, math.ceil(ra_span / bin_deg))
    tiles: List[Tuple[float, float]] = []
    for i in range(n_bins):
        start = preset["region"]["ra_min"] + i * bin_deg
        end = min(start + bin_deg, preset["region"]["ra_max"])
        tiles.append((start, end))

    frames: List[pd.DataFrame] = []
    queries = 0
    for tile in tiles:
        if queries >= preset["sdss_dr17"]["max_queries"]:
            break
        sql = _build_sql(tile, preset)
        df = _run_query(sql)
        if df.empty or len(df.columns) == 0:
            raise ValueError(f"SDSS ответ без колонок для тайла {tile}")
        missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing_cols:
            raise ValueError(
                f"В ответе нет колонок {missing_cols}; получены {df.columns.tolist()}"
            )
        frames.append(df)
        queries += 1
        merged = pd.concat(frames, ignore_index=True)
        merged = merged.drop_duplicates(subset=["obj_id"])
        if len(merged) >= target:
            break

    merged = pd.concat(frames, ignore_index=True)
    before_dedup = len(merged)
    merged = merged.drop_duplicates(subset=["obj_id"])
    after_dedup = len(merged)

    raw_path = out_raw_dir / f"sdss_dr17__{preset_name}__raw.csv"
    merged.to_csv(raw_path, index=False)

    manifest = {
        "preset": preset_name,
        "query_count": queries,
        "tiles": tiles,
        "rows_before_dedup": before_dedup,
        "rows_after_dedup": after_dedup,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_hash(),
    }
    manifest_path = out_raw_dir / f"sdss_dr17__{preset_name}__manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    return raw_path
