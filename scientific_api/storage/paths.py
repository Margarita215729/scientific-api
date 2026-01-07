"""Centralized path utilities for data and artifact locations."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable


def get_repo_root() -> Path:
    """Return repository root assuming this file lives under scientific_api/storage.

    Uses filesystem location instead of CWD to remain stable under different entrypoints.
    """

    return Path(__file__).resolve().parents[2]


def get_data_root() -> Path:
    return get_repo_root() / "data"


def get_data_raw_dir() -> Path:
    return get_data_root() / "raw"


def get_data_processed_dir() -> Path:
    return get_data_root() / "processed"


def get_outputs_dir() -> Path:
    return get_repo_root() / "outputs"


def get_reports_dir() -> Path:
    return get_repo_root() / "reports"


def ensure_dirs(paths: Iterable[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def ensure_all_dirs() -> None:
    """Create canonical top-level directories if they do not exist."""

    ensure_dirs(
        [
            get_data_raw_dir(),
            get_data_processed_dir(),
            get_outputs_dir(),
            get_reports_dir(),
        ]
    )
