"""Manifest helpers for reproducibility."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from .paths import ensure_dirs


def _git_commit() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2]
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def write_manifest(path: Path, payload: Dict[str, Any]) -> Path:
    """Write manifest JSON with timestamp and git commit.

    The caller can pass any fields; timestamp and git_commit are auto-populated
    if absent. Parent directories are created.
    """

    target = Path(path)
    ensure_dirs([target.parent])
    enriched = dict(payload)
    enriched.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
    enriched.setdefault("git_commit", _git_commit())
    with target.open("w", encoding="utf-8") as f:
        json.dump(enriched, f, ensure_ascii=False, indent=2)
    return target
