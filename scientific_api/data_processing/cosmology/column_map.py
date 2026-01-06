"""Кандидаты колонок и вспомогательные функции для DESI DR1.

Файл реализует явное разрешение имён колонок: при отсутствии обязательного
поля генерируется детальная ошибка с перечнем доступных колонок.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping

DESI_COLUMN_CANDIDATES: Dict[str, List[str]] = {
    "ra_deg": ["RA", "ra", "TARGET_RA", "target_ra"],
    "dec_deg": ["DEC", "dec", "TARGET_DEC", "target_dec"],
    "z": ["Z", "z", "Z_COSMO", "z_cosmo", "Z_NOT4CLUS", "z_not4clus"],
    "weight": [
        "WEIGHT",
        "weight",
        "WEIGHT_FKP",
        "weight_fkp",
        "WEIGHT_TOT",
        "weight_tot",
    ],
}


class ColumnResolutionError(ValueError):
    """Ошибка разрешения колонок с диагностикой доступных имён."""

    def __init__(
        self, missing: List[str], available: Iterable[str], path: str | None = None
    ) -> None:
        available_list = list(available)
        message = (
            "Отсутствуют обязательные колонки: "
            f"{missing}. Доступные колонки (первые 200): {available_list[:200]}"
        )
        if path:
            message = f"{message}; файл: {path}"
        super().__init__(message)
        self.missing = missing
        self.available = available_list
        self.path = path


def resolve_columns(
    columns: Iterable[str], mapping: Mapping[str, List[str]]
) -> Dict[str, str]:
    """Подбирает имена колонок согласно списку кандидатов.

    Raises:
        ColumnResolutionError: если хотя бы одна обязательная колонка не найдена.
    """

    available = list(columns)
    available_upper = {c.upper(): c for c in available}
    resolved: Dict[str, str] = {}
    missing: List[str] = []

    for target, candidates in mapping.items():
        found = None
        for cand in candidates:
            key = cand.upper()
            if key in available_upper:
                found = available_upper[key]
                break
        if found:
            resolved[target] = found
        else:
            missing.append(target)

    if missing:
        raise ColumnResolutionError(missing, available)

    return resolved


def resolve_desi_columns(columns: Iterable[str]) -> Dict[str, str]:
    """Разрешает колонки DESI по стандартным кандидатам."""

    return resolve_columns(columns, DESI_COLUMN_CANDIDATES)
