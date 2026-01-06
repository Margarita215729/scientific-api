"""Генерация крупного корпуса графов для обучения."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

GRAPH_ROOT = Path("outputs/datasets/graphs")
REGISTRY_PATH = Path("outputs/datasets/graph_registry.csv")


def _load_points(preset: str, source: str) -> pd.DataFrame:
    path = Path("data/processed/cosmology") / preset / f"{source}__matched.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {path}")
    df = pd.read_parquet(path)
    for col in ["x_mpc", "y_mpc", "z_mpc"]:
        if col not in df.columns:
            raise ValueError(f"Нет столбца {col} в {path}")
    return df.reset_index(drop=True)


def _sample_window(
    df: pd.DataFrame,
    window: float,
    rng: np.random.Generator,
    min_nodes: int = 800,
    max_nodes: int = 4000,
) -> Tuple[pd.DataFrame, Tuple[float, float, float]]:
    coords = df[["x_mpc", "y_mpc", "z_mpc"]].to_numpy()
    attempts = 0
    while attempts < 30:
        idx = rng.integers(0, len(df))
        cx, cy, cz = coords[idx]
        mask = (
            (np.abs(coords[:, 0] - cx) <= window / 2)
            & (np.abs(coords[:, 1] - cy) <= window / 2)
            & (np.abs(coords[:, 2] - cz) <= window / 2)
        )
        sub = df.loc[mask].copy()
        if len(sub) < min_nodes:
            attempts += 1
            continue
        if len(sub) > max_nodes:
            choose = rng.choice(len(sub), size=max_nodes, replace=False)
            sub = sub.iloc[choose].copy()
        return sub.reset_index(drop=True), (float(cx), float(cy), float(cz))
    raise RuntimeError("Не удалось подобрать окно с достаточным числом узлов")


def _build_knn_edges(coords: np.ndarray, k: int) -> pd.DataFrame:
    nn = NearestNeighbors(n_neighbors=min(k + 1, len(coords)), algorithm="auto")
    nn.fit(coords)
    dists, idxs = nn.kneighbors(coords)
    edges = []
    for i, (dist_row, idx_row) in enumerate(zip(dists, idxs)):
        for d, j in zip(dist_row[1:], idx_row[1:]):  # пропускаем саму точку
            if j <= i:
                continue
            edges.append((i, j, float(d)))
    return pd.DataFrame(edges, columns=["source", "target", "weight"])


def build_graph_corpus(
    presets: Iterable[str],
    n_graphs_per_source_per_preset: int = 500,
    window_size_mpc: float = 200.0,
    k_neighbors: int = 12,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    registry_rows: List[Dict] = []

    for preset in presets:
        for source in ["sdss_dr17", "desi_dr1"]:
            df = _load_points(preset, source)
            out_dir = GRAPH_ROOT / preset / source
            out_dir.mkdir(parents=True, exist_ok=True)
            for idx in range(n_graphs_per_source_per_preset):
                graph_id = f"graph_{idx:04d}"
                sub, center = _sample_window(df, window_size_mpc, rng)
                coords = sub[["x_mpc", "y_mpc", "z_mpc"]].to_numpy()
                edges = _build_knn_edges(coords, k_neighbors)

                nodes_path = out_dir / f"{graph_id}_nodes.parquet"
                edges_path = out_dir / f"{graph_id}_edges.parquet"
                sub.to_parquet(nodes_path, index=False)
                edges.to_parquet(edges_path, index=False)

                registry_rows.append(
                    {
                        "preset": preset,
                        "source": source,
                        "graph_id": graph_id,
                        "n_nodes": len(sub),
                        "n_edges": len(edges),
                        "center_x": center[0],
                        "center_y": center[1],
                        "center_z": center[2],
                        "path_nodes": str(nodes_path),
                        "path_edges": str(edges_path),
                    }
                )
    registry = pd.DataFrame(registry_rows)
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    registry.to_csv(REGISTRY_PATH, index=False)
    return registry
