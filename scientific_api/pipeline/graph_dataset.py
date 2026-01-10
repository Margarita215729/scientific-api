"""Генерация крупного корпуса графов для обучения."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from scientific_api.storage.paths import ensure_dirs, get_outputs_dir

GRAPH_ROOT = get_outputs_dir() / "datasets" / "graphs"
REGISTRY_PATH = get_outputs_dir() / "datasets" / "graph_registry.csv"


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
    ensure_dirs([REGISTRY_PATH.parent])
    registry.to_csv(REGISTRY_PATH, index=False)
    return registry


def build_corpus(
    preset_name: str,
    source: str,
    n_graphs: int,
    L_mpc: float,
    k: int,
    seed: int = 42,
) -> pd.DataFrame:
    """Stable callable for notebooks: build graph corpus for one preset/source.

    Saves nodes/edges parquet per graph and updates a global registry CSV.
    """

    rng = np.random.default_rng(seed)
    df = _load_points(preset_name, source)
    out_dir = GRAPH_ROOT / preset_name / source
    ensure_dirs([out_dir, REGISTRY_PATH.parent])

    registry_rows: List[Dict] = []
    for idx in range(n_graphs):
        graph_id = f"{preset_name}__{source}__{idx:04d}"
        sub, center = _sample_window(
            df,
            window=L_mpc,
            rng=rng,
            min_nodes=800,
            max_nodes=4000,
        )
        coords = sub[["x_mpc", "y_mpc", "z_mpc"]].to_numpy()
        edges = _build_knn_edges(coords, k)

        nodes_path = out_dir / f"{graph_id}_nodes.parquet"
        edges_path = out_dir / f"{graph_id}_edges.parquet"
        sub.to_parquet(nodes_path, index=False)
        edges.to_parquet(edges_path, index=False)

        registry_rows.append(
            {
                "preset": preset_name,
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

    new_registry = pd.DataFrame(registry_rows)

    if REGISTRY_PATH.exists():
        existing = pd.read_csv(REGISTRY_PATH)
        # drop potential duplicates for same preset/source to keep latest build
        existing = existing[
            ~(
                (existing["preset"] == preset_name)
                & (existing["source"] == source)
            )
        ]
        combined = pd.concat([existing, new_registry], ignore_index=True)
    else:
        combined = new_registry

    combined.to_csv(REGISTRY_PATH, index=False)
    return new_registry
