"""Извлечение признаков из сгенерированных графов."""

from __future__ import annotations

from pathlib import Path
from typing import List

import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse.linalg import eigsh

REPORT_FEATURES_PARQUET = Path("reports/tables/graph_features.parquet")
REPORT_FEATURES_CSV = Path("reports/tables/graph_features.csv")


def _build_graph(nodes_path: Path, edges_path: Path) -> nx.Graph:
    nodes_df = pd.read_parquet(nodes_path)
    edges_df = pd.read_parquet(edges_path)
    g = nx.Graph()
    for i, row in nodes_df.iterrows():
        g.add_node(int(i), **row.to_dict())
    for _, row in edges_df.iterrows():
        g.add_edge(int(row["source"]), int(row["target"]), weight=float(row["weight"]))
    return g


def _spectral_features(g: nx.Graph, k: int = 30) -> List[float]:
    n = g.number_of_nodes()
    if n == 0:
        return [np.nan] * k
    lap = nx.normalized_laplacian_matrix(g)
    k_eff = min(k, max(1, n - 2))
    try:
        vals = eigsh(lap, k=k_eff, which="SM", return_eigenvectors=False)
    except Exception:
        dense = lap.toarray()
        vals = np.linalg.eigvalsh(dense)
        vals = np.sort(vals)[:k_eff]
    vals = np.sort(np.real(vals))
    if len(vals) < k:
        vals = np.pad(vals, (0, k - len(vals)), constant_values=np.nan)
    return vals.tolist()


def _shortest_path_proxy(g: nx.Graph, sample_size: int = 200) -> float:
    if g.number_of_nodes() == 0 or g.number_of_edges() == 0:
        return np.nan
    nodes = list(g.nodes())
    sample = (
        nodes
        if len(nodes) <= sample_size
        else np.random.choice(nodes, size=sample_size, replace=False)
    )
    lengths = []
    for n in sample:
        sp = nx.single_source_dijkstra_path_length(g, n, weight="weight")
        if len(sp) > 1:
            lengths.extend(list(sp.values())[1:])
    return float(np.mean(lengths)) if lengths else np.nan


def compute_features(registry: pd.DataFrame, k_eigs: int = 30) -> pd.DataFrame:
    rows = []
    for _, row in registry.iterrows():
        g = _build_graph(Path(row["path_nodes"]), Path(row["path_edges"]))
        degrees = np.array([deg for _, deg in g.degree()])
        clustering = nx.average_clustering(g) if g.number_of_nodes() > 0 else np.nan
        components = list(nx.connected_components(g)) if g.number_of_nodes() > 0 else []
        n_components = len(components)
        giant_frac = 0.0
        if components:
            largest = max(len(c) for c in components)
            giant_frac = largest / g.number_of_nodes()
        spectral_vals = _spectral_features(g, k=k_eigs)
        sp_proxy = _shortest_path_proxy(g)

        rows.append(
            {
                "preset": row["preset"],
                "source": row["source"],
                "graph_id": row["graph_id"],
                "n_nodes": g.number_of_nodes(),
                "n_edges": g.number_of_edges(),
                "density": nx.density(g) if g.number_of_nodes() > 1 else np.nan,
                "degree_mean": float(degrees.mean()) if len(degrees) else np.nan,
                "degree_std": float(degrees.std()) if len(degrees) else np.nan,
                "degree_max": float(degrees.max()) if len(degrees) else np.nan,
                "degree_skew": (
                    float(pd.Series(degrees).skew()) if len(degrees) else np.nan
                ),
                "clustering_mean": clustering,
                "n_components": n_components,
                "giant_frac": giant_frac,
                "shortest_path_proxy": sp_proxy,
                "spectral_gap": (
                    float(spectral_vals[1] - spectral_vals[0])
                    if len(spectral_vals) >= 2
                    else np.nan
                ),
                "spectral_mean": float(np.nanmean(spectral_vals)),
                "spectral_std": float(np.nanstd(spectral_vals)),
                **{f"eig_{i}": spectral_vals[i] for i in range(k_eigs)},
            }
        )
    features = pd.DataFrame(rows)
    REPORT_FEATURES_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(REPORT_FEATURES_PARQUET, index=False)
    features.to_csv(REPORT_FEATURES_CSV, index=False)
    return features
