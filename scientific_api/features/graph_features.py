"""Graph-level feature extraction used for BOTH cosmology and quantum graphs.

We keep this minimal-but-informative set for VKR:
- topology: degree stats, clustering, assortativity, connectedness proxies
- spectral: first k eigenvalues of normalized Laplacian + spectral gap
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import networkx as nx
from scipy.sparse import csgraph
from scipy.sparse.linalg import eigsh


def _safe_float(x, default=np.nan) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def edges_to_nx(n_nodes: int, edges: pd.DataFrame) -> nx.Graph:
    g = nx.Graph()
    g.add_nodes_from(range(n_nodes))
    if len(edges) > 0:
        g.add_weighted_edges_from(edges[["source","target","weight"]].itertuples(index=False, name=None))
    return g


def topology_features(g: nx.Graph) -> Dict[str, float]:
    n = g.number_of_nodes()
    m = g.number_of_edges()
    deg = np.array([d for _, d in g.degree()], dtype=float)

    feats = {
        "n_nodes": float(n),
        "n_edges": float(m),
        "density": _safe_float(nx.density(g)),
        "deg_mean": float(np.mean(deg)) if n else np.nan,
        "deg_std": float(np.std(deg)) if n else np.nan,
        "deg_min": float(np.min(deg)) if n else np.nan,
        "deg_max": float(np.max(deg)) if n else np.nan,
    }

    # clustering can be expensive; for n>20k use approximation
    if n <= 20_000:
        feats["clustering_mean"] = _safe_float(nx.average_clustering(g))
    else:
        feats["clustering_mean"] = np.nan

    try:
        feats["assortativity"] = _safe_float(nx.degree_assortativity_coefficient(g))
    except Exception:
        feats["assortativity"] = np.nan

    # connectivity proxies
    try:
        feats["n_components"] = float(nx.number_connected_components(g))
    except Exception:
        feats["n_components"] = np.nan

    return feats


def spectral_features_from_edges(n_nodes: int, edges: pd.DataFrame, k: int = 32) -> Dict[str, float]:
    """Compute first k eigenvalues of normalized Laplacian (smallest)."""
    if n_nodes < 2 or len(edges) == 0:
        return {f"lap_eig_{i:02d}": np.nan for i in range(k)} | {"spectral_gap": np.nan}

    # build sparse adjacency
    row = edges["source"].to_numpy(dtype=int)
    col = edges["target"].to_numpy(dtype=int)
    w = edges["weight"].to_numpy(dtype=float)

    # symmetric
    import scipy.sparse as sp
    A = sp.coo_matrix((w, (row, col)), shape=(n_nodes, n_nodes))
    A = A + A.T
    L = csgraph.laplacian(A, normed=True)

    k_eff = min(k, n_nodes - 1)
    try:
        vals = eigsh(L, k=k_eff, which="SM", return_eigenvectors=False)
        vals = np.sort(np.real(vals))
    except Exception:
        vals = np.full(k_eff, np.nan)

    feats = {f"lap_eig_{i:02d}": (float(vals[i]) if i < len(vals) else np.nan) for i in range(k)}
    if len(vals) >= 2 and np.isfinite(vals[0]) and np.isfinite(vals[1]):
        feats["spectral_gap"] = float(vals[1] - vals[0])
    else:
        feats["spectral_gap"] = np.nan
    return feats


def featurize_graph(nodes: pd.DataFrame, edges: pd.DataFrame, spectral_k: int = 32) -> Dict[str, float]:
    n = int(nodes.shape[0])
    g = edges_to_nx(n, edges)
    feats = {}
    feats.update(topology_features(g))
    feats.update(spectral_features_from_edges(n, edges, k=spectral_k))
    return feats
