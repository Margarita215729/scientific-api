"""kNN graph construction utilities shared across cosmology and quantum ensembles."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


@dataclass(frozen=True)
class GraphData:
    """Portable graph representation: nodes table + edges table + metadata."""

    nodes: pd.DataFrame   # must include node_id and coordinate columns
    edges: pd.DataFrame   # columns: source, target, weight
    meta: Dict


def build_knn_edges(coords: np.ndarray, k: int, metric: str = "euclidean") -> pd.DataFrame:
    n = len(coords)
    if n < 2:
        return pd.DataFrame(columns=["source", "target", "weight"])

    k_eff = min(k + 1, n)
    nn = NearestNeighbors(n_neighbors=k_eff, metric=metric, algorithm="auto")
    nn.fit(coords)
    dists, idxs = nn.kneighbors(coords)

    edges = []
    for i, (dist_row, idx_row) in enumerate(zip(dists, idxs)):
        for d, j in zip(dist_row[1:], idx_row[1:]):  # skip self
            if j <= i:
                continue
            edges.append((i, j, float(d)))

    return pd.DataFrame(edges, columns=["source", "target", "weight"])


def save_graphdata(g: GraphData, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    g.nodes.to_parquet(out_dir / "nodes.parquet", index=False)
    g.edges.to_parquet(out_dir / "edges.parquet", index=False)

    import json
    with (out_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(g.meta, f, indent=2, ensure_ascii=False)


def load_graphdata(out_dir: Path) -> GraphData:
    import json
    nodes = pd.read_parquet(out_dir / "nodes.parquet")
    edges = pd.read_parquet(out_dir / "edges.parquet")
    with (out_dir / "meta.json").open("r", encoding="utf-8") as f:
        meta = json.load(f)
    return GraphData(nodes=nodes, edges=edges, meta=meta)
