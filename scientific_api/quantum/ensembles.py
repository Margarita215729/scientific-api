"""Quantum graph ensembles satisfying 'Набор_Критериев_для_Квантовых_Выборок'.

We generate discrete tight-binding / Anderson-type Hamiltonians on:
- 1D ring (cycle)
- 2D torus (periodic lattice)
- random regular graphs (RRG)
- rewired torus (controlled structural noise p)

We expose:
- adjacency / hopping graph edges
- Anderson onsite disorder W
- optional rewire probability p (for topology perturbation)
- ensemble generation with fixed seeds

Outputs are compatible with scientific_api.graphs.knn.GraphData for unified storage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
import networkx as nx


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def ring_graph(n: int) -> nx.Graph:
    return nx.cycle_graph(n)


def torus_2d_graph(n_side: int) -> nx.Graph:
    # periodic 2D grid
    return nx.grid_2d_graph(n_side, n_side, periodic=True)


def rrg_graph(n: int, d: int, seed: int) -> nx.Graph:
    return nx.random_regular_graph(d=d, n=n, seed=seed)


def rewire_edges(g: nx.Graph, p: float, seed: int) -> nx.Graph:
    # For torus: use watts-strogatz like rewiring, but preserve node set
    rng = _rng(seed)
    nodes = list(g.nodes())
    # convert to integer labels
    g2 = nx.convert_node_labels_to_integers(g, ordering="sorted")
    edges = list(g2.edges())
    n = g2.number_of_nodes()

    for (u, v) in edges:
        if rng.random() < p:
            # remove edge and rewire u to random w != u
            g2.remove_edge(u, v)
            w = int(rng.integers(0, n))
            while w == u or g2.has_edge(u, w):
                w = int(rng.integers(0, n))
            g2.add_edge(u, w)

    return g2


def graph_to_tables(g: nx.Graph) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Nodes table: only node_id (coords are optional for abstract graphs)
    nodes = pd.DataFrame({"node_id": list(g.nodes())})
    # Edges table with unit weight (hopping amplitude scale handled separately)
    edges = pd.DataFrame([(u, v, 1.0) for u, v in g.edges()], columns=["source","target","weight"])
    return nodes, edges


def anderson_onsite_potential(n: int, W: float, seed: int) -> np.ndarray:
    """Uniform disorder in [-W/2, W/2]."""
    rng = _rng(seed)
    return rng.uniform(-W/2.0, W/2.0, size=n)


def ipr(evecs: np.ndarray) -> np.ndarray:
    """Inverse Participation Ratio for each eigenvector (columns)."""
    # normalize safety
    psi2 = np.abs(evecs)**2
    denom = np.sum(psi2, axis=0)
    denom[denom == 0] = np.nan
    psi2 = psi2 / denom
    return np.sum(psi2**2, axis=0)


def laplacian_spectrum(g: nx.Graph, k: int = 32) -> np.ndarray:
    import scipy.sparse as sp
    from scipy.sparse.linalg import eigsh
    n = g.number_of_nodes()
    if n < 2:
        return np.array([])
    A = nx.to_scipy_sparse_array(g, format="csr", dtype=float)
    # normalized Laplacian
    from scipy.sparse import csgraph
    L = csgraph.laplacian(A, normed=True)
    k_eff = min(k, n-1)
    vals = eigsh(L, k=k_eff, which="SM", return_eigenvectors=False)
    return np.sort(np.real(vals))


def tight_binding_hamiltonian(g: nx.Graph, W: float, seed: int, t: float = 1.0):
    """Sparse Hamiltonian: H = diag(epsilon) - t * A."""
    import scipy.sparse as sp
    n = g.number_of_nodes()
    eps = anderson_onsite_potential(n, W=W, seed=seed)
    A = nx.to_scipy_sparse_array(g, format="csr", dtype=float)
    H = sp.diags(eps, offsets=0, format="csr") - t * A
    return H, eps


def quantum_graph_ensemble(
    family: str,
    n: int,
    W: float,
    p: float,
    ensemble_size: int,
    seed: int,
    rrg_degree: int = 4,
) -> List[Dict]:
    """Generate a list of ensemble members with tables+meta."""
    out = []
    base_seed = seed

    for r in range(ensemble_size):
        member_seed = base_seed + 10_000*r

        if family == "ring":
            g = ring_graph(n)
            g = nx.convert_node_labels_to_integers(g)
        elif family == "torus":
            side = int(np.sqrt(n))
            if side*side != n:
                raise ValueError("For torus, n must be a perfect square (n_side^2)")
            g = torus_2d_graph(side)
            g = nx.convert_node_labels_to_integers(g, ordering="sorted")
        elif family == "rrg":
            g = rrg_graph(n, d=rrg_degree, seed=member_seed)
        elif family == "rewired_torus":
            side = int(np.sqrt(n))
            if side*side != n:
                raise ValueError("For rewired_torus, n must be a perfect square")
            g0 = torus_2d_graph(side)
            g = rewire_edges(g0, p=p, seed=member_seed)
        else:
            raise ValueError(f"Unknown family: {family}")

        nodes, edges = graph_to_tables(g)
        H, eps = tight_binding_hamiltonian(g, W=W, seed=member_seed, t=1.0)

        out.append({
            "family": family,
            "n": int(n),
            "W": float(W),
            "p": float(p),
            "seed": int(member_seed),
            "nodes": nodes,
            "edges": edges,
            "onsite": eps,
            "H": H,
        })

    return out
