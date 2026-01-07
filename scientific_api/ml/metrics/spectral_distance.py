"""
Spectral Distance Metrics for Graphs.

This module provides functions to compute spectral distances between graphs
based on eigenvalue spectra of graph matrices (Laplacian, adjacency).

Spectral distances capture global structural properties and are useful for
comparing graphs of different sizes.

Key Functions:
--------------
- compute_laplacian_spectral_distance: L2 distance between Laplacian spectra
- compute_adjacency_spectral_distance: L2 distance between adjacency spectra
- compute_spectral_divergence: KL/JS divergence between spectral densities
- compute_pairwise_spectral_matrix: Pairwise spectral distances

Dependencies:
-------------
- NetworkX for graph operations
- SciPy for eigenvalue computation
- NumPy for numerical operations

References:
-----------
- Wilson, R. C., & Zhu, P. (2008). A study of graph spectra for comparing graphs and trees.
- Shimada, Y., et al. (2016). Graph distance for complex networks.

Notes:
------
- Spectral distances are invariant to node permutations
- Sensitive to graph size - normalize or pad spectra for fair comparison
- Computational complexity: O(n³) for eigenvalue decomposition
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import networkx as nx
import numpy as np
from scipy import stats
from scipy.sparse import linalg as sp_linalg

logger = logging.getLogger(__name__)


def compute_graph_spectrum(
    graph: nx.Graph,
    matrix_type: str = "laplacian",
    k: Optional[int] = None,
    normalized: bool = True,
) -> np.ndarray:
    """
    Compute eigenvalue spectrum of graph matrix.

    Parameters
    ----------
    graph : nx.Graph
        Input graph.
    matrix_type : str, default="laplacian"
        Type of matrix: "laplacian" or "adjacency".
    k : int, optional
        Number of eigenvalues to compute. If None, computes all.
    normalized : bool, default=True
        Whether to use normalized Laplacian/adjacency.

    Returns
    -------
    eigenvalues : np.ndarray
        Sorted eigenvalues (ascending for Laplacian, descending for adjacency).

    Notes
    -----
    - For Laplacian: eigenvalues in [0, 2] for normalized, [0, n*max_degree] for unnormalized
    - For adjacency: eigenvalues in [-1, 1] for normalized (if graph is regular)
    - Uses sparse eigenvalue solver for large graphs
    """
    n_nodes = len(graph)

    if n_nodes == 0:
        return np.array([])

    if k is None:
        k = n_nodes

    k = min(k, n_nodes - 1)  # scipy eigsh requires k < n

    if matrix_type == "laplacian":
        if normalized:
            L = nx.normalized_laplacian_matrix(graph).astype(float)
        else:
            L = nx.laplacian_matrix(graph).astype(float)

        # Compute k smallest eigenvalues
        if k < n_nodes - 1:
            eigenvalues = sp_linalg.eigsh(L, k=k, which="SM", return_eigenvectors=False)
        else:
            eigenvalues = np.linalg.eigvalsh(L.toarray())

        eigenvalues = np.sort(eigenvalues)  # Ascending order

    elif matrix_type == "adjacency":
        if normalized:
            # Normalized adjacency (symmetric normalization)
            A = nx.adjacency_matrix(graph).astype(float)
            degrees = np.array(A.sum(axis=1)).flatten()
            degrees[degrees == 0] = 1  # Avoid division by zero
            D_inv_sqrt = np.diag(1.0 / np.sqrt(degrees))
            A_norm = D_inv_sqrt @ A.toarray() @ D_inv_sqrt
            M = A_norm
        else:
            M = nx.adjacency_matrix(graph).astype(float)

        # Compute k largest eigenvalues (by magnitude)
        if k < n_nodes - 1:
            eigenvalues = sp_linalg.eigsh(M, k=k, which="LM", return_eigenvectors=False)
        else:
            eigenvalues = np.linalg.eigvalsh(M.toarray() if hasattr(M, "toarray") else M)

        eigenvalues = np.sort(eigenvalues)[::-1]  # Descending order

    else:
        raise ValueError(f"Unknown matrix_type: {matrix_type}. Use 'laplacian' or 'adjacency'.")

    return eigenvalues


def pad_or_truncate_spectrum(
    spectrum: np.ndarray,
    target_length: int,
    pad_value: float = 0.0,
) -> np.ndarray:
    """
    Pad or truncate spectrum to target length.

    Parameters
    ----------
    spectrum : np.ndarray
        Input eigenvalue spectrum.
    target_length : int
        Target length.
    pad_value : float, default=0.0
        Value to use for padding.

    Returns
    -------
    spectrum_adjusted : np.ndarray
        Adjusted spectrum of length target_length.

    Notes
    -----
    - If spectrum is longer than target, truncates to target length
    - If spectrum is shorter, pads with pad_value
    """
    current_length = len(spectrum)

    if current_length == target_length:
        return spectrum
    elif current_length > target_length:
        # Truncate
        return spectrum[:target_length]
    else:
        # Pad
        padded = np.full(target_length, pad_value)
        padded[:current_length] = spectrum
        return padded


def compute_laplacian_spectral_distance(
    graph1: nx.Graph,
    graph2: nx.Graph,
    k: Optional[int] = None,
    normalized: bool = True,
    distance_metric: str = "l2",
) -> float:
    """
    Compute spectral distance between Laplacian spectra.

    Parameters
    ----------
    graph1 : nx.Graph
        First graph.
    graph2 : nx.Graph
        Second graph.
    k : int, optional
        Number of eigenvalues to use. If None, uses min(len(graph1), len(graph2)).
    normalized : bool, default=True
        Whether to use normalized Laplacian.
    distance_metric : str, default="l2"
        Distance metric: "l2" (Euclidean) or "l1" (Manhattan).

    Returns
    -------
    distance : float
        Spectral distance.

    Notes
    -----
    - Spectra are padded/truncated to same length before comparison
    - L2 distance: sqrt(sum((s1 - s2)²))
    - L1 distance: sum(|s1 - s2|)
    """
    logger.debug(f"Computing Laplacian spectral distance ({distance_metric})...")

    # Compute spectra
    spec1 = compute_graph_spectrum(graph1, matrix_type="laplacian", k=k, normalized=normalized)
    spec2 = compute_graph_spectrum(graph2, matrix_type="laplacian", k=k, normalized=normalized)

    # Pad/truncate to same length
    target_length = min(len(spec1), len(spec2)) if k is None else k
    spec1 = pad_or_truncate_spectrum(spec1, target_length)
    spec2 = pad_or_truncate_spectrum(spec2, target_length)

    # Compute distance
    if distance_metric == "l2":
        distance = np.linalg.norm(spec1 - spec2)
    elif distance_metric == "l1":
        distance = np.sum(np.abs(spec1 - spec2))
    else:
        raise ValueError(f"Unknown distance_metric: {distance_metric}")

    logger.debug(f"Laplacian spectral distance: {distance:.6f}")
    return distance


def compute_adjacency_spectral_distance(
    graph1: nx.Graph,
    graph2: nx.Graph,
    k: Optional[int] = None,
    normalized: bool = True,
    distance_metric: str = "l2",
) -> float:
    """
    Compute spectral distance between adjacency spectra.

    Parameters
    ----------
    graph1 : nx.Graph
        First graph.
    graph2 : nx.Graph
        Second graph.
    k : int, optional
        Number of eigenvalues to use. If None, uses min(len(graph1), len(graph2)).
    normalized : bool, default=True
        Whether to use normalized adjacency.
    distance_metric : str, default="l2"
        Distance metric: "l2" or "l1".

    Returns
    -------
    distance : float
        Spectral distance.

    Notes
    -----
    - Uses largest k eigenvalues by magnitude
    - Spectra are padded/truncated to same length before comparison
    """
    logger.debug(f"Computing adjacency spectral distance ({distance_metric})...")

    # Compute spectra
    spec1 = compute_graph_spectrum(graph1, matrix_type="adjacency", k=k, normalized=normalized)
    spec2 = compute_graph_spectrum(graph2, matrix_type="adjacency", k=k, normalized=normalized)

    # Pad/truncate to same length
    target_length = min(len(spec1), len(spec2)) if k is None else k
    spec1 = pad_or_truncate_spectrum(spec1, target_length)
    spec2 = pad_or_truncate_spectrum(spec2, target_length)

    # Compute distance
    if distance_metric == "l2":
        distance = np.linalg.norm(spec1 - spec2)
    elif distance_metric == "l1":
        distance = np.sum(np.abs(spec1 - spec2))
    else:
        raise ValueError(f"Unknown distance_metric: {distance_metric}")

    logger.debug(f"Adjacency spectral distance: {distance:.6f}")
    return distance


def compute_spectral_divergence(
    graph1: nx.Graph,
    graph2: nx.Graph,
    matrix_type: str = "laplacian",
    k: Optional[int] = None,
    normalized: bool = True,
    divergence_type: str = "kl",
    bins: int = 50,
) -> float:
    """
    Compute divergence between spectral densities.

    Parameters
    ----------
    graph1 : nx.Graph
        First graph.
    graph2 : nx.Graph
        Second graph.
    matrix_type : str, default="laplacian"
        Type of matrix: "laplacian" or "adjacency".
    k : int, optional
        Number of eigenvalues to use.
    normalized : bool, default=True
        Whether to use normalized matrix.
    divergence_type : str, default="kl"
        Divergence type: "kl" (Kullback-Leibler) or "js" (Jensen-Shannon).
    bins : int, default=50
        Number of bins for histogram estimation of density.

    Returns
    -------
    divergence : float
        Spectral divergence.

    Notes
    -----
    - Spectra are converted to probability distributions via histograms
    - KL divergence is asymmetric: D_KL(P || Q) != D_KL(Q || P)
    - JS divergence is symmetric and bounded: D_JS in [0, 1]
    """
    logger.debug(f"Computing spectral divergence ({divergence_type})...")

    # Compute spectra
    spec1 = compute_graph_spectrum(graph1, matrix_type=matrix_type, k=k, normalized=normalized)
    spec2 = compute_graph_spectrum(graph2, matrix_type=matrix_type, k=k, normalized=normalized)

    # Create common bins
    min_val = min(spec1.min(), spec2.min())
    max_val = max(spec1.max(), spec2.max())
    bin_edges = np.linspace(min_val, max_val, bins + 1)

    # Compute histograms (probability distributions)
    hist1, _ = np.histogram(spec1, bins=bin_edges, density=True)
    hist2, _ = np.histogram(spec2, bins=bin_edges, density=True)

    # Normalize to probability distributions
    hist1 = hist1 / (hist1.sum() + 1e-10)
    hist2 = hist2 / (hist2.sum() + 1e-10)

    # Add small epsilon to avoid log(0)
    hist1 = hist1 + 1e-10
    hist2 = hist2 + 1e-10

    # Compute divergence
    if divergence_type == "kl":
        # KL(P || Q) = sum(P * log(P / Q))
        divergence = np.sum(hist1 * np.log(hist1 / hist2))
    elif divergence_type == "js":
        # JS(P || Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M), M = 0.5 * (P + Q)
        M = 0.5 * (hist1 + hist2)
        divergence = 0.5 * np.sum(hist1 * np.log(hist1 / M)) + 0.5 * np.sum(hist2 * np.log(hist2 / M))
    else:
        raise ValueError(f"Unknown divergence_type: {divergence_type}")

    logger.debug(f"Spectral divergence: {divergence:.6f}")
    return divergence


def compute_pairwise_spectral_matrix(
    graphs: List[nx.Graph],
    matrix_type: str = "laplacian",
    k: Optional[int] = None,
    normalized: bool = True,
    distance_metric: str = "l2",
) -> np.ndarray:
    """
    Compute pairwise spectral distance matrix.

    Parameters
    ----------
    graphs : list of nx.Graph
        List of graphs.
    matrix_type : str, default="laplacian"
        Type of matrix: "laplacian" or "adjacency".
    k : int, optional
        Number of eigenvalues to use.
    normalized : bool, default=True
        Whether to use normalized matrix.
    distance_metric : str, default="l2"
        Distance metric: "l2" or "l1".

    Returns
    -------
    spectral_matrix : np.ndarray
        Pairwise spectral distance matrix (n_graphs, n_graphs).

    Notes
    -----
    - Matrix is symmetric
    - Diagonal is zero
    - Only computes upper triangle for efficiency
    """
    n_graphs = len(graphs)
    spectral_matrix = np.zeros((n_graphs, n_graphs))

    logger.info(f"Computing pairwise spectral matrix for {n_graphs} graphs...")

    # Select distance function
    if matrix_type == "laplacian":
        distance_func = compute_laplacian_spectral_distance
    else:
        distance_func = compute_adjacency_spectral_distance

    # Compute upper triangle
    total_pairs = n_graphs * (n_graphs - 1) // 2
    pair_count = 0

    for i in range(n_graphs):
        for j in range(i + 1, n_graphs):
            dist = distance_func(
                graphs[i], graphs[j],
                k=k,
                normalized=normalized,
                distance_metric=distance_metric,
            )
            spectral_matrix[i, j] = dist
            spectral_matrix[j, i] = dist

            pair_count += 1
            if pair_count % 10 == 0:
                logger.info(f"Processed {pair_count}/{total_pairs} pairs...")

    logger.info("Pairwise spectral matrix computed")
    return spectral_matrix


# Example usage
if __name__ == "__main__":
    from app.core.logging import setup_logging

    setup_logging()

    # Example: Compute spectral distances between graphs
    logger.info("=== Example: Spectral Distances ===")

    # Create test graphs
    G1 = nx.karate_club_graph()
    G2 = nx.erdos_renyi_graph(34, 0.1, seed=42)
    G3 = nx.barabasi_albert_graph(34, 3, seed=42)

    logger.info(f"Graph 1 (Karate Club): {len(G1)} nodes, {len(G1.edges())} edges")
    logger.info(f"Graph 2 (ER Random): {len(G2)} nodes, {len(G2.edges())} edges")
    logger.info(f"Graph 3 (BA Scale-free): {len(G3)} nodes, {len(G3.edges())} edges")

    # Laplacian spectral distances
    print("\n=== Laplacian Spectral Distances (L2) ===")
    d12_lap = compute_laplacian_spectral_distance(G1, G2, k=20, normalized=True)
    d13_lap = compute_laplacian_spectral_distance(G1, G3, k=20, normalized=True)
    d23_lap = compute_laplacian_spectral_distance(G2, G3, k=20, normalized=True)

    print(f"d(G1, G2) = {d12_lap:.6f}")
    print(f"d(G1, G3) = {d13_lap:.6f}")
    print(f"d(G2, G3) = {d23_lap:.6f}")

    # Adjacency spectral distances
    print("\n=== Adjacency Spectral Distances (L2) ===")
    d12_adj = compute_adjacency_spectral_distance(G1, G2, k=20, normalized=True)
    d13_adj = compute_adjacency_spectral_distance(G1, G3, k=20, normalized=True)
    d23_adj = compute_adjacency_spectral_distance(G2, G3, k=20, normalized=True)

    print(f"d(G1, G2) = {d12_adj:.6f}")
    print(f"d(G1, G3) = {d13_adj:.6f}")
    print(f"d(G2, G3) = {d23_adj:.6f}")

    # Spectral divergence (JS)
    print("\n=== Spectral Divergence (JS) ===")
    js12 = compute_spectral_divergence(G1, G2, divergence_type="js", bins=30)
    js13 = compute_spectral_divergence(G1, G3, divergence_type="js", bins=30)
    js23 = compute_spectral_divergence(G2, G3, divergence_type="js", bins=30)

    print(f"JS(G1, G2) = {js12:.6f}")
    print(f"JS(G1, G3) = {js13:.6f}")
    print(f"JS(G2, G3) = {js23:.6f}")

    # Pairwise matrix
    print("\n=== Pairwise Spectral Matrix ===")
    graphs = [G1, G2, G3]
    spectral_matrix = compute_pairwise_spectral_matrix(
        graphs, matrix_type="laplacian", k=20, normalized=True, distance_metric="l2"
    )

    print("Pairwise Laplacian Spectral Distance Matrix:")
    print(spectral_matrix)
