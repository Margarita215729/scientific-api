"""
Spectral feature extraction for graphs.

This module computes spectral features based on the eigenvalues
of graph Laplacian and adjacency matrices.
"""

from typing import Dict, Optional

import networkx as nx
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh

from app.core.logging import get_logger

logger = get_logger(__name__)


def compute_laplacian_spectrum(
    graph: nx.Graph,
    k: int = 10,
    which: str = "SM",
    normalized: bool = True,
) -> np.ndarray:
    """
    Compute the k smallest eigenvalues of the graph Laplacian.

    Args:
        graph: NetworkX graph.
        k: Number of eigenvalues to compute.
        which: Which eigenvalues to compute ('SM' for smallest, 'LM' for largest).
        normalized: Whether to use normalized Laplacian.

    Returns:
        Array of eigenvalues, sorted in ascending order.
    """
    n_nodes = graph.number_of_nodes()

    if n_nodes == 0:
        return np.array([])

    if k >= n_nodes - 1:
        k = max(1, n_nodes - 2)
        logger.debug(f"Adjusted k to {k} (n_nodes={n_nodes})")

    # Get Laplacian matrix
    if normalized:
        L = nx.normalized_laplacian_matrix(graph)
    else:
        L = nx.laplacian_matrix(graph)

    # Compute eigenvalues
    try:
        eigenvalues = eigsh(L, k=k, which=which, return_eigenvectors=False)
        eigenvalues = np.sort(eigenvalues)
        logger.debug(
            f"Computed {len(eigenvalues)} Laplacian eigenvalues: "
            f"range [{eigenvalues[0]:.6f}, {eigenvalues[-1]:.6f}]"
        )
    except Exception as e:
        logger.warning(f"Could not compute Laplacian eigenvalues: {e}")
        eigenvalues = np.zeros(k)

    return eigenvalues


def compute_adjacency_spectrum(
    graph: nx.Graph,
    k: int = 10,
    which: str = "LM",
) -> np.ndarray:
    """
    Compute the k largest eigenvalues of the adjacency matrix.

    Args:
        graph: NetworkX graph.
        k: Number of eigenvalues to compute.
        which: Which eigenvalues to compute ('LM' for largest magnitude).

    Returns:
        Array of eigenvalues, sorted in descending order by magnitude.
    """
    n_nodes = graph.number_of_nodes()

    if n_nodes == 0:
        return np.array([])

    if k >= n_nodes - 1:
        k = max(1, n_nodes - 2)

    # Get adjacency matrix
    A = nx.adjacency_matrix(graph)

    # Compute eigenvalues
    try:
        eigenvalues = eigsh(A, k=k, which=which, return_eigenvectors=False)
        eigenvalues = np.sort(eigenvalues)[::-1]  # Sort descending
        logger.debug(
            f"Computed {len(eigenvalues)} adjacency eigenvalues: "
            f"range [{eigenvalues[-1]:.6f}, {eigenvalues[0]:.6f}]"
        )
    except Exception as e:
        logger.warning(f"Could not compute adjacency eigenvalues: {e}")
        eigenvalues = np.zeros(k)

    return eigenvalues


def compute_spectral_gap(eigenvalues: np.ndarray) -> float:
    """
    Compute spectral gap (difference between second smallest and smallest eigenvalue).

    For Laplacian, the spectral gap indicates how well-connected the graph is.

    Args:
        eigenvalues: Sorted array of eigenvalues.

    Returns:
        Spectral gap value.
    """
    if len(eigenvalues) < 2:
        return 0.0

    gap = float(eigenvalues[1] - eigenvalues[0])
    return gap


def compute_spectral_statistics(eigenvalues: np.ndarray) -> Dict[str, float]:
    """
    Compute statistics of eigenvalue distribution.

    Args:
        eigenvalues: Array of eigenvalues.

    Returns:
        Dictionary with spectral statistics.
    """
    if len(eigenvalues) == 0:
        return {
            "spectral_mean": 0.0,
            "spectral_std": 0.0,
            "spectral_min": 0.0,
            "spectral_max": 0.0,
            "spectral_median": 0.0,
        }

    stats = {
        "spectral_mean": float(np.mean(eigenvalues)),
        "spectral_std": float(np.std(eigenvalues)),
        "spectral_min": float(np.min(eigenvalues)),
        "spectral_max": float(np.max(eigenvalues)),
        "spectral_median": float(np.median(eigenvalues)),
    }

    return stats


def compute_spectral_features(
    graph: nx.Graph,
    k_laplacian: int = 10,
    k_adjacency: int = 5,
    normalized_laplacian: bool = True,
    include_raw_eigenvalues: bool = False,
) -> Dict[str, float]:
    """
    Compute spectral features from graph.

    Args:
        graph: NetworkX graph.
        k_laplacian: Number of Laplacian eigenvalues to compute.
        k_adjacency: Number of adjacency eigenvalues to compute.
        normalized_laplacian: Whether to use normalized Laplacian.
        include_raw_eigenvalues: If True, include raw eigenvalue arrays in output.

    Returns:
        Dictionary with spectral features.
    """
    logger.info(
        f"Computing spectral features for graph with "
        f"{graph.number_of_nodes()} nodes"
    )

    features = {}

    # Laplacian spectrum
    laplacian_eigenvalues = compute_laplacian_spectrum(
        graph, k=k_laplacian, normalized=normalized_laplacian
    )

    # Laplacian statistics
    laplacian_stats = compute_spectral_statistics(laplacian_eigenvalues)
    features.update({f"laplacian_{k}": v for k, v in laplacian_stats.items()})

    # Spectral gap
    if len(laplacian_eigenvalues) >= 2:
        features["spectral_gap"] = compute_spectral_gap(laplacian_eigenvalues)

    # Algebraic connectivity (second smallest Laplacian eigenvalue)
    if len(laplacian_eigenvalues) >= 2:
        features["algebraic_connectivity"] = float(laplacian_eigenvalues[1])

    # Adjacency spectrum
    adjacency_eigenvalues = compute_adjacency_spectrum(graph, k=k_adjacency)

    # Adjacency statistics
    adjacency_stats = compute_spectral_statistics(adjacency_eigenvalues)
    features.update({f"adjacency_{k}": v for k, v in adjacency_stats.items()})

    # Spectral radius (largest adjacency eigenvalue)
    if len(adjacency_eigenvalues) > 0:
        features["spectral_radius"] = float(np.abs(adjacency_eigenvalues[0]))

    # Include raw eigenvalues if requested
    if include_raw_eigenvalues:
        for i, val in enumerate(laplacian_eigenvalues):
            features[f"laplacian_eig_{i}"] = float(val)
        for i, val in enumerate(adjacency_eigenvalues):
            features[f"adjacency_eig_{i}"] = float(val)

    logger.info(f"Computed {len(features)} spectral features")

    return features


def compute_normalized_laplacian_trace(graph: nx.Graph) -> float:
    """
    Compute the trace of the normalized Laplacian matrix.

    The trace equals the number of nodes for normalized Laplacian.

    Args:
        graph: NetworkX graph.

    Returns:
        Trace value.
    """
    n_nodes = graph.number_of_nodes()
    return float(n_nodes)


def compute_spectral_entropy(eigenvalues: np.ndarray, eps: float = 1e-10) -> float:
    """
    Compute spectral entropy from eigenvalue distribution.

    Spectral entropy quantifies the irregularity of the eigenvalue distribution.

    Args:
        eigenvalues: Array of eigenvalues.
        eps: Small constant to avoid log(0).

    Returns:
        Spectral entropy value.
    """
    if len(eigenvalues) == 0:
        return 0.0

    # Normalize eigenvalues to form a probability distribution
    eigenvalues_positive = np.abs(eigenvalues) + eps
    probabilities = eigenvalues_positive / eigenvalues_positive.sum()

    # Compute entropy
    entropy = -np.sum(probabilities * np.log(probabilities + eps))

    return float(entropy)


def compute_extended_spectral_features(
    graph: nx.Graph,
    k: int = 20,
) -> Dict[str, float]:
    """
    Compute extended spectral features including entropy and distribution moments.

    Args:
        graph: NetworkX graph.
        k: Number of eigenvalues to compute.

    Returns:
        Dictionary with extended spectral features.
    """
    logger.info("Computing extended spectral features")

    features = {}

    # Compute Laplacian eigenvalues
    laplacian_eigenvalues = compute_laplacian_spectrum(graph, k=k)

    if len(laplacian_eigenvalues) > 0:
        # Spectral entropy
        features["laplacian_entropy"] = compute_spectral_entropy(laplacian_eigenvalues)

        # Higher moments
        features["laplacian_skewness"] = float(
            np.mean((laplacian_eigenvalues - laplacian_eigenvalues.mean()) ** 3)
            / (laplacian_eigenvalues.std() ** 3 + 1e-10)
        )
        features["laplacian_kurtosis"] = float(
            np.mean((laplacian_eigenvalues - laplacian_eigenvalues.mean()) ** 4)
            / (laplacian_eigenvalues.std() ** 4 + 1e-10)
        )

    # Compute adjacency eigenvalues
    adjacency_eigenvalues = compute_adjacency_spectrum(graph, k=min(k, 10))

    if len(adjacency_eigenvalues) > 0:
        features["adjacency_entropy"] = compute_spectral_entropy(adjacency_eigenvalues)

    logger.info(f"Computed {len(features)} extended spectral features")

    return features
