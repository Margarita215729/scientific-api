"""
Gromov-Wasserstein Distance for Graphs.

This module provides functions to compute Gromov-Wasserstein (GW) distances
between graphs, which measure structural similarity independent of node labels.

The GW distance compares the internal distance structures of two graphs and
finds an optimal transport plan that minimizes distortion.

Key Functions:
--------------
- compute_gw_distance: Compute GW distance between two graphs
- compute_fused_gw_distance: Compute fused GW distance (structure + features)
- compute_pairwise_gw_matrix: Compute GW distances for multiple graph pairs
- save_distance_matrix / load_distance_matrix: I/O for distance matrices

Dependencies:
-------------
- POT (Python Optimal Transport) library for GW computation
- NetworkX for graph operations
- NumPy for numerical operations

References:
-----------
- Mémoli, F. (2011). Gromov–Wasserstein distances and the metric approach to object matching.
- Peyré, G., et al. (2016). Gromov-Wasserstein averaging of kernel and distance matrices.

Notes:
------
- GW distance is a metric (satisfies triangle inequality)
- Invariant to node permutations
- Computational complexity: O(n³) for graphs with n nodes
- For large graphs, consider downsampling or using approximate methods
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import networkx as nx
import numpy as np
import pandas as pd

try:
    import ot  # Python Optimal Transport library
    POT_AVAILABLE = True
except ImportError:
    POT_AVAILABLE = False
    logging.warning(
        "POT (Python Optimal Transport) library not available. "
        "Install with: pip install POT"
    )

logger = logging.getLogger(__name__)


def graph_to_distance_matrix(
    graph: nx.Graph,
    weight_key: Optional[str] = "weight",
    use_shortest_path: bool = True,
) -> np.ndarray:
    """
    Convert graph to distance matrix.

    Parameters
    ----------
    graph : nx.Graph
        Input graph.
    weight_key : str, optional
        Edge attribute containing weights. If None, all edges have weight 1.
    use_shortest_path : bool, default=True
        If True, use shortest path distances between all node pairs.
        If False, use adjacency-based distances (0 for same node, 1 for adjacent, inf otherwise).

    Returns
    -------
    distance_matrix : np.ndarray
        Symmetric distance matrix (n_nodes, n_nodes).

    Notes
    -----
    - For disconnected graphs, shortest_path uses inf for unreachable pairs
    - Matrix rows/columns follow graph.nodes() order
    """
    n_nodes = len(graph)
    nodes = list(graph.nodes())

    if use_shortest_path:
        # Compute all-pairs shortest paths
        if weight_key is not None and nx.is_weighted(graph, weight=weight_key):
            path_lengths = dict(nx.all_pairs_dijkstra_path_length(graph, weight=weight_key))
        else:
            path_lengths = dict(nx.all_pairs_shortest_path_length(graph))

        # Build distance matrix
        distance_matrix = np.full((n_nodes, n_nodes), np.inf)
        for i, node_i in enumerate(nodes):
            for j, node_j in enumerate(nodes):
                if node_j in path_lengths[node_i]:
                    distance_matrix[i, j] = path_lengths[node_i][node_j]
    else:
        # Adjacency-based distances
        adj_matrix = nx.to_numpy_array(graph, nodelist=nodes, weight=weight_key)
        distance_matrix = np.where(adj_matrix > 0, adj_matrix, np.inf)
        np.fill_diagonal(distance_matrix, 0.0)

    return distance_matrix


def compute_gw_distance(
    graph1: nx.Graph,
    graph2: nx.Graph,
    weight_key: Optional[str] = "weight",
    use_shortest_path: bool = True,
    loss_type: str = "square_loss",
    max_iter: int = 1000,
    tol: float = 1e-9,
    verbose: bool = False,
) -> Tuple[float, np.ndarray]:
    """
    Compute Gromov-Wasserstein distance between two graphs.

    Parameters
    ----------
    graph1 : nx.Graph
        First graph.
    graph2 : nx.Graph
        Second graph.
    weight_key : str, optional
        Edge attribute containing weights.
    use_shortest_path : bool, default=True
        Whether to use shortest path distances.
    loss_type : str, default="square_loss"
        Loss function for GW computation. Options: "square_loss", "kl_loss".
    max_iter : int, default=1000
        Maximum number of iterations for GW solver.
    tol : float, default=1e-9
        Convergence tolerance for GW solver.
    verbose : bool, default=False
        Whether to print solver progress.

    Returns
    -------
    gw_distance : float
        Gromov-Wasserstein distance.
    transport_plan : np.ndarray
        Optimal transport plan (n1, n2).

    Raises
    ------
    ImportError
        If POT library is not installed.

    Notes
    -----
    - Distance matrices are computed from graphs
    - Uniform distributions assumed for node masses
    - Lower GW distance indicates more similar structure
    """
    if not POT_AVAILABLE:
        raise ImportError("POT library required. Install with: pip install POT")

    logger.info(f"Computing GW distance between graphs ({len(graph1)} nodes, {len(graph2)} nodes)...")

    # Convert graphs to distance matrices
    C1 = graph_to_distance_matrix(graph1, weight_key, use_shortest_path)
    C2 = graph_to_distance_matrix(graph2, weight_key, use_shortest_path)

    # Handle inf values (disconnected components)
    max_finite1 = np.max(C1[np.isfinite(C1)]) if np.any(np.isfinite(C1)) else 1.0
    max_finite2 = np.max(C2[np.isfinite(C2)]) if np.any(np.isfinite(C2)) else 1.0
    max_finite = max(max_finite1, max_finite2)

    C1 = np.where(np.isfinite(C1), C1, 2 * max_finite)
    C2 = np.where(np.isfinite(C2), C2, 2 * max_finite)

    # Uniform distributions
    n1, n2 = len(graph1), len(graph2)
    p = np.ones(n1) / n1
    q = np.ones(n2) / n2

    # Compute GW distance
    gw_result = ot.gromov.gromov_wasserstein(
        C1, C2, p, q,
        loss_fun=loss_type,
        max_iter=max_iter,
        tol=tol,
        verbose=verbose,
        log=True,
    )

    transport_plan = gw_result[0]
    gw_distance = gw_result[1]["gw_dist"]

    logger.info(f"GW distance: {gw_distance:.6f}")

    return gw_distance, transport_plan


def compute_fused_gw_distance(
    graph1: nx.Graph,
    graph2: nx.Graph,
    feature_key: str,
    alpha: float = 0.5,
    weight_key: Optional[str] = "weight",
    use_shortest_path: bool = True,
    loss_type: str = "square_loss",
    max_iter: int = 1000,
    tol: float = 1e-9,
    verbose: bool = False,
) -> Tuple[float, np.ndarray]:
    """
    Compute Fused Gromov-Wasserstein distance (structure + node features).

    Parameters
    ----------
    graph1 : nx.Graph
        First graph.
    graph2 : nx.Graph
        Second graph.
    feature_key : str
        Node attribute containing feature vectors.
    alpha : float, default=0.5
        Trade-off parameter: alpha * structure + (1-alpha) * features.
        alpha=1 recovers pure GW, alpha=0 recovers pure Wasserstein.
    weight_key : str, optional
        Edge attribute containing weights.
    use_shortest_path : bool, default=True
        Whether to use shortest path distances.
    loss_type : str, default="square_loss"
        Loss function for GW computation.
    max_iter : int, default=1000
        Maximum iterations.
    tol : float, default=1e-9
        Convergence tolerance.
    verbose : bool, default=False
        Print solver progress.

    Returns
    -------
    fgw_distance : float
        Fused Gromov-Wasserstein distance.
    transport_plan : np.ndarray
        Optimal transport plan (n1, n2).

    Raises
    ------
    ImportError
        If POT library is not installed.

    Notes
    -----
    - Requires node features to be stored as graph attributes
    - Features should be numeric vectors of same dimension
    - Alpha controls balance between structure and feature similarity
    """
    if not POT_AVAILABLE:
        raise ImportError("POT library required. Install with: pip install POT")

    logger.info(f"Computing Fused GW distance (alpha={alpha})...")

    # Convert graphs to distance matrices
    C1 = graph_to_distance_matrix(graph1, weight_key, use_shortest_path)
    C2 = graph_to_distance_matrix(graph2, weight_key, use_shortest_path)

    # Handle inf values
    max_finite1 = np.max(C1[np.isfinite(C1)]) if np.any(np.isfinite(C1)) else 1.0
    max_finite2 = np.max(C2[np.isfinite(C2)]) if np.any(np.isfinite(C2)) else 1.0
    max_finite = max(max_finite1, max_finite2)

    C1 = np.where(np.isfinite(C1), C1, 2 * max_finite)
    C2 = np.where(np.isfinite(C2), C2, 2 * max_finite)

    # Extract node features
    nodes1 = list(graph1.nodes())
    nodes2 = list(graph2.nodes())

    features1 = np.array([graph1.nodes[n][feature_key] for n in nodes1])
    features2 = np.array([graph2.nodes[n][feature_key] for n in nodes2])

    # Compute feature cost matrix (L2 distance)
    M = ot.dist(features1, features2, metric="euclidean")

    # Uniform distributions
    n1, n2 = len(graph1), len(graph2)
    p = np.ones(n1) / n1
    q = np.ones(n2) / n2

    # Compute Fused GW distance
    fgw_result = ot.gromov.fused_gromov_wasserstein(
        M, C1, C2, p, q,
        loss_fun=loss_type,
        alpha=alpha,
        max_iter=max_iter,
        tol=tol,
        verbose=verbose,
        log=True,
    )

    transport_plan = fgw_result[0]
    fgw_distance = fgw_result[1]["fgw_dist"]

    logger.info(f"Fused GW distance: {fgw_distance:.6f}")

    return fgw_distance, transport_plan


def compute_pairwise_gw_matrix(
    graphs: List[nx.Graph],
    weight_key: Optional[str] = "weight",
    use_shortest_path: bool = True,
    loss_type: str = "square_loss",
    max_iter: int = 1000,
    tol: float = 1e-9,
    verbose: bool = False,
) -> np.ndarray:
    """
    Compute pairwise GW distance matrix for a list of graphs.

    Parameters
    ----------
    graphs : list of nx.Graph
        List of graphs.
    weight_key : str, optional
        Edge attribute containing weights.
    use_shortest_path : bool, default=True
        Whether to use shortest path distances.
    loss_type : str, default="square_loss"
        Loss function for GW computation.
    max_iter : int, default=1000
        Maximum iterations.
    tol : float, default=1e-9
        Convergence tolerance.
    verbose : bool, default=False
        Print solver progress.

    Returns
    -------
    gw_matrix : np.ndarray
        Pairwise GW distance matrix (n_graphs, n_graphs).

    Notes
    -----
    - Matrix is symmetric: gw_matrix[i, j] = gw_matrix[j, i]
    - Diagonal is zero (graph distance to itself)
    - Only computes upper triangle for efficiency
    """
    n_graphs = len(graphs)
    gw_matrix = np.zeros((n_graphs, n_graphs))

    logger.info(f"Computing pairwise GW matrix for {n_graphs} graphs...")

    # Compute upper triangle
    total_pairs = n_graphs * (n_graphs - 1) // 2
    pair_count = 0

    for i in range(n_graphs):
        for j in range(i + 1, n_graphs):
            gw_dist, _ = compute_gw_distance(
                graphs[i], graphs[j],
                weight_key=weight_key,
                use_shortest_path=use_shortest_path,
                loss_type=loss_type,
                max_iter=max_iter,
                tol=tol,
                verbose=verbose,
            )
            gw_matrix[i, j] = gw_dist
            gw_matrix[j, i] = gw_dist

            pair_count += 1
            if pair_count % 10 == 0:
                logger.info(f"Processed {pair_count}/{total_pairs} pairs...")

    logger.info(f"Pairwise GW matrix computed")

    return gw_matrix


def save_distance_matrix(
    distance_matrix: np.ndarray,
    graph_ids: List[str],
    output_path: Path,
    metadata: Optional[Dict] = None,
) -> None:
    """
    Save distance matrix to disk.

    Parameters
    ----------
    distance_matrix : np.ndarray
        Distance matrix (n_graphs, n_graphs).
    graph_ids : list of str
        Graph identifiers corresponding to matrix rows/columns.
    output_path : Path
        Output file path (.joblib or .npz).
    metadata : dict, optional
        Additional metadata to save.

    Notes
    -----
    - .joblib format saves matrix, graph_ids, and metadata
    - .npz format saves matrix only (graph_ids as separate file)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix == ".joblib":
        data = {
            "distance_matrix": distance_matrix,
            "graph_ids": graph_ids,
            "metadata": metadata or {},
        }
        joblib.dump(data, output_path)
    elif output_path.suffix == ".npz":
        np.savez(output_path, distance_matrix=distance_matrix)
        # Save graph_ids separately
        ids_path = output_path.with_suffix(".txt")
        with open(ids_path, "w") as f:
            f.write("\n".join(graph_ids))
    else:
        raise ValueError(f"Unsupported format: {output_path.suffix}. Use .joblib or .npz")

    logger.info(f"Distance matrix saved to {output_path}")


def load_distance_matrix(input_path: Path) -> Tuple[np.ndarray, List[str], Dict]:
    """
    Load distance matrix from disk.

    Parameters
    ----------
    input_path : Path
        Input file path (.joblib or .npz).

    Returns
    -------
    distance_matrix : np.ndarray
        Distance matrix.
    graph_ids : list of str
        Graph identifiers.
    metadata : dict
        Metadata.

    Notes
    -----
    - For .npz files, graph_ids loaded from companion .txt file
    """
    input_path = Path(input_path)

    if input_path.suffix == ".joblib":
        data = joblib.load(input_path)
        distance_matrix = data["distance_matrix"]
        graph_ids = data["graph_ids"]
        metadata = data.get("metadata", {})
    elif input_path.suffix == ".npz":
        data = np.load(input_path)
        distance_matrix = data["distance_matrix"]

        # Load graph_ids from companion file
        ids_path = input_path.with_suffix(".txt")
        if ids_path.exists():
            with open(ids_path, "r") as f:
                graph_ids = [line.strip() for line in f]
        else:
            graph_ids = [f"graph_{i}" for i in range(len(distance_matrix))]

        metadata = {}
    else:
        raise ValueError(f"Unsupported format: {input_path.suffix}")

    logger.info(f"Distance matrix loaded from {input_path}")
    return distance_matrix, graph_ids, metadata


# Example usage
if __name__ == "__main__":
    from app.core.logging import setup_logging

    setup_logging()

    if not POT_AVAILABLE:
        logger.error("POT library not available. Install with: pip install POT")
    else:
        # Example: Compute GW distance between two small graphs
        logger.info("=== Example: GW Distance ===")

        # Create two simple graphs
        G1 = nx.karate_club_graph()
        G2 = nx.erdos_renyi_graph(30, 0.1, seed=42)

        logger.info(f"Graph 1: {len(G1)} nodes, {len(G1.edges())} edges")
        logger.info(f"Graph 2: {len(G2)} nodes, {len(G2.edges())} edges")

        # Compute GW distance
        gw_dist, transport = compute_gw_distance(G1, G2, use_shortest_path=True, verbose=True)

        print(f"\nGW Distance: {gw_dist:.6f}")
        print(f"Transport plan shape: {transport.shape}")

        # Example: Pairwise GW matrix for multiple graphs
        logger.info("\n=== Example: Pairwise GW Matrix ===")

        graphs = [
            nx.karate_club_graph(),
            nx.erdos_renyi_graph(30, 0.1, seed=42),
            nx.barabasi_albert_graph(30, 3, seed=42),
        ]

        gw_matrix = compute_pairwise_gw_matrix(graphs, verbose=False)

        print("\nPairwise GW Matrix:")
        print(gw_matrix)

        # Save matrix
        output_dir = Path("data/processed/distances")
        graph_ids = ["karate", "er_random", "ba_scale_free"]
        save_distance_matrix(gw_matrix, graph_ids, output_dir / "gw_matrix_example.joblib")
