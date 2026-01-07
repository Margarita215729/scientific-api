"""
Distribution Distance Metrics for Graphs.

This module provides functions to compute statistical distances between
graph property distributions (degree, clustering coefficient, path lengths).

Distribution distances capture local structural properties and are useful
for comparing graph statistics.

Key Functions:
--------------
- compute_degree_distribution_distance: Compare degree distributions
- compute_clustering_distribution_distance: Compare clustering coefficients
- compute_path_length_distribution_distance: Compare path length distributions
- compute_distribution_distance: Generic distribution distance
- compute_pairwise_distribution_matrix: Pairwise distribution distances

Dependencies:
-------------
- NetworkX for graph operations
- SciPy for statistical tests
- NumPy for numerical operations

References:
-----------
- Emmert-Streib, F., et al. (2016). Fifty years of graph matching, network alignment and network comparison.
- Soundarajan, S., et al. (2014). Generating graph snapshots from streaming edge data.

Notes:
------
- Distribution distances are sensitive to sampling and graph size
- Multiple distance metrics available: Wasserstein, KL, KS, Chi-square
- For small graphs, distributions may be sparse - use binning carefully
"""

import logging
from typing import List, Optional, Tuple

import networkx as nx
import numpy as np
from scipy import stats
from scipy.spatial import distance as sp_distance

try:
    import ot  # Python Optimal Transport for Wasserstein distance
    POT_AVAILABLE = True
except ImportError:
    POT_AVAILABLE = False

logger = logging.getLogger(__name__)


def compute_distribution_distance(
    dist1: np.ndarray,
    dist2: np.ndarray,
    distance_metric: str = "wasserstein",
    bins: Optional[np.ndarray] = None,
) -> float:
    """
    Compute distance between two distributions.

    Parameters
    ----------
    dist1 : np.ndarray
        First distribution (samples or histogram).
    dist2 : np.ndarray
        Second distribution (samples or histogram).
    distance_metric : str, default="wasserstein"
        Distance metric:
        - "wasserstein": Earth Mover's Distance (requires POT or scipy)
        - "kl": Kullback-Leibler divergence (requires histograms)
        - "js": Jensen-Shannon divergence (requires histograms)
        - "ks": Kolmogorov-Smirnov statistic (for samples)
        - "chisquare": Chi-square statistic (for histograms)
        - "l2": L2 distance (for histograms)
    bins : np.ndarray, optional
        Bin edges for histogram-based metrics. If None, uses automatic binning.

    Returns
    -------
    distance : float
        Distance between distributions.

    Notes
    -----
    - For "kl", "js", "chisquare", "l2": inputs are treated as histograms
    - For "wasserstein", "ks": inputs are treated as samples
    - KL divergence is asymmetric, others are symmetric
    """
    if distance_metric == "wasserstein":
        # Wasserstein distance (Earth Mover's Distance)
        if POT_AVAILABLE:
            # Use POT for Wasserstein
            dist = ot.emd2_1d(dist1, dist2)
        else:
            # Use scipy.stats
            dist = stats.wasserstein_distance(dist1, dist2)

    elif distance_metric == "ks":
        # Kolmogorov-Smirnov test
        ks_stat, _ = stats.ks_2samp(dist1, dist2)
        dist = ks_stat

    elif distance_metric == "kl":
        # Kullback-Leibler divergence
        # Assume inputs are histograms
        hist1 = dist1 / (dist1.sum() + 1e-10)
        hist2 = dist2 / (dist2.sum() + 1e-10)
        hist1 = hist1 + 1e-10  # Avoid log(0)
        hist2 = hist2 + 1e-10
        dist = np.sum(hist1 * np.log(hist1 / hist2))

    elif distance_metric == "js":
        # Jensen-Shannon divergence
        hist1 = dist1 / (dist1.sum() + 1e-10)
        hist2 = dist2 / (dist2.sum() + 1e-10)
        hist1 = hist1 + 1e-10
        hist2 = hist2 + 1e-10
        M = 0.5 * (hist1 + hist2)
        dist = 0.5 * np.sum(hist1 * np.log(hist1 / M)) + 0.5 * np.sum(hist2 * np.log(hist2 / M))

    elif distance_metric == "chisquare":
        # Chi-square statistic
        hist1 = dist1 + 1e-10
        hist2 = dist2 + 1e-10
        chi2_stat, _ = stats.chisquare(hist1, f_exp=hist2)
        dist = chi2_stat

    elif distance_metric == "l2":
        # L2 distance
        dist = np.linalg.norm(dist1 - dist2)

    else:
        raise ValueError(f"Unknown distance_metric: {distance_metric}")

    return float(dist)


def compute_degree_distribution_distance(
    graph1: nx.Graph,
    graph2: nx.Graph,
    distance_metric: str = "wasserstein",
    use_histogram: bool = False,
    bins: int = 20,
) -> float:
    """
    Compute distance between degree distributions.

    Parameters
    ----------
    graph1 : nx.Graph
        First graph.
    graph2 : nx.Graph
        Second graph.
    distance_metric : str, default="wasserstein"
        Distance metric (see compute_distribution_distance).
    use_histogram : bool, default=False
        If True, convert to histograms before comparison.
        If False, use raw degree sequences (for "wasserstein" and "ks").
    bins : int, default=20
        Number of bins for histogram.

    Returns
    -------
    distance : float
        Degree distribution distance.

    Notes
    -----
    - Degree sequence: list of node degrees
    - Histogram: binned degree counts
    - Wasserstein and KS work well with raw sequences
    - KL, JS, chi-square require histograms
    """
    logger.debug("Computing degree distribution distance...")

    # Extract degree sequences
    degrees1 = np.array([d for n, d in graph1.degree()])
    degrees2 = np.array([d for n, d in graph2.degree()])

    if len(degrees1) == 0 or len(degrees2) == 0:
        logger.warning("Empty degree sequence, returning inf")
        return np.inf

    # Convert to histograms if needed
    if use_histogram or distance_metric in {"kl", "js", "chisquare", "l2"}:
        max_degree = max(degrees1.max(), degrees2.max())
        bin_edges = np.arange(0, max_degree + 2)

        hist1, _ = np.histogram(degrees1, bins=bin_edges)
        hist2, _ = np.histogram(degrees2, bins=bin_edges)

        hist1 = hist1.astype(float)
        hist2 = hist2.astype(float)

        dist = compute_distribution_distance(hist1, hist2, distance_metric)
    else:
        dist = compute_distribution_distance(degrees1, degrees2, distance_metric)

    logger.debug(f"Degree distribution distance ({distance_metric}): {dist:.6f}")
    return dist


def compute_clustering_distribution_distance(
    graph1: nx.Graph,
    graph2: nx.Graph,
    distance_metric: str = "wasserstein",
    use_histogram: bool = False,
    bins: int = 20,
) -> float:
    """
    Compute distance between clustering coefficient distributions.

    Parameters
    ----------
    graph1 : nx.Graph
        First graph.
    graph2 : nx.Graph
        Second graph.
    distance_metric : str, default="wasserstein"
        Distance metric.
    use_histogram : bool, default=False
        Whether to use histograms.
    bins : int, default=20
        Number of bins for histogram.

    Returns
    -------
    distance : float
        Clustering coefficient distribution distance.

    Notes
    -----
    - Clustering coefficients in [0, 1]
    - For graphs with isolated nodes, clustering may be undefined
    """
    logger.debug("Computing clustering distribution distance...")

    # Extract clustering coefficients
    clustering1 = np.array(list(nx.clustering(graph1).values()))
    clustering2 = np.array(list(nx.clustering(graph2).values()))

    if len(clustering1) == 0 or len(clustering2) == 0:
        logger.warning("Empty clustering sequence, returning inf")
        return np.inf

    # Convert to histograms if needed
    if use_histogram or distance_metric in {"kl", "js", "chisquare", "l2"}:
        bin_edges = np.linspace(0, 1, bins + 1)

        hist1, _ = np.histogram(clustering1, bins=bin_edges)
        hist2, _ = np.histogram(clustering2, bins=bin_edges)

        hist1 = hist1.astype(float)
        hist2 = hist2.astype(float)

        dist = compute_distribution_distance(hist1, hist2, distance_metric)
    else:
        dist = compute_distribution_distance(clustering1, clustering2, distance_metric)

    logger.debug(f"Clustering distribution distance ({distance_metric}): {dist:.6f}")
    return dist


def compute_path_length_distribution_distance(
    graph1: nx.Graph,
    graph2: nx.Graph,
    distance_metric: str = "wasserstein",
    use_histogram: bool = False,
    bins: int = 20,
    sample_size: Optional[int] = 1000,
) -> float:
    """
    Compute distance between shortest path length distributions.

    Parameters
    ----------
    graph1 : nx.Graph
        First graph.
    graph2 : nx.Graph
        Second graph.
    distance_metric : str, default="wasserstein"
        Distance metric.
    use_histogram : bool, default=False
        Whether to use histograms.
    bins : int, default=20
        Number of bins for histogram.
    sample_size : int, optional
        Number of node pairs to sample for path length estimation.
        If None, computes all pairs (expensive for large graphs).

    Returns
    -------
    distance : float
        Path length distribution distance.

    Notes
    -----
    - For disconnected graphs, unreachable pairs are excluded
    - Sampling recommended for large graphs (n > 1000)
    - Path lengths are integers >= 1
    """
    logger.debug("Computing path length distribution distance...")

    def sample_path_lengths(graph, sample_size):
        """Sample shortest path lengths from graph."""
        nodes = list(graph.nodes())
        n_nodes = len(nodes)

        if n_nodes < 2:
            return np.array([])

        if sample_size is None or sample_size >= n_nodes * (n_nodes - 1):
            # Compute all pairs
            lengths = []
            for source in nodes:
                for target, length in nx.single_source_shortest_path_length(graph, source).items():
                    if source != target:
                        lengths.append(length)
            return np.array(lengths)
        else:
            # Sample pairs
            rng = np.random.RandomState(42)
            lengths = []

            for _ in range(sample_size):
                source, target = rng.choice(nodes, size=2, replace=False)
                try:
                    length = nx.shortest_path_length(graph, source, target)
                    lengths.append(length)
                except nx.NetworkXNoPath:
                    # Disconnected - skip this pair
                    pass

            return np.array(lengths)

    # Sample path lengths
    paths1 = sample_path_lengths(graph1, sample_size)
    paths2 = sample_path_lengths(graph2, sample_size)

    if len(paths1) == 0 or len(paths2) == 0:
        logger.warning("Empty path length sequence, returning inf")
        return np.inf

    # Convert to histograms if needed
    if use_histogram or distance_metric in {"kl", "js", "chisquare", "l2"}:
        max_path = max(paths1.max(), paths2.max())
        bin_edges = np.arange(1, max_path + 2)

        hist1, _ = np.histogram(paths1, bins=bin_edges)
        hist2, _ = np.histogram(paths2, bins=bin_edges)

        hist1 = hist1.astype(float)
        hist2 = hist2.astype(float)

        dist = compute_distribution_distance(hist1, hist2, distance_metric)
    else:
        dist = compute_distribution_distance(paths1, paths2, distance_metric)

    logger.debug(f"Path length distribution distance ({distance_metric}): {dist:.6f}")
    return dist


def compute_combined_distribution_distance(
    graph1: nx.Graph,
    graph2: nx.Graph,
    distance_metric: str = "wasserstein",
    weights: Optional[dict] = None,
) -> float:
    """
    Compute combined distance over multiple graph distributions.

    Parameters
    ----------
    graph1 : nx.Graph
        First graph.
    graph2 : nx.Graph
        Second graph.
    distance_metric : str, default="wasserstein"
        Distance metric.
    weights : dict, optional
        Weights for each distribution type.
        Default: {"degree": 1.0, "clustering": 1.0, "path_length": 1.0}

    Returns
    -------
    combined_distance : float
        Weighted sum of distribution distances.

    Notes
    -----
    - Combines degree, clustering, and path length distributions
    - Weights allow emphasizing specific properties
    """
    if weights is None:
        weights = {"degree": 1.0, "clustering": 1.0, "path_length": 1.0}

    distances = {}

    if "degree" in weights:
        distances["degree"] = compute_degree_distribution_distance(
            graph1, graph2, distance_metric
        )

    if "clustering" in weights:
        distances["clustering"] = compute_clustering_distribution_distance(
            graph1, graph2, distance_metric
        )

    if "path_length" in weights:
        distances["path_length"] = compute_path_length_distribution_distance(
            graph1, graph2, distance_metric, sample_size=500
        )

    # Weighted sum
    combined_distance = sum(weights[key] * distances[key] for key in distances)

    logger.info(f"Combined distribution distance: {combined_distance:.6f}")
    logger.info(f"Components: {distances}")

    return combined_distance


def compute_pairwise_distribution_matrix(
    graphs: List[nx.Graph],
    distribution_type: str = "degree",
    distance_metric: str = "wasserstein",
) -> np.ndarray:
    """
    Compute pairwise distribution distance matrix.

    Parameters
    ----------
    graphs : list of nx.Graph
        List of graphs.
    distribution_type : str, default="degree"
        Type of distribution: "degree", "clustering", "path_length", or "combined".
    distance_metric : str, default="wasserstein"
        Distance metric.

    Returns
    -------
    dist_matrix : np.ndarray
        Pairwise distribution distance matrix (n_graphs, n_graphs).

    Notes
    -----
    - Matrix is symmetric
    - Diagonal is zero
    """
    n_graphs = len(graphs)
    dist_matrix = np.zeros((n_graphs, n_graphs))

    logger.info(f"Computing pairwise {distribution_type} distribution matrix for {n_graphs} graphs...")

    # Select distance function
    if distribution_type == "degree":
        distance_func = compute_degree_distribution_distance
    elif distribution_type == "clustering":
        distance_func = compute_clustering_distribution_distance
    elif distribution_type == "path_length":
        distance_func = compute_path_length_distribution_distance
    elif distribution_type == "combined":
        distance_func = compute_combined_distribution_distance
    else:
        raise ValueError(f"Unknown distribution_type: {distribution_type}")

    # Compute upper triangle
    total_pairs = n_graphs * (n_graphs - 1) // 2
    pair_count = 0

    for i in range(n_graphs):
        for j in range(i + 1, n_graphs):
            dist = distance_func(graphs[i], graphs[j], distance_metric)
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist

            pair_count += 1
            if pair_count % 10 == 0:
                logger.info(f"Processed {pair_count}/{total_pairs} pairs...")

    logger.info("Pairwise distribution matrix computed")
    return dist_matrix


# Example usage
if __name__ == "__main__":
    from app.core.logging import setup_logging

    setup_logging()

    # Example: Compute distribution distances
    logger.info("=== Example: Distribution Distances ===")

    # Create test graphs
    G1 = nx.karate_club_graph()
    G2 = nx.erdos_renyi_graph(34, 0.1, seed=42)
    G3 = nx.barabasi_albert_graph(34, 3, seed=42)

    logger.info(f"Graph 1 (Karate Club): {len(G1)} nodes, {len(G1.edges())} edges")
    logger.info(f"Graph 2 (ER Random): {len(G2)} nodes, {len(G2.edges())} edges")
    logger.info(f"Graph 3 (BA Scale-free): {len(G3)} nodes, {len(G3.edges())} edges")

    # Degree distribution distances
    print("\n=== Degree Distribution Distances (Wasserstein) ===")
    d12_deg = compute_degree_distribution_distance(G1, G2, distance_metric="wasserstein")
    d13_deg = compute_degree_distribution_distance(G1, G3, distance_metric="wasserstein")
    d23_deg = compute_degree_distribution_distance(G2, G3, distance_metric="wasserstein")

    print(f"d(G1, G2) = {d12_deg:.6f}")
    print(f"d(G1, G3) = {d13_deg:.6f}")
    print(f"d(G2, G3) = {d23_deg:.6f}")

    # Clustering distribution distances
    print("\n=== Clustering Distribution Distances (Wasserstein) ===")
    d12_clust = compute_clustering_distribution_distance(G1, G2, distance_metric="wasserstein")
    d13_clust = compute_clustering_distribution_distance(G1, G3, distance_metric="wasserstein")
    d23_clust = compute_clustering_distribution_distance(G2, G3, distance_metric="wasserstein")

    print(f"d(G1, G2) = {d12_clust:.6f}")
    print(f"d(G1, G3) = {d13_clust:.6f}")
    print(f"d(G2, G3) = {d23_clust:.6f}")

    # Path length distribution distances
    print("\n=== Path Length Distribution Distances (Wasserstein) ===")
    d12_path = compute_path_length_distribution_distance(G1, G2, distance_metric="wasserstein", sample_size=500)
    d13_path = compute_path_length_distribution_distance(G1, G3, distance_metric="wasserstein", sample_size=500)
    d23_path = compute_path_length_distribution_distance(G2, G3, distance_metric="wasserstein", sample_size=500)

    print(f"d(G1, G2) = {d12_path:.6f}")
    print(f"d(G1, G3) = {d13_path:.6f}")
    print(f"d(G2, G3) = {d23_path:.6f}")

    # Combined distribution distance
    print("\n=== Combined Distribution Distance ===")
    d12_combined = compute_combined_distribution_distance(G1, G2, distance_metric="wasserstein")
    d13_combined = compute_combined_distribution_distance(G1, G3, distance_metric="wasserstein")
    d23_combined = compute_combined_distribution_distance(G2, G3, distance_metric="wasserstein")

    print(f"d(G1, G2) = {d12_combined:.6f}")
    print(f"d(G1, G3) = {d13_combined:.6f}")
    print(f"d(G2, G3) = {d23_combined:.6f}")

    # Pairwise matrix
    print("\n=== Pairwise Degree Distribution Matrix ===")
    graphs = [G1, G2, G3]
    dist_matrix = compute_pairwise_distribution_matrix(
        graphs, distribution_type="degree", distance_metric="wasserstein"
    )

    print("Pairwise Degree Distribution Distance Matrix:")
    print(dist_matrix)
