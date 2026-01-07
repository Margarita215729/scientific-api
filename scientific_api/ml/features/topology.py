"""
Topological feature extraction for graphs.

This module computes topological features from graphs including
degree statistics, clustering coefficients, path lengths, and centrality measures.
"""

from typing import Dict, Optional

import networkx as nx
import numpy as np

from app.core.logging import get_logger

logger = get_logger(__name__)


def compute_degree_statistics(graph: nx.Graph) -> Dict[str, float]:
    """
    Compute degree distribution statistics.

    Args:
        graph: NetworkX graph.

    Returns:
        Dictionary with degree statistics.
    """
    degrees = [d for _, d in graph.degree()]

    if len(degrees) == 0:
        return {
            "degree_mean": 0.0,
            "degree_std": 0.0,
            "degree_min": 0.0,
            "degree_max": 0.0,
            "degree_median": 0.0,
        }

    stats = {
        "degree_mean": float(np.mean(degrees)),
        "degree_std": float(np.std(degrees)),
        "degree_min": float(np.min(degrees)),
        "degree_max": float(np.max(degrees)),
        "degree_median": float(np.median(degrees)),
    }

    logger.debug(f"Computed degree statistics: mean={stats['degree_mean']:.2f}")
    return stats


def compute_clustering_coefficient(graph: nx.Graph) -> Dict[str, float]:
    """
    Compute clustering coefficient statistics.

    Args:
        graph: NetworkX graph.

    Returns:
        Dictionary with clustering statistics.
    """
    if graph.number_of_nodes() < 3:
        return {
            "avg_clustering": 0.0,
            "clustering_std": 0.0,
        }

    clustering = nx.clustering(graph)
    clustering_values = list(clustering.values())

    stats = {
        "avg_clustering": float(np.mean(clustering_values)),
        "clustering_std": float(np.std(clustering_values)),
    }

    logger.debug(f"Computed clustering: avg={stats['avg_clustering']:.4f}")
    return stats


def compute_path_statistics(
    graph: nx.Graph,
    use_largest_component: bool = True,
    sample_size: Optional[int] = None,
) -> Dict[str, float]:
    """
    Compute shortest path statistics.

    For large graphs, samples a subset of node pairs to estimate average path length.

    Args:
        graph: NetworkX graph.
        use_largest_component: If True, compute on largest connected component.
        sample_size: If set, sample this many node pairs for path length estimation.

    Returns:
        Dictionary with path statistics.
    """
    if use_largest_component and not nx.is_connected(graph):
        components = list(nx.connected_components(graph))
        largest = max(components, key=len)
        subgraph = graph.subgraph(largest)
        logger.debug(
            f"Using largest component: {len(largest)}/{graph.number_of_nodes()} nodes"
        )
    else:
        subgraph = graph

    if subgraph.number_of_nodes() < 2:
        return {
            "avg_shortest_path": 0.0,
            "diameter": 0.0,
        }

    # For small graphs, compute exact values
    if subgraph.number_of_nodes() < 100 or sample_size is None:
        try:
            avg_path = nx.average_shortest_path_length(subgraph)
            diameter = nx.diameter(subgraph)
            logger.debug(f"Computed exact path statistics: avg={avg_path:.2f}, diameter={diameter}")
        except nx.NetworkXError as e:
            logger.warning(f"Could not compute path statistics: {e}")
            avg_path = 0.0
            diameter = 0.0
    else:
        # Sample node pairs for large graphs
        nodes = list(subgraph.nodes())
        rng = np.random.default_rng(42)
        sampled_pairs = rng.choice(len(nodes), size=(sample_size, 2), replace=True)

        path_lengths = []
        max_path = 0

        for i, j in sampled_pairs:
            if i != j:
                try:
                    length = nx.shortest_path_length(subgraph, nodes[i], nodes[j])
                    path_lengths.append(length)
                    max_path = max(max_path, length)
                except nx.NetworkXNoPath:
                    pass

        avg_path = float(np.mean(path_lengths)) if path_lengths else 0.0
        diameter = float(max_path)

        logger.debug(
            f"Computed sampled path statistics ({len(path_lengths)} pairs): "
            f"avg={avg_path:.2f}, max={diameter}"
        )

    return {
        "avg_shortest_path": avg_path,
        "diameter": diameter,
    }


def compute_centrality_statistics(
    graph: nx.Graph,
    sample_size: Optional[int] = 100,
) -> Dict[str, float]:
    """
    Compute centrality measure statistics.

    For large graphs, samples nodes to estimate centrality distribution.

    Args:
        graph: NetworkX graph.
        sample_size: Number of nodes to sample for centrality computation.

    Returns:
        Dictionary with centrality statistics.
    """
    n_nodes = graph.number_of_nodes()

    if n_nodes == 0:
        return {
            "betweenness_mean": 0.0,
            "betweenness_max": 0.0,
            "closeness_mean": 0.0,
            "closeness_max": 0.0,
        }

    # Sample nodes if graph is large
    if sample_size and n_nodes > sample_size:
        nodes = list(graph.nodes())
        rng = np.random.default_rng(42)
        sampled_nodes = rng.choice(nodes, size=sample_size, replace=False)
        k = sample_size
    else:
        sampled_nodes = None
        k = None

    # Betweenness centrality
    try:
        if sampled_nodes is not None:
            betweenness = nx.betweenness_centrality(graph, k=k, seed=42)
        else:
            betweenness = nx.betweenness_centrality(graph)

        betweenness_values = list(betweenness.values())
        betweenness_mean = float(np.mean(betweenness_values))
        betweenness_max = float(np.max(betweenness_values))
    except Exception as e:
        logger.warning(f"Could not compute betweenness centrality: {e}")
        betweenness_mean = 0.0
        betweenness_max = 0.0

    # Closeness centrality (only for connected graphs or largest component)
    try:
        if nx.is_connected(graph):
            closeness = nx.closeness_centrality(graph)
            closeness_values = list(closeness.values())
            closeness_mean = float(np.mean(closeness_values))
            closeness_max = float(np.max(closeness_values))
        else:
            # Use largest component
            components = list(nx.connected_components(graph))
            largest = max(components, key=len)
            subgraph = graph.subgraph(largest)
            closeness = nx.closeness_centrality(subgraph)
            closeness_values = list(closeness.values())
            closeness_mean = float(np.mean(closeness_values))
            closeness_max = float(np.max(closeness_values))
    except Exception as e:
        logger.warning(f"Could not compute closeness centrality: {e}")
        closeness_mean = 0.0
        closeness_max = 0.0

    logger.debug(
        f"Computed centrality: betweenness_mean={betweenness_mean:.4f}, "
        f"closeness_mean={closeness_mean:.4f}"
    )

    return {
        "betweenness_mean": betweenness_mean,
        "betweenness_max": betweenness_max,
        "closeness_mean": closeness_mean,
        "closeness_max": closeness_max,
    }


def compute_connectivity_statistics(graph: nx.Graph) -> Dict[str, float]:
    """
    Compute connectivity statistics.

    Args:
        graph: NetworkX graph.

    Returns:
        Dictionary with connectivity statistics.
    """
    n_nodes = graph.number_of_nodes()
    n_edges = graph.number_of_edges()

    stats = {
        "n_nodes": float(n_nodes),
        "n_edges": float(n_edges),
        "density": float(nx.density(graph)),
    }

    if not graph.is_directed():
        components = list(nx.connected_components(graph))
        stats["n_components"] = float(len(components))

        if components:
            largest_size = len(max(components, key=len))
            stats["largest_component_size"] = float(largest_size)
            stats["largest_component_fraction"] = (
                largest_size / n_nodes if n_nodes > 0 else 0.0
            )

    logger.debug(
        f"Computed connectivity: {n_nodes} nodes, {n_edges} edges, "
        f"density={stats['density']:.6f}"
    )

    return stats


def compute_topology_features(
    graph: nx.Graph,
    include_paths: bool = True,
    include_centrality: bool = True,
    centrality_sample_size: Optional[int] = 100,
    path_sample_size: Optional[int] = None,
) -> Dict[str, float]:
    """
    Compute all topological features for a graph.

    Args:
        graph: NetworkX graph.
        include_paths: Whether to compute path statistics (can be slow for large graphs).
        include_centrality: Whether to compute centrality measures.
        centrality_sample_size: Number of nodes to sample for centrality.
        path_sample_size: Number of node pairs to sample for path statistics.

    Returns:
        Dictionary with all topological features.
    """
    logger.info(
        f"Computing topological features for graph with "
        f"{graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges"
    )

    features = {}

    # Basic connectivity
    features.update(compute_connectivity_statistics(graph))

    # Degree statistics
    features.update(compute_degree_statistics(graph))

    # Clustering
    features.update(compute_clustering_coefficient(graph))

    # Path statistics
    if include_paths:
        features.update(
            compute_path_statistics(graph, sample_size=path_sample_size)
        )

    # Centrality measures
    if include_centrality:
        features.update(
            compute_centrality_statistics(graph, sample_size=centrality_sample_size)
        )

    logger.info(f"Computed {len(features)} topological features")

    return features
