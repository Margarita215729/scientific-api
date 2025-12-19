"""
Graph consistency and utility functions.

This module provides utilities for graph downsampling, comparison,
and ensuring consistency across different graph types.
"""

from typing import List, Optional, Tuple

import networkx as nx
import numpy as np

from app.core.logging import get_logger

logger = get_logger(__name__)


def downsample_graph(
    graph: nx.Graph,
    target_nodes: int,
    method: str = "random",
    preserve_connectivity: bool = True,
    random_state: Optional[int] = None,
) -> nx.Graph:
    """
    Downsample graph to target number of nodes.

    Args:
        graph: Input NetworkX graph.
        target_nodes: Target number of nodes.
        method: Downsampling method:
            - 'random': Random node sampling
            - 'degree': Sample nodes with highest degrees
            - 'betweenness': Sample nodes with highest betweenness centrality
        preserve_connectivity: If True, ensure resulting graph is connected.
        random_state: Random seed for reproducibility.

    Returns:
        Downsampled graph.
    """
    n_nodes = graph.number_of_nodes()

    if target_nodes >= n_nodes:
        logger.info(f"Graph already has {n_nodes} nodes, no downsampling needed")
        return graph.copy()

    logger.info(
        f"Downsampling graph from {n_nodes} to {target_nodes} nodes "
        f"using {method} method"
    )

    # Select nodes based on method
    if method == "random":
        rng = np.random.default_rng(random_state)
        nodes_list = list(graph.nodes())
        selected_nodes = rng.choice(nodes_list, size=target_nodes, replace=False)

    elif method == "degree":
        # Sort nodes by degree
        degree_dict = dict(graph.degree())
        sorted_nodes = sorted(
            degree_dict.keys(), key=lambda n: degree_dict[n], reverse=True
        )
        selected_nodes = sorted_nodes[:target_nodes]

    elif method == "betweenness":
        # Compute betweenness centrality
        betweenness = nx.betweenness_centrality(graph)
        sorted_nodes = sorted(
            betweenness.keys(), key=lambda n: betweenness[n], reverse=True
        )
        selected_nodes = sorted_nodes[:target_nodes]

    else:
        raise ValueError(f"Unknown downsampling method: {method}")

    # Create subgraph
    subgraph = graph.subgraph(selected_nodes).copy()

    # Ensure connectivity if requested
    if preserve_connectivity and not nx.is_connected(subgraph):
        logger.warning(
            "Resulting subgraph is disconnected, extracting largest component"
        )
        components = list(nx.connected_components(subgraph))
        largest_component = max(components, key=len)
        subgraph = subgraph.subgraph(largest_component).copy()

    logger.info(
        f"Downsampled graph: {subgraph.number_of_nodes()} nodes, "
        f"{subgraph.number_of_edges()} edges"
    )

    return subgraph


def match_graph_sizes(
    graphs: List[nx.Graph],
    target_nodes: Optional[int] = None,
    method: str = "random",
    random_state: Optional[int] = None,
) -> List[nx.Graph]:
    """
    Downsample a list of graphs to have the same number of nodes.

    Args:
        graphs: List of NetworkX graphs.
        target_nodes: Target number of nodes. If None, use minimum across graphs.
        method: Downsampling method (see downsample_graph).
        random_state: Random seed for reproducibility.

    Returns:
        List of graphs with matched sizes.
    """
    if not graphs:
        return []

    # Find target size
    if target_nodes is None:
        target_nodes = min(g.number_of_nodes() for g in graphs)

    logger.info(f"Matching {len(graphs)} graphs to {target_nodes} nodes each")

    matched_graphs = []
    for i, graph in enumerate(graphs):
        downsampled = downsample_graph(
            graph,
            target_nodes=target_nodes,
            method=method,
            random_state=random_state,
        )
        matched_graphs.append(downsampled)

    return matched_graphs


def scale_edge_weights_to_range(
    graph: nx.Graph,
    target_range: Tuple[float, float] = (0.0, 1.0),
    weight_attr: str = "weight",
) -> nx.Graph:
    """
    Scale edge weights to a target range.

    Args:
        graph: NetworkX graph.
        target_range: Tuple of (min, max) for target range.
        weight_attr: Name of edge weight attribute.

    Returns:
        Graph with scaled edge weights (modified in place).
    """
    weights = np.array(
        [data.get(weight_attr, 1.0) for _, _, data in graph.edges(data=True)]
    )

    if len(weights) == 0:
        logger.warning("Graph has no edges, skipping weight scaling")
        return graph

    min_weight = weights.min()
    max_weight = weights.max()
    weight_range = max_weight - min_weight

    if weight_range == 0:
        logger.warning("All edge weights are equal, setting to middle of target range")
        target_value = (target_range[0] + target_range[1]) / 2
        for _, _, data in graph.edges(data=True):
            if weight_attr in data:
                data[weight_attr] = target_value
        return graph

    target_min, target_max = target_range
    target_span = target_max - target_min

    for _, _, data in graph.edges(data=True):
        if weight_attr in data:
            # Scale to [0, 1] then to target range
            normalized = (data[weight_attr] - min_weight) / weight_range
            data[weight_attr] = target_min + normalized * target_span

    logger.debug(
        f"Scaled {len(weights)} edge weights from [{min_weight:.4f}, {max_weight:.4f}] "
        f"to [{target_range[0]:.4f}, {target_range[1]:.4f}]"
    )

    return graph


def get_graph_statistics(graph: nx.Graph) -> dict:
    """
    Compute comprehensive statistics for a graph.

    Args:
        graph: NetworkX graph.

    Returns:
        Dictionary with graph statistics.
    """
    n_nodes = graph.number_of_nodes()
    n_edges = graph.number_of_edges()

    stats = {
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "density": nx.density(graph),
    }

    if n_nodes > 0:
        # Degree statistics
        degrees = [d for _, d in graph.degree()]
        stats["degree_mean"] = np.mean(degrees)
        stats["degree_std"] = np.std(degrees)
        stats["degree_min"] = np.min(degrees)
        stats["degree_max"] = np.max(degrees)

        # Edge weight statistics (if weights exist)
        weights = [data.get("weight", 1.0) for _, _, data in graph.edges(data=True)]
        if weights:
            stats["weight_mean"] = np.mean(weights)
            stats["weight_std"] = np.std(weights)
            stats["weight_min"] = np.min(weights)
            stats["weight_max"] = np.max(weights)

    # Connected components
    if not graph.is_directed():
        components = list(nx.connected_components(graph))
        stats["n_components"] = len(components)
        if components:
            stats["largest_component_size"] = len(max(components, key=len))
            stats["largest_component_fraction"] = (
                stats["largest_component_size"] / n_nodes
            )

        # Clustering coefficient (for connected graphs)
        if nx.is_connected(graph) and n_nodes > 2:
            stats["avg_clustering"] = nx.average_clustering(graph)

    return stats


def compare_graph_properties(
    graph1: nx.Graph,
    graph2: nx.Graph,
) -> dict:
    """
    Compare properties of two graphs.

    Args:
        graph1: First NetworkX graph.
        graph2: Second NetworkX graph.

    Returns:
        Dictionary with comparison metrics.
    """
    stats1 = get_graph_statistics(graph1)
    stats2 = get_graph_statistics(graph2)

    comparison = {
        "graph1_nodes": stats1["n_nodes"],
        "graph2_nodes": stats2["n_nodes"],
        "graph1_edges": stats1["n_edges"],
        "graph2_edges": stats2["n_edges"],
        "density_diff": abs(stats1["density"] - stats2["density"]),
    }

    # Degree distribution comparison
    if stats1["n_nodes"] > 0 and stats2["n_nodes"] > 0:
        comparison["degree_mean_diff"] = abs(
            stats1["degree_mean"] - stats2["degree_mean"]
        )

    return comparison


def ensure_graph_consistency(
    graphs: List[nx.Graph],
    target_nodes: Optional[int] = None,
    normalize_weights: bool = True,
    weight_range: Tuple[float, float] = (0.0, 1.0),
) -> List[nx.Graph]:
    """
    Ensure consistency across multiple graphs.

    This function:
    1. Matches graph sizes
    2. Normalizes edge weights to same range
    3. Ensures all graphs have required attributes

    Args:
        graphs: List of NetworkX graphs.
        target_nodes: Target number of nodes for all graphs.
        normalize_weights: Whether to normalize edge weights.
        weight_range: Target range for edge weights.

    Returns:
        List of consistent graphs.
    """
    logger.info(f"Ensuring consistency across {len(graphs)} graphs")

    # Match sizes
    consistent_graphs = match_graph_sizes(graphs, target_nodes=target_nodes)

    # Normalize weights
    if normalize_weights:
        for graph in consistent_graphs:
            scale_edge_weights_to_range(graph, target_range=weight_range)

    # Verify consistency
    sizes = [g.number_of_nodes() for g in consistent_graphs]
    logger.info(
        f"Graph consistency ensured: node counts = {sizes}, "
        f"weight range = {weight_range}"
    )

    return consistent_graphs


def extract_largest_component(graph: nx.Graph) -> nx.Graph:
    """
    Extract the largest connected component from a graph.

    Args:
        graph: NetworkX graph.

    Returns:
        Largest connected component as a new graph.
    """
    if nx.is_connected(graph):
        logger.debug("Graph is already connected")
        return graph.copy()

    components = list(nx.connected_components(graph))
    largest = max(components, key=len)

    logger.info(
        f"Extracted largest component: {len(largest)} nodes "
        f"out of {graph.number_of_nodes()} total "
        f"({len(largest) / graph.number_of_nodes() * 100:.1f}%)"
    )

    return graph.subgraph(largest).copy()


def relabel_nodes_sequential(graph: nx.Graph) -> nx.Graph:
    """
    Relabel graph nodes to sequential integers starting from 0.

    Args:
        graph: NetworkX graph.

    Returns:
        Graph with relabeled nodes.
    """
    n_nodes = graph.number_of_nodes()
    mapping = {node: i for i, node in enumerate(graph.nodes())}

    relabeled = nx.relabel_nodes(graph, mapping, copy=True)

    logger.debug(f"Relabeled {n_nodes} nodes to sequential integers [0, {n_nodes-1}]")

    return relabeled
