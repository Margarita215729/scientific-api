"""
Base graph utilities and normalization functions.

This module provides common functionality for working with graphs
in both cosmological and quantum domains.
"""

from typing import Dict, Optional, Tuple

import networkx as nx
import numpy as np

from app.core.logging import get_logger

logger = get_logger(__name__)


def normalize_coordinates(
    coords: np.ndarray,
    method: str = "unit_cube",
) -> np.ndarray:
    """
    Normalize node coordinates to a standard space.

    Args:
        coords: Array of coordinates (shape: [N, D] where N is number of nodes, D is dimension).
        method: Normalization method:
            - 'unit_cube': Scale to [0, 1]^D
            - 'unit_sphere': Scale to unit sphere
            - 'standardize': Zero mean, unit variance per dimension

    Returns:
        Normalized coordinates.
    """
    coords = np.asarray(coords)

    if method == "unit_cube":
        # Scale to [0, 1] in each dimension
        min_vals = coords.min(axis=0)
        max_vals = coords.max(axis=0)
        range_vals = max_vals - min_vals

        # Avoid division by zero
        range_vals[range_vals == 0] = 1.0

        normalized = (coords - min_vals) / range_vals

        logger.debug(
            f"Normalized {len(coords)} coordinates to unit cube "
            f"(original range: [{min_vals}, {max_vals}])"
        )

    elif method == "unit_sphere":
        # Center at origin
        centered = coords - coords.mean(axis=0)

        # Scale to fit in unit sphere
        max_dist = np.linalg.norm(centered, axis=1).max()
        if max_dist > 0:
            normalized = centered / max_dist
        else:
            normalized = centered

        logger.debug(
            f"Normalized {len(coords)} coordinates to unit sphere "
            f"(max distance from origin: {max_dist:.4f})"
        )

    elif method == "standardize":
        # Zero mean, unit variance
        mean_vals = coords.mean(axis=0)
        std_vals = coords.std(axis=0)

        # Avoid division by zero
        std_vals[std_vals == 0] = 1.0

        normalized = (coords - mean_vals) / std_vals

        logger.debug(
            f"Standardized {len(coords)} coordinates (zero mean, unit variance)"
        )

    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return normalized


def normalize_edge_weights(
    graph: nx.Graph,
    method: str = "minmax",
    weight_attr: str = "weight",
) -> nx.Graph:
    """
    Normalize edge weights in a graph.

    Args:
        graph: NetworkX graph.
        method: Normalization method:
            - 'minmax': Scale to [0, 1]
            - 'standardize': Zero mean, unit variance
            - 'max': Scale by maximum weight
        weight_attr: Name of edge attribute containing weights.

    Returns:
        Graph with normalized edge weights (modified in place).
    """
    # Get all edge weights
    weights = np.array(
        [data.get(weight_attr, 1.0) for _, _, data in graph.edges(data=True)]
    )

    if len(weights) == 0:
        logger.warning("Graph has no edges, skipping weight normalization")
        return graph

    if method == "minmax":
        min_weight = weights.min()
        max_weight = weights.max()
        weight_range = max_weight - min_weight

        if weight_range > 0:
            for u, v, data in graph.edges(data=True):
                if weight_attr in data:
                    data[weight_attr] = (data[weight_attr] - min_weight) / weight_range

            logger.debug(
                f"Normalized {len(weights)} edge weights using minmax "
                f"(original range: [{min_weight:.4f}, {max_weight:.4f}])"
            )

    elif method == "standardize":
        mean_weight = weights.mean()
        std_weight = weights.std()

        if std_weight > 0:
            for u, v, data in graph.edges(data=True):
                if weight_attr in data:
                    data[weight_attr] = (data[weight_attr] - mean_weight) / std_weight

            logger.debug(
                f"Standardized {len(weights)} edge weights "
                f"(mean={mean_weight:.4f}, std={std_weight:.4f})"
            )

    elif method == "max":
        max_weight = weights.max()

        if max_weight > 0:
            for u, v, data in graph.edges(data=True):
                if weight_attr in data:
                    data[weight_attr] = data[weight_attr] / max_weight

            logger.debug(
                f"Normalized {len(weights)} edge weights by max ({max_weight:.4f})"
            )

    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return graph


def get_graph_info(graph: nx.Graph) -> Dict:
    """
    Get basic information about a graph.

    Args:
        graph: NetworkX graph.

    Returns:
        Dictionary with graph statistics.
    """
    n_nodes = graph.number_of_nodes()
    n_edges = graph.number_of_edges()

    info = {
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "density": nx.density(graph) if n_nodes > 0 else 0.0,
        "is_directed": graph.is_directed(),
    }

    # Check for node attributes
    if n_nodes > 0:
        node_attrs = set()
        for node, data in graph.nodes(data=True):
            node_attrs.update(data.keys())
            break
        info["node_attributes"] = list(node_attrs)

    # Check for edge attributes
    if n_edges > 0:
        edge_attrs = set()
        for u, v, data in graph.edges(data=True):
            edge_attrs.update(data.keys())
            break
        info["edge_attributes"] = list(edge_attrs)

    # Connected components
    if not graph.is_directed():
        info["n_components"] = nx.number_connected_components(graph)
        if info["n_components"] > 0:
            largest_cc = max(nx.connected_components(graph), key=len)
            info["largest_component_size"] = len(largest_cc)

    return info


def extract_coordinates_from_graph(
    graph: nx.Graph,
    coord_attrs: Tuple[str, ...] = ("x", "y", "z"),
) -> Optional[np.ndarray]:
    """
    Extract node coordinates from graph node attributes.

    Args:
        graph: NetworkX graph with coordinate attributes.
        coord_attrs: Tuple of attribute names for coordinates.

    Returns:
        Array of coordinates (shape: [N, D]) or None if not all nodes have coordinates.
    """
    coords_list = []

    for node in graph.nodes():
        node_data = graph.nodes[node]

        # Check if all coordinate attributes exist
        if all(attr in node_data for attr in coord_attrs):
            coords = [node_data[attr] for attr in coord_attrs]
            coords_list.append(coords)
        else:
            logger.warning(
                f"Node {node} missing coordinate attributes, "
                f"expected {coord_attrs}, found {list(node_data.keys())}"
            )
            return None

    if coords_list:
        return np.array(coords_list)
    else:
        return None


def add_coordinates_to_graph(
    graph: nx.Graph,
    coords: np.ndarray,
    coord_attrs: Tuple[str, ...] = ("x", "y", "z"),
) -> nx.Graph:
    """
    Add coordinate attributes to graph nodes.

    Args:
        graph: NetworkX graph.
        coords: Array of coordinates (shape: [N, D]).
        coord_attrs: Tuple of attribute names for coordinates.

    Returns:
        Graph with added coordinate attributes (modified in place).

    Raises:
        ValueError: If number of nodes doesn't match coordinate array size.
    """
    if len(coords) != graph.number_of_nodes():
        raise ValueError(
            f"Coordinate array size ({len(coords)}) doesn't match "
            f"number of nodes ({graph.number_of_nodes()})"
        )

    if coords.shape[1] != len(coord_attrs):
        raise ValueError(
            f"Coordinate dimension ({coords.shape[1]}) doesn't match "
            f"number of coordinate attributes ({len(coord_attrs)})"
        )

    for idx, node in enumerate(graph.nodes()):
        for dim, attr in enumerate(coord_attrs):
            graph.nodes[node][attr] = float(coords[idx, dim])

    logger.debug(
        f"Added {len(coord_attrs)}-dimensional coordinates to {len(coords)} nodes"
    )

    return graph


def create_graph_from_edges(
    edges: np.ndarray,
    weights: Optional[np.ndarray] = None,
    node_attributes: Optional[Dict] = None,
) -> nx.Graph:
    """
    Create NetworkX graph from edge list.

    Args:
        edges: Array of edges (shape: [M, 2]).
        weights: Optional array of edge weights (shape: [M]).
        node_attributes: Optional dictionary mapping node IDs to attribute dictionaries.

    Returns:
        NetworkX graph.
    """
    graph = nx.Graph()

    # Add edges
    if weights is not None:
        edge_list = [
            (int(u), int(v), {"weight": float(w)}) for (u, v), w in zip(edges, weights)
        ]
    else:
        edge_list = [(int(u), int(v)) for u, v in edges]

    graph.add_edges_from(edge_list)

    # Add node attributes
    if node_attributes:
        for node, attrs in node_attributes.items():
            if node in graph:
                graph.nodes[node].update(attrs)

    logger.info(
        f"Created graph: {graph.number_of_nodes()} nodes, "
        f"{graph.number_of_edges()} edges"
    )

    return graph
