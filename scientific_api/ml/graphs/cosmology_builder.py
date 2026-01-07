"""
Graph construction from cosmological data.

This module provides functions to build graphs from galaxy catalogs
using k-nearest neighbors in 3D coordinate space.
"""

from pathlib import Path
from typing import Optional, Union

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from app.core.logging import get_logger
from scientific_api.ml.graphs.base import (
    add_coordinates_to_graph,
    create_graph_from_edges,
    normalize_coordinates,
    normalize_edge_weights,
)

logger = get_logger(__name__)


def build_knn_graph(
    coords: np.ndarray,
    k: int = 5,
    metric: str = "euclidean",
    include_self: bool = False,
) -> nx.Graph:
    """
    Build k-nearest neighbors graph from coordinates.

    Args:
        coords: Array of coordinates (shape: [N, D]).
        k: Number of nearest neighbors.
        metric: Distance metric ('euclidean', 'manhattan', 'minkowski', etc.).
        include_self: Whether to include self-loops.

    Returns:
        NetworkX graph with edges to k nearest neighbors.
    """
    n_points = len(coords)

    if k >= n_points:
        logger.warning(
            f"k={k} is >= number of points ({n_points}), "
            f"reducing to k={n_points - 1}"
        )
        k = n_points - 1

    logger.info(f"Building k-NN graph: {n_points} points, k={k}, metric={metric}")

    # Fit k-NN
    nbrs = NearestNeighbors(
        n_neighbors=k + 1 if not include_self else k,
        metric=metric,
        algorithm="auto",
    )
    nbrs.fit(coords)

    # Find k nearest neighbors
    distances, indices = nbrs.kneighbors(coords)

    # Build edge list
    edges = []
    weights = []

    for i in range(n_points):
        # Skip first neighbor if not including self (it's the point itself)
        start_idx = 0 if include_self else 1

        for j in range(start_idx, len(indices[i])):
            neighbor_idx = indices[i][j]
            distance = distances[i][j]

            # Add edge (undirected, so only add if i < neighbor to avoid duplicates)
            if i < neighbor_idx:
                edges.append([i, neighbor_idx])
                weights.append(distance)

    edges = np.array(edges)
    weights = np.array(weights)

    logger.info(
        f"Built k-NN graph: {len(edges)} edges, "
        f"distance range: [{weights.min():.4f}, {weights.max():.4f}]"
    )

    # Create graph
    graph = create_graph_from_edges(edges, weights)

    return graph


def build_cosmology_graph(
    df: pd.DataFrame,
    k: int = 5,
    coord_columns: tuple = ("x", "y", "z"),
    metric: str = "euclidean",
    normalize_coords: bool = True,
    normalize_weights: bool = True,
    additional_attributes: Optional[list] = None,
) -> nx.Graph:
    """
    Build graph from cosmological catalog DataFrame.

    Args:
        df: DataFrame with galaxy data.
        k: Number of nearest neighbors.
        coord_columns: Column names for coordinates.
        metric: Distance metric for k-NN.
        normalize_coords: Whether to normalize coordinates to unit cube.
        normalize_weights: Whether to normalize edge weights.
        additional_attributes: Additional node attributes to include from DataFrame.

    Returns:
        NetworkX graph representing galaxy catalog.
    """
    logger.info(
        f"Building cosmology graph from {len(df)} objects "
        f"(k={k}, coords={coord_columns})"
    )

    # Extract coordinates
    coords = df[list(coord_columns)].values

    # Normalize coordinates if requested
    if normalize_coords:
        coords = normalize_coordinates(coords, method="unit_cube")

    # Build k-NN graph
    graph = build_knn_graph(coords, k=k, metric=metric)

    # Add normalized coordinates to graph
    add_coordinates_to_graph(graph, coords, coord_attrs=coord_columns)

    # Add additional node attributes
    if additional_attributes:
        for idx, node in enumerate(graph.nodes()):
            for attr in additional_attributes:
                if attr in df.columns:
                    graph.nodes[node][attr] = df.iloc[idx][attr]

    # Add standard attributes
    for idx, node in enumerate(graph.nodes()):
        graph.nodes[node]["node_id"] = idx
        graph.nodes[node]["system_type"] = "cosmology"

    # Normalize edge weights if requested
    if normalize_weights:
        normalize_edge_weights(graph, method="minmax")

    logger.info(
        f"Cosmology graph created: {graph.number_of_nodes()} nodes, "
        f"{graph.number_of_edges()} edges, "
        f"density={nx.density(graph):.6f}"
    )

    return graph


def save_graph(
    graph: nx.Graph,
    output_path: Union[str, Path],
    format: str = "graphml",
) -> None:
    """
    Save graph to disk.

    Args:
        graph: NetworkX graph.
        output_path: Path where to save the graph.
        format: Output format ('graphml', 'gexf', 'edgelist', 'adjlist').

    Raises:
        ValueError: If format is not supported.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving graph to {output_path} (format: {format})")

    if format == "graphml":
        nx.write_graphml(graph, output_path)
    elif format == "gexf":
        nx.write_gexf(graph, output_path)
    elif format == "edgelist":
        nx.write_edgelist(graph, output_path, data=True)
    elif format == "adjlist":
        nx.write_adjlist(graph, output_path)
    else:
        raise ValueError(f"Unsupported format: {format}")

    logger.info("Graph saved successfully")


def load_graph(
    input_path: Union[str, Path],
    format: str = "graphml",
) -> nx.Graph:
    """
    Load graph from disk.

    Args:
        input_path: Path to saved graph.
        format: Input format ('graphml', 'gexf', 'edgelist', 'adjlist').

    Returns:
        NetworkX graph.

    Raises:
        ValueError: If format is not supported.
    """
    input_path = Path(input_path)

    logger.info(f"Loading graph from {input_path} (format: {format})")

    if format == "graphml":
        graph = nx.read_graphml(input_path, node_type=int)
    elif format == "gexf":
        graph = nx.read_gexf(input_path, node_type=int)
    elif format == "edgelist":
        graph = nx.read_edgelist(input_path, data=True)
    elif format == "adjlist":
        graph = nx.read_adjlist(input_path)
    else:
        raise ValueError(f"Unsupported format: {format}")

    logger.info(
        f"Graph loaded: {graph.number_of_nodes()} nodes, "
        f"{graph.number_of_edges()} edges"
    )

    return graph


def build_and_save_cosmology_graph(
    df: pd.DataFrame,
    output_path: Union[str, Path],
    k: int = 5,
    coord_columns: tuple = ("x", "y", "z"),
    normalize_coords: bool = True,
    normalize_weights: bool = True,
    additional_attributes: Optional[list] = None,
    format: str = "graphml",
) -> nx.Graph:
    """
    Build cosmology graph from DataFrame and save to disk.

    Args:
        df: DataFrame with galaxy data.
        output_path: Path where to save the graph.
        k: Number of nearest neighbors.
        coord_columns: Column names for coordinates.
        normalize_coords: Whether to normalize coordinates.
        normalize_weights: Whether to normalize edge weights.
        additional_attributes: Additional node attributes to include.
        format: Output format.

    Returns:
        Created NetworkX graph.
    """
    # Build graph
    graph = build_cosmology_graph(
        df,
        k=k,
        coord_columns=coord_columns,
        normalize_coords=normalize_coords,
        normalize_weights=normalize_weights,
        additional_attributes=additional_attributes,
    )

    # Save graph
    save_graph(graph, output_path, format=format)

    return graph
