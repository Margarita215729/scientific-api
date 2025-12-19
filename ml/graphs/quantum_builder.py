"""
Graph construction from quantum systems.

This module provides functions to build graphs from quantum system
Hamiltonians and grid coordinates.
"""

from pathlib import Path
from typing import Dict, Optional, Union

import networkx as nx
import numpy as np
from scipy import sparse

from app.core.logging import get_logger
from ml.graphs.base import (
    add_coordinates_to_graph,
    create_graph_from_edges,
    normalize_coordinates,
    normalize_edge_weights,
)

logger = get_logger(__name__)


def hamiltonian_to_graph(
    H: sparse.csr_matrix,
    coords: np.ndarray,
    threshold: Optional[float] = None,
    use_absolute: bool = True,
) -> nx.Graph:
    """
    Build graph from Hamiltonian matrix.

    Nodes correspond to grid points, edges correspond to non-zero
    Hamiltonian matrix elements (interactions).

    Args:
        H: Sparse Hamiltonian matrix (shape: [N, N]).
        coords: Node coordinates (shape: [N, D]).
        threshold: Optional threshold for edge inclusion. If set, only edges
                  with |H_ij| > threshold are included.
        use_absolute: If True, use absolute value of Hamiltonian elements for weights.

    Returns:
        NetworkX graph representing quantum system.
    """
    n_nodes = H.shape[0]

    if len(coords) != n_nodes:
        raise ValueError(
            f"Coordinate array size ({len(coords)}) doesn't match "
            f"Hamiltonian dimension ({n_nodes})"
        )

    logger.info(
        f"Converting Hamiltonian to graph: {n_nodes} nodes, "
        f"H.nnz={H.nnz}, sparsity={1 - H.nnz / (n_nodes**2):.6f}"
    )

    # Convert to COO format for easier iteration
    H_coo = H.tocoo()

    # Build edge list from non-zero entries
    edges = []
    weights = []

    for i, j, value in zip(H_coo.row, H_coo.col, H_coo.data):
        # Only add upper triangle (undirected graph)
        if i < j:
            weight = abs(value) if use_absolute else value

            # Apply threshold if specified
            if threshold is None or weight > threshold:
                edges.append([i, j])
                weights.append(weight)

    if len(edges) == 0:
        logger.warning("No edges passed threshold, creating empty graph")
        graph = nx.Graph()
        graph.add_nodes_from(range(n_nodes))
    else:
        edges = np.array(edges)
        weights = np.array(weights)

        logger.info(
            f"Creating graph from Hamiltonian: {len(edges)} edges, "
            f"weight range: [{weights.min():.6e}, {weights.max():.6e}]"
        )

        # Create graph
        graph = create_graph_from_edges(edges, weights)

    return graph


def build_quantum_graph(
    quantum_system: Dict,
    normalize_coords: bool = True,
    normalize_weights: bool = True,
    edge_threshold: Optional[float] = None,
    coord_scaling: str = "unit_cube",
) -> nx.Graph:
    """
    Build graph from quantum system dictionary.

    Args:
        quantum_system: Dictionary with quantum system data:
            - 'X': X coordinate grid (2D array)
            - 'Y': Y coordinate grid (2D array)
            - 'H': Hamiltonian matrix (sparse)
            - 'V': Potential values (2D array)
            - 'model_type': Model type string
            - 'parameters': Model parameters dict
        normalize_coords: Whether to normalize coordinates.
        normalize_weights: Whether to normalize edge weights.
        edge_threshold: Optional threshold for edge inclusion.
        coord_scaling: Coordinate normalization method ('unit_cube', 'unit_sphere', 'standardize').

    Returns:
        NetworkX graph representing quantum system.
    """
    logger.info(
        f"Building quantum graph from {quantum_system.get('model_type', 'unknown')} model"
    )

    # Extract coordinates and flatten
    X = quantum_system["X"]
    Y = quantum_system["Y"]

    # Flatten 2D grids to 1D arrays of coordinates
    coords = np.column_stack([X.flatten(), Y.flatten()])

    # Normalize coordinates if requested
    if normalize_coords:
        coords = normalize_coordinates(coords, method=coord_scaling)

    # Extract Hamiltonian
    H = quantum_system["H"]

    # Build graph from Hamiltonian
    graph = hamiltonian_to_graph(H, coords, threshold=edge_threshold)

    # Add coordinates to graph
    add_coordinates_to_graph(graph, coords, coord_attrs=("x", "y"))

    # Add potential values as node attributes
    V_flat = quantum_system["V"].flatten()
    for idx, node in enumerate(graph.nodes()):
        graph.nodes[node]["potential"] = float(V_flat[idx])
        graph.nodes[node]["node_id"] = idx
        graph.nodes[node]["system_type"] = "quantum"
        graph.nodes[node]["model_type"] = quantum_system.get("model_type", "unknown")

    # Normalize edge weights if requested
    if normalize_weights:
        normalize_edge_weights(graph, method="minmax")

    logger.info(
        f"Quantum graph created: {graph.number_of_nodes()} nodes, "
        f"{graph.number_of_edges()} edges, "
        f"density={nx.density(graph):.6f}"
    )

    return graph


def save_quantum_graph(
    graph: nx.Graph,
    output_path: Union[str, Path],
    format: str = "graphml",
) -> None:
    """
    Save quantum graph to disk.

    Args:
        graph: NetworkX graph from quantum system.
        output_path: Path where to save the graph.
        format: Output format ('graphml', 'gexf', 'edgelist').
    """
    from ml.graphs.cosmology_builder import save_graph

    save_graph(graph, output_path, format=format)


def load_quantum_graph(
    input_path: Union[str, Path],
    format: str = "graphml",
) -> nx.Graph:
    """
    Load quantum graph from disk.

    Args:
        input_path: Path to saved graph.
        format: Input format ('graphml', 'gexf', 'edgelist').

    Returns:
        NetworkX graph.
    """
    from ml.graphs.cosmology_builder import load_graph

    return load_graph(input_path, format=format)


def build_and_save_quantum_graph(
    quantum_system: Dict,
    output_path: Union[str, Path],
    normalize_coords: bool = True,
    normalize_weights: bool = True,
    edge_threshold: Optional[float] = None,
    format: str = "graphml",
) -> nx.Graph:
    """
    Build quantum graph and save to disk.

    Args:
        quantum_system: Dictionary with quantum system data.
        output_path: Path where to save the graph.
        normalize_coords: Whether to normalize coordinates.
        normalize_weights: Whether to normalize edge weights.
        edge_threshold: Optional threshold for edge inclusion.
        format: Output format.

    Returns:
        Created NetworkX graph.
    """
    # Build graph
    graph = build_quantum_graph(
        quantum_system,
        normalize_coords=normalize_coords,
        normalize_weights=normalize_weights,
        edge_threshold=edge_threshold,
    )

    # Save graph
    save_quantum_graph(graph, output_path, format=format)

    return graph


def create_reduced_quantum_graph(
    quantum_system: Dict,
    max_nodes: int = 1000,
    sampling_method: str = "uniform",
    normalize_coords: bool = True,
    normalize_weights: bool = True,
) -> nx.Graph:
    """
    Create a reduced quantum graph by sampling grid points.

    For large quantum systems, this creates a smaller graph by sampling
    a subset of grid points.

    Args:
        quantum_system: Dictionary with quantum system data.
        max_nodes: Maximum number of nodes in reduced graph.
        sampling_method: Sampling method:
            - 'uniform': Uniform random sampling
            - 'potential': Sample based on potential values
            - 'grid': Regular grid sampling
        normalize_coords: Whether to normalize coordinates.
        normalize_weights: Whether to normalize edge weights.

    Returns:
        Reduced NetworkX graph.
    """
    X = quantum_system["X"]
    Y = quantum_system["Y"]
    V = quantum_system["V"]
    H = quantum_system["H"]

    total_nodes = X.size

    if total_nodes <= max_nodes:
        logger.info(f"System has {total_nodes} nodes, no reduction needed")
        return build_quantum_graph(quantum_system, normalize_coords, normalize_weights)

    logger.info(
        f"Reducing quantum graph from {total_nodes} to {max_nodes} nodes "
        f"using {sampling_method} sampling"
    )

    # Generate sampling indices
    if sampling_method == "uniform":
        rng = np.random.default_rng(42)
        sample_indices = rng.choice(total_nodes, size=max_nodes, replace=False)

    elif sampling_method == "grid":
        # Sample every N-th point
        step = int(np.ceil(np.sqrt(total_nodes / max_nodes)))
        X_shape = X.shape
        sample_x = np.arange(0, X_shape[0], step)
        sample_y = np.arange(0, X_shape[1], step)
        sample_xx, sample_yy = np.meshgrid(sample_x, sample_y, indexing="ij")
        sample_indices = np.ravel_multi_index(
            (sample_xx.flatten(), sample_yy.flatten()), X_shape
        )
        sample_indices = sample_indices[:max_nodes]

    else:
        raise ValueError(f"Unknown sampling method: {sampling_method}")

    sample_indices = np.sort(sample_indices)

    # Extract subgraph from Hamiltonian
    H_reduced = H[sample_indices, :][:, sample_indices]

    # Extract coordinates
    coords_reduced = np.column_stack(
        [
            X.flatten()[sample_indices],
            Y.flatten()[sample_indices],
        ]
    )

    # Create reduced quantum system dictionary
    reduced_system = {
        "X": X.flatten()[sample_indices].reshape(-1, 1),
        "Y": Y.flatten()[sample_indices].reshape(-1, 1),
        "V": V.flatten()[sample_indices].reshape(-1, 1),
        "H": H_reduced,
        "model_type": quantum_system.get("model_type", "unknown"),
        "parameters": quantum_system.get("parameters", {}),
    }

    # Build graph from reduced system
    graph = build_quantum_graph(
        reduced_system,
        normalize_coords=normalize_coords,
        normalize_weights=normalize_weights,
    )

    logger.info(f"Reduced quantum graph created with {graph.number_of_nodes()} nodes")

    return graph
