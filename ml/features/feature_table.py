"""
Feature table construction for graphs.

This module provides functions to load graphs, compute all features,
and aggregate them into a unified DataFrame for ML model training.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union

import networkx as nx
import pandas as pd

from app.core.config import get_settings
from app.core.logging import get_logger
from ml.features.embeddings import compute_embedding_features, compute_positional_encoding_features
from ml.features.spectral import compute_extended_spectral_features, compute_spectral_features
from ml.features.topology import compute_topology_features

logger = get_logger(__name__)


def compute_all_features(
    graph: nx.Graph,
    include_topology: bool = True,
    include_spectral: bool = True,
    include_embeddings: bool = True,
    include_extended_spectral: bool = False,
    include_positional_encoding: bool = False,
    topology_kwargs: Optional[Dict] = None,
    spectral_kwargs: Optional[Dict] = None,
    embedding_kwargs: Optional[Dict] = None,
) -> Dict[str, float]:
    """
    Compute all features for a single graph.

    Args:
        graph: NetworkX graph.
        include_topology: Whether to compute topological features.
        include_spectral: Whether to compute spectral features.
        include_embeddings: Whether to compute embedding features.
        include_extended_spectral: Whether to compute extended spectral features.
        include_positional_encoding: Whether to compute positional encoding features.
        topology_kwargs: Keyword arguments for topology features.
        spectral_kwargs: Keyword arguments for spectral features.
        embedding_kwargs: Keyword arguments for embedding features.

    Returns:
        Dictionary with all computed features.
    """
    logger.info(
        f"Computing all features for graph: {graph.number_of_nodes()} nodes, "
        f"{graph.number_of_edges()} edges"
    )

    features = {}

    # Default kwargs
    topology_kwargs = topology_kwargs or {}
    spectral_kwargs = spectral_kwargs or {}
    embedding_kwargs = embedding_kwargs or {}

    # Topological features
    if include_topology:
        topo_features = compute_topology_features(graph, **topology_kwargs)
        features.update(topo_features)

    # Spectral features
    if include_spectral:
        spectral_features = compute_spectral_features(graph, **spectral_kwargs)
        features.update(spectral_features)

    # Extended spectral features
    if include_extended_spectral:
        extended_spectral = compute_extended_spectral_features(graph)
        features.update(extended_spectral)

    # Embedding features
    if include_embeddings:
        embedding_features = compute_embedding_features(graph, **embedding_kwargs)
        features.update(embedding_features)

    # Positional encoding features
    if include_positional_encoding:
        pos_enc_features = compute_positional_encoding_features(graph)
        features.update(pos_enc_features)

    logger.info(f"Computed total of {len(features)} features")

    return features


def load_graphs_from_directory(
    directory: Union[str, Path],
    format: str = "graphml",
    pattern: str = "*.graphml",
) -> List[nx.Graph]:
    """
    Load all graphs from a directory.

    Args:
        directory: Directory containing graph files.
        format: Graph file format ('graphml', 'gexf').
        pattern: File pattern to match.

    Returns:
        List of NetworkX graphs.
    """
    directory = Path(directory)

    logger.info(f"Loading graphs from {directory} with pattern {pattern}")

    graph_files = sorted(directory.glob(pattern))

    if not graph_files:
        logger.warning(f"No graph files found in {directory} matching {pattern}")
        return []

    graphs = []
    for file_path in graph_files:
        try:
            if format == "graphml":
                graph = nx.read_graphml(file_path, node_type=int)
            elif format == "gexf":
                graph = nx.read_gexf(file_path, node_type=int)
            else:
                logger.warning(f"Unknown format {format}, skipping {file_path}")
                continue

            graphs.append(graph)
            logger.debug(f"Loaded graph from {file_path.name}")
        except Exception as e:
            logger.error(f"Failed to load graph from {file_path}: {e}")

    logger.info(f"Loaded {len(graphs)} graphs from {directory}")

    return graphs


def build_feature_table(
    graphs: List[nx.Graph],
    graph_ids: Optional[List[str]] = None,
    system_types: Optional[List[str]] = None,
    include_topology: bool = True,
    include_spectral: bool = True,
    include_embeddings: bool = True,
    feature_kwargs: Optional[Dict] = None,
) -> pd.DataFrame:
    """
    Build feature table from a list of graphs.

    Args:
        graphs: List of NetworkX graphs.
        graph_ids: Optional list of graph IDs.
        system_types: Optional list of system types ('cosmology' or 'quantum').
        include_topology: Whether to include topological features.
        include_spectral: Whether to include spectral features.
        include_embeddings: Whether to include embedding features.
        feature_kwargs: Keyword arguments for feature computation.

    Returns:
        DataFrame with one row per graph and columns for all features.
    """
    logger.info(f"Building feature table for {len(graphs)} graphs")

    if not graphs:
        return pd.DataFrame()

    feature_kwargs = feature_kwargs or {}

    # Generate graph IDs if not provided
    if graph_ids is None:
        graph_ids = [f"graph_{i}" for i in range(len(graphs))]

    # Extract system types from graph attributes if not provided
    if system_types is None:
        system_types = []
        for graph in graphs:
            # Try to get from first node's attributes
            nodes = list(graph.nodes())
            if nodes:
                node_data = graph.nodes[nodes[0]]
                system_type = node_data.get("system_type", "unknown")
            else:
                system_type = "unknown"
            system_types.append(system_type)

    # Compute features for each graph
    feature_rows = []

    for i, graph in enumerate(graphs):
        logger.info(f"Computing features for graph {i+1}/{len(graphs)}")

        features = compute_all_features(
            graph,
            include_topology=include_topology,
            include_spectral=include_spectral,
            include_embeddings=include_embeddings,
            **feature_kwargs,
        )

        # Add metadata
        features["graph_id"] = graph_ids[i]
        features["system_type"] = system_types[i]

        feature_rows.append(features)

    # Create DataFrame
    df = pd.DataFrame(feature_rows)

    # Move metadata columns to front
    metadata_cols = ["graph_id", "system_type"]
    feature_cols = [col for col in df.columns if col not in metadata_cols]
    df = df[metadata_cols + feature_cols]

    logger.info(
        f"Feature table built: {len(df)} rows, {len(df.columns)} columns"
    )

    return df


def save_feature_table(
    df: pd.DataFrame,
    output_path: Union[str, Path],
    format: str = "parquet",
) -> None:
    """
    Save feature table to disk.

    Args:
        df: Feature table DataFrame.
        output_path: Path where to save the table.
        format: Output format ('parquet', 'csv').

    Raises:
        ValueError: If format is not supported.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving feature table to {output_path}")

    if format == "parquet":
        df.to_parquet(output_path, index=False)
    elif format == "csv":
        df.to_csv(output_path, index=False)
    else:
        raise ValueError(f"Unsupported format: {format}")

    logger.info(f"Feature table saved successfully")


def load_feature_table(
    input_path: Union[str, Path],
    format: str = "parquet",
) -> pd.DataFrame:
    """
    Load feature table from disk.

    Args:
        input_path: Path to saved feature table.
        format: Input format ('parquet', 'csv').

    Returns:
        Feature table DataFrame.

    Raises:
        ValueError: If format is not supported.
    """
    input_path = Path(input_path)

    logger.info(f"Loading feature table from {input_path}")

    if format == "parquet":
        df = pd.read_parquet(input_path)
    elif format == "csv":
        df = pd.read_csv(input_path)
    else:
        raise ValueError(f"Unsupported format: {format}")

    logger.info(
        f"Feature table loaded: {len(df)} rows, {len(df.columns)} columns"
    )

    return df


def build_feature_table_from_directory(
    cosmology_graph_dir: Union[str, Path],
    quantum_graph_dir: Union[str, Path],
    output_path: Union[str, Path],
    graph_format: str = "graphml",
    output_format: str = "parquet",
    include_topology: bool = True,
    include_spectral: bool = True,
    include_embeddings: bool = True,
) -> pd.DataFrame:
    """
    Build feature table from graph directories and save to disk.

    Args:
        cosmology_graph_dir: Directory with cosmology graphs.
        quantum_graph_dir: Directory with quantum graphs.
        output_path: Path where to save the feature table.
        graph_format: Format of graph files.
        output_format: Format for output table.
        include_topology: Whether to include topological features.
        include_spectral: Whether to include spectral features.
        include_embeddings: Whether to include embedding features.

    Returns:
        Feature table DataFrame.
    """
    logger.info("Building feature table from graph directories")

    # Load cosmology graphs
    cosmology_graphs = load_graphs_from_directory(
        cosmology_graph_dir,
        format=graph_format,
        pattern=f"*.{graph_format}",
    )
    cosmology_ids = [f"cosmology_{i}" for i in range(len(cosmology_graphs))]
    cosmology_types = ["cosmology"] * len(cosmology_graphs)

    # Load quantum graphs
    quantum_graphs = load_graphs_from_directory(
        quantum_graph_dir,
        format=graph_format,
        pattern=f"*.{graph_format}",
    )
    quantum_ids = [f"quantum_{i}" for i in range(len(quantum_graphs))]
    quantum_types = ["quantum"] * len(quantum_graphs)

    # Combine
    all_graphs = cosmology_graphs + quantum_graphs
    all_ids = cosmology_ids + quantum_ids
    all_types = cosmology_types + quantum_types

    # Build feature table
    df = build_feature_table(
        all_graphs,
        graph_ids=all_ids,
        system_types=all_types,
        include_topology=include_topology,
        include_spectral=include_spectral,
        include_embeddings=include_embeddings,
    )

    # Save to disk
    save_feature_table(df, output_path, format=output_format)

    return df
