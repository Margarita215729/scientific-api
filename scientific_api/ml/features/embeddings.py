"""
Node embedding and graph-level representation learning.

This module provides functions to compute node embeddings using
random walk-based methods and aggregate them to graph-level features.
"""

from typing import Dict, Optional, Tuple

import networkx as nx
import numpy as np
from sklearn.decomposition import PCA

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


def random_walk(
    graph: nx.Graph,
    start_node: int,
    walk_length: int,
    rng: np.random.Generator,
) -> list:
    """
    Perform a single random walk starting from a node.

    Args:
        graph: NetworkX graph.
        start_node: Starting node for the walk.
        walk_length: Length of the walk.
        rng: Random number generator.

    Returns:
        List of nodes in the walk.
    """
    walk = [start_node]

    for _ in range(walk_length - 1):
        current = walk[-1]
        neighbors = list(graph.neighbors(current))

        if not neighbors:
            break

        next_node = rng.choice(neighbors)
        walk.append(next_node)

    return walk


def generate_random_walks(
    graph: nx.Graph,
    num_walks: int = 10,
    walk_length: int = 80,
    random_state: Optional[int] = None,
) -> list:
    """
    Generate multiple random walks from all nodes in the graph.

    Args:
        graph: NetworkX graph.
        num_walks: Number of walks per node.
        walk_length: Length of each walk.
        random_state: Random seed for reproducibility.

    Returns:
        List of walks, where each walk is a list of nodes.
    """
    settings = get_settings()
    if random_state is None:
        random_state = settings.ML_RANDOM_SEED

    rng = np.random.default_rng(random_state)
    nodes = list(graph.nodes())

    logger.info(
        f"Generating random walks: {num_walks} walks/node, "
        f"length={walk_length}, total nodes={len(nodes)}"
    )

    walks = []
    for _ in range(num_walks):
        rng.shuffle(nodes)
        for node in nodes:
            walk = random_walk(graph, node, walk_length, rng)
            walks.append(walk)

    logger.info(f"Generated {len(walks)} random walks")
    return walks


def compute_simple_node_embeddings(
    graph: nx.Graph,
    embedding_dim: int = 128,
    num_walks: int = 10,
    walk_length: int = 80,
    window_size: int = 5,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """
    Compute simple node embeddings using random walk co-occurrence.

    This is a simplified version inspired by DeepWalk/Node2Vec.
    For production, consider using dedicated libraries like node2vec or gensim.

    Args:
        graph: NetworkX graph.
        embedding_dim: Dimension of embedding vectors.
        num_walks: Number of random walks per node.
        walk_length: Length of each walk.
        window_size: Context window size for co-occurrence.
        random_state: Random seed.

    Returns:
        Node embedding matrix (shape: [n_nodes, embedding_dim]).
    """
    logger.info(
        f"Computing simple node embeddings: dim={embedding_dim}, "
        f"walks={num_walks}, length={walk_length}"
    )

    nodes = list(graph.nodes())
    n_nodes = len(nodes)
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}

    # Generate random walks
    walks = generate_random_walks(
        graph,
        num_walks=num_walks,
        walk_length=walk_length,
        random_state=random_state,
    )

    # Build co-occurrence matrix
    cooccurrence = np.zeros((n_nodes, n_nodes))

    for walk in walks:
        for i, node in enumerate(walk):
            node_idx = node_to_idx.get(node)
            if node_idx is None:
                continue

            # Context window
            start = max(0, i - window_size)
            end = min(len(walk), i + window_size + 1)

            for j in range(start, end):
                if i != j:
                    context_node = walk[j]
                    context_idx = node_to_idx.get(context_node)
                    if context_idx is not None:
                        cooccurrence[node_idx, context_idx] += 1

    # Apply SVD/PCA to reduce dimensionality
    logger.debug("Applying PCA for dimensionality reduction")
    
    # Add small regularization to avoid singular matrix
    cooccurrence += 1e-6

    pca = PCA(n_components=min(embedding_dim, n_nodes), random_state=random_state)
    embeddings = pca.fit_transform(cooccurrence)

    # Pad if necessary
    if embeddings.shape[1] < embedding_dim:
        padding = np.zeros((n_nodes, embedding_dim - embeddings.shape[1]))
        embeddings = np.hstack([embeddings, padding])

    logger.info(f"Computed embeddings: shape={embeddings.shape}")
    return embeddings


def aggregate_embeddings(
    embeddings: np.ndarray,
    method: str = "mean",
) -> np.ndarray:
    """
    Aggregate node embeddings to graph-level representation.

    Args:
        embeddings: Node embedding matrix (shape: [n_nodes, embedding_dim]).
        method: Aggregation method ('mean', 'max', 'sum', 'std', 'concat').

    Returns:
        Graph-level embedding vector.
    """
    if method == "mean":
        graph_embedding = np.mean(embeddings, axis=0)
    elif method == "max":
        graph_embedding = np.max(embeddings, axis=0)
    elif method == "sum":
        graph_embedding = np.sum(embeddings, axis=0)
    elif method == "std":
        graph_embedding = np.std(embeddings, axis=0)
    elif method == "concat":
        # Concatenate multiple statistics
        mean_emb = np.mean(embeddings, axis=0)
        max_emb = np.max(embeddings, axis=0)
        std_emb = np.std(embeddings, axis=0)
        graph_embedding = np.concatenate([mean_emb, max_emb, std_emb])
    else:
        raise ValueError(f"Unknown aggregation method: {method}")

    logger.debug(
        f"Aggregated {len(embeddings)} node embeddings using {method}: "
        f"result shape={graph_embedding.shape}"
    )

    return graph_embedding


def compute_embedding_features(
    graph: nx.Graph,
    embedding_dim: int = 64,
    aggregation_method: str = "mean",
    num_walks: int = 10,
    walk_length: int = 80,
    random_state: Optional[int] = None,
) -> Dict[str, float]:
    """
    Compute graph-level features from node embeddings.

    Args:
        graph: NetworkX graph.
        embedding_dim: Dimension of node embeddings.
        aggregation_method: How to aggregate node embeddings ('mean', 'max', 'concat').
        num_walks: Number of random walks per node.
        walk_length: Length of each walk.
        random_state: Random seed.

    Returns:
        Dictionary with embedding-based features.
    """
    logger.info(
        f"Computing embedding features for graph with {graph.number_of_nodes()} nodes"
    )

    if graph.number_of_nodes() == 0:
        # Return zero features
        if aggregation_method == "concat":
            feature_dim = embedding_dim * 3
        else:
            feature_dim = embedding_dim

        return {f"embedding_{i}": 0.0 for i in range(feature_dim)}

    # Compute node embeddings
    node_embeddings = compute_simple_node_embeddings(
        graph,
        embedding_dim=embedding_dim,
        num_walks=num_walks,
        walk_length=walk_length,
        random_state=random_state,
    )

    # Aggregate to graph-level
    graph_embedding = aggregate_embeddings(node_embeddings, method=aggregation_method)

    # Convert to feature dictionary
    features = {f"embedding_{i}": float(val) for i, val in enumerate(graph_embedding)}

    logger.info(f"Computed {len(features)} embedding features")

    return features


def compute_positional_encoding(
    graph: nx.Graph,
    k: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Laplacian positional encoding for nodes.

    Uses eigenvectors of the graph Laplacian as positional features.

    Args:
        graph: NetworkX graph.
        k: Number of eigenvectors to compute.

    Returns:
        Tuple of (eigenvalues, eigenvectors) where eigenvectors shape is [n_nodes, k].
    """
    from scipy.sparse.linalg import eigsh

    n_nodes = graph.number_of_nodes()

    if n_nodes == 0:
        return np.array([]), np.array([])

    if k >= n_nodes:
        k = max(1, n_nodes - 1)

    # Get normalized Laplacian
    L = nx.normalized_laplacian_matrix(graph)

    # Compute eigenvectors
    try:
        eigenvalues, eigenvectors = eigsh(L, k=k, which="SM")
        logger.debug(
            f"Computed {k} Laplacian eigenvectors for positional encoding"
        )
    except Exception as e:
        logger.warning(f"Could not compute positional encoding: {e}")
        eigenvalues = np.zeros(k)
        eigenvectors = np.zeros((n_nodes, k))

    return eigenvalues, eigenvectors


def compute_positional_encoding_features(
    graph: nx.Graph,
    k: int = 10,
) -> Dict[str, float]:
    """
    Compute graph-level features from Laplacian positional encoding.

    Args:
        graph: NetworkX graph.
        k: Number of eigenvectors to use.

    Returns:
        Dictionary with positional encoding features.
    """
    logger.info("Computing positional encoding features")

    eigenvalues, eigenvectors = compute_positional_encoding(graph, k=k)

    if eigenvectors.size == 0:
        return {f"pos_enc_{i}": 0.0 for i in range(k)}

    # Aggregate eigenvectors (mean across nodes)
    aggregated = np.mean(np.abs(eigenvectors), axis=0)

    features = {f"pos_enc_{i}": float(val) for i, val in enumerate(aggregated)}

    logger.info(f"Computed {len(features)} positional encoding features")

    return features
