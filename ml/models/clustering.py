"""
Graph Clustering Models.

This module provides functions to cluster graphs in feature space
using unsupervised learning algorithms.

Clustering can reveal natural groupings of graphs based on their
topological, spectral, and structural properties.

Key Functions:
--------------
- cluster_graphs_kmeans: K-means clustering
- cluster_graphs_dbscan: DBSCAN density-based clustering
- evaluate_clustering: Evaluate clustering quality (silhouette, Davies-Bouldin)
- assign_cluster_labels: Assign cluster labels to feature table
- compute_cluster_statistics: Compute cluster-level summaries
- save_clustering / load_clustering: Model persistence

Dependencies:
-------------
- scikit-learn for clustering algorithms
- pandas for feature tables
- joblib for model serialization
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def prepare_clustering_data(
    feature_table: pd.DataFrame,
    feature_columns: Optional[List[str]] = None,
    scale_features: bool = True,
) -> Tuple[np.ndarray, Optional[StandardScaler]]:
    """
    Prepare feature table for clustering.

    Parameters
    ----------
    feature_table : pd.DataFrame
        Feature table with graph features.
    feature_columns : list of str, optional
        List of feature column names. If None, uses all numeric columns.
    scale_features : bool, default=True
        Whether to standardize features (mean=0, std=1).

    Returns
    -------
    X : np.ndarray
        Feature array (n_samples, n_features).
    scaler : StandardScaler or None
        Fitted scaler if scale_features=True, else None.

    Notes
    -----
    - Removes NaN/inf values
    - Feature scaling recommended for k-means and DBSCAN
    """
    logger.info("Preparing clustering data...")

    # Select features
    if feature_columns is None:
        # Use all numeric columns except metadata
        exclude_cols = {"graph_id", "system_type"}
        feature_columns = [
            col for col in feature_table.columns
            if col not in exclude_cols and pd.api.types.is_numeric_dtype(feature_table[col])
        ]

    logger.info(f"Using {len(feature_columns)} features for clustering")

    # Extract features
    X = feature_table[feature_columns].values

    # Handle missing/invalid values
    valid_mask = np.all(np.isfinite(X), axis=1)
    if not np.all(valid_mask):
        n_invalid = np.sum(~valid_mask)
        logger.warning(f"Removing {n_invalid} rows with NaN/inf values")
        X = X[valid_mask]

    logger.info(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")

    # Feature scaling
    scaler = None
    if scale_features:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        logger.info("Features standardized (mean=0, std=1)")

    return X, scaler


def cluster_graphs_kmeans(
    X: np.ndarray,
    n_clusters: int = 3,
    random_state: int = 42,
    n_init: int = 10,
    max_iter: int = 300,
) -> np.ndarray:
    """
    Cluster graphs using K-means.

    Parameters
    ----------
    X : np.ndarray
        Feature array (n_samples, n_features).
    n_clusters : int, default=3
        Number of clusters.
    random_state : int, default=42
        Random seed for reproducibility.
    n_init : int, default=10
        Number of times the k-means algorithm will be run with different centroid seeds.
    max_iter : int, default=300
        Maximum number of iterations of the k-means algorithm.

    Returns
    -------
    labels : np.ndarray
        Cluster labels for each sample (n_samples,).

    Notes
    -----
    - K-means requires specifying number of clusters in advance
    - Works well for spherical, evenly-sized clusters
    - Sensitive to feature scaling - always scale features first
    """
    logger.info(f"Running K-means clustering with {n_clusters} clusters...")

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=n_init,
        max_iter=max_iter,
    )

    labels = kmeans.fit_predict(X)

    # Count cluster sizes
    unique_labels, counts = np.unique(labels, return_counts=True)
    logger.info(f"K-means complete. Cluster sizes: {dict(zip(unique_labels, counts))}")

    return labels


def cluster_graphs_dbscan(
    X: np.ndarray,
    eps: float = 0.5,
    min_samples: int = 5,
) -> np.ndarray:
    """
    Cluster graphs using DBSCAN (density-based clustering).

    Parameters
    ----------
    X : np.ndarray
        Feature array (n_samples, n_features).
    eps : float, default=0.5
        Maximum distance between two samples for one to be considered as in the neighborhood of the other.
    min_samples : int, default=5
        Number of samples in a neighborhood for a point to be considered as a core point.

    Returns
    -------
    labels : np.ndarray
        Cluster labels for each sample (n_samples,).
        Noise points are labeled with -1.

    Notes
    -----
    - DBSCAN does not require specifying number of clusters
    - Can find clusters of arbitrary shape
    - Robust to outliers (noise points labeled as -1)
    - Sensitive to eps and min_samples parameters
    """
    logger.info(f"Running DBSCAN clustering with eps={eps}, min_samples={min_samples}...")

    dbscan = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)

    labels = dbscan.fit_predict(X)

    # Count cluster sizes
    unique_labels, counts = np.unique(labels, return_counts=True)
    n_clusters = np.sum(unique_labels >= 0)
    n_noise = np.sum(labels == -1)
    logger.info(f"DBSCAN complete. Found {n_clusters} clusters, {n_noise} noise points")
    logger.info(f"Cluster sizes: {dict(zip(unique_labels, counts))}")

    return labels


def evaluate_clustering(
    X: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, float]:
    """
    Evaluate clustering quality.

    Parameters
    ----------
    X : np.ndarray
        Feature array (n_samples, n_features).
    labels : np.ndarray
        Cluster labels (n_samples,).

    Returns
    -------
    metrics : dict
        Dictionary of evaluation metrics:
        - silhouette_score: [-1, 1], higher is better (measures cluster cohesion and separation)
        - davies_bouldin_score: [0, inf), lower is better (measures cluster similarity)
        - calinski_harabasz_score: [0, inf), higher is better (measures cluster variance ratio)
        - n_clusters: number of clusters (excluding noise)
        - n_noise: number of noise points (for DBSCAN)

    Notes
    -----
    - Silhouette score: 1 = well-separated clusters, 0 = overlapping, -1 = wrong assignment
    - Davies-Bouldin: lower is better, 0 = perfect separation
    - Calinski-Harabasz: higher is better, ratio of between-cluster to within-cluster variance
    - Metrics may not be defined for single-cluster or all-noise results
    """
    logger.info("Evaluating clustering quality...")

    unique_labels = np.unique(labels)
    n_clusters = np.sum(unique_labels >= 0)
    n_noise = np.sum(labels == -1)

    metrics = {
        "n_clusters": n_clusters,
        "n_noise": n_noise,
    }

    # Check if clustering is valid
    if n_clusters < 2:
        logger.warning("Less than 2 clusters found, skipping quality metrics")
        metrics["silhouette_score"] = np.nan
        metrics["davies_bouldin_score"] = np.nan
        metrics["calinski_harabasz_score"] = np.nan
        return metrics

    # Filter out noise points for metric computation
    valid_mask = labels >= 0
    X_valid = X[valid_mask]
    labels_valid = labels[valid_mask]

    if len(X_valid) < 2:
        logger.warning("Too few valid points, skipping quality metrics")
        metrics["silhouette_score"] = np.nan
        metrics["davies_bouldin_score"] = np.nan
        metrics["calinski_harabasz_score"] = np.nan
        return metrics

    # Silhouette score
    try:
        sil_score = silhouette_score(X_valid, labels_valid)
        metrics["silhouette_score"] = sil_score
        logger.info(f"Silhouette score: {sil_score:.4f}")
    except Exception as e:
        logger.warning(f"Failed to compute silhouette score: {e}")
        metrics["silhouette_score"] = np.nan

    # Davies-Bouldin index
    try:
        db_score = davies_bouldin_score(X_valid, labels_valid)
        metrics["davies_bouldin_score"] = db_score
        logger.info(f"Davies-Bouldin score: {db_score:.4f}")
    except Exception as e:
        logger.warning(f"Failed to compute Davies-Bouldin score: {e}")
        metrics["davies_bouldin_score"] = np.nan

    # Calinski-Harabasz index
    try:
        ch_score = calinski_harabasz_score(X_valid, labels_valid)
        metrics["calinski_harabasz_score"] = ch_score
        logger.info(f"Calinski-Harabasz score: {ch_score:.4f}")
    except Exception as e:
        logger.warning(f"Failed to compute Calinski-Harabasz score: {e}")
        metrics["calinski_harabasz_score"] = np.nan

    return metrics


def assign_cluster_labels(
    feature_table: pd.DataFrame,
    labels: np.ndarray,
    cluster_column: str = "cluster_id",
) -> pd.DataFrame:
    """
    Assign cluster labels to feature table.

    Parameters
    ----------
    feature_table : pd.DataFrame
        Original feature table.
    labels : np.ndarray
        Cluster labels (n_samples,).
    cluster_column : str, default="cluster_id"
        Name of new column for cluster labels.

    Returns
    -------
    feature_table_with_clusters : pd.DataFrame
        Feature table with added cluster labels.

    Notes
    -----
    - Returns a copy of the input DataFrame
    - Noise points (label -1) are preserved
    """
    feature_table_copy = feature_table.copy()
    feature_table_copy[cluster_column] = labels

    logger.info(f"Cluster labels assigned to column '{cluster_column}'")
    return feature_table_copy


def compute_cluster_statistics(
    feature_table: pd.DataFrame,
    cluster_column: str = "cluster_id",
    feature_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Compute cluster-level statistics.

    Parameters
    ----------
    feature_table : pd.DataFrame
        Feature table with cluster labels.
    cluster_column : str, default="cluster_id"
        Column name containing cluster labels.
    feature_columns : list of str, optional
        List of feature columns to summarize. If None, uses all numeric columns.

    Returns
    -------
    cluster_stats : pd.DataFrame
        Summary statistics for each cluster.
        Rows: clusters, Columns: feature means, stds, counts

    Notes
    -----
    - Computes mean and std for each feature within each cluster
    - Includes cluster sizes
    """
    logger.info("Computing cluster statistics...")

    if feature_columns is None:
        # Use all numeric columns except cluster_id and metadata
        exclude_cols = {cluster_column, "graph_id", "system_type"}
        feature_columns = [
            col for col in feature_table.columns
            if col not in exclude_cols and pd.api.types.is_numeric_dtype(feature_table[col])
        ]

    # Group by cluster
    grouped = feature_table.groupby(cluster_column)

    # Compute statistics
    cluster_stats_list = []

    for cluster_id, group in grouped:
        stats = {"cluster_id": cluster_id, "count": len(group)}

        # Feature means
        for col in feature_columns:
            stats[f"{col}_mean"] = group[col].mean()
            stats[f"{col}_std"] = group[col].std()

        cluster_stats_list.append(stats)

    cluster_stats = pd.DataFrame(cluster_stats_list)

    logger.info(f"Computed statistics for {len(cluster_stats)} clusters")
    return cluster_stats


def save_clustering(
    labels: np.ndarray,
    scaler: Optional[StandardScaler],
    feature_columns: List[str],
    cluster_method: str,
    cluster_params: Dict[str, Any],
    output_path: Path,
) -> None:
    """
    Save clustering results and metadata.

    Parameters
    ----------
    labels : np.ndarray
        Cluster labels.
    scaler : StandardScaler or None
        Fitted scaler.
    feature_columns : list of str
        List of feature column names.
    cluster_method : str
        Clustering method ("kmeans" or "dbscan").
    cluster_params : dict
        Clustering parameters (e.g., {"n_clusters": 3} or {"eps": 0.5}).
    output_path : Path
        Output file path (.joblib).

    Notes
    -----
    - Saves labels, scaler, feature columns, method, and params in single file
    - Use .joblib extension for efficiency
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    clustering_data = {
        "labels": labels,
        "scaler": scaler,
        "feature_columns": feature_columns,
        "cluster_method": cluster_method,
        "cluster_params": cluster_params,
    }

    joblib.dump(clustering_data, output_path)
    logger.info(f"Clustering saved to {output_path}")


def load_clustering(input_path: Path) -> Tuple[np.ndarray, Optional[StandardScaler], List[str], str, Dict[str, Any]]:
    """
    Load clustering results and metadata.

    Parameters
    ----------
    input_path : Path
        Input file path (.joblib).

    Returns
    -------
    labels : np.ndarray
        Cluster labels.
    scaler : StandardScaler or None
        Fitted scaler.
    feature_columns : list of str
        List of feature column names.
    cluster_method : str
        Clustering method.
    cluster_params : dict
        Clustering parameters.
    """
    input_path = Path(input_path)
    clustering_data = joblib.load(input_path)

    labels = clustering_data["labels"]
    scaler = clustering_data.get("scaler")
    feature_columns = clustering_data["feature_columns"]
    cluster_method = clustering_data["cluster_method"]
    cluster_params = clustering_data["cluster_params"]

    logger.info(f"Clustering loaded from {input_path}")
    return labels, scaler, feature_columns, cluster_method, cluster_params


# Example usage
if __name__ == "__main__":
    from app.core.logging import setup_logging
    from ml.features.feature_table import load_feature_table

    setup_logging()

    # Example: Load feature table and perform clustering
    feature_table_path = Path("data/processed/features/feature_table.parquet")

    if feature_table_path.exists():
        logger.info(f"Loading feature table from {feature_table_path}...")
        feature_table = load_feature_table(feature_table_path)

        # Prepare data
        X, scaler = prepare_clustering_data(feature_table, scale_features=True)

        # K-means clustering
        logger.info("\n=== K-means Clustering ===")
        labels_kmeans = cluster_graphs_kmeans(X, n_clusters=3)
        metrics_kmeans = evaluate_clustering(X, labels_kmeans)
        print("K-means metrics:", metrics_kmeans)

        # Assign labels
        feature_table_kmeans = assign_cluster_labels(feature_table, labels_kmeans, "kmeans_cluster")

        # Cluster statistics
        cluster_stats_kmeans = compute_cluster_statistics(feature_table_kmeans, "kmeans_cluster")
        print("\nK-means cluster statistics:")
        print(cluster_stats_kmeans[["kmeans_cluster", "count"]])

        # DBSCAN clustering
        logger.info("\n=== DBSCAN Clustering ===")
        labels_dbscan = cluster_graphs_dbscan(X, eps=0.5, min_samples=5)
        metrics_dbscan = evaluate_clustering(X, labels_dbscan)
        print("DBSCAN metrics:", metrics_dbscan)

        # Assign labels
        feature_table_dbscan = assign_cluster_labels(feature_table, labels_dbscan, "dbscan_cluster")

        # Cluster statistics
        cluster_stats_dbscan = compute_cluster_statistics(feature_table_dbscan, "dbscan_cluster")
        print("\nDBSCAN cluster statistics:")
        print(cluster_stats_dbscan[["dbscan_cluster", "count"]])

        # Save clustering results
        output_dir = Path("data/clustering")
        feature_cols = [
            col for col in feature_table.columns
            if col not in {"graph_id", "system_type"} and pd.api.types.is_numeric_dtype(feature_table[col])
        ]

        save_clustering(
            labels_kmeans,
            scaler,
            feature_cols,
            "kmeans",
            {"n_clusters": 3},
            output_dir / "kmeans_clustering.joblib"
        )

        save_clustering(
            labels_dbscan,
            scaler,
            feature_cols,
            "dbscan",
            {"eps": 0.5, "min_samples": 5},
            output_dir / "dbscan_clustering.joblib"
        )

        logger.info("Clustering complete")
    else:
        logger.error(f"Feature table not found at {feature_table_path}")
        logger.info("Run build_feature_table_from_directory() first to generate feature table")
